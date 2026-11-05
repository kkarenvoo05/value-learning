"""Stage 6 (docs/libero_plan.md): fine-tune pi05_libero on the moka rollouts, one arm per run.

Usage (openpi env; see scripts/pi05_policy_finetune.sbatch):
    .venv/bin/python <vl>/scripts/pi05_policy_finetune.py --selftest --out <dir>                    # one GPU
    .venv/bin/python <vl>/scripts/pi05_policy_finetune.py --overfit --out <dir>                     # Check C
    .venv/bin/python <vl>/scripts/pi05_policy_finetune.py --train --arm imitation --out <dir>       # arm A
    .venv/bin/python <vl>/scripts/pi05_policy_finetune.py --train --arm advantage --labels L.parquet --out <dir>
Serve a result with scripts/serve_pi05_finetuned.py (adds the advantage tag to the client's prompt).

Data: src/vl/libero_finetune_data.py (arms, action chunks, labels, prompt tag, streaming). Table = the fixed + fresh
table (900 train rollouts: 450 on the 30 fixed train states + 450 fresh layouts); val / test init states never
enter training (asserted). Arm A = successful train rollouts; arm C = all train rollouts + per-frame tag.

Model and recipe = openpi's LoRA fine-tune (`pi0_libero_low_mem_finetune`, the openpi config for LoRA on LIBERO),
applied to pi05_libero:
  [openpi] paligemma `gemma_2b_lora` + action expert `gemma_300m_lora` (rank 16 / 32, alpha = rank);
           freeze filter = `Pi0Config.get_freeze_filter()`: the LLMs' pretrained weights are frozen (bf16), everything
           else trains: LoRA, SigLIP, action in / out projections, time MLP (fp32); flow-matching loss of
           `Pi0.compute_loss(train=True)` (includes openpi's image augmentation); batch 32; AdamW b2 0.95,
           weight decay 1e-10, grad clip 1.0; no EMA; inputs through the policy's own input transform (the same
           normalisation and tokenisation as serving).
  [ours]   lora_b = 0 at init (openpi draws both factors N(0, 0.01)), so step 0 is exactly pi05_libero, the policy
           every arm is compared with; peak lr 1e-4 for LoRA (the Stage 3b LoRA finding: at 1e-5 zero-initialised
           adapters moved ~0.1%) and openpi's 2.5e-5 for the other trainable weights; warmup 500, cosine to 10%;
           a fixed step budget per arm (no selection on val: the policy is judged by rollouts on the test states);
           gradient accumulation (micro-batch 8) to fit 24-48 GB GPUs.
Val loss: flow-matching loss on a fixed set of val-state frames with fixed noise / time, prompted as the arm is at
inference (A: plain; C: tag positive). A drift check, not a selection criterion.

Saved: trainable weights only (`trainable_<step>.msgpack`, ~1.7 GB: LoRA + SigLIP + projections), merged with the
frozen pi05_libero weights at load time (`load_finetuned`); full checkpoints would be ~7 GB each.

--selftest: (1) with lora_b = 0 the LoRA model's loss equals the plain pi05_libero loss on the same batch and rng;
(2) the tagged prompts tokenize differently and are not truncated; (3) the normalised action chunk maps back to the
raw executed actions through the policy's output transform; (4) a save -> load_finetuned round trip is exact.
--overfit = Check C: one fixed batch of 32 with fixed noise / time, 300 steps; pass [ours]: loss < 20% of step 0.
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx, serialization, traverse_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl import libero_finetune_data as fd  # noqa: E402

CKPT = "/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero"
LORA = ".*lora.*"


def base_policy():
    from openpi.policies import policy_config
    from openpi.training import config as _config
    tc = _config.get_config("pi05_libero")
    return tc, policy_config.create_trained_policy(tc, CKPT)


def lora_model(tc, base_params, seed: int):
    """pi05_libero weights in the LoRA variant; lora_a ~ N(0, 0.01), lora_b = 0 [ours]. Returns (model, cfg)."""
    cfg = dataclasses.replace(tc.model, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    shapes = nnx.state(nnx.eval_shape(cfg.create, jax.random.key(0))).to_pure_dict()
    flat, base = traverse_util.flatten_dict(shapes, sep="/"), traverse_util.flatten_dict(base_params, sep="/")
    assert set(base) <= set(flat), sorted(set(base) - set(flat))[:5]
    key, merged = jax.random.PRNGKey(seed + 3000), {}
    for k, s in flat.items():
        if "lora" not in k:
            merged[k] = base[k]
        elif k.endswith("lora_b"):
            merged[k] = jnp.zeros(s.shape, jnp.float32)
        else:
            key, sub = jax.random.split(key)
            merged[k] = 0.01 * jax.random.normal(sub, s.shape, jnp.float32)
    return cfg.load(traverse_util.unflatten_dict(merged, sep="/"), remove_extra_params=False), cfg


def split(model, cfg, freeze_siglip: bool = False):
    """(graphdef, trainable fp32, frozen bf16) by openpi's freeze filter; freeze_siglip [ours, 2026-11-04 recipe
    variant] also freezes the vision encoder (PaliGemma/img), so only LoRA + action projections + time MLP train."""
    from openpi.shared import nnx_utils
    frozen_f = cfg.get_freeze_filter()
    if freeze_siglip:
        frozen_f = nnx.Any(frozen_f, nnx_utils.PathRegex(".*PaliGemma/img.*"))
    graphdef, train, frozen = nnx.split(model, nnx.All(nnx.Param, nnx.Not(frozen_f)), ...)
    train = jax.tree.map(lambda x: x.astype(jnp.float32), train)
    frozen = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, frozen)
    return graphdef, train, frozen


def counts(train) -> dict:
    out = {}
    for k, v in traverse_util.flatten_dict(train.to_pure_dict(), sep="/").items():
        kind = "lora" if "lora" in k else k.split("/")[0]
        out[kind] = out.get(kind, 0) + int(v.size)
    return out


def lora_b_rms(train) -> float:
    vs = [v for k, v in traverse_util.flatten_dict(train.to_pure_dict(), sep="/").items() if k.endswith("lora_b")]
    return float(np.sqrt(sum(float(jnp.sum(jnp.square(v))) for v in vs) / sum(v.size for v in vs)))


def batch_arrays(policy, samples):
    xs = [policy._input_transform(dict(s)) for s in samples]
    return jax.tree.map(lambda *x: np.stack(x), *xs)


def as_inputs(b):
    from openpi.models import model as _model
    b = jax.tree.map(jnp.asarray, b)
    return _model.Observation.from_dict(b), b["actions"]


def make_loss(graphdef):
    def loss_fn(train, frozen, b, rng, aug: bool):
        model = nnx.merge(graphdef, train, frozen)
        obs, actions = as_inputs(b)
        return model.compute_loss(rng, obs, actions, train=aug).mean()
    return loss_fn


def save(train, path: Path):
    path.write_bytes(serialization.to_bytes(jax.device_get(train.to_pure_dict())))


def load_finetuned(run_dir, step: str = "final", seed: int = 0):
    """(policy with the fine-tuned model, results.json). The policy keeps pi05_libero's transforms and norm stats."""
    from openpi.policies import policy as _policy
    from openpi.shared import nnx_utils
    run_dir = Path(run_dir)
    tc, policy = base_policy()
    model, cfg = lora_model(tc, nnx.state(policy._model).to_pure_dict(), seed)
    rj = run_dir / "results.json"
    fz = bool(json.loads(rj.read_text())["args"].get("freeze_siglip", False)) if rj.exists() else False
    graphdef, train, frozen = split(model, cfg, fz)
    pure = serialization.from_bytes(train.to_pure_dict(), (run_dir / f"trainable_{step}.msgpack").read_bytes())
    train.replace_by_pure_dict(jax.tree.map(jnp.asarray, pure))
    merged = nnx.merge(graphdef, jax.tree.map(lambda x: x.astype(jnp.bfloat16), train), frozen)
    policy._model = merged
    policy._sample_actions = nnx_utils.module_jit(merged.sample_actions)
    assert isinstance(policy, _policy.Policy)
    return policy, json.loads((run_dir / "results.json").read_text()) if (run_dir / "results.json").exists() else {}


def raw_samples(n_eps: int, per_ep: int, tag=None, split_name="train", seed=0):
    eps = pd.read_csv(fd.TABLE / "episodes.csv")
    rows = eps[eps.split == split_name].sort_values("episode_index").head(n_eps)
    rng, out = np.random.default_rng(seed), []
    for r in rows.itertuples():
        ep = fd.load_episode(r.raw, None, int(r.episode_index))
        for t in rng.choice(len(ep["actions"]), per_ep, replace=False):
            out.append(fd.sample(ep, int(t), tag))
    return out


def selftest(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tc, policy = base_policy()
    base_params = nnx.state(policy._model).to_pure_dict()
    ref = policy._model
    model, cfg = lora_model(tc, base_params, a.seed)
    graphdef, train, frozen = split(model, cfg)
    print("trainable", counts(train), flush=True)
    samples = raw_samples(2, 4)
    b = batch_arrays(policy, samples)
    rng = jax.random.PRNGKey(0)
    gr, sr = nnx.split(ref)
    l_ref = float(jax.jit(lambda st, b: nnx.merge(gr, st).compute_loss(rng, *as_inputs(b), train=False).mean())(sr, b))
    loss_fn = make_loss(graphdef)
    l_lora = float(jax.jit(loss_fn, static_argnums=4)(train, frozen, b, rng, False))
    res = {"loss_base": l_ref, "loss_lora_b0": l_lora}
    ok1 = abs(l_ref - l_lora) <= 0.02 * abs(l_ref) + 1e-4  # bf16 vs fp32 trainable weights, separate programs
    print(f"(1) loss pi05 {l_ref:.5f} vs LoRA(b=0) {l_lora:.5f}: {'ok' if ok1 else 'FAIL'}", flush=True)
    toks = {}
    for tag in [None, True, False]:
        s = dict(samples[0]); s["prompt"] = fd.tag_prompt(samples[0]["prompt"], tag)
        x = policy._input_transform(s)
        toks[tag] = (x["tokenized_prompt"], x["tokenized_prompt_mask"])
    n_valid = {str(k): int(v[1].sum()) for k, v in toks.items()}
    ok2 = (not np.array_equal(toks[True][0], toks[False][0]) and not np.array_equal(toks[None][0], toks[True][0])
           and max(n_valid.values()) < len(toks[None][1]))
    print(f"(2) prompt tokens {n_valid} of {len(toks[None][1])}: {'ok' if ok2 else 'FAIL'}", flush=True)
    x = policy._input_transform(dict(samples[0]))
    back = policy._output_transform({"actions": np.asarray(x["actions"]), "state": x["state"]})["actions"]
    err = float(np.abs(back - samples[0]["actions"]).max())
    ok3 = err < 1e-4
    print(f"(3) action chunk round trip max |err| {err:.2e}: {'ok' if ok3 else 'FAIL'}", flush=True)
    train2 = jax.tree.map(lambda x: x + 0.001, train)
    save(train2, out / "trainable_final.msgpack")
    train2 = jax.device_get(train2)  # host copy; free the first model before loading a second (24 GB GPUs)
    del ref, sr, model, train, frozen, policy, base_params
    import gc
    gc.collect()
    pol2, _ = load_finetuned(out, seed=a.seed)
    _, t3, _ = split(pol2._model, cfg)
    d = jax.tree.map(lambda u, v: float(jnp.abs(u.astype(jnp.float32) - v.astype(jnp.bfloat16).astype(jnp.float32)
                                                ).max()), t3, train2)
    err4 = max(jax.tree.leaves(d))
    ok4 = err4 == 0.0
    print(f"(4) save/load round trip max |diff| (bf16) {err4}: {'ok' if ok4 else 'FAIL'}", flush=True)
    (out / "trainable_final.msgpack").unlink()
    res.update(ok=[ok1, ok2, ok3, ok4], prompt_tokens=n_valid, action_err=err, roundtrip=err4)
    (out / "selftest.json").write_text(json.dumps(res, indent=2))
    assert all(res["ok"]), res


def optimizer(a, train):
    """AdamW (openpi's b2 0.95, weight decay 1e-10, clip 1.0); LoRA at a.lora_lr, other trainable weights at a.lr."""
    sched = lambda lr: optax.warmup_cosine_decay_schedule(0.0, lr, a.warmup, a.steps, lr * 0.1)
    adamw = lambda lr: optax.adamw(sched(lr), b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10)
    labels = lambda t: jax.tree_util.tree_map_with_path(
        lambda path, _: "lora" if "lora" in jax.tree_util.keystr(path) else "rest", t)
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.multi_transform({"lora": adamw(a.lora_lr), "rest": adamw(a.lr)}, labels))


def run(a, overfit=False):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tc, policy = base_policy()
    model, cfg = lora_model(tc, nnx.state(policy._model).to_pure_dict(), a.seed)
    policy._model = None
    graphdef, train, frozen = split(model, cfg, a.freeze_siglip)
    del model
    n_train = counts(train)
    if a.freeze_siglip:
        img = [k for k in traverse_util.flatten_dict(train.to_pure_dict(), sep="/") if "/img/" in f"/{k}"]
        assert not img, f"--freeze-siglip left vision weights trainable: {img[:3]}"
    print(f"loaded in {time.time() - t0:.0f} s; trainable {n_train}", flush=True)
    start = 0
    if a.resume_from:  # warm restart from a --save-every checkpoint (weights only) [ours, 2026-11-04: move off move5]
        start = a.resume_step
        pure = serialization.from_bytes(train.to_pure_dict(),
                                        (Path(a.resume_from) / f"trainable_{start}.msgpack").read_bytes())
        train.replace_by_pure_dict(jax.tree.map(jnp.asarray, pure))
        print(f"resumed weights from {a.resume_from} step {start}", flush=True)

    eps = pd.read_csv(fd.TABLE / "episodes.csv")
    labels = pd.read_parquet(a.labels) if a.labels else None
    assert (a.arm == "advantage") == (labels is not None), "arm C needs --labels, arm A must not have them"
    ep_list = [int(x) for x in Path(a.episodes).read_text().split() if not x.startswith("#")] if a.episodes else None
    sel = fd.select_episodes(eps, a.arm, ep_list)
    info = {"arm": a.arm, "episodes": len(sel), "successes": int(sel.success.sum()),
            "fresh": int((sel.init_state < 0).sum()), "labels": a.labels, "tag_dropout": a.tag_dropout}
    if labels is not None:
        lab = labels[labels.episode_index.isin(sel.episode_index)]
        info["label_positive_share"] = float(lab.positive.mean())
    print("data", info, flush=True)

    B, mb = a.batch, a.batch // a.accum
    assert B % a.accum == 0
    opt = optimizer(a, train)
    state = opt.init(train)
    if start:  # lr schedule continues at `start`; Adam moments restart from zero (not saved), bias correction fresh
        is_s = lambda x: isinstance(x, optax.ScaleByScheduleState)
        n_s = sum(is_s(x) for x in jax.tree.leaves(state, is_leaf=is_s))
        assert n_s == 2, f"expected 2 lr schedules (lora, rest), found {n_s}"
        state = jax.tree.map(lambda x: x._replace(count=jnp.full_like(x.count, start)) if is_s(x) else x,
                             state, is_leaf=is_s)
    loss_fn = make_loss(graphdef)

    @jax.jit
    def grad_fn(train, frozen, b, rng):
        return jax.value_and_grad(loss_fn)(train, frozen, b, rng, True)

    add = jax.jit(lambda acc, g: jax.tree.map(jnp.add, acc, g), donate_argnums=0)  # running sum: one gradient copy

    @jax.jit
    def update_fn(train, state, gsum):
        g = jax.tree.map(lambda x: x / a.accum, gsum)
        upd, state = opt.update(g, state, train)
        return optax.apply_updates(train, upd), state, optax.global_norm(g)

    eval_loss = jax.jit(lambda train, frozen, b, rng: loss_fn(train, frozen, b, rng, False))
    inf_tag = True if a.arm == "advantage" else None
    val = batch_arrays(policy, raw_samples(16, 16, tag=inf_tag, split_name="val", seed=1))  # 256 fixed frames

    def val_loss(train):
        ls = [float(eval_loss(train, frozen, jax.tree.map(lambda x: x[i:i + mb], val), jax.random.PRNGKey(7 + i)))
              for i in range(0, len(val["actions"]), mb)]
        return float(np.mean(ls))

    if overfit:
        fixed = batch_arrays(policy, raw_samples(4, 8))
        batches = iter(lambda: (fixed, None), None)
    else:
        stream = fd.EpisodeStream(sel, B, a.seed, group=a.group, labels=labels, tag_dropout=a.tag_dropout)
        from concurrent.futures import ThreadPoolExecutor
        # next(raw_it) on one 1-worker executor (a generator cannot be advanced from two threads at once:
        # 18269064 died with "generator already executing"; also keeps batch order deterministic), batch_arrays
        # on two workers.
        pool, raw_pool = ThreadPoolExecutor(2), ThreadPoolExecutor(1)
        raw_it = iter(stream)
        if start:  # skip the batches already trained on, so the epoch continues where it was [ours]
            t_skip = time.time()
            for _ in range(start):
                next(raw_it)
            print(f"skipped {start} batches in {time.time() - t_skip:.0f} s (epoch {stream.epoch})", flush=True)
        prep = lambda rf: (lambda s, m: (batch_arrays(policy, s), m))(*rf.result())
        submit = lambda: pool.submit(prep, raw_pool.submit(next, raw_it))
        futs = [submit() for _ in range(2)]

        def gen():
            while True:
                f = futs.pop(0)
                futs.append(submit())
                yield f.result()
        batches = gen()

    hist, tags_seen = [], np.zeros(3, np.int64)
    if start:
        hist = [h for h in json.loads((Path(a.resume_from) / "history.json").read_text()) if h["step"] <= start]
    v0 = val_loss(train)
    hist.append({"step": start, "val_loss": v0, "lora_b_rms": lora_b_rms(train), **({"resumed": True} if start else {})})
    print(f"step {start} val {v0:.4f} ({time.time() - t0:.0f} s)", flush=True)
    key, t_log, losses = jax.random.PRNGKey(a.seed), time.time(), []
    for s in range(start + 1, a.steps + 1):
        b, meta = next(batches)
        if meta is not None:
            tags_seen += np.bincount(meta[:, 2] + 1, minlength=3)
        gsum, ls = None, []
        for k in range(a.accum):
            sub = jax.tree.map(lambda x: x[k * mb:(k + 1) * mb], b)
            rng = jax.random.PRNGKey(0) if overfit else jax.random.fold_in(key, s * a.accum + k)
            l, g = grad_fn(train, frozen, sub, jax.random.fold_in(rng, k) if overfit else rng)
            gsum = g if gsum is None else add(gsum, g)
            ls.append(l)
        train, state, gn = update_fn(train, state, gsum)
        losses.append(float(np.mean([float(x) for x in ls])))
        if s % a.log_every == 0:
            fps = a.log_every * B / (time.time() - t_log); t_log = time.time()
            print(f"step {s} loss {np.mean(losses[-a.log_every:]):.4f} grad {float(gn):.3f} {fps:.1f} frames/s"
                  + (f" epoch {stream.epoch}" if not overfit else ""), flush=True)
        if s % a.eval_every == 0 or s == a.steps:
            h = {"step": s, "train_loss": float(np.mean(losses[-a.eval_every:])), "val_loss": val_loss(train),
                 "lora_b_rms": lora_b_rms(train), "elapsed_s": time.time() - t0}
            hist.append(h)
            print(h, flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
        if a.save_every and s % a.save_every == 0 and s < a.steps:
            save(train, out / f"trainable_{s}.msgpack")
    res = {"args": vars(a), "data": info, "trainable": n_train, "history": hist, "loss_first": losses[0],
           "loss_last": float(np.mean(losses[-10:])),
           "tags_seen": {"none": int(tags_seen[0]), "negative": int(tags_seen[1]), "positive": int(tags_seen[2])},
           "minutes": (time.time() - t0) / 60}
    if overfit:
        res["pass"] = res["loss_last"] < 0.2 * losses[0]
        print(f"Check C: loss {losses[0]:.4f} -> {res['loss_last']:.4f}: {'PASS' if res['pass'] else 'FAIL'}")
    else:
        save(train, out / "trainable_final.msgpack")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--overfit", action="store_true")
    g.add_argument("--train", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=fd.ARMS, default="imitation")
    ap.add_argument("--labels", help="arm C: parquet episode_index, frame_index, positive (0/1)")
    ap.add_argument("--episodes", help="optional file of train episode indices (default: every train episode)")
    ap.add_argument("--tag-dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=2.5e-5, help="non-LoRA trainable weights (openpi LoRA peak lr)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--group", type=int, default=48, help="episodes held in memory at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume-from", default=None, help="run dir with trainable_<--resume-step>.msgpack: warm restart "
                    "(weights + lr schedule position + data position; Adam moments are not saved and restart at 0)")
    ap.add_argument("--resume-step", type=int, default=0)
    ap.add_argument("--freeze-siglip", action="store_true", help="also freeze the vision encoder (recipe variant)")
    a = ap.parse_args()
    if a.overfit:
        a.steps, a.warmup, a.eval_every, a.log_every = min(a.steps, 300), 20, 100, 20
    if a.selftest:
        selftest(a)
    else:
        run(a, overfit=a.overfit)


if __name__ == "__main__":
    main()
