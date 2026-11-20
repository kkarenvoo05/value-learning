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

Discrete action loss [2026-11-06, off by default]: --discrete-weight w adds w x the cross-entropy of discretized
action tokens placed after the prompt (and its advantage tag), as pi*0.6 trains FAST tokens next to the flow expert;
--ki-stop-grad adds pi*0.6's stop gradient from the expert into the backbone. Tokenizer, attention layout and what
is paper vs ours: src/vl/discrete_actions.py. val_loss stays the flow loss (comparable to every other run); history
adds val_ce. Inference is unchanged (no discrete tokens at sampling). --selftest with these flags adds checks 5-8
(flow parity with Pi0.compute_loss, tokenizer round trip, gradient routing, CE / gradient-norm diagnostics).
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
from vl import discrete_actions as da  # noqa: E402
from vl import fast_tokens as ft  # noqa: E402
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


def make_loss(graphdef, discrete_weight: float = 0.0, ki: bool = False, n_dims: int = 7, fast: bool = False):
    """Training loss. Defaults = openpi's flow loss (Pi0.compute_loss). discrete_weight > 0 or ki [2026-11-06]: flow +
    discrete_weight * discrete action-token CE (vl.discrete_actions; ki = stop gradient from the expert into the
    backbone, pi*0.6's Knowledge Insulation)."""
    discrete = discrete_weight > 0 or ki

    def loss_fn(train, frozen, b, rng, aug: bool):
        model = nnx.merge(graphdef, train, frozen)
        w = b.pop("weight", None) if isinstance(b, dict) else None
        tk = pop_fast(b) if fast else {}
        obs, actions = as_inputs(b)
        if discrete:
            loss, ce = da.discrete_flow_loss(model, rng, obs, actions, train=aug, n_dims=n_dims, ki=ki, **tk)
        else:
            loss = model.compute_loss(rng, obs, actions, train=aug)  # (batch, action_horizon)
        per = loss.mean(axis=-1)
        if discrete:
            per = per + discrete_weight * ce
        if w is None:
            return per.mean()
        return (per * jnp.asarray(w)).mean()  # weighted arm: weights have mean 1 over the data
    return loss_fn


def make_parts(graphdef, n_dims: int = 7, fast: bool = False):
    """(flow, ce) means on a batch, eval mode (no augmentation): val diagnostics of the discrete-loss runs."""
    def parts(train, frozen, b, rng):
        model = nnx.merge(graphdef, train, frozen)
        b = dict(b)
        tk = pop_fast(b) if fast else {}
        obs, actions = as_inputs(b)
        flow, ce = da.discrete_flow_loss(model, rng, obs, actions, train=False, n_dims=n_dims, ki=False, **tk)
        return flow.mean(), ce.mean()
    return parts


def pop_fast(b) -> dict:
    """--fast-tokens [2026-11-07]: the batch's precomputed FAST tokens as discrete_flow_loss arguments."""
    return {"tokens": b.pop("fast_tokens"), "token_mask": b.pop("fast_mask"), "n_sym": ft.N_FAST}


def add_fast(b: dict, n_dims: int) -> int:
    """Host side: FAST-encode the batch's normalised action chunks (first n_dims dims) into b; returns truncations."""
    tok, mask, cut = ft.encode(np.asarray(b["actions"])[:, :, :n_dims])
    b["fast_tokens"], b["fast_mask"] = tok, mask
    return cut



def save(train, path: Path):
    path.write_bytes(serialization.to_bytes(jax.device_get(train.to_pure_dict())))


def run_args(run_dir) -> dict:
    """The run's training args: results.json (written at the end) or args.json (written at the start, 2026-11-09).
    [2026-11-09] evals of a checkpoint saved mid-training used to find no results.json and silently assume an
    unfrozen SigLIP (crash: keys {'img'} missing; 18396172, 18396976); the arm (tag default) and seed were also lost."""
    run_dir = Path(run_dir)
    for name in ("results.json", "args.json"):
        p = run_dir / name
        if p.exists():
            d = json.loads(p.read_text())
            return d["args"] if name == "results.json" else d
    raise FileNotFoundError(f"{run_dir}: no results.json or args.json, so the training args are unknown (a run "
                            "started before 2026-11-09 that is still training: evaluate it after it finishes)")


def load_finetuned(run_dir, step: str = "final", seed: int = 0):
    """(policy with the fine-tuned model, results.json). The policy keeps pi05_libero's transforms and norm stats."""
    from openpi.policies import policy as _policy
    from openpi.shared import nnx_utils
    run_dir = Path(run_dir)
    tc, policy = base_policy()
    model, cfg = lora_model(tc, nnx.state(policy._model).to_pure_dict(), seed)
    args = run_args(run_dir)
    fz = bool(args.get("freeze_siglip", False))
    graphdef, train, frozen = split(model, cfg, fz)
    pure = serialization.from_bytes(train.to_pure_dict(), (run_dir / f"trainable_{step}.msgpack").read_bytes())
    train.replace_by_pure_dict(jax.tree.map(jnp.asarray, pure))
    merged = nnx.merge(graphdef, jax.tree.map(lambda x: x.astype(jnp.bfloat16), train), frozen)
    policy._model = merged
    policy._sample_actions = nnx_utils.module_jit(merged.sample_actions)
    assert isinstance(policy, _policy.Policy)
    rj = run_dir / "results.json"
    return policy, (json.loads(rj.read_text()) if rj.exists() else {"args": args, "data": {"arm": args.get("arm")}})


def raw_samples(n_eps: int, per_ep: int, tag=None, split_name="train", seed=0):
    eps = pd.read_csv(fd.TABLE / "episodes.csv")
    rows = eps[eps.split == split_name].sort_values("episode_index").head(n_eps)
    rng, out = np.random.default_rng(seed), []
    for r in rows.itertuples():
        ep = fd.load_episode(r.raw, None, int(r.episode_index), chunk_targets="executed")  # [2026-11-14] fixed batches
        for t in rng.choice(len(ep["actions"]), per_ep, replace=False):
            out.append(fd.sample(ep, int(t), tag))
    return out


def selftest(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fd.SUBTASK_PROMPT = a.subtask_prompt
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
    if a.discrete_weight > 0 or a.ki_stop_grad:
        res["discrete"] = discrete_checks(graphdef, train, frozen, b, rng, a.discrete_dims)
    if a.fast_tokens:
        res["fast"] = fast_checks(graphdef, train, frozen, policy, rng, a.discrete_dims)
        res.setdefault("discrete", {}).setdefault("ok", []).extend(res["fast"]["ok"])
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
    res.update(ok=[ok1, ok2, ok3, ok4] + res.get("discrete", {}).get("ok", []), prompt_tokens=n_valid,
               action_err=err, roundtrip=err4)
    (out / "selftest.json").write_text(json.dumps(res, indent=2))
    assert all(res["ok"]), res


def discrete_checks(graphdef, train, frozen, b, rng, n_dims: int) -> dict:
    """--selftest with --discrete-weight / --ki-stop-grad [2026-11-06]:
    (5) the two-pass flow term equals Pi0.compute_loss (same rng, no CE weight, ki off);
    (6) the DCT tokenizer round-trips the real normalised chunk (quantisation error only, ~no clipping);
    (7) gradient routing: with ki the flow loss gives the backbone exactly zero gradient and the CE gives the expert
        side exactly zero, each nonzero on its own side; without ki the flow loss reaches the backbone;
    (8) diagnostics: CE at init (uniform = log N_SYM) and per-group gradient norms, to choose --discrete-weight."""
    flow_ref = float(jax.jit(make_loss(graphdef), static_argnums=4)(train, frozen, b, rng, False))

    def parts(ki):
        def f(tr):
            model = nnx.merge(graphdef, tr, frozen)
            obs, actions = as_inputs(b)
            flow, ce = da.discrete_flow_loss(model, rng, obs, actions, train=False, n_dims=n_dims, ki=ki)
            return flow.mean(), ce.mean()
        return f
    f_noki, f_ki = parts(False), parts(True)
    flow2, ce0 = (float(x) for x in jax.jit(f_noki)(train))
    ok5 = abs(flow2 - flow_ref) <= 0.02 * abs(flow_ref) + 1e-4
    print(f"(5) flow two-pass {flow2:.5f} vs compute_loss {flow_ref:.5f}: {'ok' if ok5 else 'FAIL'}", flush=True)

    act = np.asarray(b["actions"])
    sym = da.action_tokens(act, n_dims)
    rt = da.decode_tokens(sym, act.shape[1], n_dims)
    rt_err = float(np.sqrt(np.mean((rt - act[..., :n_dims]) ** 2)))
    clip = float(np.mean((sym == 0) | (sym == da.N_SYM - 1)))
    ok6 = rt_err < 0.03 and clip < 0.01
    print(f"(6) tokens {sym.shape[1]} per chunk, round-trip rms {rt_err:.4f}, clipped {clip:.4f}: "
          f"{'ok' if ok6 else 'FAIL'}", flush=True)

    def gnorms(fn, which):
        g = jax.jit(jax.grad(lambda tr: fn(tr)[which]))(train)
        flat = traverse_util.flatten_dict(g.to_pure_dict(), sep="/")
        sq = {True: 0.0, False: 0.0}
        for k, v in flat.items():
            sq[da.backbone_key(k)] += float(jnp.sum(jnp.square(v.astype(jnp.float32))))
        return {"backbone": sq[True] ** 0.5, "expert": sq[False] ** 0.5}
    g_flow_ki, g_ce, g_flow = gnorms(f_ki, 0), gnorms(f_ki, 1), gnorms(f_noki, 0)
    ok7 = (g_flow_ki["backbone"] == 0.0 and g_flow_ki["expert"] > 0 and g_ce["expert"] == 0.0
           and g_ce["backbone"] > 0 and g_flow["backbone"] > 0)
    print(f"(7) grad norms flow+ki {g_flow_ki}, ce {g_ce}, flow (no ki) {g_flow}: {'ok' if ok7 else 'FAIL'}",
          flush=True)
    print(f"(8) CE at init {ce0:.3f} (uniform {np.log(da.N_SYM):.3f})", flush=True)
    return {"ok": [ok5, ok6, ok7], "flow_two_pass": flow2, "flow_ref": flow_ref, "ce_init": ce0,
            "ce_uniform": float(np.log(da.N_SYM)), "tokens_per_chunk": int(sym.shape[1]), "roundtrip_rms": rt_err,
            "clipped": clip, "grad_flow_ki": g_flow_ki, "grad_ce": g_ce, "grad_flow_noki": g_flow}


def fast_checks(graphdef, train, frozen, policy, rng, n_dims: int) -> dict:
    """--selftest --fast-tokens [2026-11-07]:
    (9)  our FAST encoder (vl.fast_tokens) gives exactly the official processor's tokens on real normalised chunks
         (the downloaded processing_action_tokenizer.py imported as a plain module, read before use);
    (10) lengths: tokens per chunk on 256 train chunks, truncation rate at F_MAX (pass: < 1%);
    (11) does pi05_libero already know FAST? CE at init over the 2048 FAST rows vs uniform log 2048 (diagnostic: a
         clearly lower CE means the backbone reads and predicts FAST tokens from its pi0.5 pre-training)."""
    import importlib.util
    from transformers import PreTrainedTokenizerFast
    spec = importlib.util.spec_from_file_location("fast_processor", ft.FAST_DIR / "processing_action_tokenizer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = ft.config()
    proc = mod.UniversalActionProcessor(PreTrainedTokenizerFast(tokenizer_file=str(ft.FAST_DIR / "tokenizer.json")),
                                        scale=cfg["scale"], vocab_size=cfg["vocab_size"], min_token=cfg["min_token"])
    samples = raw_samples(32, 8, seed=3)
    bb = batch_arrays(policy, samples)
    act = np.asarray(bb["actions"])[:, :, :n_dims]
    ref = proc(act)
    tok, mask, cut = ft.encode(act)
    ours = [list(t[m]) for t, m in zip(tok, mask)]
    n_ok = sum(list(map(int, r)) == list(map(int, o)) for r, o in zip(ref, ours))
    ok9 = n_ok == len(ref)
    lens = np.array([len(r) for r in ref])
    print(f"(9) FAST encoder parity with the official processor: {n_ok}/{len(ref)}: {'ok' if ok9 else 'FAIL'}",
          flush=True)
    rt = proc.decode([list(map(int, r)) for r in ref], time_horizon=act.shape[1], action_dim=n_dims)
    rt_rms = float(np.sqrt(np.mean((rt - act) ** 2)))
    ok10 = cut / len(ref) < 0.01
    print(f"(10) tokens per chunk min/median/max {lens.min()}/{int(np.median(lens))}/{lens.max()}, truncated at "
          f"{ft.F_MAX}: {cut}/{len(ref)}, decode round-trip rms {rt_rms:.4f}: {'ok' if ok10 else 'FAIL'}", flush=True)
    parts = jax.jit(make_parts(graphdef, n_dims, fast=True))
    ces = []
    for i in range(0, 64, 8):
        sub = jax.tree.map(lambda x: x[i:i + 8], bb)
        add_fast(sub, n_dims)
        ces.append(float(parts(train, frozen, sub, rng)[1]))
    ce0 = float(np.mean(ces))
    print(f"(11) FAST CE at init {ce0:.3f} vs uniform {np.log(ft.N_FAST):.3f} "
          f"({'pi05_libero predicts FAST tokens' if ce0 < np.log(ft.N_FAST) - 1 else 'no prior FAST knowledge'})",
          flush=True)
    return {"ok": [ok9, ok10], "parity": f"{n_ok}/{len(ref)}", "len_min": int(lens.min()),
            "len_median": float(np.median(lens)), "len_max": int(lens.max()), "truncated": int(cut),
            "roundtrip_rms": rt_rms, "ce_init": ce0, "ce_uniform": float(np.log(ft.N_FAST))}


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
    (out / "args.json").write_text(json.dumps(vars(a), indent=2, default=str))  # for mid-training evals (run_args)
    t0 = time.time()
    fd.SUBTASK_PROMPT = a.subtask_prompt  # [2026-11-07] subtask text before the tag (vl.libero_finetune_data)
    fd.CHUNK_TARGETS = a.chunk_targets  # [2026-11-14] sampled = the policy's own chunk at replan frames only
    if a.table:  # [2026-11-07] RECAP round 2: a table with more rounds of rollouts (scripts/libero_round2_table.py)
        fd.TABLE = Path(a.table)
    assert not a.fast_tokens or a.discrete_weight > 0, "--fast-tokens is a tokenizer for --discrete-weight > 0"
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
    if a.init_from:  # stage 2 of the speed-tag pre-training [2026-11-06; RECAP Alg. 1: fine-tune from the pre-trained
        # checkpoint]: weights only, step 0, fresh optimiser and lr schedule. A requeue's --resume-from overrides it.
        pure = serialization.from_bytes(train.to_pure_dict(), Path(a.init_from).read_bytes())
        train.replace_by_pure_dict(jax.tree.map(jnp.asarray, pure))
        print(f"init weights from {a.init_from} (lora_b rms {lora_b_rms(train):.2e})", flush=True)
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
    weights = None
    if a.weights:  # weighted arm [Dissecting 2609.28161: continuous advantage weights; ours: column, demo weight]
        weights = pd.read_parquet(a.weights)[["episode_index", "frame_index", "weight"]]
    assert (a.arm == "weighted") == (weights is not None), "the weighted arm needs --weights, other arms must not"
    ep_list = [int(x) for x in Path(a.episodes).read_text().split() if not x.startswith("#")] if a.episodes else None
    if a.pretrain_demos:  # speed-tag pre-training [2026-11-06]: every task's fit demos, own prompts, --labels speed
        assert a.arm == "advantage" and not (a.demos or ep_list or weights is not None), \
            "--pretrain-demos: arm advantage with speed --labels only (no --demos / --episodes / --weights)"
        sel = fd.pretrain_demo_episodes(fd.DEMO_TABLE, ("fit",))
        missing = set(sel.episode_index) - set(labels.episode_index)
        assert not missing, f"--labels has no speed label for {len(missing)} demos"
    else:
        sel = fd.select_episodes(eps, a.arm, ep_list)
    n_demos = len(sel) if a.pretrain_demos else 0
    if a.demos:  # the task's LIBERO demos join the training set, tagged positive in arm C [RECAP Alg. 1 / §IV-B]
        if "task" in sel.columns and sel.task.nunique() > 1:
            # [2026-11-16, multi-task AG] one prompt per task (first episode of each); tasks without LIBERO demos in
            # DEMO_TABLE (libero_90: not in openpi's data) are skipped, so only moka's demos join
            parts = []
            for task, g in sel.groupby("task", sort=True):
                prompt = str(fd.load_episode(g.raw.iloc[0], None, 0)["prompt"])
                try:
                    parts.append(fd.demo_episodes_any([fd.DEMO_TABLE, *a.demo_table_extra], prompt))
                    print(f"--demos: {len(parts[-1])} demos for {task} ({prompt!r})")
                except AssertionError:
                    print(f"--demos: no demos for {task} ({prompt!r}); skipped")
            assert parts, "--demos: no task in the table has demos"
            demos = pd.concat(parts, ignore_index=True)
        else:
            prompt = str(fd.load_episode(sel.raw.iloc[0], None, 0)["prompt"])
            try:
                demos = fd.demo_episodes_any([fd.DEMO_TABLE, *a.demo_table_extra], prompt)
            except AssertionError:  # [2026-11-16] task-64-only chains: no libero_90 demos, train without
                print(f"--demos: no demos for {prompt!r}; skipped")
                demos = sel.iloc[:0].assign(demo=True)
        n_demos = len(demos)
        sel = pd.concat([sel.assign(demo=False), demos], ignore_index=True)
        if labels is not None:
            labels = pd.concat([labels, fd.demo_labels(demos)], ignore_index=True)
        if weights is not None:  # demos weigh 1 = the mean rollout frame [ours]
            weights = pd.concat([weights, fd.demo_labels(demos).drop(columns="positive").assign(weight=1.0)],
                                ignore_index=True)
    info = {"arm": a.arm, "episodes": len(sel), "successes": int(sel.success.sum()),
            "fresh": int((sel.init_state < 0).sum()) - n_demos, "demos": n_demos, "labels": a.labels,
            "tag_dropout": a.tag_dropout, "init_from": a.init_from,
            **({"pretrain_tasks": int(sel.task.nunique())} if a.pretrain_demos else {})}
    if a.demo_share:  # [2026-11-08] after the counts above, which count each episode once
        assert a.demos, "--demo-share needs --demos"
        sel, k, share = fd.oversample_demos(sel, a.demo_share)
        info.update(demo_repeats=k, demo_frame_share=round(share, 4))
    if a.oversample_episodes:  # [2026-11-19] data rebalancing (reports/2026-11-19-residual-failures): repeat the
        # listed episodes (whitespace-separated episode_index) k times so they are ~--oversample-share of the frames
        ids = {int(x) for x in Path(a.oversample_episodes).read_text().split()}
        sel = sel.assign(boost=sel.episode_index.isin(ids))
        assert sel.boost.any(), f"--oversample-episodes: none of {len(ids)} episodes is in the training set"
        n_boost = int(sel.boost.sum())
        sel, k, share = fd.oversample_flag(sel, "boost", a.oversample_share)
        info.update(oversample_episodes=n_boost, oversample_repeats=k, oversample_frame_share=round(share, 4))
    if a.anchor_round is not None:  # [2026-11-13] base-rollout anchor (fd.anchor_episodes), after the demo share
        assert a.arm == "imitation" and labels is None and weights is None, "--anchor-round: imitation arm only"
        anc = fd.anchor_episodes(eps, a.anchor_round)
        sel = pd.concat([sel.assign(anchor=False), anc], ignore_index=True)
        if "demo" in sel:
            sel["demo"] = sel["demo"].fillna(False).astype(bool)
        sel, k, share = fd.oversample_flag(sel, "anchor", a.anchor_share)
        info.update(anchor_round=a.anchor_round, anchor_episodes=len(anc), anchor_successes=int(anc.success.sum()),
                    anchor_repeats=k, anchor_frame_share=round(share, 4))
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
    loss_fn = make_loss(graphdef, a.discrete_weight, a.ki_stop_grad, a.discrete_dims, a.fast_tokens)
    flow_fn = make_loss(graphdef)  # val_loss = the flow loss in every run, so runs stay comparable
    discrete = a.discrete_weight > 0 or a.ki_stop_grad
    if discrete:
        print(f"discrete action loss: weight {a.discrete_weight}, ki stop-grad {a.ki_stop_grad}, dims "
              f"{a.discrete_dims}, {da.N_SYM} symbols (vl.discrete_actions)", flush=True)

    @jax.jit
    def grad_fn(train, frozen, b, rng):
        return jax.value_and_grad(loss_fn)(train, frozen, b, rng, True)

    add = jax.jit(lambda acc, g: jax.tree.map(jnp.add, acc, g), donate_argnums=0)  # running sum: one gradient copy

    @jax.jit
    def update_fn(train, state, gsum):
        g = jax.tree.map(lambda x: x / a.accum, gsum)
        upd, state = opt.update(g, state, train)
        return optax.apply_updates(train, upd), state, optax.global_norm(g)

    eval_loss = jax.jit(lambda train, frozen, b, rng: flow_fn(train, frozen, b, rng, False))
    parts_fn = jax.jit(make_parts(graphdef, a.discrete_dims, a.fast_tokens)) if discrete else None
    # --tag-dropout 1 [2026-11-07, untagged all-rollout control]: never tagged, so val prompts stay plain too
    inf_tag = True if a.arm == "advantage" and a.tag_dropout < 1 else None
    val = batch_arrays(policy, raw_samples(16, 16, tag=inf_tag, split_name="val", seed=1))  # 256 fixed frames
    n_cut = [add_fast(val, a.discrete_dims) if a.fast_tokens else 0]  # truncated chunks (val + train), logged

    def val_loss(train):
        ls = [float(eval_loss(train, frozen, jax.tree.map(lambda x: x[i:i + mb], val), jax.random.PRNGKey(7 + i)))
              for i in range(0, len(val["actions"]), mb)]
        return float(np.mean(ls))

    def val_ce(train):  # discrete runs: val CE of the action tokens (uniform over the symbols = log N_SYM)
        return float(np.mean([float(parts_fn(train, frozen, jax.tree.map(lambda x: x[i:i + mb], val),
                                             jax.random.PRNGKey(7 + i))[1]) for i in range(0, len(val["actions"]), mb)]))

    if overfit:
        fixed = batch_arrays(policy, raw_samples(4, 8))
        batches = iter(lambda: (fixed, None), None)
    else:
        stream = fd.EpisodeStream(sel, B, a.seed, group=a.group, labels=labels, tag_dropout=a.tag_dropout,
                                  action_shift=a.tag_action_shift,
                                  weights=weights)
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
        if a.fast_tokens:
            def prep(rf):
                s, m = rf.result()
                b = batch_arrays(policy, s)
                n_cut[0] += add_fast(b, a.discrete_dims)
                return b, m
        if weights is not None:
            def prep(rf):
                s, m = rf.result()
                w = np.array([x.pop("weight") for x in s], np.float32)
                b = batch_arrays(policy, s)
                b["weight"] = w
                return b, m
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
    if a.stop_after and start >= a.stop_after:  # requeued after the stop step was already saved: nothing to train
        print(f"stop-after: resumed at step {start} >= {a.stop_after}, nothing to train", flush=True)
        return
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
            if discrete:
                h["val_ce"] = val_ce(train)
            hist.append(h)
            print(h, flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
        if a.save_every and s % a.save_every == 0 and s < a.steps:
            save(train, out / f"trainable_{s}.msgpack")
        if a.stop_after and s >= a.stop_after:  # early stop; the lr schedule still spans --steps
            print(f"stop-after: stopping at step {s} of a {a.steps}-step schedule", flush=True)
            break
    res = {"args": vars(a), "data": info, "trainable": n_train, "history": hist, "loss_first": losses[0],
           "loss_last": float(np.mean(losses[-10:])),
           "tags_seen": {"none": int(tags_seen[0]), "negative": int(tags_seen[1]), "positive": int(tags_seen[2])},
           **({"fast_truncated_chunks": int(n_cut[0])} if a.fast_tokens else {}),
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
    ap.add_argument("--tag-action-shift", type=float, default=0.0, help="tag positive control [2026-11-06]: shift "
                    "action dim fd.SHIFT_DIM by +/- this (raw units) on tagged samples; use with random labels")
    ap.add_argument("--weights", help="weighted arm: parquet episode_index, frame_index, weight (mean ~1)")
    ap.add_argument("--demos", action="store_true", help="add the task's LIBERO demos (libero_demos table) to the "
                    "training set; arm C tags every demo frame positive [RECAP]")
    ap.add_argument("--oversample-episodes", help="[2026-11-19] file of episode_index values to repeat (rebalancing)")
    ap.add_argument("--oversample-share", type=float, default=0.5, help="with --oversample-episodes: their target "
                    "share of the training frames (integer repeats, so the realised share is logged)")
    ap.add_argument("--demo-table-extra", action="append", default=[], help="[2026-11-19] with --demos: more demo "
                    "tables searched after libero_demos for a task's demos (e.g. tables/libero90_t64_demos)")
    ap.add_argument("--demo-share", type=float, default=0.0, help="[2026-11-08] with --demos: repeat the demos so "
                    "they are this share of the training frames (0 = each demo once, ~2%% of frames)")
    ap.add_argument("--anchor-round", type=int, default=None, help="[2026-11-13] imitation arm: add every train "
                    "rollout of this round of --table (successes and failures; round 1 = base pi0.5) as an anchor")
    ap.add_argument("--anchor-share", type=float, default=0.5, help="with --anchor-round: anchor frame share")
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
    ap.add_argument("--stop-after", type=int, default=0, help="end training after this step while keeping the "
                    "--steps lr schedule (replicates an intermediate checkpoint of a longer run); 0 = off")
    ap.add_argument("--resume-from", default=None, help="run dir with trainable_<--resume-step>.msgpack: warm restart "
                    "(weights + lr schedule position + data position; Adam moments are not saved and restart at 0)")
    ap.add_argument("--resume-step", type=int, default=0)
    ap.add_argument("--pretrain-demos", action="store_true", help="[2026-11-06] speed-tag pre-training stage: train on "
                    "every task's fit demos (libero_demos table, own prompts) with --labels = speed labels "
                    "(scripts/libero_speed_labels.py) instead of the moka rollouts")
    ap.add_argument("--init-from", default=None, help="[2026-11-06] trainable_<step>.msgpack of a pre-training run: "
                    "start from its weights at step 0 with a fresh optimiser (stage 2)")
    ap.add_argument("--freeze-siglip", action="store_true", help="also freeze the vision encoder (recipe variant)")
    ap.add_argument("--discrete-weight", type=float, default=0.0, help="[2026-11-06] weight of the discrete action-token "
                    "CE (vl.discrete_actions, pi*0.6-style); 0 = openpi's flow loss only")
    ap.add_argument("--ki-stop-grad", action="store_true", help="[2026-11-06] stop gradient from the action expert into "
                    "the backbone (pi*0.6 Knowledge Insulation); the backbone then learns only from the discrete CE")
    ap.add_argument("--discrete-dims", type=int, default=7, help="action dims tokenized (LIBERO: 7 of the 32 padded)")
    ap.add_argument("--fast-tokens", action="store_true", help="[2026-11-07] discrete loss on the real FAST vocabulary "
                    "(vl.fast_tokens: DCT + BPE, 2048 tokens, variable length) instead of our 127-symbol code")
    ap.add_argument("--table", default=None, help="[2026-11-07] episode table (default vl.libero_finetune_data.TABLE "
                    "= moka_fixed_fresh); round 2 uses scripts/libero_round2_table.py's output")
    ap.add_argument("--subtask-prompt", action="store_true", help="[2026-11-07] subtask text before the tag "
                    "(vl.libero_finetune_data.SUBTASKS from the goal predicates; serve with the client's subtask)")
    ap.add_argument("--chunk-targets", choices=["executed", "sampled"], default="executed",
                    help="[2026-11-14] executed: a_t..a_t+9 as executed (joins separately sampled chunks); sampled: "
                    "rollouts train only on their replan frames, each with the full chunk the policy sampled there "
                    "(raw chunk_actions; vl.libero_finetune_data.CHUNK_TARGETS); demos keep executed")
    a = ap.parse_args()
    if a.overfit:
        a.steps, a.warmup, a.eval_every, a.log_every = min(a.steps, 300), 20, 100, 20
    if a.selftest:
        selftest(a)
    else:
        run(a, overfit=a.overfit)


if __name__ == "__main__":
    main()
