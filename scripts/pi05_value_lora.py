"""Stage 3b, LoRA arm (docs/libero_plan.md, 2026-10-31): arm B's value expert with LoRA adapters on the pi05 backbone.

Usage (openpi env; see scripts/pi05_value_lora.sbatch):
    .venv/bin/python <value-learning>/scripts/pi05_value_lora.py --selftest --out <dir>      # one GPU, real weights
    .venv/bin/python <value-learning>/scripts/pi05_value_lora.py --overfit --out <dir>       # Check C
    .venv/bin/python <value-learning>/scripts/pi05_value_lora.py --train --augment --out <dir>

Why. Every value model on *frozen* pi05 features has the within-episode noise that fails 3b (head swap, per-layer
heads, objective changes), while the ResNet V it is compared with fine-tunes its backbone (lr 1e-5), and on Mulligan
frozen vs fine-tuned ResNet was MAE 0.234 vs 0.121. RECAP / pi*0.6 (arXiv 2511.14759, Sec. IV-A, V-C, V-D) trains its
value function as a VLM "initialized from" a pretrained VLM and "finetune[s]" it each iteration, with Monte Carlo
201-bin CE; it never describes a frozen backbone. So this arm changes one thing relative to B.

Model = scripts/pi05_value_expert.py arm B unchanged (width 256 / MLP 1024 value expert, one query token, 201-bin
value head + outcome head, loss = sum, openpi augmentation on fit batches, single frame, no time input), except that
the PaliGemma LLM is openpi's `gemma_2b_lora` variant (rank 16, alpha 16 on attention and FFN of every layer) and
those LoRA weights train with the expert. Frozen: all pretrained LLM weights, SigLIP, the action expert (never run).
[ours: LoRA rather than full fine-tuning; SigLIP frozen (openpi's own LoRA setup); lora_b zero-initialised (openpi
initialises both factors N(0, 0.01)), so step 0 is exactly the frozen backbone; LoRA lr = expert lr / 10, the ResNet
V's backbone / head ratio and the Mulligan low-lr lesson.] Backbone weights stay bf16, LoRA weights are float32.

Selection [ours, 2026-10-31 value-objective finding: inner-CE early stopping stops early and favours a timing / layout
fit]: every eval records inner CE *and* inner MAE of E[V]; the trainable weights of every eval are saved
(ckpt_<step>.msgpack), and val predictions are written for both the best-CE and the best-MAE step
(val_predictions{,_mae}.parquet, same columns as arm B so scripts/libero_seed_table.py scores them unchanged).

--selftest: (1) with lora_b = 0 the LoRA backbone's K/V equal the frozen pi05 K/V (within 2% of max |K|: bf16, separate XLA programs), (2) a non-zero
lora_b changes them, (3) one gradient step gives non-zero gradients on LoRA and expert weights.
--overfit = Check C on 2 train episodes (as pi05_value_expert.py: CE < 0.05, MAE < 0.0025).
--subtask-weight W > 0: E2 (docs/libero_plan.md 2026-11-02; as pi05_value_expert.py): the expert's third head predicts
the pots on the stove (vl.libero_subtask), CE weight W added to the loss; its gradient also reaches the LoRA weights.
Value target and selection unchanged; inner subtask accuracy logged; val predictions gain `subtask`, `p_subtask{0,1,2}`.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, serialization, traverse_util
from openpi.models import gemma as _gemma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_value_expert import (N_SUBTASK, CENTERS, CKPT, DATASET, Frames, ValueExpert, backbone_kv, fit_temperature,  # noqa: E402
                               inner_states, truncate_kv, value_config)

LORA = r".*lora.*"


def load(seed: int, reference: bool = False):
    """(policy for its input transform, LoRA graphdef, LoRA state float32, frozen rest, gemma base config); with
    reference=True also the unmodified pi05 model built from the same weight arrays (no second checkpoint load)."""
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as _config
    tc = _config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(tc, CKPT)
    base_params = nnx.state(policy._model).to_pure_dict()
    policy._model = None  # weights live on in base_params
    cfg = dataclasses.replace(tc.model, paligemma_variant="gemma_2b_lora")
    shapes = nnx.state(nnx.eval_shape(cfg.create, jax.random.key(0))).to_pure_dict()
    flat, base = traverse_util.flatten_dict(shapes, sep="/"), traverse_util.flatten_dict(base_params, sep="/")
    key, merged = jax.random.PRNGKey(seed + 2000), {}
    for k, s in flat.items():
        if "lora" not in k:
            merged[k] = base[k]
        elif k.endswith("lora_b"):
            merged[k] = jnp.zeros(s.shape, jnp.float32)
        else:
            key, sub = jax.random.split(key)
            merged[k] = 0.01 * jax.random.normal(sub, s.shape, jnp.float32)
    assert set(base) <= set(flat), sorted(set(base) - set(flat))[:5]
    model = cfg.load(traverse_util.unflatten_dict(merged, sep="/"), remove_extra_params=False)
    graphdef, lora, rest = nnx.split(model, nnx_utils.PathRegex(LORA), ...)
    names = [k for k in traverse_util.flatten_dict(lora.to_pure_dict(), sep="/")]
    assert names and all("PaliGemma/llm" in k and "_1" not in k.split("/")[3] for k in names), names[:4]
    out = policy, graphdef, lora, rest, _gemma.get_config(tc.model.paligemma_variant)
    return (*out, tc.model.load(base_params)) if reference else out


def lora_kinds(lora) -> dict:
    """Parameter counts per LoRA tensor name (lora_a / lora_b / gating_einsum_lora ...), for the log."""
    out = {}
    for k, v in traverse_util.flatten_dict(lora.to_pure_dict(), sep="/").items():
        out[k.rsplit("/", 1)[-1]] = out.get(k.rsplit("/", 1)[-1], 0) + int(v.size)
    return out


class Model:
    """LoRA backbone + value expert; trainable = {"lora": nnx.State, "expert": linen params}. The frozen weights
    (`rest`) are an explicit argument of every jitted function: closed over, they would be baked in as constants."""

    def __init__(self, graphdef, expert: ValueExpert, n_layers=None):
        self.graphdef, self.expert, self.n_layers = graphdef, expert, n_layers

    def kv(self, lora, rest, obs, rng=None):
        kv, mask = backbone_kv(nnx.merge(self.graphdef, lora, rest), obs, rng)
        # A3 (truncated + LoRA): as pi05_value_expert.py --n-layers, the expert reads only the first n layers' K/V.
        # LoRA weights of layers >= n get no gradient (their K/V are unused) and stay at lora_b = 0.
        return (truncate_kv(kv, self.n_layers), mask) if self.n_layers else (kv, mask)

    def apply(self, params, rest, obs, rng=None):
        kv, mask = self.kv(params["lora"], rest, obs, rng)
        return self.expert.apply({"params": params["expert"]}, kv, mask)


def observation(policy, f, idx):
    from openpi.models import model as _model
    xs = [policy._input_transform(f.obs(i)) for i in idx]
    return _model.Observation.from_dict(jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *xs))


def build(a, policy, graphdef, lora, rest, base, f0):
    n = getattr(a, "n_layers", None)
    if n:
        assert 1 <= n <= base.depth, n
        print(f"truncated: value expert reads backbone layers 0..{n - 1} of {base.depth} (A3)", flush=True)
        base = dataclasses.replace(base, depth=n)
    expert = ValueExpert(base, value_config(base, a.width, a.mlp_dim), subtask=getattr(a, "subtask_weight", 0) > 0)
    m = Model(graphdef, expert, n)
    kv0, mask0 = jax.jit(m.kv)(lora, rest, observation(policy, f0, [0]))
    eparams = expert.init(jax.random.PRNGKey(a.seed), kv0, mask0)["params"]
    return m, {"lora": lora, "expert": eparams}


def optimizer(a):
    """AdamW, warmup + cosine to 0.1x; expert lr a.lr, LoRA lr a.lora_lr."""
    sched = lambda lr: optax.warmup_cosine_decay_schedule(0.0, lr, a.warmup, a.steps, lr * 0.1)
    labels = lambda p: {"lora": jax.tree.map(lambda _: "lora", p["lora"]),
                        "expert": jax.tree.map(lambda _: "expert", p["expert"])}
    return optax.multi_transform({"lora": optax.adamw(sched(a.lora_lr), weight_decay=a.weight_decay),
                                  "expert": optax.adamw(sched(a.lr), weight_decay=a.weight_decay)}, labels)


def loss_fn(params, m, rest, obs, bins, success, rng=None, outcome_weight=1.0, sub=None, subtask_weight=0.0):
    logits, ologits, _, *sl = m.apply(params, rest, obs, rng)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, bins).mean()
    oce = optax.softmax_cross_entropy_with_integer_labels(ologits, success).mean()
    if sub is None:
        return ce + outcome_weight * oce, (ce, oce, logits)
    sce = optax.softmax_cross_entropy_with_integer_labels(sl[0], sub).mean()
    return ce + outcome_weight * oce + subtask_weight * sce, (ce, oce, logits, sce)


def selftest(a):
    from openpi.shared import nnx_utils
    t0 = time.time()
    policy, graphdef, lora, rest, base, ref = load(a.seed, reference=True)
    print(f"loaded in {time.time() - t0:.0f} s; LoRA params by tensor: {lora_kinds(lora)}", flush=True)
    fr = Frames([0], 40)
    obs = observation(policy, fr, list(range(4)))
    (k0, v0), mask0 = nnx_utils.module_jit(backbone_kv.__get__(ref))(obs)
    del ref
    m, params = build(a, policy, graphdef, lora, rest, base, fr)
    kvf = jax.jit(m.kv)
    (k1, v1), mask1 = kvf(params["lora"], rest, obs)
    valid = np.asarray(mask0)[None, :, :, None, None]
    d = lambda x, y: float(np.abs(np.where(valid, np.asarray(x, np.float32) - np.asarray(y, np.float32), 0)).max())
    scale = float(np.abs(np.asarray(k0, np.float32)).max())
    err = max(d(k0, k1), d(v0, v1))
    assert bool((mask0 == mask1).all()) and err <= 2e-2 * scale, f"zero-LoRA backbone differs from pi05 by {err} (|K| {scale})"
    bumped = jax.tree.map(lambda x: x + 0.01, params["lora"])
    (k2, _), _ = kvf(bumped, rest, obs)
    moved = d(k0, k2)
    assert moved > 1e-3, f"non-zero LoRA leaves K unchanged ({moved})"
    bins = jnp.asarray(fr.rows[:4, 3], jnp.int32)
    succ = jnp.asarray(fr.rows[:4, 4], jnp.int32)
    g = jax.jit(jax.grad(lambda p, r: loss_fn(p, m, r, obs, bins, succ)[0]))(params, rest)
    gn = {k: float(optax.global_norm(g[k])) for k in g}
    assert all(v > 0 and np.isfinite(v) for v in gn.values()), gn
    lb = max(float(jnp.abs(x).max()) for p, x in traverse_util.flatten_dict(g["lora"].to_pure_dict(), sep="/").items()
             if p.endswith("lora_b"))
    msg = (f"selftest ok: zero-LoRA K/V == frozen pi05 (max |diff| {err:.2e}); +0.01 LoRA moves K by {moved:.3f} (|K| max "
           f"{scale:.1f}); grad norms {gn}, max |grad lora_b| {lb:.2e}; {time.time() - t0:.0f} s")
    print(msg)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "selftest.txt").write_text(msg + "\n")


def overfit(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    policy, graphdef, lora, rest, base = load(a.seed)
    fr = Frames(a.episodes, a.every)
    m, params = build(a, policy, graphdef, lora, rest, base, fr)
    obs = [observation(policy, fr, list(range(i, min(i + a.batch, len(fr))))) for i in range(0, len(fr), a.batch)]
    rows = fr.rows
    opt = optimizer(a)
    state = opt.init(params)

    @jax.jit
    def step(params, state, rest, obs, bins, success):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, m, rest, obs, bins, success)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, aux

    print(f"{len(fr)} frames from episodes {a.episodes}, {len(obs)} batches; {time.time() - t0:.0f} s", flush=True)
    log, nb = [], len(obs)
    for s in range(1, a.steps + 1):
        j = (s - 1) % nb
        sl = slice(j * a.batch, j * a.batch + len(obs[j].state))
        params, state, aux = step(params, state, rest, obs[j], jnp.asarray(rows[sl, 3], jnp.int32),
                                  jnp.asarray(rows[sl, 4], jnp.int32))
        if s % (25 * nb) == 0 or s == 1 or s == a.steps:
            ce, oce, lg = aux
            p = np.asarray(jax.nn.softmax(lg.astype(jnp.float32), -1))
            r = {"step": s, "ce": float(ce), "outcome_ce": float(oce), "mae": float(np.abs(p @ CENTERS -
                                                                                       rows[sl, 2]).mean()),
                 "s": round(time.time() - t0)}
            log.append(r)
            print(r, flush=True)
    final = log[-1]
    passed = final["ce"] < 0.05 and final["mae"] < 0.0025
    res = {"check": "C (overfit 2 train episodes), LoRA", "setup": vars(a), "frames": len(fr), "final": final,
           "pass": passed, "log": log}
    (out / "check_c.json").write_text(json.dumps(res, indent=2))
    print(f"Check C {'PASSES' if passed else 'FAILS'}: CE {final['ce']:.4f}, MAE {final['mae']:.4f} "
          f"(last batch) -> {out / 'check_c.json'}")


def train(a):
    """As pi05_value_expert.train: fit / 5 inner states / val, every 4th frame, batches prepared on CPU threads."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)
    policy, graphdef, lora, rest, base = load(a.seed)
    log(f"backbone loaded; LoRA params {sum(lora_kinds(lora).values()) / 1e6:.1f} M")

    eps = pd.read_csv(DATASET / "episodes.csv")
    multi = "role" in eps.columns  # stage-1 demo table (scripts/libero_demo_table.py): roles fit / inner, 40 prompts
    if multi:  # no separate val set in stage 1: "val" = the held-out inner demos (selection still on inner)
        assert set(eps.role) == {"fit", "inner"}, set(eps.role)
        inner = set()  # inner demos are held-out episodes, not held-out init states
        ids = {"fit": eps[eps.role == "fit"].episode_index.tolist(),
               "inner": eps[eps.role == "inner"].episode_index.tolist(),
               "val": eps[eps.role == "inner"].episode_index.tolist()}
    else:
        tr = eps[eps.split == "train"]
        inner = inner_states(tr.init_state.values)
        assert inner == {10, 16, 22, 26, 35}, inner
        ids = {"fit": tr[~tr.init_state.isin(inner)].episode_index.tolist(),
               "inner": tr[tr.init_state.isin(inner)].episode_index.tolist(),
               "val": eps[eps.split == "val"].episode_index.tolist()}
    if a.fit_episodes:  # as pi05_value_expert.py: a subset of the fit episodes (e.g. fresh layouts), inner / val unchanged
        keep = {int(x) for x in Path(a.fit_episodes).read_text().split("\n") if x.strip() and x[0] != "#"}
        assert keep <= set(ids["fit"]), sorted(keep - set(ids["fit"]))[:5]
        ids["fit"] = [e for e in ids["fit"] if e in keep]
    if a.smoke:
        ids = {"fit": ids["fit"][:a.smoke], "inner": ids["inner"][:1], "val": ids["val"][:2]}
    fr = {k: Frames(v, a.every, multi_prompt=multi) for k, v in ids.items()}
    log("frames: " + ", ".join(f"{k} {len(v)} ({len(ids[k])} episodes)" for k, v in fr.items()))
    inner_idx = np.arange(len(fr["inner"]))[::a.inner_stride]

    m, params = build(a, policy, graphdef, lora, rest, base, fr["fit"])
    n_exp = sum(x.size for x in jax.tree.leaves(params["expert"]))
    opt = optimizer(a)
    state = opt.init(params)
    log(f"value expert {n_exp / 1e6:.1f} M; batch {a.batch}, lr {a.lr}, LoRA lr {a.lora_lr}, wd {a.weight_decay}, "
        f"{a.steps} steps, accum {a.accum}, augment {a.augment}, outcome weight {a.outcome_weight}, "
        f"subtask weight {a.subtask_weight}")
    sub_on = a.subtask_weight > 0
    if sub_on:
        assert all((f.rows[:, 7] >= 0).all() for f in fr.values()), "--subtask-weight needs raw files with goals"
        log("subtask label share (fit frames): " + ", ".join(f"{k}: {np.mean(fr['fit'].rows[:, 7] == k):.3f}"
                                                             for k in range(N_SUBTASK)))

    B = a.batch
    pool = ThreadPoolExecutor(a.workers)

    def prepare(f, idx):
        xs = [policy._input_transform(f.obs(i)) for i in idx]
        return jax.tree.map(lambda *x: np.stack(x), *xs)

    from openpi.models import model as _model
    as_obs = lambda b: _model.Observation.from_dict(jax.tree.map(jnp.asarray, b))

    assert B % a.accum == 0, (B, a.accum)
    mb = B // a.accum  # micro-batch; gradients averaged over a.accum of them = one batch-B step

    @jax.jit
    def grad_fn(params, rest, obs, bins, success, rng, sub):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, m, rest, obs, bins, success, rng,
                                                                  a.outcome_weight, sub if sub_on else None,
                                                                  a.subtask_weight)
        return g, aux[0], aux[1], (aux[3] if sub_on else jnp.zeros(()))

    @jax.jit
    def update_fn(params, state, gs):
        g = jax.tree.map(lambda *x: sum(x) / len(x), *gs)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state

    def step_fn(params, state, batch, idx, s):
        gs, ces, oces, sces = [], [], [], []
        for k in range(a.accum):
            sl = slice(k * mb, (k + 1) * mb)
            g, ce, oce, sce = grad_fn(params, rest, as_obs(jax.tree.map(lambda x: x[sl], batch)),
                                      jnp.asarray(rows_fit[idx[sl], 3], jnp.int32), jnp.asarray(rows_fit[idx[sl], 4],
                                                                                               jnp.int32),
                                      jax.random.fold_in(aug_key, s * a.accum + k) if a.augment else None,
                                      jnp.asarray(rows_fit[idx[sl], 7], jnp.int32))
            gs.append(g); ces.append(float(ce)); oces.append(float(oce)); sces.append(float(sce))
        params, state = update_fn(params, state, gs)
        return params, state, np.mean(ces), np.mean(oces), np.mean(sces)

    @jax.jit
    def forward(params, rest, obs):
        logits, ologits, _, *sl = m.apply(params, rest, obs)
        ps = jax.nn.softmax(ologits.astype(jnp.float32), -1)[:, 1]
        return logits, ps, (jax.nn.softmax(sl[0].astype(jnp.float32), -1) if sub_on else jnp.zeros((len(ps), 0)))

    def predict(params, f, idx, with_sub=False):
        chunks = [idx[i:i + B] for i in range(0, len(idx), B)]
        pad = lambda c: np.pad(c, (0, B - len(c)), mode="edge")
        futs = [pool.submit(prepare, f, pad(c)) for c in chunks[:2]]
        lg, ps, sp = [], [], []
        for n, c in enumerate(chunks):
            batch = futs[n].result()
            if n + 2 < len(chunks):
                futs.append(pool.submit(prepare, f, pad(chunks[n + 2])))
            l, p, q = forward(params, rest, as_obs(batch))
            lg.append(np.asarray(l, np.float32)[:len(c)]); ps.append(np.asarray(p)[:len(c)])
            sp.append(np.asarray(q)[:len(c)])
        if with_sub:
            return np.concatenate(lg), np.concatenate(ps), np.concatenate(sp)
        return np.concatenate(lg), np.concatenate(ps)

    def inner_scores(params):
        """inner value CE, inner MAE, inner subtask accuracy (nan without the subtask head)."""
        lg, _, sp = predict(params, fr["inner"], inner_idx, with_sub=True)
        rows = fr["inner"].rows[inner_idx]
        ce = float(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(lg),
                                                                   jnp.asarray(rows[:, 3], jnp.int32)).mean())
        p = np.asarray(jax.nn.softmax(jnp.asarray(lg), -1))
        acc = float((sp.argmax(1) == rows[:, 7]).mean()) if sub_on else float("nan")
        return ce, float(np.abs(p @ CENTERS - rows[:, 2]).mean()), acc

    def save(params, path):
        pure = {"lora": params["lora"].to_pure_dict(), "expert": params["expert"]}
        path.write_bytes(serialization.to_bytes(jax.device_get(pure)))

    def restore(params, path):
        pure = serialization.from_bytes({"lora": params["lora"].to_pure_dict(), "expert": params["expert"]},
                                        path.read_bytes())
        lora = params["lora"]
        lora.replace_by_pure_dict(jax.tree.map(jnp.asarray, pure["lora"]))
        return {"lora": lora, "expert": jax.tree.map(jnp.asarray, pure["expert"])}

    if a.init_from:  # stage 2 from a stage-1 checkpoint: LoRA + expert weights loaded, optimiser and schedule fresh
        params = restore(params, Path(a.init_from))
        log(f"initialised from {a.init_from}: inner CE / MAE at step 0 {inner_scores(params)[:2]}")
    rng = np.random.default_rng(a.seed)
    aug_key = jax.random.PRNGKey(a.seed + 1000)
    nfit, rows_fit = len(fr["fit"]), fr["fit"].rows

    def batches():
        while True:
            perm = rng.permutation(nfit)
            for i in range(0, nfit - B + 1, B):
                yield perm[i:i + B]

    it = batches()
    queue = []
    for _ in range(a.workers + 1):
        idx = next(it)
        queue.append((idx, pool.submit(prepare, fr["fit"], idx)))
    history, best, tl, run_loss = [], {"ce": (np.inf, 0), "mae": (np.inf, 0)}, time.time(), []
    for s in range(1, (0 if a.eval_only else a.steps) + 1):
        idx, fut = queue.pop(0)
        nxt = next(it)
        queue.append((nxt, pool.submit(prepare, fr["fit"], nxt)))
        params, state, ce, oce, sce = step_fn(params, state, fut.result(), idx, s)
        run_loss.append((float(ce), float(oce), float(sce)))
        if s % a.log_every == 0:
            r = np.mean(run_loss, 0)
            log(f"step {s}: train CE {r[0]:.3f}, outcome CE {r[1]:.3f}, "
                + (f"subtask CE {r[2]:.3f}, " if sub_on else "")
                + f"{a.log_every * B / (time.time() - tl):.1f} frames/s")
            run_loss, tl = [], time.time()
        if s % a.eval_every == 0 or s == a.steps:
            ice, imae, iacc = inner_scores(params)
            history.append({"step": s, "inner_ce": ice, "inner_mae": imae,
                            **({"inner_subtask_acc": iacc} if sub_on else {})})
            save(params, out / f"ckpt_{s}.msgpack")
            for k, v in (("ce", ice), ("mae", imae)):
                if v < best[k][0]:
                    best[k] = (v, s)
            lb = [np.asarray(x, np.float32) for p, x in
                  traverse_util.flatten_dict(params["lora"].to_pure_dict(), sep="/").items() if p.endswith("lora_b")]
            history[-1]["rms_lora_b"] = float(np.sqrt(np.mean(np.concatenate([x.ravel() ** 2 for x in lb]))))
            log(f"step {s}: inner CE {ice:.3f}, inner MAE {imae:.4f} (best CE at {best['ce'][1]}, "
                f"MAE at {best['mae'][1]}); rms lora_b {history[-1]['rms_lora_b']:.2e}"
                + (f"; inner subtask acc {iacc:.3f}" if sub_on else ""))
            tl = time.time()

    if a.eval_only:  # recovery: score an existing <out>/ckpt_<step>.msgpack, no training
        assert (out / f"ckpt_{a.eval_only}.msgpack").exists(), a.eval_only
        best = {"ce": (np.nan, a.eval_only), "mae": (np.nan, a.eval_only)}
        log(f"eval only: step {a.eval_only}")
    val = fr["val"]
    r = val.rows
    res = {"setup": vars(a), "expert_params_M": n_exp / 1e6, "lora_params_M": sum(lora_kinds(lora).values()) / 1e6,
           "frames": {k: len(v) for k, v in fr.items()}, "inner_states": sorted(inner),
           "inner_eval_frames": int(len(inner_idx)), "history": history}
    y = r[:, 3].astype(np.int32)
    for sel, suffix in (("ce", ""), ("mae", "_mae")):
        step = best[sel][1]
        p_ = restore(params, out / f"ckpt_{step}.msgpack")
        lg_in, _ = predict(p_, fr["inner"], inner_idx)
        temp = fit_temperature(lg_in, fr["inner"].rows[inner_idx, 3])
        lg, ps, sp = predict(p_, val, np.arange(len(val)), with_sub=True)
        p, pt = jax.nn.softmax(jnp.asarray(lg), -1), jax.nn.softmax(jnp.asarray(lg) / temp, -1)
        ce_of = lambda x: float(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(x),
                                                                                jnp.asarray(y)).mean())
        res[f"select_{sel}"] = {"best_step": step, "temperature": temp,
                                "val": {"ce": ce_of(lg), "ce_T": ce_of(lg / temp),
                                        "mae": float(np.abs(np.asarray(p) @ CENTERS - r[:, 2]).mean()),
                                        **({"subtask_acc": float((sp.argmax(1) == r[:, 7]).mean())}
                                           if sub_on else {})}}
        pd.DataFrame({"episode_index": r[:, 0].astype(int), "init_state": r[:, 5].astype(int),
                      "frame_index": r[:, 1].astype(int), "T": r[:, 6].astype(int), "success": r[:, 4].astype(int),
                      "value": r[:, 2], "bin": y, "ev_expert": np.asarray(p) @ CENTERS,
                      "ev_expert_T": np.asarray(pt) @ CENTERS, "ps_expert": ps,
                      **({"subtask": r[:, 7].astype(int)} | {f"p_subtask{k}": sp[:, k] for k in range(N_SUBTASK)}
                         if sub_on else {})}).to_parquet(
            out / f"val_predictions{suffix}.parquet")
        np.save(out / f"val_probs{suffix}.npy", np.asarray(p).astype(np.float16))
        log(f"select by inner {sel}: step {step}, val {res[f'select_{sel}']['val']}, T {temp:.2f}")
    res["best_step"], res["val"] = res["select_ce"]["best_step"], res["select_ce"]["val"]  # arm B's keys
    res["minutes"] = (time.time() - t0) / 60
    (out / "results.json").write_text(json.dumps(res, indent=2))
    log(f"done -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--overfit", action="store_true", help="Check C")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="--train on this many fit episodes (+1 inner, 2 val)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--width", type=int, default=256, help="arm B")
    ap.add_argument("--mlp-dim", type=int, default=1024, help="arm B")
    ap.add_argument("--lr", type=float, default=1e-4, help="value expert + heads (arm B)")
    ap.add_argument("--lora-lr", type=float, default=1e-5)
    ap.add_argument("--subtask-weight", type=float, default=0.0,
                    help="--train: E2 pots-on-stove auxiliary CE weight (SeeQ: 0.1); 0 = off (no subtask head)")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=5400, help="arm B")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--init-from", default=None, metavar="CKPT",
                    help="--train: start from a stage-1 ckpt_<step>.msgpack (LoRA + expert), fresh optimiser")
    ap.add_argument("--accum", type=int, default=1, help="--train: gradient accumulation, micro-batch batch / accum")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-only", type=int, default=0, metavar="STEP",
                    help="--train recovery: skip training, write val predictions from <out>/ckpt_STEP.msgpack "
                         "(both selections = STEP; history empty)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--inner-stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=None, help="A3: value expert reads only the first N backbone "
                    "layers' K/V and has depth N (as pi05_value_expert.py; default all 18)")
    ap.add_argument("--augment", action="store_true", help="openpi train-time augmentation on fit (arm B: on)")
    ap.add_argument("--outcome-weight", type=float, default=1.0)
    ap.add_argument("--dataset", default=None,
                    help="episode table + frame targets (default Stage 2; scripts/libero_fresh_table.py for fresh)")
    ap.add_argument("--fit-episodes", default=None, metavar="FILE",
                    help="--train: fit on these episode indices only (one per line, '#' comments)")
    a = ap.parse_args()
    if a.dataset:  # Frames / train read the module global of pi05_value_expert and this module's imported copy
        import pi05_value_expert
        global DATASET
        DATASET = pi05_value_expert.DATASET = Path(a.dataset)
    if a.selftest:
        selftest(a)
    if a.overfit:
        overfit(a)
    if a.train:
        train(a)


if __name__ == "__main__":
    main()
