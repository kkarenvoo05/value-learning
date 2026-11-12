"""Stage 3b (docs/libero_plan.md): a value expert grafted onto frozen pi05_libero.

Usage (openpi env; see scripts/pi05_value_expert.sbatch):
    .venv/bin/python <value-learning>/scripts/pi05_value_expert.py --selftest           # CPU, dummy sizes
    .venv/bin/python <value-learning>/scripts/pi05_value_expert.py --overfit --out <dir>  # Check C, one GPU

Model. The value expert reads the backbone the way pi0's action expert does (`Pi0.sample_actions`): the prefix (both
images + prompt + tokenized state, openpi's own LIBERO input transform) is run once through the frozen PaliGemma LLM,
which returns every layer's keys and values; the expert is a second set of transformer weights in openpi's
`gemma.Module` (expert index 1; index 0 = PaliGemma is never run here, so it has no parameters) whose tokens attend
to that cache layer by layer. Same depth and attention-head layout as the action expert (18 layers, 8 query heads,
1 KV head, head dim 256; required to share the backbone's attention), narrower width [ours: width 512, MLP 2048].
**One learned query token** at position = number of valid prefix tokens (as the first suffix token in pi0), attending
to the valid prefix tokens and itself. No flow matching, no adaRMS. Its final-norm output feeds two linear heads
[ours]: 201-bin value logits (cross-entropy on the target bin, as in vl.libero_probe) and 2-way P(success) (cross-
entropy on the episode outcome; reported with the per-outcome clock like the 3a `outcome+clock` probe). Loss = sum of
the two [ours]. The backbone is frozen by construction: its KV cache is computed in a separate jitted function and
only the expert, query and heads are parameters of the optimised model.

Inputs: single frame, no elapsed time (Karen, 2026-10-29: single frame first; history and elapsed time as later arms).

--selftest (CPU, gemma "dummy" sizes, random weights): (1) cached path == one joint forward of prefix + query token
through a two-expert gemma.Module with the same weights; (2) changing padded prefix positions' K/V does not change
the output, changing valid ones does.
--overfit = Check C: the first 2 train episodes (every 4th frame, as the 3a probe's Check C), backbone KV computed
once and kept on the GPU, full-batch AdamW; reports train CE / MAE / outcome accuracy. Pass [ours, same bar as 3a]:
CE < 0.05 nats and MAE below half a bin (0.0025).
--train --augment: openpi's own train-time image augmentation (`preprocess_observation(train=True)`, what pi05 was
trained with: main camera 95% random crop + resize + rotate +-5 deg, every camera ColorJitter brightness 0.3 /
contrast 0.4 / saturation 0.5) on fit batches only; inner and val are never augmented [ours: openpi's scheme rather
than LeRobot's, so the frozen backbone sees its training distribution].
--blank-state: the robot state is replaced by zeros in every frame (fit, inner, val), so its prompt tokens are a
constant and the prefix carries no proprioception [ours, 2026-10-30 progress-probe follow-up: the ResNet V sees only
the two images]. --time-input: elapsed time enters as Fourier features of t / T_MAX (t = frame index), projected and
added to the query token [ours; the ResNet V has no time input, so a win here needs a ResNet + time control].
--outcome-weight 0: value CE only, as the ResNet V (2026-10-30 head-swap follow-up [ours]: does the episode-constant
outcome loss make the expert's h noisy within episodes?).
--subtask-weight W > 0: E2 (docs/libero_plan.md 2026-11-02, SeeQ Table 4 "prediction without conditioning"): a third
linear head on h predicts the pots on the stove (vl.libero_subtask, 0 / 1 / 2), CE weight W (SeeQ: 0.1) added to the
loss. Value target, selection (inner value CE) and val metrics unchanged; inner subtask accuracy logged, val
predictions gain `subtask` and `p_subtask{0,1,2}`.
--subtask-scoped: B1 on pi05 (as vl.libero_resnet --subtask-scoped; Karen: go 2026-11-05): the value target is the
value of finishing the *current* subtask (vl.libero_subtask.scoped_targets: place the first pot, then the second;
boundary = vl.libero_milestone's milestone frame) and the value head outputs 2 x 201 logits, of which the true
scope's half is read (scope from the simulator's goal predicates, as the ResNet B1). Selection and val metrics are on
the scoped target; val predictions add `scope` and `value_task` (the whole-task target). The outcome head is kept
(same recipe as the whole-task pi05 runs, so only the target changes) [ours: MC, not SeeQ's TD].
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from openpi.models import gemma as _gemma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.libero_subtask import N_SUBTASK, scoped_targets, subtask_labels  # noqa: E402  (numpy only)

CKPT = "/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero"
DATASET = Path("/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2")
RUN = Path("/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849")
N_BINS = 201
T_MAX = 520  # Stage 2 episode cap (steps)
CENTERS = np.linspace(-1, 0, N_BINS)


def value_config(base: _gemma.Config, width: int = 512, mlp_dim: int = 2048) -> _gemma.Config:
    """Backbone depth and attention-head layout, narrower width [ours]."""
    return dataclasses.replace(base, width=width, mlp_dim=mlp_dim, lora_configs={})


def time_features(t, n: int = 8):
    """(b, 2n + 1): t / T_MAX and sin / cos of it at n octave frequencies."""
    x = jnp.asarray(t, jnp.float32)[:, None] / T_MAX
    f = jnp.pi * 2.0 ** jnp.arange(n)
    return jnp.concatenate([x, jnp.sin(f * x), jnp.cos(f * x)], -1)


class ValueExpert(nn.Module):
    """One query token through expert 1 of a two-expert gemma.Module, attending to the backbone's KV cache."""

    backbone: _gemma.Config
    expert: _gemma.Config
    dtype: str = "bfloat16"
    time_input: bool = False
    subtask: bool = False  # E2: a third head; __call__ then returns 4 values (+ subtask logits)
    scoped: bool = False  # B1: 2 x N_BINS value logits; `scope` (0 / 1 per row) picks the half

    @nn.compact
    def __call__(self, kv_cache, prefix_mask, t=None, scope=None):
        b = prefix_mask.shape[0]
        query = jnp.broadcast_to(self.param("query", nn.initializers.normal(0.02), (1, 1, self.expert.width)),
                                 (b, 1, self.expert.width))
        if self.time_input:
            query = query + nn.Dense(self.expert.width, name="time_proj")(time_features(t))[:, None]
        llm = _gemma.Module(configs=[self.backbone, self.expert], embed_dtype=self.dtype, name="llm")
        mask = jnp.concatenate([prefix_mask[:, None, :], jnp.ones((b, 1, 1), bool)], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None].astype(jnp.int32)
        (_, out), _ = llm([None, query], positions=positions, mask=mask, kv_cache=kv_cache)
        h = out[:, 0].astype(jnp.float32)
        vl = nn.Dense(N_BINS * (2 if self.scoped else 1), name="value_head")(h)
        if self.scoped:
            vl = pick_scope(vl, jnp.zeros(b, jnp.int32) if scope is None else scope)
        out = vl, nn.Dense(2, name="outcome_head")(h), h
        return (*out, nn.Dense(N_SUBTASK, name="subtask_head")(h)) if self.subtask else out


def pick_scope(logits, scope):
    """(b, 2 * N_BINS) -> (b, N_BINS): row i's half `scope[i]` (0 = first pot, 1 = second pot)."""
    lg = logits.reshape(logits.shape[0], 2, N_BINS)
    return jnp.take_along_axis(lg, jnp.asarray(scope, jnp.int32)[:, None, None], axis=1)[:, 0]


def backbone_kv(self, obs, rng=None):
    """(per-layer K/V of the prefix, prefix mask) from a pi0 model, as in Pi0.sample_actions. rng given = openpi's
    train-time image augmentation."""
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    obs = _model.preprocess_observation(rng, obs, train=rng is not None)
    tokens, mask, ar = self.embed_prefix(obs)
    _, kv = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
    return kv, mask


def truncate_kv(kv, n: int):
    """First n layers of a stacked (layer, batch, seq, heads, dim) KV cache [ours, A2]. Layer i's K/V depend only on
    layers < i (causal in depth), so this equals the cache of a backbone cut after layer n - 1 (checked in --selftest);
    the value expert must then have depth n."""
    k, v = kv
    return k[:n], v[:n]


def losses(params, model, kv, pmask, bins, success, t=None, outcome_weight=1.0, sub=None, subtask_weight=0.0,
           scope=None):
    """sub given (model.subtask): + subtask_weight * subtask CE; aux gains the subtask CE as a 5th element.
    scope: B1 (model.scoped), the subtask whose value half is trained."""
    logits, ologits, *rest = model.apply({"params": params}, kv, pmask, t, scope)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, bins).mean()
    oce = optax.softmax_cross_entropy_with_integer_labels(ologits, success).mean()
    if sub is None:
        return ce + outcome_weight * oce, (ce, oce, logits, ologits)
    sce = optax.softmax_cross_entropy_with_integer_labels(rest[1], sub).mean()
    return ce + outcome_weight * oce + subtask_weight * sce, (ce, oce, logits, ologits, sce)


def selftest():
    from openpi.models.pi0 import make_attn_mask
    base = _gemma.get_config("dummy")
    vcfg = value_config(base, width=32, mlp_dim=64)
    b, t, pad = 2, 7, 3
    rng = jax.random.PRNGKey(0)
    prefix = jax.random.normal(rng, (b, t, base.width))
    pmask = jnp.array([[True] * (t - pad) + [False] * pad, [True] * t])
    model = ValueExpert(base, vcfg, dtype="float32")
    # backbone alone (expert 0) -> KV cache
    bb = _gemma.Module(configs=[base, vcfg], embed_dtype="float32")
    init = lambda *x, **kw: nn.Module.init(bb, rng, *x, method="__call__", **kw)["params"]  # gemma.Module shadows init
    bparams = init([prefix, None], positions=jnp.cumsum(pmask, 1) - 1,
                   mask=make_attn_mask(pmask, jnp.zeros(t, bool)))
    _, kv = bb.apply({"params": bparams}, [prefix, None], positions=jnp.cumsum(pmask, 1) - 1,
                     mask=make_attn_mask(pmask, jnp.zeros(t, bool)))
    vparams = model.init(jax.random.PRNGKey(1), kv, pmask)["params"]
    _, _, h = model.apply({"params": vparams}, kv, pmask)
    # joint reference: prefix + query in one forward, expert-1 weights copied from the value expert
    q = jnp.broadcast_to(vparams["query"], (b, 1, vcfg.width))
    full_mask = make_attn_mask(jnp.concatenate([pmask, jnp.ones((b, 1), bool)], 1),
                               jnp.array([False] * t + [True]))
    pos = jnp.concatenate([jnp.cumsum(pmask, 1) - 1, jnp.sum(pmask, 1, keepdims=True)], 1)
    jparams = init([prefix, q], positions=pos, mask=full_mask)
    flat = traverse_util.flatten_dict(jparams)
    flat.update({k: v for k, v in traverse_util.flatten_dict(bparams).items()})
    flat.update(traverse_util.flatten_dict(vparams["llm"]))
    (_, jout), _ = bb.apply({"params": traverse_util.unflatten_dict(flat)}, [prefix, q], positions=pos, mask=full_mask)
    err = float(jnp.abs(jout[:, 0] - h).max())
    assert err < 1e-4, f"cached path differs from the joint forward by {err}"
    # padded K/V must not matter, valid K/V must
    k, v = kv
    k_pad = k.at[:, 0, t - pad:].add(100.0)
    _, _, h_pad = model.apply({"params": vparams}, (k_pad, v.at[:, 0, t - pad:].add(100.0)), pmask)
    _, _, h_val = model.apply({"params": vparams}, (k.at[:, 0, 0].add(1.0), v.at[:, 0, 0].add(1.0)), pmask)
    assert float(jnp.abs(h_pad - h).max()) < 1e-5, "padded prefix positions leak into the value token"
    assert float(jnp.abs(h_val[0] - h[0]).max()) > 1e-3, "value token ignores the prefix"
    # truncation (A2): KV of a depth-2 backbone with the first 2 layers' weights == first 2 layers of the full cache
    nt = 2
    bcut, vcut = dataclasses.replace(base, depth=nt), dataclasses.replace(vcfg, depth=nt)
    cut = _gemma.Module(configs=[bcut, vcut], embed_dtype="float32")
    cparams = traverse_util.unflatten_dict({kk: (vv[:nt] if "layers" in kk else vv)
                                            for kk, vv in traverse_util.flatten_dict(bparams).items()})
    _, kv_cut = cut.apply({"params": cparams}, [prefix, None], positions=jnp.cumsum(pmask, 1) - 1,
                          mask=make_attn_mask(pmask, jnp.zeros(t, bool)))
    terr = max(float(jnp.abs(x - y).max()) for x, y in zip(kv_cut, truncate_kv(kv, nt)))
    assert kv[0].shape[0] == base.depth > nt and terr < 1e-5, f"truncated KV differs by {terr}"
    tmodel = ValueExpert(bcut, vcut, dtype="float32")
    tparams = tmodel.init(jax.random.PRNGKey(1), truncate_kv(kv, nt), pmask)["params"]
    tmodel.apply({"params": tparams}, truncate_kv(kv, nt), pmask)
    # B1 scoped head: scope picks the matching half of the 2 x N_BINS logits
    smodel = ValueExpert(base, vcfg, dtype="float32", scoped=True)
    sparams = smodel.init(jax.random.PRNGKey(2), kv, pmask, None, jnp.zeros(b, jnp.int32))["params"]
    assert sparams["value_head"]["kernel"].shape[-1] == 2 * N_BINS
    l0, l1, lm = (smodel.apply({"params": sparams}, kv, pmask, None, jnp.asarray(sc))[0]
                  for sc in ([0, 0], [1, 1], [0, 1]))
    assert l0.shape == (b, N_BINS) and float(jnp.abs(l0 - l1).max()) > 1e-4, "scope does not change the value logits"
    assert jnp.allclose(lm[0], l0[0]) and jnp.allclose(lm[1], l1[1]), "mixed scopes pick the wrong halves"
    raw = jnp.arange(2 * 2 * N_BINS, dtype=jnp.float32).reshape(2, 2 * N_BINS)
    assert jnp.array_equal(pick_scope(raw, jnp.array([1, 0])), jnp.stack([raw[0, N_BINS:], raw[1, :N_BINS]]))
    print("scoped head ok: scope selects its half")
    n = sum(x.size for x in jax.tree.leaves(vparams))
    print(f"truncation ok: first {nt} of {base.depth} layers, max |diff| {terr:.2e}")
    print(f"selftest ok: cached vs joint max |diff| {err:.2e}; padding ignored; {n} expert params (dummy sizes)")


class Frames:
    """Raw policy inputs (copies, uint8) + targets for some LeRobot episodes, every `every`-th frame.
    rows columns: episode_index, frame_index, value, bin, success, init_state, T (last frame index), subtask
    (vl.libero_subtask; -1 when the raw file has no `goals`), scope (B1; 0 unless scoped), value_task (the whole-task
    target; = value unless scoped). scoped: value / bin are the subtask-scoped target (vl.libero_subtask.scoped_targets,
    scale 1040 = horizon 520 + c_fail 520, checked on each episode's last frame as vl.libero_resnet.add_scope).
    multi_prompt: episodes may carry different prompts (the stage-1 demo table, 40 tasks); each frame gets its own
    episode's prompt. Off (default), every episode must share one prompt, as before."""

    def __init__(self, episodes, every: int, blank_state: bool = False, multi_prompt: bool = False,
                 scoped: bool = False):
        import pandas as pd
        eps = pd.read_csv(DATASET / "episodes.csv").set_index("episode_index")
        ft = pd.read_parquet(DATASET / "frame_targets.parquet",
                             columns=["episode_index", "frame_index", "value", "bin"]).set_index(["episode_index",
                                                                                                  "frame_index"])
        img, wrist, state, rows, self.prompt, prompts = [], [], [], [], None, []
        for e in episodes:
            r = eps.loc[e]
            with np.load(raw_path(r.raw)) as z:
                n = len(z["state"])
                img.append(z["image"][::every].copy()); wrist.append(z["wrist_image"][::every].copy())
                state.append(z["state"][::every].copy())
                self.prompt = self.prompt or str(z["prompt"])
                assert multi_prompt or str(z["prompt"]) == self.prompt
                prompts += [str(z["prompt"])] * len(range(0, n, every))
                sub = subtask_labels(z["goals"]) if "goals" in z.files else np.full(n, -1)
                if scoped:
                    from vl.libero_milestone import milestone
                    m = milestone(z["goals"])
            ts = np.arange(0, n, every)
            vt = np.array([float(ft.loc[(e, int(t))].value) for t in ts])
            bt = np.array([int(ft.loc[(e, int(t))].bin) for t in ts])
            sc, vs = np.zeros(len(ts), np.int64), vt.copy()
            if scoped:
                last = float(ft.loc[(e, n - 1)].value)
                assert np.isclose(last, 0.0 if r.success == 1 else -520 / 1040), f"episode {e}: target scale differs"
                sc, vs = scoped_targets(ts, vt, m, 1040.0)
                bt = np.rint((vs + 1.0) * (N_BINS - 1)).astype(np.int64)
                assert (bt >= 0).all() and (bt < N_BINS).all()
                self.n_reached = getattr(self, "n_reached", 0) + int(not np.isnan(m))
            for j, t in enumerate(ts):
                rows.append((e, int(t), float(vs[j]), int(bt[j]), int(r.success), int(r.init_state), n - 1,
                             int(sub[t]), int(sc[j]), float(vt[j])))
        self.img, self.wrist, self.state = np.concatenate(img), np.concatenate(wrist), np.concatenate(state)
        if blank_state:
            self.state = np.zeros_like(self.state)
        self.rows = np.array(rows, dtype=np.float64)
        self.prompts = prompts
        assert len(self.rows) == len(self.img) == len(self.prompts)

    def __len__(self):
        return len(self.rows)

    def obs(self, i):
        return {"observation/image": self.img[i], "observation/wrist_image": self.wrist[i],
                "observation/state": self.state[i], "prompt": self.prompts[i]}


def raw_path(raw) -> Path:
    """Raw episode file: absolute in a combined table (scripts/libero_fresh_table.py), else relative to Stage 2's run."""
    p = Path(raw)
    return p if p.is_absolute() else RUN / p


def inner_states(train_states, k: int = 5):
    """vl.libero_probe.inner_split's held-out train states (same rng), for early stopping. Fresh-layout episodes
    (init_state -1) are never held out, so the inner states stay those of every earlier run."""
    states = np.sort(np.unique(train_states))
    states = states[states >= 0]
    return set(np.random.default_rng(0).choice(states, size=min(k, len(states) - 1), replace=False).tolist())


def overfit(a):
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as _config
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tc = _config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(tc, CKPT)
    kv_fn = nnx_utils.module_jit(backbone_kv.__get__(policy._model))
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)

    fr = Frames(a.episodes, a.every)
    obs, rows = [fr.obs(i) for i in range(len(fr))], fr.rows
    ks, vs, masks = [], [], []
    for i in range(0, len(obs), 32):
        batch = [policy._input_transform(o) for o in obs[i:i + 32]]
        b = _model.Observation.from_dict(jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *batch))
        (k, v), m = kv_fn(b)
        ks.append(k); vs.append(v); masks.append(m)
    kv = (jnp.concatenate(ks, 1), jnp.concatenate(vs, 1))
    pmask = jnp.concatenate(masks, 0)
    bins = jnp.asarray(rows[:, 3], jnp.int32)
    success = jnp.asarray(rows[:, 4], jnp.int32)
    print(f"{len(obs)} frames from episodes {a.episodes}; KV {kv[0].shape} {kv[0].dtype}, "
          f"{int(pmask[0].sum())} valid prefix tokens, {time.time() - t0:.0f} s", flush=True)

    base = _gemma.get_config(tc.model.paligemma_variant)
    model = ValueExpert(base, value_config(base, a.width, a.mlp_dim))
    params = model.init(jax.random.PRNGKey(a.seed), jax.tree.map(lambda x: x[:, :1], kv), pmask[:1])["params"]
    n_params = sum(x.size for x in jax.tree.leaves(params))
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 50, a.steps, a.lr * 0.1)
    opt = optax.adamw(sched, weight_decay=1e-2)
    state = opt.init(params)

    @jax.jit  # data as arguments: closed-over arrays would be baked into the program as constants
    def step(params, state, kv, pmask, bins, success):
        (loss, aux), g = jax.value_and_grad(losses, has_aux=True)(params, model, kv, pmask, bins, success)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss, aux

    def evaluate(aux):
        ce, oce, logits, ologits = aux
        p = np.asarray(jax.nn.softmax(logits.astype(jnp.float32), -1))
        mae = float(np.abs(p @ CENTERS - rows[:, 2]).mean())
        acc = float((np.asarray(ologits).argmax(-1) == rows[:, 4]).mean())
        return {"ce": float(ce), "outcome_ce": float(oce), "mae": mae, "outcome_acc": acc}

    print(f"value expert: {n_params / 1e6:.1f} M params (width {a.width}, MLP {a.mlp_dim}, depth {base.depth})",
          flush=True)
    log = []
    for s in range(1, a.steps + 1):
        params, state, loss, aux = step(params, state, kv, pmask, bins, success)
        if s % 50 == 0 or s == 1:
            r = {"step": s, **evaluate(aux), "s": round(time.time() - t0)}
            log.append(r)
            print(r, flush=True)
    final = log[-1]
    passed = final["ce"] < 0.05 and final["mae"] < 0.0025
    res = {"check": "C (overfit 2 train episodes)", "episodes": a.episodes, "every": a.every, "frames": len(obs),
           "params_M": n_params / 1e6, "width": a.width, "mlp_dim": a.mlp_dim, "lr": a.lr, "steps": a.steps,
           "seed": a.seed, "final": final, "pass": passed, "log": log}
    (out / "check_c.json").write_text(json.dumps(res, indent=2))
    print(f"Check C {'PASSES' if passed else 'FAILS'}: CE {final['ce']:.4f}, MAE {final['mae']:.4f} "
          f"-> {out / 'check_c.json'}")


def fit_temperature(logits: np.ndarray, bins: np.ndarray) -> float:
    """As vl.libero_probe.fit_temperature: one softmax temperature minimising CE (grid 0.5-20, log-spaced)."""
    lg = jnp.asarray(logits, jnp.float32)
    grid = np.exp(np.linspace(np.log(0.5), np.log(20), 60))
    ce = [float(optax.softmax_cross_entropy_with_integer_labels(lg / t, jnp.asarray(bins, jnp.int32)).mean())
          for t in grid]
    return float(grid[int(np.argmin(ce))])


def log_history(path) -> list:
    """Inner-CE history from a --train run's slurm log (for --eval-only on a run cut off by the time limit)."""
    import re
    return [{"step": int(m[1]), "inner_ce": float(m[2])}
            for m in re.finditer(r"step (\d+): inner CE ([\d.]+)", Path(path).read_text())]


def train(a):
    """Full single-frame run: backbone forward on every batch (no K/V cache on disk), early stopping on the 5 inner
    train states of vl.libero_probe (selection on their value CE, never on val), then val predictions. With
    --eval-only LOG: no training, score the run's saved best.msgpack (history from LOG, its slurm log)."""
    if a.eval_only and (Path(a.out) / "results.json").exists():
        print(f"{a.out} already has results.json, nothing to do", flush=True)
        return
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    from flax import serialization
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as _config
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    tc = _config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(tc, CKPT)
    kv_fn = nnx_utils.module_jit(backbone_kv.__get__(policy._model))
    log("backbone loaded")

    eps = pd.read_csv(DATASET / "episodes.csv")
    tr = eps[eps.split == "train"]
    if a.oof_fold is not None:  # out-of-fold V for Stage 5 labels: the ResNet's folds exactly (vl.libero_oof)
        assert a.fit_episodes is None, "--oof-fold defines its own fit set"
        from vl.libero_oof import oof_split
        held, inner_m = oof_split(eps, a.oof_fold, a.oof_folds)
        inner = set(eps.init_state[inner_m].unique().tolist())
        ids = {"fit": eps[(eps.split == "train").values & ~held & ~inner_m].episode_index.tolist(),
               "inner": eps[inner_m].episode_index.tolist(),
               "val": eps[eps.split == "val"].episode_index.tolist(),
               "held": eps[held].episode_index.tolist()}
    else:
        inner = inner_states(tr.init_state.values)
        assert inner == {10, 16, 22, 26, 35}, inner  # the 3a probes' early-stopping states
        ids = {"fit": tr[~tr.init_state.isin(inner)].episode_index.tolist(),
               "inner": tr[tr.init_state.isin(inner)].episode_index.tolist(),
               "val": eps[eps.split == "val"].episode_index.tolist()}
    if a.fit_episodes:  # data-size check: a subset of the fit episodes, inner / val unchanged
        keep = {int(x) for x in Path(a.fit_episodes).read_text().split("\n") if x.strip() and x[0] != "#"}
        assert keep <= set(ids["fit"]), sorted(keep - set(ids["fit"]))[:5]
        ids["fit"] = [e for e in ids["fit"] if e in keep]
    if a.smoke:
        ids = {"fit": ids["fit"][:a.smoke], "inner": ids["inner"][:1], "val": ids["val"][:2],
               **({"held": ids["held"][:2]} if "held" in ids else {})}
    fr = {k: Frames(v, a.every, a.blank_state, scoped=a.subtask_scoped) for k, v in ids.items()}
    if a.subtask_scoped:
        assert a.subtask_weight == 0, "--subtask-scoped and --subtask-weight are separate arms"
        f = fr["fit"]
        log(f"subtask-scoped targets (fit): {f.n_reached}/{len(ids['fit'])} episodes reach the milestone, "
            f"{np.mean(f.rows[:, 8] == 0):.3f} of frames in subtask 0, {np.mean(f.rows[:, 2] != f.rows[:, 9]):.3f} retargeted")
    log("frames: " + ", ".join(f"{k} {len(v)} ({len(ids[k])} episodes)" for k, v in fr.items()))
    inner_idx = np.arange(len(fr["inner"]))[::a.inner_stride]

    B = a.batch
    pool = ThreadPoolExecutor(a.workers)

    def prepare(f, idx):
        xs = [policy._input_transform(f.obs(i)) for i in idx]
        return jax.tree.map(lambda *x: np.stack(x), *xs)

    def kv_of(batch, rng=None):
        obs = _model.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
        kv, m = kv_fn(obs) if rng is None else kv_fn(obs, rng)
        return (truncate_kv(kv, a.n_layers), m) if a.n_layers else (kv, m)

    base = _gemma.get_config(tc.model.paligemma_variant)
    if a.n_layers:  # A2 (docs/CURRENT.md, SmolVLA idea): value expert reads only the first n backbone layers
        assert 1 <= a.n_layers <= base.depth, a.n_layers
        base = dataclasses.replace(base, depth=a.n_layers)
        log(f"truncated: value expert reads backbone layers 0..{a.n_layers - 1} of {_gemma.get_config(tc.model.paligemma_variant).depth}")
    sub_on = a.subtask_weight > 0
    if sub_on:
        assert all((f.rows[:, 7] >= 0).all() for f in fr.values()), "--subtask-weight needs raw files with goals"
        log("subtask label share (fit frames): " + ", ".join(f"{k}: {np.mean(fr['fit'].rows[:, 7] == k):.3f}"
                                                             for k in range(N_SUBTASK)))
    model = ValueExpert(base, value_config(base, a.width, a.mlp_dim), time_input=a.time_input, subtask=sub_on,
                        scoped=a.subtask_scoped)
    kv0, m0 = kv_of(prepare(fr["fit"], [0]))
    params = model.init(jax.random.PRNGKey(a.seed), kv0, m0, jnp.zeros(1), jnp.zeros(1, jnp.int32))["params"]
    n_params = sum(x.size for x in jax.tree.leaves(params))
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, a.warmup, a.steps, a.lr * 0.1)
    opt = optax.adamw(sched, weight_decay=a.weight_decay)
    state = opt.init(params)
    log(f"value expert: {n_params / 1e6:.1f} M params; batch {B}, lr {a.lr}, wd {a.weight_decay}, {a.steps} steps, "
        f"augment {a.augment}, blank state {a.blank_state}, time input {a.time_input}, outcome weight {a.outcome_weight}, "
        f"subtask weight {a.subtask_weight}, subtask-scoped {a.subtask_scoped}")

    @jax.jit
    def step_fn(params, state, kv, pmask, bins, success, t, sub, scope):
        (loss, aux), g = jax.value_and_grad(losses, has_aux=True)(
            params, model, kv, pmask, bins, success, t, a.outcome_weight,
            sub if sub_on else None, a.subtask_weight, scope)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss, aux[0], aux[1], (aux[4] if sub_on else jnp.zeros(()))

    @jax.jit
    def forward(params, kv, pmask, t, scope):
        logits, ologits, _, *sl = model.apply({"params": params}, kv, pmask, t, scope)
        ps = jax.nn.softmax(ologits.astype(jnp.float32), -1)[:, 1]
        return logits, ps, (jax.nn.softmax(sl[0].astype(jnp.float32), -1) if sub_on else jnp.zeros((len(ps), 0)))

    def predict(params, f, idx, with_sub=False):
        """Value logits and P(success) for frames idx (last batch padded to B, so one compiled shape);
        with_sub: also subtask probabilities (n, N_SUBTASK)."""
        chunks = [idx[i:i + B] for i in range(0, len(idx), B)]
        futs = [pool.submit(prepare, f, np.pad(c, (0, B - len(c)), mode="edge")) for c in chunks[:2]]
        lg, ps, sp = [], [], []
        for n, c in enumerate(chunks):
            batch = futs[n].result()
            if n + 2 < len(chunks):
                c2 = chunks[n + 2]
                futs.append(pool.submit(prepare, f, np.pad(c2, (0, B - len(c2)), mode="edge")))
            kv, m = kv_of(batch)
            cp = np.pad(c, (0, B - len(c)), mode="edge")
            l, p, q = forward(params, kv, m, jnp.asarray(f.rows[cp, 1], jnp.float32), jnp.asarray(f.rows[cp, 8], jnp.int32))
            lg.append(np.asarray(l, np.float32)[:len(c)]); ps.append(np.asarray(p)[:len(c)])
            sp.append(np.asarray(q)[:len(c)])
        if with_sub:
            return np.concatenate(lg), np.concatenate(ps), np.concatenate(sp)
        return np.concatenate(lg), np.concatenate(ps)

    def inner_ce(params):
        """Inner value CE (selection) and, with the subtask head, inner subtask accuracy (else nan)."""
        lg, _, sp = predict(params, fr["inner"], inner_idx, with_sub=True)
        y = fr["inner"].rows[inner_idx, 3].astype(np.int32)
        ce = float(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(lg), jnp.asarray(y)).mean())
        acc = float((sp.argmax(1) == fr["inner"].rows[inner_idx, 7]).mean()) if sub_on else float("nan")
        return ce, acc

    rng = np.random.default_rng(a.seed)
    aug_key = jax.random.PRNGKey(a.seed + 1000)
    nfit = len(fr["fit"])

    def batches():
        while True:
            perm = rng.permutation(nfit)
            for i in range(0, nfit - B + 1, B):
                yield perm[i:i + B]

    it = batches()
    queue = []  # (indices, future of the prepared batch), a.workers + 1 batches ahead
    for _ in range(a.workers + 1):
        idx = next(it)
        queue.append((idx, pool.submit(prepare, fr["fit"], idx)))
    history, best, best_step, tl = [], np.inf, 0, time.time()
    rows_fit = fr["fit"].rows
    run_loss = []
    sub_fit = rows_fit[:, 7].astype(np.int32)
    scope_fit = rows_fit[:, 8].astype(np.int32)
    for s in range(1, 0 if a.eval_only else a.steps + 1):
        idx, fut = queue.pop(0)
        nxt = next(it)
        queue.append((nxt, pool.submit(prepare, fr["fit"], nxt)))
        kv, m = kv_of(fut.result(), jax.random.fold_in(aug_key, s) if a.augment else None)
        params, state, loss, ce, oce, sce = step_fn(params, state, kv, m, jnp.asarray(rows_fit[idx, 3], jnp.int32),
                                                    jnp.asarray(rows_fit[idx, 4], jnp.int32),
                                                    jnp.asarray(rows_fit[idx, 1], jnp.float32),
                                                    jnp.asarray(sub_fit[idx]), jnp.asarray(scope_fit[idx]))
        run_loss.append((float(ce), float(oce), float(sce)))
        if s % a.log_every == 0:
            r = np.mean(run_loss, 0)
            log(f"step {s}: train CE {r[0]:.3f}, outcome CE {r[1]:.3f}, "
                + (f"subtask CE {r[2]:.3f}, " if sub_on else "")
                + f"{a.log_every * B / (time.time() - tl):.1f} frames/s")
            run_loss, tl = [], time.time()
        if s % a.eval_every == 0 or s == a.steps:
            ice, iacc = inner_ce(params)
            history.append({"step": s, "inner_ce": ice, **({"inner_subtask_acc": iacc} if sub_on else {})})
            if ice < best:
                best, best_step = ice, s
                (out / "best.msgpack").write_bytes(serialization.to_bytes(jax.device_get(params)))
            log(f"step {s}: inner CE {ice:.3f} (best {best:.3f} at {best_step})"
                + (f", inner subtask acc {iacc:.3f}" if sub_on else ""))
            tl = time.time()

    if a.eval_only:
        history = log_history(a.eval_only)
        best_step = min(history, key=lambda h: h["inner_ce"])["step"]
        log(f"eval only: {len(history)} inner evals in {a.eval_only}, best step {best_step}")
    params = serialization.from_bytes(jax.device_get(params), (out / "best.msgpack").read_bytes())
    lg_in, _ = predict(params, fr["inner"], inner_idx)
    temp = fit_temperature(lg_in, fr["inner"].rows[inner_idx, 3])
    val = fr["val"]
    lg, ps, sp = predict(params, val, np.arange(len(val)), with_sub=True)
    p, pt = jax.nn.softmax(jnp.asarray(lg), -1), jax.nn.softmax(jnp.asarray(lg) / temp, -1)
    y = val.rows[:, 3].astype(np.int32)
    res = {"setup": vars(a), "params_M": n_params / 1e6, "frames": {k: len(v) for k, v in fr.items()},
           "inner_states": sorted(inner), "inner_eval_frames": int(len(inner_idx)), "history": history,
           "best_step": best_step, "temperature": temp,
           "val": {"ce": float(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(lg), jnp.asarray(y)).mean()),
                   "ce_T": float(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(lg) / temp,
                                                                                 jnp.asarray(y)).mean()),
                   "mae": float(np.abs(np.asarray(p) @ CENTERS - val.rows[:, 2]).mean()),
                   **({"subtask_acc": float((sp.argmax(1) == val.rows[:, 7]).mean())} if sub_on else {})},
           "minutes": (time.time() - t0) / 60}
    r = val.rows
    pd.DataFrame({"episode_index": r[:, 0].astype(int), "init_state": r[:, 5].astype(int),
                  "frame_index": r[:, 1].astype(int), "T": r[:, 6].astype(int), "success": r[:, 4].astype(int),
                  "value": r[:, 2], "bin": r[:, 3].astype(int), "ev_expert": np.asarray(p) @ CENTERS,
                  "ev_expert_T": np.asarray(pt) @ CENTERS, "ps_expert": ps,
                  **({"subtask": r[:, 7].astype(int)} | {f"p_subtask{k}": sp[:, k] for k in range(N_SUBTASK)}
                     if sub_on else {}),
                  **({"scope": r[:, 8].astype(int), "value_task": r[:, 9]} if a.subtask_scoped else {})}
                 ).to_parquet(out / "val_predictions.parquet")
    np.save(out / "val_probs.npy", np.asarray(p).astype(np.float16))
    if a.oof_fold is not None:  # same columns as vl.libero_resnet's oof_predictions.parquet, ev_expert for ev_resnet
        ho = fr["held"]
        lh, ph = predict(params, ho, np.arange(len(ho)))
        r = ho.rows
        oof = pd.DataFrame({"episode_index": r[:, 0].astype(int), "init_state": r[:, 5].astype(int),
                            "frame_index": r[:, 1].astype(int), "T": r[:, 6].astype(int),
                            "success": r[:, 4].astype(int), "value": r[:, 2], "bin": r[:, 3].astype(int),
                            "ev_expert": np.asarray(jax.nn.softmax(jnp.asarray(lh), -1)) @ CENTERS,
                            "ps_expert": ph, "fold": a.oof_fold,
                            **({"scope": r[:, 8].astype(int), "value_task": r[:, 9]} if a.subtask_scoped else {})})
        oof.to_parquet(out / "oof_predictions.parquet")
        res["oof"] = {"frames": len(oof), "episodes": int(oof.episode_index.nunique()),
                      "mae": float(np.abs(oof.ev_expert - oof.value).mean())}
        log(f"oof predictions: {res['oof']}")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    log(f"val: {res['val']}, T {temp:.2f}, best step {best_step} -> {out}")


def main():
    global DATASET
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--overfit", action="store_true", help="Check C")
    ap.add_argument("--train", action="store_true", help="full single-frame run")
    ap.add_argument("--smoke", type=int, default=0, help="--train on this many fit episodes (+1 inner, 2 val)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--inner-stride", type=int, default=4, help="early stopping on every n-th inner frame")
    ap.add_argument("--workers", type=int, default=4, help="CPU threads preparing batches")
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1], help="LeRobot episode indices (train)")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--mlp-dim", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW, --train only")
    ap.add_argument("--augment", action="store_true", help="--train: openpi train-time image augmentation on fit")
    ap.add_argument("--blank-state", action="store_true", help="--train: robot state -> zeros (no proprio tokens)")
    ap.add_argument("--time-input", action="store_true", help="--train: elapsed time added to the query token")
    ap.add_argument("--outcome-weight", type=float, default=1.0,
                    help="--train: weight of the outcome CE in the loss; 0 = value CE only, like the ResNet V (the "
                         "outcome head is then untrained and its P(success) meaningless)")
    ap.add_argument("--subtask-weight", type=float, default=0.0,
                    help="--train: E2 pots-on-stove auxiliary CE weight (SeeQ: 0.1); 0 = off (no subtask head)")
    ap.add_argument("--subtask-scoped", action="store_true",
                    help="--train: B1, subtask-scoped target, value head 2 x 201 read at the true scope")
    ap.add_argument("--dataset", default=str(DATASET),
                    help="episode table + frame targets (default Stage 2; scripts/libero_fresh_table.py for fresh)")
    ap.add_argument("--oof-fold", type=int, default=None, help="--train: out-of-fold V (vl.libero_oof, the ResNet's "
                    "folds): hold out this fold, early-stop on 5 other-fold states, write oof_predictions.parquet")
    ap.add_argument("--oof-folds", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=None, help="--train: the value expert reads only the first N "
                    "backbone layers' K/V and has depth N (A2, SmolVLA-style truncation; default all 18)")
    ap.add_argument("--fit-episodes", default=None, metavar="FILE",
                    help="--train: fit on these episode indices only (one per line, '#' comments)")
    ap.add_argument("--eval-only", default=None, metavar="LOG",
                    help="--train: skip training, score --out's best.msgpack (history from this slurm log)")
    a = ap.parse_args()
    DATASET = Path(a.dataset)
    if a.selftest:
        selftest()
    if a.overfit:
        overfit(a)
    if a.train:
        train(a)


if __name__ == "__main__":
    main()
