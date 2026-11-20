"""Advantage Guidance residual parameterization on pi0.5 (Zhou & Levine, arXiv 2610.19783, Sec. 6.2) [2026-11-11].

Paper (Eq. 5-6; code zhouzypaul/advantage-guidance utils/networks.py ConditionalAndUncondFlowField):
    v(x_t, t, s, y) = stop_grad(v_ref(x_t, t, s)) + h_theta(x_t, t, s, y)
v_ref = an unconditional BC flow, h_theta = a separate network conditioned on y through AdaLN. The reference explains
everything that does not depend on y, so the residual's only systematically learnable signal is the y-dependent part.
Sampling with contrastive guidance (Eq. 4): v = v_ref + w (h(y+) - h(y-)).

Here [ours, adaptation to a pre-trained VLA]:
  - v_ref = base pi05_libero, frozen. It is the behaviour policy of every rollout in the training table (moka_fixed_fresh:
    base pi0.5 rollouts), i.e. exactly the pi_hat of the paper's theory; the paper instead trains v_ref on the data.
    (Why not LoRA: a LoRA'd copy of pi0.5 initialised at the base and conditioned on y is the same function, init and
    loss as our naive tag fine-tunes, so the residual parameterisation would change nothing.)
  - h_theta = a small AdaLN MLP (paper: 512 x 4, AdaLN from an MLP(128, 128) of the scalar condition), applied per
    action token to [frozen action-expert output feature of that token (1024), x_t token (32), sincos(t) (64)].
    The features carry s (they attend to the image/language prefix), x_t and t. Final layer zero-initialised [ours], so
    step 0 = base pi0.5 exactly. ~1.7 M trainable parameters; nothing else trains.
  - y = +1 for a positive tag, -1 for a negative tag (our binary RECAP labels; contrastive pair y+ = 1, y- = -1 as the
    paper's defaults); no tag = pure base (h not applied). Trained with --tag-dropout 0 (the reference is separate).
  - Same data, labels, trainer, checkpoints and serving as the AdaRMS run 18355677 (pi05_policy_finetune.py patched as
    pi05_tag_adarms.py does), so against the contrastive-guidance result on that run only the parameterisation changes.
  - Velocity mix (c_pos, c_neg, c_none) as pi05_tag_adarms.TagPi0: v = (c_pos + c_neg + c_none) v_ref
    + c_pos h(+1) + c_neg h(-1); contrastive_mix(w) = (w, -w, 1) gives Eq. 4. So pi05_contrast_check.py and the
    serve path work unchanged.
  - [2026-11-13] optional head inputs (--head-state / --head-subtask, stored in residual.json "head"): the 8-d LIBERO
    robot state (observation.state[:, :8] after openpi's normalisation; pi05_libero itself never reads the state:
    discrete_state_input=False, no state_proj in pi0.5) and / or the subtask one-hot (first / second pot from the goal
    predicates; demos carry none = both 0). Concatenated to every token's input; the prompt pi0.5 sees stays the plain
    task (vl.tag_route strips the subtask text), so v_ref is unchanged. Serve: the client sends the subtask
    (pi05_tag_adarms.sbatch sets --args.send-subtask from residual.json). Without the flags nothing changes.
  - [2026-11-17] --uncond-branch (with --tag-dropout p > 0; residual.json head.uncond): every frame trains the head,
    and the tag-dropped ones fit h(0) = what the training data does relative to base pi0.5. Serve with --uncond =
    uncond_mix(w): v_ref + h(0) + w (h(+) - h(-)), so guidance steps from the data's behaviour instead of from base
    pi0.5 (AG saturation check, option 2). Untagged sampling without a mix stays base pi0.5 for every tool.
Not done (paper): action imagination (needs Q(s, a) for sampled chunks), continuous advantages with per-state
normalisation (needs imagination), training v_ref jointly.

Usage (openpi env; scripts/pi05_tag_adarms.sbatch with PY_SCRIPT=pi05_ag_residual.py):
    .venv/bin/python <vl>/scripts/pi05_ag_residual.py --residual-selftest --out <dir>
    .venv/bin/python <vl>/scripts/pi05_ag_residual.py --train --arm advantage --labels L --tag-dropout 0 --demos \
        --lr 3e-4 --accum 1 ... --out D
    .venv/bin/python <vl>/scripts/pi05_ag_residual.py --serve --run D --contrast 2 --noise-seed 1 --port P
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pi05_policy_finetune as pfm  # noqa: E402
import pi05_tag_adarms as ta  # noqa: E402
from vl import libero_finetune_data as fd  # noqa: E402
from vl import tag_route as tr  # noqa: E402

WIDTH, DEPTH, COND, TIME_DIM = 512, 4, 128, 64
STATE_DIMS = 8  # LIBERO state: eef pos (3) + axis-angle (3) + gripper qpos (2); openpi pads it to the action dim
_ResPi0 = None


def head_extra_dim(head: dict | None) -> int:
    head = head or {}
    return STATE_DIMS * bool(head.get("state")) + len(tr.MASK_SUB) * bool(head.get("subtask"))


class ResidualHead(nnx.Module):
    """h_theta(feature, x_t, t, y): per-token AdaLN MLP [AG networks.py ConditionedMLP: Dense -> LayerNorm (no affine)
    -> modulate by the condition -> GELU per layer; condition encoder MLP(128, 128) of the scalar]."""

    def __init__(self, feat_dim: int, action_dim: int, seed: int = 0, extra_dim: int = 0):
        rngs = nnx.Rngs(seed + 4000)
        self.inp = nnx.Linear(feat_dim + action_dim + TIME_DIM + extra_dim, WIDTH, rngs=rngs)
        self.c1 = nnx.Linear(1, COND, rngs=rngs)
        self.c2 = nnx.Linear(COND, COND, rngs=rngs)
        for i in range(DEPTH):
            if i:
                setattr(self, f"l{i}", nnx.Linear(WIDTH, WIDTH, rngs=rngs))
            setattr(self, f"m{i}", nnx.Linear(COND, 2 * WIDTH, rngs=rngs))
        self.out = nnx.Linear(WIDTH, action_dim, kernel_init=nnx.initializers.zeros_init(),
                              bias_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, feat, x_t, time, y, extra=None):
        from openpi.models.pi0 import posemb_sincos
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        te = posemb_sincos(f32(time), TIME_DIM, min_period=4e-3, max_period=4.0)
        te = jnp.broadcast_to(te[:, None, :], (*x_t.shape[:2], TIME_DIM))
        parts = [f32(feat), f32(x_t), te]
        if extra is not None:  # (B, E) per-sample inputs [2026-11-13], the same for every action token
            parts.append(jnp.broadcast_to(f32(extra)[:, None, :], (*x_t.shape[:2], extra.shape[-1])))
        h = self.inp(jnp.concatenate(parts, axis=-1))
        c = nnx.gelu(self.c2(nnx.gelu(self.c1(f32(y)[:, None]))))
        for i in range(DEPTH):
            if i:
                h = getattr(self, f"l{i}")(h)
            h = (h - h.mean(-1, keepdims=True)) * jax.lax.rsqrt(h.var(-1, keepdims=True) + 1e-6)  # LN, no affine
            s, b = jnp.split(getattr(self, f"m{i}")(c), 2, axis=-1)
            h = nnx.gelu(h * (1 + s[:, None, :]) + b[:, None, :])
        return self.out(h)


def res_pi0_class():
    """ResPi0(Pi0): openpi's Pi0 compute_loss / sample_actions (as pi05_tag_adarms.TagPi0) with the frozen base velocity
    plus the residual head; defined lazily (openpi imports only inside the env)."""
    global _ResPi0
    if _ResPi0 is not None:
        return _ResPi0
    import einops
    from openpi.models import model as _model
    from openpi.models.pi0 import Pi0, make_attn_mask

    class ResPi0(Pi0):
        mix = None  # (c_pos, c_neg, c_none[, c_unc]), see the module docstring; None = the observation's tag
        head_cfg = (False, False)  # (state, subtask) head inputs [2026-11-13]; static, like mix
        uncond = False  # [2026-11-17] learned no-tag branch (residual.json head.uncond): untagged frames train h(0)
        stack_w = ()  # [2026-11-18] stacked AG: weights of the frozen earlier-round heads stk0, stk1, ... (head.stack)

        def stack_v(self, feat, x_t, time):
            """[2026-11-18] Moving anchor: sum_i w_i (stk_i(+1) - stk_i(-1)), the guidance of every earlier round,
            each frozen head served at its own w. v_ref + stack_v = the previous round's whole guided sampler, so the
            new head is a residual on the policy that collected its data (AG Thm. 1's reference = behaviour policy)."""
            if not self.stack_w:
                return 0.0
            b = x_t.shape[0]
            pos, neg = jnp.ones((b,), jnp.float32), -jnp.ones((b,), jnp.float32)
            v = 0.0
            for i, w in enumerate(self.stack_w):
                hd = getattr(self, f"stk{i}")
                v = v + w * (hd(feat, x_t, time, pos) - hd(feat, x_t, time, neg))
            return v

        def head_extra(self, observation):
            """(B, E) extra head input or None (no head inputs: exactly the original computation)."""
            use_state, use_sub = self.head_cfg
            parts = []
            if use_state:
                parts.append(jnp.asarray(observation.state[:, :STATE_DIMS], jnp.float32))
            if use_sub:
                m = observation.image_masks
                parts.append(jnp.stack([jnp.asarray(m[k], jnp.float32) for k in tr.MASK_SUB], axis=-1))
            return jnp.concatenate(parts, axis=-1) if parts else None

        def tag_y(self, observation, batch: int):
            """(y, has_tag): y = +1 / -1 for a positive / negative tag in the masks (vl.tag_route), 0 without."""
            m = observation.image_masks
            if tr.MASK_POS not in m:
                z = jnp.zeros((batch,), jnp.float32)
                return z, z
            pos = jnp.asarray(m[tr.MASK_POS], jnp.float32)
            neg = jnp.asarray(m[tr.MASK_NEG], jnp.float32)
            return pos - neg, jnp.clip(pos + neg, 0, 1)

        def compute_loss(self, rng, observation, actions, *, train: bool = False):
            y, has = self.tag_y(observation, actions.shape[0])
            if self.uncond:  # [2026-11-17] every frame trains the head; tag-dropped frames (y = 0) fit h(0)
                has = jnp.ones_like(has)
            extra = self.head_extra(observation)
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond])
            feat = jax.lax.stop_gradient(suffix_out[:, -self.action_horizon:])
            v_ref = jax.lax.stop_gradient(self.action_out_proj(feat)).astype(jnp.float32)
            if self.stack_w:  # [2026-11-18] stacked AG: the reference is the previous round's guided sampler
                v_ref = v_ref + jax.lax.stop_gradient(self.stack_v(feat, jax.lax.stop_gradient(x_t), time))
            h = self.res_head(feat, jax.lax.stop_gradient(x_t), time, y, extra)
            v_t = v_ref + has[:, None, None] * h
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

        def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
            batch_size = observation.state.shape[0]
            y, has = self.tag_y(observation, batch_size)
            extra = self.head_extra(observation)
            observation = _model.preprocess_observation(None, observation, train=False)
            dt = -1.0 / num_steps
            if noise is None:
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

            def base(x_t, time):
                tb = jnp.broadcast_to(time, batch_size)
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, tb)
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                pa = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([pa, suffix_attn_mask], axis=-1)
                pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=pos,
                                                        kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
                feat = suffix_out[:, -self.action_horizon:]
                v0 = self.action_out_proj(feat).astype(jnp.float32)
                if self.stack_w:  # [2026-11-18] stacked AG: "none" / every mix is relative to the previous round
                    v0 = v0 + self.stack_v(feat, x_t, tb)
                return v0, feat, tb

            mix = self.mix

            def step(carry):
                x_t, time = carry
                v_ref, feat, tb = base(x_t, time)
                if mix is not None:
                    cp, cn, c0, *rest = mix
                    cu = rest[0] if rest else 0.0  # [2026-11-17] 4th entry: weight of h(0), the no-tag branch
                    v = (cp + cn + c0 + cu) * v_ref
                    for c, yy in ((cp, 1.0), (cn, -1.0), (cu, 0.0)):
                        if c != 0.0:
                            v = v + c * self.res_head(feat, x_t, tb, jnp.full((batch_size,), yy, jnp.float32), extra)
                else:
                    v = v_ref + has[:, None, None] * self.res_head(feat, x_t, tb, y, extra)
                return (x_t + dt * v).astype(x_t.dtype), time + dt

            x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
            return x_0

    _ResPi0 = ResPi0
    return ResPi0


def head_out_rms(train) -> float:
    from flax import traverse_util
    flat = traverse_util.flatten_dict(train.to_pure_dict(), sep="/")
    k = [v for p, v in flat.items() if p.startswith("res_head/out/kernel")]
    return float(jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(k[0], jnp.float32))))) if k else 0.0


@contextlib.contextmanager
def residual_route(seed: int = 0, head: dict | None = None):
    """Within the block, pi05_policy_finetune builds / loads ResPi0 models (frozen pi0.5 + trainable res_head only)
    and strips tags from prompts into the masks (as pi05_tag_adarms.adarms_route). head: residual.json "head" (the
    optional state / subtask inputs [2026-11-13]; None or without them = the original head)."""
    from openpi.shared import nnx_utils
    orig = pfm.base_policy, pfm.lora_model, pfm.split, pfm.lora_b_rms
    head = head or {}
    use_state, use_sub = bool(head.get("state")), bool(head.get("subtask"))

    def base_policy():
        tc, policy = orig[0]()
        policy._input_transform = tr.wrap_input_transform(policy._input_transform, subtask=use_sub)
        return tc, policy

    def lora_model(tc, base_params, s):
        model, cfg = orig[1](tc, base_params, s)  # LoRA variant with lora_b = 0 = base pi0.5 exactly; all frozen here
        model.__class__ = res_pi0_class()
        model.res_head = ResidualHead(model.action_in_proj.out_features, model.action_dim, seed,
                                      head_extra_dim(head))
        if use_state or use_sub:
            model.head_cfg = (use_state, use_sub)
        if head.get("uncond"):
            model.uncond = True
        stack = head.get("stack") or []  # [2026-11-18] frozen earlier-round heads (not matched by the res_head regex)
        for i, s in enumerate(stack):
            model.__setattr__(f"stk{i}", load_stack_head(s, model.action_in_proj.out_features, model.action_dim, seed))
        if stack:
            model.stack_w = tuple(float(s["w"]) for s in stack)
        return model, cfg

    def split(model, cfg, freeze_siglip: bool = False):
        graphdef, train, frozen = nnx.split(model, nnx.All(nnx.Param, nnx_utils.PathRegex(".*res_head.*")), ...)
        train = jax.tree.map(lambda x: x.astype(jnp.float32), train)
        frozen = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, frozen)
        return graphdef, train, frozen

    def lora_b_rms(train):  # logged at every val step: the head's output-layer size instead (no LoRA trains)
        return head_out_rms(train)

    pfm.base_policy, pfm.lora_model, pfm.split, pfm.lora_b_rms = base_policy, lora_model, split, lora_b_rms
    try:
        yield
    finally:
        pfm.base_policy, pfm.lora_model, pfm.split, pfm.lora_b_rms = orig


def load_stack_head(entry: dict, feat_dim: int, action_dim: int, seed: int = 0) -> ResidualHead:
    """[2026-11-18] One frozen head of a stacked-AG reference: the res_head of entry["run"] at entry["step"]. Its params
    land in the frozen split (bf16, as that run's head was when it was served: load_finetuned casts heads to bf16)."""
    from flax import serialization
    hd = ResidualHead(feat_dim, action_dim, seed, 0)
    p = Path(entry["run"]) / f"trainable_{entry.get('step', 'final')}.msgpack"
    pure = serialization.msgpack_restore(p.read_bytes())["res_head"]
    st = nnx.state(hd, nnx.Param)
    st.replace_by_pure_dict(jax.tree.map(jnp.asarray, pure))
    nnx.update(hd, st)
    return hd


def stack_entry(prev_run, shift: float, chk_json) -> list[dict]:
    """[2026-11-18] head.stack for a new stacked-AG round: prev_run's own stack + prev_run itself, served at the w
    whose matched shift is `shift` in its contrast check (the formula and 3-decimal rounding of
    pi05_tag_adarms.sbatch shift_to_w, so the stacked reference equals the sampler that collected the data)."""
    prev_run = Path(prev_run)
    meta = json.loads((prev_run / "residual.json").read_text())
    ph = meta.get("head", {})
    assert not (ph.get("state") or ph.get("subtask") or ph.get("uncond")), \
        f"stacking needs plain heads (no head inputs / uncond branch): {prev_run}"
    r = json.loads(Path(chk_json).read_text())["weights"]
    p = [(float(w), v["norm"]["ratio"]) for w, v in r.items() if float(w) > 0]
    k = sum(w * x for w, x in p) / sum(w * w for w, x in p)
    w = round(float(shift) / k, 3)
    return list(ph.get("stack") or []) + [{"run": str(prev_run), "step": "final", "w": w, "shift": float(shift),
                                          "check": str(chk_json)}]


def is_residual(run_dir) -> bool:
    return (Path(run_dir) / "residual.json").exists()


def load_residual(run_dir, step: str = "final", seed: int | None = None, contrast: float | None = None):
    """(policy, results.json) for a residual run; interface of pi05_tag_adarms.load_adarms (guidance not offered)."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "residual.json").read_text())
    if seed is None:
        seed = int(pfm.run_args(run_dir).get("seed", meta.get("seed", 0)))
    if meta.get("head", {}).get("subtask"):
        fd.SUBTASK_PROMPT = True  # tools that build samples from episodes (contrast check) then attach subtasks
    with residual_route(seed, meta.get("head")):
        policy, res = pfm.load_finetuned(run_dir, step, seed=seed)
    # the head runs in fp32 (its own casts); load_finetuned cast it to bf16, which only rounds the weights
    if contrast is not None:
        ta.set_mix(policy, ta.contrastive_mix(contrast))
    return policy, res


def selftest(a):
    """(1) step 0 = base: +, - and untagged losses equal each other and the plain pi05_libero loss; (2) only res_head
    trains; (3) step-0 sampling of +, -, none and contrast w = 4 equals base for the same noise; (4) random head:
    + and - differ, none and contrast w = 0 stay at base, mix (1, 0, 0) = plain +; (5) gradients reach only res_head
    and are nonzero for tagged batches, zero untagged; 40 Adam steps on one batch lower its loss; (6) save / load."""
    import optax
    from flax import traverse_util
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res, ok = {}, {}
    with residual_route(a.seed):
        tc, policy = pfm.base_policy()
        ref = policy._model
        model, cfg = pfm.lora_model(tc, nnx.state(ref).to_pure_dict(), a.seed)
        graphdef, train, frozen = pfm.split(model, cfg)
        n = pfm.counts(train)
        res["trainable"] = n
        flat = traverse_util.flatten_dict(train.to_pure_dict(), sep="/")
        ok["2_only_head"] = list(n) == ["res_head"] and all(k.startswith("res_head/") for k in flat)
        print(f"(2) trainable {n}: {ok['2_only_head']}", flush=True)
        base_samples = pfm.raw_samples(2, 4)
        tagged = {name: [dict(s, prompt=fd.tag_prompt(s["prompt"], tag)) for s in base_samples]
                  for name, tag in (("pos", True), ("neg", False), ("none", None))}
        b = {name: pfm.batch_arrays(policy, s) for name, s in tagged.items()}
        rng = jax.random.PRNGKey(0)
        loss_fn = pfm.make_loss(graphdef)
        jl = jax.jit(loss_fn, static_argnums=4)
        losses = {name: float(jl(train, frozen, dict(x), rng, False)) for name, x in b.items()}
        gr, sr = nnx.split(ref)
        bb = dict(b["none"]); bb["image_mask"] = {k: v for k, v in bb["image_mask"].items()
                                                   if k not in (tr.MASK_POS, tr.MASK_NEG)}
        l_ref = float(jax.jit(lambda st, x: nnx.merge(gr, st).compute_loss(rng, *pfm.as_inputs(x), train=False).mean()
                              )(sr, bb))
        res["losses"] = {**losses, "pi05_untagged": l_ref}
        ok["1_step0_equal"] = (losses["pos"] == losses["neg"] == losses["none"]
                               and abs(l_ref - losses["none"]) <= 0.02 * abs(l_ref) + 1e-4)
        print(f"(1) losses {res['losses']}: {ok['1_step0_equal']}", flush=True)

        def sampler(tr_state, mix=None):
            m = nnx.merge(graphdef, tr_state, frozen)
            m.mix = mix
            gd, st = nnx.split(m)
            f = jax.jit(lambda s, obs, z: nnx.merge(gd, s).sample_actions(rng, obs, noise=z))
            return lambda x, z: np.asarray(f(st, pfm.as_inputs(dict(x))[0], z), np.float32)

        nb = len(base_samples)
        z = jax.random.normal(jax.random.PRNGKey(5), (nb, ref.action_horizon, ref.action_dim))
        s0 = sampler(train)
        a0 = {name: s0(x, z) for name, x in b.items()}
        a0["cg4"] = sampler(train, ta.contrastive_mix(4.0))(b["none"], z)
        d3 = max(float(np.abs(a0[k] - a0["none"]).max()) for k in ("pos", "neg", "cg4"))
        ok["3_sample_step0_equal"] = d3 == 0.0
        print(f"(3) step-0 sampling max diff vs none (+, -, cg4) = {d3}: {ok['3_sample_step0_equal']}", flush=True)

        key = jax.random.PRNGKey(9)
        rnd = {}
        for k, v in flat.items():
            key, sub = jax.random.split(key)
            rnd[k] = v + (0.05 * jax.random.normal(sub, v.shape) if k.startswith("res_head/out/") else 0.0)
        train_r = jax.tree.map(lambda x: x, train)
        train_r.replace_by_pure_dict(traverse_util.unflatten_dict(rnd, sep="/"))
        s1 = sampler(train_r)
        a1 = {name: s1(x, z) for name, x in b.items()}
        d_pn = float(np.sqrt(np.mean((a1["pos"] - a1["neg"]) ** 2)))
        d_none = float(np.abs(a1["none"] - a0["none"]).max())
        d_w0 = float(np.abs(sampler(train_r, ta.contrastive_mix(0.0))(b["none"], z) - a0["none"]).max())
        e_mix = float(np.abs(sampler(train_r, (1.0, 0.0, 0.0))(b["none"], z) - a1["pos"]).max())
        ok["4_random_head"] = d_pn > 1e-3 and d_none == 0.0 and d_w0 == 0.0 and e_mix < 1e-5
        print(f"(4) random head: rms(+ - -) {d_pn:.4g}, none vs base {d_none}, w0 vs base {d_w0}, mix(1,0,0) vs + "
              f"{e_mix:.2g}: {ok['4_random_head']}", flush=True)

        # frozen goes in as an argument: closing over it bakes ~6 GB of weights into each executable (host OOM)
        g_raw = jax.grad(lambda t, fz, x: loss_fn(t, fz, x, rng, False))
        gfn = jax.jit(g_raw)
        gn = {name: float(optax.global_norm(gfn(train_r, frozen, dict(b[name])))) for name in ("pos", "neg", "none")}
        opt = optax.adam(1e-3)
        st, t2 = opt.init(train_r), train_r

        @jax.jit
        def step(t, s, fz, x):
            u, s = opt.update(g_raw(t, fz, x), s, t)
            return optax.apply_updates(t, u), s
        l_before = float(jl(t2, frozen, dict(b["pos"]), rng, False))
        for _ in range(40):
            t2, st = step(t2, st, frozen, dict(b["pos"]))
        l_after = float(jl(t2, frozen, dict(b["pos"]), rng, False))
        res["grad_norm"], res["fit"] = gn, [l_before, l_after]
        ok["5_grads_fit"] = gn["pos"] > 0 and gn["neg"] > 0 and gn["none"] == 0.0 and l_after < l_before
        print(f"(5) grad norms {gn}, 40 steps on one + batch: {l_before:.4f} -> {l_after:.4f}: {ok['5_grads_fit']}",
              flush=True)

        # (7) [2026-11-17] --uncond-branch: untagged frames now train h(0) (nonzero gradient), tagged losses unchanged;
        # the 4-entry mix with c_unc = 0 equals the 3-entry mix, and uncond_mix(0) = v_ref + h(0) differs from base
        mu = nnx.merge(graphdef, train_r, frozen)
        mu.uncond = True
        gd_u, tr_u, fz_u = pfm.split(mu, cfg)
        lu = pfm.make_loss(gd_u)
        gu = float(optax.global_norm(jax.jit(jax.grad(lambda t, fz, x: lu(t, fz, x, rng, False)))(
            tr_u, fz_u, dict(b["none"]))))
        lp = (float(jax.jit(lu, static_argnums=4)(tr_u, fz_u, dict(b["pos"]), rng, False)),
              float(jl(train_r, frozen, dict(b["pos"]), rng, False)))
        e4 = float(np.abs(sampler(train_r, (1.0, 0.0, 0.0, 0.0))(b["none"], z) - a1["pos"]).max())
        d_u0 = float(np.sqrt(np.mean((sampler(train_r, uncond_mix(0.0))(b["none"], z) - a0["none"]) ** 2)))
        ok["7_uncond"] = gu > 0 and abs(lp[0] - lp[1]) <= 1e-6 * abs(lp[1]) + 1e-7 and e4 < 1e-5 and d_u0 > 1e-3
        res["uncond"] = {"grad_untagged": gu, "loss_pos_uncond_vs_plain": lp, "mix4_vs_pos": e4, "rms_h0_vs_base": d_u0}
        print(f"(7) uncond: untagged grad {gu:.4g}, + loss {lp[0]:.6f} vs {lp[1]:.6f}, mix (1,0,0,0) vs + {e4:.2g}, "
              f"uncond_mix(0) vs base rms {d_u0:.4g}: {ok['7_uncond']}", flush=True)
        del mu, gd_u, tr_u, fz_u, lu

        pfm.save(train_r, out / "trainable_final.msgpack")
        (out / "residual.json").write_text(json.dumps({"seed": a.seed}))
        (out / "args.json").write_text(json.dumps({"seed": a.seed, "arm": "advantage"}))
        want = np.asarray(rnd["res_head/out/kernel"], np.float32)
        del ref, sr, gr, model, train, train_r, t2, frozen, policy, s0, s1, gfn, jl, loss_fn, b
        import gc
        gc.collect()
    pol2, _ = load_residual(out, seed=a.seed)
    got = np.asarray(nnx.state(pol2._model)["res_head"]["out"]["kernel"].value, np.float32)
    err = float(np.abs(got - want.astype(jnp.bfloat16).astype(np.float32)).max())
    ok["6_roundtrip"] = err == 0.0 and type(pol2._model).__name__ == "ResPi0"
    print(f"(6) save/load head max |diff| (bf16) {err}: {ok['6_roundtrip']}", flush=True)
    for f in ("trainable_final.msgpack", "residual.json", "args.json"):
        (out / f).unlink()
    res["ok"] = ok
    (out / "residual_selftest.json").write_text(json.dumps(res, indent=2))
    assert all(ok.values()), ok
    print("all ok", flush=True)


def stack_check(a):
    """[2026-11-18] Stacked AG equivalence on real checkpoints: a stacked run whose new head is zero (prev run's head
    with its output layer zeroed) must sample, untagged and with the standard mix at any w, exactly like prev run
    served at its stack w (same val frames, same noise z). Also: the stacked mix (1, 0, 0) / (0, 1, 0) / (0, 0, 1) all
    equal that untagged sample (zero head), and the stack term is nonzero (the reference really moved from base)."""
    import shutil
    from flax import serialization
    import libero_tag_usage as tu
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prev = Path(a.stack_from)
    stack = stack_entry(prev, a.stack_shift, a.stack_chk)
    w_prev = stack[-1]["w"]
    tmp = out / "zero_head_run"; tmp.mkdir(exist_ok=True)
    pure = serialization.msgpack_restore((prev / "trainable_final.msgpack").read_bytes())
    pure["res_head"]["out"] = {k: np.zeros_like(v) for k, v in pure["res_head"]["out"].items()}
    (tmp / "trainable_final.msgpack").write_bytes(serialization.msgpack_serialize(pure))
    for f in ("args.json", "results.json"):
        if (prev / f).exists():
            shutil.copy(prev / f, tmp / f)
    meta = json.loads((prev / "residual.json").read_text())
    meta["head"] = {**meta["head"], "stack": stack}
    (tmp / "residual.json").write_text(json.dumps(meta))
    frames = tu.val_frames(8, 4)
    var = {"m": lambda ep, t: fd.sample(ep, t, None)}
    res = {"stack": stack, "n_frames": len(frames)}
    p_ref, _ = load_residual(prev, contrast=w_prev)
    ref, _ = tu.sample_variants(p_ref, frames, 2, 8, var)
    ta.set_mix(p_ref, (0.0, 0.0, 1.0))
    base, _ = tu.sample_variants(p_ref, frames, 2, 8, var)
    del p_ref
    p_stk, _ = load_residual(tmp)
    got = {}
    for name, mix in (("plain", None), ("pos", (1.0, 0.0, 0.0)), ("neg", (0.0, 1.0, 0.0)), ("none", (0.0, 0.0, 1.0)),
                      ("w4", ta.contrastive_mix(4.0))):
        ta.set_mix(p_stk, mix)
        got[name] = tu.sample_variants(p_stk, frames, 2, 8, var)[0]["m"]
    scale = float(np.abs(ref["m"]).max()) + 1e-8
    res["rel_err_vs_prev_served"] = {n: float(np.abs(g - ref["m"]).max() / scale) for n, g in got.items()}
    res["stack_vs_base_rms"] = float(tu.rms(ref["m"] - base["m"], (0, 1, 2, 3)))
    ok = all(e < 2e-2 for e in res["rel_err_vs_prev_served"].values()) and res["stack_vs_base_rms"] > 1e-3
    res["ok"] = bool(ok)
    (out / "stack_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    shutil.rmtree(tmp)
    assert ok, "stack check FAILED"
    print("stack check ok", flush=True)


def uncond_mix(w: float) -> tuple[float, float, float, float]:
    """[2026-11-17, option 2 of the AG saturation check] Classifier-free-guidance form with a learned no-tag branch:
        v = v_ref + h(0) + w (h(+) - h(-))
    h(0) is trained on the tag-dropped frames of every episode (--uncond-branch --tag-dropout p), so v_ref + h(0) is the
    training data's own behaviour (all rounds) and the guidance steps from it, not from base pi0.5 [AG paper: the
    reference is a BC flow on the data; here it is base pi0.5 + a learned correction]. Needs a run trained with
    --uncond-branch (residual.json head.uncond)."""
    return (float(w), -float(w), 0.0, 1.0)


def anchored_mix(w: float, anchor: float) -> tuple[float, float, float]:
    """[2026-11-17, AG saturation check] Contrastive guidance from the data's behaviour instead of from base pi0.5:
        v = v_ref + (a h(+) + (1 - a) h(-)) + w (h(+) - h(-)),   a = positive share of the labels (0.4 for vb40)
    i.e. the mix (a + w, 1 - a - w, 0); the coefficients sum to 1, so v_ref keeps weight 1. The bracket is the label-
    weighted mean residual = what the training data (all rounds) does relative to base pi0.5. The standard mix
    (w, -w, 1) drops it, so later rounds always step from base pi0.5, never from the policies that collected the data
    [ours]. anchor = 0 with the same w is NOT the standard mix (it keeps h(-)); use contrastive_mix for that."""
    a = float(anchor)
    assert 0.0 <= a <= 1.0, a
    return (a + float(w), 1.0 - a - float(w), 0.0)


class SubtaskMixPolicy:
    """[2026-11-12] Per-subtask contrastive guidance weight (SeeQ / openpi lit-review suggestion: our guided failures
    are at the second moka pot). The client's --args.send-subtask puts obs["subtask"] = 0 (no pot on the stove yet) or
    1 (at least one, from the BDDL goal predicates) in every request; this picks the sampler jitted with contrast w0
    or w1 for that request. Both samplers come from ta.set_mix, so each equals a server started with that single w
    (module_jit freezes the model's graphdef, mix included, when it is built). `inner` is the TaggedPolicy wrapping
    `policy`; it pops the subtask key itself."""

    def __init__(self, policy, inner, w0: float, w1: float):
        self._policy, self._inner = policy, inner
        self.w = {0: float(w0), 1: float(w1)}
        self._fns = {}
        for s in (0, 1):
            ta.set_mix(policy, ta.contrastive_mix(self.w[s]))
            self._fns[s] = policy._sample_actions
        self.counts = {0: 0, 1: 0}

    @property
    def metadata(self):
        return self._inner.metadata

    def reset(self):
        self._inner.reset()

    def infer(self, obs: dict, noise=None) -> dict:
        import logging
        sub = obs.get("subtask")
        assert sub is not None, "--contrast-sub1 needs the client's --args.send-subtask"
        s = int(sub)
        assert s in (0, 1), f"subtask {sub!r}"
        self._policy._sample_actions = self._fns[s]
        self.counts[s] += 1
        n = self.counts[0] + self.counts[1]
        if n == 1 or n % 500 == 0:
            logging.info("subtask requests %s (w %s)", self.counts, self.w)
        return self._inner.infer(obs) if noise is None else self._inner.infer(obs, noise=noise)


class ContrastSetPolicy:
    """[2026-11-18] Per-episode contrastive guidance weight for mixed-push collection (Karen: "set up options 1 and
    2"; option 1 = a random push per episode). The client's --args.push-choices K sends obs["w_idx"] in [0, K) with
    every request (one draw per episode, scripts/libero_rollouts.py Changes 9); this serves the request with the
    sampler jitted at contrast w_set[w_idx], built as SubtaskMixPolicy builds its two (each equals a server started
    with that single w). A request without w_idx is an error, so no episode silently gets a default w."""

    def __init__(self, policy, inner, w_set):
        self._policy, self._inner = policy, inner
        self.w = [float(w) for w in w_set]
        self._fns = []
        for w in self.w:
            ta.set_mix(policy, ta.contrastive_mix(w))
            self._fns.append(policy._sample_actions)
        self.counts = [0] * len(self.w)

    @property
    def metadata(self):
        return self._inner.metadata

    def reset(self):
        self._inner.reset()

    def infer(self, obs: dict, noise=None) -> dict:
        import logging
        obs = dict(obs)
        i = obs.pop("w_idx", None)
        assert i is not None, "--contrast-set needs the client's --args.push-choices"
        i = int(i)
        assert 0 <= i < len(self.w), f"w_idx {i} outside the {len(self.w)} contrast weights"
        self._policy._sample_actions = self._fns[i]
        self.counts[i] += 1
        n = sum(self.counts)
        if n == 1 or n % 500 == 0:
            logging.info("w_idx requests %s (w %s)", self.counts, self.w)
        return self._inner.infer(obs) if noise is None else self._inner.infer(obs, noise=noise)


def serve(a):
    import logging
    import socket
    from openpi.serving import websocket_policy_server
    from serve_pi05_finetuned import TaggedPolicy
    from vl.libero_noise import MatchedNoisePolicy
    logging.basicConfig(level=logging.INFO, force=True)
    tag = {"positive": True, "negative": False, "none": None}[a.tag]
    if a.contrast is not None:
        tag = None  # the mix builds the +1 / -1 conditions itself; the prompt stays plain
    assert a.contrast_sub1 is None or a.contrast is not None, "--contrast-sub1 needs --contrast (the subtask-0 w)"
    assert a.anchor is None or (a.contrast is not None and a.contrast_sub1 is None), \
        "--anchor needs --contrast (and is not combined with --contrast-sub1)"
    policy, _ = load_residual(a.run, a.step, contrast=a.contrast)
    if a.anchor is not None:
        ta.set_mix(policy, anchored_mix(a.contrast, a.anchor))
    if a.contrast_set:
        assert a.contrast is not None and a.contrast_sub1 is None and a.anchor is None and not a.uncond, \
            "--contrast-set needs --contrast (any of the set; not with --contrast-sub1 / --anchor / --uncond)"
    if a.uncond:
        assert a.contrast is not None and a.contrast_sub1 is None and a.anchor is None, \
            "--uncond needs --contrast (not with --contrast-sub1 / --anchor)"
        assert policy._model.uncond, "--uncond: run was not trained with --uncond-branch"
        ta.set_mix(policy, uncond_mix(a.contrast))
    logging.info("serving residual run %s step %s tag %s contrast %s (mix %s) contrast-sub1 %s on port %d, host %s",
                 a.run, a.step, tag, a.contrast, policy._model.mix, a.contrast_sub1, a.port, socket.gethostname())
    m = policy._model
    inner = TaggedPolicy(policy, tag, a.noise_seed, subtask=bool(m.head_cfg[1]))  # subtask head: client must send it
    if a.contrast_sub1 is not None:
        inner = SubtaskMixPolicy(policy, inner, a.contrast, a.contrast_sub1)
    if a.contrast_set:
        inner = ContrastSetPolicy(policy, inner, a.contrast_set)
        logging.info("contrast set %s: the client's w_idx picks the w of each episode", inner.w)
    served = MatchedNoisePolicy(inner, (m.action_horizon, m.action_dim), a.noise_seed)
    websocket_policy_server.WebsocketPolicyServer(policy=served, host="0.0.0.0", port=a.port,
                                                  metadata=policy.metadata).serve_forever()


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--residual-selftest", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--step", default="final")
    ap.add_argument("--tag", choices=["positive", "negative", "none"], default="positive")
    ap.add_argument("--contrast", type=float, default=None)
    ap.add_argument("--contrast-sub1", type=float, default=None,
                    help="contrast w once a pot is on the stove (obs subtask 1); --contrast is then the subtask-0 w")
    ap.add_argument("--contrast-set", type=float, nargs="+", default=None,
                    help="[2026-11-18] contrast w per obs w_idx (client --args.push-choices), see ContrastSetPolicy")
    ap.add_argument("--anchor", type=float, default=None,
                    help="[2026-11-17] guide from the data's behaviour: mix (a + w, 1 - a - w, 0), see anchored_mix")
    ap.add_argument("--uncond", action="store_true",
                    help="[2026-11-17] serve v_ref + h(0) + w (h(+) - h(-)) (uncond_mix); --uncond-branch runs only")
    ap.add_argument("--tag-scale", type=float, default=None, help="ignored (pi05_tag_adarms.sbatch passes it)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--noise-seed", type=int, default=0)
    a, rest = ap.parse_known_args()
    if a.serve:
        return serve(a)
    if a.residual_selftest:
        o = argparse.ArgumentParser()
        o.add_argument("--out", required=True)
        o.add_argument("--seed", type=int, default=0)
        return selftest(o.parse_args(rest))
    if "--stack-check" in rest:
        o = argparse.ArgumentParser()
        o.add_argument("--stack-check", action="store_true")
        o.add_argument("--out", required=True)
        o.add_argument("--stack-from", required=True)
        o.add_argument("--stack-shift", type=float, required=True)
        o.add_argument("--stack-chk", required=True)
        return stack_check(o.parse_args(rest))
    assert "--train" in rest, "residual: use --train (or --residual-selftest / --serve)"
    uncond = "--uncond-branch" in rest  # [2026-11-17] learned no-tag branch h(0), see uncond_mix
    assert "--tag-dropout" in rest, "residual: pass --tag-dropout (0, or > 0 with --uncond-branch)"
    p_drop = float(rest[rest.index("--tag-dropout") + 1])
    assert (p_drop == 0.0) if not uncond else (0.0 < p_drop < 1.0), \
        "residual: --tag-dropout 0 (the reference is the frozen base), or 0 < p < 1 with --uncond-branch (h(0) " \
        "learns from the dropped frames)"
    head_state, head_sub = "--head-state" in rest, "--head-subtask" in rest
    rest = [x for x in rest if x not in ("--head-state", "--head-subtask", "--uncond-branch")]
    # [2026-11-18] stacked AG: --stack-from <prev run> --stack-shift <its serving shift> --stack-chk <its check json>
    stack = None
    if "--stack-from" in rest:
        sv = {}
        for k_ in ("--stack-from", "--stack-shift", "--stack-chk"):
            assert k_ in rest, f"stacked AG needs --stack-from, --stack-shift and --stack-chk ({k_} missing)"
            i = rest.index(k_)
            sv[k_] = rest[i + 1]
            del rest[i:i + 2]
        assert not (uncond or head_state or head_sub), "stacked AG: plain heads only"
        stack = stack_entry(sv["--stack-from"], float(sv["--stack-shift"]), sv["--stack-chk"])
        print(f"stacked AG: reference = base + {len(stack)} frozen head(s): " +
              ", ".join(f"{Path(s['run']).name} w {s['w']}" for s in stack), flush=True)
    if head_sub and "--subtask-prompt" not in rest:
        rest.append("--subtask-prompt")  # the data path attaches per-frame subtasks (stripped from pi0.5's prompt)
    out = Path(rest[rest.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    head = {"width": WIDTH, "depth": DEPTH, "cond": COND, "time_dim": TIME_DIM}
    if head_state or head_sub:
        head.update(state=head_state, subtask=head_sub)
    if uncond:
        head["uncond"] = True
    if stack:
        head["stack"] = stack
    (out / "residual.json").write_text(json.dumps({"route": "ag_residual", "script": Path(__file__).name, "seed": seed,
                                                   "head": head}))
    sys.argv = [sys.argv[0]] + rest
    with residual_route(seed, head):
        pfm.main()


if __name__ == "__main__":
    main()
