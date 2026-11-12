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
_ResPi0 = None


class ResidualHead(nnx.Module):
    """h_theta(feature, x_t, t, y): per-token AdaLN MLP [AG networks.py ConditionedMLP: Dense -> LayerNorm (no affine)
    -> modulate by the condition -> GELU per layer; condition encoder MLP(128, 128) of the scalar]."""

    def __init__(self, feat_dim: int, action_dim: int, seed: int = 0):
        rngs = nnx.Rngs(seed + 4000)
        self.inp = nnx.Linear(feat_dim + action_dim + TIME_DIM, WIDTH, rngs=rngs)
        self.c1 = nnx.Linear(1, COND, rngs=rngs)
        self.c2 = nnx.Linear(COND, COND, rngs=rngs)
        for i in range(DEPTH):
            if i:
                setattr(self, f"l{i}", nnx.Linear(WIDTH, WIDTH, rngs=rngs))
            setattr(self, f"m{i}", nnx.Linear(COND, 2 * WIDTH, rngs=rngs))
        self.out = nnx.Linear(WIDTH, action_dim, kernel_init=nnx.initializers.zeros_init(),
                              bias_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, feat, x_t, time, y):
        from openpi.models.pi0 import posemb_sincos
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        te = posemb_sincos(f32(time), TIME_DIM, min_period=4e-3, max_period=4.0)
        te = jnp.broadcast_to(te[:, None, :], (*x_t.shape[:2], TIME_DIM))
        h = self.inp(jnp.concatenate([f32(feat), f32(x_t), te], axis=-1))
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
        mix = None  # (c_pos, c_neg, c_none), see the module docstring; None = the observation's tag

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
            h = self.res_head(feat, jax.lax.stop_gradient(x_t), time, y)
            v_t = v_ref + has[:, None, None] * h
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

        def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
            batch_size = observation.state.shape[0]
            y, has = self.tag_y(observation, batch_size)
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
                return self.action_out_proj(feat).astype(jnp.float32), feat, tb

            mix = self.mix

            def step(carry):
                x_t, time = carry
                v_ref, feat, tb = base(x_t, time)
                if mix is not None:
                    cp, cn, c0 = mix
                    v = (cp + cn + c0) * v_ref
                    for c, yy in ((cp, 1.0), (cn, -1.0)):
                        if c != 0.0:
                            v = v + c * self.res_head(feat, x_t, tb, jnp.full((batch_size,), yy, jnp.float32))
                else:
                    v = v_ref + has[:, None, None] * self.res_head(feat, x_t, tb, y)
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
def residual_route(seed: int = 0):
    """Within the block, pi05_policy_finetune builds / loads ResPi0 models (frozen pi0.5 + trainable res_head only)
    and strips tags from prompts into the masks (as pi05_tag_adarms.adarms_route)."""
    from openpi.shared import nnx_utils
    orig = pfm.base_policy, pfm.lora_model, pfm.split, pfm.lora_b_rms

    def base_policy():
        tc, policy = orig[0]()
        policy._input_transform = tr.wrap_input_transform(policy._input_transform)
        return tc, policy

    def lora_model(tc, base_params, s):
        model, cfg = orig[1](tc, base_params, s)  # LoRA variant with lora_b = 0 = base pi0.5 exactly; all frozen here
        model.__class__ = res_pi0_class()
        model.res_head = ResidualHead(model.action_in_proj.out_features, model.action_dim, seed)
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


def is_residual(run_dir) -> bool:
    return (Path(run_dir) / "residual.json").exists()


def load_residual(run_dir, step: str = "final", seed: int | None = None, contrast: float | None = None):
    """(policy, results.json) for a residual run; interface of pi05_tag_adarms.load_adarms (guidance not offered)."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "residual.json").read_text())
    if seed is None:
        seed = int(pfm.run_args(run_dir).get("seed", meta.get("seed", 0)))
    with residual_route(seed):
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
    policy, _ = load_residual(a.run, a.step, contrast=a.contrast)
    logging.info("serving residual run %s step %s tag %s contrast %s (mix %s) on port %d, host %s", a.run, a.step, tag,
                 a.contrast, policy._model.mix, a.port, socket.gethostname())
    m = policy._model
    served = MatchedNoisePolicy(TaggedPolicy(policy, tag, a.noise_seed, subtask=False),
                                (m.action_horizon, m.action_dim), a.noise_seed)
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
    assert "--train" in rest, "residual: use --train (or --residual-selftest / --serve)"
    assert rest[rest.index("--tag-dropout") + 1] in ("0", "0.0") if "--tag-dropout" in rest else False, \
        "residual: pass --tag-dropout 0 (the reference is the frozen base, not a dropout branch)"
    out = Path(rest[rest.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    (out / "residual.json").write_text(json.dumps({"route": "ag_residual", "script": Path(__file__).name, "seed": seed,
                                                   "head": {"width": WIDTH, "depth": DEPTH, "cond": COND,
                                                            "time_dim": TIME_DIM}}))
    sys.argv = [sys.argv[0]] + rest
    with residual_route(seed):
        pfm.main()


if __name__ == "__main__":
    main()
