"""Advantage tag through AdaRMS instead of the prompt (reports/2026-11-06-tag-hypotheses/, idea 3) [ours, 2026-11-06].

Why: every tagged policy so far ignores its prompt tag, also at gripper switches and stalls (E4, 18353550). π₀.₅'s
action expert is conditioned on the flow time through adaRMS: time_emb = swish(time_mlp(sincos(t))) scales, shifts and
gates every action-expert layer. ALIGN (arXiv 2610.26455) found a tag injected there beats the prompt suffix; CFGRL
uses a learned optimality embedding the same way. Here:
    adarms_cond = time_emb + tag_scale * tag_emb[tag]      tag_emb (2, 1024): rows negative, positive; no tag adds 0
  [ours] tag_emb starts at zero, so step 0 is exactly pi05_libero (with lora_b = 0) for every tag; tag_scale (default
         40) multiplies it so that Adam at the recipe's non-LoRA lr (2.5e-5) moves it at an effective 1e-3 per step
         (Adam is invariant to gradient scale, so the multiplier is an lr for this one parameter while keeping the
         recipe's two lr groups, which the resume path asserts).
  [ours] the tag is removed from the prompt (vl.tag_route): the prompt is the plain task, the tag reaches the model
         only through adaRMS. Everything else is pi05_policy_finetune.py unchanged (LoRA recipe, data, labels, tag
         dropout, demos, val loss, checkpoints, preemption resume), so against faithful RECAP only the route changes.
  [CFGRL / RECAP] classifier-free guidance v = v_none + beta (v_pos - v_none): the prefix no longer depends on the
         tag, so one K/V cache serves both velocities (--guidance / load_adarms(guidance=...)).

How: this script patches three functions of pi05_policy_finetune (base_policy: input transform strips the tag;
lora_model: the model becomes TagPi0 with a tag_emb parameter; nothing else) and then runs its main(), so every
flag of that script works. tag_emb is a trainable leaf, so checkpoints / resume carry it with no other change.
Run settings of this script go to <out>/adarms.json; load_adarms reads them.

Usage (openpi env; scripts/pi05_tag_adarms.sbatch):
    .venv/bin/python <vl>/scripts/pi05_tag_adarms.py --adarms-selftest --out <dir>          # checks below, one GPU
    .venv/bin/python <vl>/scripts/pi05_tag_adarms.py [--tag-scale 40] --train --arm advantage --labels L ... --out D
    .venv/bin/python <vl>/scripts/pi05_tag_adarms.py --serve --run D [--tag positive] [--guidance 1.5] --port P

--adarms-selftest:
  (1) step 0 = base: with tag_emb = 0 the loss of a +, a − and an untagged batch are identical, and equal the plain
      pi05_libero loss on the untagged batch (bf16 tolerance as pi05_policy_finetune --selftest);
  (2) tagged prompts tokenize exactly as the untagged one; the masks carry the tag;
  (3) sampling at tag_emb = 0: +, − and none give identical chunks for the same noise;
  (4) with a random tag_emb: + and − differ, none is unchanged;
  (5) guidance: beta = 1 equals plain + sampling, beta = 0 equals none;
  (6) gradient of the loss w.r.t. tag_emb: nonzero rows for the tags in the batch, zero for absent tags;
  (7) save / load round trip of the trainable state including tag_emb.
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
from vl import libero_finetune_data as fd  # noqa: E402
from vl import tag_route as tr  # noqa: E402

TAG_SCALE = 40.0
_TagPi0 = None


def tag_pi0_class():
    """TagPi0(Pi0): compute_loss / sample_actions of openpi's Pi0 (src/openpi/models/pi0.py, this checkout), copied
    with the tag added to adarms_cond and the guidance line; defined lazily (openpi imports only inside the env)."""
    global _TagPi0
    if _TagPi0 is not None:
        return _TagPi0
    import einops
    from openpi.models import model as _model
    from openpi.models.pi0 import Pi0, make_attn_mask

    class TagPi0(Pi0):
        tag_scale = TAG_SCALE
        guidance = None  # float beta: classifier-free guidance toward the tag in the observation (positive)
        mix = None  # (c_pos, c_neg, c_none) velocity mix, overrides guidance; contrastive w = (w, -w, 1) [AG Eq. 4]

        def tag_cond(self, observation, batch: int):
            e = self.tag_emb.value.astype(jnp.float32) * self.tag_scale
            m = observation.image_masks
            if tr.MASK_POS not in m:
                return jnp.zeros((batch, e.shape[1]), jnp.float32)
            pos = jnp.asarray(m[tr.MASK_POS], jnp.float32)[:, None]
            neg = jnp.asarray(m[tr.MASK_NEG], jnp.float32)[:, None]
            return pos * e[1] + neg * e[0]

        def suffix(self, observation, x_t, time, cond):
            tokens, mask, ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            return tokens, mask, ar_mask, adarms_cond + cond.astype(adarms_cond.dtype)

        def compute_loss(self, rng, observation, actions, *, train: bool = False):
            cond = self.tag_cond(observation, actions.shape[0])
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.suffix(observation, x_t, time, cond)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond])
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

        def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
            batch_size = observation.state.shape[0]
            cond = self.tag_cond(observation, batch_size)
            observation = _model.preprocess_observation(None, observation, train=False)
            dt = -1.0 / num_steps
            if noise is None:
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

            def velocity(x_t, time, c):
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.suffix(
                    observation, x_t, jnp.broadcast_to(time, batch_size), c)
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                pa = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([pa, suffix_attn_mask], axis=-1)
                pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=pos,
                                                        kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
                return self.action_out_proj(suffix_out[:, -self.action_horizon:])

            mix = self.mix

            def step(carry):
                x_t, time = carry
                if mix is not None:  # v = c_pos v(+) + c_neg v(-) + c_none v(no tag); the observation's tag is unused
                    e = self.tag_emb.value.astype(jnp.float32) * self.tag_scale
                    conds = (jnp.broadcast_to(e[1], cond.shape), jnp.broadcast_to(e[0], cond.shape),
                             jnp.zeros_like(cond))
                    v = sum(c * velocity(x_t, time, k) for c, k in zip(mix, conds) if c != 0.0)
                    return x_t + dt * v, time + dt
                v = velocity(x_t, time, cond)
                if self.guidance is not None:
                    v0 = velocity(x_t, time, jnp.zeros_like(cond))
                    v = v0 + self.guidance * (v - v0)
                return x_t + dt * v, time + dt

            x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
            return x_0

    _TagPi0 = TagPi0
    return TagPi0


@contextlib.contextmanager
def adarms_route(tag_scale: float = TAG_SCALE):
    """Within the block, pi05_policy_finetune builds / loads TagPi0 models and strips tags from prompts."""
    orig_base, orig_lora, orig_rms = pfm.base_policy, pfm.lora_model, pfm.lora_b_rms

    def base_policy():
        tc, policy = orig_base()
        policy._input_transform = tr.wrap_input_transform(policy._input_transform)
        return tc, policy

    def lora_model(tc, base_params, seed):
        model, cfg = orig_lora(tc, base_params, seed)
        model.__class__ = tag_pi0_class()
        model.tag_emb = nnx.Param(jnp.zeros((2, model.action_in_proj.out_features), jnp.float32))
        model.tag_scale = float(tag_scale)
        return model, cfg

    def lora_b_rms(train):  # called at every val step: also log the tag embedding's size (no other hook needed)
        print(f"tag_emb norms (negative, positive), x tag_scale: {[round(tag_scale * x, 4) for x in tag_emb_norms(train)]}",
              flush=True)
        return orig_rms(train)

    pfm.base_policy, pfm.lora_model, pfm.lora_b_rms = base_policy, lora_model, lora_b_rms
    try:
        yield
    finally:
        pfm.base_policy, pfm.lora_model, pfm.lora_b_rms = orig_base, orig_lora, orig_rms


def contrastive_mix(w: float) -> tuple[float, float, float]:
    """Contrastive guidance [Advantage Guidance, Zhou & Levine 2026, Eq. 4]: v = v_none + w (v_pos - v_neg). The
    unconditional reference is the tag-dropout branch of the same network [ours; AG trains a separate reference]."""
    return (float(w), -float(w), 1.0)


def set_mix(policy, mix):
    """Re-jit the policy's sampler with a velocity mix (c_pos, c_neg, c_none), or None for plain sampling."""
    from openpi.shared import nnx_utils
    policy._model.mix = None if mix is None else tuple(float(c) for c in mix)
    policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)
    return policy


def load_adarms(run_dir, step: str = "final", seed: int | None = None, guidance: float | None = None,
                contrast: float | None = None):
    """(policy, results.json) for an AdaRMS run; same interface as pi05_policy_finetune.load_finetuned. The policy's
    input transform strips the prompt tag, so tools that send tagged prompts (serve_pi05_finetuned.TaggedPolicy,
    libero_tag_usage.sample_variants, libero_tag_keyframes) work unchanged. contrast=w: contrastive guidance
    (contrastive_mix), which ignores the request's tag."""
    from openpi.shared import nnx_utils
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "adarms.json").read_text())
    if seed is None:
        rj = run_dir / "results.json"
        seed = int(json.loads(rj.read_text())["args"]["seed"]) if rj.exists() else int(meta.get("seed", 0))
    with adarms_route(meta["tag_scale"]):
        policy, res = pfm.load_finetuned(run_dir, step, seed=seed)
    if guidance is not None:
        policy._model.guidance = float(guidance)
        policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)
    if contrast is not None:
        assert guidance is None, "guidance and contrast are exclusive"
        set_mix(policy, contrastive_mix(contrast))
    return policy, res


def is_adarms(run_dir) -> bool:
    return (Path(run_dir) / "adarms.json").exists()


def tag_emb_norms(train) -> list[float]:
    from flax import traverse_util
    flat = traverse_util.flatten_dict(train.to_pure_dict(), sep="/")
    return [float(x) for x in jnp.linalg.norm(jnp.asarray(flat["tag_emb"], jnp.float32), axis=1)]


def selftest(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res, ok = {}, {}
    with adarms_route(a.tag_scale):
        tc, policy = pfm.base_policy()
        ref = policy._model
        model, cfg = pfm.lora_model(tc, nnx.state(ref).to_pure_dict(), a.seed)
        graphdef, train, frozen = pfm.split(model, cfg)
        print("trainable", pfm.counts(train), flush=True)
        assert "tag_emb" in pfm.counts(train)
        base_samples = pfm.raw_samples(2, 4)  # untagged
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

        tok = {n: np.asarray(x["tokenized_prompt"]) for n, x in b.items()}
        mpos = {n: np.asarray(x["image_mask"][tr.MASK_POS]) for n, x in b.items()}
        mneg = {n: np.asarray(x["image_mask"][tr.MASK_NEG]) for n, x in b.items()}
        ok["2_prompt_stripped"] = (np.array_equal(tok["pos"], tok["none"]) and np.array_equal(tok["neg"], tok["none"])
                                   and mpos["pos"].all() and not mpos["neg"].any() and not mpos["none"].any()
                                   and mneg["neg"].all() and not mneg["pos"].any() and not mneg["none"].any())
        print(f"(2) prompt stripped, masks: {ok['2_prompt_stripped']}", flush=True)

        def sampler(tr_state, guidance=None):
            m = nnx.merge(graphdef, jax.tree.map(lambda x: x.astype(jnp.bfloat16), tr_state), frozen)
            m.guidance = guidance
            gd, st = nnx.split(m)
            f = jax.jit(lambda s, obs, z: nnx.merge(gd, s).sample_actions(rng, obs, noise=z))
            return lambda x, z: np.asarray(f(st, pfm.as_inputs(dict(x))[0], z), np.float32)

        n = len(base_samples)
        z = jax.random.normal(jax.random.PRNGKey(5), (n, ref.action_horizon, ref.action_dim))
        s0 = sampler(train)
        a0 = {name: s0(x, z) for name, x in b.items()}
        d3 = max(float(np.abs(a0["pos"] - a0["none"]).max()), float(np.abs(a0["neg"] - a0["none"]).max()))
        ok["3_sample_step0_equal"] = d3 == 0.0
        print(f"(3) step-0 sampling max |pos - none|, |neg - none| = {d3}: {ok['3_sample_step0_equal']}", flush=True)

        rnd = jax.random.normal(jax.random.PRNGKey(9), (2, ref.action_in_proj.out_features)) * 0.01
        train_r = jax.tree.map(lambda x: x, train)
        train_r.replace_by_pure_dict({**train.to_pure_dict(), "tag_emb": rnd})
        assert float(jnp.abs(train["tag_emb"].value).max()) == 0.0  # the copy did not alias
        s1 = sampler(train_r)
        a1 = {name: s1(x, z) for name, x in b.items()}
        d_pn = float(np.sqrt(np.mean((a1["pos"] - a1["neg"]) ** 2)))
        d_none = float(np.abs(a1["none"] - a0["none"]).max())
        ok["4_random_emb_moves_tags_only"] = d_pn > 1e-3 and d_none == 0.0
        print(f"(4) random tag_emb: rms(pos - neg) {d_pn:.4g}, max |none - step0 none| {d_none}: "
              f"{ok['4_random_emb_moves_tags_only']}", flush=True)

        g1 = sampler(train_r, 1.0)(b["pos"], z)
        g0 = sampler(train_r, 0.0)(b["pos"], z)
        e1, e0 = float(np.abs(g1 - a1["pos"]).max()), float(np.abs(g0 - a1["none"]).max())
        g15 = sampler(train_r, 1.5)(b["pos"], z)
        ok["5_guidance"] = e1 < 2e-2 and e0 < 2e-2 and float(np.abs(g15 - a1["pos"]).max()) > 0
        print(f"(5) guidance beta 1 vs +: {e1:.3g}, beta 0 vs none: {e0:.3g}: {ok['5_guidance']}", flush=True)

        gfn = jax.jit(jax.grad(lambda t, x: loss_fn(t, frozen, x, rng, False)))
        grads = {name: np.linalg.norm(np.asarray(gfn(train, dict(b[name]))["tag_emb"].value), axis=1)
                 for name in ("pos", "neg", "none")}
        res["grad_norm_rows_neg_pos"] = {k: [float(x) for x in v] for k, v in grads.items()}
        ok["6_gradients"] = (grads["pos"][1] > 0 and grads["pos"][0] == 0 and grads["neg"][0] > 0
                             and grads["neg"][1] == 0 and not grads["none"].any())
        print(f"(6) grad rows (neg, pos) {res['grad_norm_rows_neg_pos']}: {ok['6_gradients']}", flush=True)

        pfm.save(train_r, out / "trainable_final.msgpack")
        (out / "adarms.json").write_text(json.dumps({"tag_scale": a.tag_scale, "seed": a.seed}))
        want = np.asarray(jax.device_get(rnd), np.float32)
        del ref, sr, gr, model, train, train_r, frozen, policy, s0, s1, gfn, jl, loss_fn, b, grads
        import gc
        gc.collect()
    pol2, _ = load_adarms(out, seed=a.seed)
    got = np.asarray(nnx.state(pol2._model)["tag_emb"].value, np.float32)
    err = float(np.abs(got - want.astype(jnp.bfloat16).astype(np.float32)).max())
    ok["7_roundtrip"] = err == 0.0 and type(pol2._model).__name__ == "TagPi0"
    print(f"(7) save/load tag_emb max |diff| (bf16) {err}: {ok['7_roundtrip']}", flush=True)
    (out / "trainable_final.msgpack").unlink()
    (out / "adarms.json").unlink()
    res["ok"] = ok
    (out / "adarms_selftest.json").write_text(json.dumps(res, indent=2))
    assert all(ok.values()), ok
    print("all ok", flush=True)


def serve(a):
    import logging
    import socket
    from openpi.serving import websocket_policy_server
    from serve_pi05_finetuned import TaggedPolicy, uses_subtask
    from vl.libero_noise import MatchedNoisePolicy
    logging.basicConfig(level=logging.INFO, force=True)
    tag = {"positive": True, "negative": False, "none": None}[a.tag]
    assert a.guidance is None or tag is True, "guidance is toward the positive tag"
    if a.contrast is not None:
        tag = None  # contrastive guidance builds the +, - and no-tag velocities itself; the prompt stays plain
    policy, _ = load_adarms(a.run, a.step, guidance=a.guidance, contrast=a.contrast)
    logging.info("serving AdaRMS run %s step %s tag %s guidance %s contrast %s (mix %s) on port %d, host %s", a.run,
                 a.step, tag, a.guidance, a.contrast, policy._model.mix, a.port, socket.gethostname())
    m = policy._model
    served = MatchedNoisePolicy(TaggedPolicy(policy, tag, a.noise_seed, subtask=uses_subtask(a.run)),
                                (m.action_horizon, m.action_dim),
                                a.noise_seed)
    websocket_policy_server.WebsocketPolicyServer(policy=served, host="0.0.0.0", port=a.port,
                                                  metadata=policy.metadata).serve_forever()


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--tag-scale", type=float, default=TAG_SCALE)
    ap.add_argument("--adarms-selftest", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--step", default="final")
    ap.add_argument("--tag", choices=["positive", "negative", "none"], default="positive")
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--contrast", type=float, default=None, help="contrastive guidance weight w: "
                    "v = v_none + w (v_pos - v_neg) [Advantage Guidance Eq. 4]")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--noise-seed", type=int, default=0)
    a, rest = ap.parse_known_args()
    if a.serve:
        return serve(a)
    if a.adarms_selftest:
        o = argparse.ArgumentParser()
        o.add_argument("--out", required=True)
        o.add_argument("--seed", type=int, default=0)
        a2 = o.parse_args(rest)
        a2.tag_scale = a.tag_scale
        return selftest(a2)
    assert "--train" in rest or "--overfit" in rest, "AdaRMS: use --train / --overfit (or --adarms-selftest)"
    out = Path(rest[rest.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "adarms.json").write_text(json.dumps({"tag_scale": a.tag_scale, "route": "adarms",
                                                 "script": Path(__file__).name}))
    sys.argv = [sys.argv[0]] + rest
    with adarms_route(a.tag_scale):
        pfm.main()


if __name__ == "__main__":
    main()
