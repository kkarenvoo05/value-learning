"""Classifier-free guidance on the advantage tag for a fine-tuned pi05 policy [CFGRL / RECAP §IV-B: with indicator
dropout in training, sample with v = v_uncond + beta * (v_cond - v_uncond); beta = 1 is plain tagged sampling].

The tagged and untagged observations go through the policy's own input transform and are stacked as a batch of 2
that shares one start noise; each flow-matching step computes both velocities and moves both rows with the guided
one, so row 0 is the guided chunk. The loop is openpi's Pi0.sample_actions (src/openpi/models/pi0.py, this
checkout), copied with only the guidance line added [ours].

Self-check (openpi env; π₀.₅ needs a GPU or a 96G CPU sbatch, see scripts/debug/):
    .venv/bin/python <vl>/scripts/pi05_guidance.py --selftest [--run <fine-tuned run>]
    beta = 1 must equal policy.infer with the tag, beta = 0 policy.infer without it (same noise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl import libero_finetune_data as fd  # noqa: E402


def _guided(model, observation, noise, beta, num_steps: int = 10):
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    assert batch_size == 2, "row 0 = tagged, row 1 = untagged"
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        p_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([p_mask, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=pos,
                                                 kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        v = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        v_g = v[1] + beta * (v[0] - v[1])  # the guidance line [CFGRL]
        return x_t + dt * jnp.broadcast_to(v_g, v.shape), time + dt

    x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (jnp.concatenate([noise, noise]), 1.0))
    return x_0[:1]


class GuidedPolicy:
    """Wraps an openpi Policy: infer(obs) returns the beta-guided chunk for the positive tag (client prompt = task)."""

    def __init__(self, policy, beta: float, seed: int = 0):
        self._policy, self.beta = policy, float(beta)
        graphdef, state = nnx.split(policy._model)
        self._fn = jax.jit(lambda st, obs, noise, beta: _guided(nnx.merge(graphdef, st), obs, noise, beta))
        self._state = state
        self._rng = jax.random.PRNGKey(seed)
        m = policy._model
        self._shape = (1, m.action_horizon, m.action_dim)

    @property
    def metadata(self):
        return self._policy.metadata

    def reset(self):
        self._policy.reset()

    def infer(self, obs: dict, *, noise=None, beta=None) -> dict:
        from openpi.models import model as _model
        p = self._policy
        rows = []
        for tag in (True, None):
            o = jax.tree.map(lambda x: x, dict(obs))
            o["prompt"] = fd.tag_prompt(str(obs["prompt"]), tag)
            rows.append(p._input_transform(o))
        inputs = jax.tree.map(lambda a, b: jnp.stack([jnp.asarray(a), jnp.asarray(b)]), rows[0], rows[1])
        if noise is None:
            self._rng, k = jax.random.split(self._rng)
            noise = jax.random.normal(k, self._shape)
        noise = jnp.asarray(noise).reshape(self._shape)
        acts = self._fn(self._state, _model.Observation.from_dict(inputs), noise,
                        jnp.float32(self.beta if beta is None else beta))
        out = p._output_transform({"state": np.asarray(inputs["state"][0]), "actions": np.asarray(acts[0])})
        return out


def selftest(a):
    from pi05_policy_finetune import base_policy, load_finetuned
    from serve_pi05_finetuned import res_seed
    policy = load_finetuned(a.run, a.step, seed=res_seed(a.run))[0] if a.run else base_policy()[1]
    eps = __import__("pandas").read_csv(fd.TABLE / "episodes.csv")
    r = eps[eps.split == "val"].iloc[0]
    ep = fd.load_episode(r.raw, None, int(r.episode_index))
    g = GuidedPolicy(policy, 1.0)
    rng = np.random.default_rng(0)
    worst = {}
    for t in (0, len(ep["actions"]) // 2):
        s = fd.sample(ep, t, None)
        obs = {k: v for k, v in s.items() if k != "actions"}
        noise = rng.standard_normal(g._shape[1:]).astype(np.float32)
        for beta, tag in ((1.0, True), (0.0, None)):
            ref = policy.infer(dict(obs, prompt=fd.tag_prompt(obs["prompt"], tag)), noise=noise)["actions"]
            got = g.infer(obs, noise=noise, beta=beta)["actions"]
            err = float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-8))
            worst[beta] = max(worst.get(beta, 0.0), err)
        a15 = g.infer(obs, noise=noise, beta=1.5)["actions"]
        print(f"t={t}: beta 1.5 moves the chunk by {np.abs(a15 - ref).max():.4f} vs untagged", flush=True)
    ok = all(v < 2e-2 for v in worst.values())  # bf16, batch 2 vs batch 1 programs
    print(f"guidance self-check: rel max err beta=1 {worst[1.0]:.2e}, beta=0 {worst[0.0]:.2e}: {'ok' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", required=True)
    ap.add_argument("--run")
    ap.add_argument("--step", default="final")
    selftest(ap.parse_args())
