"""Offline pi05_droid smoke test on Mulligan val frames: does it load, act sanely, and give tokens a value head can read?

Usage (openpi's own env, one GPU; frames from scripts/pi05_dump_frames.py):
    cd /iris/u/ankile/karenvo/pi05/openpi && uv run python \
        /move/u/karenvo/Projects/value-learning/scripts/pi05_smoke.py [--frames ...] [--out ...]

Offline only: no policy server, no robot. Reads the lab's cached checkpoint in place (read-only).
Reports to --out (JSON):
  1. load time, GPU memory after load
  2. policy.infer per frame (openpi's own transforms; prompt below [ours]): latency, action chunk shape and
     range (stock DROID: 15 x 8, 7 joint velocities + gripper)
  3. prefix tokens for 6.1 (value head cross-attending into frozen pi0.5): the PaliGemma prefix (images +
     prompt, pi0.5 also tokenizes the state into it) after the full LLM forward, final norm. Shape, and
     throughput at batch 1 / 8 / 32 -> hours to extract all 439k train+val frames.
Also saves the mean-pooled prefix features (n, width) next to --out for a quick probe against `value`.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config

CKPT = "/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_droid"
PROMPT = "insert the marker into the holder"   # [ours]: the dataset's task string is just "marker_d2"
N_FRAMES_ALL = 438_853                          # train + val frames (data/frames/frame_index.parquet)


def gpu_mem_gb():
    s = jax.local_devices()[0].memory_stats() or {}
    return {k: round(s[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use") if k in s}


def prefix_hidden(self, obs: _model.Observation):
    """Final-layer prefix tokens of the VLM (what the action expert attends to), and their mask."""
    obs = _model.preprocess_observation(None, obs, train=False)
    tokens, mask, ar = self.embed_prefix(obs)
    out, _ = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
    return out[0], mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/iris/u/ankile/karenvo/pi05/frames_val64.npz")
    ap.add_argument("--out", default="/iris/u/ankile/karenvo/pi05/smoke_result.json")
    a = ap.parse_args()
    f = np.load(a.frames)
    n = len(f["side"])
    obs = [{"observation/exterior_image_1_left": f["side"][i], "observation/wrist_image_left": f["wrist"][i],
            "observation/joint_position": f["joint_position"][i], "observation/gripper_position": f["gripper_position"][i],
            "prompt": PROMPT} for i in range(n)]
    res = {"devices": [str(d) for d in jax.local_devices()], "checkpoint": CKPT, "prompt": PROMPT, "n_frames": n}

    t0 = time.time()
    policy = policy_config.create_trained_policy(_config.get_config("pi05_droid"), CKPT)
    res["load_s"] = round(time.time() - t0, 1)
    res["mem_after_load_gb"] = gpu_mem_gb()

    # 2. actions through openpi's own pipeline
    lat, acts = [], []
    for i in range(n):
        t = time.time()
        acts.append(policy.infer(obs[i])["actions"])
        lat.append(time.time() - t)
    acts = np.stack(acts)
    res["infer"] = {"first_call_s": round(lat[0], 2), "median_ms": round(1000 * float(np.median(lat[1:])), 1),
                    "action_shape": list(acts.shape[1:]),
                    "joint_vel_range": [float(acts[..., :7].min()), float(acts[..., :7].max())],
                    "gripper_range": [float(acts[..., 7].min()), float(acts[..., 7].max())],
                    "finite": bool(np.isfinite(acts).all())}

    # 3. prefix tokens
    fn = nnx_utils.module_jit(prefix_hidden.__get__(policy._model))
    items = [policy._input_transform(jax.tree.map(lambda x: x, o)) for o in obs]
    def batch(ix):
        return _model.Observation.from_dict(jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *[items[i] for i in ix]))
    tok, mask = fn(batch([0]))
    res["prefix"] = {"tokens_shape": list(tok.shape), "dtype": str(tok.dtype), "valid_tokens": int(mask[0].sum())}
    thr = {}
    for bs in (1, 8, 32):
        if bs > n:
            continue
        b = batch(list(range(bs)))
        jax.block_until_ready(fn(b))                       # compile
        t, reps = time.time(), 5
        for _ in range(reps):
            jax.block_until_ready(fn(b))
        fps = bs * reps / (time.time() - t)
        thr[bs] = {"frames_per_s": round(fps, 1), "hours_all_frames": round(N_FRAMES_ALL / fps / 3600, 2)}
    res["prefix"]["throughput"] = thr
    res["mem_peak_gb"] = gpu_mem_gb()

    pooled = []
    for s in range(0, n, 8):
        tk, mk = fn(batch(list(range(s, min(s + 8, n)))))
        m = np.asarray(mk, np.float32)[..., None]
        pooled.append((np.asarray(tk, np.float32) * m).sum(1) / m.sum(1))
    np.savez(Path(a.out).with_suffix(".pooled.npz"), pooled=np.concatenate(pooled), value=f["value"],
             outcome=f["outcome"], episode_uid=f["episode_uid"], actions=acts)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
