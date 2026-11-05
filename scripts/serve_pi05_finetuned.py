"""Serve a Stage 6 fine-tuned policy (scripts/pi05_policy_finetune.py) to the unchanged LIBERO client.

Usage (openpi env, from the openpi clone):
    .venv/bin/python <vl>/scripts/serve_pi05_finetuned.py --run <run dir> --port 8000 [--tag positive|negative|none]
    .venv/bin/python <vl>/scripts/serve_pi05_finetuned.py --base --port 8000    # plain pi05_libero, same code path

The client sends the task prompt; --tag rewrites it with vl.libero_finetune_data.tag_prompt before the policy's
input transform, exactly as in training. Default: positive for arm C runs (results.json data.arm == advantage),
none otherwise. The negative tag is the plan's Stage 6 check (C with the negative indicator must do worse).

Per-request noise scale [ours, 2026-11-04, steering pre-check v2]: if the client's observation carries
"noise_scale" (float), the flow-matching start noise is drawn as noise_scale * N(0, I) (server-side numpy rng,
--noise-seed) instead of the policy's own N(0, I). Requests without the key are served exactly as before.
"""

import argparse
import logging
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi05_policy_finetune import base_policy, load_finetuned  # noqa: E402
from vl import libero_finetune_data as fd  # noqa: E402


class TaggedPolicy:
    def __init__(self, policy, tag, noise_seed: int = 0):
        self._policy, self._tag, self.calls = policy, tag, 0
        self._rng = np.random.default_rng(noise_seed)
        m = policy._model
        self._noise_shape = (m.action_horizon, m.action_dim)

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        scale = obs.pop("noise_scale", None)
        obs["prompt"] = fd.tag_prompt(str(obs["prompt"]), self._tag)
        if self.calls == 0:
            logging.info("first prompt: %r", obs["prompt"])
        self.calls += 1
        if scale is None:
            return self._policy.infer(obs)
        noise = (float(scale) * self._rng.standard_normal(self._noise_shape)).astype(np.float32)
        return self._policy.infer(obs, noise=noise)

    def reset(self):
        self._policy.reset()


def main():
    from openpi.serving import websocket_policy_server
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--base", action="store_true")
    ap.add_argument("--step", default="final")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tag", choices=["positive", "negative", "none", "auto"], default="auto")
    ap.add_argument("--noise-seed", type=int, default=0, help="rng for requests that carry noise_scale")
    a = ap.parse_args()
    if a.base:
        policy, arm = base_policy()[1], "base"
    else:
        policy, res = load_finetuned(a.run, a.step, seed=res_seed(a.run))
        arm = res.get("data", {}).get("arm", "?")
    tag = {"positive": True, "negative": False, "none": None}.get(a.tag, True if arm == "advantage" else None)
    logging.info("serving %s (arm %s, tag %s) on port %d, host %s", a.run or "pi05_libero", arm, tag, a.port,
                 socket.gethostname())
    server = websocket_policy_server.WebsocketPolicyServer(policy=TaggedPolicy(policy, tag, a.noise_seed), host="0.0.0.0",
                                                           port=a.port, metadata=policy.metadata)
    server.serve_forever()


def res_seed(run) -> int:
    """The run's seed: lora_a is re-drawn from it before the saved weights overwrite every trainable leaf."""
    import json
    p = Path(run) / "results.json"
    return int(json.loads(p.read_text())["args"]["seed"]) if p.exists() else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
