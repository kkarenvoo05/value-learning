"""DSRL on pi0.5 LIBERO (online, noise-space SAC; vl.dsrl has the method and the paper/port notes) [2026-11-10].

A policy server like serve_pi05_finetuned.py: the unchanged LIBERO client (scripts/libero_rollouts.py --args.dsrl)
talks to it, so init states, success detection and the eval protocol are the ones every other arm uses.
- train: each chunk request gets start noise = w repeated over the action horizon, w from the SAC actor (w ~ N(0, I)
  for the first --warmup-episodes, as the official code's first collection). The client ends each episode with a
  {"dsrl_end": 1, "success": 0/1, + final images/state} request; the server then stores the episode's transitions and
  runs UTD x (decisions in the episode) SAC updates before answering (the official trajwise alternating loop).
  Checkpoints agent_<episodes>.pkl every --ckpt-every episodes (+ agent_latest.pkl), one JSON line per episode in
  dsrl_log.jsonl. Not resumable: a restart starts a fresh agent (the replay buffer is in memory).
- eval: --agent <agent_*.pkl>, actor sampled stochastically (the official eval also calls sample_actions), no updates.
Base pi0.5 (pi05_libero) is frozen in both modes.

Usage (from libero_eval.sbatch with DSRL_MODE=train|eval; see that file):
  .venv/bin/python scripts/dsrl_pi05.py --mode train --out <run dir> --port 8123
  .venv/bin/python scripts/dsrl_pi05.py --mode eval --agent <run>/agent_600.pkl --port 8123
  .venv/bin/python scripts/dsrl_pi05.py --selftest           (CPU-safe: fake pi0.5, protocol + a few updates)
"""

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl.dsrl import DSRLAgent, DSRLConfig, ReplayBuffer  # noqa: E402

END = "dsrl_end"


def sac_obs(obs, size):
    """(size, size, 3) uint8 agent view + (8,) state from a client request (image already rotated, 224 square)."""
    img = np.asarray(obs["observation/image"], np.uint8)
    if img.shape[0] != size:
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    return img, np.asarray(obs["observation/state"], np.float32).reshape(-1)


class DSRLPolicy:
    def __init__(self, policy, agent: DSRLAgent, mode: str, horizon: int, action_dim: int, out: Path | None = None,
                 warmup_episodes: int = 1, utd: int = 20, start_updates: int = 500, gamma: float = 0.999,
                 steps_per_decision: int = 5, ckpt_every: int = 100, capacity: int = 250_000, seed: int = 0):
        assert mode in ("train", "eval")
        self._policy, self.agent, self.mode = policy, agent, mode
        self.horizon, self.action_dim, self.out = horizon, action_dim, out
        self.warmup, self.utd, self.start_updates, self.ckpt_every = warmup_episodes, utd, start_updates, ckpt_every
        self.discount = float(gamma ** steps_per_decision)
        self.rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(capacity, agent.cfg, seed) if mode == "train" else None
        self.episodes, self.updates, self.cur = 0, 0, self._empty()
        self.size = agent.cfg.image_size

    @staticmethod
    def _empty():
        return {"pixels": [], "states": [], "actions": []}

    def infer(self, obs: dict, noise=None) -> dict:
        obs = dict(obs)
        if END in obs:
            return self._end(obs)
        pix, st = sac_obs(obs, self.size)
        if self.mode == "train" and self.episodes < self.warmup:
            w = self.rng.standard_normal(self.agent.cfg.noise_dim).astype(np.float32)
        else:
            w = self.agent.sample(pix, st)
        if self.mode == "train":
            self.cur["pixels"].append(pix), self.cur["states"].append(st), self.cur["actions"].append(w)
        full = np.zeros((self.horizon, self.action_dim), np.float32)
        full[:] = w[: self.action_dim] if len(w) >= self.action_dim else np.pad(w, (0, self.action_dim - len(w)))
        out = self._policy.infer(obs, noise=full)
        out["dsrl_noise_norm"] = float(np.linalg.norm(w))
        return out

    def _end(self, obs):
        ep, self.cur = self.cur, self._empty()
        if self.mode != "train" or not ep["actions"]:
            return {"actions": np.zeros((self.horizon, 7), np.float32)}
        success = bool(int(obs["success"]))
        fpix, fst = sac_obs(obs, self.size)
        self.buffer.add_episode(ep["pixels"], ep["states"], ep["actions"], success, self.discount, fpix, fst)
        self.episodes += 1
        t0, infos = time.time(), []
        if len(self.buffer) > self.start_updates:
            for _ in range(self.utd * len(ep["actions"])):
                infos.append(self.agent.update(self.buffer.sample(self.agent.cfg.batch_size)))
            self.updates += len(infos)
        rec = {"episode": self.episodes, "success": success, "decisions": len(ep["actions"]),
               "buffer": len(self.buffer), "updates": self.updates, "update_s": round(time.time() - t0, 2),
               "noise_norm": float(np.mean([np.linalg.norm(a) for a in ep["actions"]])),
               "warmup": self.episodes <= self.warmup}
        if infos:
            rec.update({k: float(np.mean([float(i[k]) for i in infos[-50:]])) for k in infos[-1]})
        logging.info("dsrl %s", rec)
        if self.out is not None:
            with open(self.out / "dsrl_log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            if self.episodes % self.ckpt_every == 0:
                self.agent.save(self.out / f"agent_{self.episodes}.pkl")
            self.agent.save(self.out / "agent_latest.pkl")
        return {"actions": np.zeros((self.horizon, 7), np.float32)}


class FakePolicy:
    """Stand-in for pi0.5 in --selftest: checks the noise it receives and returns a zero chunk."""

    def __init__(self, horizon, action_dim):
        self.horizon, self.action_dim, self.calls = horizon, action_dim, 0

    def infer(self, obs, noise=None):
        assert noise is not None and noise.shape == (self.horizon, self.action_dim), noise.shape
        assert np.allclose(noise, noise[0]), "DSRL noise must repeat over the horizon"
        self.calls += 1
        return {"actions": np.zeros((self.horizon, 7), np.float32)}


def selftest(out: Path):
    """Protocol + learner check without pi0.5: 3 short episodes (1 warmup), updates after 20 transitions."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = DSRLConfig(batch_size=16)
    p = DSRLPolicy(FakePolicy(10, 32), DSRLAgent(0, cfg), "train", 10, 32, out, warmup_episodes=1, utd=2,
                   start_updates=20, ckpt_every=2)
    rng = np.random.default_rng(0)
    def req():
        return {"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                "observation/state": rng.standard_normal(8).astype(np.float32), "prompt": "x"}
    for e, (T, s) in enumerate([(12, 1), (15, 0), (9, 1)]):
        for _ in range(T):
            assert p.infer(req())["actions"].shape == (10, 7)
        p.infer({**req(), END: 1, "success": s})
    lines = [json.loads(x) for x in (out / "dsrl_log.jsonl").read_text().splitlines()]
    assert [r["episode"] for r in lines] == [1, 2, 3] and lines[0]["updates"] == 0 and lines[-1]["updates"] > 0, lines
    assert len(p.buffer) == 36 and p.buffer.d["mask"][11] == 0 and p.buffer.d["reward"][11] == 0
    assert p.buffer.d["mask"][26] == 1 and p.buffer.d["reward"][26] == -1  # timeout: bootstrap, -1
    assert all(np.isfinite(v) for k, v in lines[-1].items() if isinstance(v, float)), lines[-1]
    a2 = DSRLAgent.load(out / "agent_2.pkl")
    ev = DSRLPolicy(FakePolicy(10, 32), a2, "eval", 10, 32)
    w = [np.abs(np.asarray(ev.agent.sample(*sac_obs(req(), 64)))).max() for _ in range(5)]
    assert max(w) <= 1.0 + 1e-6, w  # actor noise stays inside the tanh bound
    ev.infer({**req(), END: 1, "success": 1})  # eval mode ignores episode ends
    print("selftest ok:", lines[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="train: run dir for checkpoints + dsrl_log.jsonl")
    ap.add_argument("--agent", help="eval: agent_*.pkl")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-episodes", type=int, default=1)
    ap.add_argument("--utd", type=int, default=20)
    ap.add_argument("--start-updates", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=100)
    a = ap.parse_args()
    if a.selftest:
        return selftest(Path(a.out or "/tmp/dsrl_selftest"))
    assert a.mode and (a.mode == "eval") == bool(a.agent), "--mode eval needs --agent; train must not have it"
    from openpi.serving import websocket_policy_server
    from pi05_policy_finetune import base_policy
    policy = base_policy()[1]
    m = policy._model
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "dsrl_args.json").write_text(json.dumps(vars(a), indent=1))
    agent = DSRLAgent.load(a.agent, a.seed) if a.agent else DSRLAgent(a.seed)
    served = DSRLPolicy(policy, agent, a.mode, m.action_horizon, m.action_dim, out, a.warmup_episodes, a.utd,
                        a.start_updates, ckpt_every=a.ckpt_every, seed=a.seed)
    logging.info("DSRL %s on base pi05_libero (horizon %d, action dim %d), agent %s, port %d, host %s", a.mode,
                 m.action_horizon, m.action_dim, a.agent or "fresh", a.port, socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(policy=served, host="0.0.0.0", port=a.port,
                                                  metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
