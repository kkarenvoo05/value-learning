"""DSRL on pi0.5 LIBERO (online, noise-space SAC; vl.dsrl has the method and the paper/port notes) [2026-11-10].

A policy server like serve_pi05_finetuned.py: the unchanged LIBERO client (scripts/libero_rollouts.py --args.dsrl)
talks to it, so init states, success detection and the eval protocol are the ones every other arm uses.
- train: each chunk request gets start noise = w repeated over the action horizon, w from the SAC actor (w ~ N(0, I)
  for the first --warmup-episodes, as the official code's first collection). The client ends each episode with a
  {"dsrl_end": 1, "success": 0/1, + final images/state} request; the server then stores the episode's transitions and
  runs UTD x (decisions in the episode) SAC updates before answering (the official trajwise alternating loop).
  Checkpoints agent_<episodes>.pkl every --ckpt-every episodes (+ agent_latest.pkl), one JSON line per episode in
  dsrl_log.jsonl. [2026-11-12] Resumable: every --state-every episodes the full state (agent weights + Adam moments
  + rng, replay buffer as append-only chunks, episode / update counts) goes to <out>/dsrl_state; a restarted server
  (Slurm requeue) restores the newest one and cuts episodes.csv, dsrl_log.jsonl and the videos back to that episode,
  so the client (--args.resume) reruns exactly the lost episodes and the agent's own episode count equals the CSV's.
  Run 18434524 (before this) restarted a fresh agent at its requeue while the episode count went on. With
  --state-every 0 a train server refuses to start in a run dir that already has episodes.
- eval: --agent <agent_*.pkl>, actor sampled stochastically (the official eval also calls sample_actions), no updates.
  [2026-11-13, DSRL G1 probe] An eval request may carry "dsrl_cand" to choose where its noise comes from:
  "actor" (= no key), "gauss" (w ~ N(0, I) clipped to [-1, 1], the actor's support [ours]: base-like spread inside
  DSRL's action space) or "base" (plain pi0.5: i.i.d. N(0, I) noise over the horizon, no w). Every eval answer
  carries the w used ("dsrl_w", NaN for base) and the critic's mean-over-heads Q(s, w) ("dsrl_q") so an offline
  re-score can be checked against it. scripts/libero_steer_probe.py --args.cand-source uses this.
Base pi0.5 (pi05_libero) is frozen in both modes.

Usage (from libero_eval.sbatch with DSRL_MODE=train|eval; see that file):
  .venv/bin/python scripts/dsrl_pi05.py --mode train --out <run dir> --port 8123
  .venv/bin/python scripts/dsrl_pi05.py --mode eval --agent <run>/agent_600.pkl --port 8123
  .venv/bin/python scripts/dsrl_pi05.py --selftest           (CPU-safe: fake pi0.5, protocol + a few updates)
"""

import argparse
import json
import logging
import os
import pickle
import shutil
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
STATE = "dsrl_state"
CAND, CANDS = "dsrl_cand", ("actor", "gauss", "base")  # [2026-11-13] probe requests (module docstring)


def sync_run_dir(out: Path, episodes: int) -> dict:
    """[2026-11-12] Cut the client's files back to the restored agent: keep the first `episodes` rows of
    episodes.csv (plan order) and of dsrl_log.jsonl, delete the videos of later episodes. Returns what was cut."""
    import csv
    cut = {"csv": 0, "log": 0, "videos": 0}
    p = out / "episodes.csv"
    if p.exists():
        with open(p, newline="") as fh:
            r = csv.DictReader(fh)
            fields, rows = r.fieldnames, list(r)
        rows.sort(key=lambda x: int(x["episode"]))
        keep, cut["csv"] = rows[:episodes], max(len(rows) - episodes, 0)
        with open(p.with_suffix(".csv.tmp"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
        os.replace(p.with_suffix(".csv.tmp"), p)
        last = int(keep[-1]["episode"]) if keep else -1
        for v in (out / "videos").glob("ep*.mp4") if (out / "videos").exists() else []:
            if int(v.name[2:6]) > last:
                v.unlink()
                cut["videos"] += 1
    p = out / "dsrl_log.jsonl"
    if p.exists():
        lines = p.read_text().splitlines()
        cut["log"] = max(len(lines) - episodes, 0)
        p.with_suffix(".jsonl.tmp").write_text("".join(x + "\n" for x in lines[:episodes]))
        os.replace(p.with_suffix(".jsonl.tmp"), p)
    return cut


def sac_obs(obs, size):
    """(size, size, 3) uint8 agent view + (8,) state from a client request (image already rotated, 224 square)."""
    img = np.asarray(obs["observation/image"], np.uint8)
    if img.shape[0] != size:
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    return img, np.asarray(obs["observation/state"], np.float32).reshape(-1)


class DSRLPolicy:
    def __init__(self, policy, agent: DSRLAgent, mode: str, horizon: int, action_dim: int, out: Path | None = None,
                 warmup_episodes: int = 1, utd: int = 20, start_updates: int = 500, gamma: float = 0.999,
                 steps_per_decision: int = 5, ckpt_every: int = 100, capacity: int = 250_000, seed: int = 0,
                 state_every: int = 0, judge: tuple[float, float, int] = (0.0, 0.0, 0)):
        assert mode in ("train", "eval")
        self.judge_fp, self.judge_fn, self.judge_seed = judge  # [2026-11-14] see judged()
        assert 0 <= self.judge_fp < 1 and 0 <= self.judge_fn < 1
        self._policy, self.agent, self.mode = policy, agent, mode
        self.horizon, self.action_dim, self.out = horizon, action_dim, out
        self.warmup, self.utd, self.start_updates, self.ckpt_every = warmup_episodes, utd, start_updates, ckpt_every
        self.discount = float(gamma ** steps_per_decision)
        self.rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(capacity, agent.cfg, seed) if mode == "train" else None
        self.episodes, self.updates, self.cur = 0, 0, self._empty()
        self.size = agent.cfg.image_size
        self.state_every, self.chunks = state_every, []

    # [2026-11-12] resumable training state (module docstring)
    def save_state(self):
        root = self.out / STATE
        meta = self.buffer.save_chunks(root / "buffer", self.chunks)
        blob = {"agent": self.agent.full_state(), "episodes": self.episodes, "updates": self.updates,
                "rng": self.rng.bit_generator.state, "buffer": meta}
        name = f"state_ep{self.episodes:05d}.pkl"
        (root / (name + ".tmp")).write_bytes(pickle.dumps(blob))
        os.replace(root / (name + ".tmp"), root / name)
        (root / "latest.tmp").write_text(name)
        os.replace(root / "latest.tmp", root / "latest")  # the pointer moves only after the state is complete
        self.chunks = meta["chunks"]
        for old in root.glob("state_ep*.pkl"):
            if old.name != name:
                old.unlink()

    def restore_state(self) -> int:
        """Load <out>/dsrl_state/latest if present (else keep the fresh agent); returns the episode count. Buffer
        chunks not listed in that state (written after it, by the lost part of the run) are deleted."""
        root = self.out / STATE
        if not (root / "latest").exists():
            return 0
        blob = pickle.loads((root / (root / "latest").read_text().strip()).read_bytes())
        self.agent = DSRLAgent.from_full_state(blob["agent"])
        self.buffer.load_chunks(root / "buffer", blob["buffer"])
        self.chunks = list(blob["buffer"]["chunks"])
        for f in (root / "buffer").glob("rows_*"):
            if f.name not in self.chunks:
                f.unlink()
        self.episodes, self.updates = blob["episodes"], blob["updates"]
        self.rng.bit_generator.state = blob["rng"]
        return self.episodes

    @staticmethod
    def _empty():
        return {"pixels": [], "states": [], "actions": []}

    def judged(self, success: bool, episode: int) -> bool:
        """[2026-11-14] Noisy-judge robustness (Karen: zima's rewards come from a VLM judge): the outcome the agent
        learns from. A failure is reported as a success with probability judge_fp, a success as a failure with
        judge_fn; the draw u comes from an rng keyed (judge_seed, episode), so a resumed run judges every episode
        the same way, and the same u is used for both classes (lower rates flip a subset of what higher rates flip).
        Default (0, 0) = the simulator's outcome."""
        if not (self.judge_fp or self.judge_fn):
            return success
        u = np.random.default_rng([self.judge_seed, episode]).random()
        return (u >= self.judge_fn) if success else (u < self.judge_fp)

    def infer(self, obs: dict, noise=None) -> dict:
        obs = dict(obs)
        if END in obs:
            return self._end(obs)
        cand = obs.pop(CAND, None) or "actor"
        assert cand in CANDS, f"{CAND} must be one of {CANDS}, got {cand!r}"
        assert cand == "actor" or self.mode == "eval", f"{CAND} is for eval (probe) servers only"
        pix, st = sac_obs(obs, self.size)
        if cand == "base":  # plain pi0.5: its own i.i.d. noise; Q is undefined (no w)
            out = self._policy.infer(obs)
            out["dsrl_w"] = np.full(self.agent.cfg.noise_dim, np.nan, np.float32)
            out["dsrl_q"] = float("nan")
            return out
        if self.mode == "train" and self.episodes < self.warmup:
            w = self.rng.standard_normal(self.agent.cfg.noise_dim).astype(np.float32)
        elif cand == "gauss":
            w = np.clip(self.rng.standard_normal(self.agent.cfg.noise_dim), -1.0, 1.0).astype(np.float32)
        else:
            w = self.agent.sample(pix, st)
        if self.mode == "train":
            self.cur["pixels"].append(pix), self.cur["states"].append(st), self.cur["actions"].append(w)
        full = np.zeros((self.horizon, self.action_dim), np.float32)
        full[:] = w[: self.action_dim] if len(w) >= self.action_dim else np.pad(w, (0, self.action_dim - len(w)))
        out = self._policy.infer(obs, noise=full)
        out["dsrl_noise_norm"] = float(np.linalg.norm(w))
        if self.mode == "eval":
            out["dsrl_w"] = np.asarray(w, np.float32)
            out["dsrl_q"] = float(self.agent.q(pix[None], st[None], w[None]).mean())
        return out

    def _end(self, obs):
        ep, self.cur = self.cur, self._empty()
        if self.mode != "train" or not ep["actions"]:
            return {"actions": np.zeros((self.horizon, 7), np.float32)}
        success = bool(int(obs["success"]))
        judged = self.judged(success, self.episodes + 1)  # [2026-11-14] the reward signal; success stays the truth
        fpix, fst = sac_obs(obs, self.size)
        self.buffer.add_episode(ep["pixels"], ep["states"], ep["actions"], judged, self.discount, fpix, fst)
        self.episodes += 1
        t0, infos = time.time(), []
        if len(self.buffer) > self.start_updates:
            for _ in range(self.utd * len(ep["actions"])):
                infos.append(self.agent.update(self.buffer.sample(self.agent.cfg.batch_size)))
            self.updates += len(infos)
        rec = {"episode": self.episodes, "success": success, "judged": judged, "decisions": len(ep["actions"]),
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
            if self.state_every and self.episodes % self.state_every == 0:
                t1 = time.time()
                self.save_state()
                logging.info("dsrl state saved at episode %d (%.1f s)", self.episodes, time.time() - t1)
        return {"actions": np.zeros((self.horizon, 7), np.float32)}


class FakePolicy:
    """Stand-in for pi0.5 in --selftest: checks the noise it receives and returns a zero chunk."""

    def __init__(self, horizon, action_dim):
        self.horizon, self.action_dim, self.calls = horizon, action_dim, 0

    def infer(self, obs, noise=None):
        assert CAND not in obs, "the probe key must not reach pi0.5's input transform"
        self.calls += 1
        self.last_noise = noise
        if noise is None:  # "base" probe request: pi0.5's own noise
            return {"actions": np.zeros((self.horizon, 7), np.float32)}
        assert noise.shape == (self.horizon, self.action_dim), noise.shape
        assert np.allclose(noise, noise[0]), "DSRL noise must repeat over the horizon"
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
    selftest_probe(ev, req)
    selftest_judge()
    print("selftest ok:", lines[-1])
    selftest_resume(out / "resume")


def selftest_judge():
    """[2026-11-14] judged(): off = the truth; rates realised within 2.5% over 4000 episodes; the same episode is
    judged the same way twice (resume-safe); fp-only never touches successes; lower rates flip a subset."""
    cfg = DSRLConfig(batch_size=16)
    mk = lambda fp, fn: DSRLPolicy(FakePolicy(10, 32), DSRLAgent(0, cfg), "eval", 10, 32, judge=(fp, fn, 3))
    off, s25, s10, fp25 = mk(0, 0), mk(0.25, 0.25), mk(0.1, 0.1), mk(0.25, 0)
    eps = range(1, 4001)
    assert all(off.judged(s, e) == s for e in eps for s in (True, False))
    fl_f = np.array([s25.judged(False, e) for e in eps])
    fl_s = ~np.array([s25.judged(True, e) for e in eps])
    assert abs(fl_f.mean() - 0.25) < 0.025 and abs(fl_s.mean() - 0.25) < 0.025, (fl_f.mean(), fl_s.mean())
    assert all(s25.judged(False, e) == s25.judged(False, e) for e in range(1, 50))
    assert all(fp25.judged(True, e) for e in eps)
    f10 = np.array([s10.judged(False, e) for e in eps])
    assert not (f10 & ~fl_f).any(), "a 10% flip must also be a 25% flip"
    print("judge selftest ok: fp", fl_f.mean(), "fn", fl_s.mean(), "nested", int(f10.sum()), "of", int(fl_f.sum()))


def selftest_probe(ev: "DSRLPolicy", req):
    """[2026-11-13] Probe requests on an eval server: the w used reaches pi0.5 as the repeated noise, comes back in
    the answer with Q(s, w), an offline agent.q on the same request reproduces that Q, gauss stays in [-1, 1],
    base sends no noise, and a train server refuses the key."""
    for cand in ("actor", "gauss", None):
        r = req()
        el = dict(r, **({CAND: cand} if cand else {}))
        out = ev.infer(el)
        w = out["dsrl_w"]
        assert w.shape == (32,) and np.abs(w).max() <= 1.0 + 1e-6, (cand, w)
        assert np.allclose(ev._policy.last_noise, np.broadcast_to(w, (10, 32))), cand
        pix, st = sac_obs(r, 64)
        q_off = float(ev.agent.q(pix[None], st[None], w[None]).mean())
        assert abs(q_off - out["dsrl_q"]) < 1e-5, (cand, q_off, out["dsrl_q"])
    out = ev.infer({**req(), CAND: "base"})
    assert ev._policy.last_noise is None and np.isnan(out["dsrl_w"]).all() and np.isnan(out["dsrl_q"])
    g = np.stack([ev.infer({**req(), CAND: "gauss"})["dsrl_w"] for _ in range(50)])
    assert 0.2 < (np.abs(g) == 1.0).mean() < 0.45, "clipped N(0, I): ~32% of entries at the bound"
    tr = DSRLPolicy(FakePolicy(10, 32), DSRLAgent(0, DSRLConfig(batch_size=16)), "train", 10, 32)
    try:
        tr.infer({**req(), CAND: "gauss"})
        raise AssertionError("train server accepted a probe request")
    except AssertionError as e:
        assert "eval (probe) servers only" in str(e), e
    print("probe selftest ok: actor / gauss / base requests, Q(s, w) offline == served")


def selftest_resume(out: Path):
    """[2026-11-12] Resume = uninterrupted: 5 episodes straight vs 4 (state saved at 2 and 4) + a restart from the
    saved state at 4 (one more episode happened after the save and is lost) + episode 5 again. Final weights, Adam
    moments, buffer and logs must match exactly; the run dir is cut back to 4 episodes before the rerun."""
    import csv
    import jax
    shutil.rmtree(out, ignore_errors=True)
    cfg = DSRLConfig(batch_size=16)
    plan = [(12, 1), (15, 0), (9, 1), (11, 0), (10, 1)]

    def episode(p, e, run_dir=None):
        rng = np.random.default_rng(100 + e)  # the env: same observations for episode e in every run
        T, s = plan[e]
        for _ in range(T):
            p.infer({"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                     "observation/state": rng.standard_normal(8).astype(np.float32), "prompt": "x"})
        p.infer({"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                 "observation/state": rng.standard_normal(8).astype(np.float32), END: 1, "success": s})
        if run_dir is not None:  # what the client writes after each episode
            new = not (run_dir / "episodes.csv").exists()
            with open(run_dir / "episodes.csv", "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["episode", "success"])
                if new:
                    w.writeheader()
                w.writerow({"episode": e, "success": s})
            (run_dir / "videos").mkdir(exist_ok=True)
            (run_dir / "videos" / f"ep{e:04d}_x.mp4").write_bytes(b"")

    def make(run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        return DSRLPolicy(FakePolicy(10, 32), DSRLAgent(1, cfg), "train", 10, 32, run_dir, warmup_episodes=1, utd=2,
                          start_updates=20, ckpt_every=100, seed=1, state_every=2)
    pa = make(out / "a")
    for e in range(5):
        episode(pa, e)
    pb = make(out / "b")
    for e in range(5):  # episode 4 (the 5th) runs but its state is never saved: "preempted"
        episode(pb, e, out / "b")
    pc = make(out / "b")
    ep = pc.restore_state()
    cut = sync_run_dir(out / "b", ep)
    assert ep == 4 and cut == {"csv": 1, "log": 1, "videos": 1}, (ep, cut)
    assert len(list(csv.DictReader(open(out / "b" / "episodes.csv")))) == 4
    assert sorted(f.name for f in (out / "b" / dsrl_state_dir()).glob("state_ep*.pkl")) == ["state_ep00004.pkl"]
    episode(pc, 4, out / "b")
    fa, fc = pa.agent.full_state(), pc.agent.full_state()
    same = jax.tree.all(jax.tree.map(lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y))), fa, fc))
    assert same, "resumed agent differs from the uninterrupted one"
    assert pa.episodes == pc.episodes == 5 and pa.updates == pc.updates and len(pa.buffer) == len(pc.buffer)
    assert all(np.array_equal(pa.buffer.d[k][: pa.buffer.n], pc.buffer.d[k][: pc.buffer.n]) for k in pa.buffer.d)
    la = (out / "a" / "dsrl_log.jsonl").read_text().splitlines()
    lb = (out / "b" / "dsrl_log.jsonl").read_text().splitlines()
    strip = lambda ls: [{k: v for k, v in json.loads(x).items() if k != "update_s"} for x in ls]
    assert strip(la) == strip(lb), "logs differ"
    print(f"resume selftest ok: restored at episode {ep}, cut {cut}; 5 episodes, {pc.updates} updates, buffer "
          f"{len(pc.buffer)}, weights / Adam / rng / buffer / log identical to the uninterrupted run")


def dsrl_state_dir():
    return STATE


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
    ap.add_argument("--state-every", type=int, default=50, help="train: full resumable state every N episodes (0 off)")
    ap.add_argument("--judge-fp", type=float, default=0.0, help="[2026-11-14] train: report a failure as a success "
                    "with this probability (noisy judge; DSRLPolicy.judged)")
    ap.add_argument("--judge-fn", type=float, default=0.0, help="[2026-11-14] train: report a success as a failure")
    ap.add_argument("--judge-seed", type=int, default=0)
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
                        a.start_updates, ckpt_every=a.ckpt_every, seed=a.seed, state_every=a.state_every,
                        judge=(a.judge_fp, a.judge_fn, a.judge_seed))
    assert a.mode == "train" or not (a.judge_fp or a.judge_fn), "--judge-* are for train servers"
    if a.mode == "train":
        prior = (out / "dsrl_log.jsonl").exists() and (out / "dsrl_log.jsonl").read_text().strip()
        if a.state_every:
            ep = served.restore_state()
            cut = sync_run_dir(out, ep)
            logging.info("DSRL resume: agent at episode %d (updates %d, buffer %d); cut %s", ep, served.updates,
                         len(served.buffer), cut)
        else:
            assert not prior, f"{out} has episodes and --state-every 0 cannot resume: refusing to restart fresh"
    logging.info("DSRL %s on base pi05_libero (horizon %d, action dim %d), agent %s, port %d, host %s", a.mode,
                 m.action_horizon, m.action_dim, a.agent or "fresh", a.port, socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(policy=served, host="0.0.0.0", port=a.port,
                                                  metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
