"""Ground-truth value of the base policy from simulator branching (reports/2026-11-06-tag-hypotheses/, E1).

LIBERO is deterministic given the actions (open-loop replay is exact, scripts/libero_replay.py), so given the init
state the outcome depends only on the policy's sampling noise. The true value of a recorded frame,
V*(s_t) = P(success | s_t, base policy), can therefore be estimated directly: replay the recorded actions up to t,
save the simulator, and run K fresh base-policy continuations from there to the end of the episode.

Questions it answers [ours]:
  1. Where in an episode the outcome is decided (where V* jumps), so steering / headroom tests can be placed at those
     frames instead of fixed steps.
  2. A truth to score the learned value functions and the advantage labels against (instead of agreement between
     two noisy critics).
  3. Whether a correct per-frame advantage exists at all: A*_t = V*(s_{t+stride}) − V*(s_t) along the recorded path.

Runs in the LIBERO client env (Python 3.8) against the base pi05_libero server (scripts/libero_branch_value.sbatch).
For each table episode (tables/moka_fixed_fresh/episodes.csv, by episode_index): env with the recording's env seed,
the stored init state, the same 10 wait steps, then the recorded actions. At every policy step t with
t % stride == 0 (all replan points, the policy replans every 5 steps from t = 0): save, K continuations of the base
policy (replanning every 5 steps from t, as the rollout did), restore, continue the replay. Fidelity: the robot state at
each branch point is compared with the recorded state[t] (exact with ExactSim) and the images with the recorded ones
(exact on the recording GPUs; renderer-dependent elsewhere), and the replay's outcome with the recorded one.

Output: <out>/branch/epNNNN.npz (t, success (P, K), end_step (P, K), state_maxdiff (P,)), <out>/episodes.csv.

Usage:
    python scripts/libero_branch_value.py --args.episodes 3 17 ... --args.k 6 --args.stride 20 --args.out-dir <dir>
    --args.fake-policy: random chunks, no server (smoke test of replay + branching bookkeeping)
"""

import copy
import csv
import dataclasses
import logging
import pathlib
import time
from typing import List

from libero.libero import benchmark
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

import libero_rollouts as lr
from libero_steer_probe import FakePolicy, Sim, preprocess, run_policy

TABLE = pathlib.Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/episodes.csv")


@dataclasses.dataclass
class Args:
    episodes: List[int] = dataclasses.field(default_factory=list)  # episode_index in the moka table
    host: str = "127.0.0.1"
    port: int = 8000
    k: int = 6  # continuations per branch point
    stride: int = 20  # policy steps between branch points (multiple of replan_steps)
    t_max: int = 520  # no branch points at or after this policy step
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    task_suite_name: str = "libero_10"
    task_id: int = 8
    out_dir: str = "data/libero/branch_value"
    fake_policy: bool = False
    max_steps_override: int = 0  # > 0: shorter episodes (smoke test)


class ExactSim(Sim):
    """Sim (libero_steer_probe) restore made exact against an *unbranched* trajectory [ours, 2026-11-06].

    Sim.restore goes through LIBERO's set_init_state, which sets the MuJoCo state and then forces every robosuite
    observable to resample. The physics is then exact, but the observables' sampling timers are reset, so after a
    restore the observations (robot state, images) come from a different physics substep than in a run that was never
    restored: the replayed robot state drifted from the recording by up to 0.011 within 30 steps (smoke test on ep 165).
    Steer-probe's fidelity check compared two restored runs with each other, so it could not see this.
    Here every writable numeric field of MjData is copied in place (no mj_forward), and each observable's timer,
    delay, last value and the observation cache are saved and restored too. Checked: replay with two random
    15-step branches at t = 10 and 30 reproduces the recorded robot state to 4e-15 (old Sim: 0.011)."""

    _fields = None

    def save(self):
        inner = self.env.env
        d = inner.sim.data._data
        if self._fields is None:
            self._fields = []
            for n in dir(d):
                if n.startswith("_"):
                    continue
                try:
                    v = getattr(d, n)
                except Exception:
                    continue
                # efc_* / island_efcind: constraint-solver outputs sized by the current number of constraints
                # (nefc 99..183 over an episode), rebuilt from qpos/qvel at every step; the solver's warm start is
                # qacc_warmstart (fixed size, copied). Copying them fails when nefc differs (job 18353523).
                if n.startswith("efc_") or n == "island_efcind":
                    continue
                if isinstance(v, np.ndarray) and v.flags.writeable and v.dtype.kind in "fiub":
                    self._fields.append(n)
        data = {n: np.copy(getattr(d, n)) for n in self._fields}
        data["time"] = d.time
        robots = [(np.copy(r.gripper.current_action), np.copy(getattr(r.controller, "goal_pos", None)),
                   np.copy(getattr(r.controller, "goal_ori", None))) for r in inner.robots]
        obsv = {k: (o._time_since_last_sample, o._current_delay, copy.deepcopy(o._current_observed_value), o._sampled)
                for k, o in inner._observables.items()}
        return data, inner.timestep, robots, obsv, copy.deepcopy(inner._obs_cache)

    def restore(self, snap):
        data, timestep, robots, obsv, cache = snap
        inner = self.env.env
        d = inner.sim.data._data
        for n in self._fields:
            np.copyto(getattr(d, n), data[n])
        d.time = data["time"]
        inner.timestep, inner.done = timestep, False
        for r, (ga, gp, go) in zip(inner.robots, robots):
            r.gripper.current_action = np.copy(ga)
            if gp.ndim:
                r.controller.goal_pos = np.copy(gp)
            if go.ndim:
                r.controller.goal_ori = np.copy(go)
        for k, (ts, dl, val, sm) in obsv.items():
            o = inner._observables[k]
            o._time_since_last_sample, o._current_delay, o._sampled = ts, dl, sm
            o._current_observed_value = copy.deepcopy(val)
        inner._obs_cache = copy.deepcopy(cache)
        return inner._get_observations()


def table_rows(ids):
    with open(TABLE) as fh:
        rows = {int(r["episode_index"]): r for r in csv.DictReader(fh)}
    return [rows[i] for i in ids]


def env_seed(raw_path: pathlib.Path) -> int:
    """The env seed of the recording run (its episodes.csv, row with this raw file)."""
    with open(raw_path.parent.parent / "episodes.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("raw") == raw_path.name:
                return int(r["env_seed"])
    raise KeyError(f"no env_seed for {raw_path}")


def branch_episode(env, sim, client, raw, prompt, args, max_steps, prefix):
    env.reset()
    obs = env.set_init_state(raw["sim_init"])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(lr.LIBERO_DUMMY_ACTION)
    start_diff = float(np.abs(env.get_sim_state() - raw["sim_start"]).max())
    actions = raw["actions"]
    T = len(actions)
    ts, succ, ends, sdiff, pxdiff = [], [], [], [], []
    done = False
    for t in range(T):
        if t % args.stride == 0 and t < args.t_max:
            img, wrist, state = preprocess(obs, args.resize_size)
            sdiff.append(float(np.abs(state - raw["state"][t]).max()))
            pxdiff.append(max(int(np.abs(img.astype(int) - raw["image"][t]).max()),
                              int(np.abs(wrist.astype(int) - raw["wrist_image"][t]).max())))
            snap = sim.save()
            s_k, e_k = [], []
            for _ in range(args.k):
                o = sim.restore(snap)
                s, tt = run_policy(env, client, o, prompt, t, max_steps, args)
                s_k.append(s)
                e_k.append(tt)
            obs = sim.restore(snap)
            ts.append(t)
            succ.append(s_k)
            ends.append(e_k)
            logging.info(f"{prefix} t={t}: V* = {np.mean(s_k):.2f} ({sum(s_k)}/{args.k}), state diff {sdiff[-1]:.2g}, px diff {pxdiff[-1]}")
        obs, _, done, _ = env.step(actions[t].tolist())
        if done:
            break
    return dict(t=np.array(ts), success=np.array(succ, bool), end_step=np.array(ends), state_maxdiff=np.array(sdiff),
                pixel_maxdiff=np.array(pxdiff),
                start_maxdiff=np.array(start_diff), replay_success=np.array(bool(done)), T=np.array(T))


def main(args: Args) -> None:
    assert args.stride % args.replan_steps == 0, "branch points must be replan points"
    out = pathlib.Path(args.out_dir)
    (out / "branch").mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    max_steps = args.max_steps_override or lr.MAX_STEPS[args.task_suite_name]
    client = FakePolicy() if args.fake_policy else _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    fields = ["episode_index", "init_state", "layout", "success", "replay_success", "policy_steps", "n_points",
              "start_maxdiff", "state_maxdiff", "pixel_maxdiff", "seconds", "file"]
    with open(out / "episodes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in table_rows(args.episodes):
            t0 = time.time()
            raw_path = pathlib.Path(r["raw"])
            with np.load(raw_path) as z:
                raw = {k: z[k] for k in ("sim_init", "sim_start", "actions", "state", "image", "wrist_image", "prompt")}
            env, prompt = lr._get_libero_env(task, lr.LIBERO_ENV_RESOLUTION, env_seed(raw_path))
            sim = ExactSim(env)
            prefix = f"ep {r['episode_index']} ({'success' if r['success'] == '1' else 'failure'})"
            res = branch_episode(env, sim, client, raw, str(prompt), args, max_steps, prefix)
            env.close()
            name = f"ep{int(r['episode_index']):04d}.npz"
            np.savez_compressed(out / "branch" / name, episode_index=int(r["episode_index"]),
                                recorded_success=int(r["success"]), **res)
            row = {"episode_index": r["episode_index"], "init_state": r["init_state"], "layout": r["layout"],
                   "success": r["success"], "replay_success": int(res["replay_success"]),
                   "policy_steps": r["policy_steps"], "n_points": len(res["t"]),
                   "start_maxdiff": float(res["start_maxdiff"]), "state_maxdiff": float(res["state_maxdiff"].max()),
                   "pixel_maxdiff": int(res["pixel_maxdiff"].max()),
                   "seconds": round(time.time() - t0), "file": name}
            w.writerow(row)
            fh.flush()
            logging.info(f"{prefix}: done {row}")
    logging.info(f"Done -> {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
