"""Best-of-N steering pre-checks (docs/CURRENT.md direction B2, steps 1-2): do π₀.₅'s candidate action chunks
differ, and does choosing between them change the outcome?

Runs in the LIBERO client env (Python 3.8) against the base pi05_libero server (scripts/libero_steer_probe.sbatch).
For each init state, one base episode of the policy, exactly as scripts/libero_rollouts.py (wait steps, rotation,
resize, replan every 5 steps). At every replan point the server is asked for N chunks instead of one (each infer
call draws fresh flow-matching noise), then:

  Step 1, spread (every replan point): the N chunks (N, H, 7) are saved, and each candidate's first 5 actions are
      executed from the saved MuJoCo state to record where the end effector / gripper ends up (N, 3) / (N, 2).
  Step 2, headroom (only at --branch-steps): from the saved state each candidate's 5 actions are executed, then the
      base policy continues to the end of the episode, --repeats times per candidate -> success (N, R). The frames
      right after the candidate's 5 actions are saved, so an existing value function can score them offline.

The base episode then continues with candidate 0 (an ordinary policy sample), so the base episode is a normal
rollout. Branching restores the MuJoCo state, solver warm start, robosuite's step counter (the env's 1000-step horizon
would otherwise end the episode early) and the gripper's integrated command (not part of the MuJoCo state). A fidelity check repeats candidate 0 from the restored state and compares the
resulting MuJoCo state with the first execution (written to the log and the npz as restore_maxdiff).

Output: <out>/probe/epNNNN.npz per base episode and <out>/episodes.csv. Analysis: scripts/libero_steer_analyze.py.

Usage:
    python scripts/libero_steer_probe.py --args.task-ids 8 --args.init-states 9 12 --args.n-cand 8 \
        --args.branch-steps 40 130 220 310 --args.repeats 2 --args.out-dir <run dir> --args.port <port>
    --args.fake-policy: random chunks, no server (smoke test of the sim bookkeeping)
"""

import collections
import csv
import dataclasses
import logging
import pathlib
import time
from typing import List, Optional

from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

import libero_rollouts as lr


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_10"
    task_ids: List[int] = dataclasses.field(default_factory=lambda: [8])
    init_states: Optional[List[int]] = None
    base_repeats: int = 1  # base episodes per init state
    n_cand: int = 8
    # v2 [ours, 2026-11-04]: candidates 1..N-1 sample with start noise x cand_noise_scale (server: serve_pi05_finetuned.py
    # --base); candidate 0, the base episode and every continuation stay at the policy's normal noise. 1.0 = v1.
    cand_noise_scale: float = 1.0
    branch_steps: List[int] = dataclasses.field(default_factory=lambda: [40, 130, 220, 310])  # policy steps
    repeats: int = 2  # continuations per candidate at a branch step
    num_steps_wait: int = 10
    out_dir: str = "data/libero/steer_probe"
    seed: int = 7
    split_name: str = "val"
    fake_policy: bool = False
    max_steps_override: int = 0  # > 0: shorter episodes (smoke test)


class FakePolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def infer(self, element):
        a = self.rng.normal(0, 0.3, (10, 7))
        a[:, 6] = np.sign(a[:, 6])
        return {"actions": a}


def preprocess(obs, size):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, size, size))
    state = np.concatenate((obs["robot0_eef_pos"], lr._quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
    return img, wrist, state


class Sim:
    """The LIBERO env plus save / restore of everything a step depends on: the MuJoCo state, the solver warm start,
    robosuite's step counter, and the robot's non-MuJoCo state (the Panda gripper integrates its command:
    current_action += speed * sign(action); the OSC controller goals)."""

    def __init__(self, env):
        self.env = env

    def save(self):
        inner = self.env.env
        robots = [(np.copy(r.gripper.current_action), np.copy(getattr(r.controller, "goal_pos", None)),
                   np.copy(getattr(r.controller, "goal_ori", None))) for r in inner.robots]
        return (self.env.get_sim_state().copy(), np.copy(inner.sim.data.qacc_warmstart), inner.timestep, robots)

    def restore(self, snap):
        state, warm, timestep, robots = snap
        obs = self.env.set_init_state(state)
        inner = self.env.env
        inner.sim.data.qacc_warmstart[:] = warm
        inner.timestep, inner.done = timestep, False
        for r, (ga, gp, go) in zip(inner.robots, robots):
            r.gripper.current_action = np.copy(ga)
            if gp.ndim:
                r.controller.goal_pos = np.copy(gp)
            if go.ndim:
                r.controller.goal_ori = np.copy(go)
        return obs


def query(client, obs, prompt, size, noise_scale=None):
    img, wrist, state = preprocess(obs, size)
    element = {"observation/image": img, "observation/wrist_image": wrist, "observation/state": state,
               "prompt": prompt}
    if noise_scale is not None:
        element["noise_scale"] = float(noise_scale)
    return np.asarray(client.infer(element)["actions"]), (img, wrist, state)


def run_policy(env, client, obs, prompt, t, max_steps, args):
    """Base policy from the current state (policy step t) to the end. Returns (success, final policy step)."""
    plan = collections.deque()
    done = False
    while t < max_steps:
        if not plan:
            chunk, _ = query(client, obs, prompt, args.resize_size)
            plan.extend(chunk[: args.replan_steps])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        t += 1
        if done:
            break
    return bool(done), t


def execute(env, actions):
    obs, done = None, False
    for a in actions:
        obs, _, done, _ = env.step(np.asarray(a).tolist())
        if done:
            break
    return obs, bool(done)


def base_episode(env, sim, client, init_state, prompt, args, max_steps, log_prefix):
    obs = env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(lr.LIBERO_DUMMY_ACTION)
    rec = collections.defaultdict(list)
    br = collections.defaultdict(list)
    t, done, restore_maxdiff = 0, False, None
    branch = set(args.branch_steps)
    while t < max_steps and not done:
        snap = sim.save()
        cands = []
        for c in range(args.n_cand):
            ns = args.cand_noise_scale if (c > 0 and args.cand_noise_scale != 1.0) else None
            chunk, pre = query(client, obs, prompt, args.resize_size, ns)
            cands.append(chunk)
            if c == 0:
                img, wrist, state = pre
        cands = np.stack(cands)  # (N, H, 7)
        k = min(args.replan_steps, max_steps - t)
        eef, grip, cand_done = [], [], []
        is_branch = t in branch
        after_img, after_wrist, after_state, succ, steps = [], [], [], [], []
        for c in range(args.n_cand):
            sim.restore(snap)
            o, d = execute(env, cands[c, :k])
            eef.append(o["robot0_eef_pos"].copy())
            grip.append(o["robot0_gripper_qpos"].copy())
            cand_done.append(d)
            if c == 0 and restore_maxdiff is None:  # fidelity: same chunk from the same restored state, again
                s0 = env.get_sim_state().copy()
                sim.restore(snap)
                execute(env, cands[0, :k])
                restore_maxdiff = float(np.abs(env.get_sim_state() - s0).max())
                logging.info(f"{log_prefix} restore fidelity: max |state diff| = {restore_maxdiff:.3g}")
            if is_branch:
                ai, aw, ast = preprocess(o, args.resize_size)
                after_img.append(ai)
                after_wrist.append(aw)
                after_state.append(ast)
                after_snap = sim.save()
                s_r, n_r = [], []
                for r in range(args.repeats):
                    if d:
                        s_r.append(True)
                        n_r.append(t + k)
                        continue
                    o2 = sim.restore(after_snap)
                    s, tt = run_policy(env, client, o2, prompt, t + k, max_steps, args)
                    s_r.append(s)
                    n_r.append(tt)
                succ.append(s_r)
                steps.append(n_r)
        if is_branch:
            br["t"].append(t)
            br["success"].append(np.array(succ))  # (N, R)
            br["steps"].append(np.array(steps))
            br["after_image"].append(np.stack(after_img))
            br["after_wrist"].append(np.stack(after_wrist))
            br["after_state"].append(np.stack(after_state))
            br["chunks"].append(cands)
            logging.info(f"{log_prefix} branch t={t}: success per candidate {np.array(succ).mean(1).round(2).tolist()}")
        rec["t"].append(t)
        rec["image"].append(img)
        rec["wrist_image"].append(wrist)
        rec["state"].append(state)
        rec["chunks"].append(cands)
        rec["eef_after"].append(np.stack(eef))
        rec["grip_after"].append(np.stack(grip))
        rec["cand_done"].append(np.array(cand_done))
        # continue the base episode with candidate 0
        obs = sim.restore(snap)
        obs, done = execute(env, cands[0, :k])
        rec["executed"].append(cands[0, :k])
        t += k
    out = {f"rp_{k_}": np.asarray(v) for k_, v in rec.items() if k_ != "executed"}
    out["executed"] = np.concatenate(rec["executed"])
    for k_, v in br.items():
        out[f"br_{k_}"] = np.asarray(v)
    out["restore_maxdiff"] = np.array(restore_maxdiff if restore_maxdiff is not None else np.nan)
    out["cand_noise_scale"] = np.array(args.cand_noise_scale)
    return done, t, out


def main(args: Args) -> None:
    np.random.seed(args.seed)
    out = pathlib.Path(args.out_dir)
    (out / "probe").mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = args.max_steps_override or lr.MAX_STEPS[args.task_suite_name]
    client = FakePolicy() if args.fake_policy else _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    fields = ["episode", "task_id", "init_state", "repeat", "success", "policy_steps", "n_branch", "restore_maxdiff",
              "seconds", "file"]
    with open(out / "episodes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        episode = 0
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            ids = args.init_states if args.init_states is not None else list(range(len(init_states)))
            env, prompt = lr._get_libero_env(task, lr.LIBERO_ENV_RESOLUTION, args.seed)
            sim = Sim(env)
            for r in range(args.base_repeats):
                for i in ids:
                    t0 = time.time()
                    prefix = f"ep {episode} init {i} r{r}"
                    done, steps, data = base_episode(env, sim, client, init_states[i], str(prompt), args,
                                                     max_steps, prefix)
                    name = f"ep{episode:04d}.npz"
                    np.savez_compressed(out / "probe" / name, task_id=task_id, init_state_id=i, repeat=r,
                                        success=done, policy_steps=steps, prompt=np.array(str(prompt)), **data)
                    row = {"episode": episode, "task_id": task_id, "init_state": i, "repeat": r,
                           "success": int(done), "policy_steps": steps, "n_branch": len(data.get("br_t", [])),
                           "restore_maxdiff": float(data["restore_maxdiff"]), "seconds": round(time.time() - t0),
                           "file": name}
                    writer.writerow(row)
                    fh.flush()
                    logging.info(f"{prefix}: success={done} steps={steps} ({row['seconds']} s)")
                    episode += 1
            env.close()
    logging.info(f"Done: {episode} base episodes -> {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
