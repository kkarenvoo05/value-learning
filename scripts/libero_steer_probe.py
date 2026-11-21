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

DSRL probe [2026-11-13, G1 with DSRL's critic; server: scripts/dsrl_pi05.py --mode eval, which answers each
request with the noise w it used and its critic's Q(s, w)]: --args.cand-source actor|gauss sends every one of the N
candidates as a DSRL probe request (actor samples, or w ~ N(0, I) clipped to [-1, 1]); the base episode then takes
one more, ordinary request (the DSRL actor) for the chunk it executes, so the trunk is the DSRL policy whatever the
candidates are. Continuations are ordinary requests (the DSRL actor) unless --args.continue-source base. The w and
served Q of every candidate are saved (rp_cand_w / rp_cand_q, br_w / br_q). Scoring: scripts/libero_dsrl_g1.py.

Offset probe [2026-11-20, structured-offset G0; Karen: "set up the structured-offset G0 probe in sim"]:
--args.cand-offsets none x+ x- y+ y- z+ z- z+0.5 makes the N candidates the SAME policy chunk plus a constant
end-effector offset on the arm's position dims (0-2), instead of N noise samples. One chunk is queried per replan
point (plain base server); candidate c executes chunk[:5] + o_c, clipped to [-1, 1]; rotation and gripper untouched.
At branch steps the offset is held for --args.hold-chunks chunks: after the first chunk the policy is re-queried
from the candidate's own state (closed loop) and o_c is added again; then the base policy continues, --repeats
times. At other replan points only the first chunk is executed (spread bookkeeping). o_c = dir_c * scale_c *
offset_m / (offset_cal * replan_steps * hold_chunks), so that the summed offset over the hold moves the end effector
by about offset_m * scale_c open loop. offset_cal = 0.0075 m per unit of summed action over one 5-step chunk [ours:
regression of within-state end-effector differences on summed action differences between G0 candidates,
steer_v2_scale2p0 18329176, r 0.76-0.84 per axis]. The first label must be "none" (candidate 0 = the plain policy,
which is also what the base episode executes). Saved: cand_offsets (N, 7) per-step action offsets, offset_labels,
br_seg_eef (end effector after the held segment) and br_seg_steps.

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
    # DSRL probe [2026-11-13] (module docstring): "" = off; "actor" / "gauss" = candidate noise source on a DSRL server
    cand_source: str = ""
    continue_source: str = ""  # "" = the server's default policy (DSRL actor); "base" = plain pi0.5 continuations
    # offset probe [2026-11-20] (module docstring): [] = off; labels "none", "x+", "z-", "z+0.5", ... (sets n_cand)
    cand_offsets: List[str] = dataclasses.field(default_factory=list)
    offset_m: float = 0.02  # target open-loop end-effector displacement of a scale-1 offset over the held segment
    hold_chunks: int = 1  # chunks the offset is held for at branch steps (1 = one decision, as G0)
    offset_cal: float = 0.0075  # m of end-effector motion per unit of summed action over one 5-step chunk
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
        out = {"actions": a}
        if "dsrl_cand" in element:  # mimics the DSRL probe server's answer
            out["dsrl_w"] = np.clip(self.rng.normal(0, 1, 32), -1, 1).astype(np.float32)
            out["dsrl_q"] = float(self.rng.normal())
        return out


AXES = {"x": 0, "y": 1, "z": 2}


def parse_offsets(labels, offset_m, cal, replan_steps, hold_chunks):
    """Offset labels -> (N, 7) per-step action offsets. "none" = 0; "z+" / "x-" = unit axis (axis first: a leading "-" would parse as a flag); a trailing number scales it ("z+0.5")."""
    out = np.zeros((len(labels), 7), np.float32)
    per_step = offset_m / (cal * replan_steps * hold_chunks)
    for i, lab in enumerate(labels):
        if lab == "none":
            continue
        base, s = lab[:2], lab[2:]
        assert len(base) == 2 and base[0] in AXES and base[1] in "+-", f"bad offset label {lab!r}"
        out[i, AXES[base[0]]] = (1.0 if base[1] == "+" else -1.0) * float(s or 1.0) * per_step
    return out


def with_offset(actions, off):
    a = np.asarray(actions, np.float64) + off[None, :]
    a[:, :6] = np.clip(a[:, :6], -1.0, 1.0)
    return a


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


def query(client, obs, prompt, size, noise_scale=None, cand=None, info=None):
    """cand: DSRL probe request ("actor" / "gauss" / "base"); info (dict) receives the server's dsrl_w / dsrl_q."""
    img, wrist, state = preprocess(obs, size)
    element = {"observation/image": img, "observation/wrist_image": wrist, "observation/state": state,
               "prompt": prompt}
    if noise_scale is not None:
        element["noise_scale"] = float(noise_scale)
    if cand:
        element["dsrl_cand"] = cand
    out = client.infer(element)
    if info is not None:
        info["w"] = np.asarray(out.get("dsrl_w", np.full(32, np.nan)), np.float32)
        info["q"] = float(out.get("dsrl_q", np.nan))
    return np.asarray(out["actions"]), (img, wrist, state)


def run_policy(env, client, obs, prompt, t, max_steps, args):
    """Base policy from the current state (policy step t) to the end. Returns (success, final policy step)."""
    plan = collections.deque()
    done = False
    while t < max_steps:
        if not plan:
            chunk, _ = query(client, obs, prompt, args.resize_size, cand=args.continue_source or None)
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
    offs = (parse_offsets(args.cand_offsets, args.offset_m, args.offset_cal, args.replan_steps, args.hold_chunks)
            if args.cand_offsets else None)
    while t < max_steps and not done:
        snap = sim.save()
        cands, cw, cq = [], [], []
        if offs is not None:  # offset probe: one policy chunk, N offsets of it
            chunk, (img, wrist, state) = query(client, obs, prompt, args.resize_size)
            cands = [with_offset(chunk, o_c) for o_c in offs]
            cw, cq = [np.full(32, np.nan, np.float32)] * len(offs), [np.nan] * len(offs)
        for c in range(args.n_cand if offs is None else 0):
            ns = args.cand_noise_scale if (c > 0 and args.cand_noise_scale != 1.0) else None
            info = {}
            chunk, pre = query(client, obs, prompt, args.resize_size, ns, cand=args.cand_source or None, info=info)
            cands.append(chunk)
            cw.append(info["w"])
            cq.append(info["q"])
            if c == 0:
                img, wrist, state = pre
        cands = np.stack(cands)  # (N, H, 7)
        if args.cand_source:  # DSRL probe: the trunk executes its own ordinary (DSRL actor) chunk, not a candidate
            trunk, _ = query(client, obs, prompt, args.resize_size, cand=args.continue_source or None)
        elif offs is not None:  # offset probe: the plain chunk (= candidate 0 up to the env's own clipping)
            trunk = chunk
        else:
            trunk = cands[0]
        k = min(args.replan_steps, max_steps - t)
        eef, grip, cand_done = [], [], []
        is_branch = t in branch
        after_img, after_wrist, after_state, succ, steps = [], [], [], [], []
        seg_eef, seg_steps = [], []  # offset probe: end effector / policy step after the held segment
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
                seg_t = t + k
                if offs is not None:  # hold the offset for hold_chunks chunks, re-querying from the candidate's state
                    for _ in range(args.hold_chunks - 1):
                        if d or seg_t >= max_steps:
                            break
                        ch, _ = query(client, o, prompt, args.resize_size)
                        kk = min(args.replan_steps, max_steps - seg_t)
                        o, d = execute(env, with_offset(ch, offs[c])[:kk])
                        seg_t += kk
                    seg_eef.append(o["robot0_eef_pos"].copy())
                    seg_steps.append(seg_t)
                ai, aw, ast = preprocess(o, args.resize_size)
                after_img.append(ai)
                after_wrist.append(aw)
                after_state.append(ast)
                after_snap = sim.save()
                s_r, n_r = [], []
                for r in range(args.repeats):
                    if d:
                        s_r.append(True)
                        n_r.append(seg_t)
                        continue
                    o2 = sim.restore(after_snap)
                    s, tt = run_policy(env, client, o2, prompt, seg_t, max_steps, args)
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
            br["w"].append(np.stack(cw))
            br["q"].append(np.array(cq, np.float32))
            if offs is not None:
                br["seg_eef"].append(np.stack(seg_eef))
                br["seg_steps"].append(np.array(seg_steps))
            logging.info(f"{log_prefix} branch t={t}: success per candidate {np.array(succ).mean(1).round(2).tolist()}")
        rec["t"].append(t)
        rec["image"].append(img)
        rec["wrist_image"].append(wrist)
        rec["state"].append(state)
        rec["chunks"].append(cands)
        rec["cand_w"].append(np.stack(cw))
        rec["cand_q"].append(np.array(cq, np.float32))
        rec["eef_after"].append(np.stack(eef))
        rec["grip_after"].append(np.stack(grip))
        rec["cand_done"].append(np.array(cand_done))
        # continue the base episode with candidate 0 (DSRL probe: with the trunk's own chunk)
        obs = sim.restore(snap)
        obs, done = execute(env, trunk[:k])
        rec["executed"].append(trunk[:k])
        t += k
    out = {f"rp_{k_}": np.asarray(v) for k_, v in rec.items() if k_ != "executed"}
    out["executed"] = np.concatenate(rec["executed"])
    for k_, v in br.items():
        out[f"br_{k_}"] = np.asarray(v)
    out["restore_maxdiff"] = np.array(restore_maxdiff if restore_maxdiff is not None else np.nan)
    out["cand_noise_scale"] = np.array(args.cand_noise_scale)
    out["cand_source"] = np.array(args.cand_source)
    out["continue_source"] = np.array(args.continue_source)
    if offs is not None:
        out["cand_offsets"] = offs
        out["offset_labels"] = np.array(args.cand_offsets)
        out["offset_m"] = np.array(args.offset_m)
        out["hold_chunks"] = np.array(args.hold_chunks)
    return done, t, out


def main(args: Args) -> None:
    np.random.seed(args.seed)
    out = pathlib.Path(args.out_dir)
    (out / "probe").mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = args.max_steps_override or lr.MAX_STEPS[args.task_suite_name]
    if args.cand_offsets:
        assert args.cand_offsets[0] == "none", "offset probe: candidate 0 must be 'none' (the plain policy)"
        assert not args.cand_source and args.cand_noise_scale == 1.0, "offset probe runs alone (no DSRL / noise scale)"
        assert args.hold_chunks >= 1
        args = dataclasses.replace(args, n_cand=len(args.cand_offsets))
        logging.info("offset probe: %s, per-step offsets (x, y, z):\n%s", args.cand_offsets,
                     parse_offsets(args.cand_offsets, args.offset_m, args.offset_cal, args.replan_steps,
                                   args.hold_chunks)[:, :3].round(3))
    client = FakePolicy() if args.fake_policy else _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    fields = ["episode", "task_id", "init_state", "repeat", "success", "policy_steps", "n_branch", "restore_maxdiff",
              "seconds", "file"]
    # [2026-11-20] resume after a preemption / requeue: episodes already in episodes.csv with their npz are skipped
    done_eps = set()
    if (out / "episodes.csv").exists():
        with open(out / "episodes.csv") as fh:
            done_eps = {int(r["episode"]) for r in csv.DictReader(fh) if (out / "probe" / r["file"]).exists()}
        if done_eps:
            logging.info(f"resuming: {len(done_eps)} base episodes already done, skipped")
    with open(out / "episodes.csv", "a" if done_eps else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not done_eps:
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
                    if episode in done_eps:
                        episode += 1
                        continue
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
