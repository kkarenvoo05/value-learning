"""LIBERO rollouts of a pi05 policy server (docs/libero_plan.md Stage 1+), one video and one CSV row per episode.

Runs in the LIBERO client env (Python 3.8), against an openpi policy server; see scripts/libero_eval.sbatch.
The episode loop is a copy of openpi's examples/libero/main.py (215abfb), kept behaviour-identical (wait steps,
180-degree image rotation, resize-with-pad to 224, replan every 5 steps, max_steps per suite). Changes [ours]:
  1. --args.task-ids: run only these tasks of the suite (default: all)
  2. one video per episode, named by episode index, init state and outcome (main.py overwrote repeats)
  3. episodes.csv: one row per episode (task, init state, repeat, env seed, success, policy steps, error)
  4. (Stage 2) --args.save-raw: one raw/epNNNN.npz per episode with everything the policy saw and did, for the
     LeRobot conversion (scripts/libero_to_lerobot.py) and the replay check (scripts/libero_replay.py):
     image / wrist_image (T, 224, 224, 3) uint8 exactly as sent to the server, state (T, 8), executed actions
     (T, 7) (both in their original dtype), new_chunk (T,) whether step t started a new action chunk, goal predicates (T, G) after each step,
     object positions (T, 3 * n_obj) after each step, MuJoCo states: init (as stored by LIBERO), start (after
     the wait steps, before the first policy action) and end. Row t pairs the observation with the action
     taken from it. --args.split-name labels the rows (train / val)
Episode order: for each repeat r, for each init state i -> init state i, so every init state gets `repeats`
trials. Policy sampling noise is not seeded per episode by default: the server draws it from jax.random.key(0),
split on every infer call, so it depends on call order within a server's lifetime.
  5. --args.matched-noise [ours, 2026-11-06]: every request carries noise_key = [task_id, init_state, repeat, chunk
     index within the episode]; the server (scripts/serve_pi05_finetuned.py, not openpi's serve_policy.py) draws the
     start noise from that key (vl.libero_noise), so the same episode gets the same noise in every eval.
  6. --args.explore-prob p --args.explore-scale s [ours, 2026-11-08, data-variety pilot]: each new action chunk is,
     with probability p, requested with "noise_scale" = s (the server draws the flow-matching start noise as
     s * N(0, I); scripts/serve_pi05_finetuned.py, also for --base), otherwise as usual. The choice is drawn from
     an rng seeded by (explore_seed, task_id, init_state, repeat), so it is reproducible and a resumed episode
     gets the same choices. raw npz gains `noisy` (T,) = whether step t's chunk was a scaled one, and the run
     dir gets explore.json. Not with matched noise.

Usage:
    python scripts/libero_rollouts.py --args.task-suite-name libero_10 --args.task-ids 3 \
        --args.repeats 2 --args.seed 8 --args.out-dir <run dir> --args.port <port>
"""

import collections
import csv
import dataclasses
import logging
import math
import os
import pathlib
from typing import List, Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
CSV_FIELDS = ["episode", "suite", "task_id", "task", "split", "init_state", "repeat", "env_seed", "success",
              "timeout", "policy_steps", "max_steps", "goals_end", "error", "video", "raw"]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_10"
    task_ids: Optional[List[int]] = None  # default: all tasks of the suite
    init_states: Optional[List[int]] = None  # default: all init states of each task (50)
    repeats: int = 1  # trials per init state
    num_steps_wait: int = 10  # do-nothing steps while objects settle
    out_dir: str = "data/libero/run"
    seed: int = 7  # env seed (as main.py); not the policy's noise
    save_raw: bool = False  # write raw/epNNNN.npz per episode (Stage 2)
    split_name: str = ""  # written to the CSV split column
    # > 0: this many episodes per task, each from a fresh random layout (env.reset() samples the task's BDDL placement
    # regions, as the 50 saved init states were drawn) instead of the saved init states; init_states / repeats are
    # ignored, the CSV init_state is -1, and raw sim_init holds the sampled state (replay: set_init_state(sim_init))
    fresh_layouts: int = 0
    # send noise_key per request so the server draws matched (common-random-numbers) noise; needs our server
    matched_noise: bool = False
    # [2026-11-07] preemption-safe: keep the complete rows of an existing episodes.csv in out_dir and run only the
    # missing (task, init state, repeat) episodes, appending. Episode numbers stay the plan position. Not with
    # fresh_layouts (their layouts depend on the env RNG's history). Reruns are not bit-identical (GPU / render
    # nondeterminism: the same matched-noise episode took 383 vs 460 steps in two attempts of 18355679), so a
    # resumed eval is a valid draw of every episode, not a replay of the interrupted attempt.
    resume: bool = False
    # [2026-11-07] send obs["subtask"] = 0 / 1 (no moka pot on the stove yet / at least one, from the BDDL goal
    # predicates now, i.e. the state the image shows) for runs trained with --subtask-prompt (serve_pi05_finetuned)
    send_subtask: bool = False
    # [2026-11-08] exploration: share of chunks sampled with start noise x explore_scale (0 = off)
    explore_prob: float = 0.0
    explore_scale: float = 1.0
    explore_seed: int = 0
    # [2026-11-10] DSRL (scripts/dsrl_pi05.py): after each episode send {"dsrl_end": 1, "success": 0/1, final
    # images/state} so the server can store the episode and update its SAC agent (train mode; eval ignores it)
    dsrl: bool = False


def goal_predicates(env):
    """Each conjunct of the task's BDDL goal, evaluated now (success = all of them)."""
    inner = env.env
    return np.array([bool(inner._eval_predicate(g)) for g in inner.parsed_problem["goal_state"]])


def object_pos_keys(obs):
    return sorted(k for k in obs if k.endswith("_pos") and not k.startswith("robot0"))


def run_episode(env, client, init_state, task_description, args, max_steps, noise_prefix=None, explore_rng=None):
    """One episode, as main.py. Returns (success, policy_steps, error, frames, raw); raw is None unless save_raw.
    explore_rng: with args.explore_prob > 0, draws which chunks are requested with noise_scale = args.explore_scale."""
    obs = env.reset()
    action_plan = collections.deque()
    if init_state is None:  # fresh layout: keep the placement reset() sampled, record it for replay
        init_state = env.get_sim_state()
    else:
        obs = env.set_init_state(init_state)
    t, policy_steps, done, error, frames, chunk = 0, 0, False, "", [], 0
    noisy = False
    rec = collections.defaultdict(list)
    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                if t == args.num_steps_wait:
                    sim_start, pos_keys = env.get_sim_state(), object_pos_keys(obs)
                continue
            # rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
            frames.append(img)
            state = np.concatenate(
                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            new_chunk = not action_plan
            if new_chunk:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": str(task_description),
                }
                if args.send_subtask:
                    element["subtask"] = int(min(int(goal_predicates(env)[:2].sum()), 1))
                if noise_prefix is not None:  # matched noise: [task_id, init_state, repeat, chunk]
                    element["noise_key"] = np.array([*noise_prefix, chunk], dtype=np.int64)
                noisy = explore_rng is not None and bool(explore_rng.random() < args.explore_prob)
                if noisy:
                    element["noise_scale"] = float(args.explore_scale)
                chunk += 1
                action_chunk = client.infer(element)["actions"]
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])
            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            policy_steps += 1
            if args.save_raw:
                rec["image"].append(img)
                rec["wrist_image"].append(wrist_img)
                rec["state"].append(state)
                rec["actions"].append(action)
                rec["new_chunk"].append(new_chunk)
                if explore_rng is not None:
                    rec["noisy"].append(noisy)
                rec["goals"].append(goal_predicates(env))
                rec["object_pos"].append(np.concatenate([obs[k] for k in pos_keys]))
            if done:
                break
            t += 1
        except Exception as e:  # main.py logs and ends the episode as a failure; we also record it
            logging.error(f"Caught exception: {e}")
            error = repr(e)
            break
    if args.dsrl and policy_steps > 0:  # final observation: the timeout bootstrap target; success = env done
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        state = np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
        client.infer({"observation/image": img, "observation/state": state, "dsrl_end": 1,
                      "success": int(bool(done) and not error)})
    raw = None
    if args.save_raw and policy_steps > 0:
        raw = {k: np.asarray(v) for k, v in rec.items()}
        # state and actions keep their own dtype (float64): float32 actions replay with ~1e-7 drift
        raw.update(sim_init=np.asarray(init_state), sim_start=sim_start, sim_end=env.get_sim_state(),
                   object_pos_keys=np.array(pos_keys), prompt=np.array(str(task_description)))
    return bool(done), policy_steps, error, frames, raw


def main(args: Args) -> None:
    np.random.seed(args.seed)
    out = pathlib.Path(args.out_dir)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        (out / "raw").mkdir(exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]
    task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"matched noise: {args.matched_noise}")
    explore = args.explore_prob > 0
    if explore:
        assert not args.matched_noise, "exploration noise and matched noise together are not supported"
        assert 0 < args.explore_prob <= 1 and args.explore_scale > 0
        import json
        (out / "explore.json").write_text(json.dumps({"prob": args.explore_prob, "scale": args.explore_scale,
                                                      "seed": args.explore_seed}))
        logging.info(f"exploration: p {args.explore_prob} scale {args.explore_scale} seed {args.explore_seed}")

    csv_path = out / "episodes.csv"
    # a requeued SLURM job (preempted) resumes even when its stored batch script predates --args.resume
    resume = args.resume or os.environ.get("SLURM_RESTART_COUNT", "0") not in ("", "0")
    done_rows = load_done(csv_path) if resume and not args.fresh_layouts else {}
    if resume and args.fresh_layouts:
        logging.warning("--args.resume ignored with fresh layouts: starting over")
    if done_rows:
        logging.info(f"resume: {len(done_rows)} complete episodes kept from {csv_path}")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sorted(done_rows.values(), key=lambda r: int(r["episode"])):
            writer.writerow(row)
        fh.flush()
        episode = 0
        for task_id in task_ids:
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            ids = args.init_states if args.init_states is not None else list(range(len(init_states)))
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            n_ok = n = 0
            plan = ([(r, i, init_states[i]) for r in range(args.repeats) for i in ids] if not args.fresh_layouts
                    else [(r, -1, None) for r in range(args.fresh_layouts)])
            for repeat, i, init in plan:
                prev = done_rows.get((task_id, i, repeat)) if i >= 0 else None
                if prev is not None:  # resumed: already complete in an earlier attempt of this job
                    n += 1
                    n_ok += int(prev["success"])
                    episode += 1
                    continue
                success, steps, error, frames, raw = run_episode(
                    env, client, init, task_description, args, max_steps,
                    noise_prefix=(task_id, i, repeat) if args.matched_noise else None,
                    explore_rng=(np.random.default_rng([args.explore_seed, task_id, i + 1, repeat, episode])
                                 if explore else None))
                video = f"ep{episode:04d}_task{task_id}_init{'fresh' if i < 0 else f'{i:02d}'}_r{repeat}_{'success' if success else 'failure'}.mp4"
                imageio.mimwrite(out / "videos" / video, [np.asarray(x) for x in frames], fps=10)
                raw_name = ""
                if raw is not None:
                    raw_name = f"ep{episode:04d}.npz"
                    np.savez_compressed(out / "raw" / raw_name, init_state_id=i, **raw)
                goals_end = "".join(str(int(g)) for g in raw["goals"][-1]) if raw is not None else ""
                writer.writerow({"episode": episode, "suite": args.task_suite_name, "task_id": task_id,
                                 "task": task_description, "split": args.split_name, "init_state": i,
                                 "repeat": repeat, "env_seed": args.seed, "success": int(success),
                                 "timeout": int(not success and not error and steps == max_steps),
                                 "policy_steps": steps, "max_steps": max_steps, "goals_end": goals_end,
                                 "error": error, "video": video, "raw": raw_name})
                fh.flush()
                n += 1
                n_ok += success
                logging.info(f"ep {episode} task {task_id} init {i} r{repeat}: success={success} "
                             f"steps={steps} | task {n_ok}/{n}")
                episode += 1
            logging.info(f"Task {task_id} ({task_description}): {n_ok}/{n} = {n_ok / n:.3f}")
            env.close()
    logging.info(f"Done: {episode} episodes -> {csv_path}")


def load_done(csv_path):
    """{(task_id, init_state, repeat): row} of the complete rows of an earlier attempt's episodes.csv (a row cut off
    by preemption mid-write, or one missing a field, is dropped and its episode rerun)."""
    if not pathlib.Path(csv_path).exists():
        return {}
    done = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if None in row or any(row.get(k) is None for k in CSV_FIELDS) or row["success"] not in ("0", "1"):
                continue
            if row["raw"] and not row["raw"].endswith(".npz"):
                continue
            done[(int(row["task_id"]), int(row["init_state"]), int(row["repeat"]))] = row
    return done


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # main.py: seed affects object positions even with a fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from openpi examples/libero/main.py (itself from robosuite)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
