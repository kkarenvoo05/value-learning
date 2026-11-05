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
trials. Policy sampling noise is not seeded per episode: the server draws it from jax.random.key(0), split on
every infer call, so it depends on call order within a server's lifetime.

Usage:
    python scripts/libero_rollouts.py --args.task-suite-name libero_10 --args.task-ids 3 \
        --args.repeats 2 --args.seed 8 --args.out-dir <run dir> --args.port <port>
"""

import collections
import csv
import dataclasses
import logging
import math
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


def goal_predicates(env):
    """Each conjunct of the task's BDDL goal, evaluated now (success = all of them)."""
    inner = env.env
    return np.array([bool(inner._eval_predicate(g)) for g in inner.parsed_problem["goal_state"]])


def object_pos_keys(obs):
    return sorted(k for k in obs if k.endswith("_pos") and not k.startswith("robot0"))


def run_episode(env, client, init_state, task_description, args, max_steps):
    """One episode, as main.py. Returns (success, policy_steps, error, frames, raw); raw is None unless save_raw."""
    obs = env.reset()
    action_plan = collections.deque()
    if init_state is None:  # fresh layout: keep the placement reset() sampled, record it for replay
        init_state = env.get_sim_state()
    else:
        obs = env.set_init_state(init_state)
    t, policy_steps, done, error, frames = 0, 0, False, "", []
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
                rec["goals"].append(goal_predicates(env))
                rec["object_pos"].append(np.concatenate([obs[k] for k in pos_keys]))
            if done:
                break
            t += 1
        except Exception as e:  # main.py logs and ends the episode as a failure; we also record it
            logging.error(f"Caught exception: {e}")
            error = repr(e)
            break
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

    csv_path = out / "episodes.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
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
                success, steps, error, frames, raw = run_episode(env, client, init, task_description,
                                                                 args, max_steps)
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
