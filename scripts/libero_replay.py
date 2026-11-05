"""Stage 2 check (docs/libero_plan.md): replay saved episodes open-loop and compare with what was recorded.

For each raw/epNNNN.npz written by scripts/libero_rollouts.py --args.save-raw: a fresh env (same task, same
env seed), set the stored init state, the same wait steps, then step the *saved executed actions* (no policy).
Checks per episode:
  start   MuJoCo state after the wait steps equals the recorded one (max |diff|)
  frames  agentview and wrist images re-rendered at every step t, preprocessed as the client does, equal the
          stored image[t] / wrist_image[t] (counts frames that differ: tests image / frame-index sync)
  state   the 8-d state at every t equals the stored state[t]
  goals   goal predicates after every step equal the stored ones; final success equals the recorded outcome
  end     final MuJoCo state equals the recorded one (max |diff|)
Runs in the LIBERO client env (Python 3.8), no policy server needed. Writes replay.csv next to episodes.csv.

Usage:
    python scripts/libero_replay.py --args.run-dir <run dir> [--args.episodes 0 5 9]
"""

import csv
import dataclasses
import logging
import pathlib
from typing import List, Optional

from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
import tyro

import libero_rollouts as lr


@dataclasses.dataclass
class Args:
    run_dir: str
    episodes: Optional[List[int]] = None  # default: every episode with a raw file
    resize_size: int = 224
    num_steps_wait: int = 10


def replay(env, raw, args):
    env.reset()
    obs = env.set_init_state(raw["sim_init"])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(lr.LIBERO_DUMMY_ACTION)
    res = {"start_maxdiff": float(np.abs(env.get_sim_state() - raw["sim_start"]).max())}
    bad_img = bad_wrist = bad_state = bad_goals = 0
    state_maxdiff, px_maxdiff = 0.0, 0
    done = False
    for t, action in enumerate(raw["actions"]):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
        state = np.concatenate(
            (obs["robot0_eef_pos"], lr._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
        bad_img += not np.array_equal(img, raw["image"][t])
        bad_wrist += not np.array_equal(wrist, raw["wrist_image"][t])
        px_maxdiff = max(px_maxdiff, int(np.abs(img.astype(int) - raw["image"][t]).max()),
                         int(np.abs(wrist.astype(int) - raw["wrist_image"][t]).max()))
        d = float(np.abs(state - raw["state"][t]).max())
        state_maxdiff = max(state_maxdiff, d)
        bad_state += d > 0
        obs, _, done, _ = env.step(action.tolist())
        bad_goals += not np.array_equal(lr.goal_predicates(env), raw["goals"][t])
    res.update(frames=len(raw["actions"]), image_mismatch=bad_img, wrist_mismatch=bad_wrist, pixel_maxdiff=px_maxdiff,
               state_mismatch=bad_state, state_maxdiff=state_maxdiff, goal_mismatch=bad_goals,
               replay_success=int(bool(done)),
               end_maxdiff=float(np.abs(env.get_sim_state() - raw["sim_end"]).max()))
    return res


def main(args: Args) -> None:
    run = pathlib.Path(args.run_dir)
    with open(run / "episodes.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["raw"]]
    if args.episodes is not None:
        rows = [r for r in rows if int(r["episode"]) in set(args.episodes)]
    suite = benchmark.get_benchmark_dict()[rows[0]["suite"]]()
    task = suite.get_task(int(rows[0]["task_id"]))
    env, _ = lr._get_libero_env(task, lr.LIBERO_ENV_RESOLUTION, int(rows[0]["env_seed"]))
    fields = ["episode", "success", "policy_steps"]
    out_rows = []
    for r in rows:
        with np.load(run / "raw" / r["raw"]) as z:   # read once: NpzFile decompresses on every access
            raw = {k: z[k] for k in z.files}
        res = replay(env, raw, args)
        out_rows.append({"episode": r["episode"], "success": r["success"], "policy_steps": r["policy_steps"], **res})
        logging.info(f"ep {r['episode']}: {res}")
    env.close()
    with open(run / "replay.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + [k for k in out_rows[0] if k not in fields])
        w.writeheader()
        w.writerows(out_rows)
    # Sim, state, goals and outcome must match exactly; images may differ by 1 grey level on a few frames
    # (renderer noise: seen with bit-identical MuJoCo states in the smoke run, job 18155725)
    n_sim = sum(int(o["replay_success"]) == int(o["success"]) and o["start_maxdiff"] == 0 and o["end_maxdiff"] == 0
                and o["state_mismatch"] == 0 and o["goal_mismatch"] == 0 for o in out_rows)
    n_px = sum(o["pixel_maxdiff"] <= 1 for o in out_rows)
    n_exact = sum(o["image_mismatch"] == 0 and o["wrist_mismatch"] == 0 for o in out_rows)
    logging.info(f"{n_sim}/{len(out_rows)} episodes replay exactly (sim state, robot state, goals, outcome); images: "
                 f"{n_exact} identical, {n_px} within 1 grey level -> {run / 'replay.csv'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
