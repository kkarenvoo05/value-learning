"""Continuous-layout eval set [2026-11-16; mentor: "we can try sampling continuous values in the ranges, instead of from
the discrete set of 50 predefined start states"; Karen: "build the continuous eval set"].

Draws N moka layouts once from the task's BDDL placement regions (env.reset(), as --args.fresh-layouts does and as the
50 saved init states were drawn), saves each as a step-0 "source" npz, and writes a starts CSV for
scripts/libero_rollouts.py --args.start-from with R repeats per layout. Every checkpoint is then evaluated on the same
N layouts. A source npz holds what the start-from path reads: sim_init, actions (0, 7) (no prefix: k = 0),
state (1, 8) = the client's 8-d state after the wait steps, init_state_id = -1; plus obj_names / obj_pos (object
positions after the wait steps) for the success-vs-position analysis.
The start-from path does reset() -> set_init_state(sim_init) -> wait steps -> (no prefix) -> policy, so this script
builds state[0] the same way, and checks that a second set_init_state of the same sim_init lands on the same state.
Env seed EVAL_SEED [ours] differs from every collection seed (Stage 2: 9, rounds: 200+), so these draws are new.

Run (LIBERO env, one GPU for EGL rendering):
    LIBERO_CONFIG_PATH=$ROOT/libero_cfg MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero \
    $ROOT/.venv/bin/python scripts/libero_eval_layouts.py --n 100 --repeats 3 --out $ROOT/eval_sets/moka_cont100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_rollouts import (LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION, _get_libero_env,  # noqa: E402
                             _quat2axisangle, benchmark)

EVAL_SEED = 9001


def settle(env, sim_init, n_wait):
    env.reset()
    obs = env.set_init_state(sim_init)
    for _ in range(n_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return obs, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=8)
    ap.add_argument("--num-steps-wait", type=int, default=10)  # = libero_rollouts.Args.num_steps_wait
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "sources").mkdir(parents=True, exist_ok=True)
    task = benchmark.get_benchmark_dict()[a.suite]().get_task(a.task_id)
    env, desc = _get_libero_env(task, LIBERO_ENV_RESOLUTION, a.seed)
    rows, maxdiff, pos = [], 0.0, []
    for j in range(a.n):
        env.reset()
        sim_init = np.asarray(env.get_sim_state())
        obs, state = settle(env, sim_init, a.num_steps_wait)
        _, state2 = settle(env, sim_init, a.num_steps_wait)
        maxdiff = max(maxdiff, float(np.abs(state - state2).max()))
        names = sorted(k for k in obs if k.endswith("_pos") and not k.startswith("robot0") and "_to_" not in k)
        obj = np.stack([np.asarray(obs[k], float) for k in names])
        pos.append(obj)
        src = out / "sources" / f"layout{j:03d}.npz"
        np.savez_compressed(src, sim_init=sim_init, actions=np.zeros((0, 7), np.float32), state=state[None],
                            init_state_id=-1, layout_id=j, obj_names=np.array(names), obj_pos=obj, env_seed=a.seed)
        for r in range(a.repeats):
            rows.append({"start_id": j * a.repeats + r, "source_raw": str(src), "start_step": 0, "init_state": -1,
                         "env_seed": a.seed, "layout_id": j})
    env.close()
    with open(out / "starts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pos = np.stack(pos)
    d = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1)).max(-1)  # largest object displacement between layouts
    np.fill_diagonal(d, np.inf)
    info = {"task": desc, "suite": a.suite, "task_id": a.task_id, "env_seed": a.seed, "layouts": a.n,
            "repeats": a.repeats, "episodes": len(rows), "num_steps_wait": a.num_steps_wait,
            "resettle_max_state_diff": maxdiff, "objects": names,
            "obj_pos_min": pos.min(0).round(4).tolist(), "obj_pos_max": pos.max(0).round(4).tolist(),
            "nearest_layout_m": {"min": float(d.min(1).min()), "median": float(np.median(d.min(1)))}}
    (out / "eval_set.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    assert maxdiff < 1e-6, f"set_init_state is not reproducible: max |state diff| {maxdiff}"


if __name__ == "__main__":
    main()
