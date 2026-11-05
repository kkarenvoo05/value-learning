"""Stage 2 (docs/libero_plan.md): raw LIBERO rollouts -> one LeRobot dataset, plus an episode table.

Runs in the openpi env (lerobot 0.1.0), like openpi's examples/libero/convert_libero_data_to_lerobot.py, and
keeps its feature names and fps so openpi can fine-tune on the result (Stage 6): image, wrist_image, state (8),
actions (7), task, fps 10. Differences [ours]: images are the 224 x 224 frames exactly as sent to the policy
(rotated, resized with pad), not the 256 x 256 renders, and failures are kept.

Writes <root>/<repo_id>/ (LeRobot) and <root>/<repo_id>/episodes.csv: one row per LeRobot episode_index with
the run, split, init state, repeat, source episode, policy steps, success, timeout and final goal predicates.
Episodes that ended in a client error (no raw file or error set) are skipped and listed in the log.

Usage (openpi env):
    .venv/bin/python <value-learning>/scripts/libero_to_lerobot.py \
        --run-dir /iris/u/ankile/karenvo/libero/runs/stage2_moka_<job> --root /iris/u/ankile/karenvo/libero/lerobot
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}
FIELDS = ["episode_index", "run", "split", "init_state", "repeat", "source_episode", "policy_steps", "success",
          "timeout", "goals_end", "raw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Stage 2 run with train/ and val/ subdirectories")
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="libero_moka_stage2")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = Path(args.root) / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild it")
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=args.repo_id, root=out, robot_type="panda", fps=10, features=FEATURES,
                               image_writer_threads=10, image_writer_processes=5)
    rows, skipped = [], []
    for split in ("train", "val"):
        csv_path = run / split / "episodes.csv"
        if not csv_path.exists():
            continue
        for r in csv.DictReader(open(csv_path)):
            if r["error"] or not r["raw"]:
                skipped.append((split, r["episode"], r["error"]))
                continue
            with np.load(run / split / "raw" / r["raw"]) as z:   # read once: NpzFile decompresses on every access
                raw = {k: z[k] for k in z.files}
            T = len(raw["actions"])
            assert T == int(r["policy_steps"]) and len(raw["image"]) == T and len(raw["state"]) == T
            for t in range(T):
                ds.add_frame({"image": raw["image"][t], "wrist_image": raw["wrist_image"][t],
                              "state": raw["state"][t].astype(np.float32),
                              "actions": raw["actions"][t].astype(np.float32), "task": str(raw["prompt"])})
            ds.save_episode()
            rows.append({"episode_index": len(rows), "run": run.name, "split": split,
                         "init_state": r["init_state"], "repeat": r["repeat"], "source_episode": r["episode"],
                         "policy_steps": T, "success": r["success"], "timeout": r["timeout"],
                         "goals_end": r["goals_end"], "raw": f"{split}/raw/{r['raw']}"})
            print(f"{split} ep {r['episode']} -> {rows[-1]['episode_index']} ({T} frames)", flush=True)
    with open(out / "episodes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} episodes; skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
