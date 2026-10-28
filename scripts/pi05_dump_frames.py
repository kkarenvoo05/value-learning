"""Dump a few val frames in DROID's input layout for the pi05_droid smoke test (scripts/pi05_smoke.py).

Usage:
    .venv/bin/python scripts/pi05_dump_frames.py [--n 64] [--out /iris/u/ankile/karenvo/pi05/frames_val64.npz]

Runs in this repo's venv (decoding); the smoke test runs in openpi's own env and only reads the npz.
Val split only, random frames (seed 0). Per frame: side_1 and wrist_left at full 480x640 (openpi resizes
with padding itself), joint_position[7] and gripper_position[1] from the episode's data parquet, plus the
labels (value, outcome, T) so token probes can be checked against targets later. side_1 stands in for
DROID's exterior camera; the lab's pi05_droid server feeds side_2 (not downloaded here) [ours].
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vl.frames import FrameReader

VIDEOS = Path("data/videos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="/iris/u/ankile/karenvo/pi05/frames_val64.npz")
    a = ap.parse_args()
    fi = pd.read_parquet("data/frames/frame_index.parquet")
    rows = fi[fi.split == "val"].sample(a.n, random_state=0).reset_index(drop=True)
    reader = FrameReader()
    side, wrist, joints, grip = [], [], [], []
    states = {}
    for r in rows.itertuples():
        side.append(reader.read(r.path_side_1, r.from_ts_side_1, r.frame_index, r.fps))
        wrist.append(reader.read(r.path_wrist_left, r.from_ts_wrist_left, r.frame_index, r.fps))
        if r.repo not in states:
            cols = ["episode_index", "frame_index", "observation.state.joint_position",
                    "observation.state.gripper_position"]
            states[r.repo] = pd.concat(pd.read_parquet(f, columns=cols)
                                       for f in sorted((VIDEOS / r.repo / "data").glob("*/*.parquet")))
        st = states[r.repo]
        s = st[(st.episode_index == r.episode_index) & (st.frame_index == r.frame_index)]
        assert len(s) == 1, (r.repo, r.episode_index, r.frame_index)
        joints.append(np.asarray(s.iloc[0]["observation.state.joint_position"], np.float32))
        grip.append(np.asarray(s.iloc[0]["observation.state.gripper_position"], np.float32))
    reader.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, side=np.stack(side), wrist=np.stack(wrist), joint_position=np.stack(joints),
                        gripper_position=np.stack(grip), value=rows.value.to_numpy(float),
                        outcome=rows.outcome.to_numpy(str), T=rows["T"].to_numpy(int),
                        frame_index=rows.frame_index.to_numpy(int), episode_uid=rows.episode_uid.to_numpy(str))
    print(f"{a.out}: {len(rows)} frames, side {side[0].shape}, joints range "
          f"[{np.min(joints):.2f}, {np.max(joints):.2f}], gripper [{np.min(grip):.2f}, {np.max(grip):.2f}]")


if __name__ == "__main__":
    main()
