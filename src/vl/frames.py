"""Frame index (one row per training example) and exact frame decoding from local videos.

Usage:
    .venv/bin/python -m vl.frames                # -> data/frames/frame_index.parquet

One training example is one valid frame t <= T of a train/val episode. Its row says where both
camera images live (mp4 path + the episode's start time inside that file) and what its label is.
The label columns (value, bin, outcome, T, ...) are for building the loss and for breakdowns
only; `vl.dataset` never passes them to the model.

Videos (LeRobot v3): an mp4 holds one episode (r02-r05) or many back to back (c00-c05, r00,
r01). Episode e's frame t of camera c is the frame at time from_timestamp[e, c] + t / fps in
that file. `FrameReader` decodes it by presentation timestamp and checks it landed on that exact
frame (`vl.preview.frame_at` once returned t+1 because of an off-by-one; this does not guess).
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
import pandas as pd

from vl.config import load_config

SPLITS = ("train", "val")   # test frames are never indexed at this stage


def video_locations(meta_root: Path, video_root: Path, repo: str, cams: list[str]) -> pd.DataFrame:
    """Per episode and camera: local mp4 path and the episode's start time inside it."""
    eps = pd.concat(pd.read_parquet(f) for f in sorted((meta_root / repo / "meta/episodes").glob("*/*.parquet")))
    rows = []
    for cam in cams:
        k = f"videos/observation.images.{cam}"
        sub = eps[["episode_index", "length", f"{k}/chunk_index", f"{k}/file_index",
                   f"{k}/from_timestamp", f"{k}/to_timestamp"]]
        sub.columns = ["episode_index", "length", "chunk", "file", "from_ts", "to_ts"]
        sub = sub.assign(repo=repo, cam=cam)
        sub["path"] = [str(video_root / repo / f"videos/observation.images.{cam}"
                           / f"chunk-{c:03d}/file-{f:03d}.mp4") for c, f in zip(sub.chunk, sub.file)]
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def build_frame_index(manifest: pd.DataFrame, targets: pd.DataFrame, locs: pd.DataFrame,
                      cams: list[str]) -> pd.DataFrame:
    """Every in-support valid train/val frame, with both cameras' video locations."""
    tg = targets[targets.split.isin(SPLITS) & targets.in_support]
    ep = manifest[["episode_uid", "repo", "episode_index", "T", "fps", "source_type", "policy_key",
                   "has_human_frames", "soft_truncate"]]
    df = tg.merge(ep, on=["episode_uid", "repo", "episode_index"], how="left", validate="m:1")
    for cam in cams:
        loc = locs[locs.cam == cam][["repo", "episode_index", "path", "from_ts", "to_ts"]]
        loc = loc.rename(columns={"path": f"path_{cam}", "from_ts": f"from_ts_{cam}", "to_ts": f"to_ts_{cam}"})
        df = df.merge(loc, on=["repo", "episode_index"], how="left", validate="m:1")
    assert (df.frame_index <= df["T"]).all(), "frame after the valid endpoint"
    for cam in cams:
        assert df[f"path_{cam}"].notna().all(), f"episode without a {cam} video"
        # Frame t must lie inside the episode's own span of the file, not in the next episode's.
        start = df[f"from_ts_{cam}"] + df.frame_index / df.fps
        assert (start < df[f"to_ts_{cam}"] - 0.5 / df.fps).all(), f"{cam}: frame past its episode span"
    return df.reset_index(drop=True)


class FrameReader:
    """Decode frame t of an episode from a local mp4, by presentation timestamp.

    Keeps a few containers open (per process; create one per DataLoader worker). These AV1
    files have a keyframe every 2 frames, so a seek decodes at most ~2 extra frames.
    """

    def __init__(self, max_open: int = 16):
        self.max_open = max_open
        self._open: OrderedDict[str, av.container.InputContainer] = OrderedDict()

    def _container(self, path: str):
        c = self._open.pop(path, None)
        if c is None:
            c = av.open(path)
            c.streams.video[0].thread_count = 1
            while len(self._open) >= self.max_open:
                self._open.popitem(last=False)[1].close()
        self._open[path] = c
        return c

    def read(self, path: str, from_ts: float, t: int, fps: float) -> np.ndarray:
        """RGB uint8 array (H, W, 3) of the frame at from_ts + t / fps."""
        c = self._container(path)
        s = c.streams.video[0]
        tb = float(s.time_base)
        want = round((from_ts + t / fps) / tb)            # target pts in stream ticks
        half = 0.5 / fps / tb                             # half a frame, in ticks
        c.seek(want, stream=s, backward=True)
        for fr in c.decode(s):
            if fr.pts >= want - half:
                if abs(fr.pts - want) > half:
                    raise RuntimeError(f"{path}: wanted pts {want}, got {fr.pts} (frame missing?)")
                return fr.to_ndarray(format="rgb24")
        raise RuntimeError(f"{path}: no frame at pts {want} (past end of file)")

    def close(self):
        for c in self._open.values():
            c.close()
        self._open.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = load_config(cfg["dataset_config"])
    cams = cfg["cameras"]
    manifest = pd.read_parquet(ds["paths"]["manifest"])
    targets = pd.read_parquet(cfg["paths"]["frame_targets"])
    locs = pd.concat(video_locations(Path(ds["paths"]["meta_root"]), Path(cfg["paths"]["video_root"]),
                                     r, cams) for r in ds["train_pool"])
    df = build_frame_index(manifest, targets, locs, cams)
    out = Path(cfg["paths"]["frame_index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df):,} frames from {df.episode_uid.nunique():,} episodes -> {out}")
    print(df.groupby(["split", "outcome"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
