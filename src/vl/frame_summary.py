"""Per-episode summaries of the trajectory parquet, read remotely (column subset only).

Nothing but the summary is written to disk: data/manifest/frame_summary/<repo>.parquet.
Usage:
    .venv/bin/python -m vl.frame_summary --config configs/dataset.yaml
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from vl.config import all_repos, load_config

BASE_COLS = ["episode_index", "frame_index", "timestamp", "is_valid", "done", "success",
             "reward", "intervention", "source", "pen_x", "pen_y", "pen_yaw",
             "holder_x", "holder_y", "observation.state"]
OPTIONAL_COLS = ["policy_id", "arm_id", "round_id", "manifest_idx"]


def _first(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else -1


def _is_suffix(mask: np.ndarray) -> bool:
    """True if mask is all-False, or of the form 0...0 1...1."""
    f = _first(mask)
    return f < 0 or bool(mask[f:].all())


def summarize_episode(df: pd.DataFrame, fps: int) -> dict:
    df = df.sort_values("frame_index")
    n = len(df)
    fi = df.frame_index.to_numpy()
    ts = df.timestamp.to_numpy(dtype=np.float64)
    valid = df.is_valid.to_numpy() == 1
    done = df.done.to_numpy() == 1
    succ = df.success.to_numpy() == 1
    human = df.source.to_numpy() == 1
    interv = df.intervention.to_numpy() == 1
    state = np.stack(df["observation.state"].to_numpy())
    # Content fingerprint: independent of any metadata IDs, used to find duplicated recordings.
    fp = hashlib.sha1(np.round(state[:5], 4).tobytes() + str(n).encode()).hexdigest()[:16]
    out = {
        "episode_index": int(df.episode_index.iloc[0]),
        "n_rows": n,
        "frame_index_contiguous": bool((fi == np.arange(n)).all()),
        "ts_first": float(ts[0]),
        "ts_last": float(ts[-1]),
        "ts_max_abs_err_s": float(np.abs(ts - fi / fps).max()),
        "n_valid": int(valid.sum()),
        "valid_is_prefix": _is_suffix(~valid),
        "first_invalid": _first(~valid),
        "n_done": int(done.sum()),
        "first_done": _first(done),
        "done_is_suffix": _is_suffix(done),
        "success_flag_any": bool(succ.any()),
        "first_success_flag": _first(succ),
        "first_reward_pos": _first(df.reward.to_numpy() > 0),
        "n_intervention_onsets": int(interv.sum()),
        "n_human_frames": int(human.sum()),
        "first_human_frame": _first(human),
        "content_fingerprint": fp,
    }
    for k in ["pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y"]:
        v = df[k].to_numpy(dtype=np.float64)
        out[f"init_{k}"] = float(v[0])
        out[f"{k}_constant"] = bool(np.allclose(v, v[0]))
    for k in OPTIONAL_COLS:
        if k in df:
            u = sorted(set(int(x) for x in df[k]))
            out[f"frame_{k}"] = u[0] if len(u) == 1 else json.dumps(u)
    return out


def summarize_repo(fs: HfFileSystem, repo_id: str, sha: str, meta_dir: Path, fps: int) -> pd.DataFrame:
    eps = pd.read_parquet(next(meta_dir.glob("meta/episodes/*/*.parquet")))
    files = sorted(set(zip(eps["data/chunk_index"], eps["data/file_index"])))
    schema_cols = None

    def read(cf):
        nonlocal schema_cols
        path = f"datasets/{repo_id}@{sha}/data/chunk-{cf[0]:03d}/file-{cf[1]:03d}.parquet"
        with fs.open(path, "rb", block_size=2**20) as f:
            pf = pq.ParquetFile(f)
            names = set(pf.schema_arrow.names)
            cols = BASE_COLS + [c for c in OPTIONAL_COLS if c in names]
            return pf.read(columns=cols).to_pandas()

    with ThreadPoolExecutor(8) as ex:
        frames = list(ex.map(read, files))
    rows = []
    for df in frames:
        for _, g in df.groupby("episode_index"):
            rows.append(summarize_episode(g, fps))
    out = pd.DataFrame(rows)
    if out.episode_index.duplicated().any():
        raise RuntimeError(f"{repo_id}: an episode is split across data files; summarize after concat")
    return out.sort_values("episode_index").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta_root = Path(cfg["paths"]["meta_root"])
    pins = json.loads((meta_root / "pins.json").read_text())
    out_dir = Path(cfg["paths"]["manifest"]).parent / "frame_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for name in all_repos(cfg, include_audit_extra=True):
        out = out_dir / f"{name}.parquet"
        if out.exists() and not args.force:
            continue
        fps = json.loads((meta_root / name / "meta/info.json").read_text())["fps"]
        s = summarize_repo(fs, pins[name]["repo_id"], pins[name]["sha"], meta_root / name, fps)
        s.to_parquet(out)
        print(f"{name:45s} {len(s):4d} episodes summarized")


if __name__ == "__main__":
    main()
