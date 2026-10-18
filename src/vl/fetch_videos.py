"""Download pinned camera videos for the configured repos.

Usage:
    .venv/bin/python -m vl.fetch_videos --cams wrist_left side_1          # training pool
    .venv/bin/python -m vl.fetch_videos --cams wrist_left --repos real-marker-d2-r03-eval
    .venv/bin/python -m vl.fetch_videos --cams side_1 --data      # also the trajectory parquet

Files land in data/videos/<repo>/videos/observation.images.<cam>/chunk-XXX/file-XXX.mp4, at the
commit recorded in data/meta/pins.json (download by SHA, never by moving ref).
Check `df -h` on the data volume first: the training-pool wrist_left + side_1 is ~12 GB.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from vl.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--repos", nargs="+", help="default: the training pool")
    ap.add_argument("--data", action="store_true", help="also fetch data/*.parquet (trajectories)")
    ap.add_argument("--out", default="data/videos")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pins = json.loads((Path(cfg["paths"]["meta_root"]) / "pins.json").read_text())
    for name in args.repos or cfg["train_pool"]:
        snapshot_download(
            f"{cfg['hf_namespace']}/{name}",
            repo_type="dataset",
            revision=pins[name]["sha"],
            allow_patterns=[f"videos/observation.images.{c}/*" for c in args.cams]
            + (["data/*"] if args.data else []),
            local_dir=Path(args.out) / name,
        )
        print(f"{name:45s} {pins[name]['sha'][:12]}  done", flush=True)


if __name__ == "__main__":
    main()
