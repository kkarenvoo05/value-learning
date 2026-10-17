"""Labeled camera contact sheets for selected episodes, decoded from the remote videos.

ffmpeg seeks inside the pinned remote mp4 over HTTP, so only the requested frames are fetched.
Each column is one sampled frame index (same index for every camera = synchronized sampling);
the last column is always the reviewed endpoint T, plus one column after T if any exists.

Usage:
    .venv/bin/python -m vl.preview --repo real-marker-d2-r04-eval --episodes 0 1 \
        --out reports/figures
"""

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vl.config import load_config

# wrist_right is the other lens of the same stereo wrist camera; render it with --cams if needed.
CAMS = ["wrist_left", "side_1", "side_2"]
THUMB = (320, 240)


def frame_at(url: str, t: float, fps: int, size=(640, 480)) -> Image.Image | None:
    """Decode the frame starting at time t. Accurate seeking straight to the final frame
    returns nothing with these AV1 files, so seek one frame early and keep the second."""
    w, h = size
    start = max(0.0, t - 1.0 / fps)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-i", url, "-frames:v", "2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=120)
    n = len(out.stdout) // (w * h * 3)
    if out.returncode or n == 0:
        return None
    k = 0 if start == 0.0 and t < 1.0 / fps else n - 1
    return Image.frombytes("RGB", (w, h), out.stdout[k * w * h * 3:(k + 1) * w * h * 3])


def episode_sheet(row: pd.Series, eps_meta: pd.Series, cams_wanted=CAMS, n_cols: int = 6) -> Image.Image:
    fps, T, n = row.fps, int(row["T"]), int(row.raw_length)
    idx = set(np.linspace(0, T, n_cols - 1).round().astype(int).tolist() + [T])
    if row.first_human_frame >= 0:
        idx.add(int(row.first_human_frame))  # first human-controlled frame (takeover / demo start)
    idx = sorted(idx)
    if n - 1 > T:
        idx.append(n - 1)  # show what happens after the reviewed endpoint
    cams = [c for c in cams_wanted if row[f"cam_{c}"]]
    W, H = THUMB
    sheet = Image.new("RGB", (W * len(idx), 28 + (H + 18) * len(cams)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"{row.repo} ep{row.episode_index} | outcome={row.outcome} T={T} raw={n} "
                   f"| human_frames={row.n_human_frames} interv={row.n_intervention_onsets} "
                   f"| {row.policy_key}", fill="black")
    for ci, cam in enumerate(cams):
        key = f"observation.images.{cam}"
        chunk, file = int(eps_meta[f"videos/{key}/chunk_index"]), int(eps_meta[f"videos/{key}/file_index"])
        t0 = float(eps_meta[f"videos/{key}/from_timestamp"])
        url = (f"https://huggingface.co/datasets/{row.repo_id}/resolve/{row.commit}/videos/{key}/"
               f"chunk-{chunk:03d}/file-{file:03d}.mp4")
        for j, f in enumerate(idx):
            # Seek just past frame f's start (seeking to its centre overruns EOF on the last frame).
            img = frame_at(url, t0 + (f + 0.1) / fps, fps)
            x, y = j * W, 28 + ci * (H + 18)
            tag = " [T]" if f == T else " [after T]" if f > T else " [human]" if f == row.first_human_frame else ""
            label = f"{cam} f={f} t={f / fps:.2f}s{tag}"
            d.text((x + 4, y), label, fill="red" if f >= T else "black")
            if img is not None:
                sheet.paste(img.resize(THUMB), (x, y + 14))
            else:
                d.text((x + 10, y + 100), "decode failed", fill="red")
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--cams", nargs="+", default=CAMS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = pd.read_parquet(cfg["paths"]["manifest"]).set_index(["repo", "episode_index"])
    eps = pd.read_parquet(next((Path(cfg["paths"]["meta_root"]) / args.repo).glob("meta/episodes/*/*.parquet")))
    eps = eps.set_index("episode_index")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for e in args.episodes:
        row = m.loc[(args.repo, e)].copy(); row["repo"], row["episode_index"] = args.repo, e
        p = out / f"{args.repo}_ep{e:04d}.jpg"
        episode_sheet(row, eps.loc[e], args.cams).save(p, quality=85)
        print(p)


if __name__ == "__main__":
    main()
