"""How often is the operator (hand, arm, clothing) visible in a camera during the valid range?

Decodes the local pinned videos (vl.fetch_videos) at one frame per `--stride` frames over 0..T,
plus T itself, and runs an off-the-shelf COCO person segmenter (torchvision DeepLabV3-ResNet101,
VOC label 15 = person). Per sampled frame it stores the person-pixel fraction over the whole frame
and inside the Resize(256)+CenterCrop(224) box, and the maximum person probability.

The detector is not ground truth: calibrate the threshold against hand labels before reading
rates off it (reports/operator_visibility.md).

Usage:
    .venv/bin/python -m vl.operator_visibility --cam wrist_left
    -> data/audit/operator_visibility_<cam>.parquet
"""

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101

from vl.config import load_config
from vl.preview import center_crop_box

PERSON = 15  # VOC label index used by torchvision's segmentation weights


def episode_videos(meta_root: Path, repo: str, cam: str) -> pd.DataFrame:
    key = f"videos/observation.images.{cam}"
    eps = pd.concat(pd.read_parquet(p) for p in sorted((meta_root / repo).glob("meta/episodes/*/*.parquet")))
    return pd.DataFrame({"episode_index": eps.episode_index, "chunk": eps[f"{key}/chunk_index"].astype(int),
                         "file": eps[f"{key}/file_index"].astype(int), "t0": eps[f"{key}/from_timestamp"]})


def wanted_frames(T: int, stride: int) -> list[int]:
    return sorted(set(range(0, T + 1, stride)) | {T})


def decode_file(path: Path, wanted: dict[int, tuple]) -> dict[tuple, np.ndarray]:
    """Decode one mp4 once; `wanted` maps file-level frame index -> key. Returns key -> RGB."""
    out, last = {}, max(wanted)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for fr in c.decode(s):
            i = round(float(fr.time) * float(s.average_rate))
            if i in wanted:
                out[wanted[i]] = fr.to_ndarray(format="rgb24")
            if i >= last:
                break
    return out


@torch.no_grad()
def person_stats(model, frames: list[np.ndarray], dev: str) -> list[dict]:
    x = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).float() / 255
    x = F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.autocast(dev, dtype=torch.float16):
        p = model(x)["out"].float().softmax(1)[:, PERSON]
    h, w = p.shape[1:]
    x0, y0, x1, y1 = center_crop_box(w, h)
    m = p > 0.5
    return [{"person_frac": float(m[i].float().mean()),
             "person_frac_crop": float(m[i, y0:y1, x0:x1].float().mean()),
             "person_pmax": float(p[i].max())} for i in range(len(frames))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--cam", default="wrist_left")
    ap.add_argument("--stride", type=int, default=15, help="frames between samples (15 = 1 s)")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out", default="data/audit")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta_root = Path(cfg["paths"]["meta_root"])
    m = pd.read_parquet(cfg["paths"]["manifest"])
    m = m[m.repo.isin(cfg["train_pool"]) & m["T"].notna() & m[f"cam_{args.cam}"]]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).eval().to(dev)

    rows = []
    for repo, mr in m.groupby("repo"):
        vids = episode_videos(meta_root, repo, args.cam).merge(mr[["episode_index", "T", "fps"]], on="episode_index")
        for (chunk, file), g in vids.groupby(["chunk", "file"]):
            path = (Path(args.videos) / repo / f"videos/observation.images.{args.cam}"
                    / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4")
            wanted = {}
            for r in g.itertuples():
                base = round(r.t0 * r.fps)
                for f in wanted_frames(int(r.T), args.stride):
                    wanted[base + f] = (r.episode_index, f)
            frames = decode_file(path, wanted)
            missing = set(wanted.values()) - set(frames)
            if missing:
                raise RuntimeError(f"{path}: {len(missing)} sampled frames not decoded, e.g. {sorted(missing)[:3]}")
            keys = sorted(frames)
            for i in range(0, len(keys), args.batch):
                kb = keys[i:i + args.batch]
                for k, s in zip(kb, person_stats(model, [frames[k] for k in kb], dev)):
                    rows.append({"repo": repo, "episode_index": k[0], "frame_index": k[1], **s})
        print(f"{repo:40s} {len(mr):5d} episodes  {sum(r['repo'] == repo for r in rows):6d} frames", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"operator_visibility_{args.cam}.parquet", index=False)
    print(f"wrote {len(df)} frames -> {out / f'operator_visibility_{args.cam}.parquet'}")


if __name__ == "__main__":
    main()
