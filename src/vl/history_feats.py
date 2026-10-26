"""Frozen-feature cache for the short-history probe (Stage 5, deferred item "short frame history").

Runs a finished run's best.pt backbone once over every `--stride`-th frame of train and val and
saves the pooled per-camera features (ResNet-18: 512 per camera), plus the checkpoint's own
logits-derived V so the probe's single-frame control can be checked against the real model.

Usage (GPU node):
    .venv/bin/python -u -m vl.history_feats --run aug_split_cropwrist_s0 --stride 3 \
        --out /move/u/karenvo/vl-cache/history
        -> <out>/<run>_stride<k>.npz   (feats float16 (N, n_cams*512), V, row ids into the frame index)

[ours] stride 3 = 5 Hz at 15 fps, so every past offset used by the probe (0.4, 1, 2, 4, 8 s) is a
whole number of cached frames. The cache goes outside data/ (data/ is on the nearly full /sailhome).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vl.config import load_config
from vl.dataset import FrameDataset, load_split
from vl.model import build_model, model_inputs, run_data_config
from vl.train import amp_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    run = Path(cfg["paths"]["runs"]) / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg["amp"])

    idx = pd.concat([load_split(dcfg, "train"), load_split(dcfg, "val")], ignore_index=True)
    idx = idx[idx.frame_index % args.stride == 0].reset_index(drop=True)
    if args.limit:
        idx = idx.iloc[:args.limit].reset_index(drop=True)
    print(f"{run}: {len(idx):,} frames (stride {args.stride})", flush=True)

    dcfg = run_data_config(dcfg, run)
    model = build_model(dcfg, tcfg).to(device).eval()
    assert not model.per_camera_backbone, "history features assume one shared backbone"
    assert not model.history, "history features are for single-frame models"
    model.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    dl = DataLoader(FrameDataset(idx, dcfg), batch_size=256, shuffle=False, num_workers=args.workers,
                    pin_memory=True, prefetch_factor=4)
    n_cams = len(model.cams)
    feats, V = None, np.empty(len(idx), np.float32)
    with torch.no_grad():
        for i, b in enumerate(dl):
            x = model_inputs(b, device)["images"]
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                z = torch.cat([x[c] for c in model.cams])
                f = model.backbone((z.float() / 255.0 - model.mean) / model.std)
                f = torch.cat(f.chunk(n_cams), dim=1)
                logits = model.head(f)
            rows = b["row"].numpy()
            if feats is None:
                feats = np.empty((len(idx), f.shape[1]), np.float16)
            feats[rows] = f.float().cpu().numpy().astype(np.float16)
            V[rows] = model.expected_value(logits).cpu().numpy()
            if i % 50 == 0:
                print(f"batch {i}/{len(dl)}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = ["episode_uid", "repo", "episode_index", "split", "outcome", "frame_index", "T", "fps", "value", "bin"]
    idx[keep].to_parquet(out / f"{args.run}_stride{args.stride}_index.parquet", index=False)
    np.savez(out / f"{args.run}_stride{args.stride}.npz", feats=feats, V=V, cams=np.array(model.cams))
    print(f"saved {feats.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
