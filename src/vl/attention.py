"""Readout attention of a C3 attention-fusion run on the validation split (Stage 5.5 C4 input).

Usage (on a GPU node, after training; scripts/cameras.sbatch ATTN=1 runs it per seed):
    .venv/bin/python -u -m vl.attention --run c3_attn_s0
        -> data/runs/<run>/val_attn.parquet   (same rows and order as val_predictions.parquet:
           episode_uid, frame_index, then attn_<cam>_l<layer>, the readout's attention on each
           camera token in each block, mean over heads; each layer sums to 1 over cameras)

Best checkpoint, eval mode, no augmentation, same AMP as the training-time validation pass.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vl.config import load_config
from vl.dataset import FrameDataset, load_split
from vl.model import build_model, run_data_config
from vl.train import amp_dtype


def attention_table(attn: np.ndarray, cams: list[str]) -> pd.DataFrame:
    """(N, layers, tokens) readout attention -> one column per (layer, camera), layer-major."""
    n, L, T = attn.shape
    assert T == len(cams), (T, cams)
    return pd.DataFrame({f"attn_{c}_l{l}": attn[:, l, i] for l in range(L) for i, c in enumerate(cams)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-limit", type=int, help="only the first N validation episodes (smoke tests)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    run = Path(cfg["paths"]["runs"]) / args.run
    dcfg = run_data_config(dcfg, run)
    if dcfg["fusion"] != "attention":
        raise SystemExit(f"{args.run} has fusion {dcfg['fusion']!r}; attention weights exist only for 'attention'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg["amp"])
    val = load_split(dcfg, "val")
    if args.val_limit:
        val = val[val.episode_uid.isin(val.episode_uid.unique()[:args.val_limit])].reset_index(drop=True)
    model = build_model(dcfg, tcfg).to(device).eval()
    ck = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    dl = DataLoader(FrameDataset(val, dcfg), batch_size=128, shuffle=False, num_workers=args.workers,
                    pin_memory=True, prefetch_factor=4)
    attn = np.empty((len(val), len(model.head.blocks), len(model.cams)), np.float32)
    with torch.no_grad():
        for i, b in enumerate(dl):
            images = {c: v.to(device, non_blocking=True) for c, v in b["images"].items()}
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                _, a = model.head(model.features(images), return_attn=True)
            attn[b["row"].numpy()] = a.float().cpu().numpy()
            if i % 100 == 0:
                print(f"  batch {i}/{len(dl)}", flush=True)
    out = pd.concat([val[["episode_uid", "frame_index"]].reset_index(drop=True),
                     attention_table(attn, model.cams)], axis=1)
    out.to_parquet(run / "val_attn.parquet")
    sums = attn.sum(-1)
    print(f"-> {run / 'val_attn.parquet'}  (best step {ck['step']}; layer sums in [{sums.min():.4f}, {sums.max():.4f}]; "
          f"mean attention {dict(zip(out.columns[2:], out.iloc[:, 2:].mean().round(3)))})", flush=True)


if __name__ == "__main__":
    main()
