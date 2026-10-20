"""Full predicted value distributions (all 201 bins) of a run's best.pt on the validation split.

Usage (on a GPU node; see scripts/value_dist.sbatch):
    .venv/bin/python -u -m vl.value_dist --run lowlr_8k
        -> data/runs/<run>/val_probs.npz   (probs float16 (n_val, 201), same row order as
                                            val_predictions.parquet, plus V for a consistency check)

val_predictions.parquet keeps only the mean V = sum_b p_b v_b. This keeps the whole
distribution, so we can ask what V's mean is averaging over (e.g. a success mode and a failure
mode at once). Eval mode, same preprocessing and AMP as the training-time validation pass.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vl.config import load_config
from vl.dataset import FrameDataset, load_split
from vl.model import build_model
from vl.train import amp_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    run = Path(cfg["paths"]["runs"]) / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg["amp"])
    val = load_split(dcfg, "val")
    model = build_model(dcfg, tcfg).to(device).eval()
    ck = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    dl = DataLoader(FrameDataset(val, dcfg), batch_size=128, shuffle=False, num_workers=args.workers,
                    pin_memory=True, prefetch_factor=4)
    probs = np.empty((len(val), tcfg["n_bins"]), np.float16)
    V = np.empty(len(val), np.float32)
    with torch.no_grad():
        for i, b in enumerate(dl):
            x = {c: v.to(device, non_blocking=True) for c, v in b["images"].items()}
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                logits = model(x)
            rows = b["row"].numpy()
            probs[rows] = torch.softmax(logits.float(), dim=1).cpu().numpy()
            V[rows] = model.expected_value(logits).cpu().numpy()
            if i % 100 == 0:
                print(f"  batch {i}/{len(dl)}", flush=True)
    np.savez(run / "val_probs.npz", probs=probs, V=V, frame_index=val.frame_index.to_numpy(),
             episode_uid=val.episode_uid.to_numpy().astype(str))
    print(f"-> {run / 'val_probs.npz'}  (best step {ck['step']})", flush=True)


if __name__ == "__main__":
    main()
