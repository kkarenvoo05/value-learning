"""Per-frame predictions of a finished run's best.pt on training episodes (Stage 4 train/val gap).

Usage (on a GPU node; see scripts/predict.sbatch):
    .venv/bin/python -u -m vl.predict --run lowlr_8k --episodes 250
        -> data/runs/<run>/train_predictions.parquet   (same columns as val_predictions.parquet)

The training split has 1,377 episodes / 374k frames; scoring all of it would take longer than
training. A fixed random sample of whole episodes [ours: 250 episodes, seed 0, about the size of
the validation split] gives the train side of the gap at the same per-frame and per-episode
granularity as validation. Eval mode, same preprocessing and AMP as the validation pass.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vl.config import load_config
from vl.dataset import FrameDataset, load_split
from vl.model import build_model, run_data_config
from vl.train import amp_dtype
from vl.train_baseline import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    run = Path(cfg["paths"]["runs"]) / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg["amp"])

    train = load_split(dcfg, "train")
    uids = np.random.default_rng(0).choice(train.episode_uid.unique(), args.episodes, replace=False)
    sub = train[train.episode_uid.isin(uids)].reset_index(drop=True)
    print(f"{run}: {len(sub):,} frames from {args.episodes} training episodes", flush=True)

    dcfg = run_data_config(dcfg, run)
    model = build_model(dcfg, tcfg).to(device)
    ck = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    dl = DataLoader(FrameDataset(sub, dcfg), batch_size=128, shuffle=False, num_workers=args.workers,
                    pin_memory=True, prefetch_factor=4)
    pred = predict(model, dl, sub, device, dtype)
    pred.to_parquet(run / "train_predictions.parquet", index=False)
    print(f"best step {ck['step']}: train CE {pred.ce.mean():.4f}, MAE {(pred.V - pred.value).abs().mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
