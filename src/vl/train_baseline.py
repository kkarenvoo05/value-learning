"""Stage 4 baseline: train V on all training frames, select the checkpoint on validation CE.

Usage (on a GPU node; see scripts/baseline.sbatch):
    .venv/bin/python -u -m vl.train_baseline --run baseline      # -> data/runs/baseline/
    .venv/bin/python -u -m vl.train_baseline --run cosine_8k --section baseline_cosine

- Sampling [ours]: uniform over every valid training frame (shuffled passes, fixed seed),
  frames decoded on the fly by FrameDataset (Stage 3.3 decision).
- Every `eval_every` steps: cross-entropy and MAE of V on the *full* validation split (eval mode),
  averaged per frame and per episode, plus the repo-aware metrics of `vl.metrics` (repo-only
  constant, eval-repo MAE and within-repo AUC, offset-free shape error). The checkpoint with the lowest per-frame validation CE is
  kept as best.pt [ours: CE is the training objective; MAE is reported alongside].
- Before training, the trivial constant predictor (training bin histogram) is scored on the
  same validation frames, so every eval line can be read against it.
- At the end, best.pt is re-evaluated and its per-frame validation predictions saved for the
  Stage 4 analysis (val_predictions.parquet).
- Logged per interval: loss, samples/s and the fraction of time spent waiting for data, which
  shows whether the video decoder limits training.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vl.config import load_config
from vl.dataset import FrameDataset, load_split
from vl.metrics import repo_medians, score
from vl.model import build_model, value_loss
from vl.train import amp_dtype


def constant_baseline(train: pd.DataFrame, val: pd.DataFrame, n_bins: int, centers: np.ndarray) -> dict:
    """Predict the same distribution for every frame: the training bin histogram (add-one
    smoothed so unseen bins have finite CE) [ours]. Its V is the histogram mean; the best
    constant for MAE, the training median, is reported too."""
    counts = np.bincount(train.bin, minlength=n_bins) + 1.0
    p = counts / counts.sum()
    ce = -np.log(p[val.bin.to_numpy()])
    v_mean = float(p @ centers)
    v_med = float(train.value.median())
    return {"val_ce": float(ce.mean()), "val_ce_per_episode": float(pd.Series(ce).groupby(val.episode_uid.to_numpy()).mean().mean()),
            "V_constant": v_mean, "val_mae": float(np.abs(val.value - v_mean).mean()),
            "train_median": v_med, "val_mae_median": float(np.abs(val.value - v_med).mean())}


@torch.no_grad()
def predict(model, loader, val: pd.DataFrame, device, dtype) -> pd.DataFrame:
    model.eval()
    V = np.empty(len(val), np.float32)
    ce = np.empty(len(val), np.float32)
    for b in loader:
        x = {c: v.to(device, non_blocking=True) for c, v in b["images"].items()}
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x)
        rows = b["row"].numpy()
        V[rows] = model.expected_value(logits).cpu().numpy()
        ce[rows] = F.cross_entropy(logits.float(), b["bin"].to(device), reduction="none").cpu().numpy()
    model.train()
    pred = val[["episode_uid", "repo", "episode_index", "outcome", "frame_index", "T", "value", "bin"]].assign(V=V, ce=ce)
    return pred


def lr_factor(step: int, steps: int, bc: dict) -> float:
    """Linear warmup, then constant or cosine decay to 0 at the last step [ours]."""
    if step <= bc["warmup_steps"]:
        return step / bc["warmup_steps"]
    if bc.get("schedule", "constant") == "cosine":
        return 0.5 * (1 + math.cos(math.pi * (step - bc["warmup_steps"]) / (steps - bc["warmup_steps"])))
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--section", default="baseline", help="config section with the schedule (baseline, baseline_cosine)")
    ap.add_argument("--steps", type=int, help="override baseline.steps (smoke tests)")
    ap.add_argument("--val-limit", type=int, help="use only the first N validation episodes (smoke tests)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bc = cfg[args.section]
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    steps = args.steps or bc["steps"]
    run = Path(cfg["paths"]["runs"]) / args.run
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg["amp"])
    workers = bc["workers"]

    train, val = load_split(dcfg, "train"), load_split(dcfg, "val")
    if args.val_limit:
        val = val[val.episode_uid.isin(val.episode_uid.unique()[:args.val_limit])].reset_index(drop=True)
    centers = np.linspace(tcfg["v_min"], tcfg["v_max"], tcfg["n_bins"])
    trivial = constant_baseline(train, val, tcfg["n_bins"], centers)
    rmed = repo_medians(train)
    print(f"train {len(train):,} frames / {train.episode_uid.nunique()} episodes; "
          f"val {len(val):,} / {val.episode_uid.nunique()}", flush=True)
    print(f"constant predictor on val: CE {trivial['val_ce']:.4f}, MAE {trivial['val_mae']:.4f} "
          f"(median constant MAE {trivial['val_mae_median']:.4f})", flush=True)

    train_dl = DataLoader(FrameDataset(train, dcfg), batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=workers, persistent_workers=True, pin_memory=True, prefetch_factor=4,
                          generator=torch.Generator().manual_seed(cfg["seed"]))
    val_dl = DataLoader(FrameDataset(val, dcfg), batch_size=bc["eval_batch_size"], shuffle=False,
                        num_workers=workers, persistent_workers=True, pin_memory=True, prefetch_factor=4)

    meta = {"args": vars(args), "config": cfg, "steps": steps, "trivial_baseline": trivial,
            "n_train": len(train), "n_val": len(val),
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "amp": str(dtype),
            "torch": str(torch.__version__),
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    (run / "config.json").write_text(json.dumps(meta, indent=2))

    model = build_model(dcfg, tcfg).to(device).train()
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
                             {"params": model.head.parameters(), "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    base_lr = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    logf = open(run / "log.jsonl", "w")

    def write(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    best, step, epoch = math.inf, 0, 0
    t0 = t_int = time.time()
    wait = 0.0
    losses = []
    while step < steps:
        epoch += 1
        t_wait = time.time()
        for b in train_dl:
            wait += time.time() - t_wait
            step += 1
            f = lr_factor(step, steps, bc)
            for g, lr in zip(opt.param_groups, base_lr):
                g["lr"] = lr * f
            x = {c: v.to(device, non_blocking=True) for c, v in b["images"].items()}
            y = b["bin"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                logits = model(x)
            loss = value_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            if step % cfg["log_every"] == 0:
                dt = time.time() - t_int
                rec = {"kind": "train", "step": step, "epoch": epoch, "loss": float(np.mean(losses)), "lr_factor": round(f, 4),
                       "samples_per_s": round(cfg["batch_size"] * cfg["log_every"] / dt, 1),
                       "data_wait_frac": round(wait / dt, 3), "elapsed_min": round((time.time() - t0) / 60, 1)}
                write(rec)
                if step % (cfg["log_every"] * 8) == 0:
                    print(f"step {step:6d} ep {epoch}  train CE {rec['loss']:.4f}  {rec['samples_per_s']:.0f} samples/s  "
                          f"data wait {rec['data_wait_frac']:.0%}  {rec['elapsed_min']:.0f} min", flush=True)
                losses, wait, t_int = [], 0.0, time.time()
            if step % bc["eval_every"] == 0 or step == steps:
                te = time.time()
                ev = score(predict(model, val_dl, val, device, dtype), rmed)
                is_best = ev["val_ce"] < best
                if is_best:
                    best = ev["val_ce"]
                    torch.save({"model": model.state_dict(), "step": step, "eval": ev, "meta": meta}, run / "best.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "eval": ev,
                            "meta": meta}, run / "last.pt")
                write({"kind": "eval", "step": step, "epoch": epoch, **ev, "best": is_best,
                       "eval_min": round((time.time() - te) / 60, 1)})
                print(f"step {step:6d}  VAL CE {ev['val_ce']:.4f} (const {trivial['val_ce']:.4f})  "
                      f"MAE {ev['val_mae']:.4f} (const {trivial['val_mae']:.4f}, repo-const {ev['repo_const_mae']:.4f})  "
                      f"eval-repo MAE {ev['eval_repos_mae']:.4f} (repo-const {ev['eval_repos_repo_const_mae']:.4f})  "
                      f"shape {ev['offset_free_mae']:.4f} (flat {ev['offset_free_mae_flat']:.4f})  "
                      f"AUC@1/5/10s {ev['eval_repos_auc_1s']:.2f}/{ev['eval_repos_auc_5s']:.2f}/{ev['eval_repos_auc_10s']:.2f}"
                      f"{'  * best' if is_best else ''}  [{(time.time() - te) / 60:.1f} min]", flush=True)
                t_int, wait = time.time(), 0.0
            if step >= steps:
                break
            t_wait = time.time()
    logf.close()

    ck = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    pred = predict(model, val_dl, val, device, dtype)
    ev = score(pred, rmed)
    pred.to_parquet(run / "val_predictions.parquet", index=False)
    (run / "result.json").write_text(json.dumps({"best_step": ck["step"], "best_eval": ev, "trivial": trivial}, indent=2))
    print(f"best step {ck['step']}: VAL CE {ev['val_ce']:.4f}, MAE {ev['val_mae']:.4f}; "
          f"done in {(time.time() - t0) / 60:.0f} min -> {run}", flush=True)


if __name__ == "__main__":
    main()
