"""Stage 4 baseline: train V on all training frames, select the checkpoint on validation CE.

Usage (on a GPU node; see scripts/baseline.sbatch):
    .venv/bin/python -u -m vl.train_baseline --run baseline      # -> data/runs/baseline/
    .venv/bin/python -u -m vl.train_baseline --run cosine_8k --section baseline_cosine
    .venv/bin/python -u -m vl.train_baseline --run lowlr_8k --section baseline_lowlr
    .venv/bin/python -u -m vl.train_baseline --run lowlr_8k_s1 --section baseline_lowlr --seed 1
    .venv/bin/python -u -m vl.train_baseline --run side_s0 --section baseline_lowlr --cameras side_1
    .venv/bin/python -u -m vl.train_baseline --run frozen_s0 --section baseline_lowlr --freeze-backbone
    .venv/bin/python -u -m vl.train_baseline --run proprio_s0 --section baseline_lowlr --proprio
    .venv/bin/python -u -m vl.train_baseline --run dinov2s_s0 --section baseline_lowlr --backbone dinov2_vits14_reg
    .venv/bin/python -u -m vl.train_baseline --run c1_s0 --section baseline_lowlr_aug_full --per-camera-backbone
    .venv/bin/python -u -m vl.train_baseline --run c3_attn_s0 --section baseline_lowlr_aug_full --per-camera-backbone --fusion attention
    .venv/bin/python -u -m vl.train_baseline --run b2_gap2_s0 --section baseline_lowlr_aug_full --history-gap 2
    .venv/bin/python -u -m vl.train_baseline --run c2a_stack2_s0 --section baseline_lowlr_aug_full --history-gap 2 --history-stack
    .venv/bin/python -u -m vl.train_baseline --run c2b_s0 --section baseline_lowlr_aug_full --history-gaps 1 2 4 --history-stack
    .venv/bin/python -u -m vl.train_baseline --run b3_still_s0 --section baseline_lowlr_aug_full --still-time
    .venv/bin/python -u -m vl.train_baseline --run camdrop_s0 --section baseline_lowlr_aug_full --cam-dropout wrist_left=0.25
    .venv/bin/python -u -m vl.train_baseline --run r2_squash_s0 --section baseline_lowlr_aug_full \
        --resize wrist_left=squash --resize side_1=squash

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
import hashlib
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
from vl.dataset import RESIZE_MODES, FrameDataset, input_overrides, load_split
from vl.metrics import repo_medians, score
from vl.model import BACKBONES, FUSIONS, build_model, layer_decay_groups, model_inputs, value_loss
from vl.train import amp_dtype


def code_snapshot(repo: Path, paths=("src", "configs", "scripts")) -> tuple[str, str]:
    """Uncommitted code under `paths`: `git diff HEAD` plus the full text of untracked files, and its
    sha256. Saved per run (code.diff, meta code_sha256) because seeds of one job start minutes to hours
    apart and read whatever is on disk then; HEAD alone does not say what ran [ours]."""
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout
    text = run("diff", "HEAD", "--", *paths)
    for f in sorted(run("ls-files", "--others", "--exclude-standard", "--", *paths).split()):
        text += f"\n--- untracked {f}\n" + (Path(repo) / f).read_text(errors="replace")
    return text, hashlib.sha256(text.encode()).hexdigest()


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
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(**model_inputs(b, device))
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
    ap.add_argument("--seed", type=int, help="override the config seed (model head init + data order)")
    ap.add_argument("--cameras", nargs="+", help="override data.yaml cameras (Stage 5 camera ablation)")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the head; ImageNet backbone fixed (Stage 5 encoder ablation)")
    ap.add_argument("--proprio", action="store_true",
                    help="add the 10-d proprio input (data.yaml `proprio`; Stage 5 ablation 2)")
    ap.add_argument("--backbone", choices=BACKBONES,
                    help="override the ResNet-18 image backbone (Stage 5 stronger-backbone experiment)")
    ap.add_argument("--per-camera-backbone", action="store_true",
                    help="one backbone per camera instead of one shared (Stage 5.5 C1)")
    ap.add_argument("--fusion", choices=FUSIONS, default="concat",
                    help="how camera features are combined: concat MLP, or attention readout (Stage 5.5 C3)")
    ap.add_argument("--fusion-cfg", type=json.loads, default={}, metavar="JSON",
                    help='attention head overrides, e.g. \'{"d": 128, "n_layers": 1, "ff": 256, "dropout": 0.1}\'')
    ap.add_argument("--crop", action="append", metavar="CAM=x0,y0,x1,y1",
                    help="override one camera's data.yaml crop box (Stage 5.5 Gate 0); repeatable")
    ap.add_argument("--resize", action="append", metavar="CAM=MODE",
                    help=f"frame -> input mode for one camera, one of {RESIZE_MODES} (default crop); repeatable")
    ap.add_argument("--image-size", type=int,
                    help="override data.yaml image_size, the square model input per camera (Stage 5.5 Gate 0)")
    ap.add_argument("--history-gap", type=float, metavar="S",
                    help="add one past frame S seconds back per camera, masked before the episode start (Stage 5.5 B2)")
    ap.add_argument("--history-gaps", type=float, nargs="+", metavar="S",
                    help="several past frames, ascending (C2b; needs --history-stack)")
    ap.add_argument("--history-stack", action="store_true",
                    help="with --history-gap: stack current + past frame as 6 input channels (C2a early fusion)")
    ap.add_argument("--still-time", action="store_true",
                    help="add the seconds the gripper has been still to the head input (Stage 5.5 B3 upper bound)")
    ap.add_argument("--cam-dropout", action="append", metavar="CAM=P",
                    help="training only: zero CAM's feature per sample with probability P (wrist-optimism arm)")
    ap.add_argument("--steps", type=int, help="override baseline.steps (smoke tests)")
    ap.add_argument("--val-limit", type=int, help="use only the first N validation episodes (smoke tests)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bc = cfg[args.section]
    if args.seed is not None:
        cfg["seed"] = args.seed
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    if args.cameras:
        dcfg["cameras"] = args.cameras
    if args.proprio:
        dcfg["proprio"] = dict(dcfg["proprio"], enabled=True)
    if args.backbone:
        dcfg["backbone"] = args.backbone
    if args.per_camera_backbone:
        dcfg["per_camera_backbone"] = True
    dcfg["fusion"], dcfg["fusion_cfg"] = args.fusion, args.fusion_cfg
    dcfg = input_overrides(dcfg, args.crop, args.resize)
    if args.image_size:
        dcfg["image_size"] = args.image_size
    if args.history_gap is not None:
        dcfg["history"] = {"gap_s": args.history_gap}
        if args.history_stack:
            dcfg["history"]["stack"] = True
    if args.history_gaps is not None:
        if args.history_gap is not None or not args.history_stack:
            raise SystemExit("--history-gaps needs --history-stack and excludes --history-gap")
        dcfg["history"] = {"gaps_s": args.history_gaps, "stack": True}
    elif args.history_stack and args.history_gap is None:
        raise SystemExit("--history-stack needs --history-gap or --history-gaps")
    if args.still_time:
        dcfg["still_time"] = dict(dcfg["still_time"], enabled=True)
    dcfg["cam_dropout"] = {c: float(p) for c, p in (a.split("=") for a in args.cam_dropout or [])}
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

    # Training-only augmentation (Stage 5); validation is never augmented.
    train_dl = DataLoader(FrameDataset(train, dcfg, augment=bc.get("augment")), batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=workers, persistent_workers=True, pin_memory=True, prefetch_factor=4,
                          generator=torch.Generator().manual_seed(cfg["seed"]))
    val_dl = DataLoader(FrameDataset(val, dcfg), batch_size=bc["eval_batch_size"], shuffle=False,
                        num_workers=workers, persistent_workers=True, pin_memory=True, prefetch_factor=4)

    meta = {"args": vars(args), "config": cfg, "cameras": dcfg["cameras"], "backbone": dcfg.get("backbone", "resnet18"),
            "per_camera_backbone": bool(dcfg.get("per_camera_backbone", False)), "fusion": dcfg["fusion"],
            "fusion_cfg": dcfg["fusion_cfg"], "cam_dropout": dcfg["cam_dropout"],
            "proprio": bool(dcfg.get("proprio", {}).get("enabled")), "crops": dcfg["crops"], "resize": dcfg["resize"],
            "image_size": dcfg["image_size"], "history": dcfg.get("history"),
            "still_time": bool((dcfg.get("still_time") or {}).get("enabled")),
            "steps": steps, "trivial_baseline": trivial,
            "n_train": len(train), "n_val": len(val),
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "amp": str(dtype),
            "torch": str(torch.__version__),
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    code_text, meta["code_sha256"] = code_snapshot(Path(__file__).resolve().parents[2])
    (run / "code.diff").write_text(code_text)
    (run / "config.json").write_text(json.dumps(meta, indent=2))

    model = build_model(dcfg, tcfg).to(device).train()
    if args.freeze_backbone:
        model.freeze_backbone()
    # A section may override the top-level learning rates (decision 2(b)). A frozen backbone gets
    # no optimizer group, so the head's LR and schedule are unchanged.
    groups = [{"params": model.head.parameters(), "lr": bc.get("lr_head", cfg["lr_head"])}]
    if not args.freeze_backbone:
        lr_bb = bc.get("lr_backbone", cfg["lr_backbone"])
        if bc.get("layer_decay"):
            # ViT layer-wise LR decay: one group per layer, `lr_bb` at the top (Stage 5 backbone arm).
            groups[:0] = layer_decay_groups(model, lr_bb, bc["layer_decay"])
            print(f"layer decay {bc['layer_decay']}: backbone LR {groups[0]['lr']:.2e} (embeddings) "
                  f"... {groups[-2]['lr']:.2e} (final norm)", flush=True)
        else:
            groups.insert(0, {"params": model.backbone.parameters(), "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
          f"{' (backbone frozen)' if args.freeze_backbone else ''}", flush=True)
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
            y = b["bin"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                logits = model(**model_inputs(b, device))
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
