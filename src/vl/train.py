"""Train the value model. For now: Check C, the tiny-subset overfit (Stage 3.8).

Usage:
    .venv/bin/python -m vl.train --check-c --run check_c        # -> data/runs/check_c/

Check C trains on ~4 short training episodes (one success, failure, timeout and assisted
episode) for many passes. A correct pipeline must drive the training cross-entropy far below
ln 201 = 5.30 and make predicted V follow the target countdown on those same episodes. If it
cannot even memorize a few hundred frames, the loader, targets or model is broken.

The subset's items are built once through FrameDataset (the real decode/crop path) and kept in
memory, so the overfit run measures the model, not the video decoder.

Outputs in the run directory: config.json, log.jsonl (one line per logged step / eval),
predictions.parquet (per-frame V and CE at the final eval), check_c.png, model.pt.
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
from vl.dataset import FrameDataset
from vl.model import build_model, value_loss


def pick_check_c_episodes(idx: pd.DataFrame, manifest: pd.DataFrame, cfg: dict, seed: int) -> list[str]:
    """One short training episode per kind. success/failure/timeout: fully autonomous rollouts.
    assisted: a DAgger rollout with a human takeover (not a teleop or counterfactual all-human demo)."""
    ep = idx[idx.split == "train"].groupby("episode_uid").first()
    ep = ep.join(manifest.set_index("episode_uid")[["is_counterfactual"]])
    short = ep[ep["T"] + 1 <= cfg["max_frames"]]
    auto = ~short.has_human_frames
    kinds = {"success": (short.outcome == "success") & auto,
             "failure": (short.outcome == "failure") & auto,
             "timeout": (short.outcome == "timeout") & auto,
             "assisted": (short.source_type == "dagger") & short.has_human_frames
                         & ~short.is_counterfactual.fillna(False).astype(bool)}
    uids = []
    for k in cfg["kinds"]:
        uids += short[kinds[k]].sample(cfg["episodes_per_kind"], random_state=seed).index.tolist()
    return uids


def cache_items(ds: FrameDataset, workers: int) -> dict:
    """Decode every item once (real loader path) into stacked tensors."""
    dl = DataLoader(ds, batch_size=64, num_workers=workers, shuffle=False)
    parts = list(dl)
    return {"images": {c: torch.cat([p["images"][c] for p in parts]) for c in ds.cams},
            "bin": torch.cat([p["bin"] for p in parts]), "value": torch.cat([p["value"] for p in parts])}


def amp_dtype(setting: str):
    if setting == "off" or not torch.cuda.is_available():
        return None
    if setting == "bf16" or (setting == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model, data: dict, device, dtype, batch=128) -> dict:
    """Eval-mode predictions for every cached frame: V, CE, predicted-bin."""
    model.eval()
    V, ce = [], []
    n = len(data["bin"])
    for i in range(0, n, batch):
        x = {c: v[i:i + batch].to(device, non_blocking=True) for c, v in data["images"].items()}
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x)
        y = data["bin"][i:i + batch].to(device)
        V.append(model.expected_value(logits).cpu())
        ce.append(F.cross_entropy(logits.float(), y, reduction="none").cpu())
    model.train()
    V, ce = torch.cat(V).numpy(), torch.cat(ce).numpy()
    return {"V": V, "ce": ce, "mean_ce": float(ce.mean()),
            "mae": float(np.abs(V - data["value"].numpy()).mean())}


def plot_check_c(run: Path, log: list[dict], pred: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = pred.episode_uid.unique()
    fig, axes = plt.subplots(1, 1 + len(eps), figsize=(4.2 * (1 + len(eps)), 3.6))
    tr = [r for r in log if r["kind"] == "train"]
    ev = [r for r in log if r["kind"] == "eval"]
    ax = axes[0]
    ax.plot([r["step"] for r in tr], [r["loss"] for r in tr], label="train CE (train mode)")
    ax.plot([r["step"] for r in ev], [r["mean_ce"] for r in ev], "o-", label="eval CE (eval mode)")
    ax.axhline(math.log(201), color="gray", ls="--", lw=1, label="ln 201 (uniform)")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title("loss"); ax.legend(fontsize=7)
    for ax, uid in zip(axes[1:], eps):
        e = pred[pred.episode_uid == uid].sort_values("frame_index")
        t = e.frame_index / e.fps
        ax.plot(t, e.value, color="black", lw=1.5, label="target v")
        ax.plot(t, e.V, color="tab:orange", lw=1.2, label="predicted V (raw)")
        ax.set_ylim(-1.02, 0.02); ax.set_xlabel("time (s)")
        tag = "assisted " if e.has_human_frames.iloc[0] else ""
        ax.set_title(f"{tag}{e.outcome.iloc[0]}: {e.repo.iloc[0][15:]} ep{e.episode_index.iloc[0]}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(run / "check_c.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run", required=True, help="run name, -> <paths.runs>/<run>/")
    ap.add_argument("--check-c", action="store_true", help="tiny-subset overfit (Stage 3.8)")
    ap.add_argument("--steps", type=int, help="override check_c.steps")
    args = ap.parse_args()
    assert args.check_c, "only Check C is implemented so far"
    cfg = load_config(args.config)
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    cc = cfg["check_c"]
    steps = args.steps or cc["steps"]
    run = Path(cfg["paths"]["runs"]) / args.run
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg["amp"])

    idx = pd.read_parquet(dcfg["paths"]["frame_index"])
    manifest = pd.read_parquet(load_config(dcfg["dataset_config"])["paths"]["manifest"])
    uids = pick_check_c_episodes(idx, manifest, cc, cfg["seed"])
    sub = idx[idx.episode_uid.isin(uids)].reset_index(drop=True)
    t0 = time.time()
    data = cache_items(FrameDataset(sub, dcfg), cfg["workers"])
    print(f"Check C subset: {len(uids)} episodes, {len(sub)} frames, decoded in {time.time() - t0:.0f}s", flush=True)
    print(sub.groupby("episode_uid").agg(outcome=("outcome", "first"), T=("T", "first"),
                                         assisted=("has_human_frames", "first")).to_string(), flush=True)

    meta = {"args": vars(args), "config": cfg, "steps": steps, "episodes": uids, "n_frames": len(sub),
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "amp": str(dtype), "torch": str(torch.__version__),
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    (run / "config.json").write_text(json.dumps(meta, indent=2))

    model = build_model(dcfg, tcfg).to(device).train()
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
                             {"params": model.head.parameters(), "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    g = torch.Generator().manual_seed(cfg["seed"])
    n, log = len(sub), []
    logf = open(run / "log.jsonl", "w")

    def write(rec):
        log.append(rec)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    t_start = time.time()
    for step in range(1, steps + 1):
        i = torch.randint(0, n, (cfg["batch_size"],), generator=g)   # uniform over subset frames
        x = {c: v[i].to(device, non_blocking=True) for c, v in data["images"].items()}
        y = data["bin"][i].to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x)
        loss = value_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % cfg["log_every"] == 0 or step == 1:
            write({"kind": "train", "step": step, "loss": loss.item(), "elapsed_s": round(time.time() - t_start, 1)})
            print(f"step {step:5d}  train CE {loss.item():.4f}", flush=True)
        if step % cc["eval_every"] == 0 or step == steps:
            ev = evaluate(model, data, device, dtype)
            write({"kind": "eval", "step": step, "mean_ce": ev["mean_ce"], "mae": ev["mae"]})
            print(f"step {step:5d}  eval CE {ev['mean_ce']:.4f}  MAE(V, v) {ev['mae']:.4f}", flush=True)
    logf.close()

    pred = sub[["episode_uid", "repo", "episode_index", "outcome", "has_human_frames", "frame_index", "fps",
                "T", "value", "bin"]].assign(V=ev["V"], ce=ev["ce"])
    pred.to_parquet(run / "predictions.parquet", index=False)
    torch.save({"model": model.state_dict(), "meta": meta}, run / "model.pt")
    plot_check_c(run, log, pred)
    per_ep = pred.groupby("episode_uid").apply(
        lambda e: pd.Series({"outcome": e.outcome.iloc[0], "frames": len(e), "CE": e.ce.mean(),
                             "MAE": (e.V - e.value).abs().mean(), "corr": np.corrcoef(e.V, e.value)[0, 1]}),
        include_groups=False)
    print(per_ep.round(4).to_string())
    print(f"done in {time.time() - t_start:.0f}s -> {run}")


if __name__ == "__main__":
    main()
