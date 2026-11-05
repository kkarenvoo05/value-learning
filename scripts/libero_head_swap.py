"""Head swap, B vs ResNet (docs/libero_plan.md log 2026-10-30) [ours].

Question: does the value *head* (readout + loss) explain why the ResNet V reacts within episodes and arm B does not,
or do the features? Cross every cached feature set with both models' head recipes, train on fit, early-stop on the
5 inner train states, score val with the 3b pass metrics (scripts/libero_seed_table.py).

Features (features/progress_18191428, every 4th frame, all 570 episodes):
  B_h    arm B s0's final query output h (256), the input of its own value / outcome heads
  R_ft   the 3a ResNet V's fine-tuned backbone, [agentview, wrist] pooled (1024), the input of its own head
  R_in   frozen ImageNet ResNet-18, same layout (1024)
Heads:
  lin+out   B's head: Linear -> 201 bins plus Linear -> 2-way outcome, loss = sum (scripts/pi05_value_expert.py)
  lin       Linear -> 201, value CE only
  mlp       the ResNet V's head: Linear(d, 256) -> ReLU -> 201, value CE only (vl.model.ValueModel)
  mlp+out   that MLP plus the outcome head on the same 256-d hidden layer
Caveat: B_h and R_ft on fit frames come from models trained on those frames, so both are memorised there alike.
Usage: .venv/bin/python scripts/libero_head_swap.py [--seeds 0 1 2] [--out reports/libero_head_swap.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl.libero_probe import CENTERS, N_BINS, inner_split  # noqa: E402
from libero_seed_table import FPS, auc_at, milestone_dd  # noqa: E402

FEAT = Path("/iris/u/ankile/karenvo/libero/features/progress_18191428")
DATASET = Path("/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2")
SOURCES = {"B_h": "h_value_expert_aug_w256_s0_18186728.npy", "R_ft": "resnet_value.npy", "R_in": "resnet_imagenet.npy"}
HEADS = ("lin+out", "lin", "mlp", "mlp+out")


def load_meta() -> pd.DataFrame:
    z = np.load(FEAT / "meta.npz")
    meta = pd.DataFrame({k: z[k] for k in ("split", "init_state", "episode", "frame_index", "T", "success")})
    eps = pd.read_csv(DATASET / "episodes.csv")[["episode_index", "split", "source_episode"]]
    meta = meta.merge(eps, left_on=["split", "episode"], right_on=["split", "source_episode"], how="left")
    ft = pd.read_parquet(DATASET / "frame_targets.parquet", columns=["episode_index", "frame_index", "value", "bin"])
    meta = meta.merge(ft, on=["episode_index", "frame_index"], how="left", validate="one_to_one")
    assert meta.episode_index.notna().all() and meta.bin.notna().all()
    return meta


class Head(nn.Module):
    def __init__(self, d: int, kind: str):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 256), nn.ReLU()) if kind.startswith("mlp") else nn.Identity()
        w = 256 if kind.startswith("mlp") else d
        self.value = nn.Linear(w, N_BINS)
        self.outcome = nn.Linear(w, 2) if kind.endswith("+out") else None

    def forward(self, x):
        h = self.body(x)
        return self.value(h), (self.outcome(h) if self.outcome is not None else None)


def fit(X, bins, succ, tr, iv, kind, seed, epochs=40, lr=1e-3):
    torch.manual_seed(seed)
    net = Head(X.shape[1], kind)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    Xt, yb, ys = X[tr], bins[tr], succ[tr]
    best, best_state, best_ep = np.inf, None, -1
    ce = nn.functional.cross_entropy
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(tr), generator=g).split(256):
            lv, lo = net(Xt[ix])
            loss = ce(lv, yb[ix]) + (ce(lo, ys[ix]) if lo is not None else 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            c = float(ce(net(X[iv])[0], bins[iv]))   # selection on value CE, as both native recipes
        if c < best:
            best, best_ep = c, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best, best_ep


def change_snr(p: pd.DataFrame, col: str, lag: int = 5, t_max: int = 340) -> tuple[float, float, float]:
    """Within-episode change of V over `lag` grid frames (5 = 2 s), interior frames (frame index <= t_max, before the
    earliest success ends): signal = mean over frame indices of (mean change in successes - in failures), noise = sd
    of the change around its (frame index, outcome) mean. Returns (signal, noise, signal / noise)."""
    q = p.sort_values(["episode_index", "frame_index"])
    dv = q.groupby("episode_index")[col].diff(lag)
    q = q.assign(dv=dv)[dv.notna() & (q.frame_index <= t_max)]
    g = q.groupby(["frame_index", "success"]).dv.mean().unstack()
    sig = float((g[1] - g[0]).mean())
    noise = float((q.dv - q.groupby(["frame_index", "success"]).dv.transform("mean")).std())
    return sig, noise, sig / noise


def score(p: pd.DataFrame, col: str) -> dict:
    err = (p[col] - p.value).abs()
    late = (p.success == 1) & (p.frame_index >= 20 * FPS)
    # within-episode reactivity: sd of V after removing each episode's mean and the mean over episodes at each frame
    # index (two-way demean), so neither the layout level nor the shared clock trend counts
    dm = p[col] - p.groupby("episode_index")[col].transform("mean")
    dm = dm - dm.groupby(p.frame_index).transform("mean")
    r = {"mae": float(err.mean()), "mae_succ>20s": float(err[late].mean()),
         "within_ep_sd": float(dm.std()), "milestone": milestone_dd(p, col)}
    r["change_signal_2s"], r["change_noise_2s"], r["change_snr_2s"] = change_snr(p, col)
    for s in (5, 10, 20, 30):
        r[f"auc{s}"] = float(auc_at(p, col, s))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="reports/libero_head_swap.json")
    a = ap.parse_args()
    torch.set_num_threads(8)
    meta = load_meta()
    inner = inner_split(meta)
    tr = np.flatnonzero((meta.split == "train").values & ~inner)
    iv = np.flatnonzero(inner)
    va = np.flatnonzero((meta.split == "val").values)
    bins = torch.as_tensor(meta.bin.values.astype(np.int64))
    succ = torch.as_tensor(meta.success.values.astype(np.int64))
    centers = torch.as_tensor(CENTERS, dtype=torch.float32)
    rows, preds = [], []
    for src, fn in SOURCES.items():
        X = np.load(FEAT / fn).astype(np.float32)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6           # z-score on fit only
        X = torch.as_tensor((X - mu) / sd)
        for kind in HEADS:
            for seed in a.seeds:
                net, ice, ep = fit(X, bins, succ, tr, iv, kind, seed)
                with torch.no_grad():
                    ev = (torch.softmax(net(X[va])[0], -1) @ centers).numpy()
                p = meta.iloc[va][["episode_index", "init_state", "frame_index", "success", "value"]].copy()
                p["ev"] = ev
                r = {"source": src, "head": kind, "seed": seed, "inner_ce": ice, "best_epoch": ep, **score(p, "ev")}
                rows.append(r)
                preds.append(p.assign(source=src, head=kind, seed=seed))
                print(json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    pd.concat(preds).to_parquet(Path(a.out).with_suffix(".val_predictions.parquet"))
    df = pd.DataFrame(rows)
    cols = ["inner_ce", "mae", "mae_succ>20s", "within_ep_sd", "milestone", "auc5", "auc10", "auc30"]
    agg = df.groupby(["source", "head"])[cols].agg(["mean", "std"])
    pd.set_option("display.width", 250)
    print(agg.round(3).to_string())
    pos = df.groupby(["source", "head"]).milestone.apply(lambda m: f"{(m > 0).sum()}/{len(m)}")
    print(pos.to_string())


if __name__ == "__main__":
    main()
