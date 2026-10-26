"""Short-history probe on frozen features (Stage 5, deferred item "short frame history") [ours].

Question: does seeing past frames help V, on top of the current frame? Cheap version: freeze the
crop-wrist backbone (features cached by `vl.history_feats`, 5 Hz), train only a small head whose
input is the current frame's features plus features from the past. Every arm has the same head as
the real model (Linear -> 256 -> ReLU -> 201 bins, CE); only the input changes. Past frames only:
future frames would leak the outcome and break Stage 6's V(o_{t+n}) - V(o_t).

Arms: ctrl (current frame), past_<s> (current + one frame s seconds back), multi4_<s> (4 frames
spread over s seconds), elapsed (current + seconds since episode start). Near the episode start
the first frame is repeated (leaks a little "early in episode", which the elapsed arm measures).

Usage: .venv/bin/python -u -m vl.history_probe --cache /move/u/karenvo/vl-cache/history/aug_split_cropwrist_s0_stride3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
rankdata = lambda s: pd.Series(np.asarray(s)).rank().to_numpy()

WORST = [("r04-eval", 101), ("r02-eval", 44), ("r02-eval", 45), ("r01-eval", 48)]


def roc_auc_score(y, s) -> float:
    """Mann-Whitney AUC (no sklearn in the venv)."""
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def past_index(idx: pd.DataFrame, k: int) -> np.ndarray:
    """Row of the frame k cached steps earlier in the same episode (clamped to the first frame)."""
    pos = idx.groupby("episode_uid").cumcount().to_numpy()
    return np.arange(len(idx)) - np.minimum(pos, k)


def head(d_in: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 201))


def fit(Xtr, ytr, Xva, yva, seed, epochs, dev):
    torch.manual_seed(seed)
    m = head(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    best, best_state = 1e9, None
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        m.train()
        for b in torch.randperm(len(Xtr), generator=g).split(1024):
            loss = F.cross_entropy(m(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            ce = F.cross_entropy(m(Xva), yva).item()
        if ce < best:                         # [ours] epoch with lowest val CE, as best.pt in the real runs
            best, best_state = ce, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    with torch.no_grad():
        return m(Xva).float()


def scores(logits, va: pd.DataFrame, centers, hold, worst_rows) -> dict:
    P = F.softmax(logits, -1)
    yb = torch.as_tensor(va.bin.to_numpy(), device=logits.device)
    V = (P @ centers).cpu().numpy()
    p_succ = P[:, centers > -0.5].sum(1).cpu().numpy()
    err = np.abs(V - va.value.to_numpy())
    out = {"ce": F.cross_entropy(logits, yb).item(), "mae": err.mean(),
           "mae_success": err[va.outcome.to_numpy() == "success"].mean(),
           "mae_timeout": err[va.outcome.to_numpy() == "timeout"].mean(),
           "mae_failure": err[va.outcome.to_numpy() == "failure"].mean(),
           "mae_eval": err[va.repo.str.contains("eval").to_numpy()].mean()}
    h = va[hold].assign(p=p_succ[hold]).groupby("episode_uid").agg(p=("p", "mean"), s=("succ", "first"))
    out["hold_auc"] = roc_auc_score(h.s, h.p)
    for name, r in worst_rows.items():
        out[f"p_{name}"] = p_succ[r].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="path prefix written by vl.history_feats")
    ap.add_argument("--run", default="aug_split_cropwrist_s0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--save-preds", action="store_true",
                    help="also write per-frame val V, p_succ and CE (mean over head seeds) to <cache>_preds.parquet")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = pd.read_parquet(f"{args.cache}_index.parquet")
    z = np.load(f"{args.cache}.npz")
    feats = torch.as_tensor(z["feats"].astype(np.float32))
    man = pd.read_parquet("data/manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    idx = idx.merge(man, on="episode_uid", how="left", validate="m:1")
    # past_index needs each episode's rows contiguous and in frame order, one cached step apart.
    first = idx.groupby("episode_uid").cumcount().to_numpy() == 0
    stride = int(np.diff(idx.frame_index.to_numpy())[~first[1:]].min())
    assert (np.diff(idx.frame_index.to_numpy())[~first[1:]] == stride).all()
    step_s = stride / 15
    tr, va_m = (idx.split == "train").to_numpy(), (idx.split == "val").to_numpy()
    va = idx[va_m].reset_index(drop=True)
    centers = torch.linspace(-1, 0, 201, device=dev)

    # Reference: the checkpoint's own head applied to the cached features (= the real model on these frames).
    ck = torch.load(Path("data/runs") / args.run / "best.pt", map_location="cpu")["model"]
    ref = head(feats.shape[1])
    ref.load_state_dict({k.removeprefix("head."): v for k, v in ck.items() if k.startswith("head.")})

    # Hold phase: autonomous eval episodes, > 5 s in and > 2 s before T, success vs VLM timeout_holding_marker.
    t_s = va.frame_index / va.fps
    succ = va.outcome == "success"
    stall = va.machine_failure_mode == "timeout_holding_marker"
    hold = (va.repo.str.contains("eval") & (succ | stall) & (t_s > 5) & (va["T"] / va.fps - t_s > 2)).to_numpy()
    va["succ"] = succ.to_numpy()
    worst_rows = {f"{r.split('-')[0]}/{e}": (hold & va.repo.str.contains(r).to_numpy() & (va.episode_index == e).to_numpy())
                  for r, e in WORST}
    print(f"val {len(va):,} rows, hold phase {va[hold].episode_uid.nunique()} episodes "
          f"({va[hold & va.succ.to_numpy()].episode_uid.nunique()} success)", flush=True)

    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-3
    with torch.no_grad():
        ref_logits = ref(feats[va_m]).float().to(dev)
    rows = [dict(arm="ckpt_head", seed=-1, **scores(ref_logits, va, centers, hold, worst_rows))]
    print(rows[-1], flush=True)

    fz = (feats - mu) / sd
    elapsed = torch.as_tensor((idx.frame_index / idx.fps / 60.0).to_numpy(np.float32))[:, None]
    pos = torch.as_tensor(idx.groupby("episode_uid").cumcount().to_numpy())
    no_past = lambda k: (pos < k).float()[:, None]
    arms = {"ctrl": lambda: fz, "elapsed": lambda: torch.cat([fz, elapsed], 1),
            # control for long spans, which mostly clamp to the episode's first frame (start configuration)
            "first": lambda: torch.cat([fz, fz[past_index(idx, 10 ** 6)]], 1)}
    for s in (0.4, 1.2, 2, 4, 8, 12, 16, 24):
        k = round(s / step_s)
        arms[f"past_{s}"] = lambda k=k: torch.cat([fz, fz[past_index(idx, k)]], 1)
        arms[f"multi4_{s}"] = lambda k=k: torch.cat([fz] + [fz[past_index(idx, round(k * j / 3))] for j in (1, 2, 3)], 1)
        # masked: no first-frame fallback. Past features are zero where the frame k steps back doesn't
        # exist, plus a "no past" flag; flag alone controls for the flag's "episode younger than s" info.
        arms[f"masked_{s}"] = lambda k=k: torch.cat([fz, fz[past_index(idx, k)] * (1 - no_past(k)), no_past(k)], 1)
        arms[f"flag_{s}"] = lambda k=k: torch.cat([fz, no_past(k)], 1)
    y = torch.as_tensor(idx.bin.to_numpy())
    preds = va[["episode_uid", "repo", "episode_index", "outcome", "machine_failure_mode", "frame_index", "T", "value"]].copy()
    for name in args.arms or arms:
        X = arms[name]()
        Xtr, ytr, Xva, yva = X[tr].to(dev), y[tr].to(dev), X[va_m].to(dev), y[va_m].to(dev)
        for seed in range(args.seeds):
            logits = fit(Xtr, ytr, Xva, yva, seed, args.epochs, dev)
            rows.append(dict(arm=name, seed=seed, **scores(logits, va, centers, hold, worst_rows)))
            if args.save_preds:
                P = F.softmax(logits, -1)
                ce = F.cross_entropy(logits, yva, reduction="none").cpu().numpy()
                for col, x in (("V", (P @ centers).cpu().numpy()), ("p_succ", P[:, centers > -0.5].sum(1).cpu().numpy()), ("ce", ce)):
                    preds[f"{col}_{name}"] = preds.get(f"{col}_{name}", 0) + x / args.seeds
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    if args.save_preds:
        preds.to_parquet(f"{args.cache}_preds.parquet", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(f"{args.cache}_probe.csv", index=False)
    summ = res.groupby("arm", sort=False).agg(["mean", "std"]).drop(columns="seed")
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
