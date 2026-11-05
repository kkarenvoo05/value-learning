"""Progress-routed value head on frozen pi05 (docs/libero_plan.md log 2026-10-30, option 2) [ours].

Background: next to image features a value head ignores even a perfect "pot on the stove" bit (oracle test), and head
regularisation does not change that (scripts/libero_shortcut_fixes.py). Here V is forced through progress: the image
readout (pi05 layer-5 values, 33 grid tokens, the AttnProbe layout) predicts the bit y[t] (BCE), and the value MLP sees
only that prediction, **detached** (so value CE cannot turn it into a scene code), plus the arm's extra inputs:
  routed: P + clock          [P(pot on stove), Fourier(t)]
  routed: P + clock + scene2 [P, Fourier(t), a 2-d code from the readout (the only image path the value CE can shape)]
  routed: P + scene2         [P, 2-d code], no clock: the fair comparison with the ResNet V, which has no time input
Controls (no routing):
  ResNet ft + clock          the fine-tuned ResNet features + Fourier(t), MLP (as the oracle's ResNet arms):
                             the clock baseline the routed clock arms must beat (arm E's caveat)
  true bit + clock           the oracle's upper reference (P replaced by the true bit)
Value CE, 201 bins, 3 seeds; fit / inner early stopping on value CE / val and scoring as scripts/libero_head_swap.py.
Reading rule, set before the result: routing "works" if P + scene2 (no clock) reaches the ResNet V's change SNR
(~0.19) with a positive milestone excess on 3/3 seeds; the clock arms count only if they also beat ResNet ft + clock.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/libero_progress_routed.py [--seeds 0 1 2]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl.libero_probe import CENTERS, N_BINS, inner_split  # noqa: E402
from vl.libero_progress import labels  # noqa: E402
from libero_head_swap import FEAT, SOURCES, load_meta, score  # noqa: E402
from libero_oracle_bit import MLPPlus, time_features, zscore  # noqa: E402


class Routed(nn.Module):
    """Image readout -> P(pot on stove) (trained on the bit) and an optional k-d scene code; value MLP on
    [P.detach(), extra inputs, code]."""

    def __init__(self, n_tok, w, n_extra, k=0, d=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(w, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(n_tok, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.prog = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.code = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, k)) if k else None
        v_in = 1 + n_extra + k
        self.out = nn.Sequential(nn.Linear(v_in, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, N_BINS))

    def forward(self, x, e):
        t = self.proj(x) + self.pos
        q, _ = self.attn(self.query.expand(len(x), -1, -1), t, t)
        r = q[:, 0]
        a = self.prog(r)
        z = [torch.sigmoid(a).detach(), e] + ([self.code(r)] if self.code is not None else [])
        return self.out(torch.cat(z, -1)), a


class Plain(nn.Module):
    """MLPPlus with the (value logits, aux) interface; no aux."""

    def __init__(self, d, n_extra):
        super().__init__()
        self.m = MLPPlus(d, N_BINS, n_extra)

    def forward(self, x, e):
        return self.m(x, e), None


def fit(net, X, E, bins, y, tr, iv, seed, epochs, device, bs=256):
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    ce, bce = nn.functional.cross_entropy, nn.functional.binary_cross_entropy_with_logits
    best, state, best_ep = np.inf, None, -1
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(tr), generator=g).split(bs):
            j = tr[ix.numpy()]
            lv, la = net(X[j].to(device).float(), E[j].to(device))
            loss = ce(lv, bins[j].to(device))
            if la is not None:
                loss = loss + bce(la[:, 0], y[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        c = float(ce(predict(net, X, E, iv, device)[0], bins[iv].to(device)))
        if c < best:
            best, best_ep, state = c, ep, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best, best_ep


@torch.no_grad()
def predict(net, X, E, idx, device, bs=2048):
    net.eval()
    lv, la = [], []
    for i in range(0, len(idx), bs):
        v, a = net(X[idx[i:i + bs]].to(device).float(), E[idx[i:i + bs]].to(device))
        lv.append(v)
        la.append(a if a is not None else torch.zeros(len(v), 1, device=device))
    return torch.cat(lv), torch.cat(la)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="reports/libero_progress_routed.json")
    a = ap.parse_args()
    t0 = time.time()
    meta, _ = labels(load_meta().reset_index(drop=True))
    inner = inner_split(meta)
    tr = np.flatnonzero((meta.split == "train").values & ~inner)
    iv = np.flatnonzero(inner)
    va = np.flatnonzero((meta.split == "val").values)
    bins = torch.as_tensor(meta.bin.values.astype(np.int64))
    y = torch.as_tensor(meta.y.values.astype(np.float32))
    centers = torch.as_tensor(CENTERS, dtype=torch.float32)
    clock = torch.as_tensor(time_features(meta.frame_index.values.astype(np.float32)))
    none = torch.zeros(len(meta), 0)
    layer_ids = list(np.load(FEAT / "meta.npz")["layers"])
    V = np.load(FEAT / "v_layers.npy", mmap_mode="r")
    L5 = torch.from_numpy(zscore(np.asarray(V[:, layer_ids.index(5)], np.float32), tr).astype(np.float16))
    R_ft = torch.as_tensor(zscore(np.load(FEAT / SOURCES["R_ft"]).astype(np.float32), tr))
    ck = clock.shape[1]
    arms = {
        "routed: P + clock": (lambda: Routed(L5.shape[1], L5.shape[2], ck), L5, clock),
        "routed: P + clock + scene2": (lambda: Routed(L5.shape[1], L5.shape[2], ck, k=2), L5, clock),
        "routed: P + scene2": (lambda: Routed(L5.shape[1], L5.shape[2], 0, k=2), L5, none),
        "ResNet ft + clock": (lambda: Plain(R_ft.shape[1], ck), R_ft, clock),
        "true bit + clock": (lambda: Plain(ck, 1), clock, y[:, None]),
    }
    print(f"[{time.time() - t0:.0f} s] ready on {a.device}", flush=True)
    rows, preds = [], []
    for seed in a.seeds:
        for name, (make, X, E) in arms.items():
            torch.manual_seed(seed)
            net, ice, ep = fit(make(), X, E, bins, y, tr, iv, seed, a.epochs, a.device)
            lv, la = predict(net, X, E, va, a.device)
            p = meta.iloc[va][["episode_index", "init_state", "frame_index", "success", "value", "y"]].copy()
            p["ev"] = (torch.softmax(lv, -1).cpu() @ centers).numpy()
            p["p_pot"] = torch.sigmoid(la[:, 0]).cpu().numpy()
            r = {"arm": name, "seed": seed, "inner_ce": ice, "best_epoch": ep, **score(p, "ev"),
                 "pot_acc": float(((p.p_pot > 0.5) == (p.y == 1)).mean()) if name.startswith("routed") else None,
                 "minutes": round((time.time() - t0) / 60, 1)}
            rows.append(r)
            preds.append(p.assign(arm=name, seed=seed))
            print(json.dumps(r), flush=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
    pd.concat(preds).to_parquet(Path(a.out).with_suffix(".val_predictions.parquet"))
    df = pd.DataFrame(rows)
    cols = ["inner_ce", "mae", "mae_succ>20s", "milestone", "change_snr_2s", "auc10", "pot_acc"]
    pd.set_option("display.width", 250)
    print(df.groupby("arm", sort=False)[cols].agg(["mean", "std"]).round(3).to_string())
    print(df.groupby("arm", sort=False).milestone.apply(lambda m: f"{(m > 0).sum()}/{len(m)}").to_string())


if __name__ == "__main__":
    main()
