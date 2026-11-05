"""Oracle "pot on the stove" bit for the value heads (docs/libero_plan.md log 2026-10-30) [ours].

Question: frozen pi05 features decode "a pot is on the stove" (progress probe, time-matched AUC 0.96-0.99), yet value
heads on them give a noisy within-episode V (change SNR 0.08-0.11 vs the fine-tuned ResNet's 0.19). Is the head just
not extracting a cue that would help, or is the pot cue not what makes the ResNet's V clean?
Give the value head the true bit y[t] = (>= 1 pot On(cook_region) in image t; vl.libero_progress.labels, the
progress probe's label) next to the features, on train and val alike. Arms, value CE only, 201 bins, 3 seeds, fit /
inner early stopping / val as scripts/libero_head_swap.py, scored with its `score` (pass metrics + change SNR 2 s):
  pi05 L5 / L5 + bit       AttnProbe on layer-5 values (33 tokens); the bit joins the readout before the output MLP
  bit + clock              MLP on [bit, Fourier(t)] only: what the bit can do without any image
  B h / B h + bit          the head-swap MLP (1024 -> 256 -> 201 layout, d -> 256) on arm B s0's h
  ResNet ft / ResNet ft + bit   the same MLP on the fine-tuned ResNet V's backbone features
Reading rule, set before the result: if L5 + bit reaches the ResNet's SNR (~0.19) while L5 alone stays ~0.10, the cue
helps and the head was not extracting it (fix the training set-up); if L5 + bit stays near L5, the pot cue is not
what makes the ResNet's V clean.
Usage: .venv/bin/python scripts/libero_oracle_bit.py [--seeds 0 1 2] [--device cuda]
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
from vl.libero_probe import CENTERS, AttnProbe, N_BINS, inner_split  # noqa: E402
from vl.libero_progress import labels  # noqa: E402
from libero_head_swap import FEAT, SOURCES, load_meta, score  # noqa: E402


class AttnPlus(AttnProbe):
    """AttnProbe whose readout is concatenated with extra inputs (the bit) before the output MLP."""

    def __init__(self, n_tok, w, n_out, k):
        super().__init__(n_tok, w, n_out)
        d = self.out[1].in_features
        self.out = nn.Sequential(nn.LayerNorm(d + k), nn.Linear(d + k, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, n_out))

    def forward(self, x, e):
        t = self.proj(x) + self.pos
        q, _ = self.attn(self.query.expand(len(x), -1, -1), t, t)
        return self.out(torch.cat([q[:, 0], e], -1))


class MLPPlus(nn.Module):
    """The head-swap MLP (d -> 256 -> ReLU -> 201) on [features, extra]."""

    def __init__(self, d, n_out, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + k, 256), nn.ReLU(), nn.Linear(256, n_out))

    def forward(self, x, e):
        return self.net(torch.cat([x, e], -1))


def time_features(t, n=8):
    x = t[:, None] / 520.0
    f = np.pi * 2.0 ** np.arange(n)
    return np.concatenate([x, np.sin(f * x), np.cos(f * x)], 1).astype(np.float32)


def fit(net, X, E, bins, tr, iv, seed, epochs, device, bs=256):
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    ce = nn.functional.cross_entropy
    best, state, best_ep = np.inf, None, -1
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(tr), generator=g).split(bs):
            j = tr[ix.numpy()]
            loss = ce(net(X[j].to(device).float(), E[j].to(device)), bins[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        c = float(ce(logits(net, X, E, iv, device), bins[iv].to(device)))
        if c < best:
            best, best_ep, state = c, ep, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best, best_ep


@torch.no_grad()
def logits(net, X, E, idx, device, bs=2048):
    net.eval()
    return torch.cat([net(X[idx[i:i + bs]].to(device).float(), E[idx[i:i + bs]].to(device))
                      for i in range(0, len(idx), bs)])


def zscore(X, tr):
    mu, sd = X[tr].astype(np.float64).mean(0), X[tr].astype(np.float64).std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="reports/libero_oracle_bit.json")
    a = ap.parse_args()
    t0 = time.time()
    meta = load_meta()
    meta, _ = labels(meta.reset_index(drop=True))
    inner = inner_split(meta)
    tr = np.flatnonzero((meta.split == "train").values & ~inner)
    iv = np.flatnonzero(inner)
    va = np.flatnonzero((meta.split == "val").values)
    bins = torch.as_tensor(meta.bin.values.astype(np.int64))
    centers = torch.as_tensor(CENTERS, dtype=torch.float32)
    bit = torch.as_tensor(meta.y.values.astype(np.float32)[:, None])
    none = torch.zeros(len(meta), 0)
    print(f"[{time.time() - t0:.0f} s] labels: y = 1 on {meta.y.values[tr].mean():.3f} of fit, "
          f"{meta.y.values[va].mean():.3f} of val", flush=True)

    layer_ids = list(np.load(FEAT / "meta.npz")["layers"])
    V = np.load(FEAT / "v_layers.npy", mmap_mode="r")
    L5 = torch.from_numpy(zscore(np.asarray(V[:, layer_ids.index(5)], np.float32), tr).astype(np.float16))
    clock = torch.as_tensor(time_features(meta.frame_index.values.astype(np.float32)))
    B_h = torch.as_tensor(zscore(np.load(FEAT / SOURCES["B_h"]).astype(np.float32), tr))
    R_ft = torch.as_tensor(zscore(np.load(FEAT / SOURCES["R_ft"]).astype(np.float32), tr))
    arms = {
        "pi05 L5": (lambda: AttnPlus(L5.shape[1], L5.shape[2], N_BINS, 0), L5, none),
        "pi05 L5 + bit": (lambda: AttnPlus(L5.shape[1], L5.shape[2], N_BINS, 1), L5, bit),
        "bit + clock": (lambda: MLPPlus(clock.shape[1], N_BINS, 1), clock, bit),
        "B h": (lambda: MLPPlus(B_h.shape[1], N_BINS, 0), B_h, none),
        "B h + bit": (lambda: MLPPlus(B_h.shape[1], N_BINS, 1), B_h, bit),
        "ResNet ft": (lambda: MLPPlus(R_ft.shape[1], N_BINS, 0), R_ft, none),
        "ResNet ft + bit": (lambda: MLPPlus(R_ft.shape[1], N_BINS, 1), R_ft, bit),
    }
    rows, preds = [], []
    for seed in a.seeds:
        for name, (make, X, E) in arms.items():
            torch.manual_seed(seed)
            net, ice, ep = fit(make(), X, E, bins, tr, iv, seed, a.epochs, a.device)
            ev = (torch.softmax(logits(net, X, E, va, a.device), -1).cpu() @ centers).numpy()
            p = meta.iloc[va][["episode_index", "init_state", "frame_index", "success", "value"]].copy()
            p["ev"] = ev
            r = {"arm": name, "seed": seed, "inner_ce": ice, "best_epoch": ep, **score(p, "ev"),
                 "minutes": round((time.time() - t0) / 60, 1)}
            rows.append(r)
            preds.append(p.assign(arm=name, seed=seed))
            print(json.dumps(r), flush=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
    pd.concat(preds).to_parquet(Path(a.out).with_suffix(".val_predictions.parquet"))
    df = pd.DataFrame(rows)
    cols = ["inner_ce", "mae_succ>20s", "milestone", "change_signal_2s", "change_noise_2s", "change_snr_2s", "auc10"]
    pd.set_option("display.width", 250)
    print(df.groupby("arm", sort=False)[cols].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
