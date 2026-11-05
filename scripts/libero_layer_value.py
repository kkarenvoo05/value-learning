"""Per-layer value heads on frozen pi05_libero (docs/libero_plan.md log 2026-10-30, head-swap follow-up) [ours].

Question: B's h gives a noisy within-episode V (advantage SNR 0.08-0.09 vs the ResNet's 0.19-0.27). Is the noise
already in pi05's own representations, or added by the value expert's readout? Value heads straight on the cached
per-layer prefix values V (features/progress_18191428/v_layers.npy: layers 1 / 5 / 9 / 13 / 17, grid-4 pooled,
33 tokens x 256; layer 5 read "pot on the stove" best, time-matched AUC 0.987), no expert in between.
Head = vl.libero_probe.AttnProbe (the 3a grid probe and the progress probe's token head), value CE only (like the
ResNet V), 201 bins; trained on fit, early-stopped on the 5 inner train states; features z-scored on fit.
Scored as scripts/libero_head_swap.py (pass metrics + change SNR at 2 s).
Reading rule, set before the result: if some layer's SNR reaches the fine-tuned ResNet features' (~0.19 in the head
swap), the noise comes from the expert's readout; if every layer stays near B's h (~0.09), from the frozen backbone.
Usage: .venv/bin/python scripts/libero_layer_value.py [--layers 0 1 2 3 4] [--seeds 0 1 2]
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
from libero_head_swap import FEAT, load_meta, score  # noqa: E402


def fit(X, bins, tr, iv, seed, epochs, bs=256, device="cpu"):
    torch.manual_seed(seed)
    net = AttnProbe(X.shape[1], X.shape[2], N_BINS).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    ce = nn.functional.cross_entropy
    best, state, best_ep = np.inf, None, -1
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(tr), generator=g).split(bs):
            j = tr[ix.numpy()]
            loss = ce(net(X[j].to(device).float()), bins[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        c = float(ce(logits(net, X, iv, device=device), bins[iv].to(device)))
        if c < best:
            best, best_ep, state = c, ep, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best, best_ep


@torch.no_grad()
def logits(net, X, idx, bs=2048, device="cpu"):
    net.eval()
    return torch.cat([net(X[idx[i:i + bs]].to(device).float()) for i in range(0, len(idx), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="indices into meta 'layers'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="reports/libero_layer_value.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    meta = load_meta()
    layer_ids = np.load(FEAT / "meta.npz")["layers"]
    inner = inner_split(meta)
    tr = np.flatnonzero((meta.split == "train").values & ~inner)
    iv = np.flatnonzero(inner)
    va = np.flatnonzero((meta.split == "val").values)
    bins = torch.as_tensor(meta.bin.values.astype(np.int64))
    centers = torch.as_tensor(CENTERS, dtype=torch.float32)
    V = np.load(FEAT / "v_layers.npy", mmap_mode="r")
    rows, preds, t0 = [], [], time.time()

    def layer(li):
        X = np.empty(V.shape[:1] + V.shape[2:], np.float16)            # one layer at a time, fp16 (1.1 GB)
        for i in range(0, len(X), 8192):
            X[i:i + 8192] = V[i:i + 8192, li]
        s1, s2 = np.zeros(X.shape[1:]), np.zeros(X.shape[1:])
        for i in range(0, len(tr), 8192):                               # fit mean / sd in chunks
            c = X[tr[i:i + 8192]].astype(np.float64)
            s1 += c.sum(0)
            s2 += (c ** 2).sum(0)
        mu = s1 / len(tr)
        sd = np.sqrt(np.maximum(s2 / len(tr) - mu ** 2, 0)) + 1e-6
        for i in range(0, len(X), 8192):
            X[i:i + 8192] = ((X[i:i + 8192].astype(np.float32) - mu) / sd).astype(np.float16)
        return torch.from_numpy(X)

    for seed in a.seeds:                                                # every layer at the first seed first
        for li in a.layers:
            X = layer(li)
            net, ice, ep = fit(X, bins, tr, iv, seed, a.epochs, device=a.device)
            ev = (torch.softmax(logits(net, X, va, device=a.device), -1).cpu() @ centers).numpy()
            del X
            p = meta.iloc[va][["episode_index", "init_state", "frame_index", "success", "value"]].copy()
            p["ev"] = ev
            r = {"layer": int(layer_ids[li]), "seed": seed, "inner_ce": ice, "best_epoch": ep, **score(p, "ev"),
                 "minutes": round((time.time() - t0) / 60, 1)}
            rows.append(r)
            preds.append(p.assign(layer=int(layer_ids[li]), seed=seed))
            print(json.dumps(r), flush=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
    pd.concat(preds).to_parquet(Path(a.out).with_suffix(".val_predictions.parquet"))
    df = pd.DataFrame(rows)
    cols = ["inner_ce", "mae", "mae_succ>20s", "milestone", "change_signal_2s", "change_noise_2s", "change_snr_2s",
            "auc10"]
    pd.set_option("display.width", 250)
    print(df.groupby("layer")[cols].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
