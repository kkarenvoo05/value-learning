"""Three cheap fixes for the layout shortcut, on frozen pi05 features (docs/libero_plan.md log 2026-10-30) [ours].

Background: the oracle test (scripts/libero_oracle_bit.py) showed a value head next to image features ignores even a
perfect "pot on the stove" bit (true bit + clock alone: change SNR 0.22, milestone +0.33; pi05 L5 + bit: 0.10), so
per-frame value CE fits the scene/episode identity instead of progress. Each arm here changes only the training of the
head on the same input as the oracle's "pi05 L5" row (layer-5 prefix values, 33 grid tokens, z-scored on fit;
vl.libero_probe.AttnProbe layout), value CE, 201 bins, 3 seeds, fit / inner early stopping on value CE / val as
scripts/libero_head_swap.py and scored with its `score` (pass metrics + change SNR 2 s). No arm sees the true bit at
test time except the diagnostic one.
  base             the oracle's pi05 L5 head, re-run for a paired comparison
  aux              + a 2nd output predicting the bit (BCE, weight 1) from the same readout
  aux -> value     aux, and the value MLP also reads the predicted P(pot on stove)
  tokdrop 0.3/0.6  each of the 33 tokens hidden from the attention with prob p during training (never in eval)
  bottleneck 16/4  the readout squeezed to k dims before the value MLP
  bottleneck 4 + bit   diagnostic: does a squeezed head use the true bit that the wide head ignored?
Reading rule, set before the result: a fix "works" if its SNR beats base by more than 2 seed sd and its milestone
excess is positive on 3/3 seeds; the bar to aim at is the fine-tuned ResNet (SNR ~0.19, milestone ~+0.09).
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/libero_shortcut_fixes.py [--seeds 0 1 2]
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
from libero_head_swap import FEAT, load_meta, score  # noqa: E402
from libero_oracle_bit import zscore  # noqa: E402


class Probe(nn.Module):
    """AttnProbe (Linear + LayerNorm + position embedding, one query, 4-head attention) with optional token dropout,
    readout bottleneck, auxiliary bit output and extra inputs to the value MLP."""

    def __init__(self, n_tok, w, d=256, k=None, tokdrop=0.0, aux=False, aux_into_value=False, n_extra=0):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(w, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(n_tok, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.tokdrop, self.aux_into_value = tokdrop, aux_into_value
        self.squeeze = nn.Linear(d, k) if k else nn.Identity()
        r = k or d
        self.aux = nn.Linear(r, 1) if aux else None
        v_in = r + n_extra + (1 if aux_into_value else 0)
        self.out = nn.Sequential(nn.LayerNorm(v_in), nn.Linear(v_in, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, N_BINS))

    def forward(self, x, e):
        t = self.proj(x) + self.pos
        mask = None
        if self.training and self.tokdrop > 0:
            mask = torch.rand(t.shape[:2], device=t.device) < self.tokdrop       # True = hidden
            mask[mask.all(1), 0] = False                                           # keep >= 1 token per row
        q, _ = self.attn(self.query.expand(len(x), -1, -1), t, t, key_padding_mask=mask)
        r = self.squeeze(q[:, 0])
        a = self.aux(r) if self.aux is not None else None
        z = [r, e] + ([torch.sigmoid(a)] if self.aux_into_value else [])
        return self.out(torch.cat(z, -1)), a


ARMS = {
    "base": dict(),
    "aux": dict(aux=True),
    "aux -> value": dict(aux=True, aux_into_value=True),
    "tokdrop 0.3": dict(tokdrop=0.3),
    "tokdrop 0.6": dict(tokdrop=0.6),
    "bottleneck 16": dict(k=16),
    "bottleneck 4": dict(k=4),
    "bottleneck 4 + bit": dict(k=4, n_extra=1),
}


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
        c = float(ce(logits(net, X, E, iv, device), bins[iv].to(device)))   # selection on value CE, as every arm
        if c < best:
            best, best_ep, state = c, ep, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best, best_ep


@torch.no_grad()
def logits(net, X, E, idx, device, bs=2048):
    net.eval()
    return torch.cat([net(X[idx[i:i + bs]].to(device).float(), E[idx[i:i + bs]].to(device))[0]
                      for i in range(0, len(idx), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="reports/libero_shortcut_fixes.json")
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
    layer_ids = list(np.load(FEAT / "meta.npz")["layers"])
    V = np.load(FEAT / "v_layers.npy", mmap_mode="r")
    X = torch.from_numpy(zscore(np.asarray(V[:, layer_ids.index(5)], np.float32), tr).astype(np.float16))
    print(f"[{time.time() - t0:.0f} s] features {tuple(X.shape)} on {a.device}", flush=True)
    rows, preds = [], []
    for seed in a.seeds:
        for name in a.arms:
            cfg = ARMS[name]
            E = y[:, None] if cfg.get("n_extra") else torch.zeros(len(meta), 0)
            torch.manual_seed(seed)
            net, ice, ep = fit(Probe(X.shape[1], X.shape[2], **cfg), X, E, bins, y, tr, iv, seed, a.epochs, a.device)
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
    print(df.groupby("arm", sort=False).milestone.apply(lambda m: f"{(m > 0).sum()}/{len(m)}").to_string())


if __name__ == "__main__":
    main()
