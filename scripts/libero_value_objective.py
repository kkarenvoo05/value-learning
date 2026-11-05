"""Value objective test on frozen pi05 L5: Monte Carlo CE vs consistency vs bootstrapped targets [ours].

Background (docs/libero_plan.md, 2026-10-31): ~98% of arm B's within-episode 2 s change in E[V] is P(success) moving
between the success and failure humps, so V's noise is a flickering per-frame outcome classifier. Hypothesis: within a
layout the outcome is mostly policy noise, so each frame's Monte Carlo label is a near-random episode draw that a rich
feature set memorises (every 3b arm overfits in 1-1.5 passes; within-layout AUC is at chance). The four head-side
fixes (scripts/libero_shortcut_fixes.py) change what the head sees, not what it regresses onto. Here only the target
or loss changes; input, head, optimiser, selection and scoring are those of that script's "base" arm (layer-5 prefix
values, 33 grid tokens, AttnProbe, 10 epochs, early stopping on inner Monte Carlo value CE).
  base           Monte Carlo value CE (paired control; = shortcut_fixes "base")
  consist 10/100 + lam * mean (E[V](t + 2 s) - E[V](t) - 20/1040)^2 over within-episode pairs (the target's own drift
                 is exactly +1/1040 per step for both outcomes, checked on the data)
  oof            pure distillation: two-hot of an out-of-fold Monte Carlo V at the same frame (diagnostic: all outcome
                 noise replaced by what models that never saw the episode predict; no new information)
  td 2 s / 10 s  n-step bootstrapped target  y_t = -4n/1040 + V_oof(t + n grid frames), the Monte Carlo return where
                 the episode ends inside n; two-hot on the 201 bins
V_oof: 5 folds over the 25 fit init states, each fold's frames predicted by a base model trained on the other 20
(same recipe, same seed), so a frame's bootstrap value never comes from a model that saw its episode.
Comparator for the consistency arms: base smoothed post hoc, centred +-1.2 s (scripts/libero_smoothing_check.py),
since a training-time penalty that only smooths should not count as more than smoothing.
Reading rule, set before the result: an arm "works" if its change SNR at 2 s beats base by > 2 seed sd AND its
milestone excess is positive on 3/3 seeds; a consistency arm must also beat smoothed base on SNR. Bars to compare:
progress-routed head 0.168 / +0.077, fine-tuned ResNet ~0.19 / +0.09. Late-success MAE (> 20 s) is reported, not
gated: bootstrapped targets are biased toward V_oof by construction.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/libero_value_objective.py [--seeds 0 1 2] [--select ce|mae]
Added after the first run (--select ce): every soft-target arm stopped at epoch 0 under the CE rule, so --select mae
(inner MAE of E[V] against the Monte Carlo target, the same rule for every arm, V_oof folds included) is the re-run.
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
from vl.libero_probe import CENTERS, inner_split  # noqa: E402
from libero_head_swap import FEAT, load_meta, score  # noqa: E402
from libero_oracle_bit import zscore  # noqa: E402
from libero_shortcut_fixes import Probe, logits  # noqa: E402
from libero_smoothing_check import smooth  # noqa: E402

NORM = 1040.0  # horizon 520 + failure penalty 520 (configs/libero_moka.yaml)
GRID = 4  # policy steps between cached frames
LAG = 5  # grid frames = 2 s, the change-SNR lag
ARMS = ["base", "consist 10", "consist 100", "oof", "td 2 s", "td 10 s"]


def partner(meta: pd.DataFrame, n: int) -> np.ndarray:
    """Row index of the same episode's frame n grid frames later, -1 if past its end."""
    key = pd.Series(np.arange(len(meta)), index=pd.MultiIndex.from_arrays([meta.episode_index, meta.frame_index]))
    nxt = pd.MultiIndex.from_arrays([meta.episode_index, meta.frame_index + GRID * n])
    return key.reindex(nxt).fillna(-1).to_numpy(np.int64)


def two_hot(y: np.ndarray) -> torch.Tensor:
    y = np.clip(y, -1.0, 0.0)
    pos = (y + 1.0) * (len(CENTERS) - 1)
    lo = np.minimum(np.floor(pos).astype(int), len(CENTERS) - 2)
    w = pos - lo
    t = np.zeros((len(y), len(CENTERS)), np.float32)
    t[np.arange(len(y)), lo] = 1 - w
    t[np.arange(len(y)), lo + 1] += w
    return torch.from_numpy(t)


def fit(X, target, bins, tr, iv, seed, device, lam=0.0, nxt=None, epochs=10, bs=256, sel_value=None):
    """Soft-target CE (+ optional consistency penalty). Selection on inner Monte Carlo value CE for every arm, or, with
    `sel_value` (the Monte Carlo targets), on inner MAE of E[V]: CE on one-hot bimodal labels punishes any unimodal
    soft-target model, so the CE rule stops those arms at epoch 0."""
    torch.manual_seed(seed)
    net = Probe(X.shape[1], X.shape[2]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    c = torch.as_tensor(CENTERS, dtype=torch.float32, device=device)
    E0 = torch.zeros(len(X), 0)
    best, state, best_ep = np.inf, None, -1
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(tr), generator=g).split(bs):
            j = tr[ix.numpy()]
            lv, _ = net(X[j].to(device).float(), E0[j].to(device))
            loss = -(target[j].to(device) * torch.log_softmax(lv, -1)).sum(-1).mean()
            if lam > 0:
                k = nxt[j]
                has = k >= 0
                if has.any():
                    lp, _ = net(X[k[has]].to(device).float(), E0[k[has]].to(device))
                    dv = torch.softmax(lp, -1) @ c - torch.softmax(lv[torch.as_tensor(has, device=device)], -1) @ c
                    loss = loss + lam * ((dv - GRID * LAG / NORM) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        li = logits(net, X, E0, iv, device)
        ce = float(nn.functional.cross_entropy(li, bins[iv].to(device))) if sel_value is None else \
            float((torch.softmax(li, -1) @ c - sel_value[iv].to(device)).abs().mean())
        if ce < best:
            best, best_ep, state = ce, ep, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best, best_ep


@torch.no_grad()
def probs(net, X, idx, device):
    return torch.softmax(logits(net, X, torch.zeros(len(X), 0), idx, device), -1).cpu().numpy()


def hump_share(p: pd.DataFrame, P: np.ndarray) -> float:
    """Share of the 2 s change-in-E[V] variance (interior frames) carried by P(success) x hump gap."""
    hi = CENTERS > -0.5
    ps = P[:, hi].sum(1)
    vs = (P[:, hi] * CENTERS[hi]).sum(1) / np.maximum(ps, 1e-9)
    vf = (P[:, ~hi] * CENTERS[~hi]).sum(1) / np.maximum(1 - ps, 1e-9)
    q = p.assign(V=P @ CENTERS, ps=ps, gap=vs - vf)
    g = q.groupby("episode_index")
    dv, dp = g.V.diff(LAG), g.ps.diff(LAG)
    m = dv.notna() & (q.frame_index <= 340)
    return float((dp * q.gap)[m].var() / dv[m].var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--select", choices=["ce", "mae"], default="ce")
    ap.add_argument("--out", default="reports/libero_value_objective.json")
    a = ap.parse_args()
    t0 = time.time()
    meta = load_meta().reset_index(drop=True)
    inner = inner_split(meta)
    tr = np.flatnonzero((meta.split == "train").values & ~inner)
    iv = np.flatnonzero(inner)
    order = np.lexsort((meta.frame_index.values[(meta.split == "val").values],
                        meta.episode_index.values[(meta.split == "val").values]))
    va = np.flatnonzero((meta.split == "val").values)[order]  # sorted by episode, frame for the change metrics
    bins = torch.as_tensor(meta.bin.values.astype(np.int64))
    sel = torch.as_tensor(meta.value.values, dtype=torch.float32) if a.select == "mae" else None
    onehot = nn.functional.one_hot(bins, len(CENTERS)).float()
    nxt = {n: partner(meta, n) for n in (LAG, 25)}
    fit_states = np.sort(meta.init_state.iloc[tr].unique())
    layer_ids = list(np.load(FEAT / "meta.npz")["layers"])
    V = np.load(FEAT / "v_layers.npy", mmap_mode="r")
    X = torch.from_numpy(zscore(np.asarray(V[:, layer_ids.index(5)], np.float32), tr).astype(np.float16))
    print(f"[{time.time() - t0:.0f} s] features {tuple(X.shape)} on {a.device}; fit states {len(fit_states)}", flush=True)
    rows, preds = [], []
    for seed in a.seeds:
        v_oof = None
        if any(x in ("oof", "td 2 s", "td 10 s") for x in a.arms):
            v_oof = np.full(len(meta), np.nan)
            folds = np.array_split(np.random.default_rng(seed).permutation(fit_states), 5)
            for f in folds:
                held = np.isin(meta.init_state.values[tr], f)
                net, _, _ = fit(X, onehot, bins, tr[~held], iv, seed, a.device, sel_value=sel)
                v_oof[tr[held]] = probs(net, X, tr[held], a.device) @ CENTERS
            assert np.isfinite(v_oof[tr]).all()
            print(f"[{time.time() - t0:.0f} s] seed {seed} V_oof done; corr with MC target "
                  f"{np.corrcoef(v_oof[tr], meta.value.values[tr])[0, 1]:.3f}", flush=True)
        for name in a.arms:
            lam, target = 0.0, onehot
            if name.startswith("consist"):
                lam = float(name.split()[1])
            elif name == "oof":
                y = meta.value.values.copy()
                y[tr] = v_oof[tr]
                target = two_hot(y)
            elif name.startswith("td"):
                n = LAG if name == "td 2 s" else 25
                y = meta.value.values.copy()
                k = nxt[n][tr]
                y[tr] = np.where(k >= 0, -GRID * n / NORM + v_oof[np.maximum(k, 0)], meta.value.values[tr])
                target = two_hot(y)
            net, ice, ep = fit(X, target, bins, tr, iv, seed, a.device, lam=lam, nxt=nxt[LAG], sel_value=sel)
            P = probs(net, X, va, a.device)
            p = meta.iloc[va][["episode_index", "init_state", "frame_index", "success", "value"]].reset_index(drop=True)
            p["ev"] = P @ CENTERS
            r = {"arm": name, "seed": seed, "select": a.select, "inner_sel": ice, "best_epoch": ep, **score(p, "ev"),
                 "hump_share": hump_share(p, P), "minutes": round((time.time() - t0) / 60, 1)}
            rows.append(r)
            preds.append(p.assign(arm=name, seed=seed))
            print(json.dumps(r), flush=True)
            if name == "base":  # post-hoc smoothing comparator for the consistency arms
                ps_ = p.assign(ev=smooth(p, "ev", 7, "centred"))
                r2 = {"arm": "base, smoothed +-1.2 s", "seed": seed, **score(ps_, "ev")}
                rows.append(r2)
                print(json.dumps(r2), flush=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
    pd.concat(preds).to_parquet(Path(a.out).with_suffix(".val_predictions.parquet"))
    df = pd.DataFrame(rows)
    cols = ["change_snr_2s", "change_signal_2s", "change_noise_2s", "milestone", "mae_succ>20s", "mae", "auc10",
            "hump_share"]
    pd.set_option("display.width", 250)
    print(df.groupby("arm", sort=False)[cols].agg(["mean", "std"]).round(3).to_string())
    print(df.groupby("arm", sort=False).milestone.apply(lambda m: f"{(m > 0).sum()}/{len(m)}").to_string())


if __name__ == "__main__":
    main()
