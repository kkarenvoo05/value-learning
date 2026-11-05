"""Stage 3a (docs/libero_plan.md): do frozen pi05_libero features carry value? Linear / MLP probes vs a clock.

Usage:
    .venv/bin/python -m vl.libero_probe --features <features.npz> [--pooling all agentview wrist text concat]
    .venv/bin/python -m vl.libero_probe --features <features.npz> --overfit      # Check C on a tiny subset

Features: scripts/pi05_libero_features.py (mean-pooled final-layer prefix tokens, every 4th frame). Targets: the
201-bin value targets of `python -m vl.libero_targets` (frame_targets.parquet next to the LeRobot data).
Heads [ours]: linear (softmax over 201 bins) and MLP (one hidden layer of 512, GELU, dropout 0.1), cross-entropy
on the target bin, AdamW (lr 1e-3, weight decay 1e-2), batch 512, features z-scored with train statistics.
No tuning on val: early stopping (best inner-val CE over at most 60 epochs) uses 5 of the 30 train init states held
out inside train (np.random.default_rng(0) choice). Reported on the 10 value-val init states:
  ce        mean per-frame cross-entropy (nats) of the predicted bin distribution
  mae       |E[V] - v| per frame, E[V] the mean of the predicted distribution
  auc_<s>s  success-vs-failure ROC AUC of E[V] at elapsed time s (frames within +-0.5 s, one per episode)
Each head is also reported with one softmax temperature fitted on the inner train states (`+T`): the heads stop
improving after a few epochs and are over-confident, so their raw CE is worse than the clock's.
`/outcome+clock` [ours]: the head predicts only P(success | o) (2-way CE on the episode outcome, same early
stopping); the bin distribution is P(success | o) H_success[t] + P(failure | o) H_failure[t], with H the train
histograms of bins at frame t per outcome. With a constant P(success) this is `clock_by_outcome`, so the two
differ only in what the images say about the outcome. (All failures are 520-step timeouts, so H_failure[t] is
nearly one bin: a model that knows t exactly gets a large CE advantage that images alone cannot match.)
Control: `clock`, the train histogram of target bins at each frame index (add-one smoothed), i.e. what knowing
only the elapsed time gives. Writes <out>/probe_results.json and per-frame val predictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vl.metrics import auc

N_BINS, FPS = 201, 10
CENTERS = np.linspace(-1, 0, N_BINS)
POOLINGS = ["all", "agentview", "wrist", "text"]
AUC_TIMES_S = (5, 10, 20, 30)


def load(features: Path, dataset: Path) -> tuple[np.ndarray, pd.DataFrame]:
    z = np.load(features, allow_pickle=False)
    meta = pd.DataFrame({k: z[k] for k in ("split", "init_state", "episode", "frame_index", "T", "success")})
    eps = pd.read_csv(dataset / "episodes.csv")[["episode_index", "split", "source_episode"]]
    meta = meta.merge(eps, left_on=["split", "episode"], right_on=["split", "source_episode"], how="left")
    assert meta.episode_index.notna().all()
    ft = pd.read_parquet(dataset / "frame_targets.parquet", columns=["episode_index", "frame_index", "value", "bin"])
    meta = meta.merge(ft, on=["episode_index", "frame_index"], how="left", validate="one_to_one")
    assert meta.bin.notna().all() and (meta.bin >= 0).all()
    return z["feats"], meta


def inner_split(meta: pd.DataFrame, k: int = 5) -> np.ndarray:
    states = np.sort(meta[meta.split == "train"].init_state.unique())
    states = states[states >= 0]  # fresh-layout episodes (init_state -1) are never held out
    held = np.random.default_rng(0).choice(states, size=min(k, len(states) - 1), replace=False)  # smoke sets
    return (meta.split == "train").values & meta.init_state.isin(held).values


def metrics(probs: np.ndarray, meta: pd.DataFrame) -> dict:
    b = meta.bin.values.astype(int)
    ev = probs @ CENTERS
    out = {"ce": float(-np.log(np.clip(probs[np.arange(len(b)), b], 1e-12, None)).mean()),
           "mae": float(np.abs(ev - meta.value.values).mean()), "frames": int(len(b)),
           "episodes": int(meta.episode_index.nunique())}
    d = meta.assign(ev=ev)
    for s in AUC_TIMES_S:
        w = d[(d.frame_index - s * FPS).abs() <= FPS // 2]
        one = w.loc[(w.frame_index - s * FPS).abs().groupby(w.episode_index).idxmin()]
        out[f"auc_{s}s"] = float(auc(one.success.values, one.ev.values))
    return out


def clock_probs(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    T = int(max(train.frame_index.max(), query.frame_index.max())) + 1
    h = np.ones((T, N_BINS))
    np.add.at(h, (train.frame_index.values, train.bin.values.astype(int)), 1)
    h /= h.sum(1, keepdims=True)
    return h[query.frame_index.values]


class AttnProbe(nn.Module):
    """Grid test [ours]: tokens (B, n, w) -> Linear(w, d) + LayerNorm + learned position embedding -> one learned
    query attends over them (4 heads) -> LayerNorm -> MLP (d, GELU, dropout 0.1) -> n_out."""

    def __init__(self, n_tok: int, w: int, n_out: int, d: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(w, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(n_tok, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, n_out))

    def forward(self, x):
        t = self.proj(x) + self.pos
        q, _ = self.attn(self.query.expand(len(x), -1, -1), t, t)
        return self.out(q[:, 0])


def make_head(kind: str, shape, n_out: int = N_BINS) -> nn.Module:
    if kind == "attn":
        return AttnProbe(shape[0], shape[1], n_out)
    d = shape[0]
    if kind == "linear":
        return nn.Linear(d, n_out)
    return nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, n_out))


def clock_by_outcome(train: pd.DataFrame, T: int, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Train histograms of target bins at each frame index, separately for successes and failures (alpha-smoothed)."""
    h = {}
    for ok in (1, 0):
        d = train[train.success == ok]
        c = np.full((T, N_BINS), alpha)
        np.add.at(c, (d.frame_index.values, d.bin.values.astype(int)), 1)
        h[ok] = c / c.sum(1, keepdims=True)
    return h[1], h[0]


def outcome_clock_probs(p_success: np.ndarray, frames: np.ndarray, hs: np.ndarray, hf: np.ndarray) -> np.ndarray:
    """P(bin | o, t) = P(success | o) H_success[t] + P(failure | o) H_failure[t]."""
    return p_success[:, None] * hs[frames] + (1 - p_success[:, None]) * hf[frames]


def fit(kind, Xtr, ytr, Xin, yin, epochs=60, seed=0, device="cpu", log=None, n_out=N_BINS):
    torch.manual_seed(seed)
    net = make_head(kind, Xtr.shape[1:], n_out).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    Xtr, ytr = torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr, device=device)
    if Xin is not None:
        Xin, yin = torch.as_tensor(Xin, device=device), torch.as_tensor(yin, device=device)
    best, best_state, g = np.inf, None, torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(Xtr), generator=g).split(512):
            ix = ix.to(device)
            loss = nn.functional.cross_entropy(net(Xtr[ix].float()), ytr[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xin is None or len(Xin) == 0:
            continue
        ce = float(nn.functional.cross_entropy(logits_of(net, Xin, device), yin))
        if ce < best:
            best, best_ep = ce, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if log is not None and ep % 10 == 0:
            log(f"    epoch {ep}: train loss {float(loss):.3f}, inner-val CE {ce:.3f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, (best if best_state is not None else None), (best_ep if best_state is not None else epochs - 1)


def logits_of(net, X, device="cpu", bs=2048) -> torch.Tensor:
    """Logits in batches (inputs may be fp16; cast per batch)."""
    net.eval()
    X = torch.as_tensor(X, device=device)
    with torch.no_grad():
        return torch.cat([net(X[i:i + bs].float()) for i in range(0, len(X), bs)])


def predict(net, X, device="cpu", temperature=1.0):
    return torch.softmax(logits_of(net, X, device) / temperature, -1).cpu().numpy()


def fit_temperature(logits: np.ndarray, bins: np.ndarray) -> float:
    """Single softmax temperature minimizing CE on held-out frames (grid 0.5-20, log-spaced) [ours]."""
    grid = np.exp(np.linspace(np.log(0.5), np.log(20), 60))
    lg = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(bins, dtype=torch.long)
    ce = [float(nn.functional.cross_entropy(lg / t, y)) for t in grid]
    return float(grid[int(np.argmin(ce))])


def coarsen_grid(F: np.ndarray, g: int) -> np.ndarray:
    """Grid test [ours]: (n, 2 G^2 + 1, w) block means from scripts/pi05_libero_features.py --grid G (per image,
    row-major G x G cells, then the text mean) -> the same layout at g x g (g divides G). Cells are equal-sized, so
    a mean of cell means is the exact g x g block mean; g = 1 gives the per-image mean pools."""
    G = int(round(np.sqrt((F.shape[1] - 1) / 2)))
    assert 2 * G * G + 1 == F.shape[1] and G % g == 0, (F.shape, g)
    if g == G:
        return F
    n, w, k = len(F), F.shape[2], G // g
    imgs = [F[:, s * G * G:(s + 1) * G * G].astype(np.float32).reshape(n, g, k, g, k, w).mean((2, 4))
            .reshape(n, g * g, w).astype(F.dtype) for s in (0, 1)]
    return np.concatenate(imgs + [F[:, -1:]], 1)


def features_for(F: np.ndarray, pooling: str) -> np.ndarray:
    if pooling.startswith("grid"):                     # (n, tokens, w) fp16, z-scored per channel in main
        return coarsen_grid(F, int(pooling[4:]))
    if pooling == "concat":
        return F.reshape(len(F), -1).astype(np.float32)
    return F[:, POOLINGS.index(pooling)].astype(np.float32)


def with_history(X: np.ndarray, meta: pd.DataFrame, lags: list[int]) -> np.ndarray:
    """[X_t, has_1 X_(t-l1), ..., has_k X_(t-lk), has_1..has_k] [ours]; X_(t-l) of the same episode (lags must be
    multiples of the feature stride), zeros (the train mean after z-scoring) and has = 0 before the episode start.
    Grid tokens (n, tokens, w): the past tokens are appended along the token axis, then one flag token per lag
    (has in every channel), so the attention head tells past from present by its position embedding."""
    key = pd.Index(meta.episode_index.values.astype(np.int64) * 10_000 + meta.frame_index.values)
    parts, flags = [X], []
    for lag in lags:
        src = key.get_indexer(meta.episode_index.values.astype(np.int64) * 10_000 + meta.frame_index.values - lag)
        has = (src >= 0) & (meta.frame_index.values >= lag)
        if ((meta.frame_index.values >= lag) & (src < 0)).any():
            raise ValueError(f"lag {lag} is not on the feature grid")
        past = X[np.maximum(src, 0)]
        past[~has] = 0
        parts.append(past)
        flags.append(has.astype(X.dtype)[:, None])
    if X.ndim == 3:
        flags = [np.broadcast_to(h[:, :, None], (len(X), 1, X.shape[2])) for h in flags]
    return np.concatenate(parts + flags, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", default="/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2")
    ap.add_argument("--pooling", nargs="+", default=["all", "agentview", "wrist", "text", "concat"])
    ap.add_argument("--heads", nargs="+", default=["linear", "mlp"], help="linear, mlp; attn for grid features")
    ap.add_argument("--out", default=None)
    ap.add_argument("--history", type=int, nargs="+", default=None, metavar="FRAMES",
                    help="append the (z-scored) features from these many frames earlier, each with a has-past flag")
    ap.add_argument("--seed", type=int, default=0, help="head init and batch order (the inner split stays fixed)")
    ap.add_argument("--overfit", action="store_true", help="Check C: fit 2 train episodes, report train CE")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    F, meta = load(Path(a.features), Path(a.dataset))
    out = Path(a.out or Path(a.features).with_suffix(""))
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)

    if a.overfit:
        eps = meta[meta.split == "train"].episode_index.unique()[:2]
        m = meta.episode_index.isin(eps).values
        res = {}
        for kind in a.heads:
            X = features_for(F[m], "all")
            mu, sd = X.mean(0), X.std(0) + 1e-6
            X = (X - mu) / sd
            net, _, _ = fit(kind, X, meta.bin.values[m].astype(np.int64), None, None, epochs=300, device=device)
            res[kind] = metrics(predict(net, X, device), meta[m])
            log(f"overfit {kind}: {res[kind]}")
        (out / "overfit.json").write_text(json.dumps(res, indent=2))
        return

    inner = inner_split(meta)
    tr = (meta.split == "train").values & ~inner
    va = (meta.split == "val").values
    log(f"frames: train {tr.sum()} (fit) + {inner.sum()} (inner val, early stopping), val {va.sum()}; "
        f"device {device}")
    res = {"setup": {"features": str(a.features), "seed": a.seed, "history": a.history, "fit_frames": int(tr.sum()), "inner_frames": int(inner.sum()),
                     "val_frames": int(va.sum()),
                     "inner_states": sorted(meta[inner].init_state.unique().tolist())}}
    res["clock"] = metrics(clock_probs(meta[tr | inner], meta[va]), meta[va])
    log(f"clock: {res['clock']}")
    # outcome x clock: the same timing knowledge, outcome from a constant prior (= a clock) or from the images
    T_max = int(meta.frame_index.max()) + 1
    hs, hf = clock_by_outcome(meta[tr | inner], T_max)
    fr = meta.frame_index.values
    prior = float(meta[tr | inner].groupby("episode_index").success.first().mean())
    res["clock_by_outcome"] = metrics(outcome_clock_probs(np.full(va.sum(), prior), fr[va], hs, hf), meta[va])
    log(f"clock by outcome (constant P(success) = {prior:.3f}): {res['clock_by_outcome']}")
    preds = meta[va][["episode_index", "init_state", "frame_index", "T", "success", "value", "bin"]].copy()
    preds["ev_clock"] = clock_probs(meta[tr | inner], meta[va]) @ CENTERS
    for pooling in a.pooling:
        X = features_for(F, pooling)
        if X.ndim == 3:   # grid tokens: per-channel statistics over train rows and tokens, kept in fp16
            Xt = X[tr].astype(np.float32).reshape(-1, X.shape[2])
            mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
            del Xt
            X = np.concatenate([((X[i:i + 4096].astype(np.float32) - mu) / sd).astype(np.float16)
                                for i in range(0, len(X), 4096)])
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            X = (X - mu) / sd
        if a.history:
            X = with_history(X, meta, a.history)
            log(f"history lags {a.history} frames -> features {X.shape}")
        for kind in a.heads:
            log(f"{pooling} / {kind}")
            net, inner_ce, best_ep = fit(kind, X[tr], meta.bin.values[tr].astype(np.int64), X[inner],
                                         meta.bin.values[inner].astype(np.int64), seed=a.seed, device=device, log=log)
            p = predict(net, X[va], device)
            r = metrics(p, meta[va])
            r.update(inner_ce=inner_ce, best_epoch=best_ep)
            res[f"{pooling}/{kind}"] = r
            preds[f"ev_{pooling}_{kind}"] = p @ CENTERS
            log(f"  -> {r}")
            # temperature fitted on the inner train states (never on val), applied to val
            lg_in = logits_of(net, X[inner], device).cpu().numpy()
            temp = fit_temperature(lg_in, meta.bin.values[inner])
            pt = predict(net, X[va], device, temp)
            rt = metrics(pt, meta[va])
            rt.update(temperature=temp)
            res[f"{pooling}/{kind}+T"] = rt
            preds[f"ev_{pooling}_{kind}_T"] = pt @ CENTERS
            log(f"  +T {temp:.2f} -> {rt}")
            # outcome head: P(success | o) from the same features, timing from the per-outcome clock
            ys = meta.success.values.astype(np.int64)
            onet, _, oep = fit(kind, X[tr], ys[tr], X[inner], ys[inner], seed=a.seed, device=device, n_out=2)
            ps = predict(onet, X[va], device)[:, 1]
            ro = metrics(outcome_clock_probs(ps, fr[va], hs, hf), meta[va])
            ro.update(best_epoch=oep)
            res[f"{pooling}/{kind}/outcome+clock"] = ro
            preds[f"ps_{pooling}_{kind}"] = ps
            log(f"  outcome+clock -> {ro}")
    (out / "probe_results.json").write_text(json.dumps(res, indent=2))
    preds.to_parquet(out / "val_predictions.parquet")
    rows = [{"model": k, **{m: v.get(m) for m in ["ce", "mae"] + [f"auc_{s}s" for s in AUC_TIMES_S]}}
            for k, v in res.items() if k != "setup"]
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
    log(f"wrote {out / 'probe_results.json'}")


if __name__ == "__main__":
    main()
