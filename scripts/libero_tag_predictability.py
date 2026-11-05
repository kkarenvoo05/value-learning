"""Is the advantage tag predictable from the action, beyond the state? (2026-11-05, Karen: go)

Why: a tag-conditioned policy at its optimum is π(a | s, tag) ∝ π_β(a | s) · P(tag | s, a) / P(tag | s)
(Brandfonbrener et al. 2022, arXiv 2206.01079, Eq. 3). If P(tag | s, a) ≈ P(tag | s), conditioning can change nothing,
and the policy "defaults to the behavior" — what the arm C tag-usage check measured (reports/2026-11-04-tag-usage).
Test [ours]: on the 900 train rollouts (every 4th step, 104,049 frames), predict each label set's tag with an MLP from
  S   state: frozen ImageNet ResNet-18 pooled features of both cameras (2 × 512), robot state (8), time t/520,
      debounced pots on the stove, and the out-of-fold V of both value functions at t (so actions cannot help merely
      by revealing the task phase or the current value)
  S+A state + the recorded actions of the next 20 steps (the advantage's lookahead; 20 × 7, last action repeated
      past the end, with a padding mask)
  A   actions only (reference)
5 outer folds by episode (seed 0); inside each, 10% of the training episodes for early stopping (val log-loss,
patience 3). Out-of-fold P(tag) → AUC and log-loss. Headline: information the action adds,
  gain = (LL_S − LL_S+A) / H(tag)   (share of the tag's entropy explained by the action beyond the state),
with a 95% bootstrap CI over episodes (paired: same frames, same folds).
Label sets: current (arm C), expert current, RECAP-like 30% / 40% positive, value-binned calibration 40% positive
(raw and smoothed V; reports/2026-11-05-value-calibration), the outcome oracle, the stall flag (positive control:
defined from the next 20 steps of motion), and iid random labels at 70% (negative control).

Usage (one GPU): .venv/bin/python scripts/libero_tag_predictability.py --out reports/2026-11-05-tag-predictability
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
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import libero_advantage_examples as X  # noqa: E402
from vl.libero_resnet import table_meta  # noqa: E402
from vl.metrics import auc  # noqa: E402

N_ACT, FOLDS, SEED = 20, 5, 0
FEAT_CACHE = X.TABLE / "feat_r18imagenet_trainframes.npy"
CAL = Path("reports/2026-11-05-value-calibration/calibrated_labels.parquet")
RULES = Path("reports/2026-11-04-label-rules/rule_labels.parquet")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f} s] {m}", flush=True)


@torch.no_grad()
def image_features(rows: np.ndarray, device) -> np.ndarray:
    """(n, 1024) fp16 pooled ResNet-18 ImageNet features, agentview | wrist, for cache rows `rows`; cached."""
    if FEAT_CACHE.exists():
        f = np.load(FEAT_CACHE)
        assert len(f) == len(rows)
        return f
    arr = np.load(X.TABLE / "frames_every4.npy", mmap_mode="r")
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.to(device).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1).half()
    out = np.zeros((len(rows), 1024), np.float16)
    bs = 512
    for s in range(0, len(rows), bs):
        ix = rows[s:s + bs]
        order = np.argsort(ix)  # memmap reads in increasing order
        x = torch.from_numpy(np.ascontiguousarray(arr[ix[order]])).to(device)  # (b, 2, 3, 224, 224) uint8
        x = x.view(-1, 3, 224, 224).half().div_(255).sub_(mean).div_(std)
        f = net(x).view(len(ix), 1024).float().cpu().numpy().astype(np.float16)
        out[s + order] = f
        if s // bs % 40 == 0:
            log(f"  image features {s + len(ix)}/{len(rows)}")
    tmp = FEAT_CACHE.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.rename(FEAT_CACHE)
    return out


def state_action(d: pd.DataFrame, eps: pd.DataFrame):
    st, act, mask = np.zeros((len(d), 8), np.float32), np.zeros((len(d), N_ACT, 7), np.float32), np.zeros((len(d), N_ACT), np.float32)
    for e, idx in d.groupby("episode_index").indices.items():
        with np.load(eps.raw[e]) as z:
            s, a = z["state"], z["actions"]
        f = d.frame_index.to_numpy()[idx]
        st[idx] = s[f]
        j = f[:, None] + np.arange(N_ACT)[None]
        mask[idx] = (j < len(a)).astype(np.float32)
        act[idx] = a[np.minimum(j, len(a) - 1)]
    return st, act.reshape(len(d), -1), mask


class MLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(),
                               nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit_predict(Xtr, ytr, Xva, yva, Xte, device, epochs=40, bs=2048):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    t = lambda x: torch.as_tensor((x - mu) / sd, dtype=torch.float32, device=device)
    Xtr_, Xva_, Xte_ = t(Xtr), t(Xva), t(Xte)
    ytr_, yva_ = torch.as_tensor(ytr, dtype=torch.float32, device=device), torch.as_tensor(yva, dtype=torch.float32, device=device)
    torch.manual_seed(SEED)
    m = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr_), device=device)
        for s in range(0, len(perm), bs):
            i = perm[s:s + bs]
            opt.zero_grad(); lossf(m(Xtr_[i]), ytr_[i]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(lossf(m(Xva_), yva_))
        if v < best - 1e-4:
            best, state, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xte_)).cpu().numpy().astype(np.float64)


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ["episode_index", "frame_index"]
    eps = pd.read_csv(X.TABLE / "episodes.csv").set_index("episode_index")
    r = pd.read_parquet(X.LAB / "oof_resnet_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    e = pd.read_parquet(X.LAB / "oof_expert_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    o = pd.read_parquet(X.LAB / "oracle_outcome" / "labels.parquet").sort_values(key).reset_index(drop=True)
    cal = pd.read_parquet(CAL).sort_values(key).reset_index(drop=True)
    rl = pd.read_parquet(RULES).sort_values(key).reset_index(drop=True)
    for t_ in (e, o, cal, rl):
        assert (t_[key].values == r[key].values).all()
    meta = table_meta(X.TABLE, 4).reset_index(drop=True)
    meta["row"] = np.arange(len(meta))
    rows = r[key].merge(meta[key + ["row"]], on=key, how="left", validate="one_to_one").row.to_numpy()
    assert not np.isnan(rows.astype(float)).any()
    d = r[key].copy()
    log(f"{len(d)} train frames, {d.episode_index.nunique()} episodes; image features")
    img = image_features(rows, device).astype(np.float32)
    st, act, mask = state_action(d, eps)
    scal = np.stack([d.frame_index.to_numpy() / 520.0, cal.pots.to_numpy(), r.V.to_numpy(), e.V.to_numpy()], 1).astype(np.float32)
    feats = {"S": np.concatenate([img, st, scal], 1), "S+A": np.concatenate([img, st, scal, act, mask], 1),
             "A": np.concatenate([act, mask], 1)}
    rng = np.random.default_rng(SEED)
    labels = {
        "current (arm C: ResNet, 70% pos)": r.positive.to_numpy(),
        "expert, 70% pos": e.positive.to_numpy(),
        "ResNet, 40% pos": cal["raw·global·40%pos"].to_numpy(),
        "ResNet, 30% pos (RECAP pre-train)": cal["raw·global·30%pos"].to_numpy(),
        "value-binned, 40% pos": cal["raw·value·40%pos"].to_numpy(),
        "smooth 9 + value-binned, 40% pos": cal["smooth9·value·40%pos"].to_numpy(),
        "oracle outcome": o.positive.to_numpy(),
        "CONTROL+ stall flag": rl.stall.to_numpy().astype(int),
        "CONTROL- random, 70% pos": (rng.random(len(d)) < 0.7).astype(int),
    }
    ep_ids = d.episode_index.to_numpy()
    ueps = np.unique(ep_ids)
    fold_of = dict(zip(rng.permutation(ueps), np.arange(len(ueps)) % FOLDS))
    fold = np.array([fold_of[x] for x in ep_ids])
    res, oof = [], d[key].copy()
    for name, y in labels.items():
        y = y.astype(np.float64)
        P = {}
        for fs, Xf in feats.items():
            p = np.zeros(len(y))
            for k in range(FOLDS):
                te = fold == k
                tr_eps = np.unique(ep_ids[~te])
                va_eps = set(np.random.default_rng(SEED + k).choice(tr_eps, max(1, len(tr_eps) // 10), replace=False))
                va = ~te & np.isin(ep_ids, list(va_eps))
                tr = ~te & ~va
                p[te] = fit_predict(Xf[tr], y[tr], Xf[va], y[va], Xf[te], device)
            P[fs] = p
            oof[f"{name} | {fs}"] = p.astype(np.float32)
        base = y.mean()
        H = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
        L = {fs: ll(P[fs], y) for fs in P}
        per_ep = pd.DataFrame({"ep": ep_ids, "dS_SA": L["S"] - L["S+A"]}).groupby("ep").dS_SA.agg(["sum", "size"])
        bs = []
        b_rng = np.random.default_rng(SEED)
        for _ in range(2000):
            i = b_rng.integers(0, len(per_ep), len(per_ep))
            bs.append(per_ep["sum"].values[i].sum() / per_ep["size"].values[i].sum() / H)
        row = {"labels": name, "pos_share": float(base), "entropy_nats": H,
               **{f"auc_{fs}": float(auc(y.astype(int), P[fs])) for fs in P},
               **{f"ll_{fs}": float(L[fs].mean()) for fs in P},
               "gain_action_over_state": float((L["S"].mean() - L["S+A"].mean()) / H),
               "gain_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               "state_explains": float((H - L["S"].mean()) / H), "state_action_explains": float((H - L["S+A"].mean()) / H)}
        res.append(row)
        log(f"{name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != 'labels'}))
    tab = pd.DataFrame(res)
    tab.to_csv(out / "predictability_table.csv", index=False)
    oof.to_parquet(out / "oof_predictions.parquet")
    (out / "summary.json").write_text(json.dumps({"n_frames": len(d), "folds": FOLDS, "n_actions": N_ACT, "results": res}, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(tab[["labels", "pos_share", "auc_S", "auc_S+A", "auc_A", "state_explains", "state_action_explains",
               "gain_action_over_state", "gain_ci95"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
