"""Progress curves for eval episodes from their videos (Karen 2026-11-12, idea 1: "grouping the 25% failures").

    .venv/bin/python scripts/libero_failure_curves.py --out reports/2026-11-12-failure-taxonomy

Evals save only the agentview video (no raw, no wrist, no proprio), so the value models here see what a video shows:
s = frozen ImageNet ResNet-18 agentview features (the first 512 of the cached agentview | wrist features, same
preprocessing) + t / 520 [ours]. Video frame k = raw frame k (checked 2026-11-12 on stage2 val ep 0: mean abs pixel
difference 1.9, compression only).
Value models, all in success-probability units (reward 1 at a successful end, else 0, gamma 1, n = 20 transitions):
  iql_p      IQL tau 0.9 (libero_iql.train_iql with that reward; tau 0.9 tracked V* best, reports/2026-11-11-iql-frozen)
  mc_p       [control] P(success | s) by logistic regression on the episode outcome (an MLP, BCE)
Training data: the G1 caches (900 base + 450 exploration + 100 AG round-1 rollouts, train states and fresh layouts).
Validation:
  branch    5-fold OOF on the base rollouts (libero_oof.oof_split): corr with V* at the 424 branch points, overall
            and within episode (the progress-meter property)
  val       stage2 val (120 base rollouts on the val states, raw + video): AUC for the outcome at t = 100 / 200 / 300
            from video frames, and the agreement of video-frame vs raw-frame predictions
Curves: every 4th frame of every episode of base eval 18355478, residual eval 18438187 and stage2 val ->
<out>/curves.parquet (set, episode, init_state, repeat, success, policy_steps, frame_index, iql_p, mc_p).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import libero_iql as L  # noqa: E402
import libero_iql_g1 as G  # noqa: E402
from libero_value_vs_gt import corr, stat  # noqa: E402
from vl.libero_oof import oof_split  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
EVALS = {"base": RUNS / "eval_base_noise1_libero_10_18355478",
         "residual": RUNS / "eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187",
         "stage2_val": RUNS / "stage2_moka_18155849" / "val"}
POINTS = Path(__file__).resolve().parents[1] / "reports" / "2026-11-12-iql-labels" / "points.csv"
log = L.log


def agentview(D: dict) -> dict:
    """S -> agentview features (first 512) + t / 520 (last column); reward = success at the end, else 0."""
    S = np.concatenate([D["S"][:, :512], D["S"][:, -1:]], 1)
    succ = (D["vT"] == 0).astype(np.float32)  # vT = 0 success / -0.5 failure
    return {**D, "S": S, "R": np.where(D["term"], succ, 0.0).astype(np.float32), "y": succ}


def train_logit(D, tr, steps, device, seed, width=128, bs=1024, lr=3e-4):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    S, y = torch.as_tensor(D["S"], device=device), torch.as_tensor(D["y"], device=device)
    ns = L.Std(D["S"][tr], device)
    f = L.mlp(S.shape[1], width).to(device)
    o = torch.optim.Adam(f.parameters(), lr=lr)
    trt = torch.as_tensor(tr, device=device)
    for _ in range(steps):
        ix = trt[torch.randint(len(trt), (bs,), generator=g).to(device)]
        loss = nn.functional.binary_cross_entropy_with_logits(f(ns(S[ix])).squeeze(-1), y[ix])
        o.zero_grad(set_to_none=True), loss.backward(), o.step()

    @torch.no_grad()
    def pred(Sx):
        return torch.sigmoid(torch.cat([f(ns(torch.as_tensor(Sx[i:i + 8192], device=device))).squeeze(-1)
                                        for i in range(0, len(Sx), 8192)])).cpu().numpy()
    return pred


def fit(D, tr, steps, dev, seeds, width):
    """{arm: predictor(S)} averaged over seeds."""
    iq = [L.train_iql({**D, "A": np.zeros((len(D["S"]), 1), np.float32)}, tr, 0.9, steps, dev, s, width=width)
          for s in seeds]
    mc = [train_logit(D, tr, steps, dev, s, width) for s in seeds]
    return {"iql_p": lambda X: np.mean([f.V(X) for f in iq], 0), "mc_p": lambda X: np.mean([f(X) for f in mc], 0)}


def video_frames(path: Path, every: int = 4) -> tuple[np.ndarray, np.ndarray]:
    with av.open(str(path)) as c:
        fr = [f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(video=0))]
    idx = np.arange(0, len(fr), every)
    return idx, np.stack([fr[i] for i in idx])


def episode_states(name: str, run: Path, dev, cache: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows (one per 4th frame) + S (agentview features + t / 520) for every episode of an eval run; cached."""
    f = cache / f"curves_{name}.npz"
    if f.exists():
        z = np.load(f)
        return pd.read_parquet(f.with_suffix(".parquet")), z["S"]
    eps = pd.read_csv(run / "episodes.csv")
    rows, S = [], []
    for r in eps.itertuples():
        idx, fr = video_frames(run / "videos" / r.video)
        x = np.stack([fr, np.zeros_like(fr)], 1)  # wrist slot unused; agentview features are the first 512
        feat = G.r18_features(x, dev)[:, :512].astype(np.float32)
        S.append(np.concatenate([feat, (idx / 520.0)[:, None]], 1).astype(np.float32))
        rows.append(pd.DataFrame({"set": name, "episode": r.episode, "init_state": r.init_state, "repeat": r.repeat,
                                  "success": r.success, "policy_steps": r.policy_steps, "frame_index": idx}))
    d, S = pd.concat(rows, ignore_index=True), np.concatenate(S)
    np.savez(f, S=S)
    d.to_parquet(f.with_suffix(".parquet"))
    return d, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--cache-dir", default="/iris/u/karenvo/iql_cache")
    a = ap.parse_args()
    out, cd = Path(a.out), Path(a.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = agentview(L.build(20, cd / "iql_round0_n20.npz"))
    ex = agentview(G.cached(cd / "iql_explore_n20.npz", G.explore_set))
    ag = agentview(G.cached(cd / "iql_agr_r1_n20.npz", lambda: G.agr_set(dev)))
    res = {"args": vars(a)}

    # 1. branch-point validation: OOF on the base rollouts only (5 folds), as libero_iql
    d = base["d"]
    P = {k: np.full(len(d), np.nan) for k in ("iql_p", "mc_p")}
    for k in range(5):
        held, _ = oof_split(d, k)
        f = fit(base, np.flatnonzero(~held), a.steps, dev, a.seeds[:1], a.width)
        for arm in P:
            P[arm][held] = f[arm](base["S"][held])
        log(f"OOF fold {k} done")
    pts = pd.read_csv(POINTS)[["ep", "t", "v"]].rename(columns={"ep": "episode_index", "t": "frame_index"})
    m = pts.merge(d[L.KEY].assign(**P), on=L.KEY).assign(ep=lambda x: x.episode_index)
    res["branch"] = {}
    for arm in P:
        w = m.groupby("ep").apply(lambda x: corr(x[arm], x.v)).mean()
        res["branch"][arm] = {"corr_V*": stat(m, lambda x: corr(x[arm], x.v)), "within_episode": float(w)}
        log(f"branch {arm}: corr(V, V*) {res['branch'][arm]['corr_V*']['est']:+.2f} within-episode {w:+.2f}")

    # 2. final models on all training data, curves on the eval videos
    D = G.concat([base, ex, ag])
    D["y"] = np.concatenate([base["y"], ex["y"], ag["y"]])
    f = fit(D, np.arange(len(D["S"])), a.steps, dev, a.seeds, a.width)
    curves = []
    for name, run in EVALS.items():
        rows, S = episode_states(name, run, dev, cd)
        for arm in ("iql_p", "mc_p"):
            rows[arm] = f[arm](S)
        curves.append(rows)
        log(f"curves {name}: {rows.episode.nunique()} episodes")
    C = pd.concat(curves, ignore_index=True)
    C.to_parquet(out / "curves.parquet")

    # 3. val checks on stage2 val: outcome AUC from video frames; video vs raw-frame features
    v = C[C.set == "stage2_val"]
    res["val_auc"] = {arm: {t: G_auc(v[v.frame_index == t], arm) for t in (100, 200, 300)} for arm in ("iql_p", "mc_p")}
    eps = pd.read_csv(EVALS["stage2_val"] / "episodes.csv")
    raw_rows, raw_S = [], []
    for r in eps.head(20).itertuples():
        with np.load(EVALS["stage2_val"] / "raw" / r.raw) as z:
            idx = np.arange(0, r.policy_steps, 4)
            x = np.stack([z["image"][idx], np.zeros_like(z["image"][idx])], 1)
        raw_S.append(np.concatenate([G.r18_features(x, dev)[:, :512].astype(np.float32), (idx / 520.0)[:, None]], 1))
        raw_rows.append(pd.DataFrame({"episode": r.episode, "frame_index": idx}))
    R_ = pd.concat(raw_rows, ignore_index=True)
    RS = np.concatenate(raw_S).astype(np.float32)
    j = R_.assign(raw=f["iql_p"](RS)).merge(v[["episode", "frame_index", "iql_p"]], on=["episode", "frame_index"])
    res["video_vs_raw"] = {"corr": float(j.raw.corr(j.iql_p)), "mae": float((j.raw - j.iql_p).abs().mean()), "n": len(j)}
    (out / "curves_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("branch", "val_auc", "video_vs_raw")}, indent=1))
    log(f"-> {out}")


def G_auc(x: pd.DataFrame, arm: str) -> float:
    from libero_round_drift import auc
    return auc(x[arm].to_numpy(), x.success.to_numpy())


if __name__ == "__main__":
    main()
