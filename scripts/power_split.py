"""Where the width of the compare_arms interval comes from: val episodes or training seeds.

    .venv/bin/python scripts/power_split.py g0_r0 b3_still b2_gap2 b2_gap4

Same paired hierarchical bootstrap as `scripts/compare_arms.py` (episodes shared by both arms, seeds
per arm), run three ways on val CE and stall_auc [ours, 2026-10-26, Track B caveat "no gain = none
detectable"]:
  full      resample episodes and seeds (= compare_arms)
  episodes  resample episodes, every seed kept (mean over all seeds): the episode part
  seeds     all episodes kept, resample seeds: the seed part
The seed part's bootstrap variance is scaled by n/(n-1) (a resampled mean of n seeds has variance
s^2 (n-1)/n^2), then projected to k seeds per arm: half-width(k) = 1.96 sqrt(var_ep + var_seed n/k).
More seeds only shrink the seed part; the episode part is fixed by the 247 val episodes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vl.history_gain import hold_groups
from vl.history_plots import stall_seconds

R = Path(__file__).resolve().parents[1] / "data/runs"
B, FPS = 2000, 15.0
ref, arms = sys.argv[1], sys.argv[2:]


def seeds(a):
    return sorted(int(p.name[len(a) + 2:]) for p in R.glob(f"{a}_s*")
                  if p.name[len(a) + 2:].isdigit() and (p / "result.json").exists())


def load(run, uids=None):
    """Per-episode CE sums and frame counts, plus stall_auc hold frames (as compare_arms.hold_frames)."""
    p = pd.read_parquet(R / run / "val_predictions.parquet")
    z = np.load(R / run / "val_probs.npz", allow_pickle=True)
    assert (z["episode_uid"] == p.episode_uid.values).all() and (z["frame_index"] == p.frame_index.values).all()
    p["p_succ"] = z["probs"][:, np.linspace(-1, 0, z["probs"].shape[1]) > -0.5].sum(1)
    e = p.groupby("episode_uid", sort=True).agg(n=("ce", "size"), ce=("ce", "sum"))
    uids = e.index if uids is None else uids
    assert (e.index == uids).all()
    man = pd.read_parquet(R.parent / "manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    keys = ["repo", "episode_index", "frame_index"]
    p = p.merge(man, on="episode_uid", validate="m:1").merge(stall_seconds(), on=keys, how="left", validate="1:1")
    h = hold_groups(p.assign(t_s=p.frame_index / FPS, to_go=(p["T"] - p.frame_index) / FPS))
    succ, stall = h.group.str.startswith("success").to_numpy(), (h.group == "stall").to_numpy() & h["still >= 4 s now"].to_numpy()
    m = succ | stall
    return {"n": e.n.to_numpy(), "ce": e.ce.to_numpy(), "ep": pd.Index(uids).get_indexer(h.episode_uid[m]),
            "y": succ[m].astype(int), "s": h.p_succ.to_numpy()[m]}, uids


def weighted_auc(y, s, w):
    u, inv = np.unique(s, return_inverse=True)
    wp, wn = np.bincount(inv, w * (y == 1), len(u)), np.bincount(inv, w * (y == 0), len(u))
    return (wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


def metrics(t, idx):
    w = np.bincount(idx, minlength=len(t["n"]))
    return {"val_ce": (t["ce"] * w).sum() / (t["n"] * w).sum(), "stall_auc": weighted_auc(t["y"], t["s"], w[t["ep"]])}


runs, uids = {}, None
for a in [ref, *arms]:
    runs[a] = []
    for s in seeds(a):
        t, uids = load(f"{a}_s{s}", uids)
        runs[a].append(t)
rng = np.random.default_rng(0)
N = len(uids)
full = np.arange(N)
K = ["val_ce", "stall_auc"]
fixed = {a: [metrics(t, full) for t in ts] for a, ts in runs.items()}
draws = {mode: {a: {k: [] for k in K} for a in runs} for mode in ["full", "episodes", "seeds"]}
for _ in range(B):
    idx = rng.integers(0, N, N)
    for a, ts in runs.items():
        per = [metrics(t, idx) for t in ts]
        pick = rng.integers(0, len(ts), len(ts))
        pick2 = rng.integers(0, len(ts), len(ts))
        for k in K:
            draws["full"][a][k].append(np.mean([per[i][k] for i in pick]))
            draws["episodes"][a][k].append(np.mean([p[k] for p in per]))
            draws["seeds"][a][k].append(np.mean([fixed[a][i][k] for i in pick2]))

print(f"{N} val episodes, {B} draws; half-width = half the 95% interval of (arm - {ref}); "
      "sd = bootstrap sd of the difference\n")
print("| arm | metric | point Δ | full half-width | episodes sd | seeds sd (x n/(n-1)) | "
      "share of variance from seeds | projected half-width 3 / 6 / 10 / 20 seeds |")
print("|---|---|---|---|---|---|---|---|")
for a in arms:
    n = len(runs[a])
    for k in K:
        d = {m: np.array(draws[m][a][k]) - np.array(draws[m][ref][k]) for m in draws}
        hw = np.diff(np.percentile(d["full"], [2.5, 97.5]))[0] / 2
        v_ep, v_seed = d["episodes"].var(), d["seeds"].var() * n / (n - 1)
        proj = [1.96 * np.sqrt(v_ep + v_seed * n / kk) for kk in (3, 6, 10, 20)]
        pt = np.mean([f[k] for f in fixed[a]]) - np.mean([f[k] for f in fixed[ref]])
        print(f"| {a} | {k} | {pt:+.4f} | {hw:.4f} | {np.sqrt(v_ep):.4f} | {np.sqrt(v_seed):.4f} | "
              f"{v_seed / (v_ep + v_seed):.0%} | " + " / ".join(f"{x:.4f}" for x in proj) + " |")
