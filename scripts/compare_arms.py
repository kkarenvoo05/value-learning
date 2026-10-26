"""Compare arms against a reference arm on best-checkpoint val metrics.

    .venv/bin/python scripts/compare_arms.py g0_r0 c1_percam      # Stage 5.5 C1 vs the Gate 0 baseline

Prints mean ± seed sd per arm (from result.json), then a paired test on the saved per-frame val
predictions (val_predictions.parquet = the best checkpoint, same 247 val episodes for every run).

Paired test [ours, 2026-10-25; replaces "CE below the reference by > 2 reference sd, shape MAE and
AUC 1 s no worse", whose bar came from a 3-seed sd and whose "no worse" had no tolerance]:
hierarchical bootstrap, each replicate resamples val episodes (shared by both arms, so the
comparison is paired) and, independently per arm, its seeds with replacement; the metric is the mean
over the drawn seeds, computed as in `vl.metrics.score` on the drawn episodes. Reports the arm's
change vs the reference with a 95% interval that covers both episode and training-seed noise.
Win = the val CE interval is entirely below 0, and neither guard (shape MAE, AUC 1 s) is credibly
worse (its interval entirely on the bad side).

Targeted metrics [ours, pre-registered 2026-10-25 before the B3 still-time upper bound was read,
because whole-val CE dilutes effects that live on a few hold-phase frames (B3: stall-group p_succ
shifts of 0.04-0.07 with no CE change)]. Frame-level AUC of p_succ (predicted mass on V > -0.5,
from val_probs.npz), success = positive, on hold-phase frames with the groups of
`vl.history_gain.hold_groups` (eval repos, > 5 s in, > 2 s before T; still = gripper < 1 cm/s):
  stall_auc   every success frame vs stall frames still >= 4 s
  frozen_auc  frozen-success frames (still >= 4 s, episode's longest pause >= 8 s) vs stall frames
              still >= 4 s: the case history cannot see by construction
Pooled over the eval repos (paired arm vs reference, so a repo effect common to both cancels).
In the bootstrap a frame is weighted by how often its episode was drawn. Reported with their own
verdict ("targeted": both intervals checked, credibly better = interval entirely above 0); the CE
win rule is unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vl.history_gain import hold_groups
from vl.history_plots import stall_seconds
from vl.metrics import EVAL_REPOS_PREFIX

R = Path(__file__).resolve().parents[1] / "data/runs"
K = ["val_ce", "val_ce_per_episode", "val_mae", "offset_free_mae", "eval_repos_mae",
     "eval_repos_auc_1s", "eval_repos_auc_5s", "eval_repos_auc_10s"]
TARGETED = ["stall_auc", "frozen_auc"]
HIGHER_BETTER = {"eval_repos_auc_1s", "eval_repos_auc_5s", "eval_repos_auc_10s", *TARGETED}
B, FPS = 2000, 15.0
ref, arms = sys.argv[1], sys.argv[2:]


def seeds(a):
    return sorted(int(p.name[len(a) + 2:]) for p in R.glob(f"{a}_s*")
                  if p.name[len(a) + 2:].isdigit() and (p / "result.json").exists())


stats = {}
for a in [ref, *arms]:
    rs = [json.loads((R / f"{a}_s{s}" / "result.json").read_text()) for s in seeds(a)]
    if not rs:
        print(f"{a}: no finished seeds")
        continue
    ev = [r["best_eval"] for r in rs]
    stats[a] = {k: (np.mean([e[k] for e in ev]), np.std([e[k] for e in ev], ddof=1) if len(ev) > 1 else np.nan,
                    [e[k] for e in ev]) for k in K}
    stats[a]["n"] = len(ev)
    stats[a]["step"] = [r["best_step"] for r in rs]
print("| arm | n | best step | " + " | ".join(K) + " |")
print("|" + "---|" * (len(K) + 3))
for a, s in stats.items():
    print(f"| {a} | {s['n']} | {s['step']} | " + " | ".join(f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in K) + " |")
print("\nper-seed val_ce:", {a: [round(x, 4) for x in s["val_ce"][2]] for a, s in stats.items()})
if ref not in stats or len(stats) < 2:
    sys.exit()


def episode_table(run):
    """Per-episode sums/means from which every metric in K is recomputed on a bootstrap draw."""
    p = pd.read_parquet(R / run / "val_predictions.parquet")
    p = p.assign(err=(p.V - p.value).abs(),
                 ofe=((p.V - p.groupby("episode_uid").V.transform("mean"))
                      - (p.value - p.groupby("episode_uid").value.transform("mean"))).abs())
    g = p.groupby("episode_uid", sort=True)
    e = g.agg(n=("ce", "size"), ce_sum=("ce", "sum"), err_sum=("err", "sum"), ofe=("ofe", "mean"),
              repo=("repo", "first"), succ=("outcome", lambda o: int(o.iloc[0] == "success")))
    for t in (1, 5, 10):   # vl.metrics.within_repo_auc window
        w = p[(p.frame_index / FPS >= t - 0.5) & (p.frame_index / FPS < t + 0.5)]
        e[f"V{t}"] = w.groupby("episode_uid").V.mean()
    return e


def pair_auc(y, s):
    """`vl.metrics.auc` (rank AUC, ties = 1/2) as a pairwise count; NaN if one class is missing."""
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    c = pos[:, None] - neg[None, :]
    return ((c > 0).sum() + 0.5 * (c == 0).sum()) / c.size


def hold_frames(run, uids):
    """Hold-phase frames of the targeted AUCs: episode position in `uids`, stall_auc / frozen_auc
    class (1 success, 0 still stall, -1 not used) and p_succ. Labels depend only on the episode and
    proprio, so they are the same for every run."""
    p = pd.read_parquet(R / run / "val_predictions.parquet")
    z = np.load(R / run / "val_probs.npz", allow_pickle=True)
    assert (z["episode_uid"] == p.episode_uid.values).all() and (z["frame_index"] == p.frame_index.values).all()
    p["p_succ"] = z["probs"][:, np.linspace(-1, 0, z["probs"].shape[1]) > -0.5].sum(1)
    man = pd.read_parquet(R.parent / "manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    keys = ["repo", "episode_index", "frame_index"]
    p = p.merge(man, on="episode_uid", validate="m:1").merge(stall_seconds(), on=keys, how="left", validate="1:1")
    p = p.assign(t_s=p.frame_index / FPS, to_go=(p["T"] - p.frame_index) / FPS)
    h = hold_groups(p)
    succ, still = h.group.str.startswith("success").to_numpy(), h["still >= 4 s now"].to_numpy()
    stall_still = (h.group == "stall").to_numpy() & still
    frozen = (h.group == "success, frozen (pause >= 8 s)").to_numpy() & still
    return {"ep": pd.Index(uids).get_indexer(h.episode_uid), "p_succ": h.p_succ.to_numpy(),
            "stall_auc": np.select([succ, stall_still], [1, 0], -1),
            "frozen_auc": np.select([frozen, stall_still], [1, 0], -1)}


def weighted_auc(y, s, w):
    """Rank AUC (ties = 1/2) with frame weights w; equals pair_auc when w = 1."""
    u, inv = np.unique(s, return_inverse=True)
    wp, wn = np.bincount(inv, w * (y == 1), len(u)), np.bincount(inv, w * (y == 0), len(u))
    return (wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


def arrays(e, hf):
    a = {c: e[c].to_numpy() for c in ["n", "ce_sum", "err_sum", "ofe", "succ", "V1", "V5", "V10"]}
    a["eval"] = e.repo.str.startswith(EVAL_REPOS_PREFIX).to_numpy()
    a["repo"] = pd.factorize(e.repo)[0]
    a["hold"] = hf
    return a


def metrics(a, idx):
    hf = a["hold"]
    w_ep = np.bincount(idx, minlength=len(a["n"]))[hf["ep"]]
    d = {c: v[idx] for c, v in a.items() if c != "hold"}
    ev = d["eval"]
    out = {"val_ce": d["ce_sum"].sum() / d["n"].sum(), "val_ce_per_episode": (d["ce_sum"] / d["n"]).mean(),
           "val_mae": d["err_sum"].sum() / d["n"].sum(), "offset_free_mae": d["ofe"].mean(),
           "eval_repos_mae": d["err_sum"][ev].sum() / d["n"][ev].sum()}
    for t in (1, 5, 10):
        ok = ev & ~np.isnan(d[f"V{t}"])
        vals = [(pair_auc(d["succ"][m], d[f"V{t}"][m]), m.sum()) for r in np.unique(d["repo"][ok])
                for m in [ok & (d["repo"] == r)]]
        vals = [(x, n) for x, n in vals if not np.isnan(x)]
        out[f"eval_repos_auc_{t}s"] = sum(x * n for x, n in vals) / sum(n for _, n in vals)
    for k in TARGETED:
        m = hf[k] >= 0
        out[k] = weighted_auc(hf[k][m], hf["p_succ"][m], w_ep[m])
    return out


tabs = {a: [episode_table(f"{a}_s{s}") for s in seeds(a)] for a in stats}
uids = tabs[ref][0].index
assert all((t.index == uids).all() for ts in tabs.values() for t in ts), "val episodes differ between runs"
tabs = {a: [arrays(t, hold_frames(f"{a}_s{s}", uids)) for t, s in zip(ts, seeds(a))] for a, ts in tabs.items()}
y = tabs[ref][0]["hold"]
assert all((t["hold"]["ep"] == y["ep"]).all() and all((t["hold"][k] == y[k]).all() for k in TARGETED)
           for ts in tabs.values() for t in ts), "hold-phase labels differ between runs"
for k in TARGETED:
    m = y[k] >= 0
    print(f"{k}: {(y[k] == 1).sum()} success frames / {np.unique(y['ep'][y[k] == 1]).size} episodes vs "
          f"{(y[k] == 0).sum()} still-stall frames / {np.unique(y['ep'][y[k] == 0]).size} episodes")
    assert np.isclose(weighted_auc(y[k][m], y["p_succ"][m], np.ones(m.sum())), pair_auc(y[k][m], y["p_succ"][m]))
rng = np.random.default_rng(0)
full = np.arange(len(uids))
KT = K + TARGETED
point = {a: {k: np.mean([metrics(t, full)[k] for t in ts]) for k in KT} for a, ts in tabs.items()}
draws = {a: {k: [] for k in KT} for a in stats}
for _ in range(B):
    idx = rng.integers(0, len(uids), len(uids))
    for a, ts in tabs.items():
        per_seed = [metrics(t, idx) for t in ts]
        pick = rng.integers(0, len(ts), len(ts))
        for k in KT:
            draws[a][k].append(np.mean([per_seed[i][k] for i in pick]))
print(f"\nPaired test vs {ref}: change (arm - {ref}) with 95% interval, hierarchical bootstrap over "
      f"{len(uids)} val episodes x seeds, {B} draws. '*' = interval excludes 0.")
print("| arm | " + " | ".join(KT) + " | verdict | targeted |")
print("|" + "---|" * (len(KT) + 3))
for a in arms:
    if a not in stats:
        continue
    ci, cells = {}, []
    for k in KT:
        d = np.array(draws[a][k]) - np.array(draws[ref][k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        ci[k] = (lo, hi)
        cells.append(f"{point[a][k] - point[ref][k]:+.4f} [{lo:+.4f}, {hi:+.4f}]{'*' if lo > 0 or hi < 0 else ''}")
    worse = [k for k in ["offset_free_mae", "eval_repos_auc_1s"]
             if (ci[k][1] < 0 if k in HIGHER_BETTER else ci[k][0] > 0)]
    verdict = ("WIN" if ci["val_ce"][1] < 0 and not worse
               else "no win" + (f" (worse: {', '.join(worse)})" if worse else ""))
    better = [k for k in TARGETED if ci[k][0] > 0]
    tworse = [k for k in TARGETED if ci[k][1] < 0]
    targeted = ", ".join([f"better: {', '.join(better)}"] * bool(better) + [f"worse: {', '.join(tworse)}"] * bool(tworse)) or "no change"
    print(f"| {a} (n={stats[a]['n']}) | " + " | ".join(cells) + f" | {verdict} | {targeted} |")
print("\nReference point values: " + ", ".join(f"{k} {point[ref][k]:.4f}" for k in TARGETED))
