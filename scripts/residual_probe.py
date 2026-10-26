"""Does anything in the past trajectory predict what the image-only baseline gets wrong?

    .venv/bin/python scripts/residual_probe.py              # -> reports/residual_probe.md

Track B caveat [ours, 2026-10-26]: the still-time upper bound (b3_still) only covers "how long nothing
moved". Here R0's val residual r = target value - V (V = mean over g0_r0_s{0,1,2}, best checkpoints) is
regressed on causal features of the gripper's past (data/proprio/frame_proprio.parquet, 15 Hz), one
feature group at a time on top of a base model. Proprio is a perfect record of the arm's motion, so a
group that cannot predict r beyond still time is information image history cannot add either (it can
still carry things proprio does not see: the marker, the holder, the operator).

Model [ours]: additive ridge on quantile-binned features (each feature -> 8 one-hot bins), fitted on
4 of 5 episode folds, scored on the 5th (GroupKFold by episode, fixed fold assignment, seed 0);
ridge lambda per outer fold from an inner episode-fold CV over 1e0..1e6 (a fixed small lambda
overfit episode offsets: every group, even calibration alone, scored worse than predicting 0).
Base = bins of V itself (calibration, so a group must add more than re-scaling V). Score = CV MSE of r
and its reduction vs base; intervals: bootstrap over val episodes of the per-episode squared-error
sums (2000 draws). Groups:
  still        log1p(seconds still), the b3_still input
  elapsed      seconds into the episode
  motion       per window 2/5/10/20 s: path length, net displacement, dz, d gripper, "window
               reaches before start"
  retries      gripper toggles so far, z dips (down then up >= 2 cm) so far, path length so far,
               z now - min z so far
  current      x, y, z, rotation, gripper now (no memory; control: the image already shows it)
  memory       still + elapsed + motion + retries
"""
from pathlib import Path

import numpy as np
import pandas as pd

from vl.history_gain import hold_groups, kind
from vl.history_plots import stall_seconds

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/runs"
FPS, NBIN, FOLDS, B = 15.0, 8, 5, 2000
LAMS = (1e0, 1e2, 1e3, 1e4, 1e5, 1e6)
CHOSEN = []
KEYS = ["repo", "episode_index", "frame_index"]
WIN = (2, 5, 10, 20)


def features():
    p = pd.read_parquet(ROOT / "data/proprio/frame_proprio.parquet").sort_values(KEYS).reset_index(drop=True)
    g = p.groupby(["repo", "episode_index"])
    step = np.sqrt(sum(g[a].diff().fillna(0) ** 2 for a in "xyz"))
    cum = step.groupby([p.repo, p.episode_index]).cumsum()
    f = p[KEYS].copy()
    f["t_s"] = p.frame_index / FPS
    for w in WIN:
        k = int(w * FPS)
        first = g.cumcount() < k      # window reaches before the start: clip to frame 0
        x0 = {a: g[a].shift(k).fillna(g[a].transform("first")) for a in ["x", "y", "z", "gripper"]}
        f[f"path_{w}"] = cum - cum.groupby([p.repo, p.episode_index]).shift(k).fillna(0)
        f[f"disp_{w}"] = np.sqrt(sum((p[a] - x0[a]) ** 2 for a in "xyz"))
        f[f"dz_{w}"] = p.z - x0["z"]
        f[f"dgrip_{w}"] = p.gripper - x0["gripper"]
        f[f"short_{w}"] = first.astype(float)
    f["toggles"] = (p.gripper > 0.5).astype(int).groupby([p.repo, p.episode_index]).diff().abs().fillna(0) \
        .groupby([p.repo, p.episode_index]).cumsum()
    f["dips"] = np.concatenate([dips(e.z.to_numpy()) for _, e in g])
    f["path_total"] = cum
    f["z_above_min"] = p.z - g.z.cummin()
    for c in ["x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21", "gripper"]:
        f[f"now_{c}"] = p[c]
    return f.merge(stall_seconds(), on=KEYS, validate="1:1")


def dips(z, h=0.02):
    """Causal count of z dips: a fall of >= h from the running high, then a rise of >= h."""
    out, n, hi, lo, down = np.zeros(len(z)), 0, z[0], z[0], False
    for i, v in enumerate(z):
        if not down:
            hi = max(hi, v)
            if hi - v >= h:
                down, lo = True, v
        else:
            lo = min(lo, v)
            if v - lo >= h:
                n, down, hi = n + 1, False, v
        out[i] = n
    return out


GROUPS = {
    "still": ["log_still"],
    "elapsed": ["t_s"],
    "motion": [f"{c}_{w}" for w in WIN for c in ["path", "disp", "dz", "dgrip"]] + [f"short_{w}" for w in WIN],
    "retries": ["toggles", "dips", "path_total", "z_above_min"],
    "current": [f"now_{c}" for c in ["x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21", "gripper"]],
}
GROUPS["memory"] = GROUPS["still"] + GROUPS["elapsed"] + GROUPS["motion"] + GROUPS["retries"]


def onehot(col):
    q = np.unique(np.quantile(col, np.linspace(0, 1, NBIN + 1)[1:-1]))
    b = np.searchsorted(q, col, side="right")
    return np.eye(len(q) + 1)[b]


def ridge(X, r, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ r)


def cv_pred(X, r, fold):
    """Outer GroupKFold predictions; lambda per outer fold from an inner GroupKFold on its training folds."""
    out = np.zeros_like(r)
    X = np.hstack([np.ones((len(X), 1)), X])
    for k in range(FOLDS):
        tr = np.flatnonzero(fold != k)
        inner = {lam: 0.0 for lam in LAMS}
        for j in set(range(FOLDS)) - {k}:
            a, b = tr[fold[tr] != j], tr[fold[tr] == j]
            for lam in LAMS:
                inner[lam] += ((r[b] - X[b] @ ridge(X[a], r[a], lam)) ** 2).sum()
        lam = min(inner, key=inner.get)
        CHOSEN.append(lam)
        te = fold == k
        out[te] = X[te] @ ridge(X[tr], r[tr], lam)
    return out


def main():
    ps = [pd.read_parquet(R / f"g0_r0_s{s}/val_predictions.parquet") for s in range(3)]
    for q in ps[1:]:
        assert (q[KEYS].values == ps[0][KEYS].values).all()
    d = ps[0].drop(columns=["V", "ce", "bin"]).assign(V=np.mean([q.V.to_numpy() for q in ps], 0))
    d = d.merge(features(), on=KEYS, how="left", validate="1:1")
    assert d.stall_s.notna().all()
    d["log_still"] = np.log1p(d.stall_s)
    d["r"] = d.value - d.V
    man = pd.read_parquet(ROOT / "data/manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    d = d.merge(man, on="episode_uid", validate="m:1")
    d = d.assign(to_go=(d["T"] - d.frame_index) / FPS)
    eps = d.episode_uid.unique()
    fold = pd.Series(np.random.default_rng(0).permutation(len(eps)) % FOLDS, index=eps)[d.episode_uid].to_numpy()

    base = onehot(d.V.to_numpy())
    oh = {c: onehot(d[c].to_numpy()) for cs in GROUPS.values() for c in cs}
    pred = {"none": np.full(len(d), 0.0), "base": cv_pred(base, d.r.to_numpy(), fold)}
    for name, cs in GROUPS.items():
        CHOSEN.clear()
        pred[name] = cv_pred(np.hstack([base] + [oh[c] for c in cs]), d.r.to_numpy(), fold)
        print(name, "lambda per fold", CHOSEN)
    pred["still+motion"] = cv_pred(np.hstack([base] + [oh[c] for c in GROUPS["still"] + GROUPS["motion"]]),
                                   d.r.to_numpy(), fold)
    pred["still+retries"] = cv_pred(np.hstack([base] + [oh[c] for c in GROUPS["still"] + GROUPS["retries"]]),
                                    d.r.to_numpy(), fold)
    pred["still+elapsed"] = cv_pred(np.hstack([base] + [oh[c] for c in GROUPS["still"] + GROUPS["elapsed"]]),
                                    d.r.to_numpy(), fold)

    h = hold_groups(d)
    d["subset_hold"] = d.index.isin(h.index)
    k = kind(d.outcome, d.machine_failure_mode)
    subsets = {"all val frames": np.ones(len(d), bool), "hold phase, eval repos": d.subset_hold.to_numpy(),
               "hold phase, other failure": d.subset_hold.to_numpy() & (k == "other"),
               "hold phase, stall": d.subset_hold.to_numpy() & (k == "stall")}
    rng = np.random.default_rng(0)
    lines = [__doc__.split("\n\n", 1)[1].strip(), ""]
    lines += [f"Val: {len(eps)} episodes, {len(d)} frames. r = target - V(R0 mean of 3 seeds); "
              f"MSE x 1e3. Reduction = (base MSE - group MSE) / base MSE; "
              f"vs still = group MSE - still MSE (negative = better than still), 95% episode bootstrap.", ""]
    for sname, m in subsets.items():
        e = d.episode_uid[m].to_numpy()
        ue, inv = np.unique(e, return_inverse=True)
        sq = {g: np.bincount(inv, (d.r.to_numpy()[m] - p[m]) ** 2, len(ue)) for g, p in pred.items()}
        n = np.bincount(inv, minlength=len(ue))
        idx = rng.integers(0, len(ue), (B, len(ue)))
        tot = n[idx].sum(1)
        lines += [f"### {sname} ({len(ue)} episodes, {m.sum()} frames)", "",
                  "| features (+ base) | CV MSE | reduction vs base | Δ MSE vs still [95%] |", "|---|---|---|---|"]
        mse = {g: s.sum() / n.sum() for g, s in sq.items()}
        for g in pred:
            dv = (sq[g][idx].sum(1) - sq["still"][idx].sum(1)) / tot
            lo, hi = np.percentile(dv, [2.5, 97.5])
            vs = "" if g in ("still", "none") else \
                f"{(mse[g] - mse['still']) * 1e3:+.3f} [{lo * 1e3:+.3f}, {hi * 1e3:+.3f}]{'*' if hi < 0 or lo > 0 else ''}"
            lines.append(f"| {g} | {mse[g] * 1e3:.3f} | {(mse['base'] - mse[g]) / mse['base']:+.1%} | {vs} |")
        lines.append("")
    out = ROOT / "reports/residual_probe.md"
    out.write_text("# Residual probe: past trajectory vs R0's errors\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
