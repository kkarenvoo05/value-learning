"""Can hand-motion history or force stand in for C2a's past frame?

    .venv/bin/python scripts/c2a_gain_probe.py              # -> reports/c2a_gain_probe.md

Question [ours, Karen 2026-10-27]: C2a (2 s-old frame stacked into the ResNet input) beats R0 on still
stalls; proprio still time (b3_still) did not. Before training a proprio-history or force arm, check
whether those signals can *predict what C2a changes*. Target dV = V(c2a_stack2_l40s, mean of 3 seeds) -
V(g0_r0_l40s, mean of 10 seeds), per val frame, both on L40S. If a feature group predicts dV well, a
model given that group could make the same correction without images; if it explains little of the
reliable part of dV, C2a's gain comes from what the camera sees.

Same probe as scripts/residual_probe.py (additive ridge on 8 quantile bins per feature, 5 episode
folds, lambda by inner episode CV). Base = bins of V(R0), so a group must add more than "C2a rescales V".
Score = CV R^2 of dV beyond base; ceiling = split-half reliability of dV (C2a s0 vs s1+s2, R0 s0-4 vs
s5-9, Spearman-Brown to the full average), i.e. the part of dV that is not seed noise. Second target:
r = value - V(C2a), i.e. is there still something in proprio / force that C2a misses (would they add on
top of C2a?). Groups (all causal, 15 Hz):
  still        log1p(seconds still), the b3_still input
  motion       residual_probe's memory groups (still, elapsed, 2-20 s windows, retries) + short windows
               0.5 / 1 / 2 s: path, net displacement, dz, d gripper, max step speed (from positions;
               the reported velocity runs 1-2 rows ahead of video, so it is not used)
  force        telemetry.franka.ee_wrench now (6) + |F| now; |F| mean and std, fz mean over 0.5 / 1 / 2 s;
               fz change over 2 s. Only 9 of 15 repos record it (missing: c00, c01, c02, r00, r01), so all
               groups are also scored on the force-present subset, fitted on that subset only
  motion+force
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_probe import FOLDS, GROUPS, KEYS, cv_pred, features, onehot  # noqa: E402
from vl.history_gain import hold_groups, kind  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/runs"
FPS, B = 15.0, 2000
SHORT = (0.5, 1, 2)
FWIN = (0.5, 1, 2)
R0, C2A = "g0_r0_l40s", "c2a_stack2_l40s"


def short_motion():
    p = pd.read_parquet(ROOT / "data/proprio/frame_proprio.parquet").sort_values(KEYS).reset_index(drop=True)
    g = p.groupby(["repo", "episode_index"])
    step = np.sqrt(sum(g[a].diff().fillna(0) ** 2 for a in "xyz"))
    key = [p.repo, p.episode_index]
    cum = step.groupby(key).cumsum()
    f = p[KEYS].copy()
    for w in SHORT:
        k = int(round(w * FPS))
        x0 = {a: g[a].shift(k).fillna(g[a].transform("first")) for a in ["x", "y", "z", "gripper"]}
        f[f"spath_{w}"] = cum - cum.groupby(key).shift(k).fillna(0)
        f[f"sdisp_{w}"] = np.sqrt(sum((p[a] - x0[a]) ** 2 for a in "xyz"))
        f[f"sdz_{w}"] = p.z - x0["z"]
        f[f"sdgrip_{w}"] = p.gripper - x0["gripper"]
        f[f"smax_{w}"] = step.groupby(key).transform(lambda s: s.rolling(k, min_periods=1).max())
    return f


def force():
    rows = []
    for repo in sorted(p.name for p in (ROOT / "data/videos").iterdir()):
        for fp in sorted((ROOT / "data/videos" / repo / "data").rglob("*.parquet")):
            try:
                d = pd.read_parquet(fp, columns=["episode_index", "frame_index", "telemetry.franka.ee_wrench"])
            except Exception:                 # column absent in this repo
                break
            w = np.stack(d.pop("telemetry.franka.ee_wrench").to_numpy())
            d[["fx", "fy", "fz", "mx", "my", "mz"]] = w
            d["repo"] = repo
            rows.append(d)
    f = pd.concat(rows).sort_values(KEYS).reset_index(drop=True)
    key = [f.repo, f.episode_index]
    f["fmag"] = np.sqrt(f.fx ** 2 + f.fy ** 2 + f.fz ** 2)
    for w in FWIN:
        k = int(round(w * FPS))
        roll = lambda c, fn: f.groupby(key)[c].transform(lambda s: getattr(s.rolling(k, min_periods=1), fn)())  # noqa: E731
        f[f"fmag_mean_{w}"] = roll("fmag", "mean")
        f[f"fmag_std_{w}"] = roll("fmag", "std").fillna(0)
        f[f"fz_mean_{w}"] = roll("fz", "mean")
    f["dfz_2"] = f.fz - f.groupby(key).fz.shift(30).fillna(f.groupby(key).fz.transform("first"))
    return f


def mean_v(arm, seeds):
    ps = [pd.read_parquet(R / f"{arm}_s{s}/val_predictions.parquet") for s in seeds]
    for q in ps[1:]:
        assert (q[KEYS].values == ps[0][KEYS].values).all()
    return ps[0], np.mean([q.V.to_numpy() for q in ps], 0)


def main():
    d, v_r0 = mean_v(R0, range(10))
    d = d.drop(columns=["V", "ce", "bin"]).assign(V_r0=v_r0)
    q, v_c2a = mean_v(C2A, range(3))
    assert (q[KEYS].values == d[KEYS].values).all()
    d["V_c2a"] = v_c2a
    d["dV"] = d.V_c2a - d.V_r0
    d["r_c2a"] = d.value - d.V_c2a
    # split-half dV for the noise ceiling
    da = mean_v(C2A, [0])[1] - mean_v(R0, range(5))[1]
    db = mean_v(C2A, [1, 2])[1] - mean_v(R0, range(5, 10))[1]

    d = d.merge(features(), on=KEYS, how="left", validate="1:1").merge(short_motion(), on=KEYS, validate="1:1")
    fo = force()
    d = d.merge(fo, on=KEYS, how="left", validate="1:1")
    d["has_force"] = d.fx.notna()
    d["log_still"] = np.log1p(d.stall_s)
    man = pd.read_parquet(ROOT / "data/manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    d = d.merge(man, on="episode_uid", validate="m:1")
    d = d.assign(to_go=(d["T"] - d.frame_index) / FPS)

    fcols = ["fx", "fy", "fz", "mx", "my", "mz", "fmag", "dfz_2"] + \
        [f"{c}_{w}" for w in FWIN for c in ["fmag_mean", "fmag_std", "fz_mean"]]
    scols = [f"{c}_{w}" for w in SHORT for c in ["spath", "sdisp", "sdz", "sdgrip", "smax"]]
    groups = {"still": GROUPS["still"], "motion": GROUPS["memory"] + scols, "force": fcols,
              "motion+force": GROUPS["memory"] + scols + fcols}

    h = hold_groups(d)
    hold = d.index.isin(h.index)
    stall = hold & (kind(d.outcome, d.machine_failure_mode) == "stall")
    rng = np.random.default_rng(0)

    def reliability(m):
        r = np.corrcoef(da[m], db[m])[0, 1]
        return 2 * r / (1 + r)

    lines = [__doc__.split("\n\n", 1)[1].strip(), ""]
    for sname, sub, gnames in [("all val frames", np.ones(len(d), bool), ["still", "motion"]),
                               ("force-present repos", d.has_force.to_numpy(), list(groups))]:
        s = d[sub].reset_index(drop=True)
        eps = s.episode_uid.unique()
        fold = pd.Series(np.random.default_rng(0).permutation(len(eps)) % FOLDS, index=eps)[s.episode_uid].to_numpy()
        base = onehot(s.V_r0.to_numpy())
        oh = {c: onehot(s[c].to_numpy()) for g in gnames for c in groups[g]}
        lines += [f"## {sname}: {len(eps)} val episodes, {len(s)} frames "
                  f"({', '.join(sorted(r.split('d2-')[1] for r in s.repo.unique()))})", ""]
        for tname, tcol, bcol in [("dV = V(C2a) - V(R0)", "dV", "V_r0"), ("r = target - V(C2a)", "r_c2a", "V_c2a")]:
            y = s[tcol].to_numpy()
            bX = onehot(s[bcol].to_numpy())
            pred = {"none": np.zeros(len(s)), "base": cv_pred(bX, y, fold)}
            for g in gnames:
                pred[g] = cv_pred(np.hstack([bX] + [oh[c] for c in groups[g]]), y, fold)
                print(sname, tname, g, flush=True)
            subs = {"all": np.ones(len(s), bool), "hold phase, eval repos": hold[sub], "hold phase, stall": stall[sub]}
            lines += [f"### target {tname}", ""]
            if tcol == "dV":
                lines += ["Reliable share of dV variance (split-half, Spearman-Brown): " + ", ".join(
                    f"{k} {reliability(np.flatnonzero(sub)[m]):.2f}" for k, m in subs.items()) +
                    ". R^2 above this would be fitting seed noise.", ""]
            lines += ["| subset | episodes | var x 1e3 | " + " | ".join(f"R² {g}" for g in pred if g != "none") +
                      " | " + " | ".join(f"Δ R² {g} vs base [95%]" for g in gnames) + " |",
                      "|---|---|---|" + "---|" * (len(pred) - 1 + len(gnames))]
            for k, m in subs.items():
                e = s.episode_uid[m].to_numpy()
                ue, inv = np.unique(e, return_inverse=True)
                sq = {g: np.bincount(inv, (y[m] - p[m]) ** 2, len(ue)) for g, p in pred.items()}
                idx = rng.integers(0, len(ue), (B, len(ue)))
                r2 = {g: 1 - sq[g].sum() / sq["none"].sum() for g in pred}   # vs predicting 0 (dV mean ~0)
                cells = []
                for g in gnames:
                    dv = (sq["base"][idx].sum(1) - sq[g][idx].sum(1)) / sq["none"][idx].sum(1)
                    lo, hi = np.percentile(dv, [2.5, 97.5])
                    cells.append(f"{r2[g] - r2['base']:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
                lines.append(f"| {k} | {len(ue)} | {np.mean(y[m] ** 2) * 1e3:.2f} | " +
                             " | ".join(f"{r2[g]:+.3f}" for g in pred if g != "none") + " | " + " | ".join(cells) + " |")
            lines.append("")
    out = ROOT / "reports/c2a_gain_probe.md"
    out.write_text("# C2a gain probe: can proprio history or force stand in for the past frame?\n\n" +
                   "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
