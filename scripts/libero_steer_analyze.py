"""Analyze scripts/libero_steer_probe.py output (docs/CURRENT.md B2 steps 1-2).

Step 1, candidate spread, per replan point of the base episodes:
  act_spread   mean pairwise distance between the N candidates' first 5 actions (arm dims 0-5, each divided by
               the std of executed actions over all base episodes), per action step
  act_motion   the same distance between consecutive executed chunks (t vs t+5): how much the action changes
               over time anyway, as a reference scale
  grip_split   at least one candidate's gripper sign differs from the others in the 5 executed steps
  eef_spread   mean pairwise distance (mm) of the end effector after each candidate's 5 actions
  eef_motion   how far candidate 0 moves the end effector in those 5 steps (mm)
Step 2, headroom, per branch point (N candidates x R continuations of the base policy to the end):
  var_between  between-candidate variance of the true success rate, after removing binomial noise:
               var_c(p_hat) - mean_c(s2_c) / R   (s2_c = sample variance of candidate c's R outcomes)
  cv_gain      cross-validated oracle selection gain (R >= 2): choose the candidate with the best success on
               repeat r, score it on the other repeats, minus the mean over all candidates; ties are averaged.
               An unbiased estimate of what picking with a 1-rollout-per-candidate oracle buys.
  v2 (--args.cand-noise-scale != 1: candidates 1..N-1 use scaled start noise, candidate 0 is a normal sample):
  p_cand0 / p_scaled   success of candidate 0 / mean of the scaled candidates (cost of the extra noise)
  cv_gain_vs_cand0     as cv_gain, but minus candidate 0's held-out success: does picking among the more varied
                       candidates beat simply running the policy?
  offset probe (--args.cand-offsets, 2026-11-20; candidate 0 = the plain policy, others = same chunk + offset):
  per offset label: d_vs_none = mean over branch points of p_c - p_0 (paired), CI from a bootstrap over base
  episodes (clustered: the branch points of one episode share its trunk); seg_cm = median end-effector distance
  from candidate 0 after the held segment; by_t = the same per branch step. cv_gain / cv_gain_vs_cand0 get the
  clustered CI too (cv_gain_ci95_ep / cv_gain_vs_cand0_ci95_ep).
Writes <run>/analysis/{spread.csv, branches.csv, summary.json} and prints a summary.

Usage: .venv/bin/python scripts/libero_steer_analyze.py --run <runs/steer_probe_libero_10_JOB>
"""

import argparse
import itertools
import json
import pathlib

import numpy as np
import pandas as pd


def pairwise_mean(x):
    """x (N, ...) -> mean Euclidean distance over candidate pairs (over the flattened trailing dims)."""
    n = len(x)
    d = [np.linalg.norm((x[i] - x[j]).reshape(-1)) for i, j in itertools.combinations(range(n), 2)]
    return float(np.mean(d))


def cv_gain(s):
    """s (N, R) binary -> cross-validated oracle selection gain and the mean success."""
    n, r = s.shape
    if r < 2:
        return np.nan
    gains = []
    for k in range(r):
        sel = s[:, k]
        rest = np.delete(s, k, axis=1).mean(1)
        best = rest[sel == sel.max()].mean()
        gains.append(best - rest.mean())
    return float(np.mean(gains))


def cv_gain_vs(s, ref=0):
    """As cv_gain, but the selected candidate's held-out success minus candidate `ref`'s held-out success."""
    n, r = s.shape
    if r < 2:
        return np.nan
    gains = []
    for k in range(r):
        sel = s[:, k]
        rest = np.delete(s, k, axis=1).mean(1)
        gains.append(rest[sel == sel.max()].mean() - rest[ref])
    return float(np.mean(gains))


def var_between(s):
    n, r = s.shape
    p = s.mean(1)
    s2 = s.var(1, ddof=1) if r > 1 else np.zeros(n)
    return float(p.var(ddof=1) - s2.mean() / r)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_ci_cluster(df, col, n=2000, seed=0):
    """95% CI of the mean of df[col] with base episodes resampled as clusters."""
    groups = [g[col].to_numpy(float) for _, g in df.groupby("episode")]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if not groups:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    m = []
    for _ in range(n):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        m.append(np.concatenate(pick).mean())
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def offset_summary(br, files, eps):
    """Offset probe: per-label paired effect vs candidate 0 and the displacement each offset achieved."""
    labels = [str(x) for x in files[0]["offset_labels"]]
    rows = []
    for (_, e), z in zip(eps.iterrows(), files):
        if "br_t" not in z.files:
            continue
        for j, t in enumerate(z["br_t"]):
            p = z["br_success"][j].astype(float).mean(1)
            seg = z["br_seg_eef"][j]
            for c, lab in enumerate(labels):
                rows.append({"episode": e.episode, "t": int(t), "label": lab, "c": c, "p": p[c],
                             "d_vs_none": p[c] - p[0], "seg_cm": 100 * float(np.linalg.norm(seg[c] - seg[0]))})
    d = pd.DataFrame(rows)
    out = {"labels": labels, "offset_m": float(files[0]["offset_m"]), "hold_chunks": int(files[0]["hold_chunks"]),
           "per_step_offsets_xyz": np.asarray(files[0]["cand_offsets"])[:, :3].round(4).tolist(), "by_label": {}}
    for lab in labels:
        g = d[d.label == lab]
        out["by_label"][lab] = {
            "p": float(g.p.mean()), "d_vs_none": float(g.d_vs_none.mean()),
            "d_vs_none_ci95_ep": boot_ci_cluster(g, "d_vs_none") if lab != "none" else [0.0, 0.0],
            "seg_cm_median": float(g.seg_cm.median()),
            "by_t": {int(t): {"p": float(h.p.mean()), "d_vs_none": float(h.d_vs_none.mean()), "n": int(len(h))}
                     for t, h in g.groupby("t")}}
    return out, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    eps = pd.read_csv(run / "episodes.csv")
    files = [np.load(run / "probe" / f, allow_pickle=False) for f in eps.file]
    k = 5
    executed = np.concatenate([z["executed"] for z in files])
    scale = executed[:, :6].std(0) + 1e-8
    rows, brows = [], []
    for (_, e), z in zip(eps.iterrows(), files):
        chunks, ts = z["rp_chunks"], z["rp_t"]
        for j, t in enumerate(ts):
            c = chunks[j][:, :k]
            arm = c[:, :, :6] / scale
            g = np.sign(c[:, :, 6])
            row = {"episode": e.episode, "init_state": e.init_state, "success": e.success, "t": int(t),
                   "act_spread": pairwise_mean(arm) / np.sqrt(k),
                   "grip_split": bool((g != g[:1]).any()),
                   "eef_spread": 1000 * pairwise_mean(z["rp_eef_after"][j]),
                   "eef_motion": 1000 * float(np.linalg.norm(z["rp_eef_after"][j][0] - z["rp_state"][j][:3]))}
            if j + 1 < len(ts):
                nxt = chunks[j + 1][0, :k, :6] / scale
                row["act_motion"] = float(np.linalg.norm((nxt - arm[0]).reshape(-1))) / np.sqrt(k)
            rows.append(row)
        if "br_t" in z.files:
            for j, t in enumerate(z["br_t"]):
                s = z["br_success"][j].astype(float)
                brows.append({"episode": e.episode, "init_state": e.init_state, "base_success": e.success,
                              "t": int(t), "n_cand": s.shape[0], "repeats": s.shape[1], "p_mean": s.mean(),
                              "p_best_insample": s.mean(1).max(), "p_worst_insample": s.mean(1).min(),
                              "var_between": var_between(s), "cv_gain": cv_gain(s),
                              "p_cand0": s[0].mean(), "p_scaled": s[1:].mean(), "cv_gain_vs_cand0": cv_gain_vs(s, 0),
                              "p_per_cand": " ".join(f"{v:.2f}" for v in s.mean(1))})
    sp, br = pd.DataFrame(rows), pd.DataFrame(brows)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    sp.to_csv(out / "spread.csv", index=False)
    br.to_csv(out / "branches.csv", index=False)
    sp["ratio_act"] = sp.act_spread / sp.act_motion
    sp["ratio_eef"] = sp.eef_spread / sp.eef_motion.clip(lower=1e-3)
    summ = {
        "cand_noise_scale": float(files[0]["cand_noise_scale"]) if "cand_noise_scale" in files[0].files else 1.0,
        "base_episodes": int(len(eps)), "base_success": float(eps.success.mean()),
        "restore_maxdiff_max": float(eps.restore_maxdiff.max()),
        "replan_points": int(len(sp)),
        "act_spread_median": float(sp.act_spread.median()), "act_motion_median": float(sp.act_motion.median()),
        "ratio_act_median": float(sp.ratio_act.median()),
        "eef_spread_mm_median": float(sp.eef_spread.median()), "eef_motion_mm_median": float(sp.eef_motion.median()),
        "ratio_eef_median": float(sp.ratio_eef.median()),
        "grip_split_frac": float(sp.grip_split.mean()),
        "by_phase": {f"{lo}-{hi}": {"eef_spread_mm": float(g.eef_spread.median()),
                                    "ratio_eef": float(g.ratio_eef.median()), "grip_split": float(g.grip_split.mean())}
                     for (lo, hi) in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 520)]
                     for g in [sp[(sp.t >= lo) & (sp.t < hi)]] if len(g)},
    }
    if len(br):
        summ.update({
            "branch_points": int(len(br)), "p_mean": float(br.p_mean.mean()),
            "var_between_mean": float(br.var_between.mean()), "sd_between": float(np.sqrt(max(br.var_between.mean(), 0))),
            "cv_gain_mean": float(br.cv_gain.mean()), "cv_gain_ci95": boot_ci(br.cv_gain),
            "insample_best_minus_mean": float((br.p_best_insample - br.p_mean).mean()),
            "p_cand0": float(br.p_cand0.mean()), "p_scaled": float(br.p_scaled.mean()),
            "cv_gain_vs_cand0_mean": float(br.cv_gain_vs_cand0.mean()), "cv_gain_vs_cand0_ci95": boot_ci(br.cv_gain_vs_cand0),
            "by_t": {int(t): {"p_mean": float(g.p_mean.mean()), "cv_gain": float(g.cv_gain.mean()),
                              "p_cand0": float(g.p_cand0.mean()), "cv_gain_vs_cand0": float(g.cv_gain_vs_cand0.mean()),
                              "n": int(len(g))}
                     for t, g in br.groupby("t")},
        })
    if len(br) and "offset_labels" in files[0].files:
        osum, od = offset_summary(br, files, eps)
        od.to_csv(out / "offsets.csv", index=False)
        summ["offset_probe"] = osum
        summ["cv_gain_ci95_ep"] = boot_ci_cluster(br, "cv_gain")
        summ["cv_gain_vs_cand0_ci95_ep"] = boot_ci_cluster(br, "cv_gain_vs_cand0")
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
