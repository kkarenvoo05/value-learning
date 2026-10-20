"""Pre-end flicker of V and of the outcome belief, per run and per arm (Stage 5 augmentation, Steps 0 and 2).

Usage (CPU; needs data/runs/<run>/val_probs.npz from vl.value_dist):
    .venv/bin/python -m vl.flicker --arms lowlr_8k,lowlr_8k_s1,lowlr_8k_s2 aug_photo_s0,aug_photo_s1,aug_photo_s2 ...
        -> reports/flicker_stage5.md

Each --arms argument is one arm: comma-separated runs (seeds). Arm label = first run name.
Runs without val_probs.npz are skipped (and listed). Metrics, all on validation frames more than
2 s before T (the "before the end" region; reports/stage5_augmentation_plan.md, Step 0), with
frame-to-frame differences taken only between consecutive frames that are both in that region:

- |dV|: mean |V_t - V_{t-1}|. Target slope is 1/1800 per frame (the target barely moves).
- |dp_succ|: mean change of p_succ = mass on bins above -0.5 (the success range).
- TV: mean total-variation distance 0.5 * sum_b |p_t(b) - p_{t-1}(b)|.
- |V(t)-V(t-1s)|: mean change over 1 s (report only, not in the Step 2 rule). Target moves 0.0083 per s.
- R2(hump flip): share of dV explained by dp_succ * (V_if_succ - V_if_fail).
- hump MAE | success: on success episodes, |V_if_succ - v|. Does the success hump track the target?
- MAE, shape MAE: |V - v| and the offset-free (per-episode mean removed) MAE on the same frames.
- seated AUC: AUC(seated > stalled) as in vl.analysis (autonomous; all frames, not only pre-end).

Smoothing reference [ours]: the baseline's V and p_succ under a centred moving average over 0.5 s
(8 frames) and 1 s (15 frames) within each episode. Centred = uses future frames; fine for offline
advantages, not for a live controller. Augmentation is only worth it if it beats this for free.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vl.analysis import md
from vl.metrics import auc

PRE_END_S = 2.0
FPS = 15.0


def load(run: Path, v: np.ndarray) -> pd.DataFrame:
    z = np.load(run / "val_probs.npz")
    pred = pd.read_parquet(run / "val_predictions.parquet")
    assert (pred.frame_index.to_numpy() == z["frame_index"]).all()
    assert (pred.episode_uid.to_numpy().astype(str) == z["episode_uid"]).all()
    P = z["probs"].astype(np.float32)
    P /= P.sum(1, keepdims=True)
    s = v > -0.5
    pred["p_succ"] = P[:, s].sum(1)
    pred["V_if_succ"] = (P[:, s] * v[s]).sum(1) / np.maximum(pred.p_succ, 1e-6)
    pred["V_if_fail"] = (P[:, ~s] * v[~s]).sum(1) / np.maximum(1 - pred.p_succ, 1e-6)
    order = np.lexsort((pred.frame_index.to_numpy(), pred.episode_uid.to_numpy()))
    pred, P = pred.iloc[order].reset_index(drop=True), P[order]
    same = (pred.episode_uid.shift() == pred.episode_uid) & (pred.frame_index.diff() == 1)
    tv = np.r_[np.nan, 0.5 * np.abs(np.diff(P, axis=0)).sum(1)]
    pred["tv"] = np.where(same, tv, np.nan)
    pred["prev_ok"] = same
    return pred


def smooth(pred: pd.DataFrame, frames: int) -> pd.DataFrame:
    g = pred.groupby("episode_uid")
    out = pred.copy()
    for c in ["V", "p_succ", "V_if_succ", "V_if_fail"]:
        out[c] = g[c].transform(lambda x: x.rolling(frames, center=True, min_periods=1).mean())
    out["tv"] = np.nan                       # the smoothed distribution is not defined per bin here
    return out


def metrics(pred: pd.DataFrame) -> dict:
    pre = (pred["T"] - pred.frame_index) / FPS > PRE_END_S
    pair = pre & pred.prev_ok & pre.shift(fill_value=False)
    d = pred[["V", "p_succ"]].diff()[pair]
    gap = (pred.V_if_succ - pred.V_if_fail)[pair]
    from_p = d.p_succ * gap
    r2 = 1 - ((d.V - from_p) ** 2).sum() / ((d.V - d.V.mean()) ** 2).sum()
    lag = int(FPS)                           # 1 s: slower wander, which a 0.5-1 s average cannot remove
    pair1 = pre & pre.shift(lag, fill_value=False) & (pred.episode_uid.shift(lag) == pred.episode_uid)
    d1 = (pred.V - pred.V.shift(lag))[pair1]
    q = pred[pre]
    dv = q.V - q.groupby("episode_uid").V.transform("mean")
    dt = q.value - q.groupby("episode_uid").value.transform("mean")
    shape = (dv - dt).abs().groupby(q.episode_uid).mean().mean()
    succ = q[q.outcome == "success"]
    auto = pred[pred.kind == "autonomous"]
    to_T = (auto["T"] - auto.frame_index) / FPS
    seated = auto[(auto.outcome == "success") & (to_T < 0.5)].groupby("episode_uid").V.mean()
    hover = auto[(auto.outcome == "timeout") & auto.stall].groupby("episode_uid").V.mean()
    return {"|dV|": d.V.abs().mean(), "|dp_succ|": d.p_succ.abs().mean(), "TV": pred.tv[pair].mean(),
            "|V(t)-V(t-1s)|": d1.abs().mean(),
            "R2(hump flip)": r2, "hump MAE | success": (succ.V_if_succ - succ.value).abs().mean(),
            "MAE": (q.V - q.value).abs().mean(), "shape MAE": shape,
            "seated AUC": auc(np.r_[np.ones(len(seated)), np.zeros(len(hover))], np.r_[seated, hover])}


def context(runs_dir: Path) -> pd.DataFrame:
    """Episode kind and stall flag per val frame, from vl.analysis (same definitions)."""
    from vl.analysis import annotate, load_trajectories
    from vl.config import load_config
    tc = load_config("configs/train.yaml")
    dcfg = load_config(tc["data_config"])
    man = pd.read_parquet(load_config(dcfg["dataset_config"])["paths"]["manifest"])
    pred = pd.read_parquet(runs_dir / "lowlr_8k" / "val_predictions.parquet")
    traj = load_trajectories(Path(dcfg["paths"]["video_root"]), pred, Path("data/analysis/val_trajectory.parquet"))
    return annotate(pred, man, traj, {})[["episode_uid", "frame_index", "kind", "stall"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="one per arm: comma-separated run names")
    ap.add_argument("--runs-dir", default="data/runs")
    ap.add_argument("--out", default="reports/flicker_stage5.md")
    args = ap.parse_args()
    v = np.linspace(-1, 0, 201)
    rows, missing = [], []
    ctx = context(Path(args.runs_dir))
    for arm in args.arms:
        runs = arm.split(",")
        for r in runs:
            if not (Path(args.runs_dir) / r / "val_probs.npz").exists():
                missing.append(r)
                continue
            pred = load(Path(args.runs_dir) / r, v)
            n = len(pred)
            pred = pred.merge(ctx, on=["episode_uid", "frame_index"], how="left")
            assert len(pred) == n and pred.kind.notna().all()
            rows.append({"arm": runs[0], "run": r, **metrics(pred)})
            if r == "lowlr_8k":
                for k, lab in [(8, "lowlr_8k smoothed 0.5 s"), (15, "lowlr_8k smoothed 1 s")]:
                    rows.append({"arm": lab, "run": lab, **metrics(smooth(pred, k))})
    t = pd.DataFrame(rows)
    per_arm = t.drop(columns="run").groupby("arm", sort=False)
    mean, sd, n = per_arm.mean(), per_arm.std(), per_arm.size()
    summ = mean.map(lambda x: f"{x:.4f}").astype(str)
    for c in mean.columns:
        summ[c] = [f"{m:.4f} ± {s:.4f}" if k > 1 else f"{m:.4f}" for m, s, k in zip(mean[c], sd[c].fillna(0), n)]
    summ.insert(0, "seeds", n)
    lines = ["# Pre-end flicker (generated by `vl.flicker`)\n",
             f"Validation frames more than {PRE_END_S:.0f} s before T; frame-to-frame changes between consecutive "
             "frames both in that region. Mean ± sd over seeds. Smoothed rows: centred moving average of the "
             "seed-0 baseline (uses future frames). See the module docstring for definitions.\n",
             md(summ) + "\n", "## Per run\n", md(t.set_index("run").drop(columns="arm").round(4)) + "\n"]
    if missing:
        lines.append(f"Not yet available (no val_probs.npz): {', '.join(missing)}.\n")
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
