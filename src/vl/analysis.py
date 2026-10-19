"""Stage 4 breakdowns of a run's validation predictions (no GPU).

Usage:
    .venv/bin/python -m vl.analysis --run lowlr_8k --seeds lowlr_8k lowlr_8k_s1 lowlr_8k_s2
        -> reports/stage4_analysis_<run>.md      (tables only; interpretation lives in
                                                  reports/baseline_results.md)
           data/analysis/val_trajectory.parquet  (cached per-frame speed / human control)

Reads data/runs/<run>/val_predictions.parquet (best checkpoint), the manifest, the val
trajectories (end-effector position, `source`) and the Stage 1 operator-visibility audit.
Every grouping, threshold and window here is [ours]:

- kind: autonomous = eval rollout without human frames; assisted = DAgger rollout with a human
  takeover; dagger (no takeover); teleop demo. Frames of assisted episodes are split further by
  `source` (1 = human in control).
- timeout kind: soft-truncated by the reviewer at a stall; ran to the 800-step cap (T >= 799);
  recording ended early (neither).
- stall frame: robot in control, at least 3 s into the episode, and the end-effector's speed
  averaged over a centred 3 s window is below 15 mm/s. A heuristic for "the robot holds still"
  (Stage 1 stalls are slow 5-13 mm/s jitter at the holder, not zero), not a reviewed label.
- operator visible: the audit's DeepLabV3 person pixels in `wrist_left` (sampled every 15th
  frame, i.e. 1 fps). `any` = anywhere in the frame; `crop` = inside the audit's
  Resize(256)+CenterCrop(224) box, which is narrower than the training crop (full-height
  centre square), so `any` is the closer match to what the model sees.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vl.config import load_config
from vl.metrics import auc, score, within_repo_auc

FPS = 15
STALL_SPEED_MM_S = 15.0
STALL_WINDOW_S = 3


def load_trajectories(video_root: Path, val: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for repo, eps in val.groupby("repo").episode_index.unique().items():
        for f in sorted((video_root / repo / "data").rglob("*.parquet")):
            d = pd.read_parquet(f, columns=["episode_index", "frame_index", "source", "observation.state.cartesian_position"],
                                filters=[("episode_index", "in", [int(e) for e in eps])])
            if len(d):
                rows.append(d.assign(repo=repo))
    d = pd.concat(rows).sort_values(["repo", "episode_index", "frame_index"]).reset_index(drop=True)
    pos = np.stack(d.pop("observation.state.cartesian_position").to_numpy())[:, :3]
    step = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1) * FPS * 1000
    new_ep = d.episode_index.ne(d.episode_index.shift()) | d.repo.ne(d.repo.shift())
    step[new_ep.to_numpy()] = np.nan
    d["speed_mm_s"] = step
    w = STALL_WINDOW_S * FPS
    d["speed_3s"] = d.groupby(["repo", "episode_index"]).speed_mm_s.transform(
        lambda s: s.rolling(w, center=True, min_periods=w // 2).mean())
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(cache, index=False)
    return d


def annotate(pred: pd.DataFrame, man: pd.DataFrame, traj: pd.DataFrame, repo_median: dict) -> pd.DataFrame:
    m = man.set_index("episode_uid")
    p = pred.join(m[["source_type", "has_human_frames", "soft_truncate", "machine_failure_mode"]], on="episode_uid")
    p = p.merge(traj[["repo", "episode_index", "frame_index", "source", "speed_3s"]],
                on=["repo", "episode_index", "frame_index"], how="left")
    assert p.source.notna().all(), "val frames without a trajectory row"
    p["kind"] = np.select([(p.source_type == "eval") & ~p.has_human_frames,
                           (p.source_type == "dagger") & p.has_human_frames, p.source_type == "dagger"],
                          ["autonomous", "assisted", "dagger_no_takeover"], "teleop")
    p["timeout_kind"] = np.where(p.outcome != "timeout", "",
                                 np.where(p.soft_truncate, "soft_truncated", np.where(p["T"] >= 799, "cap", "ended_early")))
    p["t_s"] = p.frame_index / FPS
    p["to_T_s"] = (p["T"] - p.frame_index) / FPS
    p["stall"] = (p.source == 0) & (p.t_s >= 3) & (p.speed_3s < STALL_SPEED_MM_S)
    p["err"] = (p.V - p.value).abs()
    p["bias"] = p.V - p.value
    p["cerr"] = (p.repo.map(repo_median) - p.value).abs()
    p["repo_short"] = p.repo.str.replace("real-marker-d2-", "").str.split("-").str[0]
    return p


def group_table(p: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = p.groupby(by)
    t = pd.DataFrame({"episodes": g.episode_uid.nunique(), "frames": g.size(), "CE": g.ce.mean(), "MAE": g.err.mean(),
                      "bias (V-v)": g.bias.mean(), "repo-const MAE": g.cerr.mean()})
    ep = p.groupby(by + ["episode_uid"]).err.mean().groupby(by).mean()
    t.insert(4, "MAE per episode", ep)
    return t


def md(t: pd.DataFrame, digits: int = 3) -> str:
    return t.round(digits).to_markdown()


def seed_table(runs: list[str], runs_dir: Path) -> str:
    rows = {}
    for r in runs:
        f = runs_dir / r / "result.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        rows[r] = {"best step": res["best_step"], **{k: res["best_eval"][k] for k in
                   ["val_ce", "val_mae", "val_mae_per_episode", "eval_repos_mae", "offset_free_mae",
                    "eval_repos_auc_1s", "eval_repos_auc_5s", "eval_repos_auc_10s"]}}
    if not rows:
        return "(no finished seed runs)\n"
    t = pd.DataFrame(rows).T
    if len(t) > 1:
        t.loc["mean"] = t.mean()
        t.loc["sd"] = t.iloc[:-1].std(ddof=1)
    return md(t, 4) + "\n"


def stall_section(p: pd.DataFrame) -> str:
    out = []
    robot = p[p.source == 0]
    t = robot.groupby(["outcome", "stall"]).agg(frames=("err", "size"), MAE=("err", "mean"), bias=("bias", "mean"),
                                                 V=("V", "mean"), target=("value", "mean"))
    t["share of outcome's robot frames"] = t.frames / t.groupby(level=0).frames.transform("sum")
    out.append(md(t))
    # How much V moves inside a stall vs how much the target moves: per contiguous stall
    # segment of >= 3 s, the change of V and of v from start to end.
    s = p[p.stall].copy()
    s["seg"] = (s.frame_index.diff().ne(1) | s.episode_uid.ne(s.episode_uid.shift())).cumsum()
    seg = s.groupby("seg").agg(outcome=("outcome", "first"), n=("V", "size"), dV=("V", lambda x: x.iloc[-1] - x.iloc[0]),
                               dv=("value", lambda x: x.iloc[-1] - x.iloc[0]), V_sd=("V", "std"))
    seg = seg[seg.n >= STALL_WINDOW_S * FPS]
    t2 = seg.groupby("outcome").agg(segments=("n", "size"), mean_len_s=("n", lambda n: n.mean() / FPS),
                                    target_change=("dv", "mean"), V_change=("dV", "mean"), V_sd_within=("V_sd", "mean"))
    out.append("Stall segments of at least 3 s (change from the segment's first to last frame):\n\n" + md(t2))
    return "\n\n".join(out)


def operator_section(p: pd.DataFrame, op: pd.DataFrame) -> str:
    q = p.merge(op, on=["repo", "episode_index", "frame_index"], how="inner")
    q["vis"] = q.person_frac > 0
    q["vis_crop"] = q.person_frac_crop > 0
    out = [f"{len(q):,} val frames have an audit sample (1 fps), {q.episode_uid.nunique()} episodes.\n"]
    t = q.groupby(["outcome", "vis"]).agg(frames=("err", "size"), MAE=("err", "mean"), bias=("bias", "mean"),
                                           V=("V", "mean"), target=("value", "mean"), CE=("ce", "mean"))
    out.append(md(t))
    # Matched comparison: strata of (repo, outcome, 2 s of elapsed time); the visible-minus-not
    # difference within each stratum, averaged with the visible frames' counts as weights.
    q["tbin"] = (q.t_s // 2).astype(int)
    rows = []
    for col in ["vis", "vis_crop"]:
        for kind, sub in [("all", q), ("autonomous", q[q.kind == "autonomous"]), ("robot-control frames", q[q.source == 0])]:
            g = sub.groupby(["repo", "outcome", "tbin", col])[["V", "err", "bias", "value"]].mean().unstack(col)
            n = sub.groupby(["repo", "outcome", "tbin", col]).size().unstack(col)
            ok = n.notna().all(axis=1)
            w = n.loc[ok, True]
            d = {k: float(((g.loc[ok, (k, True)] - g.loc[ok, (k, False)]) * w).sum() / w.sum()) for k in ["V", "err", "bias", "value"]}
            rows.append({"operator": col, "frames": kind, "matched strata": int(ok.sum()), "visible frames": int(w.sum()),
                         "ΔV": d["V"], "Δtarget": d["value"], "ΔMAE": d["err"], "Δbias": d["bias"]})
    out.append("Matched (same repo, outcome and 2 s elapsed-time bin), visible minus not visible:\n\n"
               + md(pd.DataFrame(rows).set_index(["operator", "frames"])))
    # Sharp drops of V between consecutive 1 fps samples, and whether the operator just appeared.
    q = q.sort_values(["episode_uid", "frame_index"])
    prev = q.groupby("episode_uid")[["V", "vis", "value"]].shift()
    st = q.assign(dV=q.V - prev.V, appear=(~prev.vis.astype("boolean")) & q.vis, dv=q.value - prev.value).dropna(subset=["dV"])
    st = st[st.dv.abs() < 0.05]   # skip pairs straddling a gap
    rows = []
    for thr in [0.1, 0.2]:
        drop = st.dV < -thr
        rise = st.dV > thr
        rows.append({"|ΔV| > ": thr, "drops": int(drop.sum()), "P(appears | drop)": float(st.appear[drop].mean()),
                     "rises": int(rise.sum()), "P(appears | rise)": float(st.appear[rise].mean()),
                     "P(appears), all steps": float(st.appear.mean()),
                     "P(drop | appears)": float(drop[st.appear].mean()), "P(drop | no appearance)": float(drop[~st.appear].mean())})
    out.append("Sharp changes of V between consecutive 1 s samples vs the operator appearing "
               "(not visible at t-1 s, visible at t):\n\n" + md(pd.DataFrame(rows).set_index("|ΔV| > ")))
    return "\n\n".join(out)


def questions_section(p: pd.DataFrame) -> str:
    out = []
    auto = p[p.kind == "autonomous"]
    # Seat / terminal: V at fixed times before T.
    rows = []
    for o, g in auto.groupby("outcome"):
        r = {"outcome": o, "episodes": g.episode_uid.nunique()}
        for s in [5, 3, 1, 0]:
            w = g[(g.to_T_s >= s) & (g.to_T_s < s + 0.5)]
            r[f"V at T-{s}s"] = w.groupby("episode_uid").V.mean().mean()
        r["target at T"] = g[g.to_T_s == 0].value.mean()
        rows.append(r)
    out.append("Autonomous episodes, mean V (0.5 s window) before the endpoint T:\n\n" + md(pd.DataFrame(rows).set_index("outcome")))
    # Failure modes near T (VLM machine labels: diagnostics, not truth).
    f = p[(p.outcome != "success") & p.machine_failure_mode.notna()]
    rows = []
    for mode, g in f.groupby("machine_failure_mode"):
        e = g.groupby("episode_uid")
        last = g[g.to_T_s < 0.5].groupby("episode_uid").V.mean()
        before = g[(g.to_T_s >= 3) & (g.to_T_s < 3.5)].groupby("episode_uid").V.mean()
        rows.append({"machine_failure_mode": mode, "episodes": e.ngroups, "V mean": g.V.mean(),
                     "V at T-3s": before.mean(), "V at T": last.mean(), "MAE": g.err.mean()})
    out.append("Non-success episodes by VLM failure mode (machine label, diagnostic only):\n\n"
               + md(pd.DataFrame(rows).set_index("machine_failure_mode")))
    # Seated vs hovering at the holder: last 0.5 s of successes vs stall frames of timeouts.
    seated = auto[(auto.outcome == "success") & (auto.to_T_s < 0.5)].groupby("episode_uid").V.mean()
    hover = auto[(auto.outcome == "timeout") & auto.stall].groupby("episode_uid").V.mean()
    hover_late = auto[(auto.outcome == "timeout") & auto.stall & (auto.to_T_s < 0.5)].groupby("episode_uid").V.mean()
    y = np.r_[np.ones(len(seated)), np.zeros(len(hover))]
    out.append(f"Seated vs stalled at the holder (autonomous, one V per episode): seated = last 0.5 s of "
               f"successes, V {seated.mean():.3f} (n={len(seated)}); stalled = timeout stall frames, V "
               f"{hover.mean():.3f} (n={len(hover)}; last 0.5 s before T: {hover_late.mean():.3f}, n={len(hover_late)}). "
               f"AUC(seated > stalled) = {auc(y, np.r_[seated.to_numpy(), hover.to_numpy()]):.3f}.")
    # Late rounds: autonomous rollouts per eval repo.
    t = group_table(auto, ["repo_short"])
    t["success rate"] = auto.groupby("repo_short").apply(lambda g: (g.groupby("episode_uid").outcome.first() == "success").mean())
    for s_ in (1, 5, 10):
        t[f"AUC {s_}s"] = auto.groupby("repo_short").apply(lambda g: within_repo_auc(g, s_))
    out.append("Autonomous rollouts per eval round (AUC: V separates this round's successes from "
               "non-successes at fixed elapsed time):\n\n" + md(t))
    return "\n\n".join(out)


def shape_section(p: pd.DataFrame, repo_median: dict) -> str:
    """Jitter vs slow wander: the shape error (offset-free MAE) after smoothing V with a centred
    moving average. If smoothing brings it to the flat line's, the excess is frame-to-frame noise."""
    p = p.sort_values(["episode_uid", "frame_index"])
    dV = p.groupby("episode_uid").V.diff().abs()
    out = [f"Frame-to-frame |ΔV|: mean {dV.mean():.4f}, median {dV.median():.4f}; the target changes by "
           f"{1 / 120 / FPS:.5f} per frame (1/120 per second)."]
    rows = []
    for w in [1, 5, 15, 45]:
        q = p.assign(V=p.groupby("episode_uid").V.transform(lambda v: v.rolling(w, center=True, min_periods=1).mean()))
        sc = score(q, repo_median)
        rows.append({"smoothing window (frames)": w, "shape MAE": sc["offset_free_mae"], "flat line": sc["offset_free_mae_flat"],
                     "val MAE": sc["val_mae"]})
    out.append(md(pd.DataFrame(rows).set_index("smoothing window (frames)"), 4))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--train-config", default="configs/train.yaml")
    args = ap.parse_args()
    tc = load_config(args.train_config)
    dcfg = load_config(tc["data_config"])
    dscfg = load_config(dcfg["dataset_config"])
    runs_dir = Path(tc["paths"]["runs"])
    run = runs_dir / args.run
    idx = pd.read_parquet(dcfg["paths"]["frame_index"])
    rmed = idx[idx.split == "train"].groupby("repo").value.median().to_dict()
    man = pd.read_parquet(dscfg["paths"]["manifest"])
    pred = pd.read_parquet(run / "val_predictions.parquet")
    traj = load_trajectories(Path(dcfg["paths"]["video_root"]), pred, Path("data/analysis/val_trajectory.parquet"))
    p = annotate(pred, man, traj, rmed)
    op = pd.read_parquet("data/audit/operator_visibility_wrist_left.parquet")

    sec = [f"# Stage 4 analysis tables: `{args.run}` (generated by `vl.analysis`)\n",
           "Validation split, best checkpoint. MAE and bias are per frame unless labelled; "
           "repo-const MAE = each frame predicted by its repo's training median value.\n"]
    if args.seeds:
        sec += ["## Seeds (best checkpoint of each run)\n", seed_table(args.seeds, runs_dir)]
    sec += ["## By outcome and episode kind\n", md(group_table(p, ["kind", "outcome"])) + "\n",
            "## By source repo\n", md(group_table(p, ["repo_short"])) + "\n",
            "## Assisted episodes: human-controlled vs robot-controlled frames\n",
            md(group_table(p[p.kind == "assisted"].assign(control=lambda d: np.where(d.source == 1, "human", "robot")),
                           ["control", "outcome"])) + "\n"]
    tp = run / "train_predictions.parquet"
    if tp.exists():
        tr = pd.read_parquet(tp).assign(err=lambda d: (d.V - d.value).abs())
        gap = pd.DataFrame({"train (sample)": [tr.ce.mean(), tr.err.mean(), tr.groupby("episode_uid").err.mean().mean(), tr.episode_uid.nunique()],
                            "val": [p.ce.mean(), p.err.mean(), p.groupby("episode_uid").err.mean().mean(), p.episode_uid.nunique()]},
                           index=["CE", "MAE", "MAE per episode", "episodes"])
        gap["gap (val - train)"] = gap.val - gap["train (sample)"]
        sec += ["## Train/val gap\n", md(gap, 4) + "\n"]
    sec += ["## Timeouts by endpoint kind\n", md(group_table(p[p.outcome == "timeout"], ["timeout_kind"])) + "\n",
            f"## Stall frames (robot control, >= 3 s in, 3 s mean speed < {STALL_SPEED_MM_S:.0f} mm/s)\n",
            stall_section(p) + "\n",
            "## Shape: jitter vs slow wander\n", shape_section(p, rmed) + "\n",
            "## Operator visibility in wrist_left\n", operator_section(p, op) + "\n",
            "## Stage 4 questions\n", questions_section(p) + "\n"]
    out = Path(tc["paths"].get("reports", "reports")) / f"stage4_analysis_{args.run}.md"
    out.write_text("\n".join(sec))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
