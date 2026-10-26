"""Stage 5.5 B3: where does a history arm gain or lose against the no-memory baseline?

Usage:
    .venv/bin/python -m vl.history_gain                       # b2_gap2, b2_gap4 vs g0_r0
    .venv/bin/python -m vl.history_gain --arms b2_gap1        # -> reports/history_gain.md

Val split, best-checkpoint predictions of every seed (val_predictions.parquet, val_probs.npz).
Per frame: CE, V, p_succ (mass on bins above -0.5, as in `vl.flicker`). Each table reports the
reference's value and each arm's change with a 95% interval from the hierarchical bootstrap of
`scripts/compare_arms.py` (val episodes resampled, shared by all arms = paired; seeds resampled per
arm). Frame-weighted within a cell; '*' = interval excludes 0. Sections:
  1. hold phase by group [ours; the rule and groups of `vl.success_pauses`: eval repos, > 5 s in,
     > 2 s before T; stall = timeout with machine label `timeout_holding_marker`; frozen success =
     longest pause (gripper < 1 cm/s) >= 8 s]. Each group split by whether the gripper has been
     still >= 4 s at that frame. A stall gain should show as lower p_succ on stalls; the B1
     follow-up's risk is the same drop on frozen successes.
  2. time into the episode, all val frames: the first gap seconds have no past (masked), so a gain
     there is not from history; it also bounds the start-configuration confound.
  3. episode length terciles, all val frames: a gain concentrated in long episodes could come from
     length rather than stall detection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vl.history_plots import stall_seconds
from vl.success_pauses import kind

RUNS = Path("data/runs")
KEY = ["repo", "episode_index", "frame_index"]
FPS = 15
B = 2000
COLS = ["ce", "V", "p_succ"]
TIME_BINS_S = [0, 1, 2, 4, 10, 20, 40, np.inf]


def seeds(arm: str) -> list[int]:
    return sorted(int(p.name[len(arm) + 2:]) for p in RUNS.glob(f"{arm}_s*")
                  if p.name[len(arm) + 2:].isdigit() and (p / "val_predictions.parquet").exists())


def load(arms: list[str]) -> pd.DataFrame:
    """One row per val frame; columns <col>@<arm>_s<seed> for every run."""
    base, cols = None, {}
    for a in arms:
        for s in seeds(a):
            run = f"{a}_s{s}"
            p = pd.read_parquet(RUNS / run / "val_predictions.parquet")
            z = np.load(RUNS / run / "val_probs.npz", allow_pickle=True)
            assert (z["episode_uid"] == p.episode_uid.values).all() and (z["frame_index"] == p.frame_index.values).all()
            centers = np.linspace(-1, 0, z["probs"].shape[1])
            if base is None:
                base = p.drop(columns=["V", "ce"])
            assert (p.episode_uid.values == base.episode_uid.values).all() and (p.frame_index.values == base.frame_index.values).all()
            cols |= {f"ce@{run}": p.ce.values, f"V@{run}": p.V.values,
                     f"p_succ@{run}": z["probs"][:, centers > -0.5].sum(1)}
    d = pd.concat([base.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    man = pd.read_parquet("data/manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    d = d.merge(man, on="episode_uid", validate="m:1").merge(stall_seconds(), on=KEY, how="left", validate="1:1")
    assert d.stall_s.notna().all()
    d["t_s"] = d.frame_index / FPS
    d["to_go"] = (d["T"] - d.frame_index) / FPS
    return d


def hold_groups(d: pd.DataFrame) -> pd.DataFrame:
    """Hold-phase frames (eval repos, > 5 s in, > 2 s before T) with their B3 group and whether the
    gripper has been still >= 4 s at that frame. Needs columns repo, t_s, to_go, outcome,
    machine_failure_mode, episode_uid, stall_s (as `load` makes them). Shared with the targeted AUCs
    of `scripts/compare_arms.py`, so both use one definition."""
    h = d[d.repo.str.contains("-eval") & (d.t_s > 5) & (d.to_go > 2)].copy()
    k = kind(h.outcome, h.machine_failure_mode)
    longest = h.groupby("episode_uid").stall_s.transform("max")
    h["group"] = np.select([k == "stall", k == "other", longest >= 8, longest < 2],
                           ["stall", "other failure", "success, frozen (pause >= 8 s)", "success, pause < 2 s"],
                           "success, pause 2-8 s")
    h["still >= 4 s now"] = h.stall_s >= 4
    return h


def cell(d: pd.DataFrame, ref: str, arms: list[str], rng) -> dict:
    """Reference value and per-arm change, frame-weighted, with episode x seed bootstrap intervals."""
    g = d.groupby("episode_uid")
    n = g.size().to_numpy()
    runs = {a: [f"{a}_s{s}" for s in seeds(a)] for a in [ref, *arms]}
    sums = {c: g[[f"{c}@{r}" for a in runs for r in runs[a]]].sum() for c in COLS}
    idx = rng.integers(0, len(n), (B, len(n)))
    tot = n[idx].sum(1)
    out = {"episodes": len(n), "frames": int(n.sum())}
    for c in COLS:
        S = sums[c].to_numpy()                                    # episodes x runs
        col = {r: i for i, r in enumerate(sums[c].columns.str.split("@").str[1])}
        boot = {}
        for a, rs in runs.items():
            per_run = S[idx][:, :, [col[r] for r in rs]].sum(1) / tot[:, None]   # B x seeds
            pick = rng.integers(0, len(rs), (B, len(rs)))
            boot[a] = np.take_along_axis(per_run, pick, 1).mean(1)
            point = S[:, [col[r] for r in rs]].sum(0).mean() / n.sum()
            if a == ref:
                out[f"{c} {ref}"] = point
                ref_point = point
            else:
                lo, hi = np.percentile(boot[a] - boot[ref], [2.5, 97.5])
                out[f"Δ{c} {a}"] = f"{point - ref_point:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}"
    return out


def table(d: pd.DataFrame, by: list[str], ref: str, arms: list[str], rng) -> str:
    rows = [{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **cell(x, ref, arms, rng)}
            for k, x in d.groupby(by, observed=True)]
    t = pd.DataFrame(rows)
    cols = by + ["episodes", "frames"] + [c for col in COLS for c in t.columns if c.startswith((f"{col} ", f"Δ{col} "))]
    return t[cols].to_markdown(index=False, floatfmt=".3f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="g0_r0")
    ap.add_argument("--arms", nargs="+", default=["b2_gap2", "b2_gap4"])
    ap.add_argument("--report", default="reports/history_gain.md")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    d = load([args.ref, *args.arms])
    gaps = {a: json.loads((RUNS / f"{a}_s{seeds(a)[0]}" / "config.json").read_text()).get("history") for a in args.arms}

    h = hold_groups(d)
    d["time into episode (s)"] = pd.cut(d.t_s, TIME_BINS_S, right=False)
    T = d.groupby("episode_uid")["T"].transform("first") / FPS
    d["episode length"] = pd.qcut(T, 3, labels=["short", "medium", "long"])
    d["episode length"] = d["episode length"].astype(str) + d.groupby("episode length", observed=True)["T"].transform(
        lambda t: f" ({t.min() / FPS:.0f}-{t.max() / FPS:.0f} s)")
    d["outcome2"] = np.where(d.outcome == "success", "success", "failure")

    lines = ["# History gain breakdown (Stage 5.5 B3)", "",
             f"Generated by `.venv/bin/python -m vl.history_gain --ref {args.ref} --arms {' '.join(args.arms)}`; "
             "see the module docstring for definitions.", "",
             f"Reference `{args.ref}` (seeds {seeds(args.ref)}); arms "
             + ", ".join(f"`{a}` (seeds {seeds(a)}, history {gaps[a]})" for a in args.arms) + ". "
             "Δ = arm − reference, 95% interval over val episodes × seeds; '*' = excludes 0. "
             "p_succ = predicted mass on the success range (V > −0.5): should fall on stalls, rise on successes.", "",
             "## 1. Hold phase by group (eval repos)", "",
             table(h, ["group", "still >= 4 s now"], args.ref, args.arms, rng), "",
             "Same, groups pooled over stillness:", "", table(h, ["group"], args.ref, args.arms, rng), "",
             "## 2. Time into the episode (all val frames)", "",
             table(d, ["time into episode (s)"], args.ref, args.arms, rng), "",
             "## 3. Episode length terciles (all val frames)", "",
             table(d, ["episode length", "outcome2"], args.ref, args.arms, rng), ""]
    Path(args.report).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
