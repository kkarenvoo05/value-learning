"""Relabel the 900 arm C train rollouts under alternative threshold rules and compare what flips (2026-11-04, Karen).

Same out-of-fold V and the same n = 20-step advantage as labels/oof_{resnet,expert}_n20_p30 (vl.libero_advantages.
advantage; scripts/libero_oof_labels.py). Only the rule turning A into a binary tag changes. Nothing here writes to the
training label tables; this is a comparison.

Rules:
  R0  current     A > 30th percentile of A (70% positive)                 [as trained; paper ambiguous, ours]
  R1  A > 0       A > 0                                                   [ours]
  R2  top 30%     A > 70th percentile of A (30% positive)                 [ours]
  R3  no cost     A' = A + time cost (n/1040 interior, (T-t)/1040 terminal), A' > 30th percentile of A' (70% pos)

Checks per rule (ResNet and expert V):
  positive share overall / in successes / in failures; within-scene AUC of the per-episode positive share (scenes
  with both outcomes; as libero_oof_labels.summary)
  events (20 steps before; debounced goal predicates, scripts/libero_advantage_examples.py): pot falls off the stove,
  first pot placed in a success / failure
  stalls [ours]: end effector moves < 1.5 cm over the next 20 steps and the gripper opening changes < 5 mm (normal
  motion: 2-18 cm per 20 steps); split into stalls of successes / failures
  label flips per second (label changes between consecutive value frames; 5 value frames per real second)

Usage: .venv/bin/python scripts/libero_label_rules.py --out reports/2026-11-04-label-rules
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import libero_advantage_examples as X  # noqa: E402
from vl.metrics import auc  # noqa: E402

NORM, N = 1040.0, 20
STALL_M, GRIP_M = 0.015, 0.005
RULES = {"R0": "current: A > p30 (70% pos)", "R1": "A > 0", "R2": "A > p70 (30% pos)", "R3": "no time cost, > p30"}
STEP_HZ = 20


def rule_labels(d: pd.DataFrame, A: np.ndarray) -> tuple[dict, dict]:
    t, T = d.frame_index.to_numpy(float), d["T"].to_numpy(float)
    cost = np.where(d.term.to_numpy(bool), (T - t) / NORM, N / NORM)
    Anc = A + cost
    thr = {"R0": np.percentile(A, 30), "R1": 0.0, "R2": np.percentile(A, 70), "R3": np.percentile(Anc, 30)}
    lab = {"R0": A > thr["R0"], "R1": A > 0, "R2": A > thr["R2"], "R3": Anc > thr["R3"]}
    return {k: v.astype(np.int8) for k, v in lab.items()}, {k: float(v) for k, v in thr.items()}


def frame_features(d: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """Per labelled frame: stall flag and event windows (20 steps before a debounced pots-on-stove change)."""
    raw = eps.set_index("episode_index").raw
    stall, ev = np.zeros(len(d), bool), np.full(len(d), "", object)
    for e, idx in d.groupby("episode_index").indices.items():
        with np.load(raw[e]) as z:
            s, goals = z["state"], z["goals"]
        k = X.debounce(X.subtask_labels(goals))
        f = d.frame_index.to_numpy()[idx]
        fut = np.minimum(f + N, len(s) - 1)
        move = np.linalg.norm(s[fut, :3] - s[f, :3], axis=1)
        grip = np.abs((s[fut, 6] - s[fut, 7]) - (s[f, 6] - s[f, 7]))
        stall[idx] = (move < STALL_M) & (grip < GRIP_M) & (f + N < len(s))
        for c in np.flatnonzero(np.diff(k) != 0) + 1:
            kind = "place_first" if (k[c - 1] == 0 and k[c] == 1) else ("drop" if k[c] < k[c - 1] else "other_up")
            w = (f >= c - N) & (f < c)
            ev[idx[w]] = kind
    return pd.DataFrame({"stall": stall, "event": ev}, index=d.index)


def checks(d: pd.DataFrame, lab: np.ndarray) -> dict:
    s = d.success.to_numpy() == 1
    L = pd.Series(lab, index=d.index)
    out = {"positive": float(L.mean()), "positive_success": float(L[s].mean()), "positive_failure": float(L[~s].mean())}
    ep = pd.DataFrame({"share": L, "succ": d.success, "scene": d.init_state, "e": d.episode_index}).groupby("e").agg(
        share=("share", "mean"), succ=("succ", "first"), scene=("scene", "first"))
    aucs, w = [], []
    for _, g in ep[ep.scene >= 0].groupby("scene"):
        if g.succ.nunique() == 2:
            aucs.append(auc(g.succ.values, g.share.values)); w.append(len(g))
    out["within_scene_auc"] = float(np.average(aucs, weights=w))
    for name, m in [("neg_before_drop", d.event == "drop"),
                    ("neg_before_first_place_success", (d.event == "place_first") & s),
                    ("neg_before_first_place_failure", (d.event == "place_first") & ~s),
                    ("neg_stall_success", d.stall & s), ("neg_stall_failure", d.stall & ~s),
                    ("neg_moving", ~d.stall & (d.event == ""))]:
        out[name] = float(1 - L[m].mean())
    flips = L.groupby(d.episode_index).apply(lambda x: (np.diff(x.to_numpy()) != 0).sum() / (len(x) / (STEP_HZ / 4)))
    out["flips_per_s"] = float(flips.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(X.TABLE / "episodes.csv")
    T = eps.set_index("episode_index").policy_steps - 1
    res, frames = {"rules": RULES, "stall_def": {"move_m": STALL_M, "grip_m": GRIP_M, "lookahead_steps": N}}, None
    for src in ("resnet", "expert"):
        d = pd.read_parquet(X.LAB / f"oof_{src}_n20_p30" / "labels.parquet").sort_values(
            ["episode_index", "frame_index"]).reset_index(drop=True)
        d["T"] = d.episode_index.map(T).astype(float)
        labs, thr = rule_labels(d, d.A.to_numpy(float))
        assert (labs["R0"] == d.positive.to_numpy()).all(), "R0 must reproduce the trained labels"
        if frames is None:
            frames = d[["episode_index", "frame_index"]].copy()
            feat = frame_features(d, eps)
            d0 = d
        d = d.join(feat)
        res[src] = {"thresholds": thr, "checks": {r: checks(d, labs[r]) for r in RULES}, "counts": {
            "stall_frames": int(d.stall.sum()), "stall_share": float(d.stall.mean()),
            "drop_frames": int((d.event == "drop").sum()), "place_first_frames": int((d.event == "place_first").sum())}}
        agree = {f"{r}_vs_R0_changed": float((labs[r] != labs["R0"]).mean()) for r in RULES if r != "R0"}
        res[src]["frames_changed_vs_R0"] = agree
        for r in RULES:
            frames[f"{src}_{r}"] = labs[r]
        frames[f"{src}_A"] = d.A.to_numpy()
    frames["stall"] = feat.stall.to_numpy(); frames["event"] = feat.event.to_numpy()
    frames.to_parquet(out / "rule_labels.parquet")
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    rows = []
    for src in ("resnet", "expert"):
        for r in RULES:
            rows.append({"V": src, "rule": r, "desc": RULES[r], "threshold": res[src]["thresholds"][r], **res[src]["checks"][r]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "rules_table.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string(index=False))
    print(json.dumps({s: res[s]["counts"] for s in ("resnet",)}), json.dumps({s: res[s]["frames_changed_vs_R0"] for s in ("resnet", "expert")}))


if __name__ == "__main__":
    main()
