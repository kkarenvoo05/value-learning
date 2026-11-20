"""Strict success: count unstable pot placements as failures (2026-11-15; Karen: "set up the ... strict-success
relabel"; plan entry 2026-11-15).

Why: the largest single failure class of AG round 5 and DSRL is the first pot placed on the stove edge, then
drifting out of the cook region after release (scripts/libero_mentor_figures.py fig5). An edge placement is a coin
flip: if the pot stays, the episode counts as a success; if it slides, a failure. So the outcome labels disagree
exactly at the decision guidance should change. Measured on every train episode of tables/moka_agr_r7 (1600, rounds
1-8): P(success) is 0.76 when the first pot moves <= 1 cm in the 50 steps after release, 0.49 at 2-5 cm and 0.18
above 5 cm; 0.59 when the first pot leaves the cook region for 5-14 steps after settling. RECAP's largest gains came
from a stricter success definition. A VLM judge can apply the same rule ("both pots fully on the burner, not
sliding"), so the test transfers to zima.

Strict outcome [ours; thresholds from the curve above, where P(success) falls to about 0.5]:
  strict = success AND first-pot displacement in the 50 steps after release <= 2 cm
                   AND the first pot is off the cook region < 5 steps after it settles (10 steps after landing)
  release = the first step after landing with the hand > 5 cm from the pot. Episodes without a measurable release
  keep their outcome. The second pot cannot be judged: the episode ends the moment both pots are on.
Only the training outcome changes. Evals keep the simulator's success, so the comparison stays on the same metric.

The strict outcome replaces the true one everywhere the round-7 chain uses it (as scripts/libero_judge_noise.py for
a noisy judge):
  1. round-0 value targets (moka_fixed_fresh, 900 train episodes) -> the 5 OOF ResNet refits (vl.libero_resnet
     --oof-fold k --targets, round 0's recipe; run names resnet18_strict_oof_f{k}_s0_*);
  2. V for later rounds' episodes = the refit fold model that never saw the scene (libero_round_labels' rule, all
     rounds at once, which equals the round-by-round chain because each episode's V depends only on its fold model);
  3. vb40 labels with the strict outcome at terminal frames, 40% cut within each round.
Everything after the labels is round 7's residual recipe (scripts/strict_success_chain.sh).

    .venv/bin/python scripts/libero_strict_success.py outcomes        # CPU, ~2 min: outcomes + round-0 targets
    .venv/bin/python scripts/libero_strict_success.py check           # CPU: round 7's labels rebuilt from its v.parquet
    .venv/bin/python scripts/libero_strict_success.py labels [--check-v]   # GPU, after the 5 strict fold runs

Writes tables/moka_agr_r7/strict/{outcomes.csv, summary.json}, tables/moka_fixed_fresh/strict/targets.parquet,
tables/moka_agr_r7/labels/strict_vb40_r7/ and tables/moka_fixed_fresh/labels/strict_vb40/ (round 0 alone).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_judge_noise as J  # noqa: E402
import libero_round_labels as RL  # noqa: E402
from vl import libero_td as td  # noqa: E402

ROOT = Path("/iris/u/ankile/karenvo/libero")
TABLE = ROOT / "tables" / "moka_agr_r7"
TABLE0 = ROOT / "tables" / "moka_fixed_fresh"
OUT = TABLE / "strict"
REF = TABLE / "labels" / "agr_vb40_r7"
LAB = TABLE / "labels" / "strict_vb40_r7"
LAB0 = TABLE0 / "labels" / "strict_vb40"
STRICT_PATTERN = "resnet18_strict_oof_f{k}_s0_*"
SETTLE, WINDOW, RELEASE_M = 10, 50, 0.05
DISP_MAX_M, OFF_MAX = 0.02, 5   # strict failure: displacement > 2 cm, or off the cook region >= 5 steps


def placement(raw: str) -> dict:
    """First-pot placement stats from the raw rollout (goals[:, :2] = the two pots on the stove; object_pos =
    moka_pot_1 pos, pot 1 to eef, moka_pot_2 pos, pot 2 to eef)."""
    with np.load(raw) as z:
        g, op = z["goals"][:, :2].astype(bool), z["object_pos"]
    on = g.any(1)
    if not on.any():
        return {"first_pot": None}
    t1 = int(np.argmax(on))
    k = int(np.argmax(g[t1]))
    xy = op[:, [0, 1]] if k == 0 else op[:, [6, 7]]
    hand = np.linalg.norm(op[:, 3:6] if k == 0 else op[:, 9:12], axis=1)
    away = np.flatnonzero(hand[t1:] > RELEASE_M)
    rel = t1 + int(away[0]) if len(away) else None
    disp = None
    if rel is not None:
        disp = float(np.linalg.norm(xy[min(rel + WINDOW, len(g) - 1)] - xy[rel]))
    return {"first_pot": k, "landed": t1, "release": rel, "disp_50": disp,
            "off_steps": int((~g[t1 + SETTLE:, k]).sum())}


def strict_rule(success: int, disp, off: int) -> tuple[int, str]:
    if success != 1:
        return 0, "failure"
    if disp is not None and np.isfinite(disp) and disp > DISP_MAX_M:
        return 0, "moved"
    if off >= OFF_MAX:
        return 0, "off_region"
    return 1, "stable"


def cmd_outcomes(_):
    eps = pd.read_csv(TABLE / "episodes.csv", dtype={"goals_end": str}).sort_values("episode_index")
    rows = list(ThreadPoolExecutor(16).map(placement, eps.raw.tolist()))
    p = pd.DataFrame(rows, index=eps.index)
    o = pd.concat([eps[["episode_index", "round", "split", "init_state", "layout", "success"]], p], axis=1)
    rule = [strict_rule(int(s), d, int(f) if pd.notna(f) else 0)
            for s, d, f in zip(o.success, o.disp_50, o.off_steps)]
    o["strict"], o["reason"] = [r[0] for r in rule], [r[1] for r in rule]
    OUT.mkdir(parents=True, exist_ok=True)
    o.to_csv(OUT / "outcomes.csv", index=False)

    tr = o[o.split == "train"]
    s = tr.success == 1
    m = tr.disp_50.notna()
    curve = tr[m].groupby(pd.cut(tr.disp_50[m], [0, 0.01, 0.02, 0.05, 1.0]), observed=True).success.agg(
        ["size", "mean"])
    info = {"rule": f"success and disp_50 <= {DISP_MAX_M} m and off_steps < {OFF_MAX} (first pot)",
            "train_episodes": int(len(tr)), "successes": int(s.sum()), "strict_successes": int(tr.strict.sum()),
            "successes_flipped": int((s & (tr.strict == 0)).sum()),
            "flipped_share_of_successes": float((s & (tr.strict == 0)).sum() / s.sum()),
            "flip_reasons": tr[s & (tr.strict == 0)].reason.value_counts().to_dict(),
            "successes_without_release": int((s & tr.release.isna()).sum()),
            "by_round": {str(int(r)): {"episodes": int(len(g)), "success": float(g.success.mean()),
                                       "strict": float(g.strict.mean())} for r, g in tr.groupby("round")},
            "p_success_by_disp_50": {str(k): {"n": int(v["size"]), "p_success": float(v["mean"])}
                                     for k, v in curve.iterrows()}}
    (OUT / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)

    # round-0 value targets with the strict outcome (the 5 OOF refits read these)
    e0 = pd.read_csv(TABLE0 / "episodes.csv")
    e0 = e0[e0.split == "train"].sort_values("episode_index").reset_index(drop=True)
    assert len(e0) == 900
    ft = pd.read_parquet(TABLE0 / "frame_targets.parquet", columns=["episode_index", "frame_index", "value"])
    ft = ft[ft.episode_index.isin(e0.episode_index)].sort_values(td.KEY).reset_index(drop=True)
    st = o.set_index("episode_index").strict
    assert set(e0.episode_index) <= set(st.index)
    clean = J.value_targets(ft, e0, e0.set_index("episode_index").success)
    assert np.abs(clean.value.to_numpy() - ft.value.to_numpy()).max() < 1e-12, "target formula != frame_targets"
    tg = J.value_targets(ft, e0, st)
    assert tg.value.between(-1, 0).all() and len(tg) == len(ft)
    (TABLE0 / "strict").mkdir(exist_ok=True)
    tg.to_parquet(TABLE0 / "strict" / "targets.parquet")
    n0 = int(((e0.success == 1) & (e0.episode_index.map(st) == 0)).sum())
    print(f"round-0 targets: {n0} of {int(e0.success.sum())} successes flipped, "
          f"{int((np.abs(tg.value.to_numpy() - ft.value.to_numpy()) > 1e-12).sum())} frames changed "
          f"-> {TABLE0 / 'strict' / 'targets.parquet'}", flush=True)


def train_eps(outcome: str) -> pd.DataFrame:
    eps = pd.read_csv(TABLE / "episodes.csv", dtype={"goals_end": str})
    eps = eps[eps.split == "train"].copy()
    if outcome == "strict":
        st = pd.read_csv(OUT / "outcomes.csv").set_index("episode_index").strict
        eps["success"] = eps.episode_index.map(st).astype(int)
    return eps


def same_labels(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a, b = a.sort_values(td.KEY).reset_index(drop=True), b.sort_values(td.KEY).reset_index(drop=True)
    assert (a[td.KEY].values == b[td.KEY].values).all(), "label rows differ"
    return float((a.positive.values == b.positive.values).mean())


def cmd_check(_):
    """Round 7's labels from its own v.parquet with labels_from_v's per-round cut: the label path used below."""
    v = pd.read_parquet(REF / "v.parquet")
    lab, _ = RL.labels_all(train_eps("true"), v)
    agree = same_labels(lab, pd.read_parquet(REF / "labels.parquet"))
    assert agree == 1.0, agree
    print("check: agr_vb40_r7 labels reproduced exactly from its v.parquet", flush=True)


def all_v(runs: list[Path], eps: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """V per train frame for every round: round 0 from the folds' OOF predictions, later rounds from the fold model
    that never saw the scene."""
    v0 = RL.round0_v(runs)
    new = eps[eps["round"] > 1]
    folds = RL.new_episode_folds(new, RL.state_folds(pd.read_csv(TABLE0 / "episodes.csv")))
    pn = RL.predict_new(TABLE, new.episode_index.tolist(), folds, runs, cache)
    (cache / "frames_every4.npy").unlink(missing_ok=True)
    v = pd.concat([v0, pn[td.KEY + ["V"]]], ignore_index=True).sort_values(td.KEY).reset_index(drop=True)
    assert set(v.episode_index) == set(eps.episode_index)
    return v


def cmd_labels(a):
    eps = train_eps("strict")
    if a.check_v:  # the all-at-once V with round 0's runs must match round 7's chained v.parquet
        ref = pd.read_parquet(REF / "v.parquet").sort_values(td.KEY).reset_index(drop=True)
        tmp = LAB.parent / "_check_v"
        tmp.mkdir(parents=True, exist_ok=True)
        v = all_v(RL.oof_runs(), eps, tmp)
        assert (v[td.KEY].values == ref[td.KEY].values).all()
        dv = float(np.abs(v.V.to_numpy() - ref.V.to_numpy()).max())
        print(f"check-v: max |V - round 7's V| = {dv:.2e}", flush=True)
        assert dv < 1e-3, dv
        shutil.rmtree(tmp)
    runs = RL.oof_runs(STRICT_PATTERN)
    LAB.mkdir(parents=True, exist_ok=True)
    v = all_v(runs, eps, LAB)
    v.to_parquet(LAB / "v.parquet")
    lab, rounds = RL.labels_all(eps, v)
    lab.to_parquet(LAB / "labels.parquet")
    ref = pd.read_parquet(REF / "labels.parquet")
    o = pd.read_csv(OUT / "outcomes.csv").set_index("episode_index")
    m = lab.merge(ref[td.KEY + ["positive"]], on=td.KEY, suffixes=("", "_ref"), validate="one_to_one")
    assert len(m) == len(lab) == len(ref)
    flipped = m.episode_index.map((o.success == 1) & (o.strict == 0)).to_numpy(bool)
    info = {"runs": [r.name for r in runs], "rule": "vb40 (n 20, value-binned, 40% within each round), strict outcome",
            "frames": int(len(lab)), "positive": float(lab.positive.mean()),
            "frame_agreement_with_agr_vb40_r7": float((m.positive == m.positive_ref).mean()),
            "flipped_frames": int(flipped.sum()),
            "positive_on_flipped_eps": float(m.positive[flipped].mean()) if flipped.any() else None,
            "positive_on_flipped_eps_in_agr_vb40_r7": float(m.positive_ref[flipped].mean()) if flipped.any() else None}
    (LAB / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)

    # round 0 alone (moka_fixed_fresh): for an optional residual paired with round 0 (0.755)
    e1 = eps[eps["round"] == 1].assign(round=1)
    lab0, _ = RL.labels_all(e1, v[v.episode_index.isin(e1.episode_index)])
    LAB0.mkdir(parents=True, exist_ok=True)
    lab0.to_parquet(LAB0 / "labels.parquet")
    ref0 = pd.read_parquet(TABLE0 / "labels" / "resnet_valuebinned_p40" / "labels.parquet")
    (LAB0 / "README.txt").write_text(
        "Strict-success vb40 labels on round 0 alone: scripts/libero_strict_success.py labels, 2026-11-15. "
        f"OOF V refit on strict targets ({TABLE0 / 'strict' / 'targets.parquet'}); frame agreement with "
        f"resnet_valuebinned_p40 {same_labels(lab0, ref0):.4f}.\n")
    print(f"round-0 strict labels -> {LAB0}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("outcomes")
    sub.add_parser("check")
    p = sub.add_parser("labels")
    p.add_argument("--check-v", action="store_true", help="first rebuild round 7's V with round 0's runs (GPU)")
    a = ap.parse_args()
    {"outcomes": cmd_outcomes, "check": cmd_check, "labels": cmd_labels}[a.cmd](a)


if __name__ == "__main__":
    main()
