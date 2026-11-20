"""Noisy-judge robustness, offline half (Karen 2026-11-14: "2 now: offline, cheap"; plan entry 2026-11-14).

zima's rewards come from a VLM judge (Sol), not a simulator. Question: how much of residual Advantage Guidance's
gain survives when the episode outcomes it learns from are wrong at a judge's error rate? (FailBench, cited in
arXiv 2610.29677: the best of 13 VLM success detectors reaches 0.77 balanced accuracy; "Escaping Model Collapse via
Synthetic Data Verification", arXiv 2510.16657: with an imperfect verifier, self-training gains plateau or reverse.)

Simulation [ours]: each of the 900 round-0 train rollouts (tables/moka_fixed_fresh) gets one uniform draw u from an
rng keyed (seed, episode_index). Arm (fp, fn): a failure is judged a success when u < fp, a success is judged a
failure when u < fn. The same u serves every arm, so a lower rate flips a subset of what a higher rate flips. The
judged outcome then replaces the true one everywhere the round-0 chain uses it:
  1. the value targets: -(T - t)/1040 + v_T, v_T = 0 (judged success) / -0.5 (judged failure) (checked equal to
     frame_targets.parquet when nothing is flipped);
  2. the 5 out-of-fold ResNet V refits (vl.libero_resnet --oof-fold k --targets, round 0's recipe; val frames keep
     their true targets, so val metrics stay against the truth);
  3. the vb40 labels (vl.libero_td.labels_from_v, n 20, value-binned 40%), whose terminal frames read v_T.
Everything after the labels is round 0's residual recipe unchanged (scripts/judge_noise_chain.sh).

    .venv/bin/python scripts/libero_judge_noise.py targets            # flips + targets parquet per arm (CPU, ~1 min)
    .venv/bin/python scripts/libero_judge_noise.py check              # vb40 rebuilt from round 0's OOF runs (CPU)
    .venv/bin/python scripts/libero_judge_noise.py labels --arm sym25 # after the arm's 5 fold runs (CPU)

Writes tables/moka_fixed_fresh/judge_noise/<arm>/{flips.csv, targets.parquet, summary.json} and
tables/moka_fixed_fresh/labels/judge_<arm>_vb40/{labels.parquet, README.txt, summary.json}.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vl import libero_td as td
from vl.libero_advantages import NORM, V_FAIL

ROOT = Path("/iris/u/ankile/karenvo/libero")
TABLE = ROOT / "tables" / "moka_fixed_fresh"
OUT = TABLE / "judge_noise"
LABELS = TABLE / "labels"
RUNS = ROOT / "runs"
ARMS = {"sym10": (0.10, 0.10), "sym25": (0.25, 0.25), "fp25": (0.25, 0.0), "fn25": (0.0, 0.25)}
SEED = 0
RUN_PATTERN = "resnet18_judge_{arm}_oof_f{k}_s0_*"  # scripts/judge_noise_value.sbatch
ROUND0_PATTERN = "resnet18_oof_f{k}_s0_*"  # the OOF runs behind resnet_valuebinned_p40


def judge_draws(eps: pd.DataFrame) -> np.ndarray:
    return np.array([np.random.default_rng([SEED, int(e)]).random() for e in eps.episode_index])


def judged_outcomes(success: np.ndarray, u: np.ndarray, fp: float, fn: float) -> np.ndarray:
    s = np.asarray(success).astype(bool)
    return np.where(s, u >= fn, u < fp).astype(np.int8)


def value_targets(ft: pd.DataFrame, eps: pd.DataFrame, judged: pd.Series) -> pd.DataFrame:
    """Every train frame's MC target with the judged outcome (episode_index, frame_index, value)."""
    m = ft[["episode_index", "frame_index"]].merge(eps[["episode_index", "policy_steps"]], on="episode_index")
    vT = np.where(m.episode_index.map(judged).to_numpy() == 1, 0.0, V_FAIL)
    m["value"] = -((m.policy_steps - 1) - m.frame_index) / NORM + vT
    return m[["episode_index", "frame_index", "value"]]


def cmd_targets(_):
    eps = pd.read_csv(TABLE / "episodes.csv")
    eps = eps[eps.split == "train"].sort_values("episode_index").reset_index(drop=True)
    assert len(eps) == 900
    ft = pd.read_parquet(TABLE / "frame_targets.parquet", columns=["episode_index", "frame_index", "value"])
    ft = ft[ft.episode_index.isin(eps.episode_index)].sort_values(td.KEY).reset_index(drop=True)
    u = judge_draws(eps)
    clean = value_targets(ft, eps, eps.set_index("episode_index").success)
    assert np.abs(clean.value.to_numpy() - ft.value.to_numpy()).max() < 1e-12, "target formula != frame_targets"
    for arm, (fp, fn) in ARMS.items():
        j = judged_outcomes(eps.success.to_numpy(), u, fp, fn)
        flips = eps[["episode_index", "init_state", "success"]].assign(judged=j, u=u)
        tg = value_targets(ft, eps, flips.set_index("episode_index").judged)
        assert tg.value.between(-1, 0).all() and len(tg) == len(ft)
        s = flips.success == 1
        info = {"arm": arm, "fp": fp, "fn": fn, "seed": SEED, "episodes": len(flips),
                "true_success": float(s.mean()), "judged_success": float(flips.judged.mean()),
                "failures_judged_success": int(((~s) & (flips.judged == 1)).sum()), "failures": int((~s).sum()),
                "successes_judged_failure": int((s & (flips.judged == 0)).sum()), "successes": int(s.sum()),
                "balanced_accuracy": float(0.5 * ((flips.judged[s] == 1).mean() + (flips.judged[~s] == 0).mean())),
                "target_frames_changed": int((np.abs(tg.value.to_numpy() - ft.value.to_numpy()) > 1e-12).sum())}
        d = OUT / arm
        d.mkdir(parents=True, exist_ok=True)
        flips.to_csv(d / "flips.csv", index=False)
        tg.to_parquet(d / "targets.parquet")
        (d / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info), flush=True)


def oof_v(pattern: str) -> pd.DataFrame:
    """The 5 folds' held-out predictions, one V per train frame on the every-4th grid (as libero_oof_labels.build)."""
    parts = []
    for k in range(5):
        hits = sorted(h for h in glob.glob(str(RUNS / pattern.format(k=k)))
                      if (Path(h) / "oof_predictions.parquet").exists())
        assert len(hits) == 1, (pattern, k, hits)
        p = pd.read_parquet(Path(hits[0]) / "oof_predictions.parquet")
        assert (p.fold == k).all()
        parts.append(p.assign(run=Path(hits[0]).name))
    d = pd.concat(parts, ignore_index=True).sort_values(td.KEY).reset_index(drop=True)
    assert not d.duplicated(td.KEY).any() and d.episode_index.nunique() == 900
    return d


def cmd_check(_):
    d = oof_v(ROUND0_PATTERN)
    ref = pd.read_parquet(LABELS / "resnet_valuebinned_p40" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    lab = td.labels_from_v(d[td.KEY + ["T", "success"]], d.ev_resnet.to_numpy(float), 20)
    assert (lab[td.KEY].values == ref[td.KEY].values).all() and (lab.positive.values == ref.positive.values).all()
    print("check: round 0's OOF runs -> resnet_valuebinned_p40 reproduced exactly", flush=True)


def cmd_labels(a):
    arm = a.arm
    flips = pd.read_csv(OUT / arm / "flips.csv").set_index("episode_index")
    d = oof_v(RUN_PATTERN.format(arm=arm, k="{k}"))
    true_success = d.success.to_numpy()
    d["success"] = d.episode_index.map(flips.judged).to_numpy()  # the judge's outcome: the only one the chain sees
    assert (d.groupby("episode_index").success.nunique() == 1).all()
    lab = td.labels_from_v(d[td.KEY + ["T", "success"]], d.ev_resnet.to_numpy(float), 20)
    ref = pd.read_parquet(LABELS / "resnet_valuebinned_p40" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    assert (lab[td.KEY].values == ref[td.KEY].values).all()
    out = LABELS / f"judge_{arm}_vb40"
    out.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(out / "labels.parquet")
    flipped = d.episode_index.map(flips.judged).to_numpy() != true_success
    ts = true_success == 1
    info = {"arm": arm, "rule": "labels_from_v n 20 value-binned 40% (vb40's rule)", "runs": sorted(d.run.unique()),
            "positive": float(lab.positive.mean()), "frame_agreement_with_vb40": float((lab.positive.values ==
                                                                                       ref.positive.values).mean()),
            "positive_true_success_eps": float(lab.positive[ts].mean()),
            "positive_true_failure_eps": float(lab.positive[~ts].mean()),
            "positive_flipped_eps": float(lab.positive[flipped].mean()) if flipped.any() else None,
            "flipped_frames": int(flipped.sum()),
            "oof_mae_vs_true_target": float(np.abs(d.ev_resnet - d.value).mean())}
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / "README.txt").write_text(
        f"Noisy-judge labels, arm {arm} (fp, fn) = {ARMS[arm]}: scripts/libero_judge_noise.py, 2026-11-14. "
        f"OOF ResNet V refit on judged outcomes (judge_noise/{arm}/targets.parquet), then vb40's rule with the "
        f"judged outcome at terminal frames.\n{json.dumps(info, indent=2)}\n")
    print(json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("targets")
    sub.add_parser("check")
    p = sub.add_parser("labels")
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    a = ap.parse_args()
    {"targets": cmd_targets, "check": cmd_check, "labels": cmd_labels}[a.cmd](a)


if __name__ == "__main__":
    main()
