"""Sharper advantage labels on round 7: positive share 10 / 15 / 20% instead of 40% (2026-11-15; Karen: "cancel
strict and submit 10%, 15%, 20%").

Why: a general, task-agnostic knob for zima instead of a task-specific success rule (scripts/libero_strict_success.py,
not run). RECAP labels ~40% of frames positive early and ~10% once success is high; our rounds now succeed 80-85%,
so at 40% the + tag mostly separates ordinary successes from slightly better ones. A sharper cut keeps only clearly
better-than-expected frames positive. Risk: the advantage signal is weak (corr 0.04-0.14 with the sim's true
advantage) and fewer positives give the + condition less data.

Only the cut changes: the same V (round 7's chained v.parquet), the same n-step advantage and value bins
(vl.libero_td.labels_from_v, n 20, value-binned), the cut still taken within each round. No V refit.

    .venv/bin/python scripts/libero_threshold_labels.py      # CPU, seconds

Check: share 0.40 reproduces agr_vb40_r7 exactly. Writes tables/moka_agr_r7/labels/agr_vb{10,15,20}_r7/
{labels.parquet, summary.json}; each positive set is a subset of the next larger share's (same z, higher cut).

Adaptive arm (2026-11-15; Karen: "yes add the adaptive arm now") -> labels/agr_vbad_r7: each round's share = that
round's failure rate, clipped to [0.10, 0.40] [ours: a simple rule with RECAP's pattern, ~40% early and ~10% at
high success; not RECAP's exact rule]. Needs no per-task tuning: zima knows each night's judged success rate.
Same z as every other arm; only the per-round cut differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_round_labels as RL  # noqa: E402
from vl import libero_td as td  # noqa: E402

TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_agr_r7")
REF = TABLE / "labels" / "agr_vb40_r7"
SHARES = (0.10, 0.15, 0.20)
AD_MIN, AD_MAX = 0.10, 0.40


def per_round_cut(z: np.ndarray, rounds: np.ndarray, share: dict) -> np.ndarray:
    """labels_from_v's within-group cut with a share per group (same percentile rule)."""
    pos = np.zeros(len(z), np.int8)
    for r in np.unique(rounds):
        m = rounds == r
        pos[m] = z[m] > np.percentile(z[m], 100 * (1 - share[r]))
    return pos


def labels(eps: pd.DataFrame, v: pd.DataFrame, share: float) -> tuple[pd.DataFrame, np.ndarray]:
    """libero_round_labels.labels_all with the positive share as a parameter."""
    d = v[td.KEY + ["V"]].sort_values(td.KEY).reset_index(drop=True)
    e = eps.set_index("episode_index")
    d["T"] = d.episode_index.map(e.policy_steps - 1).to_numpy()
    d["success"] = d.episode_index.map(e.success).to_numpy()
    rounds = d.episode_index.map(e["round"]).to_numpy()
    return td.labels_from_v(d, d.V.to_numpy(float), RL.N, pos_share=share, groups=rounds), rounds


def main():
    eps = pd.read_csv(TABLE / "episodes.csv", dtype={"goals_end": str})
    eps = eps[eps.split == "train"]
    v = pd.read_parquet(REF / "v.parquet")
    ref = pd.read_parquet(REF / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    lab40, _ = labels(eps, v, 0.40)
    assert (lab40[td.KEY].values == ref[td.KEY].values).all() and (lab40.positive.values == ref.positive.values).all()
    print("check: share 0.40 reproduces agr_vb40_r7 exactly", flush=True)
    succ = lab40.episode_index.map(eps.set_index("episode_index").success).to_numpy() == 1
    prev = None
    for share in sorted(SHARES, reverse=True):
        lab, rounds = labels(eps, v, share)
        assert (lab[td.KEY].values == ref[td.KEY].values).all()
        pos = lab.positive.values.astype(bool)
        assert not (pos & ~ref.positive.values.astype(bool)).any(), "not a subset of the 40% positives"
        if prev is not None:
            assert not (pos & ~prev).any(), "not a subset of the larger share's positives"
        prev = pos
        out = TABLE / "labels" / f"agr_vb{round(share * 100)}_r7"
        out.mkdir(parents=True, exist_ok=True)
        lab.to_parquet(out / "labels.parquet")
        info = {"share": share, "source_v": str(REF / "v.parquet"),
                "rule": f"labels_from_v n {RL.N}, value-binned, {share:.0%} positive within each round",
                "frames": int(len(lab)), "positive": float(pos.mean()),
                "positive_success_eps": float(pos[succ].mean()), "positive_failure_eps": float(pos[~succ].mean()),
                "share_of_positives_from_successes": float(succ[pos].mean()),
                "share_of_positives_from_successes_at_40": float(succ[ref.positive.values == 1].mean()),
                "by_round": {str(int(r)): float(pos[rounds == r].mean()) for r in np.unique(rounds)}}
        (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps({k: info[k] for k in ("share", "positive", "positive_success_eps", "positive_failure_eps",
                                                "share_of_positives_from_successes",
                                                "share_of_positives_from_successes_at_40")}), flush=True)

    # adaptive: share per round = the round's failure rate, clipped
    lab, rounds = labels(eps, v, 0.40)
    assert (per_round_cut(lab.z.to_numpy(), rounds, {r: 0.40 for r in np.unique(rounds)}) ==
            lab.positive.to_numpy()).all(), "per-round cut != labels_from_v's cut"
    rate = eps.groupby("round").success.mean()
    share = {r: float(np.clip(1 - rate[r], AD_MIN, AD_MAX)) for r in np.unique(rounds)}
    lab["positive"] = per_round_cut(lab.z.to_numpy(), rounds, share)
    pos = lab.positive.values.astype(bool)
    out = TABLE / "labels" / "agr_vbad_r7"
    out.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(out / "labels.parquet")
    info = {"share": f"per round = failure rate clipped to [{AD_MIN}, {AD_MAX}]", "source_v": str(REF / "v.parquet"),
            "share_by_round": {str(int(r)): share[r] for r in share},
            "success_by_round": {str(int(r)): float(rate[r]) for r in share},
            "frames": int(len(lab)), "positive": float(pos.mean()),
            "positive_success_eps": float(pos[succ].mean()), "positive_failure_eps": float(pos[~succ].mean()),
            "share_of_positives_from_successes": float(succ[pos].mean()),
            "by_round": {str(int(r)): float(pos[rounds == r].mean()) for r in np.unique(rounds)}}
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
