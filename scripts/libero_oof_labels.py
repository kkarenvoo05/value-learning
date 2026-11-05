"""Stage 5 -> 6: arm C advantage labels on the 900 train rollouts from out-of-fold V (docs/libero_plan.md).

Usage:
    .venv/bin/python scripts/libero_oof_labels.py --source resnet   # runs/resnet18_oof_f{0..4}_s0_*
    .venv/bin/python scripts/libero_oof_labels.py --source expert   # runs/value_expert_oof_f{0..4}_*

Each fold's run (vl.libero_resnet / scripts/pi05_value_expert.py --oof-fold k) predicted only the 180 train episodes
it neither fit nor early-stopped on (vl.libero_oof), so every train frame has exactly one out-of-sample V.
Advantage = vl.libero_advantages.advantage, unchanged [paper: RECAP n-step, no discount; v_T = 0 / -0.5].
n = 20 frames (2 s) [ours: the main row of the val checks]. Threshold = 30th percentile of these train advantages
(70% positive) [paper ambiguous; ours: computed on the labelled set itself, as the val checks did on val].
Terminal frames (lookahead reaches the end) are labelled as computed, not forced negative (Karen, 2026-11-02).

Writes <out>/labels.parquet (episode_index, frame_index, positive, A, term, V) + summary.json; the trainer
(vl.libero_finetune_data.frame_labels) floors every frame to the latest labelled frame <= t.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.libero_advantages import V_FAIL, advantage  # noqa: E402
from vl.metrics import auc  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
PATTERN = {"resnet": "resnet18_oof_f{k}_s0_*", "expert": "value_expert_oof_f{k}_*",
           "resnet_all": "resnet18_trainall_s0_*"}  # resnet_all: one model, train_predictions.parquet (see build)
COL = {"resnet": "ev_resnet", "expert": "ev_expert", "resnet_all": "ev_resnet"}


def build(source: str, n: int = 20, pct: float = 30.0, k_folds: int = 5):
    parts, runs = [], []
    if source == "resnet_all":  # train all / label all [ours, 2026-11-03 meeting]: in-sample V on every train frame
        hits = sorted(d for d in glob.glob(str(RUNS / PATTERN[source])) if (Path(d) / "train_predictions.parquet").exists())
        assert len(hits) == 1, hits
        runs.append(Path(hits[0]).name)
        parts.append(pd.read_parquet(Path(hits[0]) / "train_predictions.parquet").rename(columns={COL[source]: "V"}))
        k_folds = 0
    for k in range(k_folds):
        hits = sorted(d for d in glob.glob(str(RUNS / PATTERN[source].format(k=k)))
                      if (Path(d) / "oof_predictions.parquet").exists())
        assert len(hits) == 1, (k, hits)
        runs.append(Path(hits[0]).name)
        p = pd.read_parquet(Path(hits[0]) / "oof_predictions.parquet")
        assert (p.fold == k).all()
        parts.append(p.rename(columns={COL[source]: "V"}))
    d = pd.concat(parts, ignore_index=True).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    eps = pd.read_csv(TABLE / "episodes.csv")
    train = set(eps[eps.split == "train"].episode_index)
    per_ep = d.groupby("episode_index").fold.nunique()
    assert set(per_ep.index) == train and (per_ep == 1).all(), "every train episode exactly once"
    # (resnet_all: fold == -1 everywhere; the 75 inner-state episodes were used for early stopping but not fit)
    assert not d.duplicated(["episode_index", "frame_index"]).any()
    assert (d.groupby("episode_index").frame_index.min() == 0).all() and (d.frame_index % 4 == 0).all()
    t, T, ep = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.episode_index.to_numpy()
    vT = np.where(d.success == 1, 0.0, V_FAIL)
    A0, _ = advantage(d.value.to_numpy(float), t, T, vT, ep, n)
    assert np.abs(A0).max() < 1e-9, "target-as-V must give zero advantages"
    A, term = advantage(d.V.to_numpy(float), t, T, vT, ep, n)
    thr = float(np.percentile(A, pct))
    d["A"], d["term"], d["positive"] = A, term, (A > thr).astype(np.int8)
    return d, thr, runs


def summary(d: pd.DataFrame, thr: float, runs, source, n, pct) -> dict:
    succ, inter = d.success == 1, ~d.term
    share = lambda m: float(d.positive[m].mean())
    ep = d[inter].groupby("episode_index").agg(share=("positive", "mean"), succ=("success", "first"),
                                               init_state=("init_state", "first"))
    aucs, w = [], []
    for s, g in ep[ep.init_state >= 0].groupby("init_state"):
        if g.succ.nunique() == 2:
            aucs.append(auc(g.succ.values, g.share.values)); w.append(len(g))
    return {"source": source, "runs": runs, "n_frames_lookahead": n, "pct": pct, "threshold": thr,
            "frames": len(d), "episodes": int(d.episode_index.nunique()),
            "successes": int(d.groupby("episode_index").success.first().sum()),
            "positive": share(d.index == d.index),
            "positive_interior_success": share(inter & succ), "positive_interior_failure": share(inter & ~succ),
            "positive_terminal_success": share(d.term & succ), "positive_terminal_failure": share(d.term & ~succ),
            "within_scene_auc_interior_share": float(np.average(aucs, weights=w)), "scenes_for_auc": len(aucs),
            "oof_mae": float(np.abs(d.V - d.value).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(PATTERN), required=True)
    ap.add_argument("--n", type=int, default=20, help="lookahead in frames (10 Hz)")
    ap.add_argument("--pct", type=float, default=30.0, help="threshold percentile (30 = 70%% positive)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or TABLE / "labels" / f"oof_{a.source}_n{a.n}_p{int(a.pct)}")
    d, thr, runs = build(a.source, a.n, a.pct)
    s = summary(d, thr, runs, a.source, a.n, a.pct)
    out.mkdir(parents=True, exist_ok=True)
    d[["episode_index", "frame_index", "positive", "A", "term", "V", "success", "init_state", "fold"]].to_parquet(
        out / "labels.parquet")
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), f"\n-> {out}")


if __name__ == "__main__":
    main()
