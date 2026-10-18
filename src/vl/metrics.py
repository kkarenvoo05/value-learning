"""Validation metrics shared by training runs and offline re-scoring (Stage 4).

Usage (re-score a finished run's saved predictions):
    .venv/bin/python -m vl.metrics --run baseline       # -> data/runs/<run>/metrics.json

Inputs are per-frame predictions with columns episode_uid, repo, outcome, frame_index, value
(target v), V (predicted), ce. Besides frame/episode CE and MAE, three repo-aware numbers
(reports/baseline_results.md, "repo shortcut"):

- repo_const_mae: MAE of the *repo-only constant* (each val frame predicted by its repo's
  training median value). Knows only which repo a frame is from, nothing in the image. The
  trivial baseline the model must beat.
- eval-repo metrics: the same MAEs restricted to the eval repos (r00-r04), where outcomes are
  mixed, plus the success AUC of V *within each repo* at fixed elapsed times (1, 5, 10 s),
  averaged over repos weighted by episode count. Repo identity cannot help within one repo.
- offset_free_mae: per episode, subtract the episode's mean from both V and v, then take
  the MAE. A constant per-episode (or per-repo) offset does not change it. It scores the
  *shape* of V within an episode, which is what V(o_{t+n}) - V(o_t) advantages use.
  offset_free_mae_flat is the same number for any constant (flat) prediction: the reference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vl.config import load_config

EVAL_REPOS_PREFIX = "real-marker-d2-r"


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC via ranks (ties get average rank); NaN if one class is missing."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_repo_auc(p: pd.DataFrame, t_s: float, fps: float = 15.0, half_window_s: float = 0.5) -> float:
    w = p[(p.frame_index / fps >= t_s - half_window_s) & (p.frame_index / fps < t_s + half_window_s)]
    q = w.groupby("episode_uid").agg(V=("V", "mean"), repo=("repo", "first"),
                                     s=("outcome", lambda o: int(o.iloc[0] == "success")))
    vals = [(auc(g.s.to_numpy(), g.V.to_numpy()), len(g)) for _, g in q.groupby("repo")]
    vals = [(a, n) for a, n in vals if not np.isnan(a)]
    return float(sum(a * n for a, n in vals) / sum(n for _, n in vals)) if vals else float("nan")


def repo_medians(train: pd.DataFrame) -> dict:
    return train.groupby("repo").value.median().to_dict()


def score(p: pd.DataFrame, repo_median: dict) -> dict:
    p = p.assign(err=(p.V - p.value).abs(), cerr=(p.repo.map(repo_median) - p.value).abs())
    dv = p.V - p.groupby("episode_uid").V.transform("mean")
    dt = p.value - p.groupby("episode_uid").value.transform("mean")
    p = p.assign(ofe=(dv - dt).abs(), ofe_flat=dt.abs())   # flat = any constant prediction
    ev = p[p.repo.str.startswith(EVAL_REPOS_PREFIX)]
    by_ep = p.groupby("episode_uid")[["ce", "err", "ofe", "ofe_flat"]].mean()
    out = {
        "val_ce": float(p.ce.mean()), "val_mae": float(p.err.mean()),
        "val_ce_per_episode": float(by_ep.ce.mean()), "val_mae_per_episode": float(by_ep.err.mean()),
        "repo_const_mae": float(p.cerr.mean()),
        "offset_free_mae": float(by_ep.ofe.mean()), "offset_free_mae_flat": float(by_ep.ofe_flat.mean()),
        "eval_repos_mae": float(ev.err.mean()), "eval_repos_repo_const_mae": float(ev.cerr.mean()),
    }
    for t in (1, 5, 10):
        out[f"eval_repos_auc_{t}s"] = within_repo_auc(ev, t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train-config", default="configs/train.yaml")
    args = ap.parse_args()
    tc = load_config(args.train_config)
    dcfg = load_config(tc["data_config"])
    run = Path(tc["paths"]["runs"]) / args.run
    idx = pd.read_parquet(dcfg["paths"]["frame_index"])
    m = score(pd.read_parquet(run / "val_predictions.parquet"), repo_medians(idx[idx.split == "train"]))
    (run / "metrics.json").write_text(json.dumps(m, indent=2))
    for k, v in m.items():
        print(f"{k:28s} {v:.4f}")


if __name__ == "__main__":
    main()
