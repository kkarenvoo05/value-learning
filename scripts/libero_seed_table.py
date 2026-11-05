"""Seed table for the Stage 3a grid / history tests (docs/libero_plan.md log 2026-10-29, "seeds") [ours].

Usage: .venv/bin/python scripts/libero_seed_table.py reports/libero_seeds_arms.json
ARMS.json: {arm: [[dir, ev_col, unused, results_key_or_null], ...per seed]}
Per seed: inner CE (selection metric; never val), val MAE, MAE success > 20 s, AUC 5/10/20/30 s and the milestone
time-matched excess of E[V] (vl.libero_milestone definitions). Prints mean +- sd over seeds and per-seed values.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/src")
from vl.libero_compare import auc_at  # noqa: E402
from vl.libero_milestone import FPS, HOLD, W, episodes, jump  # noqa: E402

VAL = episodes("val")
MS = VAL.set_index("episode_index").milestone
T = VAL.set_index("episode_index").policy_steps


def milestone_dd(pred: pd.DataFrame, col: str) -> float:
    by_ep = {e: g.sort_values("frame_index") for e, g in pred.groupby("episode_index")}
    treated = [e for e in by_ep if np.isfinite(MS[e]) and MS[e] - W >= 0 and MS[e] + W <= T[e]]
    dd = []
    for e in treated:
        m, g = MS[e], by_ep[e]
        ctrl = [c for c in by_ep if c != e and T[c] >= m + W and not (MS[c] < m + W + HOLD)]
        j = jump(g.frame_index.values, g[col].values, m)
        jc = np.nanmean([jump(by_ep[c].frame_index.values, by_ep[c][col].values, m) for c in ctrl]) if ctrl else np.nan
        dd.append(j - jc)
    return float(np.nanmean(dd))


def inner_ce(d: Path, key):
    r = json.loads((d / ("probe_results.json" if key else "results.json")).read_text())
    if key:
        return r[key]["inner_ce"]
    return min(h["inner_ce"] for h in r["history"])


def one(d, col, _ps, key):
    d = Path(d)
    p = pd.read_parquet(d / "val_predictions.parquet")
    err = (p[col] - p.value).abs()
    late = (p.success == 1) & (p.frame_index >= 20 * FPS)
    r = {"inner_ce": inner_ce(d, key), "mae": err.mean(), "mae_succ>20s": err[late].mean()}
    for s in (5, 10, 20, 30):
        r[f"auc{s}"] = auc_at(p, col, s)
    r["milestone"] = milestone_dd(p, col)
    return r


def main():
    arms = json.loads(Path(sys.argv[1]).read_text())
    rows, per = [], []
    for arm, seeds in arms.items():
        rs = [one(*s) for s in seeds]
        df = pd.DataFrame(rs)
        per.append(df.assign(arm=arm, seed=range(len(df))))
        rows.append({"arm": arm, "n": len(df), **{k: f"{df[k].mean():.3f} ± {df[k].std(ddof=1) if len(df) > 1 else 0:.3f}"
                                                   for k in df.columns}})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    print(pd.concat(per).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
