"""V drift check per AG round (Karen 2026-11-12, idea 5): the round labels use round 0's OOF ResNet V on every new
episode, never refit (scripts/libero_round_labels.py). As the guided policy improves, that V may misjudge its states,
which would bias the labels. Per collection round, on its own new episodes:

  bias    mean(V - G) over frames, G = the realised Monte Carlo return-to-go (frame_targets value, the V's own
          training target: -(T - t)/1040 - 0.5 if the episode fails). > 0: V too optimistic, < 0: too pessimistic.
  corr    corr(V, G) over frames, and within episodes (V's trend vs the realised one)
  auc     does V at t = 100 / 200 / 300 rank the round's successes above its failures (episodes alive at t)
  start   mean V(t = 0) vs mean G(t = 0), and the round's success rate
Split by layout (fixed train states / fresh layouts). Round 0 rows (base pi0.5) are the reference: their V is OOF.

    .venv/bin/python scripts/libero_round_drift.py [--out reports/2026-11-12-round-drift]
Reads every tables/moka_agr_r<r>/labels/agr_vb40_r<r>/v.parquet present (the latest table holds all rounds so far).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("/iris/u/ankile/karenvo/libero/tables")
KEY = ["episode_index", "frame_index"]


def auc(score, y):
    s = pd.Series(score).rank().to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((s[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def latest():
    rs = sorted(int(p.name.split("_r")[-1]) for p in TABLES.glob("moka_agr_r*") if (p / "labels").exists())
    for r in reversed(rs):
        f = TABLES / f"moka_agr_r{r}" / "labels" / f"agr_vb40_r{r}" / "v.parquet"
        if f.exists():
            return r, TABLES / f"moka_agr_r{r}", f
    raise SystemExit("no round labels yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/2026-11-12-round-drift")
    a = ap.parse_args()
    r, table, vf = latest()
    eps = pd.read_csv(table / "episodes.csv")
    eps = eps[eps.split == "train"]
    v = pd.read_parquet(vf).merge(pd.read_parquet(table / "frame_targets.parquet", columns=KEY + ["value"]), on=KEY)
    v = v.merge(eps[["episode_index", "round", "layout", "success"]], on="episode_index")
    rows, lines = [], [f"latest table moka_agr_r{r}: rounds {sorted(v['round'].unique())} (table round 1 = round 0, base)",
                       "", "| AG round | layout | episodes | success | mean V(0) / G(0) | bias V - G | corr(V, G) | "
                       "within-episode corr | AUC V @100 / 200 / 300 |", "|---|---|---|---|---|---|---|---|---|"]
    for (rd, lay), g in v.groupby(["round", "layout"]):
        e = g.groupby("episode_index")
        within = e.apply(lambda x: x.V.corr(x.value) if x.V.std() > 0 and x.value.std() > 0 else np.nan).mean()
        start = g[g.frame_index == 0]
        aucs = {}
        for t in (100, 200, 300):
            x = g[g.frame_index == t]
            aucs[t] = auc(x.V.to_numpy(), x.success.to_numpy())
        row = {"ag_round": int(rd) - 1, "layout": lay, "episodes": int(g.episode_index.nunique()),
               "success": float(start.success.mean()), "V0": float(start.V.mean()), "G0": float(start.value.mean()),
               "bias": float((g.V - g.value).mean()), "corr": float(g.V.corr(g.value)), "within": float(within),
               "auc": aucs}
        rows.append(row)
        lines.append(f"| {row['ag_round']} | {lay} | {row['episodes']} | {row['success']:.2f} | {row['V0']:+.3f} / "
                     f"{row['G0']:+.3f} | {row['bias']:+.3f} | {row['corr']:+.2f} | {row['within']:+.2f} | "
                     + " / ".join(f"{x:.2f}" for x in aucs.values()) + " |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"table": str(table), "rows": rows}, indent=2))
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
