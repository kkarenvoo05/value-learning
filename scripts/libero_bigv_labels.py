"""vb40 labels for a chain table from a V fit on a bigger table [2026-11-19; Karen: "set up the big shared critic for
the multi task run"].

The big V (scripts/libero_chainv_round_labels.py run on --big-table: 5-fold OOF ResNet, fit on every train episode
there) gives each of its episodes one out-of-fold V. Every train episode of --table (the chain) must also be in
--big-table (matched by raw path; episode ids differ between tables); its V is copied over and the labels are
libero_round_labels.labels_all on --table's train episodes only (value-binned, n = 20, 40% cut within each (round,
task)), i.e. exactly what the chain labeler does, with only V's training set changed.
Writes <out>/labels.parquet, <out>/v.parquet, <out>/summary.json (corr / MAE vs the chain table's MC targets, per
(round, task) positive shares).

    .venv/bin/python scripts/libero_bigv_labels.py --big-v <big table>/labels/<name>/v.parquet --big-table <big table> \
        --table <chain table> --out <chain table>/labels/<label name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_round_labels import N, labels_all  # noqa: E402
from vl import libero_td as td  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-v", required=True)
    ap.add_argument("--big-table", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    table, out = Path(a.table), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    big = pd.read_csv(Path(a.big_table) / "episodes.csv", dtype={"goals_end": str})
    eps = pd.read_csv(table / "episodes.csv", dtype={"goals_end": str})
    if "round" not in eps.columns:
        eps["round"] = 1
    eps = eps[eps.split == "train"]
    bv = pd.read_parquet(a.big_v)
    big_id = big[big.split == "train"].set_index("raw").episode_index
    miss = ~eps.raw.isin(big_id.index)
    assert not miss.any(), f"{int(miss.sum())} chain episodes not in the big table's train split"
    idmap = dict(zip(eps.raw.map(big_id), eps.episode_index))
    v = bv[bv.episode_index.isin(idmap)].copy()
    v["episode_index"] = v.episode_index.map(idmap)
    assert set(v.episode_index) == set(eps.episode_index), "every chain episode needs a big-V prediction"
    ft = pd.read_parquet(table / "frame_targets.parquet", columns=td.KEY + ["value"])
    ft = ft[ft.episode_index.isin(eps.episode_index)]
    m = v[td.KEY].merge(ft, on=td.KEY, how="left")  # V lives on the every-4th grid, targets on every step
    assert len(m) == len(v) and m.value.notna().all(), "big-V frames missing from the chain table's targets"
    v = v.sort_values(td.KEY).reset_index(drop=True)
    v.to_parquet(out / "v.parquet")
    lab, rounds = labels_all(eps, v)
    lab.to_parquet(out / "labels.parquet")
    d = lab.merge(ft, on=td.KEY).merge(eps[["episode_index", "success", "round"] +
                                           (["task"] if "task" in eps.columns else [])], on="episode_index")
    keys = ["round", "task"] if "task" in d.columns else ["round"]
    by = {"|".join(map(str, k if isinstance(k, tuple) else (k,))): {
        "episodes": int(g.episode_index.nunique()), "frames": int(len(g)), "positive": float(g.positive.mean()),
        "positive_success_eps": float(g[g.success == 1].positive.mean()),
        "positive_failure_eps": float(g[g.success == 0].positive.mean()),
        "corr_V_mc": float(np.corrcoef(g.V, g.value)[0, 1]), "mae_V_mc": float((g.V - g.value).abs().mean())}
        for k, g in d.groupby(keys)}
    s = {"table": str(table), "big_table": a.big_table, "big_v": a.big_v,
         "V": "ResNet-18 OOF fit on the big table's train episodes (scripts/libero_chainv_round_labels.py)", "n": N,
         "rule": "value-binned (Dissecting Eq. 4), 40% positive cut within each (round, task)",
         "frames": int(len(lab)), "positive": float(lab.positive.mean()),
         "corr_V_mc": float(np.corrcoef(d.V, d.value)[0, 1]), "mae_V_mc": float((d.V - d.value).abs().mean()),
         "by_group": by}
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
