"""Advantage labels from the IQL value functions (Karen 2026-11-12: "test labels from the IQL tau 0.9 value function").

    .venv/bin/python scripts/libero_iql_labels.py [--iql reports/2026-11-11-iql-frozen] [--arms iql_tau0.9 ...]

V = the out-of-fold V of each arm in <iql>/oof_predictions.parquet (scripts/libero_iql.py: frozen ImageNet ResNet-18
features + proprio + t, 5 OOF folds = oof_split, every 4th frame of the 900 round-0 train rollouts). Labels = exactly
the rule of resnet_valuebinned_p40 (vb40, the labels behind the 75.4% residual): vl.libero_td.labels_from_v, n = 20,
value-binned, 40% positive over the pool [same rule; only V changes]. Check first: the same call on the parquet's
V_resnet column reproduces vb40 exactly (so any difference is the V).
Writes tables/moka_fixed_fresh/labels/<name>/{labels.parquet, README.txt}, name = iql09_vb40 for iql_tau0.9 etc.,
with V / A columns, so scripts/libero_value_vs_gt.py scores them like every other source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vl import libero_td as td

LABELS = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/labels")


def name_of(arm: str) -> str:
    return {"mc": "iqlmc_vb40"}.get(arm, "iql" + arm.split("tau")[-1].replace(".", "") + "_vb40")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iql", default="reports/2026-11-11-iql-frozen")
    ap.add_argument("--arms", nargs="+", default=["iql_tau0.9", "iql_tau0.7", "mc"])
    a = ap.parse_args()
    p = pd.read_parquet(Path(a.iql) / "oof_predictions.parquet").sort_values(td.KEY).reset_index(drop=True)
    d = p[td.KEY + ["T", "success"]]
    ref = pd.read_parquet(LABELS / "resnet_valuebinned_p40" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    chk = td.labels_from_v(d, p.V_resnet.to_numpy(float), 20)
    assert (chk[td.KEY].values == ref[td.KEY].values).all() and (chk.positive.values == ref.positive.values).all(), \
        "labels_from_v on V_resnet does not reproduce resnet_valuebinned_p40"
    print("check: V_resnet -> resnet_valuebinned_p40 reproduced exactly", flush=True)
    for arm in a.arms:
        lab = td.labels_from_v(d, p[f"V:{arm}"].to_numpy(float), 20)
        out = LABELS / name_of(arm)
        out.mkdir(parents=True, exist_ok=True)
        lab.to_parquet(out / "labels.parquet")
        agree = float((lab.positive.values == ref.positive.values).mean())
        succ = d.success.to_numpy() == 1
        info = {"source": f"{a.iql}/oof_predictions.parquet V:{arm}", "rule": "labels_from_v n 20 value-binned 40%",
                "positive": float(lab.positive.mean()), "positive_success_eps": float(lab.positive[succ].mean()),
                "positive_failure_eps": float(lab.positive[~succ].mean()), "frame_agreement_with_vb40": agree}
        (out / "README.txt").write_text(
            f"Labels from IQL OOF V ({arm}), scripts/libero_iql_labels.py, 2026-11-12. Same rule as "
            f"resnet_valuebinned_p40 (labels_from_v, n 20, value-binned, 40%); only V differs.\n{json.dumps(info, indent=2)}\n")
        print(name_of(arm), json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
