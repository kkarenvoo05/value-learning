"""pi0.5 value-expert V through the AG residual's exact label rule (2026-11-14; Karen: "run the offline check").

    .venv/bin/python scripts/libero_expert_label_check.py [--out reports/2026-11-14-expert-labels]

Question: would the pi0.5 value expert's V give the AG residual better advantage labels than the ResNet-18 V it uses
now? The only earlier comparison (2026-11-10, reports/2026-11-10-value-vs-gt) put the pi0.5 V through the old p30 rule
(70% positive), not vb40 (the AG labels: vl.libero_td.labels_from_v, n 20, value-binned, 40% positive).
V sources (both 5-fold out-of-fold, every 4th frame of the 900 round-0 train rollouts, same units):
  ResNet  reports/2026-11-11-iql-frozen/oof_predictions.parquet V_resnet (check: reproduces resnet_valuebinned_p40)
  pi0.5   tables/moka_fixed_fresh/labels/oof_expert_n20_p30/labels.parquet V (the pi0.5 value expert, OOF)
Variants: n 20 / 80; V smoothed over a centred window of k = 5 / 9 value frames (scripts/libero_label_denoise.smooth;
the 11-05 denoise test: k 9 cut label flips 0.96 -> 0.34 / s); the mean of the two V's.
Score (as scripts/libero_value_vs_gt.py): corr(positive, A*) at the 424 branch points with A* (18 episodes),
bootstrap over episodes; plus the paired difference vs the current labels (same resamples).
Gate: a pi0.5 variant >= the current +0.10 -> one AG round-0 run with those labels; else keep the ResNet V.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_label_denoise import smooth  # noqa: E402
from libero_value_vs_gt import corr  # noqa: E402
from vl import libero_td as td  # noqa: E402

LABELS = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/labels")
POINTS = "reports/2026-11-12-iql-labels/points.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/2026-11-14-expert-labels")
    a = ap.parse_args()
    p = pd.read_parquet("reports/2026-11-11-iql-frozen/oof_predictions.parquet").sort_values(td.KEY).reset_index(drop=True)
    e = pd.read_parquet(LABELS / "oof_expert_n20_p30" / "labels.parquet")[td.KEY + ["V"]].rename(columns={"V": "V_expert"})
    p = p.merge(e, on=td.KEY, how="left")
    assert p.V_expert.notna().all(), "pi0.5 OOF V missing on some frames"
    d = p[td.KEY + ["T", "success"]]
    ep = d.episode_index.to_numpy()
    Vr, Ve = p.V_resnet.to_numpy(float), p.V_expert.to_numpy(float)
    ref = pd.read_parquet(LABELS / "resnet_valuebinned_p40" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    cur = td.labels_from_v(d, Vr, 20)
    assert (cur[td.KEY].values == ref[td.KEY].values).all() and (cur.positive.values == ref.positive.values).all()
    print("check: ResNet V, n 20 -> resnet_valuebinned_p40 reproduced exactly", flush=True)

    V = {"ResNet": Vr, "pi0.5": Ve, "mean(ResNet, pi0.5)": 0.5 * (Vr + Ve)}
    lab = {}
    for name, v in V.items():
        for k in (1, 5, 9):
            for n in (20, 80):
                if name.startswith("mean") and (k, n) != (1, 20):
                    continue
                key = f"{name} V" + (f", smoothed k {k}" if k > 1 else "") + f", n {n}"
                lab[key] = td.labels_from_v(d, smooth(v, ep, k), n).positive.to_numpy()
    CUR = "ResNet V, n 20"
    assert (lab[CUR] == ref.positive.values).all()

    pts = pd.read_csv(POINTS)[["ep", "t", "A_g"]].dropna()
    x = pd.DataFrame({"ep": ep, "t": d.frame_index, **lab}).merge(pts, on=["ep", "t"])
    eps = x.ep.unique()
    rng = np.random.default_rng(0)
    g = {k_: v_ for k_, v_ in x.groupby("ep")}
    samples = [pd.concat([g[i] for i in rng.choice(eps, len(eps))]) for _ in range(2000)]
    ci = lambda vals: [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    rows = []
    for key in lab:
        c = lambda y: corr(y[key].astype(float), y.A_g)
        bs = np.array([c(s) for s in samples])
        diff = np.array([c(s) - corr(s[CUR].astype(float), s.A_g) for s in samples])
        rows.append({"labels": key, "label_corr_A*": c(x), "ci95": ci(bs),
                     "minus_current": c(x) - corr(x[CUR].astype(float), x.A_g), "minus_current_ci95": ci(diff),
                     "pos_share_A*>0": float(x[x.A_g > 0][key].mean()), "pos_share_A*<0": float(x[x.A_g < 0][key].mean()),
                     "frame_agreement_with_current": float((lab[key] == lab[CUR]).mean())})
    info = {"points": int(len(x)), "episodes": int(len(eps)), "corr_V_resnet_V_expert": float(np.corrcoef(Vr, Ve)[0, 1]),
            "rows": rows}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    f = lambda v, c: f"{v:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]"
    lines = [f"points {info['points']} (A* from branching), episodes {info['episodes']}; "
             f"corr(V ResNet, V pi0.5) over all frames {info['corr_V_resnet_V_expert']:.2f}",
             "", "| labels (vb 40%) | label corr(A*) | minus current | + at A*>0 / <0 | agree w/ current |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['labels']} | {f(r['label_corr_A*'], r['ci95'])} | {f(r['minus_current'], r['minus_current_ci95'])} | "
                     f"{r['pos_share_A*>0']:.2f} / {r['pos_share_A*<0']:.2f} | {r['frame_agreement_with_current']:.0%} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
