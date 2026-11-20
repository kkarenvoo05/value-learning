"""Success per layout on the continuous-layout eval set (eval_sets/moka_cont100; scripts/libero_eval_layouts.py)
[2026-11-16; mentor: "if every state has >0% SR, or just most"].

For each eval run (one per checkpoint): success per layout (source_raw -> layout_id), layouts at 0 / R, the share of
layouts that ever succeed, and success against the two pot positions (each layout's pot 1 / pot 2 xy coloured by its
success rate, one panel per checkpoint). Layouts at 0 / 3 are not proof of 0%: one-sided 95% upper bound 63% at 3 trials,
so the per-layout table flags candidates; the pooled share of 0 / R layouts is the number to compare across rounds.

    .venv/bin/python scripts/libero_cont_eval.py --out reports/<dir> --runs r0=<ceval run dir> r1=<run dir> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load(run: Path) -> pd.DataFrame:
    df = pd.read_csv(run / "episodes.csv")
    assert "source_raw" in df, f"{run}: not a --args.start-from eval"
    meta = {}
    for src in df.source_raw.unique():
        with np.load(src) as z:
            meta[src] = (int(z["layout_id"]), z["obj_pos"][:, :2].ravel())
    df["layout"] = df.source_raw.map(lambda s: meta[s][0])
    pos = pd.DataFrame({meta[s][0]: meta[s][1] for s in meta}, index=["p1x", "p1y", "p2x", "p2y"]).T
    return df[["layout", "success"]], pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir, in checkpoint order")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    labels, rate, n, pos = [], {}, {}, None
    for spec in a.runs:
        lab, d = spec.split("=", 1)
        df, p = load(Path(d))
        pos = p if pos is None else pos
        g = df.groupby("layout").success
        labels.append(lab)
        rate[lab], n[lab] = g.mean(), g.count()
    R = pd.DataFrame(rate)
    N = pd.DataFrame(n)
    pd.concat([pos, R.add_suffix("_rate"), N.add_suffix("_n")], axis=1).to_csv(a.out / "per_layout.csv")
    summ = {}
    for lab in labels:
        r = R[lab].dropna()
        summ[lab] = {"pooled": float((R[lab] * N[lab]).sum() / N[lab].sum()), "layouts": int(r.size),
                     "trials_per_layout": sorted({int(x) for x in N[lab].dropna()}),
                     "share_zero": float((r == 0).mean()), "share_full": float((r == 1).mean()),
                     "zero_layouts": [int(i) for i in r.index[r == 0]]}
    ever = R.max(axis=1)
    summ["_all"] = {"layouts_never_solved": [int(i) for i in ever.index[ever == 0]],
                    "share_never_solved": float((ever == 0).mean())}
    (a.out / "cont_summary.json").write_text(json.dumps(summ, indent=1))

    fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.4), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        r = R[lab].reindex(pos.index)
        for x, y, mk in (("p1x", "p1y", "o"), ("p2x", "p2y", "s")):
            sc = ax.scatter(pos[x], pos[y], c=r, cmap="viridis", vmin=0, vmax=1, marker=mk, s=18, edgecolors="none")
        ax.set_title(f"{lab}: {summ[lab]['pooled']:.2f}, 0/R {summ[lab]['share_zero']:.0%}", fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("y (m)  (o pot 1, s pot 2)")
    fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.03, label="layout success rate")
    fig.savefig(a.out / "cont_positions.png", dpi=150, bbox_inches="tight")
    for lab in labels:
        s = summ[lab]
        print(f"{lab:>6}: pooled {s['pooled']:.3f}, {s['layouts']} layouts, at 0: {s['share_zero']:.0%}, "
              f"all solved: {s['share_full']:.0%}")
    print("never solved by any checkpoint:", summ["_all"]["share_never_solved"])


if __name__ == "__main__":
    main()
