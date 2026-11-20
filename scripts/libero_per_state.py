"""Per-init-state success across AG rounds [2026-11-16; mentor: "this analysis should be done for many more initial
states than just 10, so we get a better sense of if every state has >0% SR, or just most"; Karen: "set this up"].

Reads episodes.csv of eval runs (one per checkpoint), counts successes per init state, and reports:
  - a state x checkpoint table (successes / trials) -> per_state.csv
  - states at 0% for each checkpoint, with the 95% upper bound of the success rate (exact binomial, one-sided)
  - whether the weakest states climb: Spearman over checkpoints of each state's rate, and the mean rate of the
    bottom-k states at the first checkpoint, followed across the later ones (bottom k picked on the FIRST
    checkpoint only, so regression to the mean shows up as a climb; also reported: bottom k picked on split half A,
    scored on half B of every checkpoint's repeats, which removes that bias)
  - a heatmap (states sorted by mean rate) -> per_state.png

Usage (.venv/bin/python):
    scripts/libero_per_state.py --out reports/<dir> --runs base=<run dir> r0=<run dir> ... [--split val]
Runs keep their own state set; a state missing from a run is blank in the table.
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
    df = df[df["init_state"] >= 0]
    return df[["init_state", "repeat", "success"]].astype(int)


def upper95_zero(n: int) -> float:
    """One-sided 95% upper bound on p after 0 / n successes (exact: 1 - 0.05 ** (1 / n))."""
    return 1 - 0.05 ** (1 / n) if n else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir, in checkpoint order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bottom-k", type=int, default=5)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    labels, frames = [], {}
    for spec in a.runs:
        lab, d = spec.split("=", 1)
        labels.append(lab)
        frames[lab] = load(Path(d))

    states = sorted(set().union(*(set(f["init_state"]) for f in frames.values())))
    succ = pd.DataFrame(index=states, columns=labels, dtype=float)
    n = pd.DataFrame(index=states, columns=labels, dtype=float)
    for lab, f in frames.items():
        g = f.groupby("init_state")["success"]
        succ[lab] = g.sum()
        n[lab] = g.count()
    rate = succ / n

    rows = []
    for s in states:
        row = {"init_state": s}
        for lab in labels:
            row[f"{lab}_succ"] = succ.at[s, lab]
            row[f"{lab}_n"] = n.at[s, lab]
            row[f"{lab}_rate"] = rate.at[s, lab]
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out / "per_state.csv", index=False)

    summary = {"labels": labels, "states": states, "checkpoints": {}}
    for lab in labels:
        r = rate[lab].dropna()
        zero = [int(s) for s in r.index if r[s] == 0]
        summary["checkpoints"][lab] = {
            "pooled": float(succ[lab].sum() / n[lab].sum()),
            "n_states": int(r.size),
            "trials_per_state": sorted({int(x) for x in n[lab].dropna()}),
            "zero_states": zero,
            "zero_upper95": {str(s): round(upper95_zero(int(n.at[s, lab])), 3) for s in zero},
            "states_below_50": int((r < 0.5).sum()),
            "min_rate": float(r.min()),
        }

    # Does each state climb across checkpoints? Spearman of rate vs checkpoint index, per state.
    idx = np.arange(len(labels))
    trend = {}
    for s in states:
        y = rate.loc[s].to_numpy(float)
        ok = ~np.isnan(y)
        if ok.sum() >= 3 and np.ptp(y[ok]) > 0:
            trend[int(s)] = float(np.corrcoef(idx[ok], pd.Series(y[ok]).rank().to_numpy())[0, 1])  # Spearman (x has no ties)
    summary["per_state_spearman_vs_checkpoint"] = trend

    # Bottom-k states: picked on the first checkpoint (biased toward a climb), and split-half (unbiased).
    k = a.bottom_k
    first = labels[0]
    bottom = rate[first].dropna().sort_values().index[:k].tolist()
    summary["bottom_k_first"] = {"states": [int(s) for s in bottom],
                                 "mean_rate": {lab: float(rate.loc[bottom, lab].mean()) for lab in labels}}
    fa = frames[first]
    half_a = fa[fa["repeat"] % 2 == 0].groupby("init_state")["success"].mean()
    bottom_ha = half_a.sort_values().index[:k].tolist()
    split = {}
    for lab in labels:
        f = frames[lab]
        hb = f[(f["repeat"] % 2 == 1) & f["init_state"].isin(bottom_ha)]
        split[lab] = float(hb["success"].mean()) if len(hb) else None
    summary["bottom_k_split_half"] = {"picked_on": f"{first} even repeats", "scored_on": "odd repeats",
                                      "states": [int(s) for s in bottom_ha], "mean_rate": split}
    (a.out / "per_state_summary.json").write_text(json.dumps(summary, indent=1))

    order = rate.mean(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(labels), 0.8 + 0.28 * len(states)))
    m = rate.loc[order].to_numpy(float)
    im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(order)), [str(s) for s in order])
    for i, s in enumerate(order):
        for j, lab in enumerate(labels):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{int(succ.at[s, lab])}/{int(n.at[s, lab])}", ha="center", va="center",
                        fontsize=6, color="white" if m[i, j] < 0.6 else "black")
    ax.set_ylabel("init state (sorted by mean success)")
    ax.set_title(a.title or "success per init state")
    fig.colorbar(im, ax=ax, fraction=0.04, label="success rate")
    fig.tight_layout()
    fig.savefig(a.out / "per_state.png", dpi=150)

    for lab in labels:
        c = summary["checkpoints"][lab]
        print(f"{lab:>6}: pooled {c['pooled']:.3f}, {c['n_states']} states, min {c['min_rate']:.2f}, "
              f"<50% {c['states_below_50']}, at 0%: {c['zero_states'] or 'none'}")
    print("bottom-k (first ckpt):", {k_: round(v, 2) for k_, v in summary["bottom_k_first"]["mean_rate"].items()})
    print("bottom-k (split half):", {k_: (round(v, 2) if v is not None else None)
                                     for k_, v in summary["bottom_k_split_half"]["mean_rate"].items()})


if __name__ == "__main__":
    main()
