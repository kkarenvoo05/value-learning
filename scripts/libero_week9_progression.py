"""Trial progression base -> AG r0 ... r7 for the Week 9 Nov 17 page [2026-11-17; Karen: "what is the proportion of most
likely progression? is it most likely base failed then r7 succeeds? can you create a diagram that demonstrates the
proportion"].

Same 330 matched-noise val trials as the round gallery (scripts/libero_week9_gallery.py EVALS). Each trial is success or
failure in each of the 9 evals.
Left: alluvial diagram, one column per policy, success node on top and failure node below; bands are trials moving
between consecutive policies, colored by the trial's base outcome. Right: how many of the 8 rounds each trial
succeeds, split by base outcome.
Out: reports/2026-11-17-ag-rounds/fig8_progression.png, progression.json
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from libero_week9_gallery import EVALS, RUNS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-11-17-ag-rounds"
COL = {0: "#e07b39", 1: "#3b6fd8"}  # base failed / base succeeded (the trial's identity, kept through the rounds)
NODE = {1: "#3a9a5b", 0: "#8a8f98"}  # success / failure node
GAP = 34


def load() -> tuple[list[str], dict]:
    k = list(EVALS)
    s = {p: {(int(r["init_state"]), int(r["repeat"])): int(float(r["success"]))
             for r in csv.DictReader(open(RUNS / EVALS[p] / "episodes.csv"))} for p in k}
    return k, s


def band(ax, x0, x1, y0a, y0b, y1a, y1b, color):
    t = np.linspace(0, 1, 40)
    sm = t * t * (3 - 2 * t)
    x = x0 + (x1 - x0) * t
    ax.fill_between(x, y0a + (y1a - y0a) * sm, y0b + (y1b - y0b) * sm, color=color, alpha=0.45, lw=0)


def alluvial(ax, keys, S, trials):
    n = len(trials)
    W = 0.12  # node width
    # node extents: success on top [GAP+nfail, n+GAP], failure [0, nfail]
    def nodes(p):
        nf = sum(1 - S[p][t] for t in trials)
        return {0: (0.0, nf), 1: (nf + GAP, n + GAP)}
    ext = {p: nodes(p) for p in keys}
    for i, p in enumerate(keys):
        for o in (1, 0):
            lo, hi = ext[p][o]
            ax.add_patch(plt.Rectangle((i - W / 2, lo), W, hi - lo, color=NODE[o], lw=0))
            cnt = int(hi - lo)
            ax.text(i, hi + 6 if o == 1 else lo - 8, f"{'✓' if o else '✗'} {cnt}", ha="center",
                    va="bottom" if o == 1 else "top", fontsize=8.5, color="#333")
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        groups = collections.Counter((S["base"][t], S[a][t], S[b][t]) for t in trials)
        # outgoing stacking in node a: by destination (success first, top-down), then base group
        out_pos = {o: ext[a][o][1] for o in (0, 1)}
        in_pos = {o: ext[b][o][1] for o in (0, 1)}
        y_out, y_in = {}, {}
        for o in (1, 0):
            for d in (1, 0):
                for g in (1, 0):
                    c = groups[(g, o, d)]
                    y_out[g, o, d] = (out_pos[o] - c, out_pos[o]); out_pos[o] -= c
        for d in (1, 0):
            for o in (1, 0):
                for g in (1, 0):
                    c = groups[(g, o, d)]
                    y_in[g, o, d] = (in_pos[d] - c, in_pos[d]); in_pos[d] -= c
        for key, c in groups.items():
            if c:
                g = key[0]
                (a0, a1), (b0, b1) = y_out[key], y_in[key]
                band(ax, i + W / 2, i + 1 - W / 2, a0, a1, b0, b1, COL[g])
    ax.set_xticks(range(len(keys)), keys)
    ax.set_yticks([])
    ax.set_xlim(-0.75, len(keys) - 0.6)
    ax.set_ylim(-30, n + GAP + 30)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.1, n + GAP - 5, "success", fontsize=9, color=NODE[1], va="top", ha="right", fontweight="bold")
    ax.text(-0.1, 3, "failure", fontsize=9, color="#555", va="bottom", ha="right", fontweight="bold")
    ax.set_title("Each band = trials moving between consecutive policies (330 matched trials)", fontsize=10, loc="left")


def main() -> None:
    keys, S = load()
    trials = sorted(S["base"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    alluvial(a, keys, S, trials)
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[0], alpha=0.6), plt.Rectangle((0, 0), 1, 1, color=COL[1], alpha=0.6)],
             labels=["trial failed under base", "trial succeeded under base"], fontsize=8.5, frameon=False,
             loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    rounds = keys[1:]
    cnt = {g: collections.Counter(sum(S[p][t] for p in rounds) for t in trials if S["base"][t] == g) for g in (0, 1)}
    x = np.arange(len(rounds) + 1)
    for g, off in ((0, -0.2), (1, 0.2)):
        v = [cnt[g][i] for i in x]
        b.bar(x + off, v, width=0.38, color=COL[g], alpha=0.85,
              label=f"base failed (n={sum(v)})" if g == 0 else f"base succeeded (n={sum(v)})")
    b.set_xticks(x)
    b.set_xlabel(f"rounds succeeded (of {len(rounds)}, r0–r7)")
    b.set_ylabel("trials")
    b.legend(fontsize=8.5, frameon=False, loc="upper left")
    b.set_title("How consistently each trial succeeds over the rounds", fontsize=10, loc="left")
    b.spines[["top", "right"]].set_visible(False)
    b.grid(axis="y", color="#ddd", lw=0.6)
    b.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_progression.png", dpi=150, bbox_inches="tight")
    br7 = collections.Counter(f"{S['base'][t]}{S['r7'][t]}" for t in trials)
    paths = collections.Counter("".join(str(S[p][t]) for p in keys) for t in trials)
    trans = {f"{p}->{q}": dict(collections.Counter(f"{S[p][t]}{S[q][t]}" for t in trials)) for p, q in zip(keys, keys[1:])}
    res = {"base_to_r7": dict(br7), "rounds_succeeded_by_base": {str(g): dict(sorted(cnt[g].items())) for g in cnt},
           "distinct_paths": len(paths), "top_paths": paths.most_common(10), "transitions": trans}
    (OUT / "progression.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["base_to_r7"]), res["distinct_paths"])


if __name__ == "__main__":
    main()
