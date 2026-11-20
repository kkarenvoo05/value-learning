"""Throughput next to success rate over the AG rounds, for the Week 9 Nov 17 page [2026-11-17; meeting next step "Add
throughput plot alongside success rate"].

Same 330 matched-noise val trials as the round gallery (scripts/libero_week9_gallery.py EVALS: base, r0 ... r7), plus
DSRL s1 at 1800 episodes as a reference (it samples its own noise, so it shares only the start states).
Left: success rate with a 95% Wilson interval. Right: steps to success (successful episodes only; 520 = time limit),
median with the interquartile band, and the median over the trials that succeed under both base and that policy (a
paired check: a later round also solves harder trials, which are slower, so the all-successes median can hide a
speed-up). Also written: successes per 1000 env steps over all episodes (failures run longer, so it combines both).
Out: reports/2026-11-17-ag-rounds/{fig9_throughput.png, throughput.json}
"""

from __future__ import annotations

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
REF = {"DSRL": "eval_dsrl_s1_agent1800_libero_10_18509148"}
BLUE, GREY, LIGHT = "#3b6fd8", "#8a8f98", "#9db7ee"


def load(d: str) -> dict:
    return {(int(r["init_state"]), int(r["repeat"])): (int(float(r["success"])), int(r["policy_steps"]))
            for r in csv.DictReader(open(RUNS / d / "episodes.csv"))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> None:
    E = {k: load(d) for k, d in {**EVALS, **REF}.items()}
    base = E["base"]
    res = {}
    for k, e in E.items():
        ok = [s for (o, s) in e.values() if o]
        both = [e[t][1] for t in e if e[t][0] and base[t][0]]
        both_base = [base[t][1] for t in e if e[t][0] and base[t][0]]
        lo, hi = wilson(len(ok), len(e))
        res[k] = {"n": len(e), "sr": 100 * len(ok) / len(e), "ci": [100 * lo, 100 * hi],
                  "steps_med": float(np.median(ok)), "steps_q": [float(np.percentile(ok, 25)), float(np.percentile(ok, 75))],
                  "paired_n": len(both), "paired_med": float(np.median(both)), "paired_base_med": float(np.median(both_base)),
                  "succ_per_1k_steps": 1000 * len(ok) / sum(s for (_, s) in e.values())}
    keys = list(EVALS)
    x = np.arange(len(keys))
    xr = len(keys) + 0.6  # reference column
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.9))
    sr = [res[k]["sr"] for k in keys]
    a.plot(x, sr, color=BLUE, lw=2, marker="o", ms=6, zorder=3)
    a.fill_between(x, [res[k]["ci"][0] for k in keys], [res[k]["ci"][1] for k in keys], color=BLUE, alpha=0.15, lw=0)
    d = res["DSRL"]
    a.errorbar([xr], [d["sr"]], yerr=[[d["sr"] - d["ci"][0]], [d["ci"][1] - d["sr"]]], color=GREY, marker="s", ms=6, capsize=3)
    for i in (0, 1, 6, 8):
        off = (10, -4) if i == 0 else (0, 8)
        a.annotate(f"{sr[i]:.1f}", (x[i], sr[i]), textcoords="offset points", xytext=off, ha="left" if i == 0 else "center",
                   fontsize=8.5, color="#333")
    a.annotate(f"{d['sr']:.1f}", (xr, d["sr"]), textcoords="offset points", xytext=(8, -3), fontsize=8.5, color="#333")
    a.set_ylabel("success rate (%)")
    a.set_ylim(40, 100)
    a.set_title("Success rate (330 matched val trials, 95% interval)", fontsize=10, loc="left")
    med = [res[k]["steps_med"] for k in keys]
    b.plot(x, med, color=BLUE, lw=2, marker="o", ms=6, zorder=3, label="median, all successes")
    b.fill_between(x, [res[k]["steps_q"][0] for k in keys], [res[k]["steps_q"][1] for k in keys], color=BLUE, alpha=0.15, lw=0,
                   label="interquartile range")
    b.plot(x[1:], [res[k]["paired_med"] for k in keys[1:]], color=LIGHT, lw=1.6, ls="--", marker="o", ms=4,
           label="median, trials base also solves")
    b.errorbar([xr], [d["steps_med"]], yerr=[[d["steps_med"] - d["steps_q"][0]], [d["steps_q"][1] - d["steps_med"]]],
               color=GREY, marker="s", ms=6, capsize=3)
    for i in (0, 8):
        b.annotate(f"{med[i]:.0f}", (x[i], med[i]), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8.5, color="#333")
    b.annotate(f"{d['steps_med']:.0f}", (xr, d["steps_med"]), textcoords="offset points", xytext=(10, -3), fontsize=8.5, color="#333")
    b.set_ylabel("steps to success (limit 520)")
    b.set_title("Throughput: how long successful episodes take", fontsize=10, loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper right")
    for ax in (a, b):
        ax.set_xticks(list(x) + [xr], keys + ["DSRL*"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#ddd", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, xr + 0.5)
    fig.text(0.01, -0.02, "* DSRL s1 at 1800 episodes; it samples its own noise, so it shares only the start states.",
             fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_throughput.png", dpi=150, bbox_inches="tight")
    (OUT / "throughput.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:5s} sr {v['sr']:5.1f}  med {v['steps_med']:.0f} [{v['steps_q'][0]:.0f},{v['steps_q'][1]:.0f}]  "
              f"paired {v['paired_med']:.0f} vs base {v['paired_base_med']:.0f} (n {v['paired_n']})  succ/1k {v['succ_per_1k_steps']:.2f}")


if __name__ == "__main__":
    main()
