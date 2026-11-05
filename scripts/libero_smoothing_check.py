"""Does temporal smoothing of V fix B's noisy advantages? (docs/libero_plan.md 2026-10-30, smoothing check) [ours].

Usage: .venv/bin/python scripts/libero_smoothing_check.py   -> reports/libero_smoothing.{md,json}

V per val episode is smoothed with a moving average over w grid frames (grid step 0.4 s; w = 1 is no smoothing),
centred (past and future frames; fine for offline labels but peeks up to (w-1)/2 frames ahead) or trailing (past
only). Applied to each seed of B (3b arm B) and of the ResNet, and to their 3-seed ensembles. Scores:
  - change SNR at 2 s, change_snr of scripts/libero_head_swap.py (signal = successes - failures, noise = sd)
  - Stage 5 labels, n = 2 s, 70% positive, mid-episode, time-matched succ - fail and its scene-bootstrap CI
    (vl.libero_advantages.checks, unchanged)
Reading rule, set before the result: if smoothed B reaches the unsmoothed ResNet's SNR (~0.27) / label gap (+5.4),
B's deficit is fast jitter and a smoothing step fixes it; if B stays well below while the ResNet also gains, the
noise is slow and smoothing does not close the gap.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/src")
sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/scripts")
import vl.libero_advantages as la  # noqa: E402
from libero_head_swap import change_snr  # noqa: E402

R = Path("/iris/u/ankile/karenvo/libero/runs")
SEEDS = {"B": (la.ENSEMBLES["B"], "ev_expert"), "ResNet": (la.ENSEMBLES["ResNet"], "ev_resnet")}
WINDOWS = [1, 3, 5, 7, 11]  # grid frames: 0 / ±0.4 / ±0.8 / ±1.2 / ±2 s centred
KEY = ["episode_index", "frame_index"]


def smooth(d: pd.DataFrame, col: str, w: int, mode: str) -> np.ndarray:
    g = d.groupby("episode_index")[col]
    if w == 1:
        return d[col].to_numpy(float)
    f = (lambda s: s.rolling(w, center=True, min_periods=1).mean()) if mode == "centred" else \
        (lambda s: s.rolling(w, min_periods=1).mean())
    return g.transform(f).to_numpy(float)


def main():
    d = la.load_val()  # sorted by episode, frame; has V_B, V_ResNet ensembles, value, T, success, ...
    for m, (runs, col) in SEEDS.items():
        for s, r in enumerate(runs):
            p = pd.read_parquet(R / r / "val_predictions.parquet")[KEY + [col]].rename(columns={col: f"V_{m}_s{s}"})
            d = d.merge(p, on=KEY, how="left", validate="1:1")
    d = d.sort_values(KEY).reset_index(drop=True)
    assert d.filter(like="_s").notna().all().all()
    rows, srcs = [], []
    for mode in ["centred", "trailing"]:
        for w in WINDOWS:
            if w == 1 and mode == "trailing":
                continue
            tag = "raw" if w == 1 else f"{mode} {w}"
            for m in SEEDS:
                snr = [change_snr(d.assign(v=smooth(d, f"V_{m}_s{s}", w, mode)), "v") for s in range(3)]
                name = f"{m} | {tag}"
                d[f"V_{name}"] = smooth(d, f"V_{m}", w, mode)  # ensemble, for the Stage 5 labels
                ens = change_snr(d, f"V_{name}")
                rows.append({"V": m, "smoothing": tag, "window_s": round((w - 1) * 0.4, 1),
                             "snr_seed_mean": float(np.mean([x[2] for x in snr])),
                             "snr_seed_sd": float(np.std([x[2] for x in snr])),
                             "signal_seed_mean": float(np.mean([x[0] for x in snr])),
                             "noise_seed_mean": float(np.mean([x[1] for x in snr])),
                             "snr_ensemble": ens[2]})
                srcs.append(name)
    la.SOURCES, la.NS = ["ResNet"] + srcs, [20]  # checks() compares every source with "ResNet"
    res, _ = la.checks(d, np.random.default_rng(0))
    res = res[res.threshold == la.MAIN_PCT].set_index("V")
    for r in rows:
        x = res.loc[f"{r['V']} | {r['smoothing']}"]
        r["label_tm_diff"], r["label_tm_ci"] = float(x.tm_diff), [float(v) for v in x.tm_diff_ci]
        r["label_tm_auc"], r["flips_per_s"] = float(x.tm_auc_scene), float(x.flips_per_s)
    t = pd.DataFrame(rows)
    Path("reports/libero_smoothing.json").write_text(json.dumps(rows, indent=2))
    fmt = t.assign(snr=t.apply(lambda r: f"{r.snr_seed_mean:.3f} ± {r.snr_seed_sd:.3f}", axis=1),
                   label=t.apply(lambda r: f"{100 * r.label_tm_diff:+.1f} [{100 * r.label_tm_ci[0]:+.1f}, "
                                           f"{100 * r.label_tm_ci[1]:+.1f}]", axis=1))
    cols = ["V", "smoothing", "window_s", "snr", "signal_seed_mean", "noise_seed_mean", "snr_ensemble", "label",
            "label_tm_auc", "flips_per_s"]
    md = ["# Smoothing check: B vs ResNet (val)", "", "`.venv/bin/python scripts/libero_smoothing_check.py`; "
          "definitions in the script docstring. *snr* = change SNR at 2 s, mean ± sd over 3 seeds (signal / noise "
          "in V units); *snr_ensemble* = the same on the 3-seed mean V; *label* = Stage 5 time-matched succ − fail "
          "(points, 95% scene CI), n = 2 s, 70% positive, from the ensemble.", "",
          fmt[cols].sort_values(["V", "smoothing"]).to_markdown(index=False, floatfmt=".3f"), ""]
    Path("reports/libero_smoothing.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
