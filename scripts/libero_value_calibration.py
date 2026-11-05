"""Value-binned advantage calibration (Dissecting Advantage-Guided Post-Training, arXiv 2609.28161, Eq. 4) on the arm C
train rollouts, scored with our noise metrics and their ordering checks (2026-11-05, Karen: go).

Calibration [paper]: z = (A − μ_b) / max(σ_b, 0.1 σ_global), b = one of 8 equal-width bins over the frame's V; bins
with < 32 frames use the global statistics. Paper clips V to [0, 1]; ours lives in [−1, 0], so the bins span
[−1, 0] [ours: same 8 equal-width bins on our range]. Also "Global" (z = (A − μ) / σ, a no-op for a percentile tag)
as the reference. Tags from z: positive share 30% (RECAP pre-training) and 40% (RECAP fine-tuning) [paper: RECAP
App. A-F]; the current rule (70% positive) for comparison. Continuous weights exp(clip(z, −2, 2)) [Dissecting]
are written out for a weighting run.

Variants: {ResNet, expert} V × {raw A, smoothed V (k = 9, reports/2026-11-05-label-denoise)} × {global, value-binned}.
Metrics: those of scripts/libero_label_denoise.evaluate (ResNet–expert κ and r(A), flips/s, stall / drop / place
negative shares) plus
  eta2_subtask   share of A's variance explained by the subtask (pots on the stove 0 / 1; the paper's η²)
  R_prog         within-episode ordering accuracy: a moving frame's A above a stall frame's A (paper: advancing vs
                 stalled segments; ours: the stall flag of reports/2026-11-04-label-rules) [ties = 0.5]
  R_drop         within-episode ordering accuracy: the 20 steps before a pot falls score below the 20–60 steps before
  R_place        same for a successful first placement, expected sign reversed (placement window above)

Usage: .venv/bin/python scripts/libero_value_calibration.py --out reports/2026-11-05-value-calibration
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_advantage_examples as X  # noqa: E402
from libero_label_denoise import RULES, adv, evaluate, smooth  # noqa: E402

BINS, MIN_BIN, SIG_FLOOR = 8, 32, 0.1


def value_binned(A: np.ndarray, V: np.ndarray) -> np.ndarray:
    edges = np.linspace(-1, 0, BINS + 1)
    b = np.clip(np.digitize(V, edges[1:-1]), 0, BINS - 1)
    mu, sd = A.mean(), A.std()
    z = np.empty_like(A)
    for k in range(BINS):
        m = b == k
        if m.sum() >= MIN_BIN:
            z[m] = (A[m] - A[m].mean()) / max(A[m].std(), SIG_FLOOR * sd)
        else:
            z[m] = (A[m] - mu) / sd
    return z


def tag(z: np.ndarray, pos_share: float) -> np.ndarray:
    return (z > np.percentile(z, 100 * (1 - pos_share))).astype(np.int8)


def eta2(A: np.ndarray, g: np.ndarray) -> float:
    tot = ((A - A.mean()) ** 2).sum()
    return float(sum(((A[g == k].mean() - A.mean()) ** 2) * (g == k).sum() for k in np.unique(g)) / tot)


def pair_acc(hi: np.ndarray, lo: np.ndarray) -> float:
    """P(hi > lo) over all pairs, ties 0.5 (rank-based)."""
    if len(hi) == 0 or len(lo) == 0:
        return np.nan
    x = np.concatenate([hi, lo])
    r = pd.Series(x).rank().to_numpy()
    return float((r[: len(hi)].sum() - len(hi) * (len(hi) + 1) / 2) / (len(hi) * len(lo)))


def ordering(d: pd.DataFrame, A: np.ndarray, windows: dict) -> dict:
    prog, drop, place = [], [], []
    for e, idx in d.groupby("episode_index").indices.items():
        st = d.stall.to_numpy()[idx]
        mv = ~st & (d.event.to_numpy()[idx] == "") & ~d.term.to_numpy()[idx]
        if st.any() and mv.any():
            prog.append(pair_acc(A[idx][mv], A[idx][st]))
        for kind, (w_ev, w_pre) in windows.get(e, {}).items():
            for ev, pre in zip(w_ev, w_pre):
                if ev.any() and pre.any():
                    (drop if kind == "drop" else place).append(
                        pair_acc(A[idx][pre], A[idx][ev]) if kind == "drop" else pair_acc(A[idx][ev], A[idx][pre]))
    return {"R_prog": float(np.nanmean(prog)), "R_drop": float(np.nanmean(drop)) if drop else np.nan,
            "R_place": float(np.nanmean(place)) if place else np.nan, "n_prog_eps": len(prog), "n_drop": len(drop),
            "n_place": len(place)}


def event_windows(d: pd.DataFrame, eps: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Per episode: masks of the 20 steps before each debounced drop / successful first placement and of the
    20–60 steps before it; and per frame the debounced pots-on-stove count (subtask)."""
    out, pots = {}, np.zeros(len(d), np.int64)
    raw = eps.raw
    for e, idx in d.groupby("episode_index").indices.items():
        k = X.debounce(X.pots_on_stove(raw[e]))
        f = d.frame_index.to_numpy()[idx]
        pots[idx] = k[np.minimum(f, len(k) - 1)]
        succ = d.success.to_numpy()[idx[0]] == 1
        w = {"drop": ([], []), "place": ([], [])}
        for c in np.flatnonzero(np.diff(k) != 0) + 1:
            kind = "drop" if k[c] < k[c - 1] else ("place" if (k[c - 1] == 0 and k[c] == 1 and succ) else None)
            if kind:
                w[kind][0].append((f >= c - 20) & (f < c))
                w[kind][1].append((f >= c - 60) & (f < c - 20))
        out[e] = w
    return out, pots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    key = ["episode_index", "frame_index"]
    eps = pd.read_csv(X.TABLE / "episodes.csv").set_index("episode_index")
    r = pd.read_parquet(X.LAB / "oof_resnet_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    e = pd.read_parquet(X.LAB / "oof_expert_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    f = pd.read_parquet(RULES)[key + ["stall", "event"]].sort_values(key).reset_index(drop=True)
    assert (r[key].values == e[key].values).all() and (f[key].values == r[key].values).all()
    d = r[key + ["success", "init_state", "term"]].copy()
    d["T"] = d.episode_index.map(eps.policy_steps - 1).astype(float)
    d["stall"], d["event"] = f.stall.values, f.event.values
    windows, pots = event_windows(d, eps)
    d["pots"] = pots
    ep = d.episode_index.to_numpy()
    Vraw = {"resnet": r.V.to_numpy(float), "expert": e.V.to_numpy(float)}
    rows, frames = [], d[key].copy()
    for sm in (1, 9):
        A, Vs = {}, {}
        for src, V in Vraw.items():
            Vs[src] = smooth(V, ep, sm)
            A[src], term = adv(d, Vs[src], 20)
        if sm == 1:
            assert np.allclose(A["resnet"], r.A.to_numpy())
        for cal in ("global", "value"):
            z = {s: (A[s] - A[s].mean()) / A[s].std() if cal == "global" else value_binned(A[s], Vs[s]) for s in A}
            for pos in (0.7, 0.4, 0.3):
                name = f"{'smooth9' if sm > 1 else 'raw'}·{cal}·{int(pos * 100)}%pos"
                Lr, Le = tag(z["resnet"], pos), tag(z["expert"], pos)
                res = evaluate(d, z["resnet"], z["expert"], Lr, Le, term)
                res.update(ordering(d, z["resnet"], windows))
                res["eta2_subtask"] = eta2(z["resnet"], np.minimum(d.pots.to_numpy(), 1))
                rows.append({"variant": name, "smooth": sm, "calibration": cal, "pos_share": pos, **res})
                frames[name] = Lr
                print(name, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)
            if cal == "value":
                tagname = "smooth9" if sm > 1 else "raw"
                frames[f"z_resnet_{tagname}"] = z["resnet"].astype(np.float32)
                frames[f"z_expert_{tagname}"] = z["expert"].astype(np.float32)
                frames[f"w_resnet_{tagname}"] = np.exp(np.clip(z["resnet"], -2, 2)).astype(np.float32)
    frames["pots"] = d.pots.to_numpy()
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "calibration_table.csv", index=False)
    frames.to_parquet(out / "calibrated_labels.parquet")
    (out / "summary.json").write_text(json.dumps({"bins": BINS, "min_bin": MIN_BIN, "sigma_floor": SIG_FLOOR,
                                                  "variants": rows}, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["variant", "corr_A_interior", "kappa", "flips_per_s", "eta2_subtask", "R_prog", "R_drop", "R_place",
            "neg_stall", "neg_ordinary", "neg_drop", "neg_place_success"]
    print(tab[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
