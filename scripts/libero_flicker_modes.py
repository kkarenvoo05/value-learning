"""P2 (docs/libero_plan.md 2026-11-02): is V's within-episode flicker probability moving between the success and
failure lumps of the 201-bin distribution, or movement inside a lump?

Targets put successes in (-0.5, 0] and failures in [-1, -0.5] (vl.libero_targets), so with p = mass on bins > -0.5,
m_s / m_f = mean of the distribution inside / outside that range, E[V] = p m_s + (1 - p) m_f, and over any two frames
exactly  dE = dp * (mean m_s - mean m_f)  [between lumps]  +  mean p * dm_s + (1 - mean p) * dm_f  [within lumps]
(means over the two frames). Changes over 5 grid frames (2 s), interior frames as change_snr in
scripts/libero_head_swap.py; "noise" = each change minus its (frame index, outcome) mean, as there. Reports the noise
variance split Var(between) + Var(within) + 2 Cov, and the same split of the signal (success - failure mean change).

Usage: .venv/bin/python scripts/libero_flicker_modes.py  (CPU; runs with saved val_probs.npy)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = "/iris/u/ankile/karenvo/libero/runs/"
CENTERS = np.linspace(-1, 0, 201)
SUCC = CENTERS > -0.5
RUNS = {"frozen B ff s0": "value_expert_fixedfresh_w256_s0_18237898", "frozen B ff s1": "value_expert_fixedfresh_w256_s1_18237899",
        "frozen B ff s2": "value_expert_fixedfresh_w256_s2_18237900", "LoRA ff s0": "value_lora_fixedfresh_lr1e4_s0_18258391"}
LAG, T_MAX = 5, 340


def split(pred: pd.DataFrame, probs: np.ndarray) -> dict:
    P = probs.astype(np.float64)
    P /= P.sum(1, keepdims=True)
    p = P[:, SUCC].sum(1)
    ms = (P[:, SUCC] @ CENTERS[SUCC]) / np.maximum(p, 1e-12)
    mf = (P[:, ~SUCC] @ CENTERS[~SUCC]) / np.maximum(1 - p, 1e-12)
    q = pred.assign(p=p, ms=ms, mf=mf, E=P @ CENTERS).sort_values(["episode_index", "frame_index"])
    assert np.allclose(q.E, q.ev_expert, atol=2e-3), "probs do not match val_predictions row order"
    g = q.groupby("episode_index")
    d = {c: g[c].diff(LAG) for c in ("p", "ms", "mf", "E")}
    mean = {c: (q[c] + g[c].shift(LAG)) / 2 for c in ("p", "ms", "mf")}
    q = q.assign(between=d["p"] * (mean["ms"] - mean["mf"]),
                 within=mean["p"] * d["ms"] + (1 - mean["p"]) * d["mf"], dE=d["E"])
    q = q[q.dE.notna() & (q.frame_index <= T_MAX)]
    assert np.allclose(q.dE, q.between + q.within, atol=1e-9)
    out = {}
    key = ["frame_index", "success"]
    for c in ("between", "within", "dE"):
        q[c + "_n"] = q[c] - q.groupby(key)[c].transform("mean")
        m = q.groupby(key)[c].mean().unstack()
        out["signal_" + c] = float((m[1] - m[0]).mean())
    v_b, v_w, cov = q.between_n.var(), q.within_n.var(), np.cov(q.between_n, q.within_n)[0, 1]
    out.update(noise_sd=float(q.dE_n.std()), share_between=v_b / q.dE_n.var(), share_within=v_w / q.dE_n.var(),
               share_2cov=2 * cov / q.dE_n.var(), snr=out["signal_dE"] / float(q.dE_n.std()),
               mean_p_succ_lump=float(q.p.mean()))
    dp = d["p"].loc[q.index]
    out["sd_dp_noise"] = float((dp - dp.groupby([q.frame_index, q.success]).transform("mean")).std())
    if "ps_expert" in q.columns:  # the separate outcome head: same change-SNR on its P(success)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from libero_head_swap import change_snr
        out["snr_outcome_head"] = change_snr(pred, "ps_expert")[2]
        out["snr_p_succ_lump"] = change_snr(pred.assign(p_lump=p), "p_lump")[2]
    return out


def main():
    rows = {}
    for name, d in RUNS.items():
        pred = pd.read_parquet(R + d + "/val_predictions.parquet")
        rows[name] = split(pred, np.load(R + d + "/val_probs.npy"))
    t = pd.DataFrame(rows).T
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
