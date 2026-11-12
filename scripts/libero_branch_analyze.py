"""Analyze scripts/libero_branch_value.py: ground-truth value V* of recorded frames vs the learned value and labels.

Per branch point t of a recorded train episode: K base-policy continuations -> success s_k, end step e_k.
  V*(t)  = mean_k s_k                                    P(success | s_t), the quantity the labels try to rank
  G*(t)  = mean_k [ -(e_k - t)/1040 - 0.5 (1 - s_k) ]     the ResNet's own MC target (src/vl/libero_advantages.py),
                                                         so it is compared with OOF V in the same units
  A*(t)  = G*(t + 20) - G*(t) - 20/1040                   n = 20 advantage, the label formula with V replaced by G*

Reports [ours]:
  fidelity      replay state / pixel differences, replay outcome = recorded outcome
  calibration   V*(t = 0) vs the init state's success rate over its 15 recorded rollouts
  decided       where V* moves: the largest |dV*| between neighbouring points per episode, with a split-half
                reliability (continuations 1..K/2 vs K/2+1..K) so a jump is not mistaken for binomial noise
  value         OOF ResNet V vs G* (corr, MAE) on the branch grid
  labels        does the learned advantage / do the labels rank A*? corr(A_oof, A*) and the positive share of the
                labels at A* > 0 vs < 0, with the split-half reliability of A* as the ceiling
Writes <run>/analysis.json and <run>/curves.png.

Usage: .venv/bin/python scripts/libero_branch_analyze.py --run /iris/u/ankile/karenvo/libero/runs/branch_value_libero_10_<job>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
NORM, FAIL, N = 1040.0, -0.5, 20
LABELS = {"oof_resnet_n20_p30": "old arm C (70%)", "resnet_valuebinned_p40": "faithful RECAP (vb 40%)",
          "scoped_valuebinned_p40": "scoped vb 40%"}


def g_target(success, end, t):
    return -(end - t[:, None]) / NORM + FAIL * (1 - success)


def load(run: Path):
    eps = pd.read_csv(run / "episodes.csv")
    rows = []
    for r in eps.itertuples():
        z = np.load(run / "branch" / r.file)
        t, s, e = z["t"], z["success"].astype(float), z["end_step"].astype(float)
        g = g_target(s, e, t)
        h = s.shape[1] // 2
        for i, tt in enumerate(t):
            rows.append(dict(ep=int(r.episode_index), t=int(tt), v=s[i].mean(), v_a=s[i, :h].mean(), v_b=s[i, h:].mean(),
                             g=g[i].mean(), g_a=g[i, :h].mean(), g_b=g[i, h:].mean(), k=s.shape[1],
                             state_diff=float(z["state_maxdiff"][i]),
                             px_diff=int(z["pixel_maxdiff"][i]) if "pixel_maxdiff" in z.files else -1))
    return eps, pd.DataFrame(rows)


def add_advantage(d: pd.DataFrame, stride: int):
    step = N // stride if N % stride == 0 else None
    assert step, f"stride {stride} must divide n = {N}"
    for c in ("g", "g_a", "g_b"):
        d["A_" + c] = d.groupby("ep")[c].shift(-step) - d[c] - N / NORM
    return d


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    eps, d = load(run)
    stride = int(np.diff(sorted(d[d.ep == d.ep.iloc[0]].t))[0]) if len(d) > 1 else N
    d = add_advantage(d, stride)
    tab = pd.read_csv(TABLE / "episodes.csv")
    rate = tab[(tab.split == "train") & (tab.init_state >= 0)].groupby("init_state").success.mean()
    oof = pd.read_parquet(TABLE / "labels" / "oof_resnet_n20_p30" / "labels.parquet")
    d = d.merge(oof.rename(columns={"episode_index": "ep", "frame_index": "t"})[["ep", "t", "V", "A"]],
                on=["ep", "t"], how="left")
    for name in LABELS:
        p = TABLE / "labels" / name / "labels.parquet"
        if p.exists():
            lab = pd.read_parquet(p).rename(columns={"episode_index": "ep", "frame_index": "t", "positive": "pos_" + name})
            d = d.merge(lab[["ep", "t", "pos_" + name]], on=["ep", "t"], how="left")
    res = {"run": str(run), "points": int(len(d)), "episodes": int(d.ep.nunique()), "k": int(d.k.iloc[0]),
           "stride": stride}
    res["fidelity"] = {"state_maxdiff": float(d.state_diff.max()), "pixel_maxdiff": int(d.px_diff.max()),
                       "replay_outcome_matches": int((eps.replay_success == eps.success).sum()),
                       "of": int(len(eps))}
    t0 = d[d.t == 0].merge(eps.rename(columns={"episode_index": "ep"})[["ep", "init_state"]], on="ep")
    res["calibration_t0"] = [{"ep": int(r.ep), "init_state": int(r.init_state), "V*": float(r.v),
                              "state_rate_15": float(rate.get(int(r.init_state), np.nan))} for r in t0.itertuples()]
    dec = []
    for e, g in d.groupby("ep"):
        g = g.sort_values("t")
        dv = g.v.diff().to_numpy()
        i = int(np.nanargmax(np.abs(dv))) if np.isfinite(dv).any() else 0
        dec.append({"ep": int(e), "success": int(eps.set_index("episode_index").success[e]),
                    "V*_start": float(g.v.iloc[0]), "V*_last": float(g.v.iloc[-1]),
                    "largest_jump": float(dv[i]) if i else 0.0, "at_t": int(g.t.iloc[i]) if i else 0,
                    "frac_points_with_|dV|>=0.5": float(np.nanmean(np.abs(dv) >= 0.5))})
    res["decided"] = dec
    res["split_half_reliability"] = {"V*": corr(d.v_a.to_numpy(), d.v_b.to_numpy()),
                                     "A*": corr(d.A_g_a.to_numpy(), d.A_g_b.to_numpy())}
    m = d.V.notna()
    res["value"] = {"corr_oofV_G*": corr(d.V[m].to_numpy(), d.g[m].to_numpy()),
                    "mae_oofV_G*": float((d.V[m] - d.g[m]).abs().mean()),
                    "corr_oofV_V*": corr(d.V[m].to_numpy(), d.v[m].to_numpy())}
    lab = {"corr_Aoof_A*": corr(d.A.to_numpy(), d.A_g.to_numpy())}
    for name, desc in LABELS.items():
        c = "pos_" + name
        if c in d:
            ok = d[c].notna() & d.A_g.notna()
            x = d[ok]
            lab[name] = {"desc": desc, "pos_share_when_A*>0": float(x[x.A_g > 0][c].mean()),
                         "pos_share_when_A*<0": float(x[x.A_g < 0][c].mean()),
                         "corr_label_A*": corr(x[c].astype(float).to_numpy(), x.A_g.to_numpy()), "n": int(ok.sum())}
    res["labels"] = lab
    (run / "analysis.json").write_text(json.dumps(res, indent=2))
    d.to_csv(run / "points.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        E = sorted(d.ep.unique())
        fig, axs = plt.subplots(len(E), 1, figsize=(9, 2.2 * len(E)), squeeze=False)
        for ax, e in zip(axs[:, 0], E):
            g = d[d.ep == e].sort_values("t")
            se = np.sqrt(g.v * (1 - g.v) / g.k)
            ax.fill_between(g.t, g.v - se, g.v + se, alpha=0.25, color="C0")
            ax.plot(g.t, g.v, "o-", color="C0", ms=3, label="V* = P(success | s_t), sim branching")
            ax.plot(g.t, 1 + g.g / 0.5, "--", color="C0", lw=1, label="G* (same rescale)")
            ax.plot(g.t, 1 + g.V / 0.5, color="C1", lw=1, label="OOF ResNet V (rescaled: 1 + V/0.5)")
            c = "pos_resnet_valuebinned_p40"
            if c in g:
                ax.scatter(g.t, np.where(g[c] == 1, 1.08, -0.08), s=6, color="C2", label="faithful label (+ top, − bottom)")
            ax.set_ylim(-0.15, 1.15)
            ax.set_ylabel(f"ep {e} ({'S' if eps.set_index('episode_index').success[e] else 'F'})")
        axs[0, 0].legend(fontsize=7, loc="lower left")
        axs[-1, 0].set_xlabel("policy step")
        fig.tight_layout()
        fig.savefig(run / "curves.png", dpi=110)
    except Exception as ex:  # plotting is optional
        res["plot_error"] = repr(ex)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
