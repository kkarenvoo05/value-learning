"""Every learned value / label set vs the simulator ground truth, pooled over all branching runs [2026-11-10, Lars 4].

Ground truth per branch point (scripts/libero_branch_analyze.py, same definitions): V* = P(success | s_t) from K
base-policy continuations, G* = the ResNet's MC target in its units, A* = n = 20 advantage with V replaced by G*.
Branch runs: branch_value 18355241 (6 train episodes) + branch_fail a/b/c 18395412-14 (12 failed train episodes),
K = 12 continuations per point.

Per value source (labels/<name>/labels.parquet with V and A; out-of-fold where the name says so):
  value   corr(V, V*), corr(V, G*) (Pearson) and Spearman(V, V*)
  adv     corr(A, A*), sign agreement P(sign A = sign A*)
  label   corr(positive, A*), positive share at A* > 0 vs < 0, balanced accuracy of "positive" for A* > 0
Noise ceiling [ours]: split-half reliability r (continuations 1..K/2 vs K/2+1..K) -> full-K reliability by
Spearman-Brown R = 2r / (1 + r); a perfect predictor's observable corr with the estimate is at most sqrt(R).
CIs: bootstrap over episodes (2000), since points within an episode are correlated.

Usage: .venv/bin/python scripts/libero_value_vs_gt.py [--out reports/2026-11-10-value-vs-gt]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from libero_branch_analyze import TABLE, add_advantage, load

RUNS = ["branch_value_libero_10_18355241", "branch_fail_a_libero_10_18395412", "branch_fail_b_libero_10_18395413",
        "branch_fail_c_libero_10_18395414"]
ROOT = Path("/iris/u/ankile/karenvo/libero/runs")
SOURCES = {  # name -> description (V and A columns)
    "oof_resnet_n20_p30": "ResNet-18 V, 5-fold OOF (current labels' V)",
    "ens_vb40": "ResNet V ensemble",
    "oof_expert_n20_p30": "pi0.5 value expert, OOF",
    "td_vb40": "fitted n-step TD critic",
    "td_ens_vb40": "TD critic ensemble",
    "look_n80_vb40": "ResNet V, lookahead n = 80",
    "look_n160_vb40": "ResNet V, lookahead n = 160",
    "hind_v0_vb40": "realized return - V (hindsight)",
    "iql09_vb40": "IQL tau 0.9 V (frozen features, OOF), vb 40%",  # [2026-11-12] scripts/libero_iql_labels.py
    "iql07_vb40": "IQL tau 0.7 V (frozen features, OOF), vb 40%",
    "iqlmc_vb40": "MC-fit V (same frozen features, OOF), vb 40%",
}
LABEL_ONLY = {"resnet_valuebinned_p40": "faithful RECAP labels (vb 40%)", "scoped_valuebinned_p40": "subtask-scoped vb 40%",
              "control_random_p50": "random labels (control)"}


def corr(a, b, method="pearson"):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 3 or s.a.std() == 0 or s.b.std() == 0:
        return float("nan")
    if method == "spearman":  # rank then Pearson (no scipy in this env)
        s = s.rank()
    return float(s.a.corr(s.b))


def boot(d, fn, n=2000, seed=0):
    eps = d.ep.unique()
    rng = np.random.default_rng(seed)
    g = {e: x for e, x in d.groupby("ep")}
    vals = [fn(pd.concat([g[e] for e in rng.choice(eps, len(eps))])) for _ in range(n)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [np.nan, np.nan]


def stat(d, fn):
    return {"est": fn(d), "ci95": boot(d, fn)}


def reliability(d, a, b):
    r = corr(d[a], d[b])
    R = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")
    return {"split_half_r": r, "full_K_R": R, "ceiling_sqrtR": float(np.sqrt(R)) if R > 0 else 0.0,
            "ceiling_sqrt_r_(plan)": float(np.sqrt(r)) if r > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/2026-11-10-value-vs-gt")
    a = ap.parse_args()
    parts = []
    for r in RUNS:
        _, d = load(ROOT / r)
        stride = int(np.diff(sorted(d[d.ep == d.ep.iloc[0]].t))[0])
        parts.append(add_advantage(d, stride).assign(run=r, stride=stride))
    d = pd.concat(parts, ignore_index=True)
    assert not d.duplicated(["ep", "t"]).any(), "an episode appears in two branch runs"
    for name in list(SOURCES) + list(LABEL_ONLY):
        lab = pd.read_parquet(TABLE / "labels" / name / "labels.parquet")
        cols = [c for c in ("V", "A", "positive") if c in lab]
        lab = lab.rename(columns={"episode_index": "ep", "frame_index": "t", **{c: f"{c}:{name}" for c in cols}})
        d = d.merge(lab[["ep", "t"] + [f"{c}:{name}" for c in cols]], on=["ep", "t"], how="left")
    res = {"runs": RUNS, "points": int(len(d)), "points_with_A*": int(d.A_g.notna().sum()),
           "episodes": int(d.ep.nunique()), "k": sorted(d.k.unique().tolist()), "strides": sorted(d.stride.unique().tolist()),
           "ceiling": {"V*": reliability(d, "v_a", "v_b"), "A*": reliability(d, "A_g_a", "A_g_b")},
           "ceiling_by_run": {r: reliability(x, "A_g_a", "A_g_b") for r, x in d.groupby("run")}}
    rows = []
    for name, desc in {**SOURCES, **LABEL_ONLY}.items():
        row = {"source": name, "desc": desc}
        V, A, P = f"V:{name}", f"A:{name}", f"positive:{name}"
        if V in d:
            row["coverage"] = float(d[V].notna().mean())
            row["corr_V_V*"] = stat(d, lambda x: corr(x[V], x.v))
            row["spearman_V_V*"] = corr(d[V], d.v, "spearman")
            row["corr_V_G*"] = stat(d, lambda x: corr(x[V], x.g))
        if A in d:
            row["corr_A_A*"] = stat(d, lambda x: corr(x[A], x.A_g))
            row["sign_agree_A_A*"] = stat(d, lambda x: float((np.sign(x[A]) == np.sign(x.A_g))[x[A].notna() & x.A_g.notna()].mean()))
        if P in d:
            ok = d[P].notna() & d.A_g.notna()
            x = d[ok]
            row["label_corr_A*"] = stat(x, lambda y: corr(y[P].astype(float), y.A_g))
            row["pos_share_A*>0"] = float(x[x.A_g > 0][P].mean())
            row["pos_share_A*<0"] = float(x[x.A_g < 0][P].mean())
            row["balanced_acc"] = float(0.5 * (x[x.A_g > 0][P].mean() + 1 - x[x.A_g < 0][P].mean()))
        rows.append(row)
    res["sources"] = rows
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    d.to_csv(out / "points.csv", index=False)
    f = lambda s: "" if s is None else (f"{s['est']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]" if isinstance(s, dict) else f"{s:.2f}")
    c = res["ceiling"]
    lines = [f"points {res['points']} ({res['points_with_A*']} with A*), episodes {res['episodes']}, K {res['k']}",
             f"V* split-half r {c['V*']['split_half_r']:.2f} -> R {c['V*']['full_K_R']:.2f}, ceiling {c['V*']['ceiling_sqrtR']:.2f}",
             f"A* split-half r {c['A*']['split_half_r']:.2f} -> R {c['A*']['full_K_R']:.2f}, ceiling {c['A*']['ceiling_sqrtR']:.2f} "
             f"(sqrt r = {c['A*']['ceiling_sqrt_r_(plan)']:.2f})",
             "by run (A*): " + ", ".join(f"{k.split('_libero')[0]} r {v['split_half_r']:.2f}" for k, v in res["ceiling_by_run"].items()),
             "", "| source | corr(V, V*) | corr(A, A*) | sign agree | label corr(A*) | + at A*>0 / <0 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['desc']} | {f(r.get('corr_V_V*'))} | {f(r.get('corr_A_A*'))} | {f(r.get('sign_agree_A_A*'))} | "
                     f"{f(r.get('label_corr_A*'))} | {r.get('pos_share_A*>0', float('nan')):.2f} / {r.get('pos_share_A*<0', float('nan')):.2f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
