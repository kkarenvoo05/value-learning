"""IQL tau 0.9 V in other label roles (2026-11-12): value bins only, longer lookahead, hindsight baseline; scored
like libero_value_vs_gt (label corr with A*, 424 branch points). Run from the repo root."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from libero_value_vs_gt import corr, stat
from vl import libero_td as td
from vl.libero_advantages import advantage, V_FAIL
p = pd.read_parquet("reports/2026-11-11-iql-frozen/oof_predictions.parquet").sort_values(td.KEY).reset_index(drop=True)
d = p[td.KEY + ["T", "success"]]
Vr, Vi = p.V_resnet.to_numpy(float), p["V:iql_tau0.9"].to_numpy(float)
t, T, ep = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.episode_index.to_numpy()
vT = np.where(d.success.to_numpy() == 1, 0.0, V_FAIL)
def lab(A, Vbin, share=0.4):
    z = td.value_binned(A, Vbin); return (z > np.percentile(z, 100 * (1 - share))).astype(int)
v = {}
for n in (20, 80):
    Ar = advantage(Vr, t, T, vT, ep, n)[0]
    v[f"ResNet A n{n}, ResNet bins (current rule)"] = lab(Ar, Vr)
    v[f"ResNet A n{n}, IQL bins"] = lab(Ar, Vi)
    v[f"IQL A n{n}, IQL bins"] = lab(advantage(Vi, t, T, vT, ep, n)[0], Vi)
for nm, V in (("ResNet", Vr), ("IQL", Vi)):
    v[f"hindsight G - V, {nm} V"] = td.hindsight_labels(d, V).positive.to_numpy()
pts = pd.read_csv("reports/2026-11-12-iql-labels/points.csv")[["ep", "t", "A_g"]].dropna()
x = pd.DataFrame({"ep": d.episode_index, "t": d.frame_index, **v}).merge(pts, on=["ep", "t"])
print("points", len(x))
for k in v:
    s = stat(x, lambda y: corr(y[k].astype(float), y.A_g))
    print(f"{k:42s} label corr(A*) {s['est']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]  + share {x[k].mean():.2f}")
