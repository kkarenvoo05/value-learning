import json, sys
import numpy as np
from pathlib import Path
R = Path("/move/u/karenvo/Projects/value-learning/data/runs")
K = ["val_ce","val_mae","offset_free_mae","eval_repos_mae","eval_repos_auc_1s","eval_repos_auc_5s","eval_repos_auc_10s"]
arms = ["aug_full","g0_r0","g0_r1","g0_r1b","g0_r2","g0_r3","g0_r4","g0_r5","g0_r6","g0_r6_16k"]
stats = {}
for a in arms:
    rs = [json.loads((R/f"{a}_s{s}"/"result.json").read_text())["best_eval"] for s in range(3) if (R/f"{a}_s{s}"/"result.json").exists()]
    if not rs: continue
    stats[a] = {k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs], ddof=1) if len(rs)>1 else np.nan, [r[k] for r in rs]) for k in K}
    stats[a]["n"] = len(rs)
print("| arm | n | " + " | ".join(K) + " |"); print("|"+"---|"*(len(K)+2))
for a,s in stats.items():
    print(f"| {a} | {s['n']} | " + " | ".join(f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in K) + " |")
r0 = stats["g0_r0"]
print("\nDecision rule vs g0_r0 (CE drop > 2 sd of R0; shape MAE and AUC1s no worse):")
for a,s in stats.items():
    if not a.startswith("g0_") or a=="g0_r0": continue
    dce = s["val_ce"][0]-r0["val_ce"][0]
    thr = 2*r0["val_ce"][1]
    print(f"{a}: dCE {dce:+.4f} (need < -{thr:.4f}), dShape {s['offset_free_mae'][0]-r0['offset_free_mae'][0]:+.4f}, dAUC1 {s['eval_repos_auc_1s'][0]-r0['eval_repos_auc_1s'][0]:+.4f}, dMAE {s['val_mae'][0]-r0['val_mae'][0]:+.4f}  -> {'PASS' if dce < -thr and s['offset_free_mae'][0]<=r0['offset_free_mae'][0] and s['eval_repos_auc_1s'][0]>=r0['eval_repos_auc_1s'][0] else 'no'}")
print("\nper-seed val_ce:", {a: [round(x,4) for x in s['val_ce'][2]] for a,s in stats.items()})
