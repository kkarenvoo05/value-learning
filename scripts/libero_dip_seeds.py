"""Seeded milestone / dip checks: scripts/libero_dip_compare.py's statistics, per seed, for the 3-seed probe and
ResNet arms of the grid / history tests (plan log 2026-10-29, "seeds"). Needs
reports/figures/libero_milestone_probe/events.csv from vl.libero_milestone_probe.
Usage: .venv/bin/python scripts/libero_dip_seeds.py   (prints mean ± sd over seeds, then every seed)

Per seed and output (V = E[V], P = P(success) of the outcome head; the ResNet has no P):
  excess    time-matched excess jump at the milestone (vl.libero_milestone)
  S-F       mean raw jump on successes minus failures; AUC = jump -> success ROC AUC
  dip       mean over [release + 1 s, reach pot 2) minus [m - 5 s, m) (the empty-arm trip between pots)
"""
import numpy as np, pandas as pd
from vl.libero_milestone import jump, episodes, W, HOLD
from vl.metrics import auc

F = "/iris/u/ankile/karenvo/libero/features/"; R = "/iris/u/ankile/karenvo/libero/runs/"
S = F + "seeds/"
def probe(dirs, key): return [(d, f"ev_{key}", f"ps_{key}") for d in dirs]
def resnet(dirs): return [(R + d, "ev_resnet", None) for d in dirs]
ARMS = {  # label: per seed (dir, V col, P col or None); seed 0 first
 "probe mean-pool": probe([F + "feats_every4_18163142_oc", S + "single_s1", S + "single_s2"], "all_mlp"),
 "probe hist 2 s": probe([F + "feats_every4_18163142_hist20", S + "hist20_s1", S + "hist20_s2"], "all_mlp"),
 "probe hist 2/4.8/10 s": probe([F + "feats_every4_18163142_hist20_48_100", S + "hist20_48_100_s1",
                                 S + "hist20_48_100_s2"], "all_mlp"),
 "probe grid1 attn": probe([S + f"grid_s{s}" for s in range(3)], "grid1_attn"),
 "probe grid2 attn": probe([S + f"grid_s{s}" for s in range(3)], "grid2_attn"),
 "probe grid4 attn": probe([S + f"grid_s{s}" for s in range(3)], "grid4_attn"),
 "grid4 + 1 past 2 s": probe([S + f"grid4_hist20_s{s}" for s in range(3)], "grid4_attn"),
 "grid4 + past 2/4.8/10 s": probe([S + f"grid4_hist3_s{s}" for s in range(3)], "grid4_attn"),
 "ResNet": resnet(["resnet18_18166232", "resnet18_s1_18174529", "resnet18_s2_18174726"]),
 "ResNet 1 past 1.2 s": resnet([f"resnet18_hist12_s{s}_{j}" for s, j in enumerate((18174804, 18175000, 18175080))]),
 "ResNet 1 past 2 s": resnet(["resnet18_c2a20_18167078", "resnet18_hist20_s1_18174530", "resnet18_hist20_s2_18174732"]),
 "ResNet 1 past 4.8 s": resnet([f"resnet18_hist48_s{s}_{j}" for s, j in enumerate((18174861, 18175079, 18175082))]),
 "ResNet past 2/4.8/10 s": resnet(["resnet18_hist20_48_100_18167188", "resnet18_hist20_48_100_s1_18174531",
                                   "resnet18_hist20_48_100_s2_18174759"])}

ev = pd.read_csv("reports/figures/libero_milestone_probe/events.csv").set_index("episode_index")
val = episodes("val").set_index("episode_index"); T = val.policy_steps; ms = val.milestone


def stats(path, col):
    d = pd.read_parquet(f"{path}/val_predictions.parquet")
    by = {e: g.sort_values("frame_index") for e, g in d.groupby("episode_index")}
    treated = [e for e in by if np.isfinite(ms[e]) and ms[e] >= W and ms[e] + W <= T[e]]
    J, dd, y, dip = [], [], [], []
    for e in treated:
        m, g = ms[e], by[e]; f, v = g.frame_index.values, g[col].values
        ctrl = [c for c in by if c != e and T[c] >= m + W and not (ms[c] < m + W + HOLD)]
        j = jump(f, v, m)
        J.append(j); y.append(int(g.success.iloc[0]))
        dd.append(j - np.nanmean([jump(by[c].frame_index.values, by[c][col].values, m) for c in ctrl]))
        if np.isfinite(ev.loc[e, "near_q"]):
            tr = (f >= ev.loc[e, "release"] + 10) & (f < ev.loc[e, "near_q"]); pre = (f >= m - 50) & (f < m)
            if tr.any(): dip.append(v[tr].mean() - v[pre].mean())
    J, y = np.array(J), np.array(y)
    return {"excess": np.mean(dd), "S-F": J[y == 1].mean() - J[y == 0].mean(), "AUC": auc(y, J), "dip": np.mean(dip)}


rows = []
for arm, seeds in ARMS.items():
    for s, (p, v, ps) in enumerate(seeds):
        for out, col in (("V", v), ("P", ps)):
            if col: rows.append({"arm": arm, "out": out, "seed": s, **stats(p, col)})
per = pd.DataFrame(rows)
keys = ["excess", "S-F", "AUC", "dip"]
agg = per.groupby(["arm", "out"], sort=False)[keys].agg(["mean", "std"])
tab = pd.DataFrame({k: [f"{m:+.3f} ± {s:.3f}" for m, s in zip(agg[(k, "mean")], agg[(k, "std")])] for k in keys},
                   index=agg.index)
tab["excess > 0 (seeds)"] = per.groupby(["arm", "out"], sort=False).excess.apply(lambda x: f"{(x > 0).sum()}/{len(x)}")
print(tab.to_markdown())
print()
print(per.round(3).to_string(index=False))
