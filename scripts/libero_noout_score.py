import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/src"); sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/scripts")
from libero_head_swap import score
R = Path("/iris/u/ankile/karenvo/libero/runs")
ARMS = {
 "ResNet full": ("ev_resnet", ["resnet18_18166232", "resnet18_s1_18174529", "resnet18_s2_18174726"]),
 "ResNet half": ("ev_resnet", ["resnet18_half_s0_18196780", "resnet18_half_s1_18196781", "resnet18_half_s2_18196782"]),
 "B full": ("ev_expert", ["value_expert_aug_w256_s0_18186728", "value_expert_aug_w256_s1_18189370", "value_expert_aug_w256_s2_18189371"]),
 "B half": ("ev_expert", ["value_expert_half_w256_s0_18196783", "value_expert_half_w256_s1_18199177", "value_expert_half_w256_s2_18196785"]),
 "B no outcome loss": ("ev_expert", ["value_expert_noout_w256_s0_18200394", "value_expert_noout_w256_s1_18200395", "value_expert_noout_w256_s2_18200396"]),
}
rows = []
for arm, (col, runs) in ARMS.items():
    for s, r in enumerate(runs):
        p = pd.read_parquet(R / r / "val_predictions.parquet")
        res = json.loads((R / r / "results.json").read_text())
        x = score(p, col)
        x.update(arm=arm, seed=s, best_step=res.get("best_step", res.get("val", {}).get("best_step")),
                 ow=res.get("setup", {}).get("outcome_weight"))
        rows.append(x)
t = pd.DataFrame(rows)
cols = ["change_snr_2s", "change_signal_2s", "change_noise_2s", "milestone", "mae_succ>20s", "mae", "auc5", "auc10", "within_ep_sd"]
print(t.groupby("arm", sort=False)[cols].agg(lambda v: f"{v.mean():.3f} ± {v.std(ddof=0):.3f}").to_markdown())
print(t[["arm", "seed", "best_step", "ow"] + cols].round(3).to_string(index=False))
t.to_json(f"{sys.argv[1]}/noout_scores.json", orient="records", indent=1)
