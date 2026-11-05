import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/src"); sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/scripts")
from libero_head_swap import score
R = Path("/iris/u/ankile/karenvo/libero/runs")
ARMS = {
 "ResNet fixed (ref)": ("ev_resnet", ["resnet18_18166232", "resnet18_s1_18174529", "resnet18_s2_18174726"], "val_predictions.parquet"),
 "B fixed (ref)": ("ev_expert", ["value_expert_aug_w256_s0_18186728", "value_expert_aug_w256_s1_18189370", "value_expert_aug_w256_s2_18189371"], "val_predictions.parquet"),
 "ResNet fresh": ("ev_resnet", ["resnet18_fresh_s0_18220907", "resnet18_fresh_s1_18219697", "resnet18_fresh_s2_18219698"], "val_predictions.parquet"),
 "ResNet fixed+fresh": ("ev_resnet", ["resnet18_fixedfresh_s0_18237895", "resnet18_fixedfresh_s1_18237896", "resnet18_fixedfresh_s2_18237897"], "val_predictions.parquet"),
 "B fresh": ("ev_expert", ["value_expert_fresh_w256_s0_18219699", "value_expert_fresh_w256_s1_18219700", "value_expert_fresh_w256_s2_18219701"], "val_predictions.parquet"),
 "B fixed+fresh": ("ev_expert", ["value_expert_fixedfresh_w256_s0_18237898", "value_expert_fixedfresh_w256_s1_18237899"], "val_predictions.parquet"),
 "LoRA CE-sel": ("ev_expert", ["value_lora_r16_s0_18221385", "value_lora_r16_s1_18221358"], "val_predictions.parquet"),
 "LoRA MAE-sel": ("ev_expert", ["value_lora_r16_s0_18221385", "value_lora_r16_s1_18221358"], "val_predictions_mae.parquet"),
}
rows = []
for arm, (col, runs, f) in ARMS.items():
    for s, r in enumerate(runs):
        p = pd.read_parquet(R / r / f)
        res = json.loads((R / r / "results.json").read_text())
        x = score(p, col)
        bs = res.get("best_step", res.get("val", {}).get("best_step"))
        if "MAE" in arm: bs = res["select_mae"]["best_step"]
        x.update(arm=arm, seed=s, best_step=bs); rows.append(x)
t = pd.DataFrame(rows)
cols = ["change_snr_2s", "milestone", "mae_succ>20s", "mae", "auc5", "auc10"]
print(t.groupby("arm", sort=False)[cols].agg(lambda v: f"{v.mean():.3f} ± {v.std(ddof=0):.3f}").to_markdown())
print(t[["arm", "seed", "best_step"] + cols].round(3).to_string(index=False))
