import sys, json
import pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from libero_head_swap import score
R = "/iris/u/ankile/karenvo/libero/runs/"
runs = {"full18_frozen_18237898": "value_expert_fixedfresh_w256_s0_18237898",
        "trunc9_18291639": "value_expert_trunc9_fixedfresh_w256_s0_18291639",
        "trunc4_18291640": "value_expert_trunc4_fixedfresh_w256_s0_18291640", "trunc4_s1_18309645": "value_expert_trunc4_fixedfresh_w256_s1_18309645"}
out = {}
for k, r in runs.items():
    try:
        p = pd.read_parquet(R + r + "/val_predictions.parquet")
    except Exception as e:
        print(k, "missing", e); continue
    out[k] = score(p, "ev_expert")
    print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in out[k].items()})
json.dump(out, open(sys.argv[1], "w"), indent=1)
