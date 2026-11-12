#!/bin/bash
# CPU check of scripts/pi05_value_expert.py --subtask-scoped (B1 on pi05): dummy-size --selftest, then a 2-step
# --train --smoke on real episodes with the whole-task pi05 run's recipe (value_expert_fixedfresh_w256).
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi JAX_PLATFORMS=cpu
VL=/move/u/karenvo/Projects/value-learning
T=/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh
echo "host $(hostname) CPU start $(date -Is)"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python $VL/scripts/pi05_value_expert.py --selftest
.venv/bin/python $VL/scripts/pi05_value_expert.py --train --subtask-scoped --smoke 2 --steps 2 --batch 4 --warmup 1 \
    --eval-every 1 --log-every 1 --workers 2 --every 4 --width 256 --mlp-dim 1024 --lr 1e-4 --weight-decay 0.01 \
    --augment --dataset $T --fit-episodes $T/subsets/fit_fixed_fresh.txt \
    --out /iris/u/ankile/karenvo/libero/runs/smoke_value_scoped_${SLURM_JOB_ID}
.venv/bin/python - <<PY
import pandas as pd
p = pd.read_parquet("/iris/u/ankile/karenvo/libero/runs/smoke_value_scoped_${SLURM_JOB_ID}/val_predictions.parquet")
assert {"scope", "value_task"} <= set(p.columns), p.columns
for e, g in p.groupby("episode_index"):
    print(e, "frames", len(g), "scope1 from", g[g.scope == 1].frame_index.min(), "retargeted", int((g.value != g.value_task).sum()))
print("scoped smoke ok")
PY
echo "end $(date -Is)"
