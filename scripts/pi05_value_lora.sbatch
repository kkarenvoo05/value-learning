#!/bin/bash
# Stage 3b LoRA arm (scripts/pi05_value_lora.py): seeds in sequence in one job (GPU limit 2-3), openpi env.
# Usage: SEEDS="0 1 2" ARGS="--train --augment" sbatch --gres=gpu:l40s:1 scripts/pi05_value_lora.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
cd /iris/u/ankile/karenvo/pi05/openpi
# INIT_RUN=<stage-1 run dir>: init from its inner-CE-selected checkpoint (read from results.json at job start)
if [ -n "${INIT_RUN:-}" ]; then
  STEP=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['select_ce']['best_step'])" $INIT_RUN/results.json)
  ARGS="$ARGS --init-from $INIT_RUN/ckpt_${STEP}.msgpack"
  echo "init from $INIT_RUN/ckpt_${STEP}.msgpack"
fi
for S in ${SEEDS:-0}; do
  OUT=/iris/u/ankile/karenvo/libero/runs/value_lora_${NAME:-r16}_s${S}_${SLURM_JOB_ID}
  [ -f $OUT/results.json ] && continue
  echo "seed $S -> $OUT $(date -Is)"
  .venv/bin/python $VL/scripts/pi05_value_lora.py $ARGS --seed $S --out $OUT
done
echo "end $(date -Is)"
