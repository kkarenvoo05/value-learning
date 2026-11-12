#!/bin/bash
# Stage 6 policy fine-tuning (scripts/pi05_policy_finetune.py), openpi env, one arm per job.
# Usage: NAME=selftest ARGS="--selftest" sbatch scripts/pi05_policy_finetune.sbatch
#        NAME=checkc   ARGS="--overfit" sbatch ...
#        NAME=armA_s0  ARGS="--train --arm imitation --seed 0" sbatch ...
#        NAME=armC_resnet_s0 ARGS="--train --arm advantage --labels <parquet> --seed 0" sbatch ...
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:1
#SBATCH --exclude=move1,move2,move3   # 24 GB A5000s run out of memory (Check C 18267998); L40S 48 GB or rtxpro6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
cd /iris/u/ankile/karenvo/pi05/openpi
OUT=/iris/u/ankile/karenvo/libero/runs/policy_${NAME:-run}_${SLURM_JOB_ID}
# Preemption-safe: a requeued job keeps its SLURM_JOB_ID (same OUT), so warm-restart from the newest
# --save-every checkpoint instead of retraining from step 0 over the old checkpoints (2026-11-05, RECAP 18325723
# was preempted at step 4750). Adam moments restart at 0 (see --resume-from help).
# A checkpoint in this job's own OUT wins over an explicit --resume-from (that one only seeds the first attempt).
RESUME=""
if [ -d "$OUT" ]; then
    last=$(ls "$OUT" | sed -n 's/^trainable_\([0-9]\+\)\.msgpack$/\1/p' | sort -n | tail -1)
    if [ -n "$last" ]; then
        ARGS=$(echo " $ARGS " | sed -E 's/ --resume-(from|step) [^ ]+//g')
        RESUME="--resume-from $OUT --resume-step $last"; echo "requeued: resuming from step $last"
    fi
fi
.venv/bin/python $VL/scripts/pi05_policy_finetune.py $ARGS $RESUME --out $OUT
echo "end $(date -Is)"
