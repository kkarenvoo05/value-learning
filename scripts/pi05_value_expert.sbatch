#!/bin/bash
# Stage 3b value expert (scripts/pi05_value_expert.py) on one L40S, openpi env.
# Usage: ARGS="--overfit" sbatch scripts/pi05_value_expert.sbatch          (Check C)
#        [WAIT_FOR=<file>] start only once that file exists (up to 40 min)
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
OUT=${OUT:-/iris/u/ankile/karenvo/libero/runs/value_expert_${NAME:-checkc}_${SLURM_JOB_ID}}
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT"
if [ -n "${WAIT_FOR:-}" ]; then   # wait (up to 40 min) for another job to write this file, e.g. a table's DONE marker
    for i in $(seq 1 240); do [ -e "$WAIT_FOR" ] && break; sleep 10; done
    [ -e "$WAIT_FOR" ] || { echo "timed out waiting for $WAIT_FOR"; exit 1; }
fi
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python $VL/scripts/pi05_value_expert.py $ARGS --out $OUT
echo "end $(date -Is)"
