#!/bin/bash
# CPU smoke of scripts/pi05_policy_finetune.py --train (2 steps, small batch). ARGS=<arm args>, NAME=<tag>
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
echo "host $(hostname) CPU start $(date -Is) args $ARGS"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/pi05_policy_finetune.py --train $ARGS \
    --steps 2 --batch 4 --accum 1 --group 2 --warmup 1 --log-every 1 --eval-every 1000 \
    --out /iris/u/ankile/karenvo/libero/runs/smoke_${NAME}_${SLURM_JOB_ID}
echo "end $(date -Is)"
