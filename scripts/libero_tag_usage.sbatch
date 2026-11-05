#!/bin/bash
# Does the advantage-tagged policy use its tag? (scripts/libero_tag_usage.py), openpi env, one GPU.
# Usage: [ARGS="--val-eps 40 --k 4"] sbatch scripts/libero_tag_usage.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:1
#SBATCH --exclude=move1,move2,move3   # policy models OOM at 24 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
OUT=${OUT:-/iris/u/ankile/karenvo/libero/runs/tag_usage_${SLURM_JOB_ID}}
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python $VL/scripts/libero_tag_usage.py --out $OUT ${ARGS:-}
echo "end $(date -Is)"
