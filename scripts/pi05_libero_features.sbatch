#!/bin/bash
# Stage 3a feature extraction (scripts/pi05_libero_features.py) on one L40S.
# Usage: sbatch scripts/pi05_libero_features.sbatch                       (full: every 4th frame)
#        MAXEP=2 NAME=smoke sbatch scripts/pi05_libero_features.sbatch     (smoke)
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
RUN=${RUN:-/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849}
NAME=${NAME:-feats_every${EVERY:-4}}
OUT=/iris/u/ankile/karenvo/libero/features/${NAME}_${SLURM_JOB_ID}.npz
mkdir -p "$(dirname $OUT)"
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python $VL/scripts/pi05_libero_features.py --run-dir $RUN --out $OUT --every ${EVERY:-4} \
    ${MAXEP:+--max-episodes $MAXEP} ${GRID:+--grid $GRID}
echo "end $(date -Is)"
