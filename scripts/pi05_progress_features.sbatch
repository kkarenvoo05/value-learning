#!/bin/bash
# Progress probe features (scripts/pi05_progress_features.py) on one GPU, openpi env.
# Usage: sbatch scripts/pi05_progress_features.sbatch   [OUT=<dir>] [EXPERTS="<run dir>:<width>:<mlp> ..."]
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
R=/iris/u/ankile/karenvo/libero/runs
OUT=${OUT:-/iris/u/ankile/karenvo/libero/features/progress_${SLURM_JOB_ID}}
EXPERTS=${EXPERTS:-"$R/value_expert_single_s0_18178421:512:2048 $R/value_expert_aug_w256_s0_18186728:256:1024"}
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python $VL/scripts/pi05_progress_features.py --out $OUT --expert $EXPERTS
echo "end $(date -Is)"
