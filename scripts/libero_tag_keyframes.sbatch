#!/bin/bash
# Localized tag-usage test at gripper-switch / stall frames (scripts/libero_tag_keyframes.py), openpi env, one GPU.
# Usage: [ARGS="--val-eps 60 --k 4"] sbatch scripts/libero_tag_keyframes.sbatch
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris[1-4,8-9],iris-hgx-[1-2],iris-hp-z8   # as the tag checks 18352945/47: no 24 GB / Turing cards, no RTX PRO 6000 (Karen 2026-11-04)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
OUT=${OUT:-/iris/u/ankile/karenvo/libero/runs/tag_keyframes_${SLURM_JOB_ID}}
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python $VL/scripts/libero_tag_keyframes.py --out $OUT ${ARGS:-}
echo "end $(date -Is)"
