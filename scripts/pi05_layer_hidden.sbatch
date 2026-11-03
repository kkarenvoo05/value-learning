#!/bin/bash
# P1c per-layer hidden states of frozen pi05 (scripts/pi05_layer_hidden.py) on one GPU, openpi env.
# Usage: sbatch scripts/pi05_layer_hidden.sbatch                 # step 2, normal attention
#        ABLATE=1 sbatch scripts/pi05_layer_hidden.sbatch        # step 3, image tokens cannot see text/state
#        [OUT=<dir>] [EXTRA="--max-frames 256"]
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
BASE=/move/u/karenvo/libero_features   # /iris/u/ankile is 99% full; ~14 GB per run
if [ "${ABLATE:-0}" = 1 ]; then FLAG=--image-only-attn; TAG=p1c_imgonly; else FLAG=; TAG=p1c_hidden; fi
OUT=${OUT:-$BASE/${TAG}_${SLURM_JOB_ID}}
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT flag $FLAG"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python $VL/scripts/pi05_layer_hidden.py --out $OUT $FLAG ${EXTRA:-}
echo "end $(date -Is)"
