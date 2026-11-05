#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi JAX_PLATFORMS=cpu
echo "host $(hostname) CPU start $(date -Is)"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/pi05_layer_hidden.py --out /move/u/karenvo/libero_features/smoke_selfcheck_fix_cpu --max-frames 64 --batch 8
echo "end $(date -Is)"
