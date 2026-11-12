#!/bin/bash
# CPU self-check of scripts/pi05_guidance.py (beta=1 == tagged infer, beta=0 == untagged infer). RUN=<arm C run>
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi JAX_PLATFORMS=cpu
echo "host $(hostname) CPU start $(date -Is) run ${RUN:-base}"
cd /iris/u/ankile/karenvo/pi05/openpi
.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/pi05_guidance.py --selftest ${RUN:+--run $RUN}
echo "end $(date -Is)"
