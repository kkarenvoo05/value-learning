#!/bin/bash
# Stage 3.8 Check C (tiny-subset overfit) on one GPU.
#   sbatch scripts/check_c.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
df -h ~ | tail -1
.venv/bin/python -u -m vl.train --check-c --run "${RUN:-check_c}"
echo "end $(date -Is)"
