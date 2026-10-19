#!/bin/bash
#   RUN=lowlr_8k sbatch scripts/predict.sbatch
# Stage 4: score a finished run's best.pt on a sample of training episodes (train/val gap).
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
echo "git $(git rev-parse HEAD)  dirty: $(git status --porcelain -- src configs | wc -l) files"
df -h ~ | tail -1
.venv/bin/python -u -m vl.predict --run "$RUN"
echo "end $(date -Is)"
