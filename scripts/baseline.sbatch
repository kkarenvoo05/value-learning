#!/bin/bash
# Stage 4 baseline: full training set, checkpoint chosen on validation CE.
#   sbatch scripts/baseline.sbatch            (RUN=name to change the run directory)
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
echo "git $(git rev-parse HEAD)  dirty: $(git status --porcelain -- src configs | wc -l) files"
df -h ~ | tail -1
.venv/bin/python -u -m vl.train_baseline --run "${RUN:-baseline}"
echo "end $(date -Is)"
