#!/bin/bash
# Full 201-bin value distributions of a run's best.pt on validation.
#   RUN=lowlr_8k sbatch scripts/value_dist.sbatch   -> data/runs/<run>/val_probs.npz
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=humanoid
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
echo "git $(git rev-parse HEAD)"
.venv/bin/python -u -m vl.value_dist --run "${RUN:-lowlr_8k}"
echo "end $(date -Is)"
