#!/bin/bash
#   sbatch scripts/shift_check.sbatch
# Stage 5: test-time A2 crop shift on the baseline, full-aug and crop runs (no training; vl.shift_check).
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
echo "git $(git rev-parse HEAD)  dirty: $(git status --porcelain -- src configs | wc -l) files"
df -h ~ | tail -1
.venv/bin/python -u -m vl.shift_check --runs lowlr_8k lowlr_8k_s1 lowlr_8k_s2 \
  aug_full_s0 aug_full_s1 aug_full_s2 aug_crop_s0 aug_crop_s1 aug_crop_s2
.venv/bin/python -u -m vl.shift_check --report
echo "end $(date -Is)"
