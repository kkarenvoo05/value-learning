#!/bin/bash
# Stage 5 ablation 1 (cameras): the settled recipe (baseline_lowlr) on one camera, seeds 0-2 in
# sequence on one GPU. The wrist + side arm is the existing lowlr_8k / _s1 / _s2 runs.
#   NAME=side  CAMS=side_1     sbatch scripts/cameras.sbatch   -> data/runs/side_s{0,1,2}/
#   NAME=wrist CAMS=wrist_left sbatch scripts/cameras.sbatch   -> data/runs/wrist_s{0,1,2}/
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
echo "git $(git rev-parse HEAD)  dirty: $(git status --porcelain -- src configs | wc -l) files"
df -h ~ | tail -1
for s in ${SEEDS:-0 1 2}; do
  .venv/bin/python -u -m vl.train_baseline --run "${NAME}_s${s}" --section baseline_lowlr --seed "$s" --cameras $CAMS
done
echo "end $(date -Is)"
