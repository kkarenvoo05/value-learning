#!/bin/bash
# Stage 5 ablation 3 (encoder): the settled recipe (baseline_lowlr, wrist + side) with the ImageNet
# backbone frozen, seeds 0-2 in sequence on one GPU. The fine-tuned arm is the existing
# lowlr_8k / _s1 / _s2 runs.
#   sbatch scripts/frozen.sbatch   -> data/runs/frozen_s{0,1,2}/
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=humanoid
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
echo "git $(git rev-parse HEAD)  dirty: $(git status --porcelain -- src configs | wc -l) files"
df -h ~ | tail -1
for s in ${SEEDS:-0 1 2}; do
  .venv/bin/python -u -m vl.train_baseline --run "frozen_s${s}" --section baseline_lowlr --seed "$s" --freeze-backbone
done
echo "end $(date -Is)"
