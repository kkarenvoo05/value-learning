#!/bin/bash
# Stage 5 crop sweep, step 2: one seed per arm, arms in sequence on one GPU.
#   ARMS="p24 s48 s72 z75" sbatch scripts/crop_sweep.sbatch
#     -> section baseline_lowlr_crop16k_<arm>, run data/runs/crop16k_<arm>_s${SEED:-0}/
#   EXTRA="--backbone dinov2_vits14_reg" NAME=dinov2s_crop16k: same arms on another backbone
#   PREFIX=baseline_lowlr_crop8k NAME=crop8k: the 8k re-screen sections
#   DIST=1: val_probs.npz after each arm; DROP_LAST=1: delete each run's last.pt (home quota)
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
for a in ${ARMS:?set ARMS}; do
  run="${NAME:-crop16k}_${a}_s${SEED:-0}"
  .venv/bin/python -u -m vl.train_baseline --run "$run" --section "${PREFIX:-baseline_lowlr_crop16k}_${a}" --seed "${SEED:-0}" ${EXTRA:-}
  if [ -n "${DIST:-}" ]; then .venv/bin/python -u -m vl.value_dist --run "$run"; fi
  if [ -n "${DROP_LAST:-}" ]; then rm -f "data/runs/$run/last.pt"; fi
done
echo "end $(date -Is)"
