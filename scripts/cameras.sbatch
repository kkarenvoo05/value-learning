#!/bin/bash
# Stage 5 ablations: one config section (default baseline_lowlr), seeds 0-2 in sequence on one GPU.
#   NAME=side  CAMS=side_1     sbatch scripts/cameras.sbatch   -> data/runs/side_s{0,1,2}/
#   NAME=wrist CAMS=wrist_left sbatch scripts/cameras.sbatch   -> data/runs/wrist_s{0,1,2}/
#   NAME=aug_photo SECTION=baseline_lowlr_aug_photo sbatch scripts/cameras.sbatch  (both cameras)
#   NAME=proprio EXTRA=--proprio DIST=1 sbatch scripts/cameras.sbatch  (images + proprio)
#   PRE_DIST="lowlr_8k ...": first write val_probs.npz for those existing runs; DIST=1: val_probs.npz after each seed
#   ATTN=1: val_attn.parquet after each seed (C3 attention-fusion runs, vl.attention)
#   CONTRIB=1: val_cam_mask.parquet after each seed (camera masking check, vl.camera_contrib)
#   DROP_LAST=1: delete each run's last.pt after it finishes (home quota)
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
for r in ${PRE_DIST:-}; do .venv/bin/python -u -m vl.value_dist --run "$r"; done
for s in ${SEEDS:-0 1 2}; do
  .venv/bin/python -u -m vl.train_baseline --run "${NAME}_s${s}" --section "${SECTION:-baseline_lowlr}" --seed "$s" ${CAMS:+--cameras $CAMS} ${EXTRA:-}
  if [ -n "${DIST:-}" ]; then .venv/bin/python -u -m vl.value_dist --run "${NAME}_s${s}"; fi
  if [ -n "${ATTN:-}" ]; then .venv/bin/python -u -m vl.attention --run "${NAME}_s${s}"; fi
  if [ -n "${CONTRIB:-}" ]; then .venv/bin/python -u -m vl.camera_contrib --run "${NAME}_s${s}"; fi
  if [ -n "${DROP_LAST:-}" ]; then rm -f "data/runs/${NAME}_s${s}/last.pt"; fi
done
echo "end $(date -Is)"
