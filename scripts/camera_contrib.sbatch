#!/bin/bash
# Stage 5.5 C4: camera masking contributions (vl.camera_contrib) for each run in RUNS, in sequence on one GPU.
#   RUNS="c3_attn_s0 c3_attn_s1 c3_attn_s2 g0_r0_s0 ..." sbatch scripts/camera_contrib.sbatch
#SBATCH --job-name=vl-c4
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=data/runs/slurm-%j.out

set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  start $(date -Is)"
echo "git $(git rev-parse HEAD)  dirty: $(git status --porcelain -- src configs | wc -l) files"
for r in $RUNS; do .venv/bin/python -u -m vl.camera_contrib --run "$r"; done
echo "end $(date -Is)"
