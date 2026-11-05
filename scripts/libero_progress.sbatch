#!/bin/bash
# Progress probe (python -m vl.libero_progress) on one GPU.
# Usage: PROG=<pi05_progress_features out dir> sbatch scripts/libero_progress.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
cd $VL
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) prog $PROG"
.venv/bin/python -m vl.libero_progress --prog "$PROG"
echo "end $(date -Is)"
