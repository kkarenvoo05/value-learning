#!/bin/bash
# One shell command line in this repo's env on one GPU (default TITAN RTX: torch scripts) [2026-11-07].
# Usage: CMD=".venv/bin/python scripts/libero_td_labels.py labels && .venv/bin/python scripts/..." sbatch scripts/vl_python.sbatch
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
: "${CMD:?set CMD}"
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
echo "cmd: $CMD"
bash -euo pipefail -c "$CMD"
echo "end $(date -Is)"
