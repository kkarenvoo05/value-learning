#!/bin/bash
# Per-layer value heads on frozen pi05 (scripts/libero_layer_value.py) on one GPU.
# Usage: sbatch scripts/libero_layer_value.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
cd $VL
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
.venv/bin/python scripts/libero_layer_value.py --out ${OUT:-reports/libero_layer_value_gpu.json}
echo "end $(date -Is)"
