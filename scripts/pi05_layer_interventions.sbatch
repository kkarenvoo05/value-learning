#!/bin/bash
# Inference-only layer/camera diagnostic. Queue behind existing work to respect the GPU cap.
# REPO and OUT may be overridden. Default is the user-requested /iris project copy.
#SBATCH --job-name=value-layer-diagnostic
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/move/u/karenvo/libero_features/layer_diagnosis_20261105/slurm-%j.out
set -euo pipefail
REPO=${REPO:-/iris/u/karenvo/value-learning}
REPORT=$REPO/reports/2026-11-05-layer-interventions
CACHE=/move/u/karenvo/libero_features/layer_diagnosis_20261105
RUNS=/iris/u/ankile/karenvo/libero/runs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
export XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
cd /iris/u/ankile/karenvo/pi05/openpi
test -s "$CACHE/pairs.npz"
.venv/bin/python -u "$REPO/scripts/pi05_layer_interventions.py" \
    --pairs "$REPORT/pairs.csv" --cache "$CACHE/pairs.npz" \
    --out "${OUT:-$REPORT/gpu_${SLURM_JOB_ID}}" \
    --experts "$RUNS/value_expert_fixedfresh_w256_s0_18237898" \
              "$RUNS/value_expert_fixedfresh_w256_s1_18237899" \
              "$RUNS/value_expert_fixedfresh_w256_s2_18237900"
