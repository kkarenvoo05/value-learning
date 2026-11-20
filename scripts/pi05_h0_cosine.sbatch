#!/bin/bash
# [2026-11-18] scripts/pi05_h0_cosine.py over several residual runs, one process per run (frees GPU memory between).
#   RUNS="<run dir> ..." [OUT=<dir>] sbatch scripts/pi05_h0_cosine.sbatch   -> <OUT>/<run name>/results.json
#SBATCH --job-name=h0_cosine
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -uo pipefail
: "${RUNS:?RUNS=<run dirs>}"
VL=/move/u/karenvo/Projects/value-learning
OPENPI=/iris/u/ankile/karenvo/pi05/openpi
ROOT=/iris/u/ankile/karenvo/libero
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
OUT=${OUT:-$ROOT/runs/h0_cosine_${SLURM_JOB_ID}}
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT"
cd $OPENPI
rc=0
for r in $RUNS; do
    echo "=== $r"
    .venv/bin/python $VL/scripts/pi05_h0_cosine.py --run "$r" --out "$OUT/$(basename "$r")" || rc=1
done
echo "end $(date -Is)"
exit $rc
