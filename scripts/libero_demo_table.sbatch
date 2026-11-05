#!/bin/bash
# Stage-1 demo table (docs/libero_demo_pretrain_draft.md), CPU only: convert physical-intelligence/libero to raw
# files (openpi env), write targets (project venv), build the ResNet frame cache (every 4th frame).
# Usage: sbatch [--exclude=...] scripts/libero_demo_table.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
SRC=${SRC:-/iris/u/ankile/karenvo/libero/lerobot/physical-intelligence_libero}
OUT=${OUT:-/iris/u/ankile/karenvo/libero/tables/libero_demos}
echo "host $(hostname) start $(date -Is)"
cd /iris/u/ankile/karenvo/pi05/openpi
LIBERO_CONFIG_PATH=/iris/u/ankile/karenvo/libero/libero_cfg .venv/bin/python $VL/scripts/libero_demo_table.py \
    --src $SRC --out $OUT ${EXTRA:-}
cd $VL
.venv/bin/python -m vl.libero_demos --dataset $OUT
.venv/bin/python -m vl.libero_resnet --every 4 --dataset $OUT --pack-only --out $OUT/pack_check
touch $OUT/DONE
du -sh $OUT
echo "end $(date -Is)"
