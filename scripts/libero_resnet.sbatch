#!/bin/bash
# Stage 3 ResNet-18 control (python -m vl.libero_resnet) on one GPU.
# Usage: [BACKBONE=siglip_pi05] [EXTRA="--oof-fold 0"] FEATS=<features.npz> [GAP="20"] [SEED=1] [FIT=<episodes.txt> TAG=half] [INIT=<stage-1 best.pt>] sbatch scripts/libero_resnet.sbatch
#        [PREP_FRESH_RUN=<runs/fresh...>] [WAIT_FOR=<file>] EVERY=4 DATASET=<table dir> FIT=<episodes.txt> TAG=fresh ...   (frames from scripts/libero_fresh_table.py's table)
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
cd $VL
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
if [ -n "${WAIT_FOR:-}" ]; then   # wait (up to 40 min) for another job to write this file, e.g. a table's DONE marker
    for i in $(seq 1 240); do [ -e "$WAIT_FOR" ] && break; sleep 10; done
    [ -e "$WAIT_FOR" ] || { echo "timed out waiting for $WAIT_FOR"; exit 1; }
fi
if [ -n "${PREP_FRESH_RUN:-}" ]; then   # build the fixed + fresh table first (scripts/libero_fresh_table.py)
    .venv/bin/python scripts/libero_fresh_table.py --fresh-run "$PREP_FRESH_RUN" --out "$DATASET"
    touch "$DATASET/DONE"
fi
.venv/bin/python -m vl.libero_resnet ${FEATS:+--features "$FEATS"} ${EVERY:+--every $EVERY} ${DATASET:+--dataset $DATASET} \
    --out /iris/u/ankile/karenvo/libero/runs/${BACKBONE:-resnet18}${TAG:+_$TAG}${GAP:+_hist${GAP// /_}}${SEED:+_s$SEED}_${SLURM_JOB_ID} ${STEPS:+--steps $STEPS} ${SEED:+--seed $SEED} ${GAP:+--history-gap-frames $GAP} ${FIT:+--fit-episodes $FIT} ${INIT:+--init-from $INIT} ${BACKBONE:+--backbone $BACKBONE} ${EXTRA:-}
echo "end $(date -Is)"
