#!/bin/bash
# Stage 2 LeRobot conversion (scripts/libero_to_lerobot.py) as its own job: the interactive session's shared
# 16 GiB allocation killed the first attempt at 416/570 episodes.
# Usage: RUN=/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849 sbatch scripts/libero_to_lerobot.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
cd /iris/u/ankile/karenvo/pi05/openpi
echo "host $(hostname) start $(date -Is) run $RUN"
.venv/bin/python $VL/scripts/libero_to_lerobot.py --run-dir "$RUN" --root /iris/u/ankile/karenvo/libero/lerobot \
    --overwrite 2>&1 | grep -v "examples/s\|ba/s"
cd $VL
.venv/bin/python -m vl.libero_targets --dataset /iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2
echo "end $(date -Is)"
