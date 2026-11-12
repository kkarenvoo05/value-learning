#!/bin/bash
# Step-1 recipe grid [2026-11-08, Karen: set it up]: can an UNTAGGED fine-tune on all rollouts + demos match base?
# Reference: the same data with the original recipe and demos at ~2% of frames (18376076) scored 0.391 vs base 0.524.
# Everything as 18376076 (all 900 train rollouts + 29 moka demos, frozen SigLIP, 5000 steps, seed 0, tag-dropout 1.0
# = no tag ever) except the demo frame share (--demo-share) and the learning rate (R2 = 10x lower, as 18315373).
# Each lane: train -> eval @1000 -> @2000 -> @final (matched noise seed 1, val states x 33, tag none).
# Lanes start after the overnight jobs so we stay at <= 6 GPUs.
# Usage: bash scripts/step1_grid.sh <lane-dep-1> ... <lane-dep-5>   (job ids each lane waits for, afterany)
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
[ $# -eq 5 ] || { echo "need 5 lane dependencies"; exit 2; }
DEPS=("$@")
LAB=/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/labels/resnet_valuebinned_p40/labels.parquet
RUNS=/iris/u/ankile/karenvo/libero/runs
BASE="--train --arm advantage --labels $LAB --tag-dropout 1.0 --demos --freeze-siglip --save-every 1000 --seed 0"
LOW="--lora-lr 1e-5 --lr 2.5e-6"
NAMES=(grid_d50_orig grid_d75_orig grid_d50_low grid_d75_low grid_d02_low)
EXTRA=("--demo-share 0.5" "--demo-share 0.75" "--demo-share 0.5 $LOW" "--demo-share 0.75 $LOW" "$LOW")
STATES="--args.init-states 9 12 13 17 25 32 40 42 44 48"
evalj() {  # dep run step name
  sbatch --parsable -d "$1" --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_RUN=$2,FT_TAG=none,FT_STEP=$3,NAME=eval_$4_at$3_noise1,CLIENT_ARGS="$STATES" scripts/libero_eval.sbatch
}
for i in 0 1 2 3 4; do
  N=${NAMES[$i]}_s0
  T=$(sbatch --parsable -A iris -p iris --exclude='iris[1-4,8-9],iris-hgx-[1-2],iris-hp-z8' -d afterany:${DEPS[$i]} \
      --export=ALL,NAME=$N,ARGS="$BASE ${EXTRA[$i]}" scripts/pi05_policy_finetune.sbatch)
  R=$RUNS/policy_${N}_$T
  E1=$(evalj afterok:$T $R 1000 $N); E2=$(evalj afterany:$E1 $R 2000 $N); E3=$(evalj afterany:$E2 $R final $N)
  echo "$N: train $T -> evals $E1 $E2 $E3 (run $R)"
  LAST=$E3
done
# Reference curve for free: the existing original-recipe run's saved checkpoints (its final = 0.391, 18376077).
U=$RUNS/policy_armU_alluntagged_s0_18376076
R1=$(evalj afterany:$LAST $U 1000 armU_alluntagged); R2=$(evalj afterany:$R1 $U 2000 armU_alluntagged)
echo "reference armU @1000 $R1, @2000 $R2 (after the last lane)"
