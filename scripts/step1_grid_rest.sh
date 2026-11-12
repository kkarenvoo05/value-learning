#!/bin/bash
# Remainder of scripts/step1_grid.sh that hit the iris QOS submit limit (25 jobs) on 2026-11-08 21:40:
# lane 5 (2% demos = each demo once, low lr) + the two reference evals of the original-recipe run.
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
LAB=/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/labels/resnet_valuebinned_p40/labels.parquet
RUNS=/iris/u/ankile/karenvo/libero/runs
STATES="--args.init-states 9 12 13 17 25 32 40 42 44 48"
evalj() { sbatch --parsable -d "$1" --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_RUN=$2,FT_TAG=none,FT_STEP=$3,NAME=eval_$4_at$3_noise1,CLIENT_ARGS="$STATES" scripts/libero_eval.sbatch; }
DEP=$(squeue -h -j 18394908 -o %i 2>/dev/null | head -1); D=${DEP:+-d afterany:$DEP}
N=grid_d02_low_s0
T=$(sbatch --parsable -A iris -p iris --exclude='iris[1-4,8-9],iris-hgx-[1-2],iris-hp-z8' $D \
    --export=ALL,NAME=$N,ARGS="--train --arm advantage --labels $LAB --tag-dropout 1.0 --demos --freeze-siglip --save-every 1000 --seed 0 --lora-lr 1e-5 --lr 2.5e-6" \
    scripts/pi05_policy_finetune.sbatch)
R=$RUNS/policy_${N}_$T
E2=$(evalj afterok:$T $R 2000 $N); E3=$(evalj afterany:$E2 $R final $N)  # @1000 dropped (Karen 2026-11-08)
echo "$N: train $T -> evals @2000 $E2 @final $E3 (run $R)"
U=$RUNS/policy_armU_alluntagged_s0_18376076
R2=$(evalj afterany:$E3 $U 2000 armU_alluntagged)
echo "reference armU @2000 $R2"
