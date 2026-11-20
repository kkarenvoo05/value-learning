#!/bin/bash
# Filtered BC round 2 anchored with base pi0.5 rollouts instead of demos (Karen 2026-11-13: "try anchoring with base
# pi0.5 rollouts instead of demos"). Submits train -> eval @final -> eval @2000.
# Why: the demo anchor (fbc_r2_anchor50_s0, 29 moka demos repeated 37x to 50% of frames) scored 0.439 = unanchored
# round 2 (0.442); zima anchors with the base policy's own data. Here the anchor = every round-1 train rollout of
# moka_fbc_r2 (= moka_fixed_fresh, the base pi0.5 rollouts: 900 episodes, 516 successes + 384 failures, untagged),
# ~49% of frames (pi05_policy_finetune.py --anchor-round 1 --anchor-share 0.5; k = 1). Everything else = fbc_r2_s0 /
# the demo-anchored run (imitation arm on round 1 + 2 successes, frozen SigLIP, original lr, 5000 steps, seed 0),
# evals as theirs (330 val trials, matched noise seed 1), so all three pair trial for trial.
# Prior: an untagged fine-tune on all base rollouts alone scored 0.391 (18376077).
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
TABLE=$ROOT/tables/moka_fbc_r2
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
VAL="--args.init-states 9 12 13 17 25 32 40 42 44 48"
NAME=fbc_r2_baseanchor50_s0
ARGS="--train --arm imitation --freeze-siglip --save-every 500 --seed 0 --table $TABLE --anchor-round 1 --anchor-share 0.5"
cd $VL
T=$(sbatch --parsable -A iris -p iris $EX --time=10:00:00 --job-name=fbc_banchor --export=ALL,NAME=$NAME,ARGS="$ARGS" scripts/pi05_policy_finetune.sbatch)
RUN=$ROOT/runs/policy_${NAME}_$T
EV="SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_RUN=$RUN,FT_TAG=none"
EF=$(sbatch --parsable -d afterok:$T $EX --time=3:00:00 --job-name=fbc_banchor_eval --export=ALL,$EV,FT_STEP=final,NAME=eval_${NAME}_atfinal_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch)
E2=$(sbatch --parsable -d afterany:$EF $EX --time=3:00:00 --job-name=fbc_banchor_eval --export=ALL,$EV,FT_STEP=2000,NAME=eval_${NAME}_at2000_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch)
echo "$(date -Is) train $T eval@final $EF eval@2000 $E2 run $RUN"
