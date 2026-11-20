#!/bin/bash
# Multi-task AG chain [2026-11-16; mentor: "the same AG rounds of improvement tested with eg 2 different but similar
# tasks in parallel, e.g., with the trajectories split across them, and seeing if we see any positive transfer or
# tradeoffs"; Karen: moka (libero_10 task 8) + libero_90 task 64 "stack the right bowl on the left bowl and place them
# in the tray", one shared chain-only V, controls t64b50 / t64b100 beside the running moka b50cv / b100cv].
#   CHAIN=mt100   50 moka + 50 task-64 fresh layouts per round (collector CF = moka, CR = task 64); round 0 = the
#                 nested moka b50 subsample (tables/moka_agr_b50_r0, val rows kept) + base task-64 run A + val run V
#   CHAIN=t64b50  50 task-64 fresh layouts per round; round 0 = base run A + V
#   CHAIN=t64b100 100 task-64 fresh layouts per round; round 0 = base runs A + B + V (A inside B's chain: nested)
# Labels: LABELER=cv (scripts/libero_chainv_round_labels.py, one ResNet V refit each round on the chain's own episodes;
# the 40% cut within each (round, task), scripts/libero_round_labels.labels_all). Training: round 0's exact residual
# args (as scripts/ag_budget_chain.sh) + --demos (moka's 29 demos only; libero_90 has none). Contrast check on the
# chain's own val frames (--table). Evals per round: mt100 = moka 330 val trials + moka cont set + task-64 cont set;
# t64 chains = task-64 cont set (eval_sets/t64_cont100). Rounds 1..ROUNDS: scripts/ag_rounds_feed.sh (JUST_IN_TIME).
# Usage: CHAIN=mt100 RUN_A=<run dir> RUN_V=<run dir> [RUN_B=<run dir>] [DEP=afterok:<ids>] [ROUNDS=10] [SUBMIT=1]
#        scripts/ag_mt_chain.sh          (RUN_* = base pi0.5 task-64 collections, libero_eval.sbatch --save-raw)
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
: "${CHAIN:?CHAIN=mt100|t64b50|t64b100}" "${RUN_A:?}" "${RUN_V:?}"
ROUNDS=${ROUNDS:-10} SHIFT=0.99 STEPS=5000 T64=libero_90:64 MOKA=libero_10:8
CE_MOKA=$ROOT/eval_sets/moka_cont100/starts.csv CE_T64=$ROOT/eval_sets/t64_cont100/starts.csv
case $CHAIN in
    mt100)   TAG=mt100cv TPFX=mt FN=100 TA=$MOKA TB=$T64 BASE=$ROOT/tables/moka_agr_b50_r0 RUNS="$RUN_A $RUN_V" ;;
    t64b50)  TAG=t64b50cv TPFX=t64 FN=50 TA=$T64 TB=$T64 BASE=none RUNS="$RUN_A $RUN_V" ;;
    t64b100) TAG=t64b100cv TPFX=t64 FN=100 TA=$T64 TB=$T64 BASE=none RUNS="$RUN_A ${RUN_B:?RUN_B for t64b100} $RUN_V" ;;
    *) echo "CHAIN must be mt100, t64b50 or t64b100"; exit 1 ;;
esac
for f in $CE_MOKA $CE_T64 $([ $BASE != none ] && echo $BASE/episodes.csv); do [ -e $f ] || { echo "missing $f"; exit 1; }; done
PFX=agr_$TAG
TABLE0=$ROOT/tables/${TPFX}_${PFX}_r0 LAB0=$ROOT/tables/${TPFX}_${PFX}_r0/labels/${PFX}_cv40_r0
EX="--exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8"
T64ENV=",AG_SUITE=libero_90,AG_TASK=64,AG_MAX_STEPS=520"
cd $VL
run() { if [ -n "${SUBMIT:-}" ]; then sbatch --parsable "$@"; else echo "J$RANDOM"; echo "  sbatch $*" >&2; fi; }

# round 0: table (needs the base runs) + chain-only V labels in one job, then train -> check -> evals
L0=$(run ${DEP:+-d $DEP} --time=8:00:00 --job-name=${PFX}_label --export=ALL,CMD=".venv/bin/python scripts/libero_round2_table.py --base $BASE --round 1 --keep-split --out $TABLE0 --runs $RUNS && .venv/bin/python scripts/libero_chainv_round_labels.py --table $TABLE0 --out $LAB0" scripts/vl_python.sbatch)
NAME=${PFX}_r0_s0
ARGS="--train --arm advantage --labels $LAB0/labels.parquet --table $TABLE0 --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps $STEPS --save-every 1000 --seed 0"
T=$(run -d afterok:$L0 -A iris -p iris $EX --mem=56G --time=7:30:00 --job-name=${PFX}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch)
RUN=$ROOT/runs/policy_${NAME}_$T CHK=$ROOT/runs/contrast_check_$NAME
K=$(run -d afterok:$T $EX --mem=40G --time=1:00:00 --job-name=${PFX}_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16 --table $TABLE0" scripts/pi05_tag_adarms.sbatch)
POL="MODE=eval,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$CHK/results.json"
JOBS="label $L0 train $T check $K"
if [ $CHAIN = mt100 ]; then
    E=$(run -d afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${PFX}_eval --export=ALL,$POL,MATCHED_NOISE=1,NOISE_SEED=1,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch)
    C=$(run -d afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${PFX}_ceval --export=ALL,$POL,NOISE_SEED=1,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $CE_MOKA --args.split-name cont" scripts/pi05_tag_adarms.sbatch)
    JOBS="$JOBS eval $E ceval $C"
fi
C2=$(run -d afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${PFX}_ceval --export=ALL,$POL$T64ENV,NOISE_SEED=1,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $CE_T64 --args.split-name cont" scripts/pi05_tag_adarms.sbatch)
JOBS="$JOBS ceval_t64 $C2"
echo "$TAG round 0: $JOBS (policy $RUN)"

FEED_ENV=(JUST_IN_TIME=1 LABELER=cv TAG=$TAG TPFX=$TPFX TASK_A=$TA TASK_B=$TB FIXED_REPEATS=0 FRESH=$FN
          COLLECT_TIME=2:00:00 CHECK_TABLE=1 P0=$RUN CHK0=$CHK TABLE0=$TABLE0 PREV_LAB0=$LAB0 DEP_START=afterok:$K)
if [ $CHAIN = mt100 ]; then FEED_ENV+=(CONT_EVAL=$CE_MOKA CONT_EVAL_B=$CE_T64)
else FEED_ENV+=(CONT_EVAL=$CE_T64 NO_VAL_EVAL=1); fi
if [ -n "${SUBMIT:-}" ]; then
    echo "$(date -Is) $TAG round 0 submitted: $JOBS" >> $ROOT/logs/ag_rounds_feed_$TAG.log
    env "${FEED_ENV[@]}" setsid nohup scripts/ag_rounds_feed.sh 1 $ROUNDS - - > $ROOT/logs/ag_rounds_feed_$TAG.nohup 2>&1 < /dev/null &
    echo "feeder pid $! for rounds 1-$ROUNDS (log logs/ag_rounds_feed_$TAG.log)"
else
    echo "feeder: ${FEED_ENV[*]} scripts/ag_rounds_feed.sh 1 $ROUNDS - -"
    if [ -e $TABLE0/episodes.csv ]; then echo "round 1 would be:"; env "${FEED_ENV[@]}" ROUNDS=1 bash scripts/ag_rounds.sh
    else echo "(round-1 preview needs $TABLE0, built by round 0's label job)"; fi
    echo "(dry run: SUBMIT=1 to submit)"
fi
