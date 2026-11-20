#!/bin/bash
# Fixed-budget AG chain: the same number of episodes in every round, round 0 included [2026-11-16; mentor: "all rounds
# can be the same, so 50/100/200 every time, including the first, instead of the 900"; Karen: "build the subsample
# round 0"]. Replaces the stopped e50 / e200 chains (those started from the 900 base rollouts).
#   round 0: table + labels = scripts/libero_round0_subsample.py (built beforehand: tables/moka_agr_b<B>_r0), then the
#            residual trained exactly as the 900-episode round 0 (policy_ag_residual_vb40_s0_18438185's args +
#            --table) -> contrast check -> 330-trial eval at shift 0.99 (matched noise 1)
#   rounds 1..ROUNDS: scripts/ag_rounds_feed.sh -> scripts/ag_rounds.sh with TAG=b<B>, round 0 overridden, the first
#            collection after round 0's check. Per round: B fresh layouts, over two collectors [2026-11-16, Karen:
#            "switch to 100% fresh layouts"; the 30 fixed train states are no longer collected on].
# Evals per round: the 330 val trials (10 fixed val states x 33, matched noise 1) and the continuous-layout set
#   CONT_EVAL (default eval_sets/moka_cont100: 100 fresh layouts x 3, scripts/libero_eval_layouts.py), round 0 included.
# V: the frozen round-0 OOF ResNet for every budget (fit on all 900 base rollouts: option (a), stated caveat).
# Usage: B=50|100|200 [ROUNDS=10] [SUBMIT=1] [NOFEED=1: round 0 only] scripts/ag_budget_chain.sh   (dry run without SUBMIT)
# The feeder runs under nohup on this node; log logs/ag_rounds_feed_b<B>.log.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
: "${B:?B=50|100|200}"
ROUNDS=${ROUNDS:-10} SHIFT=0.99 STEPS=5000
case $B in 50) FR=0 FN=50 CT=2:00:00 ;; 100) FR=0 FN=100 CT=2:00:00 ;; 200) FR=0 FN=200 CT=3:00:00 ;;
           *) echo "B must be 50, 100 or 200"; exit 1 ;; esac
# [2026-11-16] LABELER=iql: the IQL-AG chain (TAG b<B>iql): same round-0 subsample table as b<B> (paired start), tags
# from an IQL critic trained from scratch on the chain's own episodes (round 0: a label job first; later rounds:
# ag_rounds.sh LABELER=iql). Default LABELER=vb40 (frozen round-0 OOF ResNet V).
LABELER=${LABELER:-vb40}
TABLE0=$ROOT/tables/moka_agr_b${B}_r0
case $LABELER in
    vb40) TAG=b$B PFX=agr_b$B; LAB0=$TABLE0/labels/${PFX}_vb40_r0; NEED="$LAB0/labels.parquet $LAB0/v.parquet" ;;
    iql) TAG=b${B}iql PFX=agr_b${B}iql; LAB0=$TABLE0/labels/${PFX}_iql40_r0; NEED= ;;
    # [2026-11-16; Karen: "set up chain-only V"] LABELER=cv (TAG b<B>cv): the vb40 ResNet V refit each round on the
    # chain's own episodes only (scripts/libero_chainv_round_labels.py); round 0: fit on the B subsample episodes
    cv) TAG=b${B}cv PFX=agr_b${B}cv; LAB0=$TABLE0/labels/${PFX}_cv40_r0; NEED= ;;
    *) echo "LABELER must be vb40, iql or cv"; exit 1 ;;
esac
for f in $TABLE0/episodes.csv $TABLE0/subsample.json $NEED; do
    [ -e $f ] || { echo "missing $f (run scripts/libero_round0_subsample.py first)"; exit 1; }; done
n=$(python3 -c "import json; print(json.load(open('$TABLE0/subsample.json'))['episodes'])")
[ "$n" -eq "$B" ] || { echo "$TABLE0 has $n episodes, expected $B"; exit 1; }
nf=$(python3 -c "import json; print(json.load(open('$TABLE0/subsample.json'))['fixed'])")
[ "$nf" -eq 0 ] || { echo "$TABLE0 has $nf fixed-state episodes, expected fresh layouts only"; exit 1; }
CONT_EVAL=${CONT_EVAL:-$ROOT/eval_sets/moka_cont100/starts.csv}
[ -e "$CONT_EVAL" ] || { echo "missing $CONT_EVAL"; exit 1; }
EX="--exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8"
cd $VL

NAME=${PFX}_r0_s0
ARGS="--train --arm advantage --labels $LAB0/labels.parquet --table $TABLE0 --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps $STEPS --save-every 1000 --seed 0"
run() { if [ -n "${SUBMIT:-}" ]; then sbatch --parsable "$@"; else echo "J$RANDOM"; echo "  sbatch $*" >&2; fi; }
L0=
if [ $LABELER = iql ]; then
    L0=$(run --time=2:00:00 --job-name=${PFX}_label --export=ALL,CMD=".venv/bin/python scripts/libero_iql_round_labels.py --table $TABLE0 --out $LAB0" scripts/vl_python.sbatch)
elif [ $LABELER = cv ]; then
    L0=$(run --time=8:00:00 --job-name=${PFX}_label --export=ALL,CMD=".venv/bin/python scripts/libero_chainv_round_labels.py --table $TABLE0 --out $LAB0" scripts/vl_python.sbatch)
fi
T=$(run ${L0:+-d afterok:$L0} -A iris -p iris $EX --mem=56G --time=7:30:00 --job-name=${PFX}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch)
RUN=$ROOT/runs/policy_${NAME}_$T CHK=$ROOT/runs/contrast_check_$NAME
K=$(run -d afterok:$T $EX --mem=40G --time=1:00:00 --job-name=${PFX}_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch)
E=$(run -d afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${PFX}_eval --export=ALL,MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch)
C=$(run -d afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${PFX}_ceval --export=ALL,MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$CHK/results.json,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $CONT_EVAL --args.split-name cont" scripts/pi05_tag_adarms.sbatch)
echo "$TAG round 0: ${L0:+label $L0 }train $T check $K eval $E ceval $C (policy $RUN)"

FEED_ENV=(JUST_IN_TIME=1 LABELER=$LABELER TAG=$TAG FIXED_REPEATS=$FR FRESH=$FN CONT_EVAL=$CONT_EVAL COLLECT_TIME=$CT P0=$RUN CHK0=$CHK TABLE0=$TABLE0 PREV_LAB0=$LAB0 DEP_START=afterok:$K)
if [ -n "${NOFEED:-}" ]; then  # [2026-11-16] scripts/ag_budget_rotate.sh submits the later rounds itself
    :
elif [ -n "${SUBMIT:-}" ]; then
    echo "$(date -Is) $TAG round 0 submitted: ${L0:+label $L0 }train $T check $K eval $E ceval $C" >> $ROOT/logs/ag_rounds_feed_$TAG.log
    env "${FEED_ENV[@]}" setsid nohup scripts/ag_rounds_feed.sh 1 $ROUNDS - - > $ROOT/logs/ag_rounds_feed_$TAG.nohup 2>&1 < /dev/null &
    echo "feeder pid $! for rounds 1-$ROUNDS (log logs/ag_rounds_feed_$TAG.log)"
else
    echo "feeder: ${FEED_ENV[*]} scripts/ag_rounds_feed.sh 1 $ROUNDS - -"; echo "round 1 would be:"
    env "${FEED_ENV[@]}" ROUNDS=1 bash scripts/ag_rounds.sh | sed -n 1,4p | cut -c1-260
    echo "(dry run: SUBMIT=1 to submit)"
fi
