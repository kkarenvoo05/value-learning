#!/bin/bash
# [2026-11-19] Critic tests (Karen: "set up the b50 900-episode V test as a dry run. i also want to set up the big shared
# critic for the multi task run"). Round 0 only; each copies an existing round-0 recipe and changes only V.
#
#   b50v900  moka 50-episode round 0 (tables/moka_agr_b50_r0) with labels from the frozen 900-episode OOF V
#            (agr_b50_vb40_r0, already on disk) instead of the chain-only V (agr_b50cv_cv40_r0). Train / check /
#            evals = b50cv r0's exact commands (policy_agr_b50cv_r0_s0_18546931: val 330 44.8, moka_cont100 51.0).
#            Read: b50v900 vs b50cv -> is the 50-episode chain held back by its weak V? (b100 vs b100cv is the same
#            comparison at 100: r0 79.7 vs 76.1.)
#   bigv     one shared ResNet V (the chain-V method, scripts/libero_chainv_round_labels.py: 5 OOF folds, 8000 steps)
#            fit on a big two-task table of base pi0.5 rollouts, fresh layouts only (the method's fold rule):
#              moka   the 450 fresh-layout base rollouts of moka_fixed_fresh (libero_round0_subsample --fresh 450)
#              task64 t64_base_A + B (100, the chains' round-0 runs) + 350 new (runs C-F, env seeds 303-306)
#              val    moka's 120 val-state rollouts + t64_base_V's 20 (scored only, never fit)
#            900 train episodes = the frozen moka V's size, half per task. Both chains' round-0 episodes (b50 moka
#            subsample, t64_base_A) are inside it, so each gets an out-of-fold V (scripts/libero_bigv_labels.py; check:
#            the frozen 900 V through it reproduces agr_b50_vb40_r0 exactly). Labels -> mt_agr_mt100cv_r0 (the
#            two-task round-0 table), 40% cut within (round, task) as before. Two arms on that table:
#              mtbv   train args = mt100cv r0 (moka demos only)        compare: mt100cv r0 (chain V, same data)
#              mtbvd  + --demo-table-extra (35 task-64 demos)        compare: mtd100 (chain V, same data)
#            Checks per task (as mtd100): on_t64 = t64_agr_t64b50cv_r0's val frames, on_moka = moka_agr_b50_r0's.
#            Evals at push 0.99: t64_cont100 (t64cal, FT_SHIFT_INTERP=1: the demo runs' shift is superlinear in w),
#            moka_cont100 + moka val 330 (mokacal, least squares as every moka eval).
# Usage: PARTS="b50v900 bigv" scripts/critic_tests_submit.sh                (dry run)
#        SUBMIT=1 PARTS="..." [PART=iris-hi] scripts/critic_tests_submit.sh  (submits; log logs/critic_tests.log)
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero R=$ROOT/runs TB=$ROOT/tables
EX="-A iris -p ${PART:-iris-hi} --exclude=iris[1-5,7-10],iris-hgx-[1-2],iris-hp-z8"
T64="AG_SUITE=libero_90,AG_TASK=64,AG_MAX_STEPS=520"
C64=$ROOT/eval_sets/t64_cont100/starts.csv C8=$ROOT/eval_sets/moka_cont100/starts.csv
XT=$TB/libero90_t64_demos
TARGS="--tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j" | tee -a logs/critic_tests.log
    else
        eval "$var=J$N"; echo "step $N $var (dep $dep): sbatch $*"
    fi
}
ev() {  # ev <var> <check job> <check dir> <run> <name> <extra env or -> [<starts>]   (no starts = val 330, matched noise)
    local v=$1 k=$2 chk=$3 run=$4 name=$5 env=${6#-} starts=${7:-}
    local pol="MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$run,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$chk/results.json${env:+,$env}"
    if [ -n "$starts" ]; then
        sub $v afterok:$k $EX --mem=40G --time=5:00:00 --job-name=${name%%_r0*}_ceval --export=ALL,$pol,NAME=ceval_${name}_shift099,CLIENT_ARGS="--args.start-from $starts --args.split-name cont" scripts/pi05_tag_adarms.sbatch
    else
        sub $v afterok:$k $EX --mem=40G --time=5:00:00 --job-name=${name%%_r0*}_eval --export=ALL,$pol,MATCHED_NOISE=1,NAME=eval_${name}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
    fi
}
[ -n "${SUBMIT:-}" ] && echo "$(date '+%F %T') critic_tests submit PARTS=${PARTS:?set PARTS}" >> logs/critic_tests.log
for part in ${PARTS:?set PARTS (b50v900 bigv)}; do case $part in
b50v900)
    T0=$TB/moka_agr_b50_r0 L=$T0/labels/agr_b50_vb40_r0/labels.parquet NAME=agr_b50v900_r0_s0
    [ -f $L ] || { echo "missing $L"; exit 1; }
    sub T - $EX --mem=56G --time=7:30:00 --job-name=agr_b50v900_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="--train --arm advantage --labels $L --table $T0 $TARGS" scripts/pi05_tag_adarms.sbatch
    RUN=$R/policy_${NAME}_$T CHK=$R/contrast_check_$NAME
    sub K afterok:$T $EX --mem=40G --time=2:00:00 --job-name=agr_b50v900_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    ev E $K $CHK $RUN $NAME -
    ev C $K $CHK $RUN $NAME - $C8 ;;
bigv)
    RA=$R/t64_base_A_libero_90_18549968 RB=$R/t64_base_B_libero_90_18549969 RV=$R/t64_base_V_libero_90_18550715
    for r in $RA $RB $RV; do [ -f $r/episodes.csv ] || { echo "missing $r"; exit 1; }; done
    NEW="" DEPS=""
    for x in "C 303 100" "D 304 100" "E 305 100" "F 306 50"; do
        set -- $x
        sub J - -A iris -p iris --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8 --mem=40G --time=3:00:00 --job-name=t64_base --export=ALL,SUITE=libero_90,TASKS=64,REPEATS=1,SEED=$2,NAME=t64_base_$1,CLIENT_ARGS="--args.fresh-layouts $3 --args.save-raw --args.max-steps 520 --args.split-name train" scripts/libero_eval.sbatch
        NEW="$NEW $R/t64_base_$1_libero_90_$J" DEPS="$DEPS:$J"
    done
    M450=$TB/moka_fresh450_r0 BIG=$TB/mt_bigv BL=$TB/mt_bigv/labels/bigv_cv40
    MT=$TB/mt_agr_mt100cv_r0 LAB=$TB/mt_agr_mt100cv_r0/labels/agr_mtbv_r0
    CMD=".venv/bin/python scripts/libero_round0_subsample.py --fixed-repeats 0 --fresh 450 --out $M450 --label-name agr_fresh450_vb40_r0"
    CMD="$CMD && .venv/bin/python scripts/libero_round2_table.py --base $M450 --round 1 --keep-split --out $BIG --runs $RA $RB$NEW $RV"
    CMD="$CMD && .venv/bin/python scripts/libero_chainv_round_labels.py --table $BIG --out $BL"
    CMD="$CMD && .venv/bin/python scripts/libero_bigv_labels.py --big-v $BL/v.parquet --big-table $BIG --table $MT --out $LAB"
    sub L afterok${DEPS} --mem=96G --time=24:00:00 --job-name=bigv_label --export=ALL,CMD="$CMD" scripts/vl_python.sbatch
    for arm in ${ARMS:-mtbv mtbvd}; do   # [2026-11-19] ARMS=mtbv: no task-64 demos (the demo strategy clash, reports/2026-11-19-t64-demo-collapse)
        NAME=agr_${arm}_r0_s0 XD=""; [ $arm = mtbvd ] && XD="--demo-table-extra $XT"
        sub T afterok:$L $EX --mem=56G --time=7:30:00 --job-name=agr_${arm}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="--train --arm advantage --labels $LAB/labels.parquet --table $MT $TARGS $XD" scripts/pi05_tag_adarms.sbatch
        RUN=$R/policy_${NAME}_$T
        CK64=$R/contrast_check_${NAME}_on_t64 CK8=$R/contrast_check_${NAME}_on_moka
        sub K1 afterok:$T $EX --mem=40G --time=2:00:00 --job-name=agr_${arm}_check --export=ALL,MODE=contrastcheck,OUT=$CK64,ARGS="--run $RUN --weights 1 2 4 8 16 --table $TB/t64_agr_t64b50cv_r0" scripts/pi05_tag_adarms.sbatch
        ev E1 $K1 $CK64 $RUN ${NAME}_t64cal "$T64,FT_SHIFT_INTERP=1" $C64
        sub K2 afterok:$T $EX --mem=40G --time=2:00:00 --job-name=agr_${arm}_check --export=ALL,MODE=contrastcheck,OUT=$CK8,ARGS="--run $RUN --weights 1 2 4 8 16 --table $TB/moka_agr_b50_r0" scripts/pi05_tag_adarms.sbatch
        ev E2 $K2 $CK8 $RUN ${NAME}_mokacal - $C8
        ev E3 $K2 $CK8 $RUN ${NAME}_mokacal -
    done ;;
b100s1)
    # [2026-11-19, Karen: "add the b100 seed 1 retrain"] b100 round 1 (val 53.0 after r0 79.7; its own collection
    # 43/100) retrained with --seed 1, everything else = policy_agr_b100_r1_s0_18554179 / its check / evals. Read:
    # back near r0 -> the dip was that run; still low -> something in round 1's data.
    T1=$TB/moka_agr_b100_r1 NAME=agr_b100_r1_s1
    sub T - $EX --mem=56G --time=7:30:00 --job-name=agr_b100s1_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="--train --arm advantage --labels $T1/labels/agr_b100_vb40_r1/labels.parquet --table $T1 ${TARGS/--seed 0/--seed 1}" scripts/pi05_tag_adarms.sbatch
    RUN=$R/policy_${NAME}_$T CHK=$R/contrast_check_$NAME
    sub K afterok:$T $EX --mem=40G --time=2:00:00 --job-name=agr_b100s1_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    ev E $K $CHK $RUN $NAME -
    ev C $K $CHK $RUN $NAME - $C8 ;;
*) echo "unknown part $part"; exit 1 ;;
esac; done
