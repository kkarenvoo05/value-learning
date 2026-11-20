#!/bin/bash
# [2026-11-19] Does training on two tasks help, once both tasks have demos? (Karen: "queue the correct jobs to test our
# experiment on whether having two tasks would help"). Round 0 only, every recipe setting as the existing round-0
# chains (agr_t64b50cv / agr_mt100cv / agr_b50cv: same tables, same chain-V cv40 labels, same train args, seed 0);
# the only change is --demo-table-extra tables/libero90_t64_demos (35 LIBERO-90 task-64 demos, tagged positive as the
# moka demos are), so task 64 has demos like moka:
#   t64d50 = 50 task-64 rollouts + 35 task-64 demos                        (single task, task 64)
#   mtd100 = 50 moka + 50 task-64 rollouts + 29 moka + 35 task-64 demos     (two tasks)
#   b50cv  = 50 moka rollouts + 29 moka demos  (existing: policy_agr_b50cv_r0_s0_18546931, moka cont 0.510)
# Read: task 64: mtd100 vs t64d50; moka: mtd100 vs b50cv (continuous sets, paired by row).
# Each arm: train -> contrast check (per task for mtd100, below) -> continuous-set evals at push 0.99 (the protocol) and, on task
# 64, also push 0.5 (task 64's push 0.99 is ~1.5x moka's absolute shift; see the t64sweep jobs).
# Usage: scripts/two_task_demos_submit.sh            (dry run: prints the sbatch commands)
#        SUBMIT=1 [PART=iris-hi] [ARMS="t64d50 mtd100"] [TR_MTD100=<train id>] scripts/two_task_demos_submit.sh   (submits; log logs/two_task_demos.log; PART default iris)
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
TB=$ROOT/tables
XT=$TB/libero90_t64_demos
[ -f $XT/episodes.csv ] || { echo "missing $XT"; exit 1; }
EX="-A iris -p ${PART:-iris} --exclude=iris[1-5,8-10],iris-hgx-[1-2],iris-hp-z8"
T64="AG_SUITE=libero_90,AG_TASK=64,AG_MAX_STEPS=520"
C64=$ROOT/eval_sets/t64_cont100/starts.csv C8=$ROOT/eval_sets/moka_cont100/starts.csv
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j" | tee -a logs/two_task_demos.log
    else
        eval "$var=J$N"; echo "step $N $var (dep $dep): sbatch $*"
    fi
}
arm() {  # arm <tag> <table> <labels dir name> [<existing train job id>]
    local tag=$1 table=$TB/$2 lab=$3 name=agr_$1_r0_s0
    if [ -n "${4:-}" ]; then TR=$4; echo "reuse train $TR for $tag"; else
    sub TR - $EX --mem=56G --time=7:30:00 --job-name=agr_${tag}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$name,ARGS="--train --arm advantage --labels $table/labels/$lab/labels.parquet --table $table --tag-dropout 0 --demos --demo-table-extra $XT --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0" scripts/pi05_tag_adarms.sbatch
    fi
    RUN=$ROOT/runs/policy_${name}_$TR
}
check() {  # check <tag> <table whose val frames calibrate w> <out suffix or ->; sets CK, CHK
    local tag=$1 table=$TB/$2 sfx=${3#-} name=agr_$1_r0_s0
    CHK=$ROOT/runs/contrast_check_$name${sfx:+_$sfx}
    sub CK afterok:$TR $EX --mem=40G --time=1:00:00 --job-name=agr_${tag}_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16 --table $table" scripts/pi05_tag_adarms.sbatch
}
ceval() {  # ceval <tag> <push> <stag> <task env or -> <starts> [<name suffix>]; uses the last check's CK / CHK
    local tag=$1 push=$2 stag=$3 env=$4 starts=$5 name=agr_$1_r0_s0${6:+_$6}
    sub EV afterok:$CK $EX --mem=40G --time=5:00:00 --job-name=agr_${tag}_ceval --export=ALL,MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$push,FT_SHIFT_FROM=$CHK/results.json${env:+$([ "$env" != - ] && echo ",$env")},NAME=ceval_${name}_shift$stag,CLIENT_ARGS="--args.start-from $starts --args.split-name cont" scripts/pi05_tag_adarms.sbatch
}
[ -n "${SUBMIT:-}" ] && echo "$(date '+%F %T') two_task_demos submit" >> logs/two_task_demos.log
if [[ " ${ARMS:-t64d50 mtd100} " == *" t64d50 "* ]]; then
    arm t64d50 t64_agr_t64b50cv_r0 agr_t64b50cv_cv40_r0 ${TR_T64D50:-}
    check t64d50 t64_agr_t64b50cv_r0 -
    ceval t64d50 0.99 099 $T64 $C64
    ceval t64d50 0.5 05 $T64 $C64
fi
if [[ " ${ARMS:-t64d50 mtd100} " == *" mtd100 "* ]]; then
    # [2026-11-19, Karen: "recalibrate the two-task evals per task"] one w per task, each from a check on that task's
    # own held-out frames, using the same tables (so the same seeded frames) as the single-task arm it is compared
    # with: task 64 = t64_agr_t64b50cv_r0's 20 val episodes (as t64d50), moka = moka_agr_b50_r0's 24 of 120 (as b50cv).
    # The combined-table check drew 21 moka + 3 task-64 episodes (mt100cv), so one w was ~all moka.
    arm mtd100 mt_agr_mt100cv_r0 agr_mt100cv_cv40_r0 ${TR_MTD100:-}
    check mtd100 t64_agr_t64b50cv_r0 on_t64
    ceval mtd100 0.99 099 $T64 $C64 t64cal
    ceval mtd100 0.5 05 $T64 $C64 t64cal
    check mtd100 moka_agr_b50_r0 on_moka
    ceval mtd100 0.99 099 - $C8 mokacal
fi
# Not here: mt100cv (two-task, moka demos only) at a task-64-calibrated w (check 18587063) - submitted by hand 11-19.
