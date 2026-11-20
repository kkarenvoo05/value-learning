#!/bin/bash
# [2026-11-19] Two checks before the second task is chosen again (reports/2026-11-19-t64-demo-collapse; Karen: "run 1
# and 2").
#   nodemo  Does residual AG on moka need its demos (whose strategy matches pi0.5)? Each arm = an existing round-0
#           policy's exact train / check / eval commands without --demos:
#             nd900  the 900-rollout round 0 (policy_ag_residual_vb40_s0_18438185, resnet_valuebinned_p40 labels,
#                    table moka_fixed_fresh = the trainer default): val 330 at push 0.99 = 75.5 (base 52.4)
#             ndb50  b50cv round 0 (policy_agr_b50cv_r0_s0_18546931, chain-V labels): val 44.8, moka cont 51.0
#           Read: each vs its with-demos twin (same labels, seed 0, push 0.99, matched noise 1 / cont set).
#           nd900 is the clean test (AG clearly works there); ndb50 is the setting the multi-task runs use.
#   sweep   Base pi0.5 on the four suites it was trained on, 20 fresh layouts per task (the fixed-layout sweeps of
#           10-27 / 10-29 put every task but moka at 85-100%; fresh layouts are harder: moka 0.60 fixed / 0.55
#           fresh). A task at 50-80% here would be a second task whose demos match pi0.5's strategy. Per-suite step
#           caps as always (libero_rollouts.MAX_STEPS), env seeds 400-403, no raw saved.
# Usage: PARTS="nodemo sweep" scripts/demo_task_tests_submit.sh                 (dry run)
#        SUBMIT=1 PARTS="..." [PART=iris-hi] scripts/demo_task_tests_submit.sh   (submits; log logs/demo_task_tests.log)
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero R=$ROOT/runs TB=$ROOT/tables
EX="-A iris -p ${PART:-iris-hi} --exclude=iris[1-5,7-10],iris-hgx-[1-2],iris-hp-z8"
C8=$ROOT/eval_sets/moka_cont100/starts.csv
TARGS="--tag-dropout 0 --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"   # no --demos
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j" | tee -a logs/demo_task_tests.log
    else
        eval "$var=J$N"; echo "step $N $var (dep $dep): sbatch $*"
    fi
}
arm() {  # arm <tag> <labels> <table args or ""> <cont eval: 1 or 0>
    local tag=$1 lab=$2 targ=$3 cont=$4 name=agr_$1_r0_s0
    [ -f $lab ] || { echo "missing $lab"; exit 1; }
    sub T - $EX --mem=56G --time=7:30:00 --job-name=agr_${tag}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$name,ARGS="--train --arm advantage --labels $lab $targ $TARGS" scripts/pi05_tag_adarms.sbatch
    local run=$R/policy_${name}_$T chk=$R/contrast_check_$name
    sub K afterok:$T $EX --mem=40G --time=2:00:00 --job-name=agr_${tag}_check --export=ALL,MODE=contrastcheck,OUT=$chk,ARGS="--run $run --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    local pol="MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$run,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$chk/results.json"
    sub E afterok:$K $EX --mem=40G --time=5:00:00 --job-name=agr_${tag}_eval --export=ALL,$pol,MATCHED_NOISE=1,NAME=eval_${name}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
    [ $cont = 1 ] && sub C afterok:$K $EX --mem=40G --time=5:00:00 --job-name=agr_${tag}_ceval --export=ALL,$pol,NAME=ceval_${name}_shift099,CLIENT_ARGS="--args.start-from $C8 --args.split-name cont" scripts/pi05_tag_adarms.sbatch
    return 0
}
[ -n "${SUBMIT:-}" ] && echo "$(date '+%F %T') demo_task_tests submit PARTS=${PARTS:?set PARTS}" >> logs/demo_task_tests.log
for part in ${PARTS:?set PARTS (nodemo sweep)}; do case $part in
nodemo)
    arm nd900 $TB/moka_fixed_fresh/labels/resnet_valuebinned_p40/labels.parquet "" 0
    arm ndb50 $TB/moka_agr_b50_r0/labels/agr_b50cv_cv40_r0/labels.parquet "--table $TB/moka_agr_b50_r0" 1 ;;
sweep)
    s=400
    for suite in libero_10 libero_goal libero_spatial libero_object; do
        sub J - -A iris -p ${PART:-iris-hi} --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8 --mem=40G --time=6:00:00 --job-name=fresh_sweep --export=ALL,SUITE=$suite,TASKS="0 1 2 3 4 5 6 7 8 9",REPEATS=1,SEED=$s,NAME=fresh_sweep_${suite},CLIENT_ARGS="--args.fresh-layouts 20" scripts/libero_eval.sbatch
        s=$((s + 1))
    done ;;
*) echo "unknown part $part"; exit 1 ;;
esac; done
