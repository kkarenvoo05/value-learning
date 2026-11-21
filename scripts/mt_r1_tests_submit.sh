#!/bin/bash
# [2026-11-20] Two tests of the mt100cv round-1 drop (reports/2026-11-20-mt-r1-drop; Karen: "run the base-oversampled
# retrain and the push sweep"). r1 = policy_agr_mt100cv_r1_s0_18581016 (table mt_agr_mt100cv_r1 = 100 base episodes,
# round 1, + 100 collected by the guided r0 policy, round 2; 29 moka demos; chain-V cv40 labels): moka cont 33.3,
# task-64 cont 31.4 (207 / 300) at push 0.99, vs r0 63.7 / 77.3.
#   overs  r1 retrained with the 100 base episodes (rebal/base_round1.txt, 50 moka + 50 task 64) repeated 9x
#          (--oversample-share 0.88 -> k = 9): base = 90.8% of rollout frames, as the 900 chain's round 1 (90.4%).
#          Everything else = r1's train args (same labels, seed 0). Check on the r1 table as the chain does (one
#          combined w); evals at push 0.99 = r1's three: moka val 330 (matched noise 1), moka cont, task-64 cont.
#          Read: back near r0 -> the guided-data share caused the drop.
#   sweep  the existing r1 policy at push 0.3 and 0.6 on the first start of each layout (100 per set:
#          moka_cont100 / t64_cont100 starts_rep0.csv, as the 11-19 task-64 sweep), from r1's own check.
#          Read: a lower push still well below r0 -> the guidance direction is wrong, not over-scaled.
# Usage: PARTS="overs sweep" scripts/mt_r1_tests_submit.sh                 (dry run)
#        SUBMIT=1 PARTS="..." [PART=iris-hi] scripts/mt_r1_tests_submit.sh   (submits; log logs/mt_r1_tests.log)
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero R=$ROOT/runs TB=$ROOT/tables ES=$ROOT/eval_sets
EX="-A iris -p ${PART:-iris-hi} --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8"   # iris7 hung 11-20
T1=$TB/mt_agr_mt100cv_r1 L1=$TB/mt_agr_mt100cv_r1/labels/agr_mt100cv_cv40_r1/labels.parquet
R1=$R/policy_agr_mt100cv_r1_s0_18581016 K1=$R/contrast_check_agr_mt100cv_r1_s0
T64="AG_SUITE=libero_90,AG_TASK=64,AG_MAX_STEPS=520"
for f in $L1 $T1/rebal/base_round1.txt $R1/trainable_final.msgpack $K1/results.json $ES/moka_cont100/starts_rep0.csv \
         $ES/t64_cont100/starts_rep0.csv; do [ -e $f ] || { echo "missing $f"; exit 1; }; done
TARGS="--tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j" | tee -a logs/mt_r1_tests.log
    else
        eval "$var=J$N"; echo "step $N $var (dep $dep): sbatch $*"
    fi
}
[ -n "${SUBMIT:-}" ] && echo "$(date '+%F %T') mt_r1_tests submit PARTS=${PARTS:?set PARTS}" >> logs/mt_r1_tests.log
for part in ${PARTS:?set PARTS (overs overs77 seed1 sweep)}; do case $part in
overs|overs77)
    # [2026-11-20, Karen: "set up the oversampled retrain at 70-80% base"] overs77 = the same retrain with --oversample-share
    # 0.71 -> k = 3 (base ~77% of rollout frames; k 9 gave 90.8%, no oversampling ~52%). Read: above r0 -> the guided
    # episodes help once weighted down; at / below r0 -> they add nothing at any weight tried.
    if [ $part = overs ]; then JP=agr_mtos OS=0.88; else JP=agr_mtos77 OS=0.71; fi
    NAME=${JP}_r1_s0
    sub T - $EX --mem=56G --time=7:30:00 --job-name=${JP}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="--train --arm advantage --labels $L1 --table $T1 $TARGS --oversample-episodes $T1/rebal/base_round1.txt --oversample-share $OS" scripts/pi05_tag_adarms.sbatch
    RUN=$R/policy_${NAME}_$T CHK=$R/contrast_check_$NAME
    sub K afterok:$T $EX --mem=40G --time=2:00:00 --job-name=${JP}_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16 --table $T1" scripts/pi05_tag_adarms.sbatch
    POL="MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$CHK/results.json"
    sub E afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${JP}_eval --export=ALL,$POL,MATCHED_NOISE=1,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
    sub C afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${JP}_ceval --export=ALL,$POL,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $ES/moka_cont100/starts.csv --args.split-name cont" scripts/pi05_tag_adarms.sbatch
    sub C2 afterok:$K $EX --mem=40G --time=3:00:00 --job-name=${JP}_ceval --export=ALL,$POL,$T64,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $ES/t64_cont100/starts.csv --args.split-name cont" scripts/pi05_tag_adarms.sbatch ;;
seed1)
    # [2026-11-20, Karen: "submit the seed-1 reruns at top priority"] b100 r1 retrained with --seed 1 (same data / labels)
    # matched r0 (val 77.0 / moka cont 78.0 vs seed 0 53.0 / 45.3), so retrain b50cv r1 and mt100cv r1 with seed 1 too.
    # Everything else = their r1 runs (policy_agr_b50cv_r1_s0_18581088, policy_agr_mt100cv_r1_s0_18581016), same checks
    # and evals at push 0.99. Read: back near r0 -> the r1 drop is mostly training-seed variance; still low -> real.
    TA1="--tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 1"
    for c in b50cv mt100cv; do
        if [ $c = b50cv ]; then JP=agr_b50cvs1 TB1=$TB/moka_agr_b50cv_r1 KT=""; else JP=agr_mts1 TB1=$T1 KT="--table $T1"; fi
        NAME=agr_${c}_r1_s1
        sub T - $EX --mem=56G --time=7:30:00 --job-name=${JP}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="--train --arm advantage --labels $TB1/labels/agr_${c}_cv40_r1/labels.parquet --table $TB1 $TA1" scripts/pi05_tag_adarms.sbatch
        RUN=$R/policy_${NAME}_$T CHK=$R/contrast_check_$NAME
        sub K afterok:$T $EX --mem=40G --time=2:00:00 --job-name=${JP}_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16 $KT" scripts/pi05_tag_adarms.sbatch
        POL="MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$CHK/results.json"
        sub E afterok:$K $EX --mem=40G --time=5:00:00 --job-name=${JP}_eval --export=ALL,$POL,MATCHED_NOISE=1,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
        sub C afterok:$K $EX --mem=40G --time=5:00:00 --job-name=${JP}_ceval --export=ALL,$POL,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $ES/moka_cont100/starts.csv --args.split-name cont" scripts/pi05_tag_adarms.sbatch
        [ $c = mt100cv ] && sub C2 afterok:$K $EX --mem=40G --time=5:00:00 --job-name=${JP}_ceval --export=ALL,$POL,$T64,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $ES/t64_cont100/starts.csv --args.split-name cont" scripts/pi05_tag_adarms.sbatch
    done ;;
sweep)
    for s in 0.3 0.6; do
        st=$(echo $s | tr -d .)
        POL="MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$R1,FT_SHIFT_TARGET=$s,FT_SHIFT_FROM=$K1/results.json"
        sub S - $EX --mem=40G --time=2:00:00 --job-name=mtsweep --export=ALL,$POL,NAME=sweep_mt100cv_r1_shift${st}_rep0,CLIENT_ARGS="--args.start-from $ES/moka_cont100/starts_rep0.csv --args.split-name cont" scripts/pi05_tag_adarms.sbatch
        sub S - $EX --mem=40G --time=2:00:00 --job-name=mtsweep --export=ALL,$POL,$T64,NAME=sweep_mt100cv_r1_shift${st}_rep0,CLIENT_ARGS="--args.start-from $ES/t64_cont100/starts_rep0.csv --args.split-name cont" scripts/pi05_tag_adarms.sbatch
    done ;;
*) echo "unknown part $part"; exit 1 ;;
esac; done
