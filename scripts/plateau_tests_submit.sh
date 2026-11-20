#!/bin/bash
# [2026-11-19] Tests from reports/2026-11-19-residual-failures (Karen: "set up the data rebalancing test as a dry run.
# also look into 2 and 3 ... prioritize these runs"). Each copies an existing round's exact recipe and changes one thing.
#
#   rebal  (data rebalancing) round 7's data and labels (tables/moka_agr_r7, agr_vb40_r7; r7 = 83.0 on the 330 val
#          trials), r7's train args + --oversample-episodes:
#            rbag   every AG-round episode x3: AG rounds 41% -> 67% of frames, base 58% -> 32%
#            rbfail the 163 AG-round failures x2: 11.7% -> 21% of frames
#          Read: vs r7 83.0 (same data / labels / seed; only the sampling differs).
#   mid    (mid-task starts) round 6 again from r5 at push 0.99, r6's seeds, 100 episodes as every round, but
#          50 normal (30 train states x 1 + 20 fresh) + 50 mid-task starts (midtask/mid6_starts.csv:
#          30 at the second pot, libero_midtask_starts.py default; 20 after a lost / tipped pot, --kind recovery).
#          Table / labels / train / check / eval exactly as rs6. Read: vs r6 77.3 and rs6 (new-seed r6).
#   cap700 (diagnostic) r5 (87.6) on the 330 val trials, matched noise 1, push 0.99, step cap 700 instead of 520.
#          Read within the run: success by step 520 vs by step 700 on the same trials.
# Usage: PARTS="rebal mid cap700" scripts/plateau_tests_submit.sh          (dry run)
#        SUBMIT=1 PARTS="mid cap700" scripts/plateau_tests_submit.sh      (submits; log logs/plateau_tests.log)
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero R=$ROOT/runs TB=$ROOT/tables
EX="-A iris -p ${PART:-iris-hi} --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8"
TRAIN_STATES="0 1 2 3 4 6 8 10 11 16 18 19 20 21 22 23 24 26 27 28 30 34 35 36 37 38 43 45 46 47"
P5=$R/policy_agr_r5_s0_18484605 K5=$R/contrast_check_agr_r5_s0/results.json
TARGS="--tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j" | tee -a logs/plateau_tests.log
    else
        eval "$var=J$N"; echo "step $N $var (dep $dep): sbatch $*"
    fi
}
check_eval() {  # check_eval <tag> <run> <train job>
    local tag=$1 run=$2 t=$3 chk=$R/contrast_check_agr_$1_r6_s0
    [ "$tag" = rbag ] || [ "$tag" = rbfail ] && chk=$R/contrast_check_agr_${tag}_r7_s0
    sub K afterok:$t $EX --mem=40G --time=1:00:00 --job-name=agr_${tag}_check --export=ALL,MODE=contrastcheck,OUT=$chk,ARGS="--run $run --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    sub E afterok:$K $EX --mem=40G --time=3:00:00 --job-name=agr_${tag}_eval --export=ALL,MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$run,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$chk/results.json,NAME=eval_$(basename $chk | sed 's/^contrast_check_//')_shift099_noise1 scripts/pi05_tag_adarms.sbatch
}
[ -n "${SUBMIT:-}" ] && echo "$(date '+%F %T') plateau_tests submit PARTS=${PARTS:?set PARTS}" >> logs/plateau_tests.log
for part in ${PARTS:?set PARTS (rebal mid cap700)}; do case $part in
rebal)
    T7=$TB/moka_agr_r7 L7=$T7/labels/agr_vb40_r7/labels.parquet
    for arm in "rbag ag_rounds.txt 0.7" "rbfail ag_failures.txt 0.2"; do
        set -- $arm
        sub T - $EX --mem=56G --time=7:30:00 --job-name=agr_$1_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=agr_$1_r7_s0,ARGS="--train --arm advantage --labels $L7 --table $T7 $TARGS --oversample-episodes $T7/rebal/$2 --oversample-share $3" scripts/pi05_tag_adarms.sbatch
        check_eval $1 $R/policy_agr_$1_r7_s0_$T $T
    done ;;
mid)
    POL="MODE=eval,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$P5,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$K5,NOISE_SEED=106"
    sub CF - $EX --mem=40G --time=2:00:00 --job-name=agr_mid_collect --export=ALL,$POL,REPEATS=1,SEED=212,NAME=agr_mid_r6_collect_fixed,CLIENT_ARGS="--args.init-states $TRAIN_STATES --args.save-raw --args.split-name train" scripts/pi05_tag_adarms.sbatch
    sub CR - $EX --mem=40G --time=2:00:00 --job-name=agr_mid_collect --export=ALL,$POL,REPEATS=1,SEED=213,NAME=agr_mid_r6_collect_fresh,CLIENT_ARGS="--args.fresh-layouts 20 --args.save-raw --args.split-name train" scripts/pi05_tag_adarms.sbatch
    sub CM - $EX --mem=40G --time=2:00:00 --job-name=agr_mid_collect --export=ALL,$POL,REPEATS=1,SEED=214,NAME=agr_mid_r6_collect_midtask,CLIENT_ARGS="--args.start-from $ROOT/midtask/mid6_starts.csv --args.save-raw --args.split-name train" scripts/pi05_tag_adarms.sbatch
    TM=$TB/moka_agr_mid_r6 LM=$TB/moka_agr_mid_r6/labels/agr_mid_vb40_r6
    RUNS="$R/agr_mid_r6_collect_fixed_libero_10_$CF $R/agr_mid_r6_collect_fresh_libero_10_$CR $R/agr_mid_r6_collect_midtask_libero_10_$CM"
    sub TL afterok:$CF:$CR:$CM --time=1:00:00 --job-name=agr_mid_label --export=ALL,CMD=".venv/bin/python scripts/libero_round2_table.py --base $TB/moka_agr_r5 --round 7 --out $TM --runs $RUNS && .venv/bin/python scripts/libero_round_labels.py --table $TM --round 7 --prev $TB/moka_agr_r5/labels/agr_vb40_r5 --out $LM" scripts/vl_python.sbatch
    sub T afterok:$TL $EX --mem=56G --time=7:30:00 --job-name=agr_mid_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=agr_mid_r6_s0,ARGS="--train --arm advantage --labels $LM/labels.parquet --table $TM $TARGS" scripts/pi05_tag_adarms.sbatch
    check_eval mid $R/policy_agr_mid_r6_s0_$T $T ;;
cap700)
    sub E - $EX --mem=40G --time=5:00:00 --job-name=agr_cap700_eval --export=ALL,MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$P5,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$K5,AG_MAX_STEPS=700,NAME=eval_agr_r5_s0_shift099_noise1_cap700 scripts/pi05_tag_adarms.sbatch ;;
*) echo "unknown part $part"; exit 1 ;;
esac; done
