#!/bin/bash
# Advantage Guidance residual parameterisation, first run + evals (2026-11-11; reports/2026-11-11-advantage-guidance).
# Prints the sbatch chain; SUBMIT=1 submits it. One GPU slot, used sequentially (GPU cap <= 6 incl. interactive):
#   train (frozen pi0.5 + residual head, the AdaRMS vb40 run's data / labels / demos, tag dropout 0)
#   -> contrast check (shift / noise per w, as for 18355677)
#   -> eval 1: contrastive guidance at the w whose action shift matches AdaRMS w 8 (shift/noise 0.99; +0.16 vs base)
#   -> eval 2: AdaRMS run 18355677 at w 16 (does the dose-response keep rising?)
#   -> eval 3: residual at the shift of AdaRMS w 4 (0.57; AdaRMS there: -0.03 vs base)
# Evals = the 330 val trials, matched noise seed 1 (same as base 0.524 and every guidance eval).
# DEP_TRAIN: dependency of the training job (selftest afterok + the slot it waits for).
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
ADARMS=$ROOT/runs/policy_armC_adarms_vb40_s0_18355677
LAB=$ROOT/tables/moka_fixed_fresh/labels/resnet_valuebinned_p40/labels.parquet
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
NAME=ag_residual_vb40_s0
ARGS="--train --arm advantage --labels $LAB --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
EV="MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1"
cd $VL
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j"
    else
        eval "$var=J$N"; echo "step $N $var (dep ${dep}): sbatch $*"
    fi
}
sub T "${DEP_TRAIN:--}" -A iris -p iris $EX --time=7:30:00 --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch
RUN=$ROOT/runs/policy_${NAME}_$T
CHK=$ROOT/runs/contrast_check_${NAME}
sub C afterok:$T $EX --time=1:00:00 --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
sub E1 afterok:$C $EX --time=3:00:00 --job-name=cg_eval --export=ALL,$EV,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
sub E2 afterany:$E1 $EX --time=3:00:00 --job-name=cg_eval --export=ALL,$EV,FT_RUN=$ADARMS,FT_CONTRAST=16,NAME=eval_adarms_vb40_cg16_noise1 scripts/pi05_tag_adarms.sbatch
sub E3 afterany:$E2 $EX --time=3:00:00 --job-name=cg_eval --export=ALL,$EV,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=0.57,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift057_noise1 scripts/pi05_tag_adarms.sbatch
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
