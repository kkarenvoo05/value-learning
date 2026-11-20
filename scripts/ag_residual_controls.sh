#!/bin/bash
# Checks on the 75.4% residual before trusting the rounds built on it (Karen 2026-11-12: "submit the shuffled-label
# residual and seed 2"). Prints the sbatch chain; SUBMIT=1 submits it.
#   1. shuffled-label residual: round 0's exact residual recipe (scripts/ag_residual_chain.sh) with labels
#      control_shuffled_vb40 (vb40 label sequences permuted across episodes; same 41% positive share and clumping,
#      demos still positive, no relation to the episode's states or outcome). Same null control as AdaRMS 18438131.
#      Read: ~base -> the residual's gain comes from the advantage labels; ~70% -> it is demo-likeness / amplification.
#   2. second seed of the real-label residual (--seed 1; round 0 = seed 0, 75.4%).
# Each: train (7.5 h limit, ~3.1 h) -> contrast check (w 1 2 4 8 16) -> eval at action shift 0.99 x noise (the
# residual's own shift-matched w, as 18438187), 330 val trials, matched noise seed 1.
# One lane, strictly sequential (shuffled first), so +1 GPU: cap <= 6 incl. interactive (rounds peak 3 + IQL 1).
# Job names agc_* (not agr_/cg_eval), so a failure here does not stop scripts/ag_rounds_feed.sh.
# Submit limit (iris QOS MaxSubmitPU 25): with SUBMIT=1 each sbatch is retried every 5 min until slurm accepts it.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
LABS=$ROOT/tables/moka_fixed_fresh/labels
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
EV="MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1"
cd $VL
for f in $LABS/control_shuffled_vb40/labels.parquet $LABS/resnet_valuebinned_p40/labels.parquet; do
    [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j out
        until out=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@" 2>&1); do
            echo "$out" | grep -q QOSMaxSubmitJobPerUserLimit || { echo "sbatch failed: $out"; exit 1; }
            sleep 300
        done
        j=$out; eval "$var=$j"; echo "$(date -Is) step $N $var: $j"
    else
        eval "$var=J$N"; echo "step $N $var (dep ${dep}): sbatch $*"
    fi
}
DEP=${DEP_START:--}
for spec in "ag_residual_shufvb40_s0 control_shuffled_vb40 0" "ag_residual_vb40_s1 resnet_valuebinned_p40 1"; do
    read -r NAME LSET SEED <<< "$spec"
    echo "# $NAME: labels $LSET, seed $SEED"
    ARGS="--train --arm advantage --labels $LABS/$LSET/labels.parquet --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed $SEED"
    sub T "$DEP" -A iris -p iris $EX --time=7:30:00 --job-name=agc_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch
    RUN=$ROOT/runs/policy_${NAME}_$T
    CHK=$ROOT/runs/contrast_check_${NAME}
    sub K afterok:$T $EX --time=1:00:00 --job-name=agc_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    sub E afterok:$K $EX --time=3:00:00 --job-name=agc_eval --export=ALL,$EV,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
    DEP=afterany:$E   # next lane step waits for the GPU, not for success
done
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
