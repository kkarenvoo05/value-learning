#!/bin/bash
# Per-init-state evals on the 30 TRAIN states for base pi0.5 and AG rounds 0-7 [2026-11-16; mentor: "this analysis
# should be done for many more initial states than just 10"; Karen: "set this up"].
# The 10 val states already have 33 trials per round (the 330-trial evals); this adds the 30 train states, where
# every round collects data, so together 40 of the 50 states. The 10 test states stay untouched.
# Per checkpoint: 30 train states x REPEATS (default 10 = 300 episodes), env seed 7 and matched noise seed 1 as
# the val evals, at each round's matched shift 0.99 (w read from its own contrast check). 0 / 10 -> 95% upper bound
# 26%: enough to flag stuck states, too coarse to rank methods per state.
# Two lanes chained with afterany (GPU cap <= 6 incl. interactive, Karen 2026-11-07). Prints the chain; SUBMIT=1
# submits. Score: scripts/libero_per_state.py (reports/2026-11-16-per-state).
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
REPEATS=${REPEATS:-10} SHIFT=0.99
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
cd $VL
TRAIN_STATES=$(.venv/bin/python -c "import yaml; print(*yaml.safe_load(open('configs/libero_moka.yaml'))['split']['train'])")
[ "$(echo $TRAIN_STATES | wc -w)" -eq 30 ] || { echo "expected 30 train states"; exit 1; }
CLIENT="--args.init-states $TRAIN_STATES --args.split-name train"

declare -A RUN CHK
RUN[r0]=$ROOT/runs/policy_ag_residual_vb40_s0_18438185 CHK[r0]=$ROOT/runs/contrast_check_ag_residual_vb40_s0
for r in 1 2 3 4 5 6 7; do
    d=$(ls -d $ROOT/runs/policy_agr_r${r}_s0_[0-9]* | head -1)
    RUN[r$r]=$d CHK[r$r]=$ROOT/runs/contrast_check_agr_r${r}_s0
done
for k in "${!RUN[@]}"; do
    [ -e "${RUN[$k]}/trainable_final.msgpack" ] && [ -e "${CHK[$k]}/results.json" ] || { echo "missing $k: ${RUN[$k]}"; exit 1; }
done

sub() {  # sub <lane var> <name> <sbatch args...>
    local lane=$1 name=$2; shift 2
    local dep=${!lane:-}
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+-d afterany:$dep} "$@"); printf -v $lane %s "$j"; echo "$name: $j${dep:+ (after $dep)}"
    else
        printf -v $lane %s "J_$name"; echo "$name (after ${dep:-none}): sbatch $*"
    fi
}
base() {
    sub $1 base $EX --mem=40G --time=4:00:00 --job-name=ps_eval \
        --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=$REPEATS,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,NAME=ps_base_train_noise1,CLIENT_ARGS="$CLIENT" \
        scripts/libero_eval.sbatch
}
round() {
    sub $1 $2 $EX --mem=40G --time=4:00:00 --job-name=ps_eval \
        --export=ALL,MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=${RUN[$2]},FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=${CHK[$2]}/results.json,REPEATS=$REPEATS,SEED=7,NAME=ps_$2_train_shift099_noise1,CLIENT_ARGS="$CLIENT" \
        scripts/pi05_tag_adarms.sbatch
}
A=${DEP_A:-} B=${DEP_B:-}
# endpoints first (base, r0, r7), then the middle rounds
base A; round B r0; round A r7; round B r5; round A r3; round B r6; round A r1; round B r2; round A r4
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
