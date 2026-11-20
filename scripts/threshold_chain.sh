#!/bin/bash
# Sharper advantage labels on round 7: positive share 15 / 10 / 20% vs 40% (Karen 2026-11-15: "cancel strict and
# submit 10%, 15%, 20%"). Prints the sbatch chain; SUBMIT=1 submits it. First (CPU, seconds):
#   .venv/bin/python scripts/libero_threshold_labels.py   (labels agr_vb{10,15,20}_r7 from round 7's own V)
# Per share: residual head from zero init on moka_agr_r7 with round 7's exact arguments (scripts/ag_rounds.sh step 3),
# only --labels changed -> contrast check (w 1 2 4 8 16) -> eval at action shift 0.99 x noise (330 val trials, matched
# noise seed 1). Paired with round 7 (40%: train 18524601, eval 18524603, 0.830): same data, same V, same recipe.
# The matched shift keeps the guided action change the same size, so a sharper label cannot win by pushing harder.
# One train lane (each train waits for the previous one; checks / evals run beside the next train): peak +2 GPUs.
# Order: 15% (the main arm) first, then 10%, 20%. Seeds of one recipe differ by ~8 points: a trend across the three
# shares is more informative than any single arm. Job names th_*. Queue: waits until <= CAP jobs, retries on QOS.
# Resume: SHARES=<shares not yet submitted> DEP_START=afterany:<last submitted train>.
# Share "ad" (2026-11-15, Karen: "yes add the adaptive arm now"): labels agr_vbad_r7, each round's share = its
# failure rate clipped to [0.10, 0.40] (libero_threshold_labels.py). Submitted with SHARES=ad DEP_START=afterany:<20% train>.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
TABLE=$ROOT/tables/moka_agr_r7
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
EV="MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_SHIFT_TARGET=0.99"
SHARES=${SHARES:-"15 10 20"} CAP=${CAP:-20}
cd $VL
for s in $SHARES; do [ -e "$TABLE/labels/agr_vb${s}_r7/labels.parquet" ] || { echo "missing labels agr_vb${s}_r7 (run libero_threshold_labels.py)"; exit 1; }; done
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j out
        while [ "$(squeue -u "$USER" -h | wc -l)" -gt "$CAP" ]; do sleep 120; done
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
for s in $SHARES; do
    NAME=agr_r7_vb${s}_s0
    echo "# $NAME: labels agr_vb${s}_r7"
    ARGS="--train --arm advantage --labels $TABLE/labels/agr_vb${s}_r7/labels.parquet --table $TABLE --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
    sub T "$DEP" -A iris -p iris $EX --mem=56G --time=7:30:00 --job-name=th_train \
        --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch
    RUN=$ROOT/runs/policy_${NAME}_$T
    CHK=$ROOT/runs/contrast_check_$NAME
    sub K afterok:$T $EX --mem=40G --time=1:00:00 --job-name=th_check \
        --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    sub E afterok:$K $EX --mem=40G --time=3:00:00 --job-name=th_eval \
        --export=ALL,$EV,FT_RUN=$RUN,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
    DEP=afterany:$T
done
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
