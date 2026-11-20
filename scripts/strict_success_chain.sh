#!/bin/bash
# Strict-success relabel of round 7 (Karen 2026-11-15: "set up the ... strict-success relabel"). Prints the sbatch
# chain; SUBMIT=1 submits it. First (CPU): .venv/bin/python scripts/libero_strict_success.py outcomes && ... check
#
# Question: does counting unstable first-pot placements as failures (scripts/libero_strict_success.py docstring:
# moved > 2 cm in the 50 steps after release, or off the cook region >= 5 steps; 124 of 1053 train successes) make
# residual Advantage Guidance fix the stove-edge failure? Only the training outcome changes; the eval keeps the
# simulator's success.
#   1. value job (one TITAN, ~1.5 h): the 5 OOF ResNet refits of round 0 on the strict targets (judge_noise_chain.sh's
#      recipe, run names resnet18_strict_oof_f{k}_s0_<job>), then libero_strict_success.py labels --check-v (first
#      rebuilds round 7's chained V with round 0's runs, then the strict V and labels for every round of moka_agr_r7)
#   2. residual head trained from zero init on moka_agr_r7 = round 7's exact arguments (scripts/ag_rounds.sh), labels
#      strict_vb40_r7
#   3. contrast check (w 1 2 4 8 16) -> 4. eval at action shift 0.99 x noise, 330 val trials, matched noise seed 1
#   Compare paired with round 7 (train 18524601, eval 18524603): same data, same recipe, only the outcomes differ.
#   Seeds of one recipe differ by ~8 points, so a gain under ~0.05 needs a second seed.
# ROUND0=1 adds a second lane after step 4 starts: round 0's residual recipe on moka_fixed_fresh with the round-0-only
# strict labels (labels/strict_vb40), paired with round 0 seed 0 (18438187, 0.755) and the noisy-judge arm sym25.
# Job names ss_* so no watcher or feeder reacts to them. Queue: waits until <= CAP jobs, retries on the QOS limit.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
TABLE=$ROOT/tables/moka_agr_r7
TABLE0=$ROOT/tables/moka_fixed_fresh
TG=$TABLE0/strict/targets.parquet
LAB=$TABLE/labels/strict_vb40_r7
LAB0=$TABLE0/labels/strict_vb40
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
EV="MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_SHIFT_TARGET=0.99"
CAP=${CAP:-20} ROUND0=${ROUND0:-0}
cd $VL
for f in "$TG" "$TABLE/strict/outcomes.csv" "$TABLE/episodes.csv"; do [ -e "$f" ] || { echo "missing $f (run libero_strict_success.py outcomes)"; exit 1; }; done
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
lane() {  # lane <name> <dep> <train args>: residual train -> contrast check -> eval
    local name=$1 dep=$2 args=$3
    sub T "$dep" -A iris -p iris $EX --mem=56G --time=7:30:00 --job-name=ss_train \
        --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$name,ARGS="$args" scripts/pi05_tag_adarms.sbatch
    local run=$ROOT/runs/policy_${name}_$T chk=$ROOT/runs/contrast_check_$name
    sub K afterok:$T $EX --mem=40G --time=1:00:00 --job-name=ss_check \
        --export=ALL,MODE=contrastcheck,OUT=$chk,ARGS="--run $run --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    sub E afterok:$K $EX --mem=40G --time=3:00:00 --job-name=ss_eval \
        --export=ALL,$EV,FT_RUN=$run,FT_SHIFT_FROM=$chk/results.json,NAME=eval_${name}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
}

if [ -n "${V_JOB:-}" ]; then VJ=$V_JOB; echo "# value job $VJ (reused)"; else
    CMD="for k in 0 1 2 3 4; do .venv/bin/python -m vl.libero_resnet --every 4 --dataset $TABLE0 --seed 0 --oof-fold \$k --targets $TG --out $ROOT/runs/resnet18_strict_oof_f\${k}_s0_\$SLURM_JOB_ID; done && .venv/bin/python scripts/libero_strict_success.py labels --check-v"
    # iris2's TITAN: "CUDA unknown error" twice (round 7 labels 18524600, first ss_value 18528551)
    sub VJ - --exclude=iris2 --time=8:00:00 --job-name=ss_value --export=ALL,CMD="$CMD" scripts/vl_python.sbatch
fi
# round 7's arguments (scripts/ag_rounds.sh step 3) with the strict labels
lane agr_r7_strict_s0 afterok:$VJ "--train --arm advantage --labels $LAB/labels.parquet --table $TABLE --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
if [ "$ROUND0" = 1 ]; then  # round 0's arguments (judge_noise_chain.sh) with the round-0-only strict labels
    lane ag_residual_strict_s0 "afterok:$VJ,afterany:$E" "--train --arm advantage --labels $LAB0/labels.parquet --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
fi
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
