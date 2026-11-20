#!/bin/bash
# Filtered BC round 2 with a demo anchor (Karen 2026-11-12: "set up the anchored filtered BC rerun"). Prints the sbatch
# chain; SUBMIT=1 submits it.
# Why: round 2 (fbc_round2.sh: successes of rounds 1 + 2, no demos) fell to 44-51% vs round 1 55.5%. zima's round loop
# (zima docs/learning/notes/setup.md, bc-sources.md) co-trains every fine-tune with the base policy's own data
# (DROID_SHARE=0.5) and raises that share if held-out success drops. Our step-1 grid already showed the anchor works
# here: an untagged all-rollouts fine-tune went 0.391 (demos ~2% of frames, 18376076) -> 0.515 at demo share 0.5
# (grid_d50_orig final; reports/2026-11-09-step1-followup).
# Run: fbc_r2_s0's exact arguments (round-1 recipe: imitation arm, frozen SigLIP, original lr, 5000 steps, seed 0,
# table moka_fbc_r2) + --demos --demo-share 0.5 [ours: the grid's passing share at the original lr; the 29 moka demos
# in the openpi LeRobot copy, oversampled to ~50% of frames]. Evals @final and @2000 as round 2 (330 val trials,
# matched noise seed 1), so it pairs with fbc_r2_s0 (44.2% / 48.8%) trial for trial.
# GPU cap: SUBMIT=1 waits until scripts/ag_residual_controls.sh has submitted the shuffled residual's eval and starts
# after it (afterany), i.e. beside the seed-1 residual: +1 GPU, peak 6 incl. interactive. DEP=<dep> overrides.
# Submissions retry every 5 min at the 25-job submit limit.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
TABLE=$ROOT/tables/moka_fbc_r2
CLOG=$ROOT/logs/ag_residual_controls.log
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"   # as pi05_tag_adarms.sbatch: 48 GB, iris9-10 bad
VAL="--args.init-states 9 12 13 17 25 32 40 42 44 48"
NAME=fbc_r2_anchor50_s0
ARGS="--train --arm imitation --freeze-siglip --save-every 1000 --seed 0 --table $TABLE --demos --demo-share 0.5"
cd $VL
[ -e $TABLE/episodes.csv ] || { echo "missing $TABLE"; exit 1; }
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
if [ -z "${DEP:-}" ]; then
    if [ -n "${SUBMIT:-}" ]; then
        until E=$(sed -n 's/.* step 3 E: \([0-9]*\)$/\1/p' $CLOG 2>/dev/null) && [ -n "$E" ]; do sleep 300; done
        DEP=afterany:$E; echo "$(date -Is) after shuffled-residual eval $E"
    else DEP="afterany:<shuffled-residual eval>"; fi
fi
sub T "$DEP" -A iris -p iris $EX --time=10:00:00 --job-name=fbc_anchor --export=ALL,NAME=$NAME,ARGS="$ARGS" scripts/pi05_policy_finetune.sbatch
RUN=$ROOT/runs/policy_${NAME}_$T
EV="SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_RUN=$RUN,FT_TAG=none"
sub EF afterok:$T $EX --time=3:00:00 --job-name=fbc_anchor_eval --export=ALL,$EV,FT_STEP=final,NAME=eval_${NAME}_atfinal_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch
sub E2 afterany:$EF $EX --time=3:00:00 --job-name=fbc_anchor_eval --export=ALL,$EV,FT_STEP=2000,NAME=eval_${NAME}_at2000_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
