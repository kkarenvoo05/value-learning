#!/bin/bash
# Noisy-judge robustness (Karen 2026-11-14: "2 now: offline, cheap, and it starts while the GPUs are busy").
# Prints the sbatch chain; SUBMIT=1 submits it. Flips + targets first: .venv/bin/python scripts/libero_judge_noise.py targets
#
# A. Residual Advantage Guidance trained on a noisy judge's outcomes (scripts/libero_judge_noise.py docstring), per arm:
#    1. value refit: the 5 OOF ResNet folds of round 0 (vl.libero_resnet --every 4 --oof-fold k, 8000 steps, seed 0)
#       with --targets = the judged-outcome targets; then the vb40 labels with the judged outcome (one TITAN job, ~1 h)
#    2. round 0's residual recipe exactly (as scripts/ag_residual_controls.sh), labels judge_<arm>_vb40, seed 0
#    3. contrast check (w 1 2 4 8 16) -> eval at action shift 0.99 x noise, 330 val trials, matched noise seed 1
#    Compare paired with round 0 seed 0 (18438187, 0.755); seed spread is ~8 points (seed 1: 0.676).
#    Value jobs run side by side (TITANs); the residual steps form ONE sequential lane (+1 GPU on the 48 GB pool).
# B. DSRL with the same judge (sym25: failures and successes each flipped with p 0.25, balanced accuracy ~0.75,
#    FailBench's best VLM detector 0.77): a fresh agent, the s0 recipe (seed 0, env seed 51, 30 train states) but
#    600 episodes (30 x 20, the small-budget range), checkpoints every 100, resumable; then held-out evals of
#    agent_300 and agent_600 (as 18445300: 330 val trials, env seed 7, no matched noise). Compare with s0's
#    agent_300 / agent_600 evals (small-budget curve). dsrl_log.jsonl records success (truth) and judged.
# Job names jn_* so neither the rounds feeder nor other watchers react to them.
# Queue: iris QOS allows 25 submitted jobs per user; with SUBMIT=1 each sbatch waits until this user has <= CAP
# (default 20) jobs queued or running, leaving room for other sessions, and retries on the QOS limit. Run it under
# nohup. If it dies, resume from its log: DSRL=0 (if B was submitted), V_<arm>=<value job id> for submitted value
# jobs, ARMS=<arms whose residual lane is not yet submitted>, DEP_START=afterany:<last submitted lane eval>.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
TABLE=$ROOT/tables/moka_fixed_fresh
LABS=$TABLE/labels
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
EV="MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1"
VAL="--args.init-states 9 12 13 17 25 32 40 42 44 48"
ARMS=${ARMS:-"sym25 fp25 sym10"}
CAP=${CAP:-20}
DSRL=${DSRL:-1}
cd $VL
read -r -a TRAIN <<< "$(.venv/bin/python -c "import yaml; print(*yaml.safe_load(open('configs/libero_moka.yaml'))['split']['train'])")"
[ ${#TRAIN[@]} -eq 30 ] || { echo "expected 30 train states"; exit 1; }
for arm in $ARMS; do [ -e "$TABLE/judge_noise/$arm/targets.parquet" ] || { echo "missing targets for $arm"; exit 1; }; done
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

# B first in the queue order: it is one long job, and its evals only need its checkpoints.
if [ "$DSRL" = 1 ]; then
    echo "# DSRL, noisy judge sym25, 600 episodes"
    D_ARGS="--ckpt-every 100 --state-every 50 --judge-fp 0.25 --judge-fn 0.25 --judge-seed 0"
    sub D - -A iris -p iris --exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8 --time=1-12:00:00 --mem=64G \
        --job-name=jn_dsrl_train --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=20,SEED=51,DSRL_MODE=train,DSRL_SEED=0,DSRL_ARGS="$D_ARGS",NAME=dsrl_judge_sym25_s0,CLIENT_ARGS="--args.init-states ${TRAIN[*]} --args.split-name train" \
        scripts/libero_eval.sbatch
    DRUN=$ROOT/runs/dsrl_judge_sym25_s0_libero_10_$D
    for n in 300 600; do
        sub DE afterok:$D $EX --time=4:00:00 --mem=40G --job-name=jn_dsrl_eval \
            --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,DSRL_MODE=eval,DSRL_AGENT=$DRUN/agent_$n.pkl,NAME=eval_dsrl_judge_sym25_s0_agent${n},CLIENT_ARGS="$VAL" \
            scripts/libero_eval.sbatch
    done
fi

declare -A V
for arm in $ARMS; do
    TG=$TABLE/judge_noise/$arm/targets.parquet
    CMD="for k in 0 1 2 3 4; do .venv/bin/python -m vl.libero_resnet --every 4 --dataset $TABLE --seed 0 --oof-fold \$k --targets $TG --out $ROOT/runs/resnet18_judge_${arm}_oof_f\${k}_s0_\$SLURM_JOB_ID; done && .venv/bin/python scripts/libero_judge_noise.py labels --arm $arm"
    reuse=V_$arm   # resume: V_sym25=<job id> reuses an already submitted value job
    if [ -n "${!reuse:-}" ]; then V[$arm]=${!reuse}; echo "# $arm: value job ${!reuse} (reused)"; continue; fi
    sub VJ - --time=8:00:00 --job-name=jn_value --export=ALL,CMD="$CMD" scripts/vl_python.sbatch
    V[$arm]=$VJ
done

DEP=${DEP_START:-}
for arm in $ARMS; do
    NAME=ag_residual_judge_${arm}_s0
    echo "# $NAME: labels judge_${arm}_vb40 (after value job ${V[$arm]})"
    ARGS="--train --arm advantage --labels $LABS/judge_${arm}_vb40/labels.parquet --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed 0"
    sub T "afterok:${V[$arm]}${DEP:+,$DEP}" -A iris -p iris $EX --time=7:30:00 --job-name=jn_train \
        --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch
    RUN=$ROOT/runs/policy_${NAME}_$T
    CHK=$ROOT/runs/contrast_check_${NAME}
    sub K afterok:$T $EX --time=1:00:00 --job-name=jn_check \
        --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    sub E afterok:$K $EX --time=3:00:00 --job-name=jn_eval \
        --export=ALL,$EV,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift099_noise1 \
        scripts/pi05_tag_adarms.sbatch
    DEP=afterany:$E   # the lane's next arm waits for the GPU, not for success
done
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
