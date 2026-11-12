#!/bin/bash
# Iterated filtered BC, round 2 (Lars 2026-11-10: "if every state succeeds sometimes, more rollouts + keep successes
# should converge toward ~100%"). Built 2026-11-10. Prints the sbatch chain; SUBMIT=1 submits it.
#
# Round 1 = R1 arm A (frozen SigLIP, original lr, 516 round-1 train successes, no demos) = 0.555 on the 330 val trials
# (eval 18376075, matched noise seed 1). Round 2:
#   1. collect with the round-1 policy (no tag): round 1's design [ours: same as round 1 / recap_round2.sh] =
#      30 fixed train states x 15 repeats (2 jobs) + 450 fresh layouts (2 jobs), new env seeds 41-44, raw saved
#   2. table = round-1 table + the 4 runs (scripts/libero_round2_table.py, round 2)
#   3. fine-tune FROM BASE on every success of rounds 1 + 2, round 1's exact arguments + --table
#      [RECAP Alg. 1: retrain from the pre-trained checkpoint on all data so far; ours for filtered BC], seeds 0 and 1
#   3b. round-1 recipe seed 1 on the round-1 table (so round 1 vs round 2 is 2 seeds vs 2 seeds)
#   4. per lane: eval @final, then @2000 (val states x 33, env seed 7, matched noise seed 1; same as 0.555)
# GPU cap (Karen 2026-11-07): <= 6 incl. the interactive session. Peak here: 4 collectors, then 3 lanes.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
R1=${R1:-$ROOT/runs/policy_armA_frozensiglip_s0_18310771}
BASE_TABLE=$ROOT/tables/moka_fixed_fresh
TABLE=$ROOT/tables/moka_fbc_r2
TR="-A iris -p iris --exclude=iris[1-4,8-9],iris-hgx-[1-2],iris-hp-z8"
EX="--exclude=iris[1-4,8-9],iris-hgx-[1-2],iris-hp-z8"   # iris8: nvidia-smi hangs in D state (2026-11-10 21:10)
VAL="--args.init-states 9 12 13 17 25 32 40 42 44 48"
R1ARGS="--train --arm imitation --freeze-siglip --save-every 1000"   # + defaults = R1's results.json args
cd $VL
read -r -a TRAIN <<< "$(.venv/bin/python -c "import yaml; print(*yaml.safe_load(open('configs/libero_moka.yaml'))['split']['train'])")"
[ ${#TRAIN[@]} -eq 30 ] || { echo "expected 30 train states"; exit 1; }
HALF_A="${TRAIN[*]:0:15}"; HALF_B="${TRAIN[*]:15:15}"

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
COL="SUITE=libero_10,TASKS=8,FT_RUN=$R1,FT_TAG=none,FT_STEP=final"
sub CA - $EX --export=ALL,$COL,REPEATS=15,SEED=41,NAME=fbc_r2_collect_fixed_a,CLIENT_ARGS="--args.init-states $HALF_A --args.save-raw --args.split-name train" scripts/libero_eval.sbatch
sub CB - $EX --export=ALL,$COL,REPEATS=15,SEED=42,NAME=fbc_r2_collect_fixed_b,CLIENT_ARGS="--args.init-states $HALF_B --args.save-raw --args.split-name train" scripts/libero_eval.sbatch
sub CC - $EX --export=ALL,$COL,REPEATS=1,SEED=43,NAME=fbc_r2_collect_fresh_a,CLIENT_ARGS="--args.fresh-layouts 225 --args.save-raw --args.split-name train" scripts/libero_eval.sbatch
sub CD - $EX --export=ALL,$COL,REPEATS=1,SEED=44,NAME=fbc_r2_collect_fresh_b,CLIENT_ARGS="--args.fresh-layouts 225 --args.save-raw --args.split-name train" scripts/libero_eval.sbatch
RUNS=""; for n in fixed_a:$CA fixed_b:$CB fresh_a:$CC fresh_b:$CD; do RUNS="$RUNS $ROOT/runs/fbc_r2_collect_${n%%:*}_libero_10_${n##*:}"; done
sub TB afterok:$CA:$CB:$CC:$CD --export=ALL,CMD=".venv/bin/python scripts/libero_round2_table.py --base $BASE_TABLE --round 2 --out $TABLE --runs$RUNS" scripts/vl_python.sbatch

lane() {  # lane <name> <train args>: train after the table, then eval @final, then @2000
    local name=$1 args=$2
    sub T afterok:$TB $TR --export=ALL,NAME=$name,ARGS="$args" scripts/pi05_policy_finetune.sbatch
    local run=$ROOT/runs/policy_${name}_$T
    local ev="SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_RUN=$run,FT_TAG=none"
    sub EF afterok:$T $EX --export=ALL,$ev,FT_STEP=final,NAME=eval_${name}_atfinal_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch
    sub E2 afterany:$EF $EX --export=ALL,$ev,FT_STEP=2000,NAME=eval_${name}_at2000_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch
}
lane fbc_r2_s0 "$R1ARGS --seed 0 --table $TABLE"
lane fbc_r2_s1 "$R1ARGS --seed 1 --table $TABLE"
lane fbc_r1_s1 "$R1ARGS --seed 1"
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
