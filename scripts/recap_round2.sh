#!/bin/bash
# RECAP round 2 for one round-1 policy run (item 4, 2026-11-07; paper: pi*0.6 Alg. 1 / Sec. V-D) [built, not run].
# Prints the sbatch chain; SUBMIT=1 submits it. Run it only for a round-1 policy that at least matches base: round 2
# trains on that policy's own rollouts, so a policy worse than base only adds worse data.
#
# Usage: scripts/recap_round2.sh <round-1 policy run dir>            (dry run: prints the commands)
#        SUBMIT=1 scripts/recap_round2.sh <run>                      (submits; prints the job ids)
#
# Chain (one lane, each step after the previous):
#   1. collect with the policy (tag positive [RECAP: rollouts of the current policy, conditioned on improvement]):
#      the 30 fixed train states x 15 repeats (round 1's design) at env seed 11, then 450 fresh layouts at seed 12
#      [ours: new env seeds so the episodes are not round 1's], raw files saved
#   2. table = round 1's table + both runs (scripts/libero_round2_table.py), frame cache packed
#   3. 5-fold OOF ResNet V on the new table (vl.libero_resnet --oof-fold k, round 1's recipe)
#   4. value-binned 40% labels (scripts/libero_td_labels.py oof)
#   5. policy fine-tune FROM BASE (pi05_libero) on all rounds [RECAP: "from the pre-trained checkpoint ... avoiding
#      drift"], with round 1's arguments + --table / --labels replaced; same route (prompt or AdaRMS)
#   6. tag check, 7. eval (matched noise seed 1, the 330 val trials, tag positive)
set -euo pipefail
RUN=${1:?round-1 policy run dir}
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
BASE_TABLE=${BASE_TABLE:-$ROOT/tables/moka_fixed_fresh}
TR="-A iris -p iris --exclude=iris[1-4,8-9],iris-hgx-[1-2],iris-hp-z8"
cd $VL
TRAIN_STATES=$(.venv/bin/python -c "import yaml; print(*yaml.safe_load(open('configs/libero_moka.yaml'))['split']['train'])")
ADARMS=$([ -f "$RUN/adarms.json" ] && echo 1 || echo "")
ARGS1=$(.venv/bin/python - "$RUN" <<'EOF'
import json, sys
a = json.load(open(f"{sys.argv[1]}/results.json"))["args"]
assert a["arm"] == "advantage" and a.get("labels"), "round 2 is for advantage-tagged runs (RECAP)"
flags = {"arm": "--arm", "tag_dropout": "--tag-dropout", "lora_lr": "--lora-lr", "lr": "--lr", "steps": "--steps",
         "discrete_weight": "--discrete-weight", "discrete_dims": "--discrete-dims", "seed": "--seed"}
out = ["--train"] + [f"{f} {a[k]}" for k, f in flags.items() if a.get(k) is not None]
out += [f for k, f in {"demos": "--demos", "freeze_siglip": "--freeze-siglip", "ki_stop_grad": "--ki-stop-grad",
                       "fast_tokens": "--fast-tokens", "subtask_prompt": "--subtask-prompt"}.items() if a.get(k)]
print(" ".join(out + ["--save-every 1000"]))
EOF
)
SUBTASK=$(.venv/bin/python -c "import json; print(1 if json.load(open('$RUN/results.json'))['args'].get('subtask_prompt') else '')")
NAME=r2_$(basename $RUN | sed 's/^policy_//')
N=0; prev=""
sub() {  # sub <var> <sbatch args...>: print, or submit after the previous step
    local var=$1; shift
    N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${prev:+-d afterok:$prev} "$@"); prev=$j; eval "$var=$j"
        echo "step $N: $j"
    else
        eval "$var=JOB$N"; echo "step $N (after ${prev:-none}): sbatch $*"; prev=JOB$N
    fi
}
if [ -n "$ADARMS" ]; then SERVE=(scripts/pi05_tag_adarms.sbatch); MODE=",MODE=eval"; else SERVE=(scripts/libero_eval.sbatch); MODE=",SUITE=libero_10,TASKS=8"; fi
sub C1 --export=ALL$MODE,FT_RUN=$RUN,FT_TAG=positive,FT_SUBTASK=$SUBTASK,REPEATS=15,SEED=11,NAME=${NAME}_collect_fixed,CLIENT_ARGS="--args.init-states $TRAIN_STATES --args.save-raw" "${SERVE[@]}"
sub C2 --export=ALL$MODE,FT_RUN=$RUN,FT_TAG=positive,FT_SUBTASK=$SUBTASK,REPEATS=1,SEED=12,NAME=${NAME}_collect_fresh,CLIENT_ARGS="--args.fresh-layouts 450 --args.save-raw" "${SERVE[@]}"
TABLE=$ROOT/tables/moka_${NAME}
OUTC() { if [ -n "$ADARMS" ]; then echo "$ROOT/runs/${NAME}_collect_$1_libero_10_$2"; else echo "$ROOT/runs/${NAME}_collect_$1_libero_10_$2"; fi; }
sub TB --export=ALL,CMD=".venv/bin/python scripts/libero_round2_table.py --base $BASE_TABLE --round 2 --out $TABLE --runs $(OUTC fixed $C1) $(OUTC fresh $C2) && .venv/bin/python -m vl.libero_resnet --every 4 --dataset $TABLE --pack-only --out $ROOT/runs/pack_${NAME}" scripts/vl_python.sbatch
for k in 0 1 2 3 4; do
    sub F$k --export=ALL,TAG=${NAME}oof_f$k,SEED=0,EVERY=4,DATASET=$TABLE,EXTRA="--oof-fold $k" scripts/libero_resnet.sbatch
done
sub LB --export=ALL,CMD=".venv/bin/python scripts/libero_td_labels.py oof --table $TABLE --pattern 'resnet18_${NAME}oof_f{k}_s0_*' --name r2_vb40" scripts/vl_python.sbatch
LAB=$TABLE/labels/r2_vb40/labels.parquet
if [ -n "$ADARMS" ]; then
    sub T --export=ALL,MODE=train,NAME=$NAME,ARGS="$ARGS1 --labels $LAB --table $TABLE" scripts/pi05_tag_adarms.sbatch
    P=$ROOT/runs/policy_${NAME}_$T
    sub K --export=ALL,MODE=tagusage,ARGS="--run $P --labels $LAB --steps 1000 3000 final --arms tagged" scripts/pi05_tag_adarms.sbatch
    sub E --export=ALL,MODE=eval,FT_RUN=$P,FT_TAG=positive,FT_SUBTASK=$SUBTASK,MATCHED_NOISE=1,NOISE_SEED=1,NAME=eval_${NAME}_pos_noise1 scripts/pi05_tag_adarms.sbatch
else
    sub T $TR --export=ALL,NAME=$NAME,ARGS="$ARGS1 --labels $LAB --table $TABLE" scripts/pi05_policy_finetune.sbatch
    P=$ROOT/runs/policy_${NAME}_$T
    sub K $TR --export=ALL,ARGS="--run $P --labels $LAB --steps 1000 3000 final --arms tagged" scripts/libero_tag_usage.sbatch
    sub E --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_SUBTASK=$SUBTASK,FT_RUN=$P,FT_TAG=positive,NAME=eval_${NAME}_pos_noise1,CLIENT_ARGS="--args.init-states 9 12 13 17 25 32 40 42 44 48" scripts/libero_eval.sbatch
fi
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
