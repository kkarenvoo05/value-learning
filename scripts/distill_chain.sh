#!/bin/bash
# Distil the RL gains back into pi0.5 (PLD-style; Karen 2026-11-14: "set up the distillation experiment with the
# chunk target fix"). Prints the sbatch chain; SUBMIT=1 submits it.
#
# Question: can a plain fine-tune of pi0.5 (no residual head, no DSRL actor at inference) keep the teacher's gain?
# Every fine-tune on own rollouts so far landed at 0.40-0.51 (base 0.524), all with "executed" targets that join two
# separately sampled chunks. PLD (arXiv 2511.00091, Table 3, LIBERO-90 pi0 SFT): PLD data 0.871, base-policy rollouts
# 0.488, human 0.815; PLD never compares the distilled model with base + residual, so that comparison is ours.
#   1. collect 300 episodes each, raw + sampled chunks saved (scripts/libero_rollouts.py Changes 8), 30 train states
#      x 10, new env seeds 61-63, in parallel (24 GB cards):
#        hybrid  DSRL eval server (s0 agent_final_ep1800, 0.861 held out), base pi0.5 for a prefix k ~ U[0, 0.6 x 520)
#                then the actor (PLD Sec. 3.2; its probing ablation peaks at alpha 0.6)
#        dsrl    the same server, actor from step 0 (--args.dsrl-base-prefix 0: same bookkeeping)
#        base    plain pi0.5 (openpi server): the chunk-fix-only control (filtered BC round 1 with executed targets:
#                0.555, 18376075)
#   2. table per collection (scripts/libero_distill_table.py: round-0 table + the run as round 2; checks the chunks)
#   3. fine-tune FROM BASE on the collection's successes, filtered BC round 1's recipe (imitation arm, frozen SigLIP,
#      5000 steps, batch 32, LoRA) + --chunk-targets sampled (replan frames only, each with its sampled 10-action
#      chunk) + --save-every 500; one sequential lane: hybrid, dsrl, base
#   4. per run: eval @final, then @2000 (330 val trials, env seed 7, matched noise seed 1; pairs with every arm)
# Read: distilled hybrid >= ~0.75 -> the gain survives distillation (zima's loop can ship a new generation);
# ~0.5 -> pi0.5 does not absorb it by fine-tuning; base control ~0.55 -> the chunk fix alone changes little.
# Sample count: ~250 successes x ~80 replan frames ~ 20k samples, ~8 epochs at 5000 x 32: watch @2000 vs @final.
# Job names ds_* . Queue: as scripts/judge_noise_chain.sh (waits until <= CAP jobs, retries on the QOS limit).
# Resume from its log: TB_<kind>=<table job id> for submitted collections, PREV_TRAIN / TRAIN_KINDS for the lane.
#
# Kind "ag" (2026-11-15, Karen: "set up the AG distillation"): the same 300-episode collection and fine-tune with the
# residual Advantage Guidance policy as the teacher (AG_RUN, default round 5 = 0.876 held out; served at action shift
# 0.99 x noise from AG_CHK's contrast check, as its eval), env seed 64, noise seed 164. Question for zima: does a
# batch-only, judge-noise-robust teacher distil as well as DSRL (0.882)? Run: KINDS=ag scripts/distill_chain.sh.
# AG round collections have only 100 episodes (round 6 = the round-5 policy: 85 successes), so a fresh 300 matches
# the DSRL collection (255 successes).
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
TEACHER=${TEACHER:-$ROOT/runs/dsrl_s0_libero_10_18434524/agent_final_ep1800.pkl}
AG_RUN=${AG_RUN:-$ROOT/runs/policy_agr_r5_s0_18484605}
AG_CHK=${AG_CHK:-$ROOT/runs/contrast_check_agr_r5_s0}
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
VAL="--args.init-states 9 12 13 17 25 32 40 42 44 48"
KINDS=${KINDS:-"hybrid dsrl base"}
CAP=${CAP:-20}
cd $VL
case " $KINDS " in *" hybrid "*|*" dsrl "*) [ -e "$TEACHER" ] || { echo "missing $TEACHER"; exit 1; } ;; esac
case " $KINDS " in *" ag "*) for f in "$AG_RUN/trainable_final.msgpack" "$AG_CHK/results.json"; do
    [ -e "$f" ] || { echo "missing $f"; exit 1; }; done ;; esac
read -r -a TRAIN <<< "$(.venv/bin/python -c "import yaml; print(*yaml.safe_load(open('configs/libero_moka.yaml'))['split']['train'])")"
[ ${#TRAIN[@]} -eq 30 ] || { echo "expected 30 train states"; exit 1; }
COLARGS="--args.init-states ${TRAIN[*]} --args.save-raw --args.split-name train"
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

declare -A TB TBL
for kind in $KINDS; do
    case $kind in
        hybrid) seed=61; ex="DSRL_MODE=eval,DSRL_AGENT=$TEACHER"; extra="--args.dsrl-base-prefix 0.6 --args.dsrl-prefix-seed 0" ;;
        dsrl)   seed=62; ex="DSRL_MODE=eval,DSRL_AGENT=$TEACHER"; extra="--args.dsrl-base-prefix 0" ;;
        base)   seed=63; ex="NOOP=1"; extra="" ;;
        ag)     seed=64; extra=""   # residual AG server (pi05_tag_adarms.sbatch eval mode, as the round collections)
                ex="MODE=eval,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$AG_RUN,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$AG_CHK/results.json,NOISE_SEED=164" ;;
        *) echo "unknown kind $kind"; exit 1 ;;
    esac
    NAME=distill_collect_$kind
    TBL[$kind]=$ROOT/tables/moka_distill_$kind
    reuse=TB_$kind   # resume: TB_hybrid=<table job id> reuses an already submitted collection + table
    if [ -n "${!reuse:-}" ]; then TB[$kind]=${!reuse}; echo "# $kind: table job ${!reuse} (reused)"; continue; fi
    sb=scripts/libero_eval.sbatch; [ "$kind" = ag ] && sb=scripts/pi05_tag_adarms.sbatch
    sub C - $EX --time=8:00:00 --mem=64G --job-name=ds_collect \
        --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=10,SEED=$seed,$ex,NAME=$NAME,CLIENT_ARGS="$COLARGS $extra" \
        $sb
    RUNC=$ROOT/runs/${NAME}_libero_10_$C
    sub T afterok:$C --time=1:00:00 --job-name=ds_table \
        --export=ALL,CMD=".venv/bin/python scripts/libero_distill_table.py --run $RUNC --out ${TBL[$kind]}" \
        scripts/vl_python.sbatch
    TB[$kind]=$T
done

PREV=${PREV_TRAIN:-}   # resume: PREV_TRAIN=<last submitted ds_train id>, TRAIN_KINDS=<kinds not yet trained>
# A train kind "<kind>_executed" (2026-11-14, Karen: "add the executed-target control fine-tune") trains the same
# collection with the old executed targets (every frame of the successes, actions[t:t+10]): with base_executed vs base
# the targets are the only difference.
for tkind in ${TRAIN_KINDS:-$KINDS}; do
    kind=${tkind%_executed}; targets=sampled; [ "$kind" != "$tkind" ] && targets=executed
    NAME=distill_${kind}_${targets}_s0
    T=${TBL[$kind]}
    ARGS="--train --arm imitation --freeze-siglip --save-every 500 --seed 0 --table $T --episodes $T/new_episodes.txt --chunk-targets $targets"
    sub F "afterok:${TB[$kind]}${PREV:+,afterany:$PREV}" -A iris -p iris $EX --time=10:00:00 --job-name=ds_train \
        --export=ALL,NAME=$NAME,ARGS="$ARGS" scripts/pi05_policy_finetune.sbatch
    RUN=$ROOT/runs/policy_${NAME}_$F
    EVA="SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_RUN=$RUN,FT_TAG=none"
    sub EF afterok:$F $EX --time=3:00:00 --mem=40G --job-name=ds_eval \
        --export=ALL,$EVA,FT_STEP=final,NAME=eval_${NAME}_atfinal_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch
    sub E2 afterany:$EF $EX --time=3:00:00 --mem=40G --job-name=ds_eval \
        --export=ALL,$EVA,FT_STEP=2000,NAME=eval_${NAME}_at2000_noise1,CLIENT_ARGS="$VAL" scripts/libero_eval.sbatch
    PREV=$F
done
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
