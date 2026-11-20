#!/bin/bash
# Rollout rounds with the Advantage Guidance residual policy (mentor 2026-11-11: "run the current policy for more
# rounds, 100 episodes per round, 5-10 rounds; does success climb past 75%?"). Built 2026-11-11.
# Prints the sbatch chain; SUBMIT=1 submits it.
#
# Round 0 = the residual run policy_ag_residual_vb40_s0_18438185 (frozen pi05_libero + AdaLN head, trained on
# moka_fixed_fresh with resnet_valuebinned_p40 labels), served at action shift 0.99 x noise (w 9.26) = 75.4% on the
# 330 val trials. Each round r = 1..ROUNDS:
#   1. collect 100 episodes with policy r-1 at the same matched shift SHIFT (contrastive guidance; its w is re-read
#      from policy r-1's own contrast check): the 30 train states x 2 repeats (60) + 40 fresh layouts, two jobs in
#      parallel, ordinary sampling noise (no matched noise), new env / noise seeds per round [ours: 60/40 keeps round
#      0's fixed/fresh mix roughly; the 10 val states are never collected on, so the eval stays held out]
#   2. table r = table r-1 + both runs (libero_round2_table.py, table round r + 1 since round 0 is round 1 there),
#      then labels (libero_round_labels.py: V on the new episodes from the OOF fold model that never saw their scene,
#      no V refit; value-binned 40% cut within each round), one TITAN job
#   3. residual head trained FROM ZERO INIT on all rounds so far, round 0's exact arguments + --table / --labels
#      [RECAP Alg. 1: retrain from the pre-trained checkpoint on all data so far; the reference stays base pi0.5]
#   4. contrast check (shift / noise per w) -> 5. eval on the 330 val trials (matched noise seed 1, same as 75.4%)
#      at the w whose shift matches SHIFT. Round r+1's collection waits only for step 4, so it runs next to 5.
# Critical path ~4.5 h per round (collect ~0.5, table+labels ~0.2, train ~3.1, check ~0.2); eval ~2.3 h beside it.
# Peak GPUs: 2 collectors + 1 eval. GPU cap (Karen 2026-11-07): <= 6 incl. the interactive session.
#
# Caveat [ours]: v_ref = base pi0.5 is the behaviour policy of round 0 only; later rounds come from the guided policy.
# What the new data shares across tags is absorbed equally by h(+1) and h(-1) and cancels in w (h(+) - h(-)), but
# watch the contrast check's shift per w round over round.
#
# Env: ROUNDS (default 5; table rounds >= 10 get ids past the demo range, libero_round2_table.id_base),
#      SHIFT (default 0.99; 1.64 = the residual's w 16), EVAL_EVERY (default 1; the last round is always evaluated),
#      DEP_START (dependency of round 1's collectors, e.g. afterany:18452438), STEPS (default 5000, round 0's).
#      [2026-11-16, per-round budget; mentor: "test the same setup but with both 50 and 200 episodes per round"]
#      FIXED_REPEATS (default 2: x 30 train states) + FRESH (default 40 fresh layouts) = episodes per round;
#      TAG (default none) names a separate chain: runs / tables / labels / job names agr_<TAG>_..., so it never
#      touches the 100-per-round chain. Round 0 (policy, table, frozen OOF V) is shared by every budget.
#      e.g. 50: TAG=e50 FIXED_REPEATS=1 FRESH=20; 200: TAG=e200 FIXED_REPEATS=4 FRESH=80 COLLECT_TIME=4:00:00.
#      Collection seeds depend only on r, so round r of every budget uses the same env / noise seeds [ours].
#      [2026-11-16, Karen: "switch to 100% fresh layouts"] FIXED_REPEATS=0: no fixed-state collection; both collectors
#      draw fresh layouts (ceil(FRESH/2) with seed 200+2r, the rest with 201+2r).
#      CONT_EVAL=<starts.csv> (scripts/libero_eval_layouts.py): a second eval per round on the continuous-layout set
#      (--args.start-from, no matched noise; noise seed 1), job <PFX>_ceval, beside the 330-trial val eval.
#      [2026-11-18] STACK=1 (stacked AG, moving anchor): round r's reference = round r-1's whole guided sampler (base +
#      every earlier head at its own w, frozen; pi05_ag_residual.py --stack-from / residual.json head.stack). The new
#      head starts from zero and trains ONLY on round r's collection (the data the reference collected; no demos), so
#      each round adds a step on top of the last instead of re-aiming one step from base. SHIFT = the per-round step
#      relative to the previous policy; SHIFT0 (default 0.99) = the serving shift of P0, used by round 1's collectors
#      and in the first stack entry.
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
ROUNDS=${ROUNDS:-5} SHIFT=${SHIFT:-0.99} EVAL_EVERY=${EVAL_EVERY:-1} STEPS=${STEPS:-5000}
FIXED_REPEATS=${FIXED_REPEATS:-2} FRESH=${FRESH:-40} COLLECT_TIME=${COLLECT_TIME:-2:00:00}
[ "$FIXED_REPEATS" -ge 0 ] && [ "$FRESH" -ge 2 ] || { echo "need FIXED_REPEATS >= 0 and FRESH >= 2"; exit 1; }
[ -z "${CONT_EVAL:-}" ] || [ -e "$CONT_EVAL" ] || { echo "missing CONT_EVAL $CONT_EVAL"; exit 1; }
# [2026-11-16, Karen: "set up the 100 iql ag run"] LABELER=iql: tags from an IQL critic retrained from scratch each
# round on the chain's own episodes (scripts/libero_iql_round_labels.py, Q - V top 40% within each round, every round
# relabelled); label dirs <PFX>_iql40_r<r>. Default LABELER=vb40 (frozen round-0 OOF ResNet V; unchanged).
LABELER=${LABELER:-vb40}
# [2026-11-18, Karen: "set up the collection-push test"] COLLECT_SHIFT: the collectors' push (default SHIFT); training,
# check and evals are unchanged (evals stay at SHIFT). Not with STACK (there the collection push is the stacked step).
[ -z "${COLLECT_SHIFT:-}" ] || [ -z "${STACK:-}" ] || { echo "COLLECT_SHIFT and STACK do not combine"; exit 1; }
# [2026-11-18, Karen: "set up options 1 and 2"] collectors only, no spaces: COLLECT_EXPORT = extra --export entries
# (e.g. ",FT_SHIFT_SET=0.5:0.99:1.5:2.0,PUSH_SEED=6": a random push per episode), COLLECT_CLIENT = extra client args
# joined by "+" (e.g. "--args.explore-prob+0.5+--args.explore-scale+2.0+--args.explore-seed+6": start noise x 2.0
# on half the chunks). Training, check and evals are unchanged.
case "${COLLECT_EXPORT:-}${COLLECT_CLIENT:-}" in *" "*) echo "COLLECT_EXPORT / COLLECT_CLIENT: no spaces"; exit 1 ;; esac
CC=${COLLECT_CLIENT:+ ${COLLECT_CLIENT//+/ }}
# [2026-11-19] COLLECT_SEED_OFFSET = added to every collector env / noise seed (default 0): a new-seed rerun of a round
# (e.g. 1000 -> 1212 / 1213 / 1106 for round 6; never a multiple of 2 below 100, which would reuse another round's seeds)
SO=${COLLECT_SEED_OFFSET:-0}
# [2026-11-16, multi-task AG; mentor: "2 different but similar tasks in parallel, with the trajectories split across
# them"] TASK_A / TASK_B = suite:task_id of collector CF / CR (default libero_10:8 = moka for both, unchanged).
# Non-moka tasks get AG_SUITE / AG_TASK / AG_MAX_STEPS=520 on their jobs (scripts/pi05_tag_adarms.sbatch). With
# FIXED_REPEATS=0 the FRESH layouts are split over CF / CR as before, so TASK_B=libero_90:64 FRESH=100 = 50 + 50.
# TPFX = table name prefix (default moka). NO_VAL_EVAL=1 drops the moka 330-trial eval (task-64-only chains).
# [2026-11-19] CHECK_TIME (default 1:00:00): the contrast check's time limit (mt100cv r1's check hit 1 h).
# CONT_EVAL runs with TASK_A; CONT_EVAL_B (optional) adds a second ceval with TASK_B. CHECK_TABLE=1: the contrast
# check takes its val frames from the round's table (pi05_contrast_check.py --table) instead of moka_fixed_fresh.
TASK_A=${TASK_A:-libero_10:8} TASK_B=${TASK_B:-${TASK_A}} TPFX=${TPFX:-moka}
task_env() {  # extra --export entries for a non-moka task (none for moka, so default commands stay unchanged)
    [ "$1" = libero_10:8 ] || echo ",AG_SUITE=${1%%:*},AG_TASK=${1##*:},AG_MAX_STEPS=520"; }
ENV_A=$(task_env $TASK_A) ENV_B=$(task_env $TASK_B) SUITE_A=${TASK_A%%:*} SUITE_B=${TASK_B%%:*}
[ -z "${CONT_EVAL_B:-}" ] || [ -e "$CONT_EVAL_B" ] || { echo "missing CONT_EVAL_B $CONT_EVAL_B"; exit 1; }
# [2026-11-16, Karen: "set up chain-only V"] LABELER=cv: the same ResNet V as vb40, fit from scratch each round on the
# chain's own episodes only (scripts/libero_chainv_round_labels.py, 5 OOF folds, vb40 rule, every round relabelled);
# label dirs <PFX>_cv40_r<r>.
case $LABELER in vb40) LABTAG=vb40 LABEL_TIME=1:00:00 ;; iql) LABTAG=iql40 LABEL_TIME=2:00:00 ;;
                 cv) LABTAG=cv40 LABEL_TIME=${CV_LABEL_TIME:-8:00:00} ;;
                 *) echo "LABELER must be vb40, iql or cv"; exit 1 ;; esac
PFX=agr${TAG:+_$TAG}                       # agr (100 per round, unchanged) or agr_<TAG>
EVAL_JOB=${TAG:+${PFX}_eval}; EVAL_JOB=${EVAL_JOB:-cg_eval}
[ "$ROUNDS" -ge 1 ] || { echo "ROUNDS must be >= 1"; exit 1; }
# [2026-11-16] P0 / CHK0 / TABLE0 / PREV_LAB0 override round 0 (fixed-budget chains: scripts/ag_budget_chain.sh; round 0
# = a subsample of the base rollouts, scripts/libero_round0_subsample.py). With DEP_START set, P0 / CHK0 may not exist yet.
P0=${P0:-$ROOT/runs/policy_ag_residual_vb40_s0_18438185}
CHK0=${CHK0:-$ROOT/runs/contrast_check_ag_residual_vb40_s0}
TABLE0=${TABLE0:-$ROOT/tables/moka_fixed_fresh}
EX="--exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8"   # iris9 excluded again 2026-11-13: one of its cards fails (18463158: no CUDA device) though 18486750 passed
TR="-A iris -p ${TRAIN_PART:-iris} $EX"   # [2026-11-18] TRAIN_PART (e.g. iris-hi when iris is at its 25-job limit; with SBATCH_PARTITION for the rest)
# [2026-11-12] memory: the sbatch default is 96G, but MaxRSS is ~8-14 GB (eval / collect / check; the pi05 restore
# spikes past 16 GB) and ~27 GB (residual train); smaller requests fit nodes whose RAM, not GPUs, is taken
MEM_RUN=${MEM_RUN:---mem=40G} MEM_TRAIN=${MEM_TRAIN:---mem=56G}
STAG=$(echo "$SHIFT" | tr -d .)   # 0.99 -> 099, as in eval_ag_residual_vb40_s0_shift099_noise1
cd $VL
for f in $([ -z "${DEP_START:-}" ] && echo "$P0/trainable_final.msgpack $CHK0/results.json") "$TABLE0/episodes.csv"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
TRAIN_STATES=$(.venv/bin/python -c "import yaml; print(*yaml.safe_load(open('configs/libero_moka.yaml'))['split']['train'])")
[ "$(echo $TRAIN_STATES | wc -w)" -eq 30 ] || { echo "expected 30 train states"; exit 1; }

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

PREV_RUN=$P0 PREV_CHK=$CHK0 PREV_TABLE=$TABLE0 PREV_LAB=${PREV_LAB0:-round0} DEP=${DEP_START:--}
if [[ "$DEP" =~ ^afterok:([0-9]+)$ ]] && ! squeue -h -j "${BASH_REMATCH[1]}" 2>/dev/null | grep -q .; then
    # [2026-11-16] DEP_START already finished (slurm may have forgotten it): require success, then no dependency
    st=$(sacct -j "${BASH_REMATCH[1]}" -X -n -P --format=State 2>/dev/null | head -1)
    [ "$st" = COMPLETED ] || { echo "DEP_START job ${BASH_REMATCH[1]} is '$st', not COMPLETED"; exit 1; }
    DEP=-
fi
START_ROUND=${START_ROUND:-1}
# [2026-11-18] START_ROUND > 1 with PREV_T=- PREV_K=-: branch a new chain (TAG) off an existing round, i.e. round
# START_ROUND-1 = P0 / CHK0 / TABLE0 / PREV_LAB0 (e.g. the 900 chain's r5), so round START_ROUND keeps its own seeds and
# table round number (the collection-push test, COLLECT_SHIFT below).
if [ "$START_ROUND" -gt 1 ] && [ "${PREV_T:?set PREV_T (- = branch off P0 / CHK0 / TABLE0)}" = - ]; then
    [ -n "${TAG:-}" ] && [ -n "${PREV_LAB0:-}" ] || { echo "branching (PREV_T=-) needs TAG and PREV_LAB0"; exit 1; }
elif [ "$START_ROUND" -gt 1 ]; then  # resume (slurm's 25-job submit limit): PREV_T / PREV_K = round START_ROUND-1's
    # train and contrast-check job ids. Depends on PREV_K while slurm still knows it, else needs its finished output.
    : "${PREV_T:?set PREV_T}" "${PREV_K:?set PREV_K}"
    p=$((START_ROUND - 1))
    PREV_RUN=$ROOT/runs/policy_${PFX}_r${p}_s0_$PREV_T PREV_CHK=$ROOT/runs/contrast_check_${PFX}_r${p}_s0
    PREV_TABLE=$ROOT/tables/${TPFX}_${PFX}_r$p PREV_LAB=$ROOT/tables/${TPFX}_${PFX}_r$p/labels/${PFX}_${LABTAG}_r$p
    if squeue -h -j "$PREV_K" 2>/dev/null | grep -q .; then DEP=afterok:$PREV_K
    else [ -e "$PREV_CHK/results.json" ] || { echo "round $p check $PREV_K not queued and $PREV_CHK/results.json missing"; exit 1; }; DEP=-; fi
fi
for r in $(seq $START_ROUND $ROUNDS); do
    echo "# round $r: collect with $(basename $PREV_RUN)"
    PSHIFT=${COLLECT_SHIFT:-$SHIFT}; [ -n "${STACK:-}" ] && [ "$r" -eq 1 ] && PSHIFT=${SHIFT0:-0.99}   # [2026-11-18] P0's own shift
    POL="MODE=eval,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$PREV_RUN,FT_SHIFT_TARGET=$PSHIFT,FT_SHIFT_FROM=$PREV_CHK/results.json,NOISE_SEED=$((100 + r + SO))${COLLECT_EXPORT:-}"
    if [ "$FIXED_REPEATS" -ge 1 ]; then
        CFN=${PFX}_r${r}_collect_fixed NR=$FRESH
        sub CF "$DEP" $EX $MEM_RUN --time=$COLLECT_TIME --job-name=${PFX}_collect --export=ALL,$POL,REPEATS=$FIXED_REPEATS,SEED=$((200 + 2 * r + SO)),NAME=$CFN,CLIENT_ARGS="--args.init-states $TRAIN_STATES --args.save-raw --args.split-name train$CC" scripts/pi05_tag_adarms.sbatch
    else  # fresh layouts only: split over the two collectors
        CFN=${PFX}_r${r}_collect_fresha NR=$((FRESH / 2))
        sub CF "$DEP" $EX $MEM_RUN --time=$COLLECT_TIME --job-name=${PFX}_collect --export=ALL,$POL$ENV_A,REPEATS=1,SEED=$((200 + 2 * r + SO)),NAME=$CFN,CLIENT_ARGS="--args.fresh-layouts $((FRESH - NR)) --args.save-raw --args.split-name train$CC" scripts/pi05_tag_adarms.sbatch
    fi
    sub CR "$DEP" $EX $MEM_RUN --time=$COLLECT_TIME --job-name=${PFX}_collect --export=ALL,$POL$ENV_B,REPEATS=1,SEED=$((201 + 2 * r + SO)),NAME=${PFX}_r${r}_collect_fresh,CLIENT_ARGS="--args.fresh-layouts $NR --args.save-raw --args.split-name train$CC" scripts/pi05_tag_adarms.sbatch
    TABLE=$ROOT/tables/${TPFX}_${PFX}_r$r
    LAB=$TABLE/labels/${PFX}_${LABTAG}_r$r
    if [ $LABELER = iql ]; then LABEL_CMD=".venv/bin/python scripts/libero_iql_round_labels.py --table $TABLE --out $LAB"
    elif [ $LABELER = cv ]; then LABEL_CMD=".venv/bin/python scripts/libero_chainv_round_labels.py --table $TABLE --out $LAB"
    else LABEL_CMD=".venv/bin/python scripts/libero_round_labels.py --table $TABLE --round $((r + 1)) --prev $PREV_LAB --out $LAB"; fi
    RUNS="$ROOT/runs/${CFN}_${SUITE_A}_$CF $ROOT/runs/${PFX}_r${r}_collect_fresh_${SUITE_B}_$CR"
    # [2026-11-18] STACK: also list round r's own train episodes (the stacked head trains on these only)
    EPCMD=${STACK:+" && .venv/bin/python scripts/round_episodes.py --table $TABLE --round $((r + 1)) --out $TABLE/round_episodes.txt"}
    sub TL afterok:$CF:$CR --time=$LABEL_TIME --job-name=${PFX}_label --export=ALL,CMD=".venv/bin/python scripts/libero_round2_table.py --base $PREV_TABLE --round $((r + 1)) --out $TABLE --runs $RUNS && $LABEL_CMD$EPCMD" scripts/vl_python.sbatch
    NAME=${PFX}_r${r}_s0
    ARGS="--train --arm advantage --labels $LAB/labels.parquet --table $TABLE --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps $STEPS --save-every 1000 --seed 0"
    [ -z "${STACK:-}" ] || ARGS="--train --arm advantage --labels $LAB/labels.parquet --table $TABLE --episodes $TABLE/round_episodes.txt --tag-dropout 0 --stack-from $PREV_RUN --stack-shift $PSHIFT --stack-chk $PREV_CHK/results.json --freeze-siglip --lr 3e-4 --accum 2 --steps $STEPS --save-every 1000 --seed 0"
    sub T afterok:$TL $TR $MEM_TRAIN --time=7:30:00 --job-name=${PFX}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch
    RUN=$ROOT/runs/policy_${NAME}_$T
    CHK=$ROOT/runs/contrast_check_$NAME
    sub K afterok:$T $EX $MEM_RUN --time=${CHECK_TIME:-1:00:00} --job-name=${PFX}_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16${CHECK_TABLE:+ --table $TABLE}" scripts/pi05_tag_adarms.sbatch
    if [ $((r % EVAL_EVERY)) -eq 0 ] || [ "$r" -eq "$ROUNDS" ]; then
        # [2026-11-17] EVAL_TIME: eval / ceval time limit (default 3 h; b50cv round 0's 330-trial eval timed out at 3 h)
        [ -n "${NO_VAL_EVAL:-}" ] || sub E afterok:$K $EX $MEM_RUN --time=${EVAL_TIME:-3:00:00} --job-name=$EVAL_JOB --export=ALL,MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift${STAG}_noise1 scripts/pi05_tag_adarms.sbatch
        if [ -n "${CONT_EVAL:-}" ]; then
            sub C afterok:$K $EX $MEM_RUN --time=${EVAL_TIME:-3:00:00} --job-name=${PFX}_ceval --export=ALL,MODE=eval$ENV_A,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$CHK/results.json,NAME=ceval_${NAME}_shift${STAG},CLIENT_ARGS="--args.start-from $CONT_EVAL --args.split-name cont" scripts/pi05_tag_adarms.sbatch
        fi
        if [ -n "${CONT_EVAL_B:-}" ]; then
            sub C2 afterok:$K $EX $MEM_RUN --time=${EVAL_TIME:-3:00:00} --job-name=${PFX}_ceval --export=ALL,MODE=eval$ENV_B,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$CHK/results.json,NAME=ceval_${NAME}_shift${STAG},CLIENT_ARGS="--args.start-from $CONT_EVAL_B --args.split-name cont" scripts/pi05_tag_adarms.sbatch
        fi
    fi
    PREV_RUN=$RUN PREV_CHK=$CHK PREV_TABLE=$TABLE PREV_LAB=$LAB DEP=afterok:$K
done
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
