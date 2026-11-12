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
set -euo pipefail
VL=/move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
ROUNDS=${ROUNDS:-5} SHIFT=${SHIFT:-0.99} EVAL_EVERY=${EVAL_EVERY:-1} STEPS=${STEPS:-5000}
[ "$ROUNDS" -ge 1 ] || { echo "ROUNDS must be >= 1"; exit 1; }
P0=$ROOT/runs/policy_ag_residual_vb40_s0_18438185
CHK0=$ROOT/runs/contrast_check_ag_residual_vb40_s0
TABLE0=$ROOT/tables/moka_fixed_fresh
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
TR="-A iris -p iris $EX"
STAG=$(echo "$SHIFT" | tr -d .)   # 0.99 -> 099, as in eval_ag_residual_vb40_s0_shift099_noise1
cd $VL
for f in "$P0/trainable_final.msgpack" "$CHK0/results.json" "$TABLE0/episodes.csv"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
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

PREV_RUN=$P0 PREV_CHK=$CHK0 PREV_TABLE=$TABLE0 PREV_LAB=round0 DEP=${DEP_START:--}
START_ROUND=${START_ROUND:-1}
if [ "$START_ROUND" -gt 1 ]; then  # resume (slurm's 25-job submit limit): PREV_T / PREV_K = round START_ROUND-1's
    # train and contrast-check job ids. Depends on PREV_K while slurm still knows it, else needs its finished output.
    : "${PREV_T:?set PREV_T}" "${PREV_K:?set PREV_K}"
    p=$((START_ROUND - 1))
    PREV_RUN=$ROOT/runs/policy_agr_r${p}_s0_$PREV_T PREV_CHK=$ROOT/runs/contrast_check_agr_r${p}_s0
    PREV_TABLE=$ROOT/tables/moka_agr_r$p PREV_LAB=$ROOT/tables/moka_agr_r$p/labels/agr_vb40_r$p
    if squeue -h -j "$PREV_K" 2>/dev/null | grep -q .; then DEP=afterok:$PREV_K
    else [ -e "$PREV_CHK/results.json" ] || { echo "round $p check $PREV_K not queued and $PREV_CHK/results.json missing"; exit 1; }; DEP=-; fi
fi
for r in $(seq $START_ROUND $ROUNDS); do
    echo "# round $r: collect with $(basename $PREV_RUN)"
    POL="MODE=eval,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$PREV_RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$PREV_CHK/results.json,NOISE_SEED=$((100 + r))"
    sub CF "$DEP" $EX --time=2:00:00 --job-name=agr_collect --export=ALL,$POL,REPEATS=2,SEED=$((200 + 2 * r)),NAME=agr_r${r}_collect_fixed,CLIENT_ARGS="--args.init-states $TRAIN_STATES --args.save-raw --args.split-name train" scripts/pi05_tag_adarms.sbatch
    sub CR "$DEP" $EX --time=2:00:00 --job-name=agr_collect --export=ALL,$POL,REPEATS=1,SEED=$((201 + 2 * r)),NAME=agr_r${r}_collect_fresh,CLIENT_ARGS="--args.fresh-layouts 40 --args.save-raw --args.split-name train" scripts/pi05_tag_adarms.sbatch
    TABLE=$ROOT/tables/moka_agr_r$r
    LAB=$TABLE/labels/agr_vb40_r$r
    RUNS="$ROOT/runs/agr_r${r}_collect_fixed_libero_10_$CF $ROOT/runs/agr_r${r}_collect_fresh_libero_10_$CR"
    sub TL afterok:$CF:$CR --time=1:00:00 --job-name=agr_label --export=ALL,CMD=".venv/bin/python scripts/libero_round2_table.py --base $PREV_TABLE --round $((r + 1)) --out $TABLE --runs $RUNS && .venv/bin/python scripts/libero_round_labels.py --table $TABLE --round $((r + 1)) --prev $PREV_LAB --out $LAB" scripts/vl_python.sbatch
    NAME=agr_r${r}_s0
    ARGS="--train --arm advantage --labels $LAB/labels.parquet --table $TABLE --tag-dropout 0 --demos --freeze-siglip --lr 3e-4 --accum 2 --steps $STEPS --save-every 1000 --seed 0"
    sub T afterok:$TL $TR --time=7:30:00 --job-name=agr_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="$ARGS" scripts/pi05_tag_adarms.sbatch
    RUN=$ROOT/runs/policy_${NAME}_$T
    CHK=$ROOT/runs/contrast_check_$NAME
    sub K afterok:$T $EX --time=1:00:00 --job-name=agr_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16" scripts/pi05_tag_adarms.sbatch
    if [ $((r % EVAL_EVERY)) -eq 0 ] || [ "$r" -eq "$ROUNDS" ]; then
        sub E afterok:$K $EX --time=3:00:00 --job-name=cg_eval --export=ALL,MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=$SHIFT,FT_SHIFT_FROM=$CHK/results.json,NAME=eval_${NAME}_shift${STAG}_noise1 scripts/pi05_tag_adarms.sbatch
    fi
    PREV_RUN=$RUN PREV_CHK=$CHK PREV_TABLE=$TABLE PREV_LAB=$LAB DEP=afterok:$K
done
[ -n "${SUBMIT:-}" ] || echo "(dry run: SUBMIT=1 to submit)"
