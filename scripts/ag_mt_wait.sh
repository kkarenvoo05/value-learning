#!/bin/bash
# [2026-11-16] Submit the multi-task AG setup as the queue frees (25-job limit), under the feeders' lock:
#   1. the base task-64 val run V (20 fresh layouts, env seed 302, split val) once <= 24 jobs are queued
#   2. scripts/ag_mt_chain.sh for CHAIN in mt100 t64b50 t64b100 (in that order), each once <= 19 are queued
#      (round 0 = 6 / 4 / 4 jobs, then its feeder), depending on the base runs it needs (A: 18549968, B: 18549969)
# Usage: setsid nohup scripts/ag_mt_wait.sh > logs/ag_mt_wait.nohup 2>&1 &   (log logs/ag_mt_wait.log)
set -uo pipefail
VL=/move/u/karenvo/Projects/value-learning ROOT=/iris/u/ankile/karenvo/libero
LOG=$ROOT/logs/ag_mt_wait.log LOCK=$ROOT/logs/.ag_rounds_feed.lock
JA=${JA:-18549968} JB=${JB:-18549969} JV=${JV:-}
cd $VL
queued() { squeue -u karenvo -h -o "%j" | grep -vc "^bash$"; }
dep() {  # afterok on the base jobs still queued; finished ones must be COMPLETED
    local d=
    for j in "$@"; do
        if squeue -h -j $j 2>/dev/null | grep -q .; then d=${d:+$d:}$j
        else st=$(sacct -j $j -X -n -P --format=State | head -1)
             [ "$st" = COMPLETED ] || { echo "base job $j is '$st'" >> $LOG; return 1; }; fi
    done
    echo ${d:+afterok:$d}
}
if [ -z "$JV" ]; then
    while true; do
        exec 9>$LOCK; flock 9
        if [ "$(queued)" -le 23 ]; then
            # smoke: the residual collect path on task 64 (pi05_tag_adarms.sbatch AG_SUITE / AG_TASK / AG_MAX_STEPS)
            JS=$(sbatch --parsable --exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8 --mem=40G --time=1:00:00 --job-name=t64_smoke \
                 --export=ALL,MODE=eval,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$ROOT/runs/policy_ag_residual_vb40_s0_18438185,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$ROOT/runs/contrast_check_ag_residual_vb40_s0/results.json,AG_SUITE=libero_90,AG_TASK=64,AG_MAX_STEPS=520,REPEATS=1,SEED=999,NAME=smoke_t64_collect,CLIENT_ARGS="--args.fresh-layouts 2 --args.save-raw --args.split-name train" scripts/pi05_tag_adarms.sbatch)
            echo "$(date -Is) smoke submitted: $JS" >> $LOG
            JV=$(SUITE=libero_90 TASKS=64 REPEATS=1 SEED=302 NAME=t64_base_V CLIENT_ARGS="--args.fresh-layouts 20 --args.save-raw --args.max-steps 520 --args.split-name val" \
                 sbatch --parsable --exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8 --mem=40G --time=3:00:00 --job-name=t64_base scripts/libero_eval.sbatch)
            flock -u 9; echo "$(date -Is) val run V submitted: $JV" >> $LOG; break
        fi
        flock -u 9; sleep 300
    done
fi
RA=$ROOT/runs/t64_base_A_libero_90_$JA RB=$ROOT/runs/t64_base_B_libero_90_$JB RV=$ROOT/runs/t64_base_V_libero_90_$JV
for c in mt100 t64b50 t64b100; do
    while true; do
        exec 9>$LOCK; flock 9
        if [ "$(queued)" -le 19 ]; then
            if [ $c = t64b100 ]; then D=$(dep $JA $JB $JV) || exit 1; else D=$(dep $JA $JV) || exit 1; fi
            out=$(CHAIN=$c RUN_A=$RA RUN_B=$RB RUN_V=$RV DEP=$D SUBMIT=1 scripts/ag_mt_chain.sh 2>&1); rc=$?
            flock -u 9
            echo "$(date -Is) $c submitted (dep ${D:-none}, rc $rc): $out" >> $LOG
            [ $rc -eq 0 ] || { echo "$(date -Is) STOP: $c failed" >> $LOG; exit 1; }
            break
        fi
        flock -u 9; sleep 600
    done
done
echo "$(date -Is) done" >> $LOG
