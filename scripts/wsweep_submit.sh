#!/bin/bash
# Dense guidance-push sweep on AG rounds 5 and 7 [2026-11-17; Karen: "queue the dense sweep on r5 and r7. make it the
# next run"]. Push (action shift / policy noise) 0.6, 0.8, 1.2, 1.5, 1.8; 0.99 already exists (r5 18484607 0.876,
# r7 18524603 0.830). w = push / slope from each round's own contrast check (pi05_tag_adarms.sbatch shift_to_w), same
# eval as the rounds: 330 val trials, matched noise seed 1 (paired with the 0.99 evals).
# Submits one eval at a time whenever < 25 iris jobs are queued, under the feeders' lock. The feeders submit only at
# <= 19, so the sweep takes freed slots first. Each submitted job is moved to the top of my pending jobs (scontrol
# top). State: logs/wsweep.state (submitted labels), so a rerun resumes. Log: logs/wsweep.log.
# Usage: setsid nohup bash scripts/wsweep_submit.sh >> /iris/u/ankile/karenvo/libero/logs/wsweep.log 2>&1 < /dev/null &
set -uo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
LOCK=$ROOT/logs/.ag_rounds_feed.lock
STATE=$ROOT/logs/wsweep.state
touch "$STATE"
declare -A RUN=([r5]=$ROOT/runs/policy_agr_r5_s0_18484605 [r7]=$ROOT/runs/policy_agr_r7_s0_18524601)
# most informative first: the untested range above 0.99, then the far ends
ORDER="r5:1.5 r7:1.5 r5:1.2 r7:1.2 r5:1.8 r7:1.8 r5:0.8 r7:0.8 r5:0.6 r7:0.6"
EX="--exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8"
queued() { squeue -u karenvo -h -o %q | grep -vc interactive; }
echo "$(date -Is) start on $(hostname), pid $$"
for item in $ORDER; do
    r=${item%%:*} s=${item#*:}
    grep -qx "$item" "$STATE" && continue
    stag=$(echo "$s" | tr -d .)
    name=eval_agr_${r}_s0_shift${stag}_noise1
    while :; do
        exec 9>"$LOCK"; flock 9
        if [ "$(queued)" -lt 25 ]; then
            j=$(sbatch --parsable $EX --mem=40G --time=5:00:00 --job-name=wsweep_eval \
                --export=ALL,MODE=eval,MATCHED_NOISE=1,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=${RUN[$r]},FT_SHIFT_TARGET=$s,FT_SHIFT_FROM=$ROOT/runs/contrast_check_agr_${r}_s0/results.json,NAME=$name \
                scripts/pi05_tag_adarms.sbatch)
            rc=$?
            flock -u 9
            if [ $rc -eq 0 ] && [ -n "$j" ]; then
                echo "$item" >> "$STATE"
                top=$(scontrol top "$j" 2>&1) && top=ok
                echo "$(date -Is) $item -> $j ($name; scontrol top: $top)"
                break
            fi
            echo "$(date -Is) $item: sbatch failed ($j); retry in 2 min"
        else
            flock -u 9
        fi
        sleep 120
    done
done
echo "$(date -Is) all 10 sweep evals submitted"
