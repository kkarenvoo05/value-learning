#!/bin/bash
# Feeds scripts/ag_rounds.sh one whole round at a time under slurm's 25-job submit limit (iris QOS MaxSubmitPU 25)
# [2026-11-11]. Every 10 min: if a round job failed / was cancelled / has an unsatisfiable dependency -> stop; if
# <= 19 non-interactive jobs are queued -> submit round R (6 jobs) resumed from the previous round's train / check ids.
# Usage: scripts/ag_rounds_feed.sh <first round> <last round> <prev train job> <prev check job>
# Log: /iris/u/ankile/karenvo/libero/logs/ag_rounds_feed.log (each submission's job ids, so a stop can be resumed by hand:
#   ROUNDS=<last> START_ROUND=<r> PREV_T=<id> PREV_K=<id> SUBMIT=1 scripts/ag_rounds.sh).
set -uo pipefail
R=$1 LAST=$2 PT=$3 PK=$4
VL=/move/u/karenvo/Projects/value-learning
LOG=/iris/u/ankile/karenvo/libero/logs/ag_rounds_feed.log
SINCE=${SINCE:-2026-11-11T21:00}
cd $VL
echo "$(date -Is) feeder start: rounds $R-$LAST, prev train $PT check $PK" >> $LOG
while [ "$R" -le "$LAST" ]; do
    bad=$(sacct -u karenvo -S $SINCE -X -n -P --format=JobID,JobName,State | grep -E "\|agr_|\|cg_eval" |
          grep -E "FAILED|CANCELLED|TIMEOUT|OUT_OF_ME|NODE_FAIL" | grep -v -E "^1846032[0-3]\|")
    never=$(squeue -u karenvo -h -o "%i %j %R" | grep -E "agr_|cg_eval" | grep DependencyNeverSatisfied)
    if [ -n "$bad$never" ]; then
        echo "$(date -Is) STOP before round $R: $bad $never" >> $LOG; exit 1
    fi
    n=$(squeue -u karenvo -h -o "%j" | grep -vc "^bash$")
    if [ "$n" -le 19 ]; then
        out=$(ROUNDS=$R START_ROUND=$R PREV_T=$PT PREV_K=$PK SUBMIT=1 scripts/ag_rounds.sh 2>&1)
        echo "$(date -Is) round $R submitted ($n queued before):" >> $LOG; echo "$out" >> $LOG
        T=$(echo "$out" | sed -n 's/^step [0-9]* T: \([0-9]*\)$/\1/p') K=$(echo "$out" | sed -n 's/^step [0-9]* K: \([0-9]*\)$/\1/p')
        [ -n "$T" ] && [ -n "$K" ] && [ "$(echo "$out" | grep -c '^step ')" -eq 6 ] ||
            { echo "$(date -Is) STOP: round $R submission incomplete" >> $LOG; exit 1; }
        PT=$T PK=$K R=$((R + 1))
        continue
    fi
    sleep 600
done
echo "$(date -Is) feeder done: all rounds through $LAST submitted" >> $LOG
