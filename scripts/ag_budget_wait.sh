#!/bin/bash
# Submit scripts/ag_budget_chain.sh for each budget in turn once the queue has room [2026-11-16]: under the feeders'
# lock, waits until <= 19 non-interactive jobs are queued (round 0 = 3 jobs, then its feeder adds 6 per round).
# Usage: scripts/ag_budget_wait.sh 200 50   (log logs/ag_budget_wait.log)
set -uo pipefail
VL=/move/u/karenvo/Projects/value-learning LOG=/iris/u/ankile/karenvo/libero/logs/ag_budget_wait.log
LOCK=/iris/u/ankile/karenvo/libero/logs/.ag_rounds_feed.lock
cd $VL
for b in "$@"; do
    while true; do
        exec 9>$LOCK; flock 9
        n=$(squeue -u karenvo -h -o "%j" | grep -vc "^bash$")
        if [ "$n" -le 19 ]; then
            out=$(B=$b SUBMIT=1 scripts/ag_budget_chain.sh 2>&1); rc=$?
            flock -u 9
            echo "$(date -Is) b$b submitted ($n queued before, rc $rc): $out" >> $LOG
            [ $rc -eq 0 ] || { echo "$(date -Is) STOP: b$b failed" >> $LOG; exit 1; }
            break
        fi
        flock -u 9; sleep 600
    done
done
echo "$(date -Is) done: $*" >> $LOG
