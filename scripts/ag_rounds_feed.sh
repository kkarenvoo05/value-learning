#!/bin/bash
# Feeds scripts/ag_rounds.sh one whole round at a time under slurm's 25-job submit limit (iris QOS MaxSubmitPU 25)
# [2026-11-11]. Every 10 min: if a round job failed / was cancelled / has an unsatisfiable dependency -> stop; if
# <= 19 non-interactive jobs are queued -> submit round R (6 jobs) resumed from the previous round's train / check ids.
# Usage: scripts/ag_rounds_feed.sh <first round> <last round> <prev train job> <prev check job>
# Log: /iris/u/ankile/karenvo/libero/logs/ag_rounds_feed.log (each submission's job ids, so a stop can be resumed by hand:
#   ROUNDS=<last> START_ROUND=<r> PREV_T=<id> PREV_K=<id> SUBMIT=1 scripts/ag_rounds.sh).
# [2026-11-16] TAG / FIXED_REPEATS / FRESH / COLLECT_TIME pass through to ag_rounds.sh (export them). With TAG set:
# own log ag_rounds_feed_<TAG>.log, and only that chain's jobs (agr_<TAG>_*) can stop it. Round 1: prev ids "-".
# Several feeders share one lock around count + submit, so together they stay under the 25-job limit.
set -uo pipefail
R=$1 LAST=$2 PT=$3 PK=$4
VL=/move/u/karenvo/Projects/value-learning
LOG=/iris/u/ankile/karenvo/libero/logs/ag_rounds_feed${TAG:+_$TAG}.log
LOCK=/iris/u/ankile/karenvo/libero/logs/.ag_rounds_feed.lock
if [ -n "${TAG:-}" ]; then PAT="\\|agr_${TAG}_" QPAT=" agr_${TAG}_"   # sacct "id|name|state" / squeue "id name reason"
else PAT='\|agr_|\|cg_eval' QPAT='agr_|cg_eval'; fi
SINCE=${SINCE:-2026-11-11T21:00}
cd $VL
echo "$(date -Is) feeder start: rounds $R-$LAST, prev train $PT check $PK" >> $LOG
while [ "$R" -le "$LAST" ]; do
    bad=$(sacct -u karenvo -S $SINCE -X -n -P --format=JobID,JobName,State | grep -E "$PAT" |
          grep -E "FAILED|CANCELLED|TIMEOUT|OUT_OF_ME|NODE_FAIL" | grep -v -E "^(1846032[0-3]${IGNORE:+|$IGNORE})\|")   # IGNORE: regex of job ids recovered by hand (e.g. requeued after a bad node)
    never=$(squeue -u karenvo -h -o "%i %j %R" | grep -E "$QPAT" | grep DependencyNeverSatisfied)
    if [ -n "$bad$never" ]; then
        echo "$(date -Is) STOP before round $R: $bad $never" >> $LOG; exit 1
    fi
    # [2026-11-16] JUST_IN_TIME=1: submit round R only once round R-1's check has left the queue (round R's collection
    # waits for it anyway), so several chains don't fill the 25-job limit with rounds that cannot start yet
    # (first round of a chain: wait for DEP_START's job, e.g. round 0's check)
    JK=$PK; [ "$JK" = - ] && [[ "${DEP_START:-}" =~ ^afterok:([0-9]+)$ ]] && JK=${BASH_REMATCH[1]}
    if [ -n "${JUST_IN_TIME:-}" ] && [ "$JK" != - ] && squeue -h -j "$JK" 2>/dev/null | grep -q .; then sleep 600; continue; fi
    exec 9>$LOCK; flock 9
    # [2026-11-19] count only jobs the iris QOS limits (MaxSubmitPU 25: partition list contains iris, e.g. iris-hi,iris;
    # iris-hi alone has no submit limit); FEED_MAX = most queued before a round is submitted (19 + 6 jobs = 25; 8-job
    # multi-task rounds: 17)
    n=$(squeue -u karenvo -h -o "%P" | grep -cE "(^|,)iris(,|$)")
    if [ "$n" -le "${FEED_MAX:-19}" ]; then
        out=$(ROUNDS=$R START_ROUND=$R PREV_T=$PT PREV_K=$PK SUBMIT=1 scripts/ag_rounds.sh 2>&1)
        flock -u 9
        echo "$(date -Is) round $R submitted ($n queued before):" >> $LOG; echo "$out" >> $LOG
        T=$(echo "$out" | sed -n 's/^step [0-9]* T: \([0-9]*\)$/\1/p') K=$(echo "$out" | sed -n 's/^step [0-9]* K: \([0-9]*\)$/\1/p')
        [ -n "$T" ] && [ -n "$K" ] && [[ "$(echo "$out" | grep -c '^step ')" =~ ^[678]$ ]] ||   # 7 with CONT_EVAL, 8 with CONT_EVAL_B (multi-task)
            { echo "$(date -Is) STOP: round $R submission incomplete" >> $LOG; exit 1; }
        PT=$T PK=$K R=$((R + 1))
        continue
    fi
    flock -u 9
    sleep 600
done
echo "$(date -Is) feeder done: all rounds through $LAST submitted" >> $LOG
