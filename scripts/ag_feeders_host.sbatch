#!/bin/bash
# Hosts the AG-chain feeders (scripts/ag_rounds_feed.sh) in a long CPU-only batch job, so they outlive interactive
# sessions [2026-11-17; the feeders ran under nohup inside an interactive job and would die with it].
#   <FEED_DIR>/<TAG>.env : one feeder each. KEY=VALUE lines (exported to the feeder, e.g. TAG, LABELER, FRESH, P0,
#                          DEP_START, IGNORE) + FEED_ARGS="<first round> <last round> <prev train> <prev check>".
#                          Feeder output -> <FEED_DIR>/<TAG>.out; its log stays logs/ag_rounds_feed_<TAG>.log.
#   <FEED_DIR>/*.pending : one-off submissions (bash scripts), run once the queue has room (< 25 jobs, same lock as the
#                          feeders), then renamed .done (success) or .failed.
# To restart (e.g. after this job's time limit): update each .env's FEED_ARGS / DEP_START from the feeder log's last
# "round R submitted" block (train / check ids), then sbatch this file again. Never run two hosts at once.
# Usage: [OLD_JOB=<id>] sbatch scripts/ag_feeders_host.sbatch   [FEED_DIR=<dir>]
#    or, on an interactive node (the #SBATCH lines are then plain comments; the feeders live as long as that session):
#    setsid nohup bash scripts/ag_feeders_host.sbatch >> /iris/u/ankile/karenvo/libero/logs/feeders/host.log 2>&1 < /dev/null &
#SBATCH --job-name=ag_feeders
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -uo pipefail
DIR=${FEED_DIR:-/iris/u/ankile/karenvo/libero/logs/feeders}
LOCK=/iris/u/ankile/karenvo/libero/logs/.ag_rounds_feed.lock
cd /move/u/karenvo/Projects/value-learning
exec 8>"$DIR/.host.lock"   # one host at a time (held by this script and every feeder it starts)
flock -n 8 || { echo "$(date -Is) another feeder host holds $DIR/.host.lock; not starting"; exit 1; }
echo "host $(hostname) start $(date -Is), feeders from $DIR"
# OLD_JOB=<job id>: start only once that job (the interactive job whose feeders these replace) has ended, or once
# <FEED_DIR>/GO exists (stop the old feeders by hand first, then touch GO).
if [ -n "${OLD_JOB:-}" ]; then
    until [ -e "$DIR/GO" ] || ! squeue -h -j "$OLD_JOB" 2>/dev/null | grep -q .; do sleep 20; done
    echo "$(date -Is) go ($([ -e "$DIR/GO" ] && echo GO file || echo "job $OLD_JOB ended"))"; sleep 30
fi
for f in "$DIR"/*.env; do
    t=$(basename "$f" .env)
    args=$(set -a; source "$f"; echo "$FEED_ARGS")
    # resume from the feeder log if it records a later round than the .env: its last "round R submitted (" block's
    # train / check ids -> rounds R+1..last
    log=/iris/u/ankile/karenvo/libero/logs/ag_rounds_feed_$t.log
    last=$(grep -n " round [0-9]* submitted (" "$log" 2>/dev/null | tail -1)
    if [ -n "$last" ]; then
        r=$(echo "$last" | sed 's/.* round \([0-9]*\) submitted (.*/\1/')
        blk=$(tail -n +"${last%%:*}" "$log")
        T=$(echo "$blk" | sed -n 's/^step [0-9]* T: \([0-9]*\)$/\1/p' | head -1)
        K=$(echo "$blk" | sed -n 's/^step [0-9]* K: \([0-9]*\)$/\1/p' | head -1)
        set -- $args
        [ -n "$T" ] && [ -n "$K" ] && [ $((r + 1)) -ge "$1" ] && args="$((r + 1)) $2 $T $K"
    fi
    ( set -a; source "$f"; set +a; exec scripts/ag_rounds_feed.sh $args ) >> "$DIR/$t.out" 2>&1 < /dev/null &
    echo "$(date -Is) feeder $t pid $! (rounds/prev: $args)"
done
(   # one-off submissions, when there is room
    while compgen -G "$DIR/*.pending" > /dev/null; do
        for p in "$DIR"/*.pending; do
            exec 9>$LOCK; flock 9
            if [ "$(squeue -u karenvo -h | wc -l)" -lt 25 ]; then
                if out=$(bash "$p" 2>&1); then mv "$p" "${p%.pending}.done"; s=done; else mv "$p" "${p%.pending}.failed"; s=FAILED; fi
                echo "$(date -Is) $(basename "$p"): $s: $out"
            fi
            flock -u 9
        done
        sleep 300
    done
    echo "$(date -Is) no pending submissions left"
) &
wait
echo "end $(date -Is)"
