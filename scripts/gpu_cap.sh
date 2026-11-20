#!/bin/bash
# Keep at most CAP of my batch GPU jobs running [2026-11-17: "try to keep the number of gpus used to 6"].
# Slurm has no per-user running cap we can set, so this holds every pending job (user hold) and releases them as slots
# free up. The interactive shell job ("bash") is not counted or touched; nor are CPU-only hosts (ag_feeders*, gpu_cap).
#   running  = my RUNNING / CONFIGURING / COMPLETING jobs (any partition)
#   granted  = jobs this script already released that are still pending (they will start soon, so they use a slot)
#   ready    = held jobs whose dependencies are all satisfied
# [2026-11-19 v3; Karen: "make sure to check both iris hi and iris for gpu's. do not just rely on one cluster, continue
# checking both for opportunities"] Both partitions: every job this script releases is first moved to the partition
# list "iris-hi,iris" (same nodes; slurm starts it on whichever can run it first: iris-hi = 4 GPUs per user, never
# preempted; iris = 20 per user, PreemptMode REQUEUE: a preempted train warm-restarts from its newest checkpoint,
# an eval reruns). Granted jobs still waiting on one partition are moved too, every pass. Hence no iris-hi room
# accounting any more (v2 released at most 4 iris-hi jobs at a time and left iris-only jobs stuck behind them).
# Release order: priority regexes in $PRIOF (one per line, file order, oldest first within a line), then *_train jobs,
# then everything else, oldest first. A held job whose dependency failed is released so it shows
# DependencyNeverSatisfied and the feeders clean it up. CAP is read each pass from $CAPF (default 6).
# If this script stops, held jobs stay held: `scontrol release <ids>` (or restart this script).
# Run: sbatch scripts/gpu_cap.sbatch (CPU-only batch host, outlives sessions), or under setsid nohup on a login node.
CAPF=/iris/u/ankile/karenvo/libero/logs/.gpu_cap
GRANT=/iris/u/ankile/karenvo/libero/logs/.gpu_cap_granted
PRIOF=/iris/u/ankile/karenvo/libero/logs/.gpu_cap_priority
# [2026-11-20; Karen: focus GPUs on the round-1 drop] Paused jobs: one regex per line in $PAUSEF, matched against the job
# name or the job id; matching held jobs are never released (delete the line to un-pause). Re-read every pass.
PAUSEF=/iris/u/ankile/karenvo/libero/logs/.gpu_cap_pause
BOTH=iris-hi,iris
touch "$GRANT"
exec 9>/iris/u/ankile/karenvo/libero/logs/.gpu_cap.lock   # one loop at a time
flock -n 9 || { echo "$(date '+%F %T') another gpu_cap loop holds the lock; not starting"; exit 1; }
echo "$(date '+%F %T') start pid $$ host $(hostname) (v3: both partitions)"
both() {  # both <job> <its partition>: offer the job to iris-hi and iris (only single-partition iris / iris-hi jobs)
    case $2 in iris|iris-hi)
        scontrol update JobId=$1 Partition=$BOTH 2>/dev/null && echo "$(date '+%F %T') $1: $2 -> $BOTH" ||
            echo "$(date '+%F %T') $1: could not add the second partition (kept $2)" ;;
    esac
}
while true; do
    CAP=$(cat "$CAPF" 2>/dev/null || echo 6)
    q=$(squeue -u karenvo -h -o "%i|%j|%T|%r|%P|%E" | grep -vE "\|(bash|ag_feeders[^|]*|gpu_cap)\|")   # CPU-only hosts are not GPU jobs
    running=$(echo "$q" | grep -cE "\|(RUNNING|CONFIGURING|COMPLETING)\|")
    pend=$(echo "$q" | grep "|PENDING|")
    granted=$(echo "$pend" | grep -v "JobHeldUser" | cut -d'|' -f1 | grep -xFf "$GRANT" || true)
    ng=$(echo -n "$granted" | grep -c . || true)
    # granted jobs still waiting on a single partition -> both
    echo "$pend" | grep -v "JobHeldUser" | awk -F'|' '$5=="iris" || $5=="iris-hi" {print $1, $5}' |
        while read -r j p; do grep -qx "$j" "$GRANT" && both $j $p; done
    # hold every pending job not granted and not already held (dependency-failed jobs are left alone)
    tohold=$(echo "$pend" | grep -v "JobHeldUser" | grep -v "(failed)" | cut -d'|' -f1 | grep -vxFf "$GRANT" || true)
    [ -n "$tohold" ] && scontrol hold $tohold && echo "$(date '+%F %T') hold $(echo $tohold)"
    # held jobs with a failed dependency -> release so the feeders see DependencyNeverSatisfied
    dead=$(echo "$pend" | grep "JobHeldUser" | grep "(failed)" | cut -d'|' -f1)
    for j in $dead; do scontrol release $j && echo $j >> "$GRANT" && echo "$(date '+%F %T') release $j (dependency failed)"; done
    slots=$((CAP - running - ng))
    if [ "$slots" -gt 0 ]; then
        held=$(echo "$pend" | grep "JobHeldUser" | grep -v "unfulfilled\|(failed)")
        pause=$(grep -v '^ *$' "$PAUSEF" 2>/dev/null | paste -sd'|' || true)
        [ -n "$pause" ] && held=$(echo "$held" | grep -vE "^(($pause)\||[0-9]+\|($pause)\|)" || true)
        prio=""
        while read -r pat; do
            [ -n "$pat" ] || continue
            for j in $(echo "$held" | grep -E "^[0-9]+\|($pat)\|" | cut -d'|' -f1 | sort -n); do
                echo "$prio" | grep -qx "$j" || prio=$(printf "%s\n%s" "$prio" "$j")
            done
        done < <(cat "$PRIOF" 2>/dev/null)
        held=$(echo "$held" | grep -vE "^($(echo $prio | tr ' ' '|'))\|" || true)
        # long training runs (job name *_train) before evals, then oldest first
        rest=$( (echo "$held" | grep "_train|" | cut -d'|' -f1 | sort -n
                 echo "$held" | grep -v "_train|" | cut -d'|' -f1 | sort -n) | grep . || true)
        ready=$(printf "%s\n%s\n" "$prio" "$rest" | grep . | head -n "$slots")
        for j in $ready; do
            both $j "$(echo "$pend" | awk -F'|' -v j=$j '$1==j {print $5}')"
            scontrol release $j && echo $j >> "$GRANT" && echo "$(date '+%F %T') release $j (running $running, granted $ng, cap $CAP)"
        done
    fi
    sleep 10
done
