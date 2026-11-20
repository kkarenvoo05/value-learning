#!/bin/bash
# Fixed-budget AG chains run in rotation, two rounds per turn [2026-11-16; Karen: "instead of doing all 10 rounds for
# each budget at once, run round 0 and 1 for 100 then alternate to 200 and 50"]:
#   b100 r0-1 -> b200 r0-1 -> b50 r0-1 -> b100 r2-3 -> b200 r2-3 -> b50 r2-3 -> ... -> r10 (last turn: one round).
# A turn is submitted once the previous turn's last contrast check has COMPLETED (its eval still runs beside the next
# turn's collection) and the queue has room for the turn's jobs (25-job limit, same lock as ag_rounds_feed.sh).
# Round 0 = scripts/ag_budget_chain.sh NOFEED=1 (subsample table, round-0 args); rounds >= 1 = scripts/ag_rounds.sh
# with TAG=b<B> and that budget's round 0. Stops on a failed / cancelled check.
# State (resumable after the interactive job ends: just rerun): logs/ag_budget_rotate.state; log logs/ag_budget_rotate.log
# Usage: setsid nohup scripts/ag_budget_rotate.sh > logs/ag_budget_rotate.nohup 2>&1 &   [DRY=1: print the plan only]
set -uo pipefail
VL=/move/u/karenvo/Projects/value-learning ROOT=/iris/u/ankile/karenvo/libero
LOG=$ROOT/logs/ag_budget_rotate.log STATE=$ROOT/logs/ag_budget_rotate.state LOCK=$ROOT/logs/.ag_rounds_feed.lock
LAST_ROUND=${LAST_ROUND:-10} ORDER=${ORDER:-"100 200 50"}
cd $VL

BLOCKS=()   # "<budget> <first round> <last round>"
for s in $(seq 0 2 $LAST_ROUND); do
    e=$((s + 1)); [ $e -gt $LAST_ROUND ] && e=$LAST_ROUND
    for b in $ORDER; do BLOCKS+=("$b $s $e"); done
done
if [ -n "${DRY:-}" ]; then printf '%s\n' "${BLOCKS[@]}" | nl -v0; exit 0; fi

# State: NEXT (index into BLOCKS), WAIT_K (check to wait for), T_b<B> / K_b<B> / R0T_b<B> (last train / check, round-0 train)
[ -e $STATE ] || { echo "missing $STATE"; exit 1; }
source $STATE
save() { { echo "NEXT=$NEXT WAIT_K=$WAIT_K"; for b in $ORDER; do eval "echo T_b$b=\${T_b$b:-} K_b$b=\${K_b$b:-} R0T_b$b=\${R0T_b$b:-}"; done; } > $STATE.tmp && mv $STATE.tmp $STATE; }
say() { echo "$(date -Is) $*" >> $LOG; }
budget_env() {  # budget_env <B>: ag_rounds.sh settings of that budget
    local b=$1 fr fn ct; case $b in 50) fr=0 fn=50 ct=2:00:00 ;; 100) fr=0 fn=100 ct=2:00:00 ;; 200) fr=0 fn=200 ct=3:00:00 ;; esac
    local r0t; eval "r0t=\${R0T_b$b}"
    local t0=$ROOT/tables/moka_agr_b${b}_r0
    echo "TAG=b$b FIXED_REPEATS=$fr FRESH=$fn COLLECT_TIME=$ct P0=$ROOT/runs/policy_agr_b${b}_r0_s0_$r0t CHK0=$ROOT/runs/contrast_check_agr_b${b}_r0_s0 TABLE0=$t0 PREV_LAB0=$t0/labels/agr_b${b}_vb40_r0 CONT_EVAL=$ROOT/eval_sets/moka_cont100/starts.csv"
}
say "rotate start: NEXT=$NEXT WAIT_K=$WAIT_K, ${#BLOCKS[@]} turns"
while [ "$NEXT" -lt "${#BLOCKS[@]}" ]; do
    read b s e <<< "${BLOCKS[$NEXT]}"
    if [ "$WAIT_K" != - ]; then
        if squeue -h -j "$WAIT_K" 2>/dev/null | grep -q .; then sleep 600; continue; fi
        st=$(sacct -j "$WAIT_K" -X -n -P --format=State | head -1)
        [ "$st" = COMPLETED ] || { say "STOP before turn $NEXT (b$b r$s-$e): check $WAIT_K is '$st'"; exit 1; }
    fi
    need=$(( (s == 0 ? 4 : 0) + 7 * (e - (s == 0 ? 1 : s) + 1) ))   # round 0: 4 jobs; a round: 7 (incl. the continuous eval)
    exec 9>$LOCK; flock 9
    n=$(squeue -u karenvo -h -o "%j" | grep -vc "^bash$")
    if [ $((n + need)) -gt 25 ]; then flock -u 9; sleep 600; continue; fi
    out=""
    if [ "$s" -eq 0 ]; then
        o=$(B=$b NOFEED=1 SUBMIT=1 scripts/ag_budget_chain.sh 2>&1) || { flock -u 9; say "STOP: b$b round 0 failed: $o"; exit 1; }
        out="$o"
        r0t=$(echo "$o" | sed -n 's/^b[0-9]* round 0: train \([0-9]*\) check \([0-9]*\) .*/\1/p')
        r0k=$(echo "$o" | sed -n 's/^b[0-9]* round 0: train \([0-9]*\) check \([0-9]*\) .*/\2/p')
        [ -n "$r0t" ] && [ -n "$r0k" ] || { flock -u 9; say "STOP: b$b round 0 ids not parsed: $o"; exit 1; }
        eval "R0T_b$b=$r0t T_b$b=$r0t K_b$b=$r0k"
        o=$(env $(budget_env $b) DEP_START=afterok:$r0k START_ROUND=1 ROUNDS=$e SUBMIT=1 scripts/ag_rounds.sh 2>&1); rc=$?
    else
        eval "pt=\${T_b$b} pk=\${K_b$b}"
        o=$(env $(budget_env $b) DEP_START=skip-file-check START_ROUND=$s ROUNDS=$e PREV_T=$pt PREV_K=$pk SUBMIT=1 scripts/ag_rounds.sh 2>&1); rc=$?
    fi
    flock -u 9
    out="$out
$o"
    say "turn $NEXT b$b rounds $s-$e submitted ($n queued before, rc $rc):
$out"
    T=$(echo "$o" | sed -n 's/^step [0-9]* T: \([0-9]*\)$/\1/p' | tail -1) K=$(echo "$o" | sed -n 's/^step [0-9]* K: \([0-9]*\)$/\1/p' | tail -1)
    [ $rc -eq 0 ] && [ -n "$T" ] && [ -n "$K" ] || { say "STOP: turn $NEXT submission incomplete"; exit 1; }
    eval "T_b$b=$T K_b$b=$K"; WAIT_K=$K NEXT=$((NEXT + 1)); save
done
say "rotate done: all ${#BLOCKS[@]} turns submitted"
