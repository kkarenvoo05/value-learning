#!/bin/bash
# Round 6 first (Karen 2026-11-14 ~11:45: "hold my pending jobs so round 6 goes first").
# The distill + noisy-judge jobs pending at that point are held (scontrol hold) and the remaining lanes unsubmitted.
# This watcher waits until round 6's agr_train is RUNNING (its collection and training have gone first), or until
# 19:00 (the interactive job running it ends 19:15), then releases the held jobs and resumes both chains
# (distill base lane; noisy-judge test dropped 2026-11-14 12:20-12:30 at Karen's request), which queue behind round 6 with CAP 20.
set -u
VL=/move/u/karenvo/Projects/value-learning
LOG=/iris/u/ankile/karenvo/libero/logs/round6_first_watch.log
HELD="18507522 18508053 18508042 18507747 18507697 18507772"
DEADLINE=$(date -d "today 19:00" +%s)
cd $VL
echo "$(date -Is) start: waiting for round 6 agr_train RUNNING (deadline 19:00)" >> $LOG
while :; do
    sub=$(grep -c "round 6 submitted" /iris/u/ankile/karenvo/libero/logs/ag_rounds_feed.log)
    run=$(squeue -u karenvo -h -t R -o "%j" | grep -c "^agr_train$")
    [ "$sub" -ge 1 ] && [ "$run" -ge 1 ] && { echo "$(date -Is) round 6 train running" >> $LOG; break; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date -Is) deadline reached (round 6 submitted: $sub)" >> $LOG; break; }
    sleep 300
done
scontrol release $HELD >> $LOG 2>&1; echo "$(date -Is) released: $HELD" >> $LOG
KINDS=base TB_base=18507522 PREV_TRAIN=18507772 TRAIN_KINDS=base CAP=20 SUBMIT=1 \
    bash scripts/distill_chain.sh >> /iris/u/ankile/karenvo/libero/logs/distill_chain.log 2>&1 &
wait
echo "$(date -Is) both chains done submitting" >> $LOG
