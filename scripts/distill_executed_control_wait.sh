#!/bin/bash
# Executed-target control (2026-11-14, Karen: "add the executed-target control fine-tune"): waits until the resumed
# distill chain (started by scripts/round6_first_watch.sh) submits the base sampled fine-tune, then submits
# distill_base_executed_s0 (+ evals @final, @2000) behind it on the same lane.
set -u
cd /move/u/karenvo/Projects/value-learning
DL=/iris/u/ankile/karenvo/libero/logs/distill_chain.log
START=12
until F=$(tail -n +$((START + 1)) $DL | sed -n 's/^.* step 1 F: \([0-9]*\)$/\1/p' | head -1); [ -n "$F" ]; do sleep 120; done
echo "$(date -Is) # executed-target control after base sampled train $F" >> $DL
KINDS=base TB_base=18507522 PREV_TRAIN=$F TRAIN_KINDS=base_executed CAP=20 SUBMIT=1 bash scripts/distill_chain.sh >> $DL 2>&1
