#!/bin/bash
# Base pi05 on the continuous eval set moka_cont100 (300 start rows) [2026-11-17; Karen: "queue base on the continuous
# eval set"]. Same client settings as the chains' ceval jobs (no matched noise: --args.start-from runs alone), stock
# pi05_libero server. Waits for < 25 queued iris jobs under the feeders' lock, submits once, exits.
# Usage: setsid nohup bash scripts/ceval_base_submit.sh >> /iris/u/ankile/karenvo/libero/logs/ceval_base.log 2>&1 < /dev/null &
set -uo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero
echo "$(date -Is) start on $(hostname), pid $$"
while :; do
    exec 9>"$ROOT/logs/.ag_rounds_feed.lock"; flock 9
    if [ "$(squeue -u karenvo -h -o %q | grep -vc interactive)" -lt 25 ]; then
        j=$(SUITE=libero_10 TASKS=8 REPEATS=33 SEED=7 NAME=ceval_base \
            CLIENT_ARGS="--args.start-from $ROOT/eval_sets/moka_cont100/starts.csv --args.split-name cont" \
            sbatch --parsable --exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8 --mem=40G --time=4:00:00 \
            --job-name=ceval_base scripts/libero_eval.sbatch) && { flock -u 9; echo "$(date -Is) submitted $j"; exit 0; }
        echo "$(date -Is) sbatch failed ($j)"
    fi
    flock -u 9; sleep 120
done
