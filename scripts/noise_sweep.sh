#!/bin/bash
# Noise sweep [2026-11-09, mentor: "maybe 2x is still not enough to move the needle"; Karen: set it up].
# Same design as the data-variety pilot (reports/2026-11-08-explore-pilot/, scripts/libero_explore_pilot.py) so the
# pilot's control 18388271 and x2.0 p0.5 18388273 are reused as-is: base pi05_libero on our server (BASE_OURS=1),
# 30 fixed train states x 3 repeats = 90 episodes per setting, env seed 21, explore seed 0, raw saved.
# New settings: start noise x3, x5, x8 on half the chunks (p 0.5), and x2 on every chunk (p 1.0).
# 2 GPU lanes (Karen's cap: 6 GPUs incl. the interactive session; the step-1 follow-ups hold 3), 2 settings each
# (~35 min per setting), then a CPU analysis job -> reports/2026-11-09-noise-sweep/.
# Reading rule [set before the result, the pilot's rule unchanged]: the largest-spread setting whose success is at
# most 10 pts below control (point estimate) and whose early arm spread is >= 1.5x control qualifies. Settings
# 10-15 pts below are reported as borderline. A qualifying setting beyond x2.0 p0.5 -> with Karen's go, 450
# episodes at it -> table -> OOF ResNet V -> vb40 labels -> coach test, compared by action info (not ideal shift)
# with the x2.0 explore round; none -> more start noise does not add variety without breaking the episodes.
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
RUNS=/iris/u/ankile/karenvo/libero/runs
ST="0 1 2 3 4 6 8 10 11 16 18 19 20 21 22 23 24 26 27 28 30 34 35 36 37 38 43 45 46 47"
OUT=reports/2026-11-09-noise-sweep
run() {  # $1 dependency ("" = none), $2 name, $3 prob, $4 scale
    sbatch --parsable ${1:+-d $1} \
        --export=ALL,SUITE=libero_10,TASKS=8,REPEATS=3,SEED=21,BASE_OURS=1,NOISE_SEED=0,NAME=noise_sweep_$2,CLIENT_ARGS="--args.init-states $ST --args.save-raw --args.split-name noise_sweep --args.explore-prob $3 --args.explore-scale $4 --args.explore-seed 0" \
        scripts/libero_eval.sbatch
}
A1=$(run "" x3p0 0.5 3.0);            A2=$(run afterany:$A1 x8p0 0.5 8.0)
B1=$(run "" x5p0 0.5 5.0);            B2=$(run afterany:$B1 x2p0_all 1.0 2.0)
echo "lane A: x3.0 $A1 -> x8.0 $A2 | lane B: x5.0 $B1 -> x2.0 every chunk $B2"
d() { echo "$RUNS/noise_sweep_$1_libero_10_$2"; }
AN=$(sbatch --parsable -A iris -p iris -c 2 --mem 16G -t 1:00:00 -d afterany:$A2:$B2 \
    --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out \
    --wrap "cd $PWD && .venv/bin/python scripts/libero_explore_pilot.py --runs control=$RUNS/explore_pilot_control_libero_10_18388271 x2.0=$RUNS/explore_pilot_x2p0_libero_10_18388273 x3.0=$(d x3p0 $A1) x5.0=$(d x5p0 $B1) x8.0=$(d x8p0 $A2) x2.0_all=$(d x2p0_all $B2) --out $OUT")
echo "analysis $AN -> $OUT"
