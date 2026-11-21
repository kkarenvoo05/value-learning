#!/bin/bash
# [2026-11-20] Structured-offset G0 (reports/2026-11-20-offset-g0/report.md; Karen: "set up the structured-offset G0
# probe in sim"). Base pi05_libero; candidates = the same chunk + end-effector offsets (libero_steer_probe.py offset
# mode); simulator branching as G0 on the 10 val init states.
#   smoke  val state 9, 120 steps, none z+ z-, hold 4, branch 40, R 1 (real-server path, achieved displacement)
#   h1     afterok smoke: offset on one chunk (one decision, as G0); 20 base episodes x 4 branch points, 8 cands, R 4
#   h4     afterok smoke: offset held 4 chunks (closed loop)
# Usage: scripts/offset_g0_submit.sh                       (dry run)
#        SUBMIT=1 [PART=iris-hi] scripts/offset_g0_submit.sh  (submits; log logs/offset_g0.log)
# Jobs are held and released by gpu_cap (scripts/gpu_cap.sh) like every other GPU job.
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
EX="-A iris -p ${PART:-iris-hi} --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8"   # iris7 hung 11-20
OFFS="none x+ x- y+ y- z+ z- z+0.5"
FULL="--args.offset-m 0.02 --args.branch-steps 40 130 220 310 --args.repeats 4 --args.base-repeats 2"
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j" | tee -a logs/offset_g0.log
    else
        eval "$var=J$N"; echo "step $N $var (dep $dep): sbatch $*"
    fi
}
[ -n "${SUBMIT:-}" ] && echo "$(date '+%F %T') offset_g0 submit" >> logs/offset_g0.log
sub S - $EX --mem=48G --time=0:45:00 --job-name=offg0_smoke \
    --export=ALL,NAME=offset_g0_smoke,INIT=9,CLIENT_ARGS="--args.cand-offsets none z+ z- --args.hold-chunks 4 --args.branch-steps 40 --args.repeats 1 --args.max-steps-override 120" \
    scripts/libero_steer_probe.sbatch
for H in 1 4; do
    sub P afterok:$S $EX --mem=48G --time=16:00:00 --job-name=offg0_h$H \
        --export=ALL,NAME=offset_g0_h$H,CLIENT_ARGS="--args.cand-offsets $OFFS --args.hold-chunks $H $FULL" \
        scripts/libero_steer_probe.sbatch
done
