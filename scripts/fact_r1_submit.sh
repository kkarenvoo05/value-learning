#!/bin/bash
# [2026-11-21] Round-1 factorial for the small-chain drop (reports/2026-11-21-r1-factorial; Karen: "run the free label
# check first, then build arms A-D"). Every arm = a round-1 retrain on a subset of the 900 chain's round-7 table
# tables/moka_agr_r7 (900 base pi0.5 rollouts, round 1, + 100 guided rollouts from each of the chain's r0..r6
# policies, rounds 2..8), with its frozen-900-V labels agr_vb40_r7 (thresholded 40% per round, so a subset keeps
# every episode's label; r1 vs r7 label files agree on 99.3% of shared frames). No --demos (demos hurt: nd900 / ndb50).
# Episode lists: tables/moka_agr_r7/rebal/fact_*.txt, made by rebal/fact_lists.py (rng 20261121; fact_lists.json):
#   A    100 base + 100 guided from r0 only (49% guided frames)              -> does the drop reproduce from the mix alone?
#   B    100 base + 100 guided, 14-15 from each of r0..r6 (48%)              -> A vs B: one policy vs many (explanation 1)
#   C    700 base + all 700 guided (48%)                                    -> B vs C: more base + more data, same share
#   D    100 base repeated 7x (52.2% of frames, as C's 52.4%) + all 700 guided (616k vs C's 619k frames)
#                                                                            -> C vs D: distinct base episodes (2) vs volume (3)
#   R100 / R700  base only (r0 controls for A/B/D and C)
# base100 = b100 chain's 100 r0 base episodes (fresh layouts), so A = b100 r1 with the 900 chain's r0 as the guided
# policy. Train args otherwise = the chains' (5000 steps, lr 3e-4, accum 2, freeze SigLIP); check on the r7 table (its
# 120 base val episodes, as every moka table); evals at push 0.99: moka val 330 (matched noise 1) + moka cont 300.
# Usage: ARMS="A B C D" SEEDS="0 1" scripts/fact_r1_submit.sh            (dry run)
#        SUBMIT=1 ARMS=... SEEDS=... scripts/fact_r1_submit.sh           (submits held, gpu_cap releases; logs/fact_r1.log)
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
ROOT=/iris/u/ankile/karenvo/libero R=$ROOT/runs TB=$ROOT/tables ES=$ROOT/eval_sets
EX="--hold -A iris -p ${PART:-iris-hi} --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8"   # iris7 hung 11-20
T7=$TB/moka_agr_r7 L7=$TB/moka_agr_r7/labels/agr_vb40_r7/labels.parquet RB=$TB/moka_agr_r7/rebal
for f in $L7 $ES/moka_cont100/starts.csv $RB/fact_base100.txt; do [ -e $f ] || { echo "missing $f"; exit 1; }; done
N=0
sub() {  # sub <var> <dependency or -> <sbatch args...>
    local var=$1 dep=$2; shift 2; N=$((N + 1))
    if [ -n "${SUBMIT:-}" ]; then
        local j; j=$(sbatch --parsable ${dep:+$([ "$dep" != - ] && echo "-d $dep")} "$@"); eval "$var=$j"
        echo "step $N $var: $j ($JP)" | tee -a logs/fact_r1.log
    else
        eval "$var=J$N"; echo "step $N $var (dep $dep): sbatch $*"
    fi
}
[ -n "${SUBMIT:-}" ] && echo "$(date '+%F %T') fact_r1 submit ARMS=${ARMS:?set ARMS} SEEDS=${SEEDS:?set SEEDS}" >> logs/fact_r1.log
for seed in ${SEEDS:?set SEEDS}; do for arm in ${ARMS:?set ARMS (A B C D R100 R700)}; do
    [ -e $RB/fact_$arm.txt ] || { echo "missing $RB/fact_$arm.txt"; exit 1; }
    JP=agr_fact${arm}s$seed NAME=agr_fact${arm}_r1_s$seed
    OV=""; [ $arm = D ] && OV="--oversample-episodes $RB/fact_base100.txt --oversample-share 0.5236"
    sub T - $EX --mem=56G --time=7:30:00 --job-name=${JP}_train --export=ALL,MODE=train,PY_SCRIPT=pi05_ag_residual.py,NAME=$NAME,ARGS="--train --arm advantage --labels $L7 --table $T7 --episodes $RB/fact_$arm.txt --tag-dropout 0 --freeze-siglip --lr 3e-4 --accum 2 --steps 5000 --save-every 1000 --seed $seed $OV" scripts/pi05_tag_adarms.sbatch
    RUN=$R/policy_${NAME}_$T CHK=$R/contrast_check_$NAME
    sub K afterok:$T $EX --mem=40G --time=2:00:00 --job-name=${JP}_check --export=ALL,MODE=contrastcheck,OUT=$CHK,ARGS="--run $RUN --weights 1 2 4 8 16 --table $T7" scripts/pi05_tag_adarms.sbatch
    POL="MODE=eval,NOISE_SEED=1,PY_SCRIPT=pi05_ag_residual.py,FT_RUN=$RUN,FT_SHIFT_TARGET=0.99,FT_SHIFT_FROM=$CHK/results.json"
    sub E afterok:$K $EX --mem=40G --time=5:00:00 --job-name=${JP}_eval --export=ALL,$POL,MATCHED_NOISE=1,NAME=eval_${NAME}_shift099_noise1 scripts/pi05_tag_adarms.sbatch
    sub C afterok:$K $EX --mem=40G --time=5:00:00 --job-name=${JP}_ceval --export=ALL,$POL,NAME=ceval_${NAME}_shift099,CLIENT_ARGS="--args.start-from $ES/moka_cont100/starts.csv --args.split-name cont" scripts/pi05_tag_adarms.sbatch
done; done
