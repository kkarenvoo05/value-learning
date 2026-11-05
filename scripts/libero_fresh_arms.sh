#!/bin/bash
# Fresh-layout arms (docs/libero_plan.md 2026-10-31): fresh-only (375 fresh) and fixed + fresh (375 + 450) for the
# ResNet V and pi05 arm B, 3 seeds each. One arm per call (Karen 2026-10-31: fresh first, 6 jobs; fixed+fresh later),
# as two chains = 2 GPUs (Karen: at most 3; another session may hold one):
#   ResNet chain: s0, s1, s2 on an A5000 / L40S / TITAN RTX (40 GB; ResNet ran ~1 h per seed on TITAN before)
#   expert chain: s0, s1, s2 on an A5000 / L40S (fixed+fresh: 96 GB, 8 h)
# The first fresh ResNet job builds the table (+ DONE marker) and the frame cache; the expert chain starts with it.
# Recipes unchanged from the half-data check (ResNet 8,000 steps; B = w256 / MLP 1024, aug, lr 1e-4, 5,400 steps).
# Usage: scripts/libero_fresh_arms.sh <fresh run dir> fresh|fixedfresh
set -euo pipefail
RUN=$1 ARM=$2
[[ $ARM =~ ^(fresh|fixedfresh)$ ]] || { echo "arm must be fresh or fixedfresh"; exit 1; }
cd /move/u/karenvo/Projects/value-learning
T=/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh
SUB=$T/subsets
B="--train --augment --width 256 --mlp-dim 1024 --lr 1e-4 --steps 5400 --dataset $T"
GPU="--gres=gpu:1 --exclude=move1,move2,move5"   # expert: A5000 / L40S
RGPU="--gres=gpu:1 --exclude=move5 --mem=40G"       # ResNet: also TITAN RTX
jobs=()
resnet() {  # arm seed dependency [extra env...]
    local fit=$SUB/fit_fresh375_s0.txt; [ $1 = fixedfresh ] && fit=$SUB/fit_fixed_fresh.txt
    env "${@:4}" EVERY=4 DATASET=$T FIT=$fit TAG=$1 SEED=$2 \
        sbatch --parsable --job-name=job $RGPU ${3:+--dependency=$3} scripts/libero_resnet.sbatch
}
expert() {  # arm seed dependency
    local fit=$SUB/fit_fresh375_s0.txt extra="--time=08:00:00"
    [ $1 = fixedfresh ] && { fit=$SUB/fit_fixed_fresh.txt; extra="--time=08:00:00 --mem=96G"; }
    WAIT_FOR=$T/DONE NAME=${1}_w256_s$2 ARGS="$B --seed $2 --fit-episodes $fit" \
        sbatch --parsable --job-name=job $GPU $extra ${3:+--dependency=$3} scripts/pi05_value_expert.sbatch
}
if [ $ARM = fresh ]; then   # builds the table first
    j=$(resnet fresh 0 "" PREP_FRESH_RUN=$RUN); PREP=$j; jobs+=("resnet fresh s0 (+table) $j")
    first=after:$PREP          # expert chain starts with the table job (WAIT_FOR covers the ~5 min build)
else
    [ -e $T/frames_every4.npy ] || { echo "run the fresh arm first (it builds $T)"; exit 1; }
    j=$(resnet fixedfresh 0 ""); jobs+=("resnet fixedfresh s0 $j"); first=""
fi
for s in 1 2; do j=$(resnet $ARM $s afterany:$j); jobs+=("resnet $ARM s$s $j"); done
j=$(expert $ARM 0 "$first"); jobs+=("expert $ARM s0 $j")
for s in 1 2; do j=$(expert $ARM $s afterany:$j); jobs+=("expert $ARM s$s $j"); done
printf '%s\n' "${jobs[@]}"
