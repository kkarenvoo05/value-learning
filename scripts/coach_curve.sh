#!/bin/bash
# Coach-test data curve [2026-11-08, Karen: build A]: does the label/action signal grow with more episodes?
# Usage: CMD="bash scripts/coach_curve.sh old|explore" sbatch scripts/vl_python.sbatch
set -euo pipefail
PY=.venv/bin/python; S=scripts/libero_tag_ideal_shift.py; R=reports/2026-11-08-coach-curve
case "$1" in
  old)  # 450 fixed-state base rollouts, the V and labels behind the policy runs
    $PY $S --table /iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh --labels resnet_valuebinned_p40 \
      --v-labels oof_resnet_n20_p30 --layout fixed --curve 56 112 225 450 --curve-seeds 5 --out $R/old_fixed ;;
  explore)  # after scripts/coach_explore_rules.sh: per round (old 450 vs explore 450) and pooled up to 900
    T=/iris/u/ankile/karenvo/libero/tables/moka_explore
    for L in explore_vb40 explore_thr40_round; do
      $PY $S --table $T --labels $L --layout fixed --group-col round --curve 56 112 225 450 --curve-seeds 5 \
        --out $R/explore_${L}_by_round
      $PY $S --table $T --labels $L --layout fixed --curve 112 225 450 900 --curve-seeds 5 --out $R/explore_${L}_pooled
    done ;;
  *) echo "usage: $0 old|explore"; exit 2 ;;
esac
