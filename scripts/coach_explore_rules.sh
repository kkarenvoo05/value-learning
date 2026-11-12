#!/bin/bash
# Coach test on the explore table under 4 label rules [2026-11-08, Karen: go]. Same OOF ResNet V for all.
#   explore_vb40        value-binned, 40% over the pool           (ours; the rule of the policy runs)
#   explore_thr40       one threshold on A, 40% over the pool      (RECAP Eq. 5 rule)
#   explore_vb40_round  value-binned, 40% within each round        (RECAP per-round share)
#   explore_thr40_round one threshold, 40% within each round       (RECAP exact: rule + share)
# Run via: CMD="bash scripts/coach_explore_rules.sh" sbatch scripts/vl_python.sbatch
set -euo pipefail
T=/iris/u/ankile/karenvo/libero/tables/moka_explore
P='resnet18_explore_oof_f{k}_s0_*'
R=reports/2026-11-08-explore-pilot
PY=.venv/bin/python
$PY scripts/libero_td_labels.py oof --table $T --pattern "$P" --name explore_vb40
$PY scripts/libero_td_labels.py oof --table $T --pattern "$P" --name explore_thr40 --rule thr
$PY scripts/libero_td_labels.py oof --table $T --pattern "$P" --name explore_vb40_round --share-by round
$PY scripts/libero_td_labels.py oof --table $T --pattern "$P" --name explore_thr40_round --rule thr --share-by round
for L in explore_vb40 explore_thr40 explore_vb40_round explore_thr40_round; do
  O=$R/coach_explore; [ $L != explore_vb40 ] && O=$R/coach_${L}
  $PY scripts/libero_tag_ideal_shift.py --table $T --labels $L --layout fixed --group-col round --out $O
done
