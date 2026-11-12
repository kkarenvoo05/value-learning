# Learned values and labels vs simulator ground truth (mentor item 4)

**Question.** Is the value function's advantage informative about actions? Score every value / label source against
the branching ground truth, with the ground truth's own noise ceiling.

**Setup.** `scripts/libero_value_vs_gt.py` (CPU, inference only). 18 train episodes from 4 branching runs
(branch_value 18355241: 6 episodes, K = 6; branch_fail a/b/c 18395412-14: 12 failed episodes, K = 12), 442 branch
points, 424 with A*. V* = P(success | s_t), G* = MC target in the ResNet's units, A* = n = 20 advantage on G*
(definitions as `scripts/libero_branch_analyze.py`). CIs: bootstrap over episodes. Ceiling [ours]: split-half r →
Spearman-Brown R = 2r/(1+r) for the full-K estimate → max observable corr √R.

**Ceiling.** V*: r 0.88 → R 0.94, ceiling **0.97**. A*: r 0.31 → R 0.48, ceiling **0.69** (√r = 0.56 would be the
ceiling for a half-K estimate; the plan's 0.57 used that). By run: fail_a 0.02, fail_b 0.43, fail_c 0.47, value 0.30.

| source | corr(V, V*) | corr(A, A*) | sign agree | label corr(A*) | + at A*>0 / <0 |
|---|---|---|---|---|---|
| ResNet-18 V, 5-fold OOF (current labels' V) | +0.22 [-0.04, +0.41] | +0.07 [+0.02, +0.12] | +0.54 [+0.50, +0.58] | +0.09 [-0.00, +0.17] | 0.70 / 0.69 |
| ResNet V ensemble | +0.28 [+0.01, +0.45] | +0.06 [-0.02, +0.16] | +0.54 [+0.52, +0.57] | +0.06 [-0.02, +0.15] | 0.37 / 0.29 |
| pi0.5 value expert, OOF | +0.29 [+0.04, +0.49] | +0.04 [-0.06, +0.17] | +0.53 [+0.50, +0.56] | -0.00 [-0.09, +0.10] | 0.66 / 0.66 |
| fitted n-step TD critic | +0.23 [+0.00, +0.40] | +0.04 [-0.02, +0.10] | +0.52 [+0.49, +0.55] | +0.03 [-0.07, +0.13] | 0.38 / 0.34 |
| TD critic ensemble | +0.28 [+0.04, +0.46] | +0.05 [-0.05, +0.16] | +0.54 [+0.51, +0.57] | +0.02 [-0.06, +0.11] | 0.39 / 0.35 |
| ResNet V, lookahead n = 80 | +0.22 [-0.04, +0.41] | +0.14 [+0.06, +0.23] | +0.54 [+0.50, +0.58] | +0.16 [+0.09, +0.25] | 0.34 / 0.26 |
| ResNet V, lookahead n = 160 | +0.22 [-0.04, +0.41] | +0.08 [+0.01, +0.18] | +0.53 [+0.49, +0.56] | +0.05 [-0.04, +0.14] | 0.26 / 0.23 |
| realized return - V (hindsight) | +0.22 [-0.04, +0.41] | +0.08 [+0.01, +0.14] | +0.52 [+0.49, +0.54] | +0.05 [-0.05, +0.13] | 0.10 / 0.09 |
| faithful RECAP labels (vb 40%) |  |  |  | +0.10 [+0.02, +0.20] | 0.40 / 0.29 |
| subtask-scoped vb 40% |  |  |  | +0.05 [-0.02, +0.10] | 0.42 / 0.32 |
| random labels (control) |  |  |  | +0.01 [-0.08, +0.09] | 0.48 / 0.50 |

**Observations.**
- Every learned advantage correlates with A* at 0.04–0.14 against a ceiling of 0.69, i.e. it recovers at most
  ~20% of the correlation a perfect predictor could show. Sign agreement 0.52–0.54 (chance 0.50).
- Best: lookahead n = 80 (A 0.14 [0.06, 0.23], labels 0.16 [0.09, 0.25]). The current ResNet labels (vb 40%):
  0.10 [0.02, 0.20]; π₀.₅ expert and TD critic ≈ 0 (CIs cover 0); random-label control 0.01.
- V vs V*: 0.22–0.29 against a ceiling of 0.97, ensembles and the π₀.₅ expert at the top (0.28–0.29).

**Interpretation (to check).** The ground truth is noisy but not the bottleneck: A* is reliable enough (ceiling 0.69)
that a good advantage would show corr 0.4+; ours show ≤ 0.14. So the labels are weak because the value functions
do not resolve action-level differences, not only because the truth is noisy. This answers the mentor's "is the advantage
informative?": barely, about actions.

**Caveats.** 12 of 18 episodes are failures chosen for branching, so V* has a restricted range (lowers V's corr);
fail_a's A* is nearly pure noise (r 0.02). K = 6 in branch_value. One n (20); n = 80 sources are compared with the
n = 20 A*. Data: `summary.json`, `points.csv`, `table.md`.
