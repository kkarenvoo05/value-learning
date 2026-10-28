# Advantage labels from R0: do they make sense? (Stage 6 first step, val only)

`.venv/bin/python -m vl.advantages` → this file, `reports/figures/stage6/advantages/` (figures, `advantage_checks.csv`).
Definitions, controls and thresholds: the module docstring of `src/vl/advantages.py`. 247 val episodes, 64,906 frames (57,249 interior at n = 2 s). Intervals: episode bootstrap, 1000 draws. Arithmetic check passed: with the target as V every advantage is 0.

Columns: *pos* = share of frames labelled positive; *interior* = lookahead stays inside the episode (A depends on V only); *AUC within round* = episode-level success AUC of the interior positive share, inside each eval round (round-free), weighted mean; *speed ρ* = Spearman of success duration vs interior positive share inside each repo (negative = faster successes get more positives); *still* = hold phase, gripper still ≥ 4 s now (`vl.history_gain.hold_groups`): pause before a seat (success) vs stall (timeout); *moving succ* = success hold frames while moving; *flips/s* = label changes per second on interior frames.

## Threshold: 70% positive (ε = 30th percentile of val advantages)

| V | n (s) | ε | pos, terminal | pos succ | pos fail | pos timeout | succ − fail | 95% CI | AUC within round | speed ρ | still: succ | still: stall | succ − stall 95% CI | moving succ | flips/s | agree w/ R0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 1 | -0.009 | 0.895 | 0.765 | 0.526 | 0.558 | 0.215 | [+0.186, +0.243] | 0.878 | -0.371 | 0.677 | 0.557 | [+0.054, +0.206] | 0.642 | 0.837 |  |
| R0 | 2 | -0.016 | 0.890 | 0.781 | 0.472 | 0.495 | 0.293 | [+0.255, +0.332] | 0.910 | -0.442 | 0.706 | 0.515 | [+0.101, +0.307] | 0.682 | 0.617 |  |
| R0 | 5 | -0.030 | 0.883 | 0.803 | 0.318 | 0.346 | 0.463 | [+0.394, +0.530] | 0.879 | -0.515 | 0.686 | 0.350 | [+0.145, +0.550] | 0.724 | 0.378 |  |
| R0 single | 1 | -0.010 | 0.900 | 0.754 | 0.558 | 0.573 | 0.185 | [+0.157, +0.211] | 0.873 | -0.388 | 0.639 | 0.572 | [+0.012, +0.139] | 0.629 | 2.730 | 0.810 |
| R0 single | 2 | -0.016 | 0.895 | 0.765 | 0.509 | 0.517 | 0.250 | [+0.219, +0.279] | 0.904 | -0.455 | 0.656 | 0.520 | [+0.075, +0.222] | 0.673 | 2.102 | 0.836 |
| R0 single | 5 | -0.030 | 0.889 | 0.788 | 0.360 | 0.358 | 0.430 | [+0.366, +0.497] | 0.884 | -0.540 | 0.632 | 0.373 | [+0.111, +0.432] | 0.701 | 1.475 | 0.890 |
| C2a | 1 | -0.009 | 0.912 | 0.761 | 0.546 | 0.558 | 0.206 | [+0.176, +0.240] | 0.815 | -0.540 | 0.597 | 0.556 | [-0.022, +0.111] | 0.618 | 0.763 | 0.703 |
| C2a | 2 | -0.016 | 0.916 | 0.796 | 0.466 | 0.449 | 0.343 | [+0.302, +0.382] | 0.920 | -0.572 | 0.659 | 0.443 | [+0.136, +0.321] | 0.688 | 0.519 | 0.729 |
| C2a | 5 | -0.031 | 0.911 | 0.803 | 0.308 | 0.313 | 0.491 | [+0.428, +0.548] | 0.885 | -0.520 | 0.670 | 0.288 | [+0.249, +0.558] | 0.684 | 0.320 | 0.815 |
| time+round | 1 | -0.005 | 0.730 | 0.760 | 0.696 | 0.545 | 0.175 | [+0.130, +0.213] | 0.569 | -0.966 | 0.389 | 0.401 | [-0.132, +0.104] | 0.507 | 0.135 | 0.590 |
| time+round | 2 | -0.010 | 0.728 | 0.779 | 0.655 | 0.509 | 0.234 | [+0.179, +0.284] | 0.543 | -0.939 | 0.326 | 0.345 | [-0.159, +0.099] | 0.467 | 0.093 | 0.615 |
| time+round | 5 | -0.028 | 0.734 | 0.816 | 0.557 | 0.452 | 0.341 | [+0.277, +0.407] | 0.614 | -0.732 | 0.353 | 0.356 | [-0.207, +0.192] | 0.241 | 0.069 | 0.645 |

Mid-episode only (lookahead ends ≥ 5 s before T, so V never sees the final scene):

| V | n (s) | pos succ | pos fail+timeout | succ − fail 95% CI | AUC within round | still: succ | still: stall | succ − stall 95% CI | time-matched succ − fail | 95% CI | time-matched still succ − stall | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 1 | 0.716 | 0.551 | [+0.134, +0.197] | 0.665 | 0.633 | 0.564 | [+0.005, +0.176] | 0.032 | [+0.007, +0.055] | 0.078 | [+0.018, +0.168] |
| R0 | 2 | 0.738 | 0.496 | [+0.200, +0.285] | 0.808 | 0.645 | 0.530 | [+0.010, +0.251] | 0.061 | [+0.027, +0.093] | 0.127 | [+0.029, +0.252] |
| R0 | 5 | 0.757 | 0.335 | [+0.345, +0.500] | 0.788 | 0.627 | 0.336 | [+0.103, +0.584] | 0.137 | [+0.071, +0.199] | 0.246 | [+0.053, +0.427] |
| R0 single | 1 | 0.704 | 0.571 | [+0.106, +0.162] | 0.677 | 0.602 | 0.570 | [-0.024, +0.106] | 0.016 | [-0.004, +0.037] | 0.040 | [-0.005, +0.115] |
| R0 single | 2 | 0.717 | 0.519 | [+0.165, +0.233] | 0.753 | 0.599 | 0.526 | [-0.000, +0.191] | 0.040 | [+0.012, +0.066] | 0.081 | [+0.011, +0.192] |
| R0 single | 5 | 0.741 | 0.347 | [+0.320, +0.468] | 0.775 | 0.579 | 0.370 | [+0.033, +0.442] | 0.118 | [+0.057, +0.177] | 0.189 | [-0.001, +0.378] |
| C2a | 1 | 0.717 | 0.558 | [+0.122, +0.193] | 0.671 | 0.543 | 0.557 | [-0.075, +0.048] | 0.019 | [-0.008, +0.045] | 0.022 | [-0.050, +0.109] |
| C2a | 2 | 0.755 | 0.452 | [+0.256, +0.351] | 0.819 | 0.542 | 0.433 | [+0.024, +0.207] | 0.075 | [+0.039, +0.112] | 0.131 | [+0.065, +0.207] |
| C2a | 5 | 0.736 | 0.304 | [+0.355, +0.509] | 0.816 | 0.549 | 0.280 | [+0.123, +0.482] | 0.112 | [+0.047, +0.173] | 0.201 | [+0.052, +0.344] |
| time+round | 1 | 0.851 | 0.603 | [+0.187, +0.305] | 0.696 | 0.405 | 0.428 | [-0.184, +0.141] | +0.00000 | [+0.000, +0.000] | +0.00000 | [+0.000, +0.000] |
| time+round | 2 | 0.848 | 0.574 | [+0.202, +0.338] | 0.719 | 0.348 | 0.384 | [-0.245, +0.154] | +0.00000 | [+0.000, +0.000] | +0.00000 | [+0.000, +0.000] |
| time+round | 5 | 0.827 | 0.511 | [+0.232, +0.405] | 0.748 | 0.428 | 0.410 | [-0.327, +0.329] | +0.00000 | [+0.000, +0.000] | +0.00000 | [+0.000, +0.000] |

## Threshold: 30% positive (ε = 70th percentile of val advantages)

| V | n (s) | ε | pos, terminal | pos succ | pos fail | pos timeout | succ − fail | 95% CI | AUC within round | speed ρ | still: succ | still: stall | succ − stall 95% CI | moving succ | flips/s | agree w/ R0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 1 | 0.007 | 0.277 | 0.339 | 0.232 | 0.236 | 0.104 | [+0.060, +0.144] | 0.905 | -0.436 | 0.319 | 0.171 | [+0.073, +0.216] | 0.519 | 0.705 |  |
| R0 | 2 | 0.009 | 0.374 | 0.346 | 0.217 | 0.185 | 0.153 | [+0.107, +0.192] | 0.892 | -0.427 | 0.323 | 0.096 | [+0.145, +0.315] | 0.556 | 0.487 |  |
| R0 | 5 | 0.016 | 0.402 | 0.339 | 0.163 | 0.120 | 0.210 | [+0.153, +0.264] | 0.878 | -0.372 | 0.340 | 0.048 | [+0.153, +0.504] | 0.560 | 0.279 |  |
| R0 single | 1 | 0.008 | 0.261 | 0.324 | 0.269 | 0.264 | 0.058 | [+0.012, +0.101] | 0.865 | -0.406 | 0.349 | 0.223 | [+0.039, +0.207] | 0.491 | 2.291 | 0.842 |
| R0 single | 2 | 0.011 | 0.362 | 0.332 | 0.237 | 0.216 | 0.111 | [+0.065, +0.156] | 0.882 | -0.464 | 0.367 | 0.172 | [+0.108, +0.281] | 0.536 | 1.709 | 0.868 |
| R0 single | 5 | 0.018 | 0.405 | 0.320 | 0.187 | 0.148 | 0.163 | [+0.107, +0.214] | 0.872 | -0.367 | 0.361 | 0.102 | [+0.153, +0.404] | 0.518 | 1.251 | 0.906 |
| C2a | 1 | 0.007 | 0.293 | 0.341 | 0.231 | 0.229 | 0.111 | [+0.072, +0.150] | 0.942 | -0.367 | 0.388 | 0.152 | [+0.179, +0.308] | 0.514 | 0.682 | 0.746 |
| C2a | 2 | 0.010 | 0.423 | 0.337 | 0.199 | 0.188 | 0.147 | [+0.101, +0.191] | 0.919 | -0.390 | 0.411 | 0.115 | [+0.203, +0.405] | 0.563 | 0.437 | 0.776 |
| C2a | 5 | 0.019 | 0.432 | 0.308 | 0.180 | 0.139 | 0.160 | [+0.098, +0.215] | 0.888 | -0.274 | 0.432 | 0.066 | [+0.222, +0.573] | 0.560 | 0.290 | 0.848 |
| time+round | 1 | +0.00000 | 0.725 | 0.179 | 0.468 | 0.358 | -0.207 | [-0.257, -0.161] | 0.611 | -0.447 | 0.153 | 0.194 | [-0.137, +0.048] | 0.421 | 0.143 | 0.629 |
| time+round | 2 | +0.00014 | 0.696 | 0.186 | 0.443 | 0.316 | -0.162 | [-0.211, -0.115] | 0.634 | -0.640 | 0.122 | 0.132 | [-0.098, +0.078] | 0.296 | 0.100 | 0.646 |
| time+round | 5 | 0.002 | 0.591 | 0.148 | 0.340 | 0.198 | -0.081 | [-0.133, -0.027] | 0.532 | -0.494 | 0.074 | 0.066 | [-0.059, +0.071] | 0.008 | 0.073 | 0.699 |

Mid-episode only (lookahead ends ≥ 5 s before T, so V never sees the final scene):

| V | n (s) | pos succ | pos fail+timeout | succ − fail 95% CI | AUC within round | still: succ | still: stall | succ − stall 95% CI | time-matched succ − fail | 95% CI | time-matched still succ − stall | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 1 | 0.326 | 0.241 | [+0.038, +0.131] | 0.780 | 0.295 | 0.169 | [+0.019, +0.213] | 0.069 | [+0.034, +0.103] | 0.118 | [+0.022, +0.199] |
| R0 | 2 | 0.327 | 0.199 | [+0.084, +0.176] | 0.816 | 0.257 | 0.091 | [+0.034, +0.314] | 0.084 | [+0.044, +0.122] | 0.141 | [+0.007, +0.261] |
| R0 | 5 | 0.326 | 0.133 | [+0.121, +0.253] | 0.768 | 0.280 | 0.032 | [+0.060, +0.619] | 0.112 | [+0.051, +0.173] | 0.204 | [+0.019, +0.547] |
| R0 single | 1 | 0.320 | 0.274 | [-0.004, +0.093] | 0.774 | 0.341 | 0.235 | [-0.008, +0.195] | 0.058 | [+0.023, +0.091] | 0.106 | [+0.008, +0.188] |
| R0 single | 2 | 0.320 | 0.227 | [+0.034, +0.146] | 0.811 | 0.342 | 0.174 | [+0.048, +0.288] | 0.077 | [+0.031, +0.119] | 0.160 | [+0.021, +0.265] |
| R0 single | 5 | 0.316 | 0.157 | [+0.090, +0.223] | 0.777 | 0.333 | 0.090 | [+0.078, +0.517] | 0.107 | [+0.042, +0.164] | 0.219 | [+0.019, +0.424] |
| C2a | 1 | 0.320 | 0.237 | [+0.035, +0.125] | 0.820 | 0.300 | 0.155 | [+0.057, +0.217] | 0.068 | [+0.029, +0.102] | 0.121 | [+0.023, +0.199] |
| C2a | 2 | 0.307 | 0.195 | [+0.054, +0.159] | 0.823 | 0.247 | 0.105 | [+0.052, +0.219] | 0.076 | [+0.032, +0.117] | 0.118 | [+0.019, +0.196] |
| C2a | 5 | 0.283 | 0.152 | [+0.061, +0.199] | 0.790 | 0.263 | 0.056 | [+0.087, +0.454] | 0.082 | [+0.018, +0.137] | 0.143 | [+0.029, +0.272] |
| time+round | 1 | 0.206 | 0.400 | [-0.259, -0.134] | 0.754 | 0.178 | 0.204 | [-0.161, +0.113] | +0.00000 | [+0.000, +0.000] | +0.00000 | [+0.000, +0.000] |
| time+round | 2 | 0.225 | 0.368 | [-0.209, -0.083] | 0.747 | 0.182 | 0.146 | [-0.104, +0.168] | +0.00000 | [+0.000, +0.000] | +0.00000 | [+0.000, +0.000] |
| time+round | 5 | 0.188 | 0.244 | [-0.138, +0.016] | 0.625 | 0.125 | 0.064 | [-0.024, +0.233] | +0.00000 | [+0.000, +0.000] | +0.00000 | [+0.000, +0.000] |
## Reading (Claude, 2026-10-26, to be checked by Karen)

Main setting below: n = 2 s, threshold at the 30th percentile (70% positive), R0 = 3-seed ensemble + centred
0.5 s. Figures: `01_distributions.png`, `02_checks.png`, `03_examples.png`, `04_time_to_end.png`.

1. **The raw success-vs-failure split is mostly not R0's doing.** Interior frames of successes are 78% positive
   vs 47% for failures and 50% for timeouts, which looks good, but the time+round control (no image) gets almost
   the same split (78% vs 65% / 51%). Failures run longer, and later seconds get lower advantages from the clock
   alone. Two other sources also inflate it: in the last n seconds the known outcome enters the advantage directly
   (terminal frames: 89% positive; `04_time_to_end.png`), and the last frames
   *show* the outcome (a seated marker). Neither needs a good V.
2. **After removing those, R0's labels still carry real information, but a small amount.** Mid-episode (lookahead
   ends ≥ 5 s before T) and time-matched (each label minus the mean label of frames at the same second of the same
   round), successes get **+6.1 pp more positives [+2.7, +9.3]** at n = 2 s, +3.2 [+0.7, +5.6] at 1 s,
   +13.7 [+7.1, +19.9] at 5 s. The control is 0 here by construction. Same picture at the 30%-positive threshold
   (+8.4 [+4.4, +12.2] at 2 s). So roughly a quarter of the raw split survives.
3. **Stalls vs pauses before a seat: R0 gets the direction right.** With the gripper still ≥ 4 s at the holder,
   pause-before-seat frames get +12.7 pp more positives than stall frames [+2.9, +25.2] time-matched (n = 2 s),
   +24.6 [+5.3, +42.7] at 5 s; the control has no signal. Small sample: 23 stall vs 13 frozen-success episodes.
   This is what Lars's throughput argument needs (a stall should not look like progress), but V's stall/pause
   ambiguity from Track B is still there: it is a lean, not a separation.
4. **Labels flicker.** R0 flips label 0.62 times per second on interior frames at n = 2 s (one change every ~1.6 s);
   the unsmoothed single model flips 2.1/s, so the ensemble + smoothing recipe matters 3.4× here. At n = 5 s it is
   0.38/s. `03_examples.png` shows why: in r04 ep 39, a clean success, V wanders ±0.1 during the approach and
   58% of its first 8 s are labelled negative. This is the shape error (0.044 vs 0.037 flat) turning into labels.
5. **Speed check is not diagnostic.** Faster successes get more positives (Spearman −0.44 within repo), but the
   control does it more strongly (−0.94), from the clock alone. A cleaner test of "advantage rewards speed" needs
   a time-matched comparison of fast vs slow successes, not done here.
6. **Failure endings are mostly labelled positive (caution for conditioning).** In the last n = 2 s, success
   frames are 100% positive, but so are **72% of failure and 57% of timeout frames**. A terminal failure frame gets
   Â = −(T−t)/1800 − 0.5 − V(o_t), which is positive whenever V had already dropped below about −0.5: the failure
   was "no worse than expected". That is correct RECAP arithmetic (advantage is relative to V), but a policy
   conditioned on "positive" would be trained on the final actions of failed episodes. Options [ours, not tested]:
   force terminal failure frames negative, or drop terminal frames from the positive set.
7. **Longer lookahead is better on every check here:** stronger time-matched signal, stronger stall separation,
   fewer flips. The cost is coarser credit (a 5 s window covers most of an approach). R0 vs the control agree on
   ~62% of interior labels at 2 s, so the labels are substantially different from a clock rule.

8. **C2a as V (added 2026-10-27; ensemble ×3 of `c2a_stack2` + centred 0.5 s): stalls get fewer positives, but
   so do pauses before a seat, so separation barely moves.** Mid-episode at n = 2 s, still-stall frames drop from
   53% positive (R0) to 43%, and still pause-before-seat frames from 65% to 54% (C2a's known frozen-success cost).
   Time-matched pause-vs-stall: +13.1 pp [+6.5, +20.7] vs R0's +12.7 [+2.9, +25.2] (same size, tighter interval);
   successes vs failures +7.5 [+3.9, +11.2] vs +6.1. At n = 5 s C2a is a little *lower* (+11.2 / +20.1 vs +13.7 /
   +24.6; overlapping intervals). Flips 0.52/s vs 0.62/s at 2 s. C2a agrees with R0 on 73% of interior labels at
   2 s. Reading: C2a's CE and stall_auc gain makes stall labels more negative, not better separated from paused
   successes. It is no reason to prefer or reject C2a for advantages; n = 2 s is now about as good as 5 s.

**Verdict.** The labels make sense in direction on both checks and are not a clock artefact, but the per-frame
signal is weak and noisy: a few percentage points of time-matched separation at 1–2 s. Proposed first choice for
advantage conditioning [ours]: **n = 5 s with the offline recipe (ensemble ×3 + centred 0.5 s)**, n = 2 s as the
comparison arm, and a decision on failure-ending frames (item 6) before any policy is trained. Threshold reading (70% vs 30% positive) does not change the verdict; RECAP's literal "30th
percentile" (70% positive) is the default.

**What this does not show.** Whether conditioning on these labels improves a policy (needs robot rollouts); and
these are val episodes only. Train-set labels need out-of-fold R0 (k models on disjoint train folds) after the
freeze, or R0's fit makes train advantages ~0.
