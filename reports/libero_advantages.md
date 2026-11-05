# LIBERO advantage labels from the ResNet and π₀.₅ (3b arm B) V: do they make sense? (Stage 5 first step, val only)

`.venv/bin/python -m vl.libero_advantages` → this file, `reports/figures/libero_advantages/` (figures, `advantage_checks.csv`, `val_labels.parquet`). Definitions, controls and thresholds: the module docstring of `src/vl/libero_advantages.py`. 120 val episodes (63 successes), 13,944 frames (every 4th), 10 init states. Intervals: scene bootstrap, 1000 draws. Arithmetic check passed: with the target as V every advantage is 0.

Failure types (val): stuck_pot2 11, stalled_earlier 11, stuck_placing_pot2 9, knocked_over 7, near_miss 7, slow 7, stuck_pot1 5.

Columns: *terminal* = lookahead reaches the end, so the known outcome enters; *pos succ / fail* = share positive on interior frames (A depends on V only); *AUC within scene* = episode-level success AUC of the interior positive share inside each init state with both outcomes, weighted mean; *flips/s* = label changes per second of interior frames.

## Threshold: 70% positive (ε = 30th percentile of val advantages)

| V | n (s) | ε | terminal: succ | terminal: fail | pos succ | pos fail | succ − fail | 95% CI | AUC within scene | flips/s | agree w/ ResNet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet | 1.200 | -0.018 | 1.000 | 0.614 | 0.715 | 0.680 | 0.035 | [-0.009, +0.078] | 0.724 | 0.600 |  |
| ResNet | 2.000 | -0.024 | 1.000 | 0.607 | 0.743 | 0.653 | 0.090 | [+0.054, +0.129] | 0.787 | 0.484 |  |
| ResNet | 4.800 | -0.039 | 1.000 | 0.614 | 0.822 | 0.570 | 0.252 | [+0.204, +0.293] | 0.938 | 0.282 |  |
| ResNet s0 | 1.200 | -0.021 | 1.000 | 0.585 | 0.706 | 0.689 | 0.017 | [-0.024, +0.065] | 0.668 | 0.585 | 0.872 |
| ResNet s0 | 2.000 | -0.026 | 1.000 | 0.575 | 0.728 | 0.667 | 0.061 | [+0.023, +0.109] | 0.768 | 0.492 | 0.878 |
| ResNet s0 | 4.800 | -0.041 | 1.000 | 0.570 | 0.805 | 0.588 | 0.217 | [+0.180, +0.252] | 0.961 | 0.298 | 0.913 |
| ResNet+2s | 1.200 | -0.019 | 1.000 | 0.690 | 0.719 | 0.675 | 0.044 | [+0.007, +0.074] | 0.661 | 0.630 | 0.679 |
| ResNet+2s | 2.000 | -0.024 | 1.000 | 0.688 | 0.751 | 0.642 | 0.108 | [+0.076, +0.139] | 0.740 | 0.487 | 0.693 |
| ResNet+2s | 4.800 | -0.042 | 1.000 | 0.697 | 0.818 | 0.565 | 0.253 | [+0.205, +0.291] | 0.944 | 0.320 | 0.792 |
| B | 1.200 | -0.015 | 1.000 | 0.719 | 0.686 | 0.703 | -0.017 | [-0.054, +0.027] | 0.556 | 0.522 | 0.632 |
| B | 2.000 | -0.020 | 1.000 | 0.716 | 0.690 | 0.694 | -0.005 | [-0.054, +0.052] | 0.629 | 0.436 | 0.637 |
| B | 4.800 | -0.035 | 1.000 | 0.694 | 0.730 | 0.640 | 0.089 | [+0.010, +0.167] | 0.867 | 0.306 | 0.672 |
| B s0 | 1.200 | -0.014 | 1.000 | 0.754 | 0.693 | 0.696 | -0.003 | [-0.051, +0.048] | 0.571 | 0.536 | 0.628 |
| B s0 | 2.000 | -0.019 | 1.000 | 0.747 | 0.707 | 0.678 | 0.028 | [-0.017, +0.073] | 0.710 | 0.474 | 0.627 |
| B s0 | 4.800 | -0.032 | 1.000 | 0.722 | 0.752 | 0.619 | 0.133 | [+0.067, +0.194] | 0.886 | 0.317 | 0.667 |
| ResNet ff | 1.200 | -0.017 | 1.000 | 0.637 | 0.716 | 0.679 | 0.038 | [+0.000, +0.070] | 0.639 | 0.640 | 0.759 |
| ResNet ff | 2.000 | -0.022 | 1.000 | 0.642 | 0.745 | 0.649 | 0.096 | [+0.059, +0.124] | 0.800 | 0.520 | 0.771 |
| ResNet ff | 4.800 | -0.035 | 1.000 | 0.670 | 0.825 | 0.562 | 0.263 | [+0.229, +0.287] | 0.938 | 0.327 | 0.821 |
| B ff | 1.200 | -0.015 | 1.000 | 0.830 | 0.692 | 0.696 | -0.004 | [-0.054, +0.046] | 0.475 | 0.561 | 0.630 |
| B ff | 2.000 | -0.020 | 1.000 | 0.818 | 0.700 | 0.682 | 0.018 | [-0.032, +0.075] | 0.606 | 0.450 | 0.628 |
| B ff | 4.800 | -0.034 | 1.000 | 0.823 | 0.750 | 0.610 | 0.140 | [+0.069, +0.200] | 0.828 | 0.324 | 0.678 |
| clock | 1.200 | -0.00000 | 1.000 | +0.00000 | 0.637 | 0.504 | 0.133 | [+0.109, +0.149] | 1.000 | 0.290 | 0.521 |
| clock | 2.000 | -0.00000 | 1.000 | +0.00000 | 0.640 | 0.504 | 0.136 | [+0.112, +0.153] | 1.000 | 0.159 | 0.503 |
| clock | 4.800 | -0.00069 | 1.000 | +0.00000 | 0.809 | 0.627 | 0.182 | [+0.150, +0.205] | 1.000 | 0.024 | 0.646 |

Mid-episode only (lookahead ends ≥ 5 s before T, so V never sees the final scene):

| V | n (s) | pos succ | pos fail | succ − fail 95% CI | AUC within scene | time-matched succ − fail | 95% CI | time-matched AUC within scene |
|---|---|---|---|---|---|---|---|---|
| ResNet | 1.200 | 0.723 | 0.674 | [+0.008, +0.090] | 0.726 | 0.039 | [+0.014, +0.071] | 0.736 |
| ResNet | 2.000 | 0.750 | 0.649 | [+0.070, +0.133] | 0.794 | 0.054 | [+0.023, +0.086] | 0.738 |
| ResNet | 4.800 | 0.828 | 0.585 | [+0.195, +0.285] | 0.913 | 0.086 | [+0.050, +0.125] | 0.807 |
| ResNet s0 | 1.200 | 0.720 | 0.682 | [-0.002, +0.083] | 0.685 | 0.035 | [+0.007, +0.062] | 0.680 |
| ResNet s0 | 2.000 | 0.742 | 0.664 | [+0.038, +0.122] | 0.772 | 0.048 | [+0.018, +0.081] | 0.724 |
| ResNet s0 | 4.800 | 0.815 | 0.604 | [+0.175, +0.248] | 0.950 | 0.073 | [+0.042, +0.107] | 0.823 |
| ResNet+2s | 1.200 | 0.714 | 0.659 | [+0.018, +0.086] | 0.653 | 0.040 | [+0.015, +0.069] | 0.686 |
| ResNet+2s | 2.000 | 0.747 | 0.627 | [+0.082, +0.150] | 0.700 | 0.059 | [+0.023, +0.099] | 0.703 |
| ResNet+2s | 4.800 | 0.809 | 0.568 | [+0.195, +0.278] | 0.896 | 0.088 | [+0.048, +0.138] | 0.832 |
| B | 1.200 | 0.675 | 0.696 | [-0.067, +0.032] | 0.549 | 0.013 | [-0.014, +0.041] | 0.550 |
| B | 2.000 | 0.674 | 0.690 | [-0.071, +0.046] | 0.613 | 0.024 | [-0.010, +0.061] | 0.570 |
| B | 4.800 | 0.705 | 0.656 | [-0.024, +0.122] | 0.770 | 0.053 | [+0.014, +0.105] | 0.710 |
| B s0 | 1.200 | 0.669 | 0.688 | [-0.069, +0.034] | 0.489 | 0.010 | [-0.010, +0.030] | 0.509 |
| B s0 | 2.000 | 0.678 | 0.676 | [-0.051, +0.059] | 0.568 | 0.016 | [-0.011, +0.043] | 0.553 |
| B s0 | 4.800 | 0.720 | 0.639 | [+0.016, +0.143] | 0.767 | 0.045 | [+0.016, +0.082] | 0.715 |
| ResNet ff | 1.200 | 0.709 | 0.672 | [-0.002, +0.074] | 0.603 | 0.027 | [+0.004, +0.051] | 0.592 |
| ResNet ff | 2.000 | 0.738 | 0.647 | [+0.052, +0.124] | 0.726 | 0.042 | [+0.017, +0.071] | 0.703 |
| ResNet ff | 4.800 | 0.824 | 0.578 | [+0.211, +0.272] | 0.908 | 0.081 | [+0.045, +0.125] | 0.802 |
| B ff | 1.200 | 0.687 | 0.686 | [-0.050, +0.059] | 0.537 | 0.024 | [-0.003, +0.053] | 0.508 |
| B ff | 2.000 | 0.696 | 0.676 | [-0.036, +0.086] | 0.641 | 0.033 | [+0.001, +0.070] | 0.572 |
| B ff | 4.800 | 0.748 | 0.618 | [+0.058, +0.194] | 0.854 | 0.066 | [+0.021, +0.123] | 0.724 |
| clock | 1.200 | 0.721 | 0.557 | [+0.135, +0.183] | 1.000 | +0.00000 | [+0.000, +0.000] | 0.500 |
| clock | 2.000 | 0.721 | 0.558 | [+0.137, +0.183] | 1.000 | +0.00000 | [+0.000, +0.000] | 0.500 |
| clock | 4.800 | 0.924 | 0.698 | [+0.185, +0.254] | 1.000 | +0.00000 | [+0.000, +0.000] | 0.500 |

Share positive on interior frames by failure type:

| V | n (s) | success | stuck_pot2 | stuck_placing_pot2 | near_miss | stalled_earlier | knocked_over | stuck_pot1 | slow |
|---|---|---|---|---|---|---|---|---|---|
| ResNet | 1.200 | 0.715 | 0.694 | 0.619 | 0.618 | 0.723 | 0.674 | 0.773 | 0.673 |
| ResNet | 2.000 | 0.743 | 0.673 | 0.596 | 0.584 | 0.672 | 0.664 | 0.731 | 0.664 |
| ResNet | 4.800 | 0.822 | 0.564 | 0.589 | 0.577 | 0.572 | 0.568 | 0.442 | 0.637 |
| ResNet s0 | 1.200 | 0.706 | 0.703 | 0.612 | 0.629 | 0.727 | 0.681 | 0.776 | 0.713 |
| ResNet s0 | 2.000 | 0.728 | 0.684 | 0.597 | 0.601 | 0.697 | 0.658 | 0.765 | 0.685 |
| ResNet s0 | 4.800 | 0.805 | 0.590 | 0.603 | 0.582 | 0.588 | 0.575 | 0.510 | 0.644 |
| ResNet+2s | 1.200 | 0.719 | 0.686 | 0.648 | 0.618 | 0.692 | 0.663 | 0.772 | 0.665 |
| ResNet+2s | 2.000 | 0.751 | 0.660 | 0.622 | 0.574 | 0.646 | 0.643 | 0.726 | 0.642 |
| ResNet+2s | 4.800 | 0.818 | 0.535 | 0.568 | 0.579 | 0.596 | 0.596 | 0.497 | 0.562 |
| B | 1.200 | 0.686 | 0.713 | 0.657 | 0.665 | 0.737 | 0.668 | 0.817 | 0.684 |
| B | 2.000 | 0.690 | 0.709 | 0.646 | 0.671 | 0.724 | 0.647 | 0.805 | 0.678 |
| B | 4.800 | 0.730 | 0.657 | 0.579 | 0.694 | 0.653 | 0.574 | 0.663 | 0.671 |
| B s0 | 1.200 | 0.693 | 0.733 | 0.652 | 0.646 | 0.709 | 0.658 | 0.824 | 0.673 |
| B s0 | 2.000 | 0.707 | 0.711 | 0.631 | 0.657 | 0.687 | 0.647 | 0.760 | 0.669 |
| B s0 | 4.800 | 0.752 | 0.641 | 0.599 | 0.659 | 0.605 | 0.577 | 0.593 | 0.654 |
| ResNet ff | 1.200 | 0.716 | 0.646 | 0.678 | 0.645 | 0.733 | 0.674 | 0.778 | 0.614 |
| ResNet ff | 2.000 | 0.745 | 0.625 | 0.647 | 0.590 | 0.697 | 0.669 | 0.717 | 0.608 |
| ResNet ff | 4.800 | 0.825 | 0.532 | 0.602 | 0.557 | 0.570 | 0.615 | 0.475 | 0.561 |
| B ff | 1.200 | 0.692 | 0.712 | 0.651 | 0.655 | 0.727 | 0.666 | 0.809 | 0.666 |
| B ff | 2.000 | 0.700 | 0.705 | 0.629 | 0.647 | 0.702 | 0.663 | 0.798 | 0.649 |
| B ff | 4.800 | 0.750 | 0.627 | 0.599 | 0.602 | 0.618 | 0.593 | 0.608 | 0.614 |
| clock | 1.200 | 0.637 | 0.504 | 0.504 | 0.504 | 0.504 | 0.504 | 0.504 | 0.504 |
| clock | 2.000 | 0.640 | 0.504 | 0.504 | 0.504 | 0.504 | 0.504 | 0.504 | 0.504 |
| clock | 4.800 | 0.809 | 0.627 | 0.627 | 0.627 | 0.627 | 0.627 | 0.627 | 0.627 |

## Threshold: 30% positive (ε = 70th percentile of val advantages)

| V | n (s) | ε | terminal: succ | terminal: fail | pos succ | pos fail | succ − fail | 95% CI | AUC within scene | flips/s | agree w/ ResNet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet | 1.200 | 0.019 | 0.971 | 0.398 | 0.342 | 0.241 | 0.101 | [+0.060, +0.148] | 0.863 | 0.521 |  |
| ResNet | 2.000 | 0.026 | 0.849 | 0.309 | 0.348 | 0.233 | 0.115 | [+0.078, +0.159] | 0.890 | 0.416 |  |
| ResNet | 4.800 | 0.051 | 0.618 | 0.175 | 0.360 | 0.224 | 0.136 | [+0.091, +0.189] | 0.939 | 0.321 |  |
| ResNet s0 | 1.200 | 0.020 | 0.961 | 0.398 | 0.338 | 0.246 | 0.092 | [+0.055, +0.127] | 0.882 | 0.537 | 0.889 |
| ResNet s0 | 2.000 | 0.029 | 0.858 | 0.284 | 0.335 | 0.244 | 0.090 | [+0.052, +0.127] | 0.911 | 0.438 | 0.894 |
| ResNet s0 | 4.800 | 0.056 | 0.538 | 0.130 | 0.357 | 0.241 | 0.116 | [+0.073, +0.158] | 0.922 | 0.342 | 0.908 |
| ResNet+2s | 1.200 | 0.020 | 0.850 | 0.608 | 0.340 | 0.242 | 0.098 | [+0.059, +0.138] | 0.779 | 0.580 | 0.722 |
| ResNet+2s | 2.000 | 0.027 | 0.723 | 0.547 | 0.335 | 0.240 | 0.094 | [+0.057, +0.127] | 0.734 | 0.461 | 0.732 |
| ResNet+2s | 4.800 | 0.050 | 0.440 | 0.222 | 0.362 | 0.238 | 0.124 | [+0.088, +0.159] | 0.913 | 0.329 | 0.782 |
| B | 1.200 | 0.015 | 1.000 | 0.626 | 0.327 | 0.249 | 0.078 | [+0.012, +0.131] | 0.731 | 0.487 | 0.672 |
| B | 2.000 | 0.020 | 0.922 | 0.509 | 0.322 | 0.244 | 0.078 | [+0.013, +0.142] | 0.773 | 0.396 | 0.686 |
| B | 4.800 | 0.036 | 0.463 | 0.222 | 0.369 | 0.230 | 0.139 | [+0.070, +0.211] | 0.884 | 0.273 | 0.695 |
| B s0 | 1.200 | 0.016 | 0.830 | 0.696 | 0.335 | 0.245 | 0.090 | [+0.035, +0.139] | 0.714 | 0.507 | 0.670 |
| B s0 | 2.000 | 0.023 | 0.488 | 0.596 | 0.342 | 0.243 | 0.099 | [+0.046, +0.149] | 0.789 | 0.431 | 0.689 |
| B s0 | 4.800 | 0.036 | 0.279 | 0.278 | 0.380 | 0.236 | 0.145 | [+0.073, +0.214] | 0.876 | 0.325 | 0.679 |
| ResNet ff | 1.200 | 0.018 | 0.655 | 0.579 | 0.353 | 0.237 | 0.117 | [+0.074, +0.161] | 0.898 | 0.589 | 0.788 |
| ResNet ff | 2.000 | 0.024 | 0.527 | 0.516 | 0.360 | 0.228 | 0.132 | [+0.090, +0.176] | 0.938 | 0.498 | 0.797 |
| ResNet ff | 4.800 | 0.044 | 0.401 | 0.246 | 0.400 | 0.208 | 0.193 | [+0.149, +0.236] | 0.980 | 0.390 | 0.788 |
| B ff | 1.200 | 0.018 | 0.937 | 0.766 | 0.351 | 0.226 | 0.126 | [+0.075, +0.169] | 0.879 | 0.523 | 0.664 |
| B ff | 2.000 | 0.026 | 0.798 | 0.635 | 0.354 | 0.216 | 0.138 | [+0.085, +0.186] | 0.887 | 0.399 | 0.661 |
| B ff | 4.800 | 0.037 | 0.515 | 0.335 | 0.398 | 0.188 | 0.210 | [+0.138, +0.276] | 0.950 | 0.317 | 0.664 |
| clock | 1.200 | +0.00000 | 1.000 | +0.00000 | 0.189 | 0.150 | 0.040 | [+0.032, +0.044] | 1.000 | 0.379 | 0.644 |
| clock | 2.000 | +0.00000 | 1.000 | +0.00000 | 0.193 | 0.152 | 0.041 | [+0.034, +0.046] | 1.000 | 0.273 | 0.645 |
| clock | 4.800 | +0.00000 | 1.000 | +0.00000 | 0.164 | 0.127 | 0.037 | [+0.031, +0.042] | 1.000 | 0.267 | 0.654 |

Mid-episode only (lookahead ends ≥ 5 s before T, so V never sees the final scene):

| V | n (s) | pos succ | pos fail | succ − fail 95% CI | AUC within scene | time-matched succ − fail | 95% CI | time-matched AUC within scene |
|---|---|---|---|---|---|---|---|---|
| ResNet | 1.200 | 0.362 | 0.253 | [+0.056, +0.159] | 0.858 | 0.034 | [+0.013, +0.058] | 0.747 |
| ResNet | 2.000 | 0.372 | 0.245 | [+0.087, +0.170] | 0.906 | 0.049 | [+0.026, +0.077] | 0.830 |
| ResNet | 4.800 | 0.376 | 0.235 | [+0.094, +0.197] | 0.937 | 0.064 | [+0.042, +0.093] | 0.893 |
| ResNet s0 | 1.200 | 0.359 | 0.259 | [+0.058, +0.140] | 0.868 | 0.028 | [+0.010, +0.052] | 0.728 |
| ResNet s0 | 2.000 | 0.361 | 0.255 | [+0.061, +0.146] | 0.922 | 0.040 | [+0.023, +0.062] | 0.836 |
| ResNet s0 | 4.800 | 0.374 | 0.253 | [+0.075, +0.169] | 0.922 | 0.050 | [+0.035, +0.070] | 0.868 |
| ResNet+2s | 1.200 | 0.357 | 0.252 | [+0.067, +0.144] | 0.784 | 0.027 | [+0.011, +0.048] | 0.688 |
| ResNet+2s | 2.000 | 0.350 | 0.250 | [+0.061, +0.131] | 0.761 | 0.026 | [+0.011, +0.045] | 0.665 |
| ResNet+2s | 4.800 | 0.367 | 0.251 | [+0.080, +0.150] | 0.839 | 0.043 | [+0.022, +0.071] | 0.771 |
| B | 1.200 | 0.344 | 0.259 | [+0.022, +0.140] | 0.769 | 0.022 | [-0.002, +0.048] | 0.662 |
| B | 2.000 | 0.329 | 0.252 | [+0.010, +0.142] | 0.759 | 0.025 | [-0.002, +0.057] | 0.644 |
| B | 4.800 | 0.355 | 0.237 | [+0.049, +0.186] | 0.834 | 0.054 | [+0.026, +0.096] | 0.785 |
| B s0 | 1.200 | 0.353 | 0.259 | [+0.039, +0.144] | 0.702 | 0.017 | [+0.001, +0.037] | 0.625 |
| B s0 | 2.000 | 0.355 | 0.254 | [+0.041, +0.152] | 0.816 | 0.028 | [+0.010, +0.051] | 0.670 |
| B s0 | 4.800 | 0.364 | 0.244 | [+0.050, +0.186] | 0.761 | 0.045 | [+0.022, +0.079] | 0.739 |
| ResNet ff | 1.200 | 0.374 | 0.248 | [+0.080, +0.171] | 0.928 | 0.043 | [+0.030, +0.062] | 0.863 |
| ResNet ff | 2.000 | 0.383 | 0.239 | [+0.102, +0.192] | 0.957 | 0.052 | [+0.037, +0.073] | 0.877 |
| ResNet ff | 4.800 | 0.424 | 0.217 | [+0.162, +0.254] | 0.980 | 0.089 | [+0.062, +0.127] | 0.930 |
| B ff | 1.200 | 0.370 | 0.237 | [+0.083, +0.180] | 0.913 | 0.052 | [+0.026, +0.088] | 0.852 |
| B ff | 2.000 | 0.379 | 0.226 | [+0.091, +0.199] | 0.952 | 0.060 | [+0.035, +0.097] | 0.819 |
| B ff | 4.800 | 0.414 | 0.194 | [+0.147, +0.286] | 0.960 | 0.094 | [+0.056, +0.150] | 0.930 |
| clock | 1.200 | 0.213 | 0.165 | [+0.040, +0.053] | 1.000 | +0.00000 | [+0.000, +0.000] | 0.500 |
| clock | 2.000 | 0.217 | 0.168 | [+0.041, +0.054] | 1.000 | +0.00000 | [+0.000, +0.000] | 0.500 |
| clock | 4.800 | 0.187 | 0.142 | [+0.038, +0.050] | 1.000 | +0.00000 | [+0.000, +0.000] | 0.500 |

Share positive on interior frames by failure type:

| V | n (s) | success | stuck_pot2 | stuck_placing_pot2 | near_miss | stalled_earlier | knocked_over | stuck_pot1 | slow |
|---|---|---|---|---|---|---|---|---|---|
| ResNet | 1.200 | 0.342 | 0.208 | 0.314 | 0.294 | 0.197 | 0.263 | 0.140 | 0.269 |
| ResNet | 2.000 | 0.348 | 0.205 | 0.299 | 0.299 | 0.196 | 0.230 | 0.141 | 0.250 |
| ResNet | 4.800 | 0.360 | 0.211 | 0.281 | 0.338 | 0.172 | 0.224 | 0.102 | 0.230 |
| ResNet s0 | 1.200 | 0.338 | 0.234 | 0.306 | 0.280 | 0.214 | 0.268 | 0.142 | 0.256 |
| ResNet s0 | 2.000 | 0.335 | 0.232 | 0.312 | 0.295 | 0.208 | 0.259 | 0.139 | 0.245 |
| ResNet s0 | 4.800 | 0.357 | 0.220 | 0.302 | 0.344 | 0.212 | 0.240 | 0.098 | 0.241 |
| ResNet+2s | 1.200 | 0.340 | 0.208 | 0.283 | 0.312 | 0.208 | 0.270 | 0.146 | 0.265 |
| ResNet+2s | 2.000 | 0.335 | 0.199 | 0.286 | 0.309 | 0.218 | 0.266 | 0.134 | 0.262 |
| ResNet+2s | 4.800 | 0.362 | 0.213 | 0.275 | 0.308 | 0.226 | 0.247 | 0.139 | 0.246 |
| B | 1.200 | 0.327 | 0.198 | 0.334 | 0.301 | 0.228 | 0.285 | 0.124 | 0.251 |
| B | 2.000 | 0.322 | 0.184 | 0.335 | 0.307 | 0.229 | 0.274 | 0.104 | 0.248 |
| B | 4.800 | 0.369 | 0.180 | 0.318 | 0.323 | 0.201 | 0.272 | 0.081 | 0.211 |
| B s0 | 1.200 | 0.335 | 0.205 | 0.304 | 0.285 | 0.228 | 0.287 | 0.132 | 0.261 |
| B s0 | 2.000 | 0.342 | 0.200 | 0.310 | 0.288 | 0.227 | 0.281 | 0.123 | 0.253 |
| B s0 | 4.800 | 0.380 | 0.183 | 0.324 | 0.287 | 0.224 | 0.275 | 0.088 | 0.238 |
| ResNet ff | 1.200 | 0.353 | 0.251 | 0.241 | 0.241 | 0.217 | 0.260 | 0.128 | 0.290 |
| ResNet ff | 2.000 | 0.360 | 0.239 | 0.240 | 0.241 | 0.212 | 0.249 | 0.123 | 0.264 |
| ResNet ff | 4.800 | 0.400 | 0.233 | 0.225 | 0.189 | 0.206 | 0.234 | 0.090 | 0.226 |
| B ff | 1.200 | 0.351 | 0.189 | 0.285 | 0.236 | 0.218 | 0.246 | 0.135 | 0.253 |
| B ff | 2.000 | 0.354 | 0.175 | 0.264 | 0.239 | 0.229 | 0.240 | 0.110 | 0.229 |
| B ff | 4.800 | 0.398 | 0.143 | 0.253 | 0.189 | 0.213 | 0.200 | 0.071 | 0.207 |
| clock | 1.200 | 0.189 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| clock | 2.000 | 0.193 | 0.152 | 0.152 | 0.152 | 0.152 | 0.152 | 0.152 | 0.152 |
| clock | 4.800 | 0.164 | 0.127 | 0.127 | 0.127 | 0.127 | 0.127 | 0.127 | 0.127 |
## Reading

[ours, 2026-10-29] Main row: ResNet ensemble, n = 2 s, 70% positive.

1. **The labels carry outcome information beyond scene and clock.** Mid-episode and time-matched (same scene,
   same frame), successes get +5.4 points more positives [+2.3, +8.6]; time-matched within-scene AUC 0.74. The
   clock control is exactly 0 / 0.50 there. This is about the same size as on the station data (R0, n = 2 s:
   +6.1 [+2.7, +9.3], `reports/advantages.md`).
2. **The raw columns overstate it.** Every failure runs the full 52 s, so episode length alone separates outcomes
   on the raw share: the image-free clock gets raw succ − fail +13.6 points and within-scene AUC 1.000. Use the
   time-matched columns for any comparison.
3. **Longer lookahead separates more**: time-matched +3.9 / +5.4 / +8.6 points at 1.2 / 2 / 4.8 s, and fewer label
   flips (0.60 → 0.28 per s). Labels are still noisy: at 2 s about one flip every 2 s (figure 02: single frames
   alternate between positive and negative inside smooth progress).
4. **Ensemble vs single seed, history vs none: no clear difference.** The intervals overlap everywhere. The 2 s-history
   ResNet agrees with the plain ensemble on only 69% of interior labels, so the choice of V changes *which* frames
   are positive more than *how well* the labels track outcome.
5. **Failure frames are mostly positive near the end** (terminal, 70% threshold: 61% of failure frames, all success
   frames). Near the end of a failure V already sits near −0.5, so the real return is no worse than V expected. This
   is the plan's open "decide on failure-ending frames" question (Mulligan: 72%). Not decided here.
6. **By failure type** (interior, 2 s): stuck_placing_pot2 and near_miss get the fewest positives (0.60 / 0.58 vs
   0.74 for successes); stuck_pot1 the most (0.73). A guess, not checked: V gives up early there and then stays flat.
   This matches figure 02 (stuck_pot2): a sharp drop gives a burst of negatives, then the flat stretch scores as
   positive.

Status: a dry run with the **control** V. In the plan the fine-tuning labels come from the 3b π₀.₅ value expert;
these numbers are the bar it has to match. Train-rollout labels (Stage 6 arm C) also need an out-of-fold V.

### Addendum [ours, 2026-10-30]: π₀.₅ value expert B vs the ResNet

Rows `B` (3-seed ensemble of 3b arm B, w256 + augmentation) and `B s0` added; same frames, thresholds and checks.
The ResNet rows are unchanged from the 2026-10-29 run. Main row: n = 2 s, 70% positive, mid-episode, time-matched.

1. **B's labels track the outcome about half as well as the ResNet's.** Time-matched succ − fail: B +2.4 points
   [−1.0, +6.1] vs ResNet +5.4 [+2.3, +8.6]; time-matched within-scene AUC 0.57 vs 0.74. Paired scene bootstrap
   of B − ResNet (same 10 scenes resampled for both): −2.7 [−5.5, −0.5] / −3.0 [−6.6, −0.1] / −3.2 [−6.7, −0.4]
   points at n = 1.2 / 2 / 4.8 s. So the ResNet is better at every lookahead, by a margin that only just clears
   scene noise. B separates outcomes at all only at 4.8 s (+5.3 [+1.4, +10.5]).
2. **The raw columns flip.** B's raw mid-episode succ − fail is −1.6 points at 2 s (ResNet +10.2). B knows the
   outcome from the layout early and then stays flat, so within an episode V(t + n) − V(t) is near the step cost
   for both outcomes. This matches the Stage 4 plots (`reports/figures/libero_stage4_b_vs_resnet/`).
3. **B's labels are different labels, not noisier copies**: they agree with the ResNet's on 64% of interior
   frames (the ResNet seeds agree with their ensemble 87–88%). B flips a little less (0.44 vs 0.48 per s).
4. **Failure-ending frames**: B labels 72% of terminal failure frames positive (ResNet 61%). B already sits near
   −0.5 on doomed layouts, so the real return is no worse than B expected (open question 5 above, sharper with B).

Reading: on the labels Stage 5 actually uses, B is worse than the ResNet. This agrees with the 3b pass rule,
which was set to measure this.
