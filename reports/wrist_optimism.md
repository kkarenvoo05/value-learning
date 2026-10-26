# Where the wrist camera makes not-success frames look better (C4 follow-up; `scripts/wrist_optimism.py`)

Saved masking outputs only (no new runs). Per frame, mean over 3 seeds: wrist ΔV = V_full − V_without_wrist (> 0: the wrist raises V); wrist ΔCE = CE_full − CE_without_wrist (> 0: the wrist makes the loss worse). R0 / C1: `mean` intervention; C3: `drop` (token masked, no stand-in). bias = V − target. Groupings [ours]; machine_failure_mode is a VLM label (diagnostic).

## 1. All arms: does the wrist add optimism on not-success frames?

| arm   | frames      |   bias full |   bias no wrist |   wrist ΔV |   wrist ΔCE |   share of frames wrist hurts |
|:------|:------------|------------:|----------------:|-----------:|------------:|------------------------------:|
| R0    | success     |      -0.076 |          -0.131 |      0.055 |      -0.947 |                         0.255 |
| R0    | not success |       0.154 |           0.127 |      0.027 |       0.133 |                         0.565 |
| C1    | success     |      -0.081 |          -0.127 |      0.046 |      -0.807 |                         0.276 |
| C1    | not success |       0.145 |           0.135 |      0.01  |       0.093 |                         0.544 |
| C3    | success     |      -0.077 |          -0.149 |      0.073 |      -1.401 |                         0.231 |
| C3    | not success |       0.15  |           0.069 |      0.082 |       0.292 |                         0.646 |

## 2. R0, not-success frames: where the wrist harm sits

### outcome × phase

|                          |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:-------------------------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| ('failure', 'first 5 s') |         25 |     1513 |   -0.627 |   -0.452 |       -0.468 |      0.017 |                  0.263 |    -0.219 |       0.175 |           0.158 |
| ('failure', 'last 5 s')  |         35 |     2496 |   -0.52  |   -0.463 |       -0.461 |     -0.002 |                  0.145 |    -0.181 |       0.056 |           0.059 |
| ('failure', 'middle')    |         13 |     2341 |   -0.625 |   -0.474 |       -0.478 |      0.003 |                  0.191 |    -0.171 |       0.151 |           0.147 |
| ('timeout', 'first 5 s') |         40 |     3038 |   -0.713 |   -0.467 |       -0.502 |      0.035 |                 -0.104 |    -0.251 |       0.246 |           0.211 |
| ('timeout', 'last 5 s')  |         40 |     3040 |   -0.521 |   -0.448 |       -0.502 |      0.054 |                  0.149 |    -0.209 |       0.073 |           0.019 |
| ('timeout', 'middle')    |         39 |    10793 |   -0.653 |   -0.482 |       -0.513 |      0.032 |                  0.162 |    -0.235 |       0.172 |           0.14  |

### stall (robot holding still) vs moving

|                    |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:-------------------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| ('failure', False) |         35 |     3913 |   -0.577 |   -0.448 |       -0.467 |      0.018 |                  0.318 |    -0.201 |       0.129 |           0.11  |
| ('failure', True)  |         16 |     2437 |   -0.596 |   -0.491 |       -0.473 |     -0.018 |                 -0.015 |    -0.164 |       0.105 |           0.123 |
| ('timeout', False) |         40 |     6067 |   -0.683 |   -0.457 |       -0.502 |      0.045 |                  0.065 |    -0.239 |       0.226 |           0.182 |
| ('timeout', True)  |         40 |    10804 |   -0.616 |   -0.482 |       -0.513 |      0.031 |                  0.138 |    -0.23  |       0.134 |           0.103 |

### hold phase (> 5 s in, > 2 s before T)

| hold   |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:-------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| False  |         75 |     7472 |   -0.617 |   -0.462 |       -0.488 |      0.027 |                  0.085 |    -0.225 |       0.155 |           0.129 |
| True   |         60 |    15749 |   -0.628 |   -0.475 |       -0.503 |      0.028 |                  0.156 |    -0.218 |       0.154 |           0.126 |

### machine_failure_mode

| fmode                                     |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:------------------------------------------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| (none)                                    |          1 |      146 |   -0.54  |   -0.074 |       -0.084 |      0.01  |                  0.997 |     0.053 |       0.466 |           0.456 |
| dropped_marker_during_transport           |          1 |       57 |   -0.516 |   -0.578 |       -0.57  |     -0.007 |                  0.392 |    -0.295 |      -0.062 |          -0.055 |
| jammed_partial_insert                     |          1 |      400 |   -0.611 |   -0.538 |       -0.515 |     -0.023 |                 -0.187 |    -0.146 |       0.073 |           0.096 |
| marker_released_at_holder                 |          2 |      693 |   -0.627 |   -0.42  |       -0.414 |     -0.006 |                 -0.279 |    -0.156 |       0.207 |           0.213 |
| marker_slipped_from_gripper               |          2 |      145 |   -0.522 |   -0.482 |       -0.523 |      0.041 |                  0.733 |    -0.268 |       0.04  |          -0.001 |
| missed_grasp_after_alignment              |          8 |     1927 |   -0.625 |   -0.551 |       -0.525 |     -0.027 |                  0.055 |    -0.154 |       0.073 |           0.1   |
| no_useful_approach                        |          2 |      103 |   -0.515 |   -0.544 |       -0.533 |     -0.011 |                  0.176 |    -0.249 |      -0.03  |          -0.019 |
| other                                     |          1 |      128 |   -0.535 |   -0.221 |       -0.245 |      0.024 |                  0.952 |    -0.014 |       0.314 |           0.29  |
| pregrasp_misalignment                     |          6 |     1497 |   -0.637 |   -0.531 |       -0.536 |      0.005 |                  0.171 |    -0.272 |       0.106 |           0.1   |
| timeout_holding_marker                    |         47 |    17082 |   -0.63  |   -0.461 |       -0.502 |      0.04  |                  0.149 |    -0.233 |       0.169 |           0.128 |
| transport_orientation_wrong_for_insertion |          1 |      283 |   -0.578 |   -0.556 |       -0.53  |     -0.027 |                 -0.062 |    -0.211 |       0.022 |           0.049 |
| uncompensated_grasp_angle                 |          1 |      290 |   -0.58  |   -0.268 |       -0.31  |      0.042 |                  0.427 |    -0.067 |       0.312 |           0.27  |
| wrong_hole_partial_insert                 |          2 |      470 |   -0.573 |   -0.528 |       -0.511 |     -0.018 |                 -0.134 |    -0.247 |       0.045 |           0.063 |

### episode kind

| kind       |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:-----------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| assisted   |          1 |      146 |   -0.54  |   -0.074 |       -0.084 |      0.01  |                  0.997 |     0.053 |       0.466 |           0.456 |
| autonomous |         74 |    23075 |   -0.625 |   -0.473 |       -0.501 |      0.028 |                  0.128 |    -0.222 |       0.152 |           0.125 |

### repo

| repo_short   |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:-------------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| c03          |          1 |      146 |   -0.54  |   -0.074 |       -0.084 |      0.01  |                  0.997 |     0.053 |       0.466 |           0.456 |
| r00          |         15 |     4051 |   -0.598 |   -0.48  |       -0.517 |      0.036 |                  0.059 |    -0.205 |       0.118 |           0.081 |
| r01          |         18 |     4100 |   -0.589 |   -0.501 |       -0.527 |      0.027 |                  0.116 |    -0.257 |       0.089 |           0.062 |
| r02          |         26 |     7708 |   -0.612 |   -0.448 |       -0.476 |      0.028 |                  0.277 |    -0.2   |       0.164 |           0.136 |
| r03          |          7 |     3702 |   -0.684 |   -0.541 |       -0.529 |     -0.012 |                 -0.041 |    -0.239 |       0.143 |           0.156 |
| r04          |          8 |     3514 |   -0.666 |   -0.416 |       -0.475 |      0.059 |                  0.071 |    -0.231 |       0.25  |           0.191 |

### control: success frames, by phase

| phase     |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:----------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| first 5 s |        172 |    12961 |   -0.114 |   -0.197 |       -0.201 |      0.004 |                 -0.432 |     0.014 |      -0.083 |          -0.087 |
| last 5 s  |        172 |    13072 |   -0.021 |   -0.069 |       -0.174 |      0.105 |                 -2.557 |     0.007 |      -0.048 |          -0.153 |
| middle    |        161 |    15652 |   -0.095 |   -0.187 |       -0.244 |      0.057 |                 -0.029 |    -0.036 |      -0.092 |          -0.15  |

## 3. Per episode (R0, not success)

53 of 75 not-success episodes have mean wrist ΔCE > 0. Top 10% of episodes (8) hold 48% of the positive frame-weighted harm. Rank correlation of per-episode wrist ΔCE, R0 vs C1 0.97, R0 vs C3 0.52.

| episode_uid                                     | repo   | outcome   | mode                         | kind       |   frames |   dCE |     dV |   bias |   bias_nw |   dCE C1 |   dCE C3 |
|:------------------------------------------------|:-------|:----------|:-----------------------------|:-----------|---------:|------:|-------:|-------:|----------:|---------:|---------:|
| real-marker-d2-r02-eval@d06a4aa79829:91         | r02    | timeout   | timeout_holding_marker       | autonomous |      307 | 2.124 |  0.326 |  0.394 |     0.068 |    2.308 |    2.613 |
| real-marker-d2-r02-eval@d06a4aa79829:45         | r02    | timeout   | timeout_holding_marker       | autonomous |      243 | 1.137 |  0.117 |  0.328 |     0.211 |    1.365 |    1.807 |
| real-marker-d2-r02-eval@d06a4aa79829:141        | r02    | failure   | timeout_holding_marker       | autonomous |      133 | 1.093 |  0.067 |  0.294 |     0.227 |    1.056 |    0.777 |
| real-marker-d2-c03-dagger-mixed@d60b6b0bf496:97 | c03    | failure   | (none)                       | assisted   |      146 | 0.997 |  0.01  |  0.466 |     0.456 |    1.022 |   -0.389 |
| real-marker-d2-r02-eval@d06a4aa79829:131        | r02    | failure   | other                        | autonomous |      128 | 0.952 |  0.024 |  0.314 |     0.29  |    0.917 |    0.301 |
| real-marker-d2-r03-eval@d0eded30bc32:130        | r03    | failure   | pregrasp_misalignment        | autonomous |       54 | 0.835 |  0.021 | -0.011 |    -0.031 |    0.695 |   -0.048 |
| real-marker-d2-r01-eval@6db676f48697:144        | r01    | failure   | marker_slipped_from_gripper  | autonomous |       97 | 0.802 |  0.062 |  0.089 |     0.027 |    0.625 |    0.578 |
| real-marker-d2-r02-eval@d06a4aa79829:142        | r02    | timeout   | timeout_holding_marker       | autonomous |      263 | 0.801 |  0.101 |  0.307 |     0.206 |    0.487 |    0.286 |
| real-marker-d2-r04-eval@6802af7602ca:99         | r04    | failure   | timeout_holding_marker       | autonomous |      211 | 0.755 |  0.103 |  0.215 |     0.112 |    0.676 |    0.086 |
| real-marker-d2-r02-eval@d06a4aa79829:44         | r02    | timeout   | timeout_holding_marker       | autonomous |      362 | 0.734 |  0.082 |  0.308 |     0.226 |    0.838 |    2.075 |
| real-marker-d2-r02-eval@d06a4aa79829:110        | r02    | failure   | timeout_holding_marker       | autonomous |       93 | 0.694 |  0.022 |  0.127 |     0.104 |    0.564 |    0.243 |
| real-marker-d2-r02-eval@d06a4aa79829:164        | r02    | timeout   | timeout_holding_marker       | autonomous |      301 | 0.649 |  0.061 |  0.135 |     0.074 |    0.629 |    0.806 |
| real-marker-d2-r04-eval@6802af7602ca:140        | r04    | failure   | missed_grasp_after_alignment | autonomous |       94 | 0.626 |  0.007 |  0.179 |     0.172 |    0.736 |   -0.176 |
| real-marker-d2-r00-eval@7a36afcd5632:92         | r00    | timeout   | timeout_holding_marker       | autonomous |      525 | 0.622 |  0.231 |  0.33  |     0.099 |    0.266 |    1     |
| real-marker-d2-r01-eval@6db676f48697:108        | r01    | failure   | marker_slipped_from_gripper  | autonomous |       48 | 0.595 | -0     | -0.057 |    -0.057 |    0.351 |    0.32  |
| real-marker-d2-r00-eval@7a36afcd5632:12         | r00    | failure   | missed_grasp_after_alignment | autonomous |       58 | 0.557 | -0.003 | -0.036 |    -0.032 |    0.381 |    0.538 |
| real-marker-d2-r00-eval@7a36afcd5632:14         | r00    | timeout   | timeout_holding_marker       | autonomous |      150 | 0.498 |  0.126 |  0.221 |     0.095 |    0.679 |    0.589 |
| real-marker-d2-r04-eval@6802af7602ca:49         | r04    | timeout   | timeout_holding_marker       | autonomous |      601 | 0.477 |  0.144 |  0.279 |     0.135 |    0.593 |    0.375 |
| real-marker-d2-r02-eval@d06a4aa79829:247        | r02    | failure   | pregrasp_misalignment        | autonomous |       74 | 0.446 |  0.004 |  0.044 |     0.041 |    0.326 |   -0.099 |
| real-marker-d2-r02-eval@d06a4aa79829:136        | r02    | failure   | uncompensated_grasp_angle    | autonomous |      290 | 0.427 |  0.042 |  0.312 |     0.27  |    0.318 |    0.928 |

### least wrist harm (most wrist help)

| episode_uid                              | repo   | outcome   | mode                      | kind       |   frames |    dCE |     dV |   bias |   bias_nw |   dCE C1 |   dCE C3 |
|:-----------------------------------------|:-------|:----------|:--------------------------|:-----------|---------:|-------:|-------:|-------:|----------:|---------:|---------:|
| real-marker-d2-r00-eval@7a36afcd5632:93  | r00    | failure   | marker_released_at_holder | autonomous |      151 | -0.52  | -0.004 | -0.016 |    -0.012 |   -0.558 |   -0.337 |
| real-marker-d2-r02-eval@d06a4aa79829:75  | r02    | timeout   | timeout_holding_marker    | autonomous |      563 | -0.492 | -0.1   |  0.136 |     0.236 |   -0.435 |   -0.052 |
| real-marker-d2-r04-eval@6802af7602ca:101 | r04    | timeout   | timeout_holding_marker    | autonomous |      800 | -0.428 | -0.045 |  0.216 |     0.261 |   -0.644 |   -0.238 |
| real-marker-d2-r00-eval@7a36afcd5632:4   | r00    | timeout   | timeout_holding_marker    | autonomous |      300 | -0.301 | -0.088 |  0.044 |     0.132 |   -0.228 |   -0.025 |
| real-marker-d2-r00-eval@7a36afcd5632:54  | r00    | timeout   | timeout_holding_marker    | autonomous |      491 | -0.293 | -0.061 |  0.098 |     0.159 |   -0.387 |   -0.026 |
| real-marker-d2-r02-eval@d06a4aa79829:21  | r02    | timeout   | timeout_holding_marker    | autonomous |      251 | -0.29  | -0.026 |  0.032 |     0.058 |   -0.353 |   -0.216 |
| real-marker-d2-r03-eval@d0eded30bc32:34  | r03    | timeout   | timeout_holding_marker    | autonomous |      826 | -0.283 | -0.022 |  0.167 |     0.19  |   -0.273 |   -0.009 |
| real-marker-d2-r04-eval@6802af7602ca:19  | r04    | failure   | marker_released_at_holder | autonomous |      542 | -0.212 | -0.006 |  0.269 |     0.275 |   -0.433 |    0.568 |

## 4. Hold-phase stall frames: successes vs not

If the wrist raises V on stalled not-success frames as much as on stalled successes, it cannot tell them apart; if it raises them *more*, the wrist view of a stall looks like a success.

|             |   episodes |   frames |   target |   V full |   V no wrist |   wrist ΔV |   wrist ΔCE (>0 hurts) |   side ΔV |   bias full |   bias no wrist |
|:------------|-----------:|---------:|---------:|---------:|-------------:|-----------:|-----------------------:|----------:|------------:|----------------:|
| not success |         51 |    11729 |   -0.623 |   -0.486 |       -0.507 |      0.021 |                  0.115 |    -0.218 |       0.137 |           0.117 |
| success     |         92 |     8171 |   -0.079 |   -0.19  |       -0.311 |      0.121 |                 -0.34  |    -0.074 |      -0.111 |          -0.232 |

- R0: AUC(success vs not, hold-phase stall frames) with wrist 0.915, without wrist 0.810
- C1: AUC(success vs not, hold-phase stall frames) with wrist 0.918, without wrist 0.806
- C3: AUC(success vs not, hold-phase stall frames) with wrist 0.907, without wrist 0.646
