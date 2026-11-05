# C2a gain probe: can proprio history or force stand in for the past frame?

.venv/bin/python scripts/c2a_gain_probe.py              # -> reports/c2a_gain_probe.md

Question [ours, Karen 2026-10-27]: C2a (2 s-old frame stacked into the ResNet input) beats R0 on still
stalls; proprio still time (b3_still) did not. Before training a proprio-history or force arm, check
whether those signals can *predict what C2a changes*. Target dV = V(c2a_stack2_l40s, mean of 3 seeds) -
V(g0_r0_l40s, mean of 10 seeds), per val frame, both on L40S. If a feature group predicts dV well, a
model given that group could make the same correction without images; if it explains little of the
reliable part of dV, C2a's gain comes from what the camera sees.

Same probe as scripts/residual_probe.py (additive ridge on 8 quantile bins per feature, 5 episode
folds, lambda by inner episode CV). Base = bins of V(R0), so a group must add more than "C2a rescales V".
Score = CV R^2 of dV beyond base; ceiling = split-half reliability of dV (C2a s0 vs s1+s2, R0 s0-4 vs
s5-9, Spearman-Brown to the full average), i.e. the part of dV that is not seed noise. Second target:
r = value - V(C2a), i.e. is there still something in proprio / force that C2a misses (would they add on
top of C2a?). Groups (all causal, 15 Hz):
  still        log1p(seconds still), the b3_still input
  motion       residual_probe's memory groups (still, elapsed, 2-20 s windows, retries) + short windows
               0.5 / 1 / 2 s: path, net displacement, dz, d gripper, max step speed (from positions;
               the reported velocity runs 1-2 rows ahead of video, so it is not used)
  force        telemetry.franka.ee_wrench now (6) + |F| now; |F| mean and std, fz mean over 0.5 / 1 / 2 s;
               fz change over 2 s. Only 9 of 15 repos record it (missing: c00, c01, c02, r00, r01), so all
               groups are also scored on the force-present subset, fitted on that subset only
  motion+force

## all val frames: 247 val episodes, 64906 frames (c00-teleop-mixed, c01-dagger-mixed, c02-dagger-mixed, c03-dagger-mixed, c04-dagger-mixed, c05-dagger-mixed, r00-eval, r01-eval, r02-eval, r03-eval, r04-eval)

### target dV = V(C2a) - V(R0)

Reliable share of dV variance (split-half, Spearman-Brown): all 0.89, hold phase, eval repos 0.90, hold phase, stall 0.93. R^2 above this would be fitting seed noise.

| subset | episodes | var x 1e3 | R² base | R² still | R² motion | Δ R² still vs base [95%] | Δ R² motion vs base [95%] |
|---|---|---|---|---|---|---|---|
| all | 247 | 4.13 | +0.054 | +0.160 | +0.159 | +0.106 [+0.052, +0.153]* | +0.105 [+0.037, +0.162]* |
| hold phase, eval repos | 104 | 7.59 | +0.087 | +0.209 | +0.221 | +0.122 [+0.047, +0.178]* | +0.134 [+0.052, +0.205]* |
| hold phase, stall | 45 | 7.68 | +0.135 | +0.298 | +0.295 | +0.163 [+0.060, +0.232]* | +0.160 [+0.025, +0.253]* |

### target r = target - V(C2a)

| subset | episodes | var x 1e3 | R² base | R² still | R² motion | Δ R² still vs base [95%] | Δ R² motion vs base [95%] |
|---|---|---|---|---|---|---|---|
| all | 247 | 29.93 | -0.020 | -0.011 | -0.005 | +0.009 [+0.001, +0.020]* | +0.016 [-0.002, +0.036] |
| hold phase, eval repos | 104 | 49.79 | -0.013 | -0.008 | -0.006 | +0.005 [-0.005, +0.017] | +0.007 [-0.012, +0.030] |
| hold phase, stall | 45 | 37.46 | -0.060 | -0.037 | -0.026 | +0.023 [+0.003, +0.059]* | +0.033 [-0.005, +0.100] |

## force-present repos: 134 val episodes, 39762 frames (c03-dagger-mixed, c04-dagger-mixed, c05-dagger-mixed, r02-eval, r03-eval, r04-eval)

### target dV = V(C2a) - V(R0)

Reliable share of dV variance (split-half, Spearman-Brown): all 0.89, hold phase, eval repos 0.90, hold phase, stall 0.94. R^2 above this would be fitting seed noise.

| subset | episodes | var x 1e3 | R² base | R² still | R² motion | R² force | R² motion+force | Δ R² still vs base [95%] | Δ R² motion vs base [95%] | Δ R² force vs base [95%] | Δ R² motion+force vs base [95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 134 | 5.34 | +0.053 | +0.192 | +0.187 | +0.126 | +0.171 | +0.139 [+0.065, +0.199]* | +0.134 [+0.044, +0.203]* | +0.073 [+0.016, +0.123]* | +0.118 [+0.033, +0.185]* |
| hold phase, eval repos | 75 | 8.43 | +0.075 | +0.237 | +0.239 | +0.174 | +0.221 | +0.162 [+0.069, +0.235]* | +0.164 [+0.058, +0.248]* | +0.099 [+0.033, +0.159]* | +0.146 [+0.047, +0.227]* |
| hold phase, stall | 25 | 8.95 | +0.099 | +0.333 | +0.294 | +0.227 | +0.265 | +0.235 [+0.076, +0.329]* | +0.196 [-0.031, +0.323] | +0.129 [+0.009, +0.218]* | +0.166 [-0.033, +0.287] |

### target r = target - V(C2a)

| subset | episodes | var x 1e3 | R² base | R² still | R² motion | R² force | R² motion+force | Δ R² still vs base [95%] | Δ R² motion vs base [95%] | Δ R² force vs base [95%] | Δ R² motion+force vs base [95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 134 | 41.15 | +0.003 | -0.004 | -0.022 | -0.007 | -0.011 | -0.007 [-0.038, +0.028] | -0.025 [-0.062, +0.016] | -0.009 [-0.050, +0.037] | -0.014 [-0.050, +0.026] |
| hold phase, eval repos | 75 | 56.46 | +0.012 | -0.000 | -0.026 | -0.008 | -0.014 | -0.012 [-0.048, +0.027] | -0.038 [-0.084, +0.010] | -0.020 [-0.068, +0.031] | -0.026 [-0.069, +0.020] |
| hold phase, stall | 25 | 47.90 | -0.157 | -0.080 | -0.084 | -0.032 | -0.066 | +0.077 [+0.028, +0.149]* | +0.073 [+0.006, +0.167]* | +0.125 [+0.055, +0.223]* | +0.091 [+0.027, +0.181]* |

