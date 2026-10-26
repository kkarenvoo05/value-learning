# Residual probe: past trajectory vs R0's errors

.venv/bin/python scripts/residual_probe.py              # -> reports/residual_probe.md

Track B caveat [ours, 2026-10-26]: the still-time upper bound (b3_still) only covers "how long nothing
moved". Here R0's val residual r = target value - V (V = mean over g0_r0_s{0,1,2}, best checkpoints) is
regressed on causal features of the gripper's past (data/proprio/frame_proprio.parquet, 15 Hz), one
feature group at a time on top of a base model. Proprio is a perfect record of the arm's motion, so a
group that cannot predict r beyond still time is information image history cannot add either (it can
still carry things proprio does not see: the marker, the holder, the operator).

Model [ours]: additive ridge on quantile-binned features (each feature -> 8 one-hot bins), fitted on
4 of 5 episode folds, scored on the 5th (GroupKFold by episode, fixed fold assignment, seed 0);
ridge lambda per outer fold from an inner episode-fold CV over 1e0..1e6 (a fixed small lambda
overfit episode offsets: every group, even calibration alone, scored worse than predicting 0).
Base = bins of V itself (calibration, so a group must add more than re-scaling V). Score = CV MSE of r
and its reduction vs base; intervals: bootstrap over val episodes of the per-episode squared-error
sums (2000 draws). Groups:
  still        log1p(seconds still), the b3_still input
  elapsed      seconds into the episode
  motion       per window 2/5/10/20 s: path length, net displacement, dz, d gripper, "window
               reaches before start"
  retries      gripper toggles so far, z dips (down then up >= 2 cm) so far, path length so far,
               z now - min z so far
  current      x, y, z, rotation, gripper now (no memory; control: the image already shows it)
  memory       still + elapsed + motion + retries

Val: 247 episodes, 64906 frames. r = target - V(R0 mean of 3 seeds); MSE x 1e3. Reduction = (base MSE - group MSE) / base MSE; vs still = group MSE - still MSE (negative = better than still), 95% episode bootstrap.

### all val frames (247 episodes, 64906 frames)

| features (+ base) | CV MSE | reduction vs base | Δ MSE vs still [95%] |
|---|---|---|---|
| none | 30.685 | +0.8% |  |
| base | 30.929 | +0.0% | -0.167 [-0.445, +0.053] |
| still | 31.096 | -0.5% |  |
| elapsed | 31.016 | -0.3% | -0.080 [-0.228, +0.040] |
| motion | 30.976 | -0.2% | -0.120 [-0.235, -0.019]* |
| retries | 30.864 | +0.2% | -0.232 [-0.599, +0.050] |
| current | 30.728 | +0.7% | -0.368 [-0.886, +0.076] |
| memory | 31.018 | -0.3% | -0.078 [-0.189, +0.020] |
| still+motion | 31.302 | -1.2% | +0.206 [-0.301, +0.713] |
| still+retries | 31.175 | -0.8% | +0.079 [-0.054, +0.236] |
| still+elapsed | 31.128 | -0.6% | +0.032 [-0.002, +0.073] |

### hold phase, eval repos (104 episodes, 23824 frames)

| features (+ base) | CV MSE | reduction vs base | Δ MSE vs still [95%] |
|---|---|---|---|
| none | 52.526 | +0.7% |  |
| base | 52.894 | +0.0% | -0.535 [-1.306, +0.102] |
| still | 53.429 | -1.0% |  |
| elapsed | 53.160 | -0.5% | -0.269 [-0.709, +0.075] |
| motion | 53.100 | -0.4% | -0.328 [-0.616, -0.060]* |
| retries | 52.846 | +0.1% | -0.583 [-1.525, +0.153] |
| current | 52.584 | +0.6% | -0.845 [-2.173, +0.290] |
| memory | 53.207 | -0.6% | -0.222 [-0.476, +0.025] |
| still+motion | 53.321 | -0.8% | -0.108 [-1.470, +1.200] |
| still+retries | 53.614 | -1.4% | +0.186 [-0.152, +0.582] |
| still+elapsed | 53.508 | -1.2% | +0.079 [-0.007, +0.186] |

### hold phase, other failure (14 episodes, 3617 frames)

| features (+ base) | CV MSE | reduction vs base | Δ MSE vs still [95%] |
|---|---|---|---|
| none | 28.413 | -0.3% |  |
| base | 28.320 | +0.0% | +0.012 [-0.042, +0.047] |
| still | 28.309 | +0.0% |  |
| elapsed | 28.365 | -0.2% | +0.057 [+0.004, +0.156]* |
| motion | 28.011 | +1.1% | -0.298 [-0.688, +0.000] |
| retries | 28.306 | +0.0% | -0.002 [-0.114, +0.087] |
| current | 28.409 | -0.3% | +0.100 [-0.013, +0.179] |
| memory | 27.981 | +1.2% | -0.328 [-0.782, +0.008] |
| still+motion | 26.757 | +5.5% | -1.552 [-4.233, +0.729] |
| still+retries | 28.272 | +0.2% | -0.036 [-0.154, +0.052] |
| still+elapsed | 28.321 | -0.0% | +0.013 [-0.019, +0.073] |

### hold phase, stall (45 episodes, 12093 frames)

| features (+ base) | CV MSE | reduction vs base | Δ MSE vs still [95%] |
|---|---|---|---|
| none | 46.852 | -1.3% |  |
| base | 46.245 | +0.0% | +0.183 [-0.127, +0.706] |
| still | 46.062 | +0.4% |  |
| elapsed | 46.164 | +0.2% | +0.102 [-0.058, +0.364] |
| motion | 45.738 | +1.1% | -0.324 [-0.655, -0.027]* |
| retries | 46.301 | -0.1% | +0.239 [-0.131, +0.934] |
| current | 46.656 | -0.9% | +0.594 [-0.035, +1.712] |
| memory | 45.638 | +1.3% | -0.423 [-0.778, -0.141]* |
| still+motion | 45.156 | +2.4% | -0.905 [-2.214, +0.248] |
| still+retries | 45.975 | +0.6% | -0.087 [-0.150, -0.023]* |
| still+elapsed | 46.048 | +0.4% | -0.014 [-0.070, +0.031] |

