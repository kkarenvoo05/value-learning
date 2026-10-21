# Video <-> proprio timing recheck (Stage 5)

220 training episodes (20 per repo, seed 0), frames 0..T. Lag L: corr(video motion[k + L], parquet signal[k]); L > 0 = video late. Video motion = mean |frame difference| at 160x120 (gray); `fingers` = the finger region of the wrist view. Parquet signals: `pos` = |Δ EE position|, `rot` = EE rotation angle per step, `grip` = |Δ gripper|. Pooled curve = mean of the per-episode curves; 95% CI of its sub-frame peak from 1,000 episode bootstraps.

## Pooled lag peaks

| group                     | pair           |   episodes |   pooled peak |   sub-frame | 95% CI         |   r(0) |   r(+6) |   r(-6) | per-episode argmax (r>=0.5)                       |
|:--------------------------|:---------------|-----------:|--------------:|------------:|:---------------|-------:|--------:|--------:|:--------------------------------------------------|
| teleop (c00)              | side_1~pos     |         20 |             0 |        0.08 | [-0.02, +0.17] |  0.416 |   0.202 |   0.119 | +0:9, +1:1                                        |
| teleop (c00)              | wrist_left~pos |         20 |             0 |        0.09 | [-0.02, +0.30] |  0.449 |   0.321 |   0.201 | +0:8, +5:1, +8:1                                  |
| teleop (c00)              | wrist_left~rot |         20 |             0 |        0.01 | [-0.09, +0.11] |  0.525 |   0.402 |   0.369 | +0:10, +2:1, +5:1                                 |
| teleop (c00)              | fingers~grip   |         20 |             0 |        0.26 | [+0.15, +0.36] |  0.473 |   0.23  |   0.177 | -2:1, +0:8, +1:1                                  |
| DAgger (c01-c05)          | side_1~pos     |        100 |             0 |        0.11 | [-0.01, +0.19] |  0.262 |   0.203 |   0.137 | -3:1, -1:1, +0:15, +5:1, +6:1, +12:2              |
| DAgger (c01-c05)          | wrist_left~pos |        100 |             0 |       -0    | [-0.02, +0.02] |  0.402 |   0.346 |   0.259 | +0:32, +2:1, +9:1, +12:1                          |
| DAgger (c01-c05)          | wrist_left~rot |        100 |             0 |       -0.1  | [-0.13, -0.06] |  0.451 |   0.375 |   0.378 | -6:2, -4:1, -3:1, -2:1, +0:35, +3:1, +8:1         |
| DAgger (c01-c05)          | fingers~grip   |        100 |             0 |        0.06 | [+0.03, +0.10] |  0.395 |   0.196 |   0.165 | +0:36, +1:1                                       |
| autonomous eval (r00-r04) | side_1~pos     |        100 |             0 |        0.13 | [+0.09, +0.16] |  0.517 |   0.461 |   0.288 | -3:1, -2:1, +0:50, +1:2, +3:6, +6:4, +11:1, +12:5 |
| autonomous eval (r00-r04) | wrist_left~pos |        100 |             0 |       -0.02 | [-0.03, -0.01] |  0.67  |   0.628 |   0.476 | -7:1, +0:67, +4:1, +6:11, +10:1, +12:9            |
| autonomous eval (r00-r04) | wrist_left~rot |        100 |             0 |       -0.03 | [-0.04, -0.01] |  0.77  |   0.672 |   0.654 | -10:1, -6:1, +0:90, +6:3, +12:1                   |
| autonomous eval (r00-r04) | fingers~grip   |        100 |             0 |        0.04 | [+0.01, +0.09] |  0.646 |   0.343 |   0.342 | +0:82, +4:1                                       |

## Autocorrelation of EE speed (6-step action chunks)

Mean over episodes of corr(speed[k + L], speed[k]). A bump at L = 6 (and 12) is the chunk period.

|                           |   0 |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |    10 |    11 |    12 |
|:--------------------------|----:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| teleop (c00)              |   1 | 0.921 | 0.847 | 0.755 | 0.663 | 0.581 | 0.506 | 0.447 | 0.404 | 0.367 | 0.334 | 0.299 | 0.262 |
| DAgger (c01-c05)          |   1 | 0.856 | 0.822 | 0.791 | 0.729 | 0.71  | 0.711 | 0.629 | 0.579 | 0.56  | 0.509 | 0.499 | 0.502 |
| autonomous eval (r00-r04) |   1 | 0.783 | 0.788 | 0.748 | 0.704 | 0.63  | 0.748 | 0.576 | 0.569 | 0.538 | 0.505 | 0.445 | 0.55  |

## Parquet-internal checks

Reported `cartesian_velocity` x vs Δ position x, corr(vel[k + 1 + L], pos[k + 1] − pos[k]): L=-3: 0.915, L=-2: 0.961, L=-1: 0.945, L=+0: 0.907, L=+1: 0.862, L=+2: 0.813, L=+3: 0.768

Euler convention (mean corr of finite-difference world angular velocity with the reported rotational velocity, best lag in −2..2): extrinsic `xyz` 0.958, intrinsic `XYZ` -0.410

## Reading (2026-10-20)

- **Video frame t and parquet row t are synchronous (lag 0), for both cameras.** Every pooled curve
  (4 signal pairs × 3 groups) peaks at L = 0; sub-frame peaks are within −0.10…+0.26 frames
  (< 18 ms at 15 fps). Teleop (no action chunks) gives one smooth peak at 0 with no side peaks.
- **The Stage 3 "+6" was the action chunk, not an offset.** The eval runner executes 6-step chunks
  (`n_action_steps: 6` in `results.json`); autonomous EE speed autocorrelation bumps at L = 6 and 12,
  teleop's decays smoothly. That periodicity puts side peaks at ±6/±12 in autonomous lag curves, so
  per-episode argmaxes sometimes land there (wrist~pos: 11 at +6, 9 at +12 of 100). Pooling resolves it.
- **Euler angles are extrinsic `xyz`** (r = 0.96 vs −0.41 for intrinsic `XYZ`). Roll sits at ±π and wraps
  inside episodes, so the input uses the 6D rotation (first two columns of R), not raw angles.
- **`observation.state.cartesian_velocity` is not synchronous with position**: it best matches the
  displacement 1–2 rows later (peak at L = −2). Velocities are left out of the input [ours].
