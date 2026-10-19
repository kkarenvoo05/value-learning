# Stage 5 ablations

All arms use the settled recipe (`baseline_lowlr`: 8k steps, cosine, backbone 1e-5 / head 1e-4,
checkpoint = lowest val CE) and 3 seeds. Validation split only; the test repos are untouched.
Seed noise of the baseline: val CE sd 0.002, MAE sd 0.004. Differences under ~2 sd are not evidence.

## 1. Cameras (2026-10-19)

Arms: wrist + side (`lowlr_8k`, `_s1`, `_s2`; the Stage 4 baseline), side only (`side_s{0,1,2}`,
job 17970458), wrist only (`wrist_s{0,1,2}`, job 17970459). Both new jobs ran on humanoid1, one
L40S each, with 20 CPUs (`scripts/cameras.sbatch`, `--cameras`), git `eb88526`. The single-camera
model is the same network with one 512-d feature into the head.

| arm | val CE | val MAE | MAE per ep. | eval-repo MAE | shape MAE (flat 0.037) | AUC 1 / 5 / 10 s | best steps |
|---|---|---|---|---|---|---|---|
| **wrist + side** | **3.730 ± 0.002** | **0.121 ± 0.004** | 0.101 ± 0.002 | 0.196 ± 0.008 | 0.047 ± 0.004 | 0.67 / 0.64 / 0.84 | 3k, 2k, 2k |
| side only | 3.799 ± 0.017 | 0.130 ± 0.009 | 0.109 ± 0.006 | 0.205 ± 0.016 | 0.048 ± 0.006 | 0.59 / 0.55 / 0.82 | 3k, 2k, 1.5k |
| wrist only | 3.963 ± 0.019 | 0.191 ± 0.010 | 0.169 ± 0.010 | 0.248 ± 0.012 | 0.067 ± 0.007 | 0.59 / 0.66 / 0.77 | 1.5k, 2k, 1.5k |

Trivial baselines: global constant MAE 0.257, repo-only constant 0.140 (eval repos 0.222).

- **The side view carries most of the signal.** Side only is close to both cameras: CE is clearly
  worse (+0.07, ~4 sd of the single-camera spread), MAE worse by 0.009 (borderline, ~2 sd), and
  the 1–5 s AUC is lower. **Wrist only is much worse**: MAE 0.191 does not even beat the repo-only
  constant (0.140), its shape error is the worst, and its val CE peaks early (1.5–2k steps) and
  then rises, so it overfits even at the low LR.
- **Keep both cameras** (the baseline). The wrist view adds a small but consistent gain on CE and
  early AUC on top of the side view; alone it is not useful.
- **Round cue at frame 0** (mean V at frame 0, autonomous val episodes, mean of 3 seeds):

  | round | wrist + side | side only | wrist only |
  |---|---:|---:|---:|
  | r00 | −0.526 | −0.499 | −0.243 |
  | r01 | −0.533 | −0.514 | −0.310 |
  | r02 | −0.493 | −0.461 | −0.356 |
  | r03 | −0.454 | −0.444 | −0.360 |
  | r04 | −0.401 | −0.393 | −0.328 |

  Side only reproduces the baseline's ordering (low for the early, mostly failing rounds, higher
  for r04). Wrist only also differs by round but in the *opposite* order, with r00 most
  optimistic, so it does not carry the outcome-related round cue. **The round cue that V uses is
  in the side view.** The question for Lars (day/lighting/camera setup per round) can focus on `side_1`.
- **Drops when the arm leaves the side view** (`vl.side_occlusion`, `reports/side_occlusion_{side_s0,wrist_s0}.md`):
  P(ΔV < −0.2 in 1 s | the arm leaves the side view) vs when occlusion doesn't jump:

  | arm | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | wrist + side | 0.081 vs 0.008 | 0.072 vs 0.008 | 0.045 vs 0.003 |
  | side only | 0.063 vs 0.008 | 0.027 vs 0.008 | 0.009 vs 0.001 |
  | wrist only | 0 vs 0.003 | 0.018 vs 0.017 | 0 vs 0.002 |

  The cluster disappears without `side_1`, so these drops are driven by what the side view shows at
  that moment. Whether that is a real event (pulling back after a miss) or a viewpoint artifact is
  still open. Side only also gets slightly worse on high-occlusion frames (matched ΔMAE +0.011…+0.024
  vs +0.004…+0.008 for both cameras), so the wrist view helps when the arm blocks the side view.
