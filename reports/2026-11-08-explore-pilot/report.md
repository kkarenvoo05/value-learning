# Data-variety pilot: does exploration noise add action variety without ruining episodes?

2026-11-08. Karen: option A (exploration noise, no policy mix), pilot first.

## Question
The coach test (`reports/2026-11-08-coach-test/`) found the bottleneck is the data: one policy doing nearly the
same thing in the same spots. Before collecting 450 new episodes, check which noise scale (if any) makes the robot
act differently in the same spots while still finishing the task often enough.

## Setup
- Policy: base pi05_libero via `serve_pi05_finetuned.py --base` (BASE_OURS=1) for all arms.
- 30 fixed train init states × 3 repeats = 90 episodes per arm, libero_10 task 8, env seed 21, raw actions saved.
- Exploration [ours]: each new action chunk, with p 0.5, is sampled from flow-matching start noise × s
  (`libero_rollouts.py --args.explore-prob 0.5 --args.explore-scale s`). Arms: control (p 0), ×1.5, ×2.0.
- Jobs: control 18388271, ×1.5 18388272, ×2.0 18388273. Analysis `scripts/libero_explore_pilot.py`.
- Spread = std across the 3 repeats of the executed action at each early step (t < 60, 3 s), RMS over arm dims,
  averaged over states and steps.
- Rule (set before the result): largest scale with success ≥ control − 10 pts (point estimate) and early arm
  spread ≥ 1.5× control.

## Results

| arm | success | vs control (paired, 95% CI) | steps (successes) | arm spread | ratio | gripper spread | qualifies |
|---|---|---|---|---|---|---|---|
| control | 0.644 | — | 424 | 0.0155 | — | 0.0014 | — |
| ×1.5 | 0.489 | −0.156 [−0.267, −0.044] | 423 | 0.0190 | 1.23 | 0.0028 | no |
| ×2.0 | 0.567 | −0.078 [−0.200, +0.056] | 427 | 0.0255 | 1.64 | 0.0121 | **yes** |

Noisy vs normal chunks (×2.0): mean |arm action| 0.108 vs 0.105; gripper flips per chunk 0.081 vs 0.046.
Noisy share of chunks 0.49 in both arms.

## Verdict
**×2.0 passes the rule.** The server does apply the scale: spread rises with the scale (1.0 → 1.23 → 1.64×) and
noisy chunks at ×2.0 flip the gripper almost twice as often.

## Caveats
- Success is non-monotone (×1.5 lower than ×2.0). With 90 episodes per arm the CIs overlap, so this is most
  likely noise; the true cost of ×2.0 could be anywhere from ~0 to ~20 pts.
- Spread is measured only in the first 3 s, not at grasps or placements, where the labels matter most.
- 1.64× more early arm variety does not guarantee more action information in the labels; the coach test on the
  new data is the real check.

## Next
With Karen's go: 450 episodes at ×2.0, p 0.5 on the 30 fixed train states (15 each) → table → 5-fold OOF ResNet
V → vb40 labels → coach test on the new table (needs `libero_tag_ideal_shift.py --table`). Policy training only if
the ideal shift clearly rises above the current 0.0151.
