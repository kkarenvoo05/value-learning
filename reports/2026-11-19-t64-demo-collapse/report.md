# Why t64d50 (task-64 residual AG + 35 task-64 demos) scores 0% (2026-11-19)

Karen: "investigate why the t64d50 demo run collapsed to 0%". Offline; no new jobs.

## Facts
- Eval: ceval_agr_t64d50_r0_s0_shift099 (18592250), t64_cont100, push 0.99 (w 2.08, interpolated): 0 / 101 so far,
  goals 00 in every episode (neither stacked nor in tray). Base 60.7 (300); same run without demos (t64b50cv) 38.7;
  the two-task run without task-64 demos at its task-64 w 74% (142 / 191, partial).
- Not the demo format: gripper sign (-1 open at the start), start state, action ranges match the rollouts.
- Not the demo amount: 35 demos vs moka's 29; 40.4% positive frames vs moka b50cv's 40.3%.
- **The demos solve the task in the opposite order from base pi0.5** (`order_demos_vs_rollouts.png`, rows 1-3
  demos, 4-6 rollouts; columns: after first grasp, first release, after second grasp, end):
  - demos (LIBERO-90 human, via jesbu1/libero_90_lerobot): right bowl onto the left bowl where it sits, then the
    stack into the tray. First release 44 cm (median) from the end point; 0 of 35 release first in the tray.
  - base rollouts (43 successes): left bowl into the tray, then the right bowl onto it. 91% release first within
    8 cm of the end point (median 2 cm).
- Eval video (ep 4; `eval_ep4_*.png`): grasps the left bowl (base order), carries it over the tray, never releases
  it there, and later sets it back on the table beside the right bowl.

## Interpretation
- Demos are tagged all positive and never negative, so h(+) - h(-) contains a "demo vs rollout" component. For moka
  that is harmless: pi05_libero was trained on these libero_10 demos, so base already follows the demo strategy.
  Task 64 (libero_90) is new to pi0.5, which found its own order. On task 64 the guidance pushes the base-order
  states toward the demo order, e.g. do not drop a lone bowl in the tray (no demo ever does; they carry a stack):
  the two strategies fight, and the episode completes neither.
- Consistent with the check: with demos the task-64 contrast tripled (0.31 vs 0.09) and grew faster than linearly
  in w, i.e. a large, systematic direction, not a sharper version of the rollout signal.

## Open
- Dose: push 0.5 (18592251) queued. mtd100's task-64 evals (18592252-53) and the big-critic mtbvd arm use the same
  demos; expect the same failure on task 64 (moka side unaffected).
- Fix options: drop the task-64 demos (two-task without them already beats base); or use demos whose order
  matches the policy (none available: all 35 human demos stack first).
