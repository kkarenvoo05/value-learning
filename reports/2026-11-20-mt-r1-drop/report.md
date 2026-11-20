# Why the two-task chain (mt100cv) dropped at round 1 (2026-11-20)

Karen: "investigate why the multi task round 1 dropped". Offline; no new jobs.

## Facts
- mt100cv r0 -> r1 (push 0.99): moka cont 63.7 -> 33.3 (300 / 300), task-64 cont 77.3 -> 31.4 (207 / 300). Both
  evals ran on iris5 / iris6 only, 0 client errors, rate steady across the run. (The r1 moka val 4.5% is void: 280 /
  330 client errors on hung iris7; rerun 18604938.)
- Failures moved to the second stage and time out: timeouts 0.36 -> 0.67 (moka), 0.23 -> 0.69 (task 64); task-64
  failures 137 / 142 = "01" (base bowl in the tray, never stacked); moka failures 001 / 101 rise. Successes are not
  slower (median 386 vs 388; 302 vs 288 steps).
- Ruled out:
  - Infrastructure: no iris7 episodes, no errors.
  - Labels: 40.0% positive in every (round, task). Successes vs failures 46-52% vs 26-28% positive (r0: 49-51% vs 29%).
    Per stage, second-stage frames of successes 50-60% positive vs 20-27% for failures, in both rounds.
  - V: corr(V, MC) 0.42 (r0 0.43).
  - Calibration: contrast curve like r0's (0.105 / 0.386 / 1.137 at w 1 / 4 / 16 vs 0.104 / 0.317 / 0.978); w 13.1 vs 15.5.
  - Training: completed 5000 steps, loss 0.0115 (r0 0.0090, on twice the data).
  - Serving: v = v_ref + w (h+ - h-); the previous round's push is not re-added.
- Same pattern in every small chain, not in the 900 chain:

  | chain   | round-1 table       | r0 -> r1 (val)               | r0 -> r1 (moka cont)    |
  |---------|---------------------|------------------------------|-------------------------|
  | 900     | 900 base + 100 AG   | 75.5 -> 79.1                 | -                       |
  | b100    | 100 base + 100 AG   | 79.7 -> 53.0 (r2 69.7)       | 77.3 -> 45.3 (r2 74.7)  |
  | b100iql | 100 base + 100 AG   | 63.9 -> 55.5                 | 66.0 -> 60.7            |
  | mt100cv | 100 base + 100 AG   | (rerun)                      | 63.7 -> 33.3; t64 77.3 -> 31.4 |

## Interpretation (not proven)
- The dip tracks the share of round-1 data collected by the guided r0 policy: 50% in the small chains, 10% in the 900
  chain. The residual assumes v_ref (base pi0.5) is the behaviour policy of every training episode; from round 1 on,
  half the episodes come from a different (guided) policy, so h(+) - h(-) can absorb a "which policy collected this"
  component. This is the same kind of confound as the task-64 demos (all positive, different strategy), only weaker.
- Seed noise is not excluded: the b100 r1 seed-1 retrain (18593303, requeued off iris7) answers that.

## Correction (2026-11-20, Karen: "for the previous ag round 0-7, didn't they also use the pushed policy's data?")
- Yes: the 900 chain adds 100 guided-policy rollouts every round; guided frame share 10 / 24 / 34 / 42% at r1 / r3 / r5
  / r7 (moka_agr_r*), with no drop (79.1 / 80.9 / 87.6 / 83.0). The small chains are at 47-48% after one round. So the
  guided share alone does not explain the drop. What still differs: (1) the share arrives in one step, from one policy
  (r0), not gradually from seven; (2) 100 base episodes instead of 900 anchor h(+) - h(-); (3) single seeds.
- Reading the oversampled retrain (base 90.8% of frames, still 100 unique base episodes): recovers -> the sudden mix
  shift matters; still drops -> few base episodes (or noise) is the better explanation.

## Tests that would separate the explanations
1. Seed: b100 r1 seed 1 (queued).
2. Data share: retrain mt100cv r1 with the round-0 (base) episodes oversampled to 90% of frames
   (`--oversample-episodes`, as the 900 chain's mix). If the drop goes away, the guided-data share is the cause.
3. Push: r1 at push 0.3 / 0.6 on both cont sets (existing policy, 4 evals). If a lower push is still below r0, the
   direction is wrong, not over-scaled.
4. Pending: b100cv / b50cv r1 evals (held / requeued) add two more small chains.
