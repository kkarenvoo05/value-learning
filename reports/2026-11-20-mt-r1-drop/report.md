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

## Results of the tests (2026-11-20 13:35 – 2026-11-21 10:30)

All push 0.99 unless stated, one seed unless stated, 0 client errors. "Paired" = same episodes / starts.

**1. Training seed (same round-1 data, labels and args, `--seed 1`).**

| chain | r0 | r1 seed 0 | r1 seed 1 | eval |
|---|---|---|---|---|
| b100 | 79.7 | 53.0 | **77.0** | val 330 |
| b100 | 77.3 | 45.3 | **78.0** | moka cont 300 |
| b50cv (50 base) | 49.4 | 11.7 | **10.9** | val, paired, 239/330 (running) |
| b50cv | 52.6 | 14.6 | **14.1** | moka cont, paired, 192/300 (running) |
| mt100cv (50 + 50 base) | 67.5 | 31.4 | **17.2** | val, paired, 169/330 (running) |

b100's drop was mostly training-seed variance; b50cv's and mt100cv's reproduce. Contrast checks do not flag the bad
seed (b100: shift/noise at w 1 0.182 vs 0.198).

**2. Base share of round-1 frames (mt100cv r1 retrained, same 100 unique base episodes, same labels, seed 0).**

| base share | val (paired) | moka cont | t64 cont |
|---|---|---|---|
| r0 (reference) | 64.9 | 63.7 | 77.3 |
| ~52% (r1 as run) | 32.1 | 33.3 | 33.3 |
| 77% (mtos77) | 23.6 (330; r1 24.3 on the same 305) | 30.8 (279/300, TIMEOUT; r1 34.1 paired) | 43.0 |
| 91% (mtos) | 64.3 | 54.3 | 46.0 |
| base pi0.5 | 52.4 | ~55 | 60.7 |

Only ~91% helps, and it recovers r0 only on val (10 fixed states); on fresh layouts it is near base.

**3. Push strength (rep0 starts, 100 each).**

| | moka | task 64 |
|---|---|---|
| base | 55–56 | 58–61 |
| mt100cv r0 @0.99 / @0.6 | 58–66 / 62 | 76–77 / 79 |
| mt100cv r1 @0.99 / @0.6 / @0.3 | 34 / 48 / 52 | 32 / 59 / 57 |
| b50cv r0 @0.99 / @0.6 | 53 / 77 | |

No push makes r1 beat base: the r1 direction is wrong, not over-scaled.

**4. More small chains at r1 (seed 0).** b50cv: val 44.8 → 11.5, moka cont 51.0 → 15.3. b100cv (chain's own V):
val 76.1 → 67.8, moka cont 79.3 → 69.0.

### Reading (interpretation, not proven)
- The drops that survive a second seed are the chains with ~50 base episodes per task (b50cv, mt100cv); with 100
  (b100) it is within seed noise; with 900 there is none, even at 42% guided frames by r7. So the tolerable guided
  share appears to grow with the number of base episodes per task (explanations 2 and 1 together).
- Oversampling the same 100 episodes to 91% fixes val but not fresh layouts: weight on base data helps, but too few
  distinct base layouts limits generalisation.
- Single-seed retrains vary by up to ~25 points, so every remaining number needs ≥ 2 seeds before it is attributed.
- Pending: seed-1 evals (b50cv, mt100cv incl. both cont sets), b200cv r1 (200 base), b100 rounds 3–9 (volume).

## Seed-1 finals and an eval correction (mentor session, 2026-11-21 14:00)

Recomputed from `episodes.csv` (paired by start; 0 client errors in every eval below; script
`reports/2026-11-21-nov21-update/compute_r1_seeds.py`).

| chain | eval | r0 | r1 seed 0 | r1 seed 1 | s1 − s0 paired |
|---|---|---|---|---|---|
| b50cv | val 330 | 44.8 | 11.5 | **10.0** | −1.5 [−5.8, +2.7] |
| b50cv | moka cont 300 | 51.0 | 15.3 | **14.7** | −0.7 [−6.0, +4.7] |
| mt100cv | val | 65.8 | 32.1 (305) | **14.5** (330) | −17.7 [−24.3, −11.1] |
| mt100cv | moka cont | 63.7 | 33.3 | **12.3** (227/300, stalled) | −22.0 [−30.0, −13.7] |
| mt100cv | t64 cont | 77.3 | 33.3 | **73.3** (296/300, stalled) | **+40.5 [+32.8, +47.6]** |

- **Correction:** mt100cv r1 seed 0 val eval 18598985 has 280 client-errored episodes logged as failures (4.5%);
  the rerun 18604938 (32.1%, 305 trials) is the valid number. The 4.5% had been quoted in
  `reports/2026-11-20-plateau-and-matched-budget/` and the Nov 20 site page (both corrected).
- Observation: seed 1 of mt100cv drops further on moka (val 14.5, cont 12.3) but **does not drop on task 64**
  (73.3 vs r0 77.3, −4.1 [−10.1, +1.7]). Seed 0 dropped on both tasks.
- Interpretation: within one two-task policy, which task collapses changes with the training seed, so "the r1 drop"
  is not one task-independent failure. Any factorial arm needs both seeds and both eval sets before reading.
- Both mt100cv seed-1 cevals (18615179 / 80) have logged "Still waiting for server" since 10:40.
