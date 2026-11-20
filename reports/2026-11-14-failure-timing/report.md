# Why the plateaued arms fail: pot timing + video review (2026-11-14)

Karen: "start with the video review and the first-pot timing on AG round 4 and DSRL agent_1300".
Arms: AG round 4 (18463162, 0.794) and DSRL s0 agent_1300 (18500794, 0.824); base (18355478) and the round-0
residual (18438187) for reference. 330 val trials each (10 val init states x 33). Every failure is a 520-step timeout.

## Method

- Pot state per video frame: the pot detector of `libero_failure_taxonomy.py` (same training frames, MLP, smoothing)
  on ResNet-18 agentview features of every 4th frame. `scripts/libero_failure_timing.py`.
- Check: the detector's end state on failures vs the simulator's `goals_end`: 0.91 (AG r4), 0.83 (DSRL 1300); the
  taxonomy's was 0.93. DSRL errors are mostly "no pot" read as "one pot" (6 of 20).
- Video review: 20 contact sheets (14 frames each, every ~40 steps), 2 failures per hard state (12, 13, 42, 44) plus
  2 from the rest, per arm, sampled with seed 0; 14 read (7 per arm).

## Timing

| | AG r4 | DSRL 1300 | residual r0 | base |
|---|---|---|---|---|
| successes: first pot (median step) | 184 | 160 | 192 | 204 |
| successes: first -> second pot (median / p90) | 182 / 223 | 160 / 173 | 200 / 229 | 234 / 277 |
| failures: first pot (median, IQR) | 196 (184-240) | 200 (152-344) | 192 (176-208) | 244 (204-382) |
| failures ending with 1 pot / 0 pots (sim) | 61 / 7 | 40 / 18 | n/a | n/a |
| one-pot failures whose first pot left >= the p90 success gap | **84%** | **75%** | | |
| failures with a detected pot drop (successes) | 44% (11%) | 48% (13%) | 41% (16%) | 31% (28%) |

- The one-pot failures are **stuck, not slow**. Their first pot lands on time (median 196 / 188), which leaves about
  320 steps, while successes need 182 / 160 (p90 223 / 173) for the second pot. A longer cap would rescue the
  remaining ~16-25% (first pot late) and some slow grasps, not the majority.
- About half of the failures show a pot coming off the stove at some point (vs 11-13% of successes, the detector's
  noise floor), i.e. knock-offs.
- DSRL's 18 no-pot failures: 10 never placed a pot (5 at state 44, 4 at 12), 8 placed one and lost it.

## Video review (14 failures)

| pattern | AG r4 | DSRL 1300 |
|---|---|---|
| hovers at a pot for 150-250 steps without closing the gripper | 3 (states 13, 40, 42) | 2 (9: second pot; 12: first pot) |
| slow second grasp, still carrying at the cap (would likely finish with more time) | 1 (12) | 0 |
| knocks the first pot off while placing the second | 2 (44, 44) | 1 (13) |
| keeps adjusting the first pot on the stove for ~200 steps, then knocks it off or runs out of time | 1 (13) | 2 (42, 42) |
| holds the first pot pressed against the stove's side, never lifts it over | 0 | 2 (44, 44) |

- AG's typical failure: first pot placed, then a hover loop above the second pot.
- DSRL's state-44 weakness (0.42-0.70 vs AG 0.70-0.85) is the pressed-against-the-stove failure: the pot is carried
  too low.
- Knocking the first pot over is shared; once a pot lies on its side neither policy recovers it.

## Reading

- Not a time-budget problem for most failures: a longer cap is a diagnostic at most, not a fix.
- Two failure modes that more rounds of the same data do not target: (1) hover loops at a grasp, (2) collisions with
  the placed pot. Both are second-stage events, but the collection spends most of each episode on the easy first pot.
- Candidate next steps (not yet run): data that starts mid-task (first pot already on, saved MuJoCo states), so the
  rounds see many second-pot grasps and placements; a check whether the hover is a gripper-closing problem (gripper
  action stays open while the arm oscillates) using rollouts with raw actions.
- Caveat: the 10 val states are both the diagnosis set and the eval set; confirm the patterns on train-state rollouts
  before tuning to them.
