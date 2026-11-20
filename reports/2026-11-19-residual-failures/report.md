# Why more AG data stops helping: the residual failures (2026-11-19)

Karen: "do an in-depth analysis of the remaining failures and see if my current experiments are not currently
handling them". Offline only; nothing new was run.

## Question
AG plateaus at ~85% (rounds 5-7). Which failures are left, does the training data contain the behaviour that would
fix them, and do the labels point at it? Which running / planned experiments target which failure?

## Data and method
- Episodes: every train episode of `tables/moka_agr_r7` (the round-7 training table): 900 base pi0.5 rollouts
  (table round 1) + 7 AG collections of 100 (table rounds 2-8 = collected by AG policies r0-r6 at push 0.99;
  60 train-state + 40 fresh-layout episodes each). Saved goal predicates, pot positions, hand-to-pot vectors,
  actions, robot state. Plus the four round-6 variant collections (orig r6, pmix, push 1.5, noise x2).
- Labels: `labels/agr_vb40_r7/labels.parquet` (the labels round 7 trained on; 177,931 labelled frames).
- Groups: base / AG r1-3 (table rounds 2-4) / **AG r4-7** (table rounds 5-8, the plateau policies).
- Failure classes [ours; thresholds as `scripts/libero_week9_nov17.py` where shared]: a pot is placed after 10
  consecutive on-stove steps and lost after 20 off; lifted = > 3 cm above its start height.
- Grasp pose [ours]: at a successful second-pot lift, pot - eef = (-2.1, 0.2, 6.6) cm, sd ~1 cm (200 successes).
  "At the grasp pose" = within 3 cm of it; "low" = hand >= 1.5 cm below it. Stall = within 8 cm of the grasp pose,
  eef speed < 0.2 cm / step.
- Caveat: these are collections, not the 330-trial val evals (no saved sim state there). Same policies and push.
- Scripts: `analysis/fail_deep.py` (per-episode features -> `fail_feats.parquet`), `analysis/stall.py`
  (-> `stall_feats.parquet`), `analysis/labphase.py` (labels by phase); the tables below were computed inline from
  these files.

## Results

### 1. What is left (failures per 100 episodes)

| failure | base | AG r1-3 | AG r4-7 | share of AG r4-7 failures |
|---|---|---|---|---|
| second pot lifted, not placed in time | 15.6 | 12.0 | 7.8 | 40% |
| second pot never lifted: hand stalls low beside it | 7.7 | 7.0 | 4.5 | 23% |
| second pot never lifted: other | 8.2 | 2.7 | 2.5 | 13% |
| placed pot lost (slid or knocked off) | 3.3 | 3.7 | 2.5 | 13% |
| pot tipped over | 0.3 | 0.3 | 0.8 | 4% |
| no pot ever placed | 7.6 | 3.0 | 1.2 | 6% |
| **success** | 57.3 | 71.3 | 80.8 | (77 failures) |

- AG removed most "no pot" and mis-aimed second grasps (-84%, -70%). The modes that shrink least are pot loss
  (-24%), the low stall (-42%) and tipped pots (no drop).

### 2. The mechanisms
- **Low stall at the second pot.** In never-lifted failures the hand's closest approach is 1.5-2.5 cm *below* the
  grasp pose (79 of 80 with a > 2 cm height error are too low, 1 too high). It then sits within 8 cm for a median
  180 steps at 0.13 cm / step (successes: 46 steps at 0.31), gripper closed only 12% of that time (successes 41%).
  56% of one-pot failures never close the gripper at the grasp pose; successes do it once in 92% of cases.
- **Slow first pot, then out of time.** In AG r4-7, "lifted, not placed" failures place the first pot at a median
  step 299 (successes 174) and are still holding the second pot at the 520 cap in 91% of cases; 55% had less time
  left after the lift than 90% of successes need (78 steps). The delay is the *carry*: the first pot is lifted almost
  on time (step 155 vs 123) but takes 234 steps to set down (successes 51), carried 3.9 cm above its start (successes
  9.2) with the gripper closed only 59% of the time: dragged low against the stove, released and regrasped.
- Common thread: **height errors** (hand / pot too low near the stove and the second pot), not mis-aim or slowness
  of motion (eef speed while carrying 0.68 vs 0.78 cm / step).
- Not a fixed set of hard states: on the train states the 5 worst states hold 42% of AG r4-7 failures vs 36%
  expected if every state had the same rate (95th pct 43%). On the val evals every state succeeds sometimes; state 13
  is the weakest (48-73% in r5-r7). Round 6's drop shows on 9 of 10 val states (round-level, not state-level).

### 3. Is the fix in the data?

| behaviour that would fix it | in the 1,600 training episodes |
|---|---|
| lost pot put back on the stove | 7 re-placements of 62 losses; 4 of the 62 episodes still succeed |
| tipped pot recovered | 0 of 22 |
| escape from a >= 60-step low stall | 50 of 189 such episodes succeed (26%; >= 100 steps: 12 of 111) |
| second grasp retried after a failed close | 59 of 1,053 successes (6%) |

- Recovery from a lost or tipped pot is essentially absent: AG reweights behaviours the data contains, so these
  (17% of late failures) cannot be learned from more of the same data.
- Stall escapes are present (26%), so they are learnable in principle, but see 4.

### 4. Do the labels point at the fix?

| phase (AG rounds) | share of frames | P(pos) success eps / failure eps | AUC of A for eventual success |
|---|---|---|---|
| first pot | 48% | 0.42 / 0.32 | 0.55 |
| second-pot approach / grasp | 37% | 0.45 / 0.21 | 0.64 |
| carry / place second pot | 15% | 0.56 / 0.21 | 0.72 |
| low-stall frames | 4% of all frames | 0.38 / 0.15 | 0.60 |

- Labels are weakest in the first-pot phase, where the slow-carry failures start (AUC 0.55; base 0.57).
- **The negative class is mostly successes in late rounds**: 65-75% of negatives in table rounds 5-8 come from
  successful episodes (round 1: 42%), and failure frames are only 20-28% of each late round. The contrast the head
  learns is mostly fast vs slow success.
- **Dilution**: base rollouts are 58% of labelled frames (104k of 178k); each AG round adds ~6%. The plateau
  policy's own failures (15-23 episodes per round) are ~1-2% of frames per round.

### 5. Do the round-6 variants change the failure mix? (one collection of 100 each)

| collection | success | lifted, not placed | never lifted | pot lost / tipped | no pot |
|---|---|---|---|---|---|
| push 0.99 (original r6) | 85 | 6 | 4 | 3 | 2 |
| push 1.5 | 87 | 3 | 4 | 3 | 3 |
| noise x2 | 85 | 5 | 6 | 3 | 1 |
| random push per episode | 78 | 10 | 3 | 4 | 5 |

- Within random push: push 2.0 places the first pot sooner (median step 154 vs 176-183) but loses a pot in 3 of 31
  episodes (other pushes 1 of 69). Counts are too small to say a variant shifts a specific mode; what random push
  clearly adds is more failures (22 vs 15) for the negative class.

## Verdict

Observations
- The plateau policies fail mainly in the second half: late second-pot placement (40%) and a low stall beside the
  second pot (23%), plus pot loss / tipping (17%). Both main modes involve the hand or pot being too low.
- Recovery from a lost or tipped pot is (nearly) absent from 1,600 training episodes. Stall escapes are present.
- Late-round labels mostly separate fast from slow successes; the latest policy's failures are ~1-2% of the frames.

Interpretations
- More data of the same kind cannot fix pot loss / tipping (no support) and fixes the stall slowly (weak, diluted
  signal on 4% of frames). This is a better explanation of "more data, no gain" than guidance strength (ruled out)
  or the reference (no gain).
- The time cap is a partial factor: about half of the late-placement failures would likely finish with more time.

## Which experiments target which failure

| failure (share of late failures) | exploration (pmix / push 1.5 / noise) | labels (IQL) | episodes per round | stacked reference | multi-task |
|---|---|---|---|---|---|
| late placement (40%) | indirectly (more slow episodes as negatives) | maybe | slowly | no | no |
| low stall (23%) | maybe (more escapes?) | maybe (credit for the escape step) | slowly | no | no |
| pot lost / tipped (17%) | no: adds losses, not recoveries | no | no | no | no |
| first-pot carry height | no | weak signal (AUC 0.55) | slowly | no | no |

Not covered by any running or planned run:
1. **Dilution / label focus**: downweight the 900 base rollouts or oversample the latest round's failures; labels
   tied to failure (positives / negatives from success vs failure, not speed). Uses existing data, one train + eval.
2. **Second-half and recovery data**: mid-task starts (saved sim states) at the second pot, and from a knocked or
   tipped first pot, so rounds see the hard part many times.
3. **Time-cap diagnostic**: eval the best round with a 700-step cap (diagnostic only, not a fix) to measure how
   much of the ceiling is "too slow".

## Caveats
- Collections, not val evals; thresholds (3 cm lift, 3 cm grasp pose, 0.2 cm / step stall) are ours.
- Phase AUCs measure whether A predicts the episode outcome, a proxy for action quality.
- One collection per policy; the round-6 variant counts are small.
- No new video review; the mechanisms come from the saved state, consistent with the 11-14 video review.
