# Nov 13–20 results: plateau, distillation, noisy judge, matched-budget per state

Compiled 2026-11-20 (mentor session) from the eval run folders in `/iris/u/ankile/karenvo/libero/runs/`, through
Nov 20 10:00 PT. Most experiments are Karen's (design and chronology in `docs/libero_plan.md` log, 11-13 to 11-20);
this report recomputes their numbers on one footing and adds the matched-budget per-state comparison.

## Question

1. Where do AG rounds and DSRL level off on moka, and does anything tried this week move the plateau?
2. Can the gain be distilled back into π₀.₅ (deployable without a residual head or DSRL actor)?
3. How do AG and DSRL hold up when the success judge is noisy?
4. At a matched budget, does either method leave a start state unsolved that the other solves?

## Setup

- LIBERO libero_10 task 8 (moka pots), π₀.₅ `pi05_libero` frozen. 330 held-out trials = val states 9 12 13 17 25
  32 40 42 44 48 × 33 repeats, env seed 7, matched noise seed 1 (`MATCHED_NOISE=1`). Base 52.4% (18355478).
- Success with Wilson 95% intervals; differences paired by (init state, repeat), bootstrap 5,000 draws, rng 0.
  [ours]
- Every policy is **one training seed**.
- Files: `evals.csv` (every eval: key, success, CI, n), `paired.csv` (every comparison), `per_state_matched.csv`
  (every trial of base, AG r5, DSRL s0 @1300 / @1357, DSRL s1 @1300), `compute_evals.py` / `compute_per_state.py`
  (glob patterns → run folders), `highlights.html` and `per_state_matched.html` (charts).

## Tables

Plateau (330 trials unless noted):

| policy | success | note |
|---|---|---|
| AG rounds r0–r7 (900 base + 100/round) | 75.5, 79.1, 74.5, 80.9, 79.4, **87.6**, 77.3, 83.0 | one seed per round |
| AG r5 / r7, eval noise seed 2 | 84.8 / 80.0 | −2.7 / −3.0 paired |
| DSRL seed 0, own episodes 100 … 1357 | 54.8, 48.2, 79.7, 83.0, 82.7, 65.2, 79.7, 83.0, 80.9, 82.4, 86.1 | at 100, 200, 400, 600, 800, 900, 1000, 1100, 1200, 1300, 1357 |
| DSRL seed 1 at 1300 / 1800 / 2400 / 3000 / 3300 | 87.0, 89.1, **90.0**, 86.1, 88.2 | 3300 − 1800: −0.9 [−5.8, +3.9] |

Paired differences (percentage points, 95% CI):

| comparison | Δ |
|---|---|
| DSRL s1 @2400 − AG r5 | +2.4 [−2.4, +7.3] |
| push 1.8 − push 0.99, r5 / r7 | −9.7 / −11.5 (raw 77.9 / 71.5) |
| w = 1 (r5 / r6 / r7) | 63.6 / 62.7 / 63.9 |
| anchored reference − standard, r5 / r7 | −9.7 [−15.5, −3.9] / −28.2 [−34.2, −22.1] |
| learned no-tag branch (best, w 2) − standard guidance on the same model | −7.9 [−13.9, −1.5] |
| stacked residual − starting policy, r0 / r5 / r6 | −7.3 [−13.6, −0.6] / −0.9 / −1.2 |
| IQL labels − value-binned, r7 data / b100 round 0 | −4.2 [−10.0, +1.5] / −15.8 [−22.1, −9.1] |
| no demos − with demos, 900 rollouts / 50 rollouts | +5.8 [−0.6, +11.8] / **+27.9 [+20.9, +35.2]** |
| round 6 redo − original r6: rerun / extra noise / push 1.5 / random push | +3.0 / +3.0 / +4.8 / **+7.0 [+1.5, +12.4]** |
| random push − r6 rerun | +3.9 [−1.8, +9.4] |
| 700-step limit − 520, r5 | +0.3 |
| b100 chain: r1 − r0 / r2 − r0 | **−26.7 [−33.6, −19.4]** / −7.6 |
| two-task chain (mt100cv) r1, moka val | 32.1% on 305 trials (r0 65.8%); corrected 11-21: the first eval (18598985) logged 280 client errors as failures and read 4.5% |

Distillation (LoRA, filtered BC round 1's recipe, `--chunk-targets sampled`, final checkpoint; teacher: 300 rollouts
on train states, successes only):

| teacher (collection success) | distilled success | vs base |
|---|---|---|
| base (57.3%) | 49.7% (executed targets 50.3%) | −2.7 [−9.1, +3.6] |
| hybrid base → DSRL actor (76.7%) | 80.9% | +28.5 |
| DSRL (85.0%) | **88.2%** | +35.8 [+29.7, +41.8]; −0.9 vs DSRL s1 @1800 |
| AG (85.3%) | **89.1%** on 266/330 (eval stopped) | +38.0; +0.8 vs AG r5 on the same trials |

Noisy judge (each episode's success flipped with p = 0.25 both ways):

| method | flipped − clean |
|---|---|
| AG round 0 | +0.6 [−5.5, +6.7] (76.1 vs 75.5) |
| DSRL @400 / @600 | **−36.7 [−44.2, −29.1] / −47.9 [−54.5, −41.2]** (43.0 / 35.2) |

Matched budget per state (AG r5 at 1.4k episodes incl. the 900 base pool vs DSRL s0 @1300 / @1357 and s1 @1300):
the worst state is 24/33 for AG (state 13), and 15/33 (state 44), 19/33 (state 12) and 17/33 (state 13) for the
three DSRL agents; base's worst is 4/33. Of 30 per-state paired tests, 9 exclude zero (about 1.5 expected by chance). No trial is failed by all four
agents (7 by three).

## Verdict

Observations:
- Both methods level off at 85–90% on the 330 trials. DSRL seed 1 does not improve after 1,300–1,800 episodes;
  AG rounds after round 0 move within 74.5–87.6% with no trend.
- Guidance strength is not the limit: push 0.8–1.2 is flat, higher pushes and small fixed weights are worse.
- Distilling DSRL or AG successes into π₀.₅ matches the teacher (88–89%); distilling base's own successes does not
  beat base.
- AG is unaffected by 25% symmetric label flips; DSRL with the same flip rate ends below base.
- Removing the 29 demos raises round 0, strongly at 50 rollouts.
- Moving the guidance reference (anchored, learned no-tag, stacked) and IQL labels did not help.
- Small fixed-budget chains drop sharply at round 1; the 900 chain never did.
- At matched budget, no start state is solved by one method and unsolved by another; per-state gaps are a few
  trials out of 33.

Interpretations:
- The plateau is in what the learned head (or DSRL actor) can extract from this data, not in how hard it is applied.
- Distillation is a workable deployment path for either method; the fine-tune recipe no longer costs points on its
  own (the earlier −13 to −16 came from a different recipe).
- For a VLM judge with real errors, AG is the safer method; DSRL's critic propagates flipped rewards.
- The demos may be part of the small-chain drop: every b50 / b100 / mt chain still trains with them.

## Caveats

- One training seed per policy; two evals of the same or a near-identical policy agree on only 72–75% of trials
  (r6 push 0.8 vs 0.99: 72%; Nov 13 control rerun of r0: 75%), so single differences below ~6 points are not
  readable.
- The DSRL seed-0 checkpoint named `ep1800` trained on 1,357 of its own episodes (the run restarted at row 443).
  Earlier docs called it 1,800.
- The noisy-judge test flips labels independently and symmetrically; a real judge's errors are likely lopsided and
  concentrated on near-misses (one pot placed).
- AG-distilled eval stopped at 266/330 trials.
- Per-state results cover 10 of the 50 saved states; the start region is a few-centimetre jitter (stove 2 × 2 cm,
  each pot 5 × 5 cm, fixed yaw).

## Next

- Second training seeds (or head ensembles) for AG rounds before reading single-round gains.
- Re-run the small chains without demos; this tests whether the demos drive the round-1 drop.
- Noisy-judge test with an asymmetric, near-miss-concentrated error model; held-out evals at the measured zima
  judge FP / FN rates.
- Distillation from the union of AG and DSRL successes; finish the AG-distilled eval to 330.
