# Nov 21 update: round-1 seeds, base share, offset G0, push direction (mentor session)

Compiled 2026-11-21 14:00 PT. The experiments are Karen's (`docs/libero_plan.md` log 11-20 11:00 to 11-21 12:50;
reports `2026-11-20-mt-r1-drop`, `2026-11-20-offset-g0`, `2026-11-21-r1-factorial`, `2026-11-21-push-direction`).
This report recomputes the round-1 comparisons from `episodes.csv` with paired CIs, finishes the seed-1 numbers,
reruns the push-direction analysis on the policies finished so far, and collects the charts (`highlights.html`).

## Question

1. Is the small chains' round-1 drop training-seed noise or real, and per chain / task?
2. Does a larger base share of round-1 frames remove the drop?
3. Do deliberate end-effector offsets create choices worth ranking (imagination premise)?
4. Does round 1 re-apply round 0's push ("double push")?

## Setup

- LIBERO libero_10 task 8 (moka), π₀.₅ frozen, residual AG at push 0.99. val = 330 trials (10 val states × 33,
  matched noise 1), moka cont = 300 fresh layouts (`moka_cont100`), t64 cont = libero_90 task 64, 300 fresh layouts.
- Paired by start; bootstrap 5,000, rng 0 [ours]. Client-errored episodes checked (`episodes.csv` `error`): 0 in
  every eval used. `compute_r1_seeds.py` → `r1_seeds.json` (every number in the tables).
- Push direction: Karen's `scripts/push_direction_analyze.py` on the 10 of 17 policies with `push.npz` at 13:40
  (symlinked into a scratch root so her `analysis.json` is untouched) → `push_direction_partial.json`.

## Tables

Round 1 vs training seed (success %, paired differences in points):

| chain | eval | base | r0 | r1 s0 | r1 s1 | s1 − s0 |
|---|---|---|---|---|---|---|
| b100 | val | 52.4 | 79.7 | 53.0 | 77.0 | +23.9 [+17.6, +30.6] |
| b100 | moka cont | 55.0 | 77.3 | 45.3 | 78.0 | +32.7 [+25.3, +40.0] |
| b100cv | val / moka cont | | 76.1 / 79.3 | 67.6 / 69.0 | – | |
| b50cv | val | 52.4 | 44.8 | 11.5 | 10.0 | −1.5 [−5.8, +2.7] |
| b50cv | moka cont | 55.0 | 51.0 | 15.3 | 14.7 | −0.7 [−6.0, +4.7] |
| mt100cv | val | 52.4 | 65.8 | 32.1 (305) | 14.5 | −17.7 [−24.3, −11.1] |
| mt100cv | moka cont | 55.0 | 63.7 | 33.3 | 12.3 (227/300) | −22.0 [−30.0, −13.7] |
| mt100cv | t64 cont | 60.7 | 77.3 | 33.3 | 73.3 (296/300) | +40.5 [+32.8, +47.6] |

mt100cv round 1 vs base share of frames (seed 0): val 32.1 / 23.6 / 64.2, moka cont 33.3 / 30.8 (279) / 54.3,
t64 cont 33.3 / 43.0 / 46.0 at ~52 / 77 / 91%.

Offset G0 (Karen's numbers, verified against `analysis/summary.json` and `cv_gain_vs` in
`scripts/libero_steer_analyze.py`): best-of-8 minus plain chunk +12.5 [+5.6, +19.8] (1 chunk), +12.1 [+4.9, +19.3]
(held 4), noise G0 +0.6 [−2.8, +4.2]. Per direction: no CI above 0 in both probes.

Push direction (arm motion, 30-d; per-frame cosine / β with episode-bootstrap CIs):

| pair | val outcome | cos | β |
|---|---|---|---|
| b100cv r0 vs 900 r0 | – | 0.58 [0.52, 0.64] | 0.88 [0.78, 0.95] |
| mt100cv r0 vs 900 r0 | – | 0.54 [0.48, 0.60] | 0.68 [0.49, 0.80] |
| mt r1 91% base vs mt r0 | 64.2 (no drop) | 0.28 [0.22, 0.35] | 0.56 [0.42, 0.63] |
| mt r1 77% base vs mt r0 | 23.6 (drop) | 0.31 [0.26, 0.36] | 0.54 [0.41, 0.63] |
| mt r1 s0 vs mt r0 | 32.1 (drop) | −0.11 [−0.19, −0.03] | 0.37 [0.12, 0.54] |
| b100cv r1 vs b100cv r0 | 67.6 (−8.5) | 0.06 [−0.04, 0.16] | 0.59 [0.21, 0.84] |

## Verdict

Observations:
- b100's round-1 drop does not reproduce with a second training seed (77.0 vs 53.0 on the same data). b50cv's does
  (10–15% with both seeds). mt100cv drops on moka with both seeds; on task 64 seed 1 holds (73.3) and seed 0 drops.
- **Correction:** the Nov 20 "mt100cv r1 moka val 4.5%" came from eval 18598985, which logged 280 client errors as
  failures; the valid rerun is 32.1% (305 trials). Fixed in `reports/2026-11-20-plateau-and-matched-budget/`
  (`compute_evals.py` now drops errored episodes and pins the round-1 eval ids), `docs/findings.md`, and the Nov 20
  site page.
- 91% base frames restores val but not fresh layouts; 77% does nothing.
- Offset candidates pass the G0 gate (+12 points); no fixed direction helps.
- Round-1 pushes are not more aligned with round 0 in the retrains that drop; two r0s from different chains agree
  more than any r1 does with its r0.

Interpretations:
- Single retrains on identical data can differ by 25–40 points on one eval set, and which task collapses can change
  with the seed. Every chain number before today is one seed; the factorial's two-seed, ≥ 10-point rule is needed.
- The double-push hypothesis is not supported by the 10 policies scored so far.
- Offset G0 makes imagination worth testing on structured proposals; the +12 is an oracle upper bound.

## Caveats

- mt100cv seed-1 cont evals are partial (moka 227/300, t64 296/300); both show "Still waiting for server" since
  10:40.
- Push direction: 10 of 17 policies; the b50cv and b100 r0 / r1 pairs and all seed-1 policies are not in yet.
- Offset G0: 20 base episodes, one offset size; hold-4 overshot (2.8–3.1 cm).

## Next

- Read the factorial (arms A–D, 2 seeds) by Karen's rule when done (~1 day).
- Finish the push-direction scoring on all 17 policies.
- G1 on offset proposals (Karen's pre-registered next step).
