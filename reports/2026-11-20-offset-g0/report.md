# Structured-offset G0: can deliberate end-effector offsets create choices worth ranking?

**Status 2026-11-21: done. Gate passes in both probes; no single direction replicates (see Results).** Setup below was written before any result.
Karen: "set up the structured-offset G0 probe in sim".

## Question

Action imagination (Advantage Guidance §6.3) trains on extra candidate actions per state, ranked by a critic. It only
helps if, at a given state, some candidates really do better than others, and better than the plain policy. G0 with
π₀.₅'s own samples said no: candidates end ~1.9 mm apart, and even the simulator's cross-validated best-of-8 gain was
+0.006 [−0.028, +0.042] (+0.006 / +0.007 / +0.015 at start noise ×1 / ×1.5 / ×2; `reports/2026-11-04-steer-probe/`,
findings "Policy fine-tuning"). Scaling the start noise does not widen the choices usefully: the flow pulls start
points back to the same action until it breaks (×3: 28% success, gripper spread 120×;
`reports/2026-11-09-noise-sweep/`).

This probe asks the same question with **structured** candidates: the same policy chunk plus a constant end-effector
offset of ~2 cm along ±x / ±y / ±z. The residual failures are mostly height errors (hand 1.5–2.5 cm too low at the
second pot and while carrying; `reports/2026-11-19-residual-failures/`), about 10× base's sample spread, which start
noise cannot reach.

## Setup

- **Code:** `scripts/libero_steer_probe.py --args.cand-offsets ...` (offset mode, module docstring), analysis
  `scripts/libero_steer_analyze.py` (offset summary added), submit `scripts/offset_g0_submit.sh`.
- **Policy:** base pi05_libero, plain openpi server (as G0 v1). The trunk episode and every continuation are the plain
  policy; only the candidate segment is offset.
- **Candidates (N = 8) [ours]:** `none x+ x- y+ y- z+ z- z+0.5`. Candidate 0 = the plain chunk. All candidates share
  the same queried chunk, so they differ only by the offset (G0's candidates differed by noise).
  - Offset = constant addition to action dims 0-2 (position deltas), clipped to [−1, 1]; rotation and gripper
    untouched.
  - Size: per-step offset = 0.02 m × scale / (0.0075 × 5 × hold), so the summed offset moves the end effector ~2 cm
    open loop (1 cm for `z+0.5`). 0.0075 m per unit of summed action over one chunk is our calibration from G0's
    candidates (within-state end-effector difference vs summed action difference, `steer_v2_scale2p0` 18329176,
    r 0.76–0.84 per axis); fake-policy smoke test: 0.8 cm achieved vs 0.67 predicted for one chunk.
  - Per-step size: hold 1 → 0.533 (policy's own |action| p95 is 0.44–0.75 per axis), hold 4 → 0.133.
  - `z+` is the pre-specified hypothesis direction (height errors). `z+0.5` is a gentler version of it.
- **Two probes, reported separately, never pooled:**
  - **hold 1:** offset on the first chunk only (5 steps, one decision, as G0).
  - **hold 4:** offset held for 4 chunks (20 steps, 1 s). After the first chunk the policy is re-queried from the
    candidate's own state and the offset added again (closed loop), so the policy can partly correct.
- **States and branching (as G0 v2):** the 10 val init states × 2 base episodes, branch steps 40 / 130 / 220 / 310
  (≈ first-pot approach / first-pot carry / second-pot approach / second-pot carry), so 80 branch points per probe.
  R = 4 continuations per candidate [ours: G0 used 2, judged underpowered].
- **Achieved displacement** is recorded (`br_seg_eef`) and reported per label, since closed-loop correction and
  clipping change it.

## Statistics

As G0 (`libero_steer_analyze.py`): between-candidate variance of success (binomial noise removed), cross-validated
oracle gain (pick the best candidate on one repeat, score on the others). New for this probe:
- `cv_gain_vs_cand0`: the picked candidate's held-out success minus candidate 0's (the plain policy).
- Per label `d_vs_none` = mean over branch points of p_c − p_0 (paired).
- CIs from a bootstrap over **base episodes** (the 4 branch points of one episode share its trunk).

## Reading rule (pre-registered)

- **Gate (imagination premise):** `cv_gain_vs_cand0` with a 95% episode-clustered CI above 0, in either probe.
  Picking among offset candidates then beats running the plain policy, so there is something worth ranking.
  - **Pass →** G1 on offset proposals: train a critic Q(s, a) on data that contains these offsets and test whether it
    ranks them at held-out branch points. Only then imagination (G2/G3).
- **`cv_gain` vs the candidate mean is not the gate.** It will be positive if some offsets simply hurt (avoiding a bad
  candidate), which says nothing about beating base. Reported only.
- **A single direction helps** (`d_vs_none` CI above 0): a fixed push helps at those states. This is a residual/push
  result, not imagination. It counts as evidence only for `z+` / `z+0.5` (pre-specified) or a direction that
  replicates in both probes. With 7 labels × 2 probes, any other single hit is treated as chance. Read it by branch
  step too (the height errors sit at the second pot, branch steps 220 / 310).
- **Fail:** gate CI includes 0 in both probes and no direction helps → deliberate ±2 cm offsets do not create useful
  choices either. With G0 and G1 this closes imagination on this task for both noise and offset proposals.
- If offsets mostly hurt (all `d_vs_none` < 0) with clearly larger between-candidate variance than G0 (SD 0.08):
  the outcome is sensitive to position, but the plain policy is already near the best of these directions.

## Caveats (known before running)

- One offset size (2 cm), axis-aligned, constant in the world frame. Height errors are relative to the pot, so the
  useful offset may differ by state; a fixed +z may overshoot where the hand is already right.
- Hold 4 lets the policy correct during the hold; achieved displacement may be well below 2 cm.
- 80 points × R 4 per probe: G0's CI half-width was ~0.035 at 40 points × R 2. Expect about ±0.02–0.03 here.
- Sim only. On the real robot this oracle cannot be run (no mid-episode restore); see the 2026-11-20 discussion
  in `docs/libero_plan.md`.

## Run

| step | job | what |
|---|---|---|
| GPU smoke | 18611395 | val state 9, 120 steps, `none z+ z-`, hold 4, branch 40, R 1: real-server path + achieved displacement |
| probe hold 1 | 18611396 | afterok smoke; 20 base episodes, 8 candidates, R 4 |
| probe hold 4 | 18611397 | afterok smoke; same, hold 4 |

Score when done (vl env): `.venv/bin/python scripts/libero_steer_analyze.py --run <run dir>` per probe; the offset
summary is `analysis/summary.json` → `offset_probe`, per-point rows `analysis/offsets.csv`.

## Results (2026-11-21)

Both probes COMPLETED (h1 18611396 06:55, h4 18611397 07:10), 20 base episodes × 4 branch points × 8 candidates × R 4,
restore max |diff| 4e-17 / 1e-16. Scored with `libero_steer_analyze.py --run <dir>` (`analysis/summary.json`).

| | hold 1 | hold 4 | G0 (noise, ref) |
|---|---|---|---|
| base episodes successful | 10/20 | 8/20 | |
| p(none) over branch points | 0.469 | 0.472 | |
| between-candidate SD | 0.20 | 0.22 | 0.08 |
| **cv_gain_vs_cand0** (gate) | **+0.125 [+0.056, +0.198]** | **+0.121 [+0.049, +0.193]** | +0.006 [−0.028, +0.042] |
| cv_gain vs candidate mean (not the gate) | +0.10 [+0.07, +0.14] | +0.11 [+0.08, +0.15] | |
| achieved end-effector displacement (median, 2 cm labels) | 2.1–2.6 cm | 2.8–3.1 cm | 0.19 cm |

CIs: 95%, bootstrap over base episodes. Per direction, `d_vs_none` (p_c − p_none) [95% CI]:

| label | hold 1 | hold 4 |
|---|---|---|
| x+ | +0.053 [−0.003, +0.119] | +0.009 [−0.097, +0.119] |
| x− | −0.053 [−0.144, +0.031] | −0.047 [−0.125, +0.031] |
| y+ | −0.009 [−0.088, +0.072] | +0.019 [−0.047, +0.078] |
| y− | +0.097 [−0.013, +0.194] | +0.047 [−0.047, +0.141] |
| **z+** (pre-specified) | −0.019 [−0.131, +0.094] | +0.025 [−0.075, +0.116] |
| z− | +0.106 [+0.034, +0.184] | +0.016 [−0.072, +0.106] |
| z+0.5 (pre-specified) | +0.006 [−0.056, +0.072] | +0.000 [−0.066, +0.062] |

h1 z− by branch step: +0.19 / +0.14 / +0.06 / +0.04 at 40 / 130 / 220 / 310 (mostly early, not at the second pot).

### Verdict
- Observation: the gate passes in both probes (CI above 0). Picking the best of 8 offset candidates, chosen on one
  repeat and scored on the others, beats the plain chunk by ~12 points; noise candidates in G0 gave +0.6.
- Observation: no direction helps in both probes. z+ / z+0.5 (the height hypothesis) do nothing. z− in h1 is the only
  CI above 0; it does not replicate in h4 and is not pre-specified, so by the rule it is treated as chance.
- Interpretation: which offset helps depends on the state; a fixed push would not capture this, a state-dependent
  ranker could. That is the imagination premise G0 with noise lacked.
- Per the pre-registered rule: **pass → G1 on offset proposals** (train Q(s, a) on data containing these offsets,
  test whether it ranks them at held-out branch points).

### Caveats
- 20 base episodes (10 states × 2), one offset size; the oracle uses mid-episode restore (sim only).
- The cv gain is an upper bound for a learned ranker: it uses the simulator's own outcome on a held-out repeat.
- Hold 4 overshot (2.8–3.1 cm vs 2 cm target; controller lag), so the two probes differ in size as well as duration.
