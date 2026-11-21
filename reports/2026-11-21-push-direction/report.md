# Push direction: does round 1 re-apply round 0's push? (2026-11-21)

**Status: sampling (iris1 interactive GPU); results pending.** Karen: "write the push-direction check. create visuals
for me to better understand the pushes". Follows the label check in `reports/2026-11-21-r1-factorial` (labels treat
base and guided episodes alike, so the drop is not in the labels).

## Question
The "double push" hypothesis: round-1 data is half rollouts collected WITH r0's push; if r1's residual learns that
push from those rollouts, the r1 policy applies it again on top of base, overshooting. Prediction: r1's push points
along r0's (cos near 1, β ≥ 1) in the chains that drop (b50cv, mt100cv) and less so where there is no drop (900, b100
seed 1).

## Setup
- **Code:** `scripts/pi05_push_direction.py` (one policy → `push.npz`), `scripts/push_direction_analyze.py`
  (CPU scoring → `analysis.json`), sbatch `MODE=pushdir` in `scripts/pi05_tag_adarms.sbatch` (list file; not used
  for this run). Page: `build_page.py` + `page_template.html` + `page_text.json` → `push_direction.html`.
- **Frames [ours]:** 24 of the 120 base val episodes × 8 frames = 192 val frames (states never trained on; same
  episodes in every moka table), 2 noise draws each; same frames, noise and frozen base for every policy.
- **Push [ours]:** a_w − a_none, executed 5 steps × 7 dims, at each policy's eval w (push target 0.99 from its own
  contrast check, as the evals). a_none = frozen base π₀.₅, identical across policies (checked).
- **Metrics:** cos and β = Σ⟨P_a, P_b⟩ / Σ|P_b|² on the 35-d normalised push; 95% CIs by bootstrap over episodes.
  xyz in cm = summed raw push over the 5 steps × 0.75 cm (G0 calibration, open loop).
- **Policies:** list `runs/push_direction/push_list.txt` (17: 900 r0/r1/r2; b100 r0, r1 s0/s1, r2; b100cv r0/r1;
  b50cv r0, r1 s0/s1; mt100cv r0, r1 s0/s1, overs 91% / 77%).
