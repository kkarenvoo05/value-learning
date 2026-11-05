# B2 steps 1–2: best-of-N steering pre-checks (job 18296012)

Base π₀.₅, 10 val states (one base episode each, base success 6/10), N = 8 candidate chunks at every replan
point (920 points). Headroom: at policy steps 40/130/220/310 each candidate executed from the restored state and
the base policy continued to the end, 2× per candidate (40 branch points). Restore exact (max state diff 4.9e-17).
Run: `/iris/u/ankile/karenvo/libero/runs/steer_probe_libero_10_18296012`; analysis `scripts/libero_steer_analyze.py`
→ `analysis.json`.

## Step 1: candidate spread
- End-effector position spread after 5 steps: median **1.9 mm**, vs 22.7 mm of arm motion per chunk (ratio 0.10).
- Arm action spread / motion (per-dim std units): 0.42. Gripper split (candidates disagree on open/close): 7% of
  replan points (1% in steps 0–100, up to 14% after step 400).

## Step 2: simulator headroom
- Mean success per branch point 0.58; between-candidate SD of success (binomial noise removed) **0.08**.
- Cross-validated oracle gain (pick on repeat 1, score on repeat 2): **+0.006, 95% CI [−0.028, +0.042]**.
  By step: 40 +0.043, 130 −0.037, 220 +0.017, 310 0.000 (n = 10 each).
- In-sample best − mean 0.103 is selection on binomial noise (it does not survive cross-validation).

## Verdict
Pre-set stop rule (docs/CURRENT.md §B2): oracle gain ≈ 0 → a learned Q cannot steer base π₀.₅ with N = 8
samples. **B2 stopped at step 2.** π₀.₅'s samples are nearly identical (≈2 mm), so there is little to choose
between. Caveats: 10 states, 2 repeats per candidate (CI ±0.035); a larger N or added sampling noise / temperature
could widen the spread, but that changes the base policy.
