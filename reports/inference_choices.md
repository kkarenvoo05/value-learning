# Inference-time choices on R0 (2026-10-26)

`.venv/bin/python data/runs/inference_choices.py` (CPU, ~40 min with 4 processes) → `data/runs/inference_choices.csv`.
Inputs: the saved `val_probs.npz` of `g0_r0_s{0,1,2}` (best checkpoints; recomputed CE / MAE match each
`result.json` to 4 decimals). No retraining: every row post-processes the same three models.

Variants [ours]: single model (metric = mean over the 3 seeds) or 3-seed ensemble (mean of distributions);
moving average of the 201-bin distribution within each episode, centred (uses future frames: offline only) or
causal (trailing: usable live), 0.5 s = 8 frames, 1 s = 15; temperature fit by cross-fitting over val episode
groups (as the 2026-10-22 study), after smoothing. Intervals: paired bootstrap over the 247 val episodes,
400 draws (no seed resampling, since all rows use the same models); `*` = interval excludes 0.
Flicker = mean |V_t − V_{t−1}| between consecutive frames more than 2 s before T (`vl.flicker` definition).

## Main rows (Δ vs single model, no smoothing, no temperature)

| variant | val CE | Δ CE [95%] | val MAE | shape MAE | AUC 1 s | ECE p_succ | flicker \|dV\| | stall_auc | frozen_auc |
|---|---|---|---|---|---|---|---|---|---|
| single (R0 as reported) | 3.672 | | 0.1136 | 0.0487 | 0.621 | 0.024 | 0.0142 | 0.800 | 0.820 |
| single + T (1.10–1.19) | 3.660 | −0.011 [−0.026, +0.002] | 0.1168* | 0.0481 | 0.617 | 0.018 | 0.0138* | 0.801 | 0.821 |
| single, centred 0.5 s | 3.655 | −0.017 [−0.020, −0.014]* | 0.1133 | 0.0458* | 0.617 | 0.023 | 0.0036* | 0.805* | 0.822 |
| single, causal 0.5 s | 3.682 | +0.010 [−0.000, +0.022] | 0.1144* | 0.0453* | 0.617 | 0.023 | 0.0036* | 0.803 | 0.821 |
| ensemble ×3 | 3.641 | −0.030 [−0.033, −0.027]* | 0.1127* | 0.0469* | 0.606 | 0.015 | 0.0127* | 0.807* | 0.828 |
| **ensemble ×3, centred 0.5 s** | **3.630** | **−0.042 [−0.047, −0.036]*** | 0.1124* | 0.0444* | 0.600 | 0.015 | 0.0033* | 0.811* | 0.831 |
| ensemble ×3, centred 1 s | 3.633 | −0.038 [−0.044, −0.031]* | 0.1120* | 0.0433* | 0.606 | 0.017 | 0.0021* | 0.814* | 0.833 |
| ensemble ×3, centred 0.5 s + T (1.04–1.08) | 3.628 | −0.043 [−0.053, −0.033]* | 0.1136 | 0.0441* | 0.606 | 0.018 | 0.0032* | 0.812* | 0.833 |
| **ensemble ×3, causal 0.5 s** | **3.656** | −0.015 [−0.027, −0.002]* | 0.1136 | 0.0439* | 0.631 | 0.015 | 0.0033* | 0.810* | 0.830 |
| ensemble ×3, causal 1 s | 3.686 | +0.015 [−0.005, +0.036] | 0.1148* | 0.0421* | 0.633 | 0.017 | 0.0021* | 0.810* | 0.830 |

All 20 rows (every combination, plus eval-repo MAE, AUC 5 s, |dp_succ|, 1 s wander) are in the CSV.

## Reading (Claude, to be checked)

1. **Ensembling is the biggest free gain**: −0.030 CE, credible, and slightly better MAE, shape and stall_auc.
   Larger than any trained arm in Stage 5.5 (none passed −0.025). Cost: 3 forward passes per frame.
2. **Centred 0.5 s smoothing adds −0.012 CE on top of the ensemble and cuts flicker 74%** (0.0127 → 0.0033),
   shape MAE −0.0025. 1 s smooths more (flicker −83%, best shape) for slightly worse CE than 0.5 s. Either
   is fine offline; 0.5 s is the CE optimum on this grid.
3. **Temperature is not worth it.** Alone −0.011 CE (n.s.) at a credible +0.003 MAE cost; on the smoothed
   ensemble ~0 (−0.001), T ≈ 1.04–1.08, and ECE gets *worse* (0.015 → 0.018) because smoothing already
   softens the distribution. Same verdict as the 2026-10-22 study on the older arms.
4. **Causal smoothing costs accuracy** (it lags: V reaches the seat late): causal 0.5 s on the ensemble is
   +0.015 CE vs the unsmoothed ensemble, but still −0.015 vs today's single model, with the same 74% flicker cut.
   Causal 1 s is too slow (+0.045 vs the ensemble).
5. **Unchanged by any choice:** AUC 1 s (all intervals span 0; ±0.03–0.05) and the shape gap to a flat line:
   the best shape MAE is 0.042, still above the flat line's 0.0365. Smoothing removes frame flicker, not the
   slower wander (1 s wander 0.035 → 0.022–0.026).
6. Caveat: chosen on val, so the chosen row's val numbers are a little optimistic (5 smoothing options × 2
   temperature options); the r05 test reports it once.

## Proposed recipe (for Karen)

- **Offline (advantage labels, reports, the r05 test): 3-seed ensemble, centred 0.5 s average of the
  distribution, no temperature.** V and p_succ computed from the smoothed distribution.
- **Live (Stage 8 rollout scoring): 3-seed ensemble, causal 0.5 s, no temperature.**
- Report the single-model numbers next to it, so the gain from post-processing stays separate from the model.
