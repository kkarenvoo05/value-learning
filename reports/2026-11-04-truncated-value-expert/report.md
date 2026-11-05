# A2: truncated π₀.₅ value expert (SmolVLA idea), seed 0

Value expert reads only the first N of PaliGemma's 18 layers (`scripts/pi05_value_expert.py --n-layers N`, K/V
cache sliced, exact by causality). Frozen-π₀.₅ recipe (w256, mlp 1024, lr 1e-4, 5400 steps, augment, fixed+fresh
825 fit episodes, seed 0). Training jobs 18291639 (N=9) / 18291640 (N=4) were host-OOM-killed at 64G after
training (best.msgpack saved); scored with `--eval-only <log>` at 96G: 18305874 / 18305875. Scores:
`score(p, "ev_expert")` from `scripts/libero_head_swap.py` (`score_a2.py` here) on the 120 val episodes.

| backbone layers | change SNR | noise | milestone | MAE | AUC10 | best step |
|---|---|---|---|---|---|---|
| 18 (18237898, s0) | 0.152 | 0.101 | +0.042 | 0.160 | 0.797 | — |
| 9 | 0.158 | 0.092 | +0.029 | 0.162 | 0.808 | 5000 |
| 4 | **0.180** | **0.085** | +0.035 | 0.169 | 0.809 | 5000 |

Reference: frozen π₀.₅ 3 seeds SNR 0.166 ± 0.025, MAE 0.170; ResNet 0.247 / 0.189.

## Verdict (seed 0 only)
- Noise falls monotonically with depth cut; accuracy cost small (MAE +0.009 at N=4, still ≤ 0.175).
- Pre-set bar (SNR clearly above 0.166 ± 0.025, MAE ≤ ~0.175): N=4 is inside one sd → **not cleared on one seed**.
- N=9 ≈ full; consistent with the P1c per-layer probe (flicker enters at layers 17–18).
- Both runs' best inner CE was at step 5000 of 5400: slightly under-trained, like the full model.
- Next: seeds 1–2 for N=4 (and N=9) before any A3 (truncated + LoRA).

## Update 2026-11-04 17:45: N=4 seed 1 (18309645)
SNR **0.182**, noise 0.084, MAE 0.170, AUC10 0.820, milestone +0.021 — replicates seed 0 (0.180 / 0.085 / 0.169).
Two-seed mean 0.181 vs full frozen π₀.₅ 0.166 ± 0.025 (3 seeds): consistent, modest gain; still below the ~0.22
needed to make label construction worthwhile and below the ResNet 0.247. Seed 2: 18309646 (queued).
