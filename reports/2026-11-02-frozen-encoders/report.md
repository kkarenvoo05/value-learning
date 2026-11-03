# Frozen encoders: where does π₀.₅'s value noise come from? (P1, P1b, P1c)

2026-11-02 to 2026-11-03 · LIBERO libero_10 task 8 ("put both moka pots on the stove") · status: **P1c step 2 queued**

## Question

π₀.₅ value models give a jittery within-episode V (advantage-label SNR 0.07 for B, vs 0.25–0.27 for the
fine-tuned ResNet). Is the noise already in π₀.₅'s image features, added by its language-model (LLM) layers,
or added by fine-tuning?

## Setup

- Frames: every 4th frame of the fixed + fresh table (`/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh`),
  117,993 frames, both cameras. Val = the 10 value-val init states (120 episodes, 13,944 frames); inner = 5 train
  states (early stopping). Fit set varies by row (below).
- Encoders [ours]: `resnet_in` (ImageNet ResNet-18), `siglip_pi05` (π₀.₅'s SigLIP, mean of 256 tokens),
  `siglip_web` (timm SigLIP So400m/14 webli), `resnet_ft` / `siglip_pi05_ft` (fine-tuned value-model backbones
  18237895 / 18269859), `llm_final` (frozen π₀.₅ final-layer prefix hidden states, image-token means),
  `llm_v{1,5,9,13,17}` (per-layer attention values V, 256-d), `*_pca256` (PCA to 256-d per camera).
- Measure: the same MLP value head (`scripts/libero_head_swap.py`, 3 seeds) on each frozen feature set, scored by
  `score()`; main metric **head SNR 2 s** (within-episode change in V, success minus failure, time-matched, over
  its noise). Also a linear pot-on-stove probe and robot-state R².
- Code: `scripts/libero_frozen_encoders.py` (extract / analyze, `--fit-subset`, `--tag`),
  `scripts/libero_p1b_prepare.py`, `scripts/libero_p1c_pca.py`.
- Data files here: `p1_fit825.json`, `p1b_fit375fixed.json`, `p1c_pca256.json`, `p1c_fit375fresh.json`,
  `p1c_fit375fixed_p1dir.json` (copies of `analysis*.json` from `/iris/u/ankile/karenvo/libero/features/`).

## Results

### P1: frozen vs fine-tuned (fit = fixed + fresh, 825 episodes)

| encoder | head SNR 2 s | milestone | MAE | AUC 10 s | probe event SNR |
|---|---|---|---|---|---|
| resnet_in | 0.168 ± 0.004 | 0.045 | 0.186 | 0.756 | 4.9 |
| siglip_pi05 | 0.172 ± 0.011 | 0.056 | 0.171 | 0.826 | 6.9 |
| siglip_web | 0.160 ± 0.015 | 0.057 | 0.186 | 0.698 | 5.1 |
| resnet_ft | 0.169 ± 0.005 | 0.072 | 0.188 | 0.762 | 5.8 |
| siglip_pi05_ft | 0.084 ± 0.003 | 0.058 | 0.197 | 0.726 | 6.3 |

### P1b: language-model layers, matched head (fit = 375 fixed episodes)

| encoder | head SNR 2 s |
|---|---|
| siglip_pi05 | 0.122 ± 0.004 |
| llm_v1 / v5 / v9 / v13 / v17 | 0.093 / 0.098 / 0.089 / 0.080 / 0.060 |
| llm_final | 0.059 ± 0.009 |
| siglip_pi05_ft | 0.078 ± 0.012 |
| resnet_in | 0.155 ± 0.011 |

### P1c: layout variety vs size; PCA control

| fit set | episodes | resnet_in | siglip_pi05 | siglip_web |
|---|---|---|---|---|
| fixed layouts | 375 | 0.155 | 0.122 | 0.135 |
| fresh layouts | 375 | 0.193 | 0.194 | 0.160 |
| fixed + fresh | 825 | 0.168 | 0.172 | 0.160 |

PCA-256 (fit 375 fixed): siglip_pi05 0.115 ± 0.004 (vs 0.122 full); llm_final 0.079 ± 0.001 (vs 0.059 full).

## Verdict

- **Observed:** frozen π₀.₅ SigLIP is as steady as an ImageNet ResNet at 825 episodes and better than web SigLIP.
  Robot (VLA) training did not make the vision features noisy.
- **Observed:** two separate drops, each about halving head SNR: the frozen LLM layers (0.122 → 0.059) and SigLIP
  fine-tuning (0.172 → 0.084 at 825; 0.122 → 0.078 at 375).
- **Observed:** at equal size, fresh layouts beat fixed layouts by a lot (SigLIP 0.122 → 0.194); adding fixed
  episodes to fresh lowers SNR. SigLIP is more layout-sensitive than the ResNet.
- **Interpretation:** the fixed-layout fit invites a layout shortcut. The 256-d projection explains ~0.007 of the
  layer-1 drop; ~0.02 remains (≈1.6 sd), not decisive. Gentler SigLIP fine-tuning alone cannot fix B, whose value
  reads the LLM prefix.

## Caveats

- One fine-tuned SigLIP checkpoint at one learning rate.
- V rows are a learned 256-d projection, not hidden states (P1c step 2 fixes this).
- P1b's LLM rows exist only on the 375-fixed fit set, the shortcut-prone one.
- Retracted along the way: "frozen SigLIP needs more data" (it needs more layouts).

## Next

- P1c step 2 (job 18279695): per-layer hidden states h0–h18 on all 1,020 episodes; score on fit_fixed,
  fit_fresh375_s0, fit_fixed_fresh. h0 should ≈ siglip_pi05 (sanity).
- Step 3 (Karen pre-approved): if h1–h3 fall below h0 by > ~2 sd, `ABLATE=1 sbatch scripts/pi05_layer_hidden.sbatch`
  (image tokens cannot attend to prompt / state).

Log entries: `docs/libero_plan.md` 2026-11-02 22:30 to 2026-11-03.
