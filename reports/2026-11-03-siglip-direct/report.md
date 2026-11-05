# SigLIP-direct value (no LLM): frozen vs gentle fine-tune — 2026-11-03

**Question.** If V reads π₀.₅'s SigLIP directly (skipping the Gemma LLM), is it steadier than the whole frozen π₀.₅ (B)
and more useful than the ResNet?

**Setup** [ours]. `vl.libero_resnet --backbone siglip_pi05` (mean-pooled 256 tokens → 1152-d per camera, concat → 256 →
201 bins), fixed+fresh fit (825 episodes, every 4th frame), 8,000 steps, head lr 1e-4, inner-CE selection, seed 0,
job 18286678 (A5000). Arms: `--freeze-backbone`; `--backbone-lr 1e-6` (10× below 18269859). Scored with
`libero_head_swap.score` on the 120 val episodes (10 fixed held-out states), same as every reference row.

| s0 | SNR 2 s | signal | noise | milestone | MAE | MAE succ>20 s | AUC 5 / 10 s | inner CE (step) |
|---|---|---|---|---|---|---|---|---|
| ResNet ff (18237895) | **0.261** | **0.0183** | 0.0701 | **+0.095** | 0.192 | 0.120 | 0.73 / 0.76 | 3.34–3.38 |
| frozen B (18237898) | 0.152 | | | +0.042 | **0.160** | **0.119** | 0.70 / **0.80** | |
| SigLIP ft 1e-5 (18269859) | 0.129 | 0.0105 | 0.0815 | +0.060 | 0.207 | 0.223 | 0.66 / 0.72 | 3.584 (1,500) |
| **SigLIP frozen** | 0.221 | 0.0126 | **0.0571** | +0.017 | 0.215 | 0.194 | **0.77** / 0.70 | 3.397 (8,000) |
| **SigLIP lr 1e-6** | 0.107 | 0.0104 | 0.0971 | +0.093 | 0.187 | 0.150 | 0.71 / 0.78 | 3.389 (5,000) |

**Observations.**
- Frozen SigLIP has the lowest 2 s noise of any arm (0.057 vs ResNet 0.070) and SNR 0.221, above frozen B (0.152);
  but its signal is 31% below the ResNet's, its milestone is ≈ 0 (+0.017), and it is the least accurate (MAE 0.215,
  late-success MAE 0.194).
- lr 1e-6 restores the milestone (+0.093 = ResNet) and does not overfit (inner CE flat from step 3,000), but noise is
  the highest of all arms (0.097) → SNR 0.107.
- Neither arm matches B's accuracy (MAE 0.160) or the ResNet's SNR.

**Interpretation.**
- Removing the LLM does buy steadiness, but frozen SigLIP is steady partly because it is sluggish: it does not see the
  pot landing (milestone ≈ 0). Its SNR is not "better features", it is a smaller, slower V.
- Any SigLIP fine-tuning, even at 1e-6, adds per-frame flicker (noise 0.082–0.097). So the SigLIP-fine-tuning noise
  source (P1) is not an overfitting artefact that a lower lr removes.
- B's accuracy edge (MAE 0.160) comes from the LLM path, not from SigLIP: SigLIP-direct loses it in both arms. The
  LLM adds noise *and* accuracy.
- Verdict: SigLIP-direct is not more useful than the ResNet for advantage labels on moka. Not worth more seeds as is.

**Caveats.** One seed (earlier seed sd ≈ ±0.025 SNR, ±0.015 milestone); frozen 0.221 vs ResNet mean 0.247 is within
~1 sd, but the milestone gap (0.017 vs 0.094) is ~5 sd. Mean pooling may discard the pot's location: an
attention-pooled head on frozen tokens is the one untested SigLIP variant. No language input (moka is single-task).

**Next.** Step 0 (PaliGemma per-layer, 18286806) and the π₀.₅ per-layer probe (18288157, adds accuracy per layer)
decide whether the LLM's accuracy can be kept without its noise. Optional cheap follow-up: attention pooling over
frozen SigLIP tokens.
