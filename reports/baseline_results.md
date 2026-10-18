# Stage 4 baseline: first results (2026-10-18)

Run `data/runs/baseline/` (job 17947767, move5, RTX PRO 6000, git `0cd26b5`, 40 min):
`sbatch scripts/baseline.sbatch` → `vl.train_baseline`. Settings in `configs/train.yaml` [ours]:
batch 32, 40k steps (~3.4 passes over 373,947 train frames), AdamW lr 1e-4 backbone / 1e-3
head, 500-step linear warmup then constant, bf16. Full validation set (64,906 frames,
247 episodes) every 2k steps; best.pt = lowest per-frame val CE.

Throughput: 850–1,270 samples/s with 28 decoding workers; the GPU waited for data 42–58% of the
time (the video decoder is the bottleneck).

## Learning curve: the model overfits after ~2k steps

| step | train CE | val CE | val MAE (frame) | val MAE (episode) |
|---:|---:|---:|---:|---:|
| 2,000 | 3.41 | **3.873** | **0.119** | **0.099** |
| 4,000 | 3.06 | 4.212 | 0.132 | 0.114 |
| 10,000 | 2.38 | 5.247 | 0.134 | 0.114 |
| 20,000 | 1.83 | 6.670 | 0.137 | 0.114 |
| 30,000 | 1.62 | 8.295 | 0.146 | 0.119 |
| 40,000 | 1.41 | 8.734 | 0.137 | 0.113 |
| constant predictor | – | 4.530 | 0.257 | – |

- Best checkpoint = **step 2,000** (only 0.17 passes over the training frames). Afterwards,
  train CE keeps falling while val CE rises past the constant predictor (from step ~6k). Val
  MAE plateaus around 0.13–0.14.
- CE worsens much faster than MAE. With hard one-hot targets on 201 narrow bins (0.6 s each),
  a confident peak a few bins off costs a lot of CE but little MAE. So the overfitting shows
  mainly as **over-confident distributions**; the expected value V degrades slowly.
- Likely cause: memorization of training episodes (neighbouring frames are near-duplicates;
  1,373 episodes). Not yet tested.

## Best checkpoint (step 2,000) on validation

| outcome | episodes | frames | target mean | V mean | MAE | bias (V − v) | constant MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| success | 172 | 41,685 | −0.077 | −0.145 | 0.081 | −0.067 | 0.213 |
| failure | 35 | 6,350 | −0.584 | −0.442 | 0.162 | +0.142 | 0.296 |
| timeout | 40 | 16,871 | −0.640 | −0.460 | 0.196 | +0.180 | 0.352 |

V is pulled toward the middle: too pessimistic on successes, too optimistic on failures and
timeouts.

## Stronger trivial baselines: repo identity explains most of the gain

Validation success rate by repo: c00–c05 (collection, with human help) 98–100%; r00 4%,
r01 16%, r02 34%, r03 52%, r04 53%. Any cue that reveals the repo or round (setup, lighting,
operator presence) also reveals the likely outcome.

| predictor (val, per-frame MAE) | MAE |
|---|---:|
| global constant (training histogram mean) | 0.257 |
| elapsed time only (train mean value per 1 s of elapsed time) | 0.235 |
| repo + elapsed time (train mean per repo × 1 s) | 0.146 |
| **repo only (train median value per repo)** | **0.140** |
| **model, best checkpoint** | **0.119** |

- The model beats the repo-only constant by only 0.021, against 0.138 over the global
  constant. Most of the apparent skill could come from recognising the repo. These two
  "baselines" use repo or elapsed time, which the model never receives; they are reference
  points, not model inputs.
- Episode-level AUC of V for "this episode will succeed", at matched elapsed time:
  - Pooled over all val repos, at t ≈ 1 s (before the robot does anything): model 0.894,
    repo-only constant 0.826.
  - **Within eval repos r01–r04** (103 val episodes, 43% success; AUC computed within each
    repo, then averaged):

    | t | 1 s | 3 s | 5 s | 8 s | 12 s | 16 s |
    |---|---:|---:|---:|---:|---:|---:|
    | AUC | 0.57 | 0.64 | 0.62 | 0.70 | 0.80 | 0.66 |

    This is close to chance at the start and rises as the episode unfolds, the expected
    pattern for a progress signal. By 16 s, successes have mostly finished, so fewer and
    harder episodes remain.
  - On eval repos the model's MAE is 0.199 vs 0.240 for the repo constant.

## Open decisions (for Karen)

1. **Checkpoint selection:** CE and MAE disagree after step 2k. Keep CE (the objective), or
   select on MAE, or report both.
2. **Overfitting:** fewer steps / lower LR, a frozen backbone (Stage 5 ablation 3), or soft
   (Gaussian, "HL-Gauss") targets over neighbouring bins. The last is a target-construction
   change, not paper-specified.
3. **Repo shortcut:** make the repo-only constant the trivial baseline to beat, and report
   metrics within the eval repos, where outcomes vary. Check what in the images lets the
   model tell repos apart.
