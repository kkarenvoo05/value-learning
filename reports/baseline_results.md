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

## Synchronized video/value plots (best checkpoint, step 2k)

`python -m vl.value_plots --run baseline` → `reports/figures/stage4/baseline/*.png` (filmstrip
of both cropped views + curve) and `data/review_videos/stage4/baseline/*.mp4` (moving cursor).
9 validation episodes, chosen by a fixed rule (median-MAE episode of each kind, plus the worst
episodes; the worst-error episode is also the assisted failure). Shading = human control
(`source` = 1). Read from the still frames by Claude; **not yet watched as video by Karen**.

| # | kind | episode | outcome, T | MAE | what V does |
|---|---|---|---|---:|---|
| 01 | autonomous success | r03/61 | success, 15.7 s | 0.198 | starts pessimistic (−0.4), dips to −0.6 at ~4 s while the arm passes in front of the side camera, then rises steadily from ~8 s as the marker reaches the holder; ≈ 0 at the seat |
| 02 | autonomous failure | r01/110 | failure, 35.7 s | 0.093 | flat ≈ −0.57 for 30 s with the marker held at the holder |
| 03 | timeout (soft-truncated stall) | r01/23 | timeout, 34.5 s | 0.118 | flat ≈ −0.55 through the stall |
| 04 | timeout at the 53 s cap | r03/125 | timeout, 53.3 s | 0.191 | flat ≈ −0.57 for 45 s while the target climbs −0.94 → −0.5 |
| 05 | assisted success | c01/102 | success, 22.5 s | 0.021 | tracks the target, unaffected by the two takeovers |
| 06 | assisted failure (= worst error) | c03/97 | failure, 9.7 s | 0.467 | ≈ −0.05 throughout (confident success); brief dip during the takeover at 9 s |
| 07 | teleop demo | c00/182 | success, 11.1 s | 0.009 | on the target |
| 08 | typical error | c02/61 | success, 10.7 s | 0.038 | close to the target, slightly pessimistic early |
| 09 | worst success | r01/103 | success, 28.3 s | 0.407 | flat ≈ −0.55 for 25 s at the holder, then **jumps to ≈ 0 at the seat** |

What this suggests [Claude's reading of the stills; to be checked against the videos]:
- **V recognizes a seated marker** (01, 09: sharp rise exactly at seating), and rises with
  visible progress toward the holder (01).
- **"Holding the marker at the holder" gets V ≈ −0.55, whatever happens next** (02, 03, 04,
  09). That is the average of the two outcome groups (eventual seat ≈ −0.1, stall/failure ≈
  −0.7). It is the image-only ambiguity predicted in audit Section 5a: one frame cannot show
  how long the robot has been stuck. It is not a bug, but it is where most eval-repo error
  comes from.
- **Repo prior:** collection-repo episodes get V ≈ −0.1 regardless of outcome. The worst error
  (06) is a c03 failure predicted as a confident success; c03 val episodes are 98% success.
  Eval-repo episodes start near −0.4 to −0.6.
- **Possible occlusion sensitivity:** in 01 V drops by ~0.2 while the arm fills the side
  camera. Check on more episodes (Stage 4 occlusion question).

## Repo-aware metrics (`vl.metrics`, step-2k baseline)

| metric | model | reference |
|---|---:|---:|
| val MAE (frame) | 0.119 | repo-only constant 0.140 |
| eval-repo (r00–r04) MAE | 0.194 | repo-only constant 0.222 |
| within-repo success AUC at 1 / 5 / 10 s | 0.59 / 0.62 / 0.81 | 0.5 = chance |
| offset-free (shape) MAE, per episode | **0.052** | flat line 0.037 |

The shape error is worse than a flat line. Once each episode's average level is removed, V's
within-episode ups and downs fit the straight countdown worse than no movement at all. The
jumps (e.g. −0.55 → 0 at seating) count as error against a straight target, so some of this is
expected. Still, it means this V's frame-to-frame changes, the input to Stage 6 advantages, are
mostly not tracking the countdown.

## Open decisions (for Karen)

1. **Checkpoint selection:** CE and MAE disagree after step 2k. Keep CE (the objective), or
   select on MAE, or report both.
2. **Overfitting:** fewer steps / lower LR, a frozen backbone (Stage 5 ablation 3), or soft
   (Gaussian, "HL-Gauss") targets over neighbouring bins. The last is a target-construction
   change, not paper-specified.
3. **Repo shortcut:** make the repo-only constant the trivial baseline to beat, and report
   metrics within the eval repos, where outcomes vary. Check what in the images lets the
   model tell repos apart.
