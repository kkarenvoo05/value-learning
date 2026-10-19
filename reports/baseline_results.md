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

## Decision 2(a): shorter schedule with cosine decay (2026-10-18)

Run `data/runs/cosine_8k/` (job 17950654, move4, **L40S**, git `400b17a`, 25 min). Same as
the baseline except: 8k steps (~0.68 passes), 500-step warmup then cosine decay to 0, full-val
eval every 500 steps [ours]. The GPU type also differs from the baseline (RTX PRO 6000).

| step | val CE | val MAE | eval-repo MAE | shape MAE | AUC 1/5/10 s |
|---:|---:|---:|---:|---:|---|
| 500 | 3.985 | 0.144 | 0.229 | 0.043 | 0.54 / 0.57 / 0.79 |
| **1,000 (best)** | **3.839** | 0.129 | 0.197 | 0.048 | 0.56 / 0.61 / 0.76 |
| 2,000 | 3.970 | 0.126 | 0.203 | 0.059 | 0.57 / 0.58 / 0.77 |
| 4,500 | 4.636 | 0.131 | 0.219 | 0.063 | 0.59 / 0.58 / 0.75 |
| 8,000 | 5.734 | 0.135 | 0.221 | 0.071 | 0.57 / 0.55 / 0.75 |
| references | constant 4.530 | repo-const 0.140 | repo-const 0.222 | flat 0.037 | 0.5 |
| step-2k baseline | 3.873 | 0.119 | 0.194 | 0.052 | 0.59 / 0.62 / 0.81 |

- **The success criterion is not met.** Val CE again bottoms out early (step 1k) and rises
  while the LR is still high; MAE at the best checkpoint (0.129) is worse than the baseline's
  0.119, and only 0.011 better than the repo-only constant.
- Overfitting starts at ~1k steps = 32k frames ≈ 0.09 passes: most training frames have not
  been seen once. So it is not memorizing revisited frames. More likely the model learns
  features of the 1,373 training *episodes* (each 1k-step stretch already samples ~23 frames
  of every episode, and neighbouring frames are near-duplicates) that do not carry over to
  new episodes. The LR schedule acts too late to matter.
- The shape error gets worse at every eval (0.043 → 0.071) and is never below a flat line:
  training makes V's within-episode movement jumpier, not more countdown-like.
- Differences between this run and the baseline at their best (CE 3.84 vs 3.87, MAE 0.129 vs
  0.119) are within what one seed and a different GPU could produce; the 3-seed retrain will
  say how large that noise is.

## Decision 2(b): 10x lower learning rates (2026-10-18)

Run `data/runs/lowlr_8k/` (job 17954831, move4, L40S, git `8ada382`, 19 min). Same as 2(a)
except backbone LR 1e-5 and head LR 1e-4 (both 10x lower) [ours].

| step | val CE | val MAE | eval-repo MAE | shape MAE | AUC 1/5/10 s |
|---:|---:|---:|---:|---:|---|
| 500 | 4.228 | 0.235 | 0.280 | 0.031 | 0.60 / 0.53 / 0.79 |
| 1,500 | 3.745 | 0.132 | 0.213 | 0.042 | 0.63 / 0.61 / 0.78 |
| **3,000 (best)** | **3.727** | **0.117** | 0.188 | 0.050 | 0.69 / 0.63 / 0.85 |
| 4,500 | 3.794 | 0.113 | 0.184 | 0.052 | 0.65 / 0.56 / 0.83 |
| 8,000 | 3.878 | 0.116 | 0.189 | 0.053 | 0.69 / 0.57 / 0.83 |
| references | constant 4.530 | repo-const 0.140 | repo-const 0.222 | flat 0.037 | 0.5 |
| step-2k baseline | 3.873 | 0.119 | 0.194 | 0.052 | 0.59 / 0.62 / 0.81 |

- **Success criterion met.** Val CE is flat (3.73–3.88) over the last 6.5k steps instead of
  climbing past the constant, so the best checkpoint is no longer an early accident. Best-
  checkpoint MAE 0.117 ≤ 0.119, and MAE keeps improving slightly after the CE minimum.
- Best on every summary number so far: val CE, val MAE, eval-repo MAE (0.188, 0.034 below
  the repo-only constant) and within-repo AUC. The gaps to the baseline are small; whether
  they exceed seed noise is for the 3-seed retrain.
- **Still unsolved: shape.** The shape error is below flat only at step 500, when V barely
  moves; afterwards it is 0.042–0.054 vs 0.037. Lowering the LR did not change this, so the
  within-episode jumpiness is not an overfitting effect. It is the main open problem for
  Stage 6 advantages.
- Within-repo AUC at 1 s (0.69) is above chance before the robot has moved. Something in the
  first frames predicts the outcome even inside one repo (e.g. starting pose, marker
  placement, or day/round within a repo). Worth checking with the repo-cue question.

## 3-seed retrain of the settled recipe (`baseline_lowlr`, 2026-10-19)

Seeds 0 (`lowlr_8k`, job 17954831), 1 (`lowlr_8k_s1`, job 17958964) and 2 (`lowlr_8k_s2`, job
17958965), all on move4 L40S, git `c07f46d` for seeds 1–2 (`--seed` only changes the head init
and data order; the backbone is the same pretrained ResNet-18). Best checkpoints:

| seed | best step | val CE | val MAE | MAE per ep. | eval-repo MAE | shape MAE | AUC 1/5/10 s |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 3,000 | 3.727 | 0.117 | 0.098 | 0.188 | 0.050 | 0.69 / 0.63 / 0.85 |
| 1 | 2,000 | 3.732 | 0.123 | 0.103 | 0.196 | 0.047 | 0.67 / 0.65 / 0.83 |
| 2 | 2,000 | 3.730 | 0.124 | 0.101 | 0.204 | 0.042 | 0.65 / 0.64 / 0.83 |
| **mean ± sd** | | **3.730 ± 0.002** | **0.121 ± 0.004** | 0.101 ± 0.002 | 0.196 ± 0.008 | 0.047 ± 0.004 | 0.67 / 0.64 / 0.84 (± 0.01–0.02) |
| step-2k baseline (1 seed) | 2,000 | 3.873 | 0.119 | 0.099 | 0.194 | 0.052 | 0.59 / 0.62 / 0.81 |

- **Seed noise (sd):** val CE 0.002, val MAE 0.004, eval-repo MAE 0.008, shape 0.004, AUC
  0.01–0.02. Stage 5 differences smaller than about 2 sd (MAE ≈ 0.008, eval-repo MAE ≈ 0.016) are
  not evidence.
- **Seed 0 was the lucky one.** Its MAE 0.117 met the "≤ 0.119" criterion; the 3-seed mean
  0.121 does not beat the step-2k baseline's 0.119 on MAE (within noise). The real gain of the
  low LR is **CE (3.73 vs 3.87, ~70 sd) and stability**: no seed overfits, so the best
  checkpoint is no longer an early accident. The recipe stays settled on that basis. Every seed
  beats the repo-only constant (0.140) and, on eval repos, its 0.222.
- All three seeds are still worse than a flat line on shape (0.042–0.050 vs 0.037).
- The analysis below (seed 0) was rerun on seeds 1 and 2 (`reports/stage4_analysis_lowlr_8k_s{1,2}.md`)
  and every conclusion holds: the hedging on autonomous failures/timeouts (V still worse than the repo
  constant there), the timeout-kind ordering, operator ΔMAE ≤ 0 with 0 drops > 0.2 at an appearance,
  and seated-vs-stalled AUC 1.0.

**Train/val gap** (`vl.predict`, job 17958970: best.pt of seed 0 on 250 random training
episodes, 72,243 frames, eval mode): CE 3.554 train vs 3.727 val (gap 0.17); MAE 0.091 vs 0.117
(gap 0.026); per-episode MAE 0.078 vs 0.098. A modest gap: at its best checkpoint the low-LR
model is not memorizing (cf. the constant-LR baseline at 40k steps: train CE 1.41 vs val 8.73).

## Stage 4 analysis of the settled recipe (`lowlr_8k`, best step 3k, 2026-10-19)

Tables: `reports/stage4_analysis_lowlr_8k.md`, generated by
`.venv/bin/python -m vl.analysis --run lowlr_8k --seeds lowlr_8k lowlr_8k_s1 lowlr_8k_s2`.
Plots: `python -m vl.value_plots --run lowlr_8k` → `reports/figures/stage4/lowlr_8k/` (11
episodes: the same fixed rule as before plus one "drop failure"). All groupings and thresholds
below are [ours]; they are defined in the `vl.analysis` docstring.

### Where the error is (validation, per-frame MAE; repo-const = repo-only constant)

| episodes | n | MAE | bias (V − v) | repo-const MAE |
|---|---:|---:|---:|---:|
| teleop / DAgger collection (c00–c05), success | 127 | 0.011–0.032 | −0.01 to −0.02 | 0.026–0.040 |
| autonomous success | 45 | 0.218 | −0.216 | 0.447 |
| autonomous failure | 34 | 0.133 | +0.116 | **0.065** |
| autonomous timeout | 40 | 0.186 | +0.179 | **0.106** |
| assisted failure (c03/97) | 1 | 0.464 | +0.464 | 0.473 |

- On autonomous rollouts V **hedges toward the middle**. It is far better than the repo constant
  on successes and *worse* than it on failures and timeouts, because the repo median of a
  mostly-failing round is already close to the failure targets. The net gain on eval repos (0.188 vs
  0.222) comes from recognizing the successes.
- Per round (autonomous only): the model beats the repo constant on r02–r04 (0.19–0.22 vs
  0.21–0.30), ties on r01 and loses on r00 (0.139 vs 0.084; 1 success in 16). Within-round
  success AUC at 10 s is 0.72–0.88 for r01–r04. **V stays meaningful on the late rounds.**
- **Assisted episodes:** human-controlled and robot-controlled frames have the same error
  (0.035 vs 0.031, successes). Validation has only one assisted failure (c03/97, a confident
  success prediction), so assisted failures cannot be analysed on this split.

### Timeout endpoints

| timeout kind | episodes | MAE | bias | repo-const MAE |
|---|---:|---:|---:|---:|
| soft-truncated at a stall | 31 | 0.174 | +0.166 | 0.087 |
| recording ended early | 5 | 0.108 | +0.101 | 0.113 |
| ran to the 53 s cap | 4 | 0.283 | +0.280 | 0.170 |

Reviewer-truncated timeouts are **not** worse than the others. The capped ones are worst
because a 53 s episode's targets start at −0.94, while V sits at about −0.45 on the same images (plot 09,
r04/101: 45 s at the holder, V ≈ −0.2). So the arbitrary truncation point is not a major error
source. **No need to ask Lars about changing T.** (Small n: 4 capped, 5 ended early.)

### Stalls (robot in control, ≥ 3 s in, 3 s mean end-effector speed < 15 mm/s; a heuristic)

- 64% of timeout robot frames, 38% of failure frames and 30% of success frames are stalls.
- Over stall segments (≥ 3 s; mean 17 s in timeouts) the target rises by +0.145 and V by
  +0.024: **V is flat during stalls, as expected from one image**. Per the plan this is not
  counted as a model failure. Stall frames are not where timeouts' error concentrates (MAE
  0.164 stall vs 0.224 non-stall).

### Operator visibility in `wrist_left` (audit samples at 1 fps, 4,670 val frames)

- Matched by repo, outcome and 2 s of elapsed time, visible-minus-not: ΔV = +0.02 (any person
  pixel) to +0.06 (inside the crop, autonomous only), with Δtarget +0.01 and ΔMAE ≤ 0. A small
  upward shift, and no loss of accuracy.
- Sharp drops are **not** associated with the operator appearing: P(appears | ΔV < −0.1 in 1 s)
  = 0.025 vs 0.046 for all steps; 0 of the 41 drops larger than 0.2.
- **Decision (by the pre-set rule "mask only if V reacts"): no operator masking.** The Stage 5
  masking ablation is not needed. Caveat: the audit's crop box is narrower than the training
  crop, and the detector misses 13–22% of visible frames.

### Stage 4 questions

| question | answer | evidence |
|---|---|---|
| V rises at grasp/seat? | **Seat: yes.** Grasp: not visible as a step | successes V −0.29 at T−5 s → −0.09 at T−1 s → −0.02 at T; plots 01, 10 jump to ≈ 0 at seating |
| seated vs hovering above the holder? | **Yes** | AUC 1.000: last 0.5 s of successes (V −0.02) vs timeout stall frames (V −0.46) |
| V falls at a drop? | **Not testable on val** | only 5 episodes with a drop/slip/release VLM label; the one plotted (11, r01/109) is a missed grasp (the marker never leaves the table), V flat −0.55 |
| reacts to occlusion? | operator: no (above); arm occluding the side view: not measured | needs an occlusion measure; earlier plot 01 (step-2k baseline) showed a −0.2 dip |
| meaningful on late-round autonomous rollouts? | **Yes** (r02–r04) | see per-round numbers above |

### Shape: jitter and slow wander

Frame-to-frame |ΔV| averages 0.017 (median 0.005). The target changes by 0.00056 per frame,
**30× less**. Smoothing V helps the shape error but never reaches the flat line's 0.0365: 0.050
raw, 0.044 (1 s window), 0.039 (3 s). So there are two parts: frame-level noise (plot 01: ±0.2
swings while the arm moves), and a structural plateau-then-jump (−0.5 at the holder, ≈ 0 at the
seat), which a straight countdown cannot match. For Stage 6, V(o_{t+n}) − V(o_t) will be
dominated by this noise unless n spans seconds or V is smoothed.

### Phase-one gate (Claude's assessment; Karen/Lars decide)

- **Beats the trivial baselines on validation: yes, overall** (MAE 0.121 ± 0.004 over 3 seeds vs repo-only 0.140;
  eval repos 0.196 ± 0.008 vs 0.222; within-round AUC well above chance). Not on autonomous failures and
  timeouts taken alone.
- **Interpretable held-out curves: partly.** Seating and progress toward the holder are visible
  and V separates seated from hovering perfectly. But V is jittery, and flat at the holder whatever happens next.
- **Failure cases understood: mostly.** (1) The stall ambiguity: one frame cannot show how
  long the robot has been stuck, so long timeouts sit at about −0.45 against targets near −0.9. (2) A repo/round
  prior that hedges V toward the middle. (3) The one assisted failure, which looks like a success. Not
  understood: what in the first frames predicts the outcome within a round (AUC@1 s 0.55–0.78 in r01–r04).
- **Evidence for history:** the stall ambiguity is exactly what a short history or an elapsed-time input
  would resolve, which supports Stage 5's "current frame vs short history" ablation.

### Repo cues: what lets V tell rounds apart, and why V at 1 s predicts within a round

Autonomous eval episodes, train + val (800 episodes; V from `lowlr_8k`, val only):
- **Mean V at frame 0 differs by round**: about −0.55 for r00–r02, −0.46 for r03, −0.40 for r04, before
  the robot has moved. So a still image reveals the round. `reports/figures/stage4/first_frames_by_repo.png`
  (`vl.analysis --repo-sheet`: frame 0 of 4 random training episodes per repo) shows no
  obvious cue at a glance. Glare on the table in `side_1` varies between repos (strongest in
  c00), and r00–r04 look alike at this size. Locating the cue would need a probe (e.g. a
  repo classifier on frame 0 with saliency); not done. **Still open: ask Lars whether rounds
  were recorded on different days, lighting or camera setups.**
- **Within a round, the first frame is at chance:** within-round success AUC of V at frame 0
  is 0.55 (r01–r04, 103 val episodes) and rises to 0.66 by 0.5–1.5 s. Neither the initial
  layout nor the policy explains it. Each initial-state variable (pen x/y/yaw, |sin yaw|,
  holder x/y) has within-round success AUC 0.34–0.66 per round, averaging 0.47–0.53. The
  AUC within round × policy (0.62) is about the same as within round (0.66). So the early
  within-round signal comes from the **first second of robot motion**, not a static scene cue.
