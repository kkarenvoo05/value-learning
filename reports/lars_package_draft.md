# Progress package for Lars: draft (2026-10-26)

Draft by Claude for Karen to edit. Follows the PROGRESS.md item "what was built (architecture, inputs, targets,
split), metrics + 3 representative plots/videos, failure examples, one comparison, next decision; with config,
pinned revisions and checkpoint". **Not final:** the test-set numbers (r05) come after the merge gate and are
left as TODO; the recipe below assumes Gate B = no history and Gate C = R0 (both still Karen's call).

## 1. What was built

**Question.** Does a small RECAP-style state-value function V(o) learn useful task progress from real-robot
Mulligan Insert Marker data?

**Data.** 13 Mulligan Insert Marker repos pinned to their `release-1` commits (`data/meta/pins.json`). Outcomes
and valid endpoints T from the reviewed release labels; 4 train-pool episodes with unknown outcome excluded
(never turned into failures). Group-aware split (episodes sharing a start never cross splits): train 1,377 /
val 247 episodes (373,947 / 64,906 frames), test `r05-eval` 150 and `r05-screen` 200 (untouched).

**Targets** (`configs/targets.yaml`). [paper] −1 per step, no discount, large failure penalty, 201 bins on
(−1, 0); [Lars] timeouts = failure; [ours] horizon 900 steps (60 s at 15 fps; longest episode 840),
failure penalty 900 steps, so successes land in (−0.5, 0] and failures in [−1, −0.5]. Hard one-hot to the
nearest of 201 centres (bin = 0.6 s). Within an episode the target is a straight countdown.

**Model** (R0, the Gate 0 baseline). Per frame, `wrist_left` + `side_1` crops (`configs/data.yaml`) resized
to 224; one ImageNet ResNet-18 shared by both cameras, fine-tuned at LR 1e-5; features concatenated
(1024) → 256 → 201 logits, cross-entropy. AdamW, 8k steps × batch 32, cosine with 500 warm-up, head LR 1e-4,
LeRobot-style photometric + small affine augmentation on both cameras (`baseline_lowlr_aug_full` in
`configs/train.yaml`). Checkpoint = lowest val CE (evaluated every 500 steps). 3 seeds.
V = Σ_b p_b · v_b; p_succ = mass on bins above −0.5.

**Artifacts.** Checkpoints `data/runs/g0_r0_s{0,1,2}/best.pt` (config.json and code hash next to each);
data under `/iris/u/ankile/karenvo/vl-data`.

## 2. Metrics (validation, 247 episodes, mean over 3 seeds)

| | R0 | repo-only constant | flat line (shape) |
|---|---|---|---|
| val CE | 3.672 ± 0.010 | | |
| val MAE of V | 0.114 | 0.140 | |
| eval-round MAE (r00–r04) | 0.183 | 0.222 | |
| shape MAE (episode mean removed) | 0.049 | | 0.037 |
| within-round success AUC at 1 / 5 / 10 s | 0.62 / 0.61 / 0.82 | 0.5 by construction | |
| success vs still-stall AUC, hold phase (p_succ) | 0.80 | | |

How to read: V beats the strongest trivial baseline (each round's training median), and inside one round,
where the round can't help, it separates eventual success from failure increasingly well over the first
10 s. Two caveats to state up front: (1) **shape**: within an episode V wiggles around the straight-line
target more than a flat line would (0.049 vs 0.037), which is what advantages V(o_{t+n}) − V(o_t) see;
(2) **resolution**: with 247 val episodes × 3 seeds, a CE change below ~0.025 can't be detected, so every
"tie" below is "no effect larger than that".

TODO after the merge gate: the same table on `r05-eval` and `r05-screen` (reported separately).

## 3. Three representative plots (synchronized cameras + V and p_succ vs target)

Same recipe as R0 (`aug_full_s0`, R0 reproduces it: CE 3.670 vs 3.672); folder
`reports/figures/stage4/aug_full_s0/`, videos in `data/review_videos/stage4/aug_full_s0/`.
- `01_autonomous_success_r04-eval_ep0039_dist.png`: V climbs with the approach, the success hump takes over at seating.
- `02_autonomous_failure_r04-eval_ep0020_dist.png`: an autonomous failure; where V drops relative to the event.
- `03_timeout_stall_r02-eval_ep0079_dist.png`: a stall at the holder; V stays hedged (~−0.45) while the target
  sits near −0.9 (the stall ambiguity).

TODO (Karen): pick which three after watching; regenerate on `g0_r0_s0` if the package must show the exact
checkpoint (`vl.value_plots --run g0_r0_s0 --dist --episodes-from lowlr_8k`, CPU ~3 min).

## 4. Failure examples (what V gets wrong, and why we think so)

1. **Stall vs pause at the holder.** One frame can't show how long the robot has been stuck, so long timeouts
   sit at ~−0.45 against targets near −0.9. Tested directly in Stage 5.5: a perfect "seconds still" input
   recovers only ~a third of the optimism (CE −0.016, n.s.), because a quarter of successes also freeze ≥ 8 s
   before seating. Image history (2 s / 4 s) gets the same small stall gain and hurts paused successes.
   (`reports/still_time_gain.md`, `reports/history_gain.md`)
2. **Round prior.** V at frame 0 already differs by round (≈ −0.55 r00–r02, −0.46 r03, −0.40 r04) before the
   robot moves: a session-appearance cue we haven't located. Within a round, frame 0 is at chance (AUC 0.55)
   and the signal appears in the first second of motion.
3. **Jitter.** Frame-to-frame |ΔV| is ~30× the target slope; it is almost entirely flips of the
   success/failure hump balance between near-identical frames. Augmentation doesn't remove it; a 0.5 s
   average does (see §6).
4. Example episodes: `09_worst_error_r04-eval_ep0101`, `10_worst_success_r01-eval_ep0103`,
   `06_assisted_failure_c03-dagger-mixed_ep0097` (looks like a success to V).

## 5. One comparison: what didn't help (all vs R0, paired bootstrap over episodes × seeds)

| change (one at a time) | val CE change [95%] | verdict |
|---|---|---|
| side camera only / wrist camera only (Stage 5) | +0.07 / +0.23 (MAE 0.130 / 0.191) | both cameras needed; side carries most |
| proprio added (Stage 5) | ~0 | no gain |
| frozen ImageNet encoder (Stage 5) | +0.35 | fine-tuning essential |
| DINOv2 ViT-S/14, three LR recipes (Stage 5) | worse | ruled out |
| crop / resize variants R1–R6 (Gate 0) | −0.023 (R6, 320 px) … +0.005 | no win |
| per-camera ResNets (C1) | −0.002 [−0.03, +0.03] | tie |
| attention fusion, 3 variants (C3) | +0.019 … +0.055 | worse |
| image history 2 s / 4 s (B2) | −0.004 / +0.001 | tie |
| perfect stillness signal (upper bound) | −0.016 [−0.041, +0.006] | tie |

(Stage 5 rows were vs the older `lowlr_8k` baseline; check before sending.) Message: at this data size the
single-frame, shared-ResNet model is where the easy gains stop; the remaining errors are either signal
limits (stalls) or below what 247 val episodes can resolve.

Camera contributions (C4, masking one camera at a time; `reports/camera_contrib.md`): in the last 5 s of
successes the wrist camera dominates (removing it costs +2.2–3.0 nats/frame, the side camera +0.04–0.2), as
Lars expected. On failures it is the side camera (removing it costs +0.6–1.9 nats), and removing the wrist
slightly *improves* the loss: the wrist view makes failures look promising, a candidate source of V's
optimism at stalls. Same pattern in R0, per-camera ResNets and attention fusion.

## 6. Inference recipe

From `reports/inference_choices.md` (post-processing the same 3 R0 models; paired bootstrap over val episodes):
- **3-seed ensemble + centred 0.5 s average of the distribution, no temperature**: val CE 3.630 vs 3.672
  single (−0.042 [−0.047, −0.036]), frame flicker −77%, shape MAE 0.044 vs 0.049, stall_auc 0.811 vs 0.800.
  Ensembling alone is −0.030, more than any trained change in Stage 5.5.
- Live (causal) version: ensemble + trailing 0.5 s, CE 3.656, same flicker cut.
- Temperature: ~0 on top of this, and slightly worse calibration; dropped.
- Not fixed by any post-processing: AUC 1 s, and the shape gap to a flat line (0.042 vs 0.037).

## 7. Next decision (for Lars)

- Freeze R0 (+ inference recipe) as the Stage 5.5 baseline, run the r05 test once.
- Phase 2 setup: see `reports/stage6_setup_prep.md` (π₀.₅ checkpoint and token access, IRIS compute, Rohan's
  rollout format, whether failures are kept, success labels, task).
- Whether to start advantage labels with this V while the π₀.₅ value head (6.1) is built.
