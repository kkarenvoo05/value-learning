# Operator visibility in `wrist_left` (2026-10-18)

Question: how often is the human operator (hand, arm, clothing) visible in the wrist camera
during the valid range 0..T, and does it track the outcome (shortcut risk for V)?

## Method

- Videos: pinned training-pool `wrist_left` downloaded with
  `.venv/bin/python -m vl.fetch_videos --cams wrist_left` (5.3 GB).
- Sampling: one frame per second (every 15th frame) over 0..T plus T, for every train/val
  episode: **31,519 frames, 1,620 episodes**. [ours]
- Detector: torchvision DeepLabV3-ResNet101 (COCO-trained, VOC labels), "person" pixels at
  probability > 0.5. A frame counts as *visible* if any person pixel is present
  (`person_frac > 0`), and *inside the crop* if any is inside the Resize(256)+CenterCrop(224) box.
  `.venv/bin/python -m vl.operator_visibility --cam wrist_left`
  -> `data/audit/operator_visibility_wrist_left.parquet`. [ours]
- Frame decoding was checked against `ffmpeg select=eq(n,f)` (exact match). This check found
  that `vl.preview.frame_at` returned frame f+1. Fixed; see "Side finding" below.

## Validation against hand labels

180 frames, stratified by detector score (60 with zero person pixels, 120 across four
positive bands), shuffled, scored blind to the detector output (labels by Claude from 400×300
tiles; disputed tiles re-viewed larger): `reports/operator_visibility_labels.csv`.
"Visible" = any part of a person or their clothing anywhere in the frame; `?` = too small or
ambiguous to call.

| detector band (person_frac) | Y | N | ? |
|---|---|---|---|
| 0 | 2 | 56 | 2 |
| (0, 0.5%] | 20 | 6 | 4 |
| (0.5%, 2%] | 37 | 3 | 0 |
| (2%, 5%] | 30 | 0 | 0 |
| > 5% | 20 | 0 | 0 |

Weighted by how common each band is (the bounds take `?` as N, then as Y):
**precision 0.91–0.94, recall 0.78–0.87; true prevalence 21–24% of frames** against 19.8%
flagged. The detector undercounts slightly. Its misses are slivers of clothing at the
frame edge, and its false positives are dark edge regions in the faintest band. The
breakdowns below use the detector, so they are slight underestimates.

## Results [data]

**How often:** **19.8% of sampled valid frames** (12.7% inside the crop box); **57.6% of
episodes** have at least one such frame, 13.1% have it in half or more of their frames. It is
not limited to a few rounds: every repo is between 14% and 27% of frames.

| | failure | success | timeout |
|---|---|---|---|
| P(visible), all frames | 0.101 | 0.198 | 0.238 |
| P(visible inside crop) | 0.055 | 0.124 | 0.164 |
| episodes with any visible frame | 0.411 | 0.615 | 0.555 |

The same numbers hold for fully autonomous episodes only (0.102 / 0.199 / 0.238), so this is
not the operator taking over. Human-controlled frames show it slightly more often (0.214 vs
0.159 for robot frames in assisted episodes).

**When:** it depends on the task phase, i.e. where the arm is pointing. Rows are elapsed time
in the episode, entries P(visible):

| elapsed | failure | success | timeout |
|---|---|---|---|
| 0–2 s | 0.08 | 0.09 | 0.07 |
| 2–5 s (grasp) | 0.04 | 0.03 | 0.03 |
| 5–10 s | 0.11 | 0.17 | 0.16 |
| 10–20 s (at holder) | 0.17 | 0.34 | 0.30 |
| 20–40 s | 0.09 | 0.35 | 0.31 |

Rows are time before T:

| before T | failure | success | timeout |
|---|---|---|---|
| 0–1 s | 0.16 | 0.31 | 0.33 |
| 1–3 s | 0.17 | 0.37 | 0.32 |
| 3–5 s | 0.10 | 0.29 | 0.31 |
| 5–10 s | 0.08 | 0.15 | 0.29 |

## Interpretation

- The operator shows up mainly when the arm is **at the holder**: the wrist camera then looks
  toward the side where the operator sits. Failures show it less often because most of them
  end early (missed grasp, drop) before the arm reaches the holder.
- **Between success and timeout,** the pair that the stall images already make hard to tell
  apart, the rates are similar at the same phase: 2–5 percentage points apart in the last
  5 s before T, and 4 points apart at 10–40 s elapsed. The one larger gap, 5–10 s before T
  (0.29 vs 0.15), is also phase. By then a timeout has been stalled at the holder for a
  while, but a success is still approaching. Seeing the operator mostly tells the model the
  task phase, and the arm pose in the image already shows that. Presence alone carries
  little extra outcome information.
- **Not ruled out:** this measures *presence*, not *what the operator does*. A hand
  moving toward the robot to stop a trial or reset it could still be a cue.
  That is better tested on the trained model: in Stage 4, compare predictions on matched
  frames with and without the operator, and look at the frames where V drops sharply.

**Recommendation [ours, for Karen/Lars to decide]:** do not crop or mask for the baseline. The
operator is in about 1 in 5 frames and mostly at the corners, and the evidence above does
not show an outcome shortcut. Keep `person_frac` per frame
so Stage 4 can break metrics down by operator-visible vs not. Masking the bottom-left corner
is the fallback if Stage 4 shows V reacting to the operator.

## Side finding: preview frames were one frame late

`vl.preview.frame_at(t)` returned the frame *after* the requested one (it kept the second of
the two frames ffmpeg outputs after seeking one frame early). Every Stage 1 filmstrip was
therefore labelled one frame (67 ms) late; the column marked T showed T+1. Fixed in
`src/vl/preview.py` (checked exact at frames 0, 1, 44, n−2, n−1) and the 16 review sheets
regenerated. A 67 ms shift does not change any Stage 1 conclusion. The manifest, endpoints
and targets never used this function.
