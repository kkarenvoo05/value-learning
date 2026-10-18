# Stage 3 data pipeline: checks and decisions (2026-10-18)

Covers PROGRESS checkpoints 3.1–3.6. Everything below ran on move2; commands are in the README.
Generated tables: `reports/data_integrity.md` (3.1). Generated figures (gitignored, regenerate
with `python -m vl.data_figures`): `reports/figures/stage3/`, review videos in
`data/review_videos/`. Choices are tagged [paper], [lars] or [ours].

## 3.1 Data on disk

- Downloaded `side_1` and the trajectory parquet for the 11 train-pool repos
  (`vl.fetch_videos --cams side_1 --data`); `wrist_left` was already there. 11 GB under
  `data/videos/` (home NFS, 6.8 GB left). Test repos not downloaded.
- `vl.check_data` [data]:
  - **sha256 of all 1,785 files equals the LFS sha256 at the pinned commit**; nothing missing,
    nothing extra.
  - **Video spans:** for every episode and both cameras, packet timestamps hold one frame at
    `from_timestamp + k/15` for every k = 0..T (3,248 episode-camera spans, packets only).
  - **Parquet:** row count = manifest `raw_length`, `frame_index` = 0..n−1, `timestamp` = k/15,
    the `is_valid` count equals the manifest's `n_valid` (read remotely in Stage 1), and for
    soft-truncated episodes `is_valid` covers exactly 0..T. 0 problems in 1,624 episodes.
  - One known exception: **r00/35 `wrist_left` is one frame short** (frames 0..395 of 397;
    T = 395). The audit already noted it. Its last slot in the packed file is the *next*
    episode's first frame, so the frame index now asserts that every indexed frame lies inside
    its own episode's `[from_timestamp, to_timestamp)` in both cameras (a test covers this).
- On-disk layout (for "where do the pixels for (repo, ep, t) live"):
  `data/meta/<repo>/meta/episodes/*.parquet` gives, per episode and camera, `chunk_index`,
  `file_index` and `from_timestamp`. The frame is at time `from_timestamp + t/15` in
  `data/videos/<repo>/videos/observation.images.<cam>/chunk-CCC/file-FFF.mp4`. r02–r04 store one
  episode per file (`from_timestamp` = 0); c00–c05, r00 and r01 pack up to 61 episodes per file.
  Trajectory rows are in `data/videos/<repo>/data/chunk-CCC/file-FFF.parquet`.

## 3.2 Frame index

- `vl.frames` → `data/frames/frame_index.parquet`: one row per in-support valid frame of a
  train/val episode, with label columns (for the loss and breakdowns only) and both cameras'
  `path`, `from_ts`, `to_ts`.
- **438,853 frames from 1,620 episodes** (train 373,947 / 1,373; val 64,906 / 247). Per split
  and outcome, the counts match `reports/targets_summary.md` exactly (train 38,750 failure /
  233,254 success / 101,943 timeout; val 6,350 / 41,685 / 16,871). No frame index > T. Test
  frames are not indexed.

## 3.3 Frame decoding

- `vl.frames.FrameReader` seeks by presentation timestamp and **refuses to return a frame more
  than half a frame away from the requested time**. It never takes "the next frame decoded",
  which is how the old `vl.preview.frame_at` bug arose.
- Verified [data]: **434/434 frames** (2 random episodes per repo × 11 repos × 2 cameras;
  frames 0, 1, T/2, T−1, T and 5 random ones) are byte-identical to the same frame taken from a
  plain sequential decode. This includes episodes in the middle of packed files.
  Synthetic tests (`tests/test_frames.py`): a numbered video with three packed "episodes",
  forward and backward seeks. The tests fail if the reader is changed to return frame t+1.
- Video format: AV1, 640×480, 15 fps, yuv420p, **a keyframe every 2 frames**, so a random
  seek decodes at most ~2 extra frames.
- **Speed (4 CPU cores on move2, no other load) and decision [ours]:**
  - One process: 57 samples/s decode-only, 43 samples/s with crop+resize (2 frames per sample).
  - DataLoader: 3 workers 86 samples/s, **4 workers 112 samples/s** (3.5 batches/s of 32).
  - GPU decode is not an option: the TITAN RTX (Turing) has no AV1 hardware decoder.
  - **Decision: decode on the fly.** It is exact, needs no extra storage, and gives about one
    pass over the training frames per hour. Pre-extracting 224×224 JPEGs (~877k images,
    ~10 GB) would be roughly 4× faster but does not fit in the home quota (6.8 GB free) and
    would need storage outside home. Revisit if Stage 4 is loader-bound. Local `/scr` has room
    but is not "NFS under Karen's name" (Lars's rule), so ask first.

## 3.4 Crops and preprocessing [ours unless tagged]

- Fixed square crop per camera, then bilinear resize to 224×224 [paper: 224×224 after a fixed
  per-camera crop], ImageNet mean/std [torchvision default for the pretrained ResNet-18]. No
  flips, no random crops. Boxes in `configs/data.yaml`:
  - `wrist_left` **(80, 0, 560, 480)**: full height, centred. The standard 420 px centre
    crop (110–530, 30–450) cut the top of the holder during approach and insertion in about
    half the sampled episodes.
  - `side_1` **(140, 0, 560, 420)**: 420 px, moved to the top and 30 px right. The scene
    (marker, holder, gripper at insertion) sits in the upper middle; the bottom ~60 px is
    empty table, and the centre crop cut the gripper at the top edge. The marker is ~15×5 px
    in the source and ~8×3 px at 224.
- Chosen from 11-repo sheets at t = 0, 0.4T, 0.7T, 0.9T, T and checked on the 8 initial-state
  extremes (min/max pen x/y, holder x/y) at 0, 0.5T, 0.85T, T. Marker, holder and gripper stay
  inside both boxes in all of them.
- The side camera is fixed. During approach the arm passes close in front of it and can fill
  most of the `side_1` frame. This is real content, not a decode error (wrist and side agree
  frame by frame).
- Images stay uint8 until the GPU; `normalize`/`unnormalize` round-trip exactly (test).
- Figures: `reports/figures/stage3/crops_{wrist_left,side_1}.jpg`. Each panel shows the raw
  frame with the box and, beside it, the model's input after normalize → unnormalize.

## 3.5 Check B: data alignment

- `reports/figures/stage3/check_b_alignment.jpg`: 8 training samples (failure/success/timeout
  mix, 4 repos) taken through `FrameDataset` → `default_collate` → `normalize` → back. Each is
  annotated with references recomputed **independently** of the frame index: the timestamp read
  from the trajectory parquet row, `is_valid` of that row, and v_t from the Section 8 formula.
  **8/8 agree** (value to 1e-9, bin center within half a bin, parquet ts = t/15,
  `is_valid` = 1). Viewed: in every pair the gripper state and arm position agree between the
  wrist and side views.
- **Camera-to-camera sync (quantitative):** for 3 episodes per repo, each camera's image
  motion (mean absolute frame difference, 80×60 gray) is correlated with end-effector speed from
  the parquet at lags −10..10.
  - `side_1`: wherever the correlation is clear (r ≥ 0.5, 16 of 33 episodes) the best lag is **0** in 16/16.
  - `wrist_left`: of the 22 episodes with r ≥ 0.5, the best lag is 0 in 17. In the other 5 it
    is +6 frames, with r within 0.06 of the lag-0 value (a flat curve, not an offset).
  - No episode shows the two cameras clearly peaking at different lags.
- The recurring +6 frames is probably a property of the motion, not of the files.
  - Autonomous gripper closings start on 6-frame boundaries (steps 42, 48, 54, …). This looks
    like the policies' action chunks, which would make end-effector speed roughly periodic
    with a 6-frame period. **Hypothesis, not checked.**
  - Gripper-close test (5 episodes per repo): the frame of fastest closing in the parquet vs
    the frame of largest change in the wrist finger region gives lag 0 in 23/53 episodes and
    +6 in 11/53. A constant video/parquet offset would give one lag everywhere.
  - The model sees images only, so this does not affect the baseline. **Recheck before the
    Stage 5 proprioception ablation.**
- **Review videos** (`data/review_videos/`, what the model sees: 224 wrist | 224 side, with t
  and the target value; frame counts = T+1):
  - `success_autonomous_r04-eval_ep0135.mp4`
  - `success_teleop_c00-teleop-mixed_ep0244.mp4`
  - `failure_r00-eval_ep0029.mp4`
  - `timeout_stall_r00-eval_ep0053.mp4`
  - `intervention_c02-dagger-mixed_ep0044.mp4`

## 3.6 Dataset and DataLoader

- `vl.dataset.FrameDataset` item: `{"images": {wrist_left, side_1: uint8 (3, 224, 224)},
  "bin", "value", "row"}`. Only `images` may go to the model. `row` points back into the frame
  index for breakdowns. Proprioception, actions, elapsed time, T, outcome, policy and
  time-to-go never leave the index. `tests/test_dataset.py` asserts the exact item keys and
  camera keys, and that item i is frame t of its own episode.
- One `FrameReader` per worker process: it is created lazily and re-created after a fork.
- Sampling [ours]: uniform over valid training frames (a shuffled pass, fixed seed,
  `drop_last`).
- Mix over the first ~1,900 sampled training frames vs the index:
  - success 0.60 vs 0.62, timeout 0.28 vs 0.27, failure 0.12 vs 0.10.
  - Every repo within 1 point of its share in the index.

## Open items carried forward

- Video ↔ parquet timing for proprioception (above): before Stage 5.
- Loader throughput (112 samples/s) may limit Stage 4. Pre-extraction needs a storage decision.
- The Q6 GPU question is still open (needed from 3.8 on).
