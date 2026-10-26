# Project progress

Legend: `[x]` done and verified · `[~]` partly done · `[ ]` not started.
Details for finished items are in `reports/dataset_audit.md`.

## Stage 0 — Setup
- [x] Project skeleton, `pyproject.toml`, `.venv` (uv), `.gitignore` (data/videos/checkpoints excluded)
- [x] `README.md` and `CLAUDE.md` for future sessions
- [x] `git init`
- [x] Storage. Lars: training data goes on NFS storage under Karen's name, **not** under `/move/`.
      **Moved (2026-10-23):** `data/` → `/iris/u/ankile/karenvo/vl-data` (38 TB NFS). Copied with
      rsync from orionp2 (no move2 login needed, `/sailhome` is mounted everywhere): 7,210 files,
      19,440,967,026 bytes, `rsync --checksum` dry-run clean. Old `/sailhome` copy deleted (home 96% → 9%).
      **Temporary solution (2026-10-17):** `data/` in the project is a symlink to
      `/sailhome/karenvo/vl-data` (NFS home, 20 GB quota, ~19 GB free after moving
      `~/.cursor-server` to `/move/u/karenvo/.cursor-server` behind a symlink). Fits the
      ~12 GB training cameras + data with a few GB to spare; the 3.3 GB test videos make it
      tight, so fetch those later. Home filling up breaks the shell, so check `df -h ~` before
      each download.
      **Permanent:** `/iris/u/karenvo` does not exist and Karen is not in the `finnlab` group that
      owns `/iris/u`. Ask Lars to get it created, then move `vl-data` there and re-point `data`.
      No other NFS mount on move2 is writable for this account (`/scr` is a local move2 disk).
      **2026-10-20: quota ran out** (96%, 855 MB free; each DINOv2 run is 363 MB). Job 17999270 seed 2
      died writing a checkpoint, and at 22:46 the uv Python binaries in `~/.local/share/uv/python`
      (3.11.16, which `.venv` uses, plus 3.8 / 3.9) were left as 0-byte files (cause unknown), so
      `.venv/bin/python` silently does nothing until it is reinstalled
- [ ] Compute (GPU for Stage 3+): ask Lars (Q6)
- [ ] Phase 2 compute (π₀.₅ fine-tuning): IRIS cluster
- [x] First commit

## Stage 1 — Dataset inspection, audit, splits (2026-10-17)
- [x] Inspect metadata before any video download (format, FPS, cameras, fields, provenance)
- [x] Schema consistency check across the 13 repos
- [x] Pin each repo to its `release-1` commit (`data/meta/pins.json`, in every manifest row)
- [x] Storage estimate (27.4 GB total; ~12 GB train-pool `side_1` + `wrist_left` + data)
- [x] Determine label precedence from release docs; resolve outcomes and valid endpoints T
- [x] Verify endpoints against per-frame `done` / `is_valid` (all 2,095 known episodes agree)
- [x] Flag unknown outcomes (4 train-pool episodes excluded, not turned into failures)
- [x] Standardized episode manifest (`data/manifest/episodes.parquet`)
- [x] Overlap audit (shared starts, duplicate content, source identity); train/test: no overlap
- [x] Group-aware splits: train 1,377 / val 247 / test_r05_eval 150 / test_r05_screen 200
- [x] Tests: identity, split disjointness, endpoints, synchronized sampling (14 passing)
- [x] Contact sheets for success, failure, soft-truncated timeout, intervention, CF demo, teleop
- [x] Video/trajectory sync check (ffprobe frame counts = row counts)
- [x] Initial-state-extremes crop check at f=0
- [x] Audit report + regenerable tables
- [x] Rebuilt on move2 (2026-10-17): same 15 pinned commits, identical split counts, all 26
      tests pass (the 8 manifest integrity tests now run instead of skipping)
- [~] Review ~10 episodes across categories: 10 random (seed 0) + 6 odd cases, 10-column
      filmstrips in `reports/figures/stage1_review/` (audit Section 5a). Labels, endpoints and sync
      agree in all 16. **Reviewed from still frames by Claude; Karen should still watch a few as
      video.**
- [x] Odd episodes: c02/82 is an aborted trial labeled timeout (same target either way);
      r03/5, 34, 35, 50 are genuine ~55 s stalls, kept
- [x] Mid-episode crop check: center crop keeps marker/gripper/holder opening at insertion in
      all 16; holder top leaves the wrist frame during approach (audit Section 5)
- [x] Operator visible in `wrist_left` measured (2026-10-18, `reports/operator_visibility.md`):
      ~20% of valid frames (21–24% after correcting with 180 hand labels; detector precision
      0.91–0.94, recall 0.78–0.87), 13% inside the crop, 58% of episodes. It follows the task
      phase (arm at the holder) rather than outcome: success vs timeout ≤5 points apart at
      matched time. **For review:** recommendation is no masking for the baseline, plus a
      visible/not-visible breakdown in Stage 4
- [x] Fixed `vl.preview.frame_at` returning frame f+1 (filmstrips were labelled 67 ms late;
      regenerated). Manifest, endpoints and targets were never affected

## Open questions
- [x] Q1: timeouts → **failure, as in RECAP** (Lars, 2026-10-17). Not censored.
- [x] Q2: cameras → **`wrist_left` + `side_1`**. Lars's meeting notes: of the 4 cameras, only 2
      are used for policies, the left wrist view and the side-on view. `side_1` is the only side
      camera in r00–r02, so it is that view. Which cameras and whether to add proprioception
      remain empirical questions (Stage 5).
- [x] Q3: storage → NFS under Karen's name, not under `/move/` (Lars). Now
      `/iris/u/ankile/karenvo/vl-data` (2026-10-23, see Stage 0).
- [ ] Q4: does Lars have an existing loader that applies reviewed outcomes and endpoint trimming?
- [ ] Q5: should `r05-screen` be a separate diagnostic distribution?
- [ ] Q6: what compute (GPU) and policy-evaluation setup can Karen use? (move2 has an idle
      TITAN RTX 24 GB: may Karen use it?)
- [ ] Q7: soft truncation: what rule did the reviewer use to pick the stall frame (all 210
      `soft_truncate` edits are human, `cv2-editor`)? And why do 47 training timeouts end
      before the cap without being soft-truncated (operator stopped the trial?) And why do 4 r03
      rollouts run 826–840 steps although r03 `results.json` has `max_steps: 800` and
      `no_auto_timeout: false`? r03/35 is a success at 826 steps, after the cap. Kept as
      labeled for the baseline (all fit inside H = 60 s)

- [ ] Q8: does a marker fully seated in *any* of the holder's 4 holes count as success, or is
      there a target hole? The release docs don't say. The VLM tag "wrong_hole_partial_insert"
      (16 train/val episodes: 14 failure, 2 timeout, 0 success) comes from a VLM prior, not a
      task definition. Doesn't change the baseline: partial inserts are not seated, so not
      success either way (r00/29, reviewed by Karen 2026-10-18)

## Guidance from the meeting with Lars
(Transcript: https://notes.granola.ai/t/393c9237-febd-434a-8d84-3418ff58cedb)
- Start point: implement the RECAP way of learning V on the Mulligan Insert Marker data.
- Simplest possible version first, get it working, then iterate.
- Understanding over code: know what V learns and verify it behaves as expected.
- Expected behavior: values rise roughly linearly on episodes that go well and drop where
  mistakes happen. Visualize this (Stage 4). The drops can only appear in the model's
  *predictions*; the training targets are straight countdowns.
- Model: ImageNet-pretrained torchvision ResNet-18 plus a categorical value head; swap in a larger
  backbone later only if needed.
- If one-GPU training is fast, try a few variations and compare (Stage 5).
- Next (Karen): RECAP value function implementation this weekend (2026-10-17/18).

## Stage 2 — Targets (Check A: target arithmetic)
- [x] Target design (seconds, C_fail = H = 60 s, v = R/120 s, 201 bins on [−1, 0]); all values
      in `configs/targets.yaml` with provenance tags
- [x] Freeze failure penalty, normalization, support, timeout rule (timeout = failure, Lars)
- [x] Implement target construction (`src/vl/targets.py`); out-of-support frames get `bin = -1`,
      are counted per split/outcome, never clipped
- [x] Tests on hand-computable synthetic episodes (`tests/test_targets.py`, 11 tests; checked
      that they fail on a floor-instead-of-round bin bug and on a missing failure penalty)
- [x] Run on the real manifest → `data/targets/frame_targets.parquet` (575,406 frames, 1,970
      episodes), `reports/targets_summary.md`, `reports/figures/target_examples.png`.
      0 out-of-support targets; independently recomputed per-episode first/last values match
- [x] Stall frames (marker held at the holder for tens of seconds) have rising targets on a
      nearly static image, so they are unfittable from one image (audit 5a). **Reviewed by
      Karen 2026-10-18:** kept in the baseline; analyzed separately in Stage 4
- [x] **Stage 2 reviewed by Karen (2026-10-18).** Decisions:
      - C_fail = H = 60 s, v = R/120 s kept. It is the smallest penalty that puts every failure
        frame below every success frame; not tuned so the longest episode hits exactly −1
        (lowest target −0.966). [ours]
      - ~~Seconds rather than steps~~ **Changed to timesteps (Karen + mentor, 2026-10-19):** reward
        −1 per timestep (frame), failure penalty 900 steps, R_t = −(T − t) − 900·[failed],
        v_t = R_t / 1800. Every repo is 15 fps, so this is exactly the old seconds version
        (60 s = 900 steps): rebuilt `frame_targets` and `frame_index` have identical bins and
        values within 1.1e-16 on all 575,406 / 438,853 frames, so **no retraining needed**; all
        trained runs stay valid. Stage 6 lookaheads are chosen in seconds and converted at 15 fps
      - Timeout endpoints kept as reviewed, including the 173 of 214 training timeouts that
        end before the 53.3 s cap (reviewer-truncated or recording ended); broken down in
        Stage 4; open question Q7
      - r03/35 (success at 826 steps, after the 800-step cap; validation split) kept as
        success, pending Q7
      - Targets rise over time for failures and timeouts too, as in RECAP's reward (−1 per
        step, −C_fail at the end); drops at mistakes can appear only in predictions

## Stage 3 — Data and model (target: Sun 2026-10-18)
Eight checkpoints, done in order. Each one ends with something Karen checks directly
(**You check**) before the next starts, so every piece is understood before it is built on.

### 3.1 Data on disk
- [x] Download pinned training-pool `side_1`, `wrist_left`, data parquet (2026-10-18;
      `vl.fetch_videos --cams side_1 --data`). 11 GB in `data/videos/` (home NFS, 6.8 GB left)
- [x] Integrity (`vl.check_data` → `reports/data_integrity.md`): sha256 of all 1,785 files =
      pinned LFS hashes; every episode × camera has frames 0..T at from_ts + k/15; parquet rows,
      frame_index, timestamps and is_valid agree with the manifest. Only exception: r00/35
      `wrist_left` is one frame short after T (known, harmless; the frame index now guards
      against reading into the next episode)
- **You check:** the on-disk layout for one episode: which mp4 file, which time range inside it
  (LeRobot v3 packs many episodes per file), which parquet rows. Be able to answer "where do the
  pixels for r02 episode 17, frame 300 live?" (worked answer: `reports/data_checks.md` 3.1)

### 3.2 Frame index: what one training example is
- [x] `vl.frames` → `data/frames/frame_index.parquet`: 438,853 frames from 1,620 episodes
      (train 373,947 / val 64,906); t ≤ T only; train/val only (test untouched)
- [x] Counts match `reports/targets_summary.md` per split and outcome exactly
- [x] **Checked by Karen (2026-10-18):** r00/29 t=100 recomputed by hand (v = −0.5311,
  bin 94), 157 rows = T+1, no label column reaches the model

### 3.3 Frame decoding
- [x] `FrameReader.read(path, from_ts, t, fps)` decodes by timestamp and raises if it would land
      on another frame. 434/434 real frames byte-identical to sequential decoding (11 repos ×
      2 cams, incl. packed files); synthetic tests fail on a t+1 bug
- [x] Decode speed: 4 workers → 112 samples/s (AV1, keyframe every 2 frames; no AV1 hardware
      decode on the TITAN RTX). **Decision [ours]: decode on the fly** (exact, no extra disk,
      ~1 h per pass over train frames); pre-extraction (~10 GB) doesn't fit the home quota.
      Revisit if Stage 4 is loader-bound
- **You check:** the measured speed, and the decision with its reason (frames/s needed for
  batch 32 at a useful GPU utilization vs frames/s the decoder gives)

### 3.4 Crops and preprocessing
- [x] Fixed square crops [ours], `configs/data.yaml`: `wrist_left` (80, 0, 560, 480) full height
      (centre crop cut the holder's top); `side_1` (140, 0, 560, 420), moved up and right onto
      the scene. 224×224 bilinear, ImageNet mean/std. No flips, no random crops
- [x] Contact sheets `reports/figures/stage3/crops_*.jpg` (phases, stall, failure end,
      initial-state extremes; raw + box next to the model's input after normalize→unnormalize)
- **You check:** the sheet. Nothing needed to judge progress (marker, gripper, holder opening)
  is cut off; un-normalize a tensor and confirm it renders back to the same image

### 3.5 Check B: data alignment
- [x] `reports/figures/stage3/check_b_alignment.jpg`: 8 samples through the real loader path,
      annotated with references recomputed independently (parquet timestamp and is_valid,
      Section 8 formula): 8/8 agree
- [x] Sync, measured: camera motion vs end-effector speed peaks at lag 0 for `side_1` in 16/16
      clear episodes and for `wrist_left` in 17/22 (other 5: flat curve, +6 within 0.06).
      Video ↔ proprio timing looked ambiguous at 0 or +6 frames. **Rechecked in Stage 5: lag 0;
      +6 is the 6-step action chunk** (`reports/proprio_sync.md`)
- [x] Review videos of what the model sees, `data/review_videos/` (5 episodes)
- **You check:** both views show the same moment (gripper pose agrees between wrist and side),
  late frames have v_t near 0 (success) or near −0.5 (failure), and t never exceeds T

### 3.6 Dataset and DataLoader
- [x] `vl.dataset.FrameDataset` → `{"images": {wrist_left, side_1}, "bin", "value", "row"}`; only
      `images` go to the model; labels stay in the index (looked up by `row`)
- [x] Tests: exact item keys / camera keys, item i is frame t of its episode, crop box, normalize
      round-trip (`tests/test_dataset.py`; 36 tests pass)
- [x] Uniform sampling over valid train frames [ours]; sampled mix vs index: success 0.60/0.62,
      timeout 0.28/0.27, failure 0.12/0.10; every repo within 1 point
- [x] Throughput: 3 workers 86, 4 workers 112 samples/s
- **You check:** the logged mix matches the index, and one batch's shapes, dtypes and value ranges

### 3.7 Model
- [x] `src/vl/model.py`: shared ImageNet ResNet-18 (same weights for both views) → concat 2×512
      → MLP (1024 → 256 → 201) → cross-entropy on the hard bin. Prediction V = Σ p_b · v_b over
      the 201 bin centers on [−1, 0] [paper: 201-bin categorical; architecture ours]. Takes the
      uint8 image dict; ImageNet normalization is inside `forward`. 11,490,569 parameters
      (backbone 11,176,512 + head 314,057)
- [x] Sanity on a real training batch (`python -m vl.model`, CPU): logits (16, 201); at
      initialization CE 5.380 (ln 201 = 5.303), max p 0.016 (uniform 0.005), V ∈ [−0.497, −0.487]
      (uniform −0.5). Not exactly uniform because the last layer has PyTorch's default random
      init [ours: kept]
- [x] Tests (`tests/test_model.py`, 9): shape; one shared backbone; changing only `wrist_left` or
      only `side_1` changes the output; cameras read by name; normalization inside; one-hot on
      bin b → v_b; uniform → V = −0.5 and CE = ln 201; a few steps fit one batch. Checked
      that the tests fail if normalization is dropped or one camera is fed twice
- Note [paper vs ours]: Mulligan's Insert Marker critic uses **one ResNet-18 per camera**
  (audit Section 5); our plan shares one backbone. Possible later ablation. With a shared
  backbone in train mode, BatchNorm statistics are computed over wrist and side images together
- **You check:** walk through `forward` line by line and explain why the head outputs 201
  logits rather than one number, and what Σ p_b · v_b means

### 3.8 Training step and Check C: tiny-subset overfit
- [~] GPU: Karen OK'd the `move` partition's rtxpro6000 (move5) for Stage 3.8 (2026-10-18). Q6
      (move2 TITAN RTX / standing compute) is still open with Lars
- [x] `src/vl/train.py --check-c`, `configs/train.yaml`, `scripts/check_c.sbatch`. Settings
      [ours, not paper]: 224×224 per view, batch 32, AdamW (wd 0.01), lr 1e-4 backbone / 1e-3
      head, bf16 autocast, seed 0; config, git commit and device logged to the run's
      `config.json`
- [x] **Check C passed (2026-10-18, job 17947295, RTX PRO 6000, 1,500 steps, 57 s)** on 4
      training episodes / 720 frames: autonomous success r02/54, failure r00/82, soft-truncated
      timeout r02/241, assisted DAgger success c03/93. Eval-mode CE 5.30 → **0.018**;
      MAE(V, v) **0.0014** (≈ 0.17 s); per-episode corr(V, v) ≥ 0.994. Plot:
      `reports/figures/stage3/check_c.png`; run in `data/runs/check_c/`
- Reading it: the pipeline (loader → targets → model → loss → optimizer) can learn the
  targets. It says nothing about generalization: 720 frames are memorized. Even the timeout's
  stall frames were fit (two small blips at 12.5–13.5 s), which on this few frames means
  memorizing tiny image differences. Whether V can handle stalls on *held-out* episodes is a
  Stage 4 question
- **You check:** the loss curve, and a predicted-vs-target value plot for each of the 4 episodes.
  Stall frames may stay hard even here (one image cannot tell how long the robot has been stuck)

## Stage 4 — Baseline training and evaluation (target: Mon 2026-10-19)
- [x] Train baseline (job 17947767, 2026-10-18, 40 min; `reports/baseline_results.md`). 40k steps
      × 32, best.pt = lowest val CE = **step 2,000**: val CE 3.873, MAE 0.119 (constant: 4.530,
      0.257). **Overfits after ~2k steps**: train CE 3.41 → 1.41, val CE 3.87 → 8.73 (worse than
      the constant from ~6k), val MAE plateaus at 0.13–0.14 (over-confident distributions)
- [x] Trivial baselines (`reports/baseline_results.md`), val MAE: global constant 0.257,
      elapsed-time-only 0.235, repo + time 0.146, **repo-only constant 0.140** vs model 0.119.
      Repo almost fixes the outcome (c00–c05 ~100% success, r00 4% … r04 53%), so most of the
      gain could be repo recognition. Within eval repos r01–r04, V's success AUC rises from 0.57
      at 1 s to 0.80 at 12 s (a progress signal). Original plan item: constant predictor;
      optional elapsed-time-only predictor (elapsed time only, never eventual duration or outcome)

### Baseline recipe: settle before Stage 5 (ablations compare against a fixed baseline)
- [x] **Decision 1, checkpoint metric (Karen, 2026-10-18):** select the checkpoint on per-frame
      validation **CE** (the training objective), and always report val **MAE** of V next to it
      (per frame and per episode). Already what `vl.train_baseline` does. [ours]
- [x] **Decision 2, overfitting** (best checkpoint at 0.17 passes; val CE worse than the
      constant from ~6k steps). Try in this order, one change at a time:
      - [x] (a) Shorter schedule with learning-rate decay (e.g. cosine to ~0 over a few thousand
            steps, more frequent val evals). Labels unchanged. **Try first.** Submitted
            2026-10-18: `RUN=cosine_8k SECTION=baseline_cosine sbatch scripts/baseline.sbatch`.
            8k steps (~0.68 passes), 500-step warmup then cosine to 0, full-val eval every 500
            steps; everything else as the step-2k baseline [ours]. Ran as job 17950654 on an
            **L40S** (move4; rtxpro6000s were full), 25 min. **Result: does not fix it.** Best
            at step 1,000 (val CE 3.839 vs 3.873, MAE 0.129 vs 0.119); val CE rises from step
            1.5k and passes the constant at 4.5k, before the LR has decayed much. Overfitting
            starts at ~0.09 passes, so it is not caused by revisiting frames, and the schedule
            is not the lever. Shape error grows steadily (0.043 at 500 → 0.071 at 8k, flat
            0.037). `reports/baseline_results.md`, "Decision 2(a)"
      - [x] (b) Lower learning rate (backbone and/or head), if (a) still overfits early.
            `RUN=lowlr_8k SECTION=baseline_lowlr`, both LRs 10x lower (backbone 1e-5, head
            1e-4), otherwise identical to (a) [ours]. Job 17954831, L40S, 19 min. **Meets the
            success criterion:** val CE stays within 3.73–3.88 from step 1.5k to 8k (never near
            the constant's 4.53); best at step 3k, val CE 3.727, MAE 0.117 (≤ 0.119); eval-repo
            MAE 0.188 vs repo-const 0.222; within-repo AUC 0.69 / 0.63 / 0.85. Shape error still
            worse than flat (0.050 vs 0.037): not an overfitting problem.
            **Settled recipe: `baseline_lowlr`** (confirmed by the 3-seed retrain on CE/stability; see below)
      - [ ] (c) not needed: (b) removed the over-confidence (kept for reference)
      - (c) Soft targets (Gaussian over neighbouring bins, "HL-Gauss") instead of one-hot.
            Changes target construction, not in RECAP: only with Karen's OK, and only if (a)/(b)
            leave the model over-confident
      - Success criterion: the best checkpoint is no longer an early accident (val CE flat or
        improving over a stretch of evals), and val MAE ≤ the step-2k baseline's 0.119
- [~] **Repo-aware evaluation** (the repo shortcut, `reports/baseline_results.md`):
      - [x] Add the repo-only constant (val MAE 0.140) as the trivial baseline the model must
            beat, next to the global constant, in every run's eval (`vl.metrics`, logged at every
            eval of `vl.train_baseline`; `python -m vl.metrics --run <run>` re-scores old runs)
      - [x] Report metrics within the eval repos r00–r04 (outcomes vary there), incl. the
            within-repo success AUC at 1/5/10 s. Step-2k baseline: eval-repo MAE 0.194 vs
            repo-constant 0.222; AUC 0.59 / 0.62 / 0.81
      - [x] Report the *within-episode* signal: `offset_free_mae` (subtract each episode's mean
            from V and v, then MAE), vs the same number for a flat line. A per-repo offset cancels
            in it, as it does in V(o_{t+n}) − V(o_t), the quantity Stage 6 advantages use. (The
            target's own change is the same fixed slope in every episode, so "does ΔV track Δv"
            is not informative.) **Step-2k baseline: 0.052, worse than flat (0.037)**: V's
            within-episode wiggles and jumps add error against the straight countdown
      - [~] Find what in the images reveals the repo. Done: `vl.analysis --repo-sheet` ->
            `reports/figures/stage4/first_frames_by_repo.png`; V at frame 0 already differs by
            round (r00–r02 ≈ −0.55, r04 ≈ −0.40), but no cue is obvious by eye (side-view glare
            varies, strongest in c00). Within a round, frame-0 V is at chance (AUC 0.55); the
            1 s AUC (0.66) comes from the first second of motion, not the initial layout or the
            policy. **Open: ask Lars** whether rounds differ in day/lighting/camera setup
            (`reports/baseline_results.md`, "Repo cues")
- [x] Retrain the baseline with the settled recipe, **3 seeds** (2026-10-19; seeds 1–2 jobs
      17958964/17958965, L40S, ~20 min each, `RUN=lowlr_8k_s<k> SECTION=baseline_lowlr SEED=<k>`;
      seed 0 = `lowlr_8k`). Best val CE 3.730 ± 0.002, MAE **0.121 ± 0.004**, eval-repo MAE
      0.196 ± 0.008, shape 0.047 ± 0.004 (flat 0.037), AUC 1/5/10 s 0.67/0.64/0.84 ± 0.01–0.02.
      **Seed 0 was lucky**: the mean MAE does not beat the step-2k baseline's 0.119; the gain is
      CE (3.73 vs 3.87) and no overfitting. **Stage 5 differences below ~2 sd (MAE 0.008,
      eval-repo MAE 0.016) are not evidence**
- [x] Karen's observations from the step-2k videos: Karen reports them addressed (2026-10-18)
- [x] Metrics (`vl.analysis`, `reports/stage4_analysis_lowlr_8k.md`): CE, MAE per frame and per
      episode, by outcome × kind (autonomous/assisted/DAgger/teleop) and source repo. Train/val
      gap (`vl.predict`, job 17958970, 250 train episodes): CE 3.55 vs 3.73, MAE 0.091 vs 0.117.
      **V hedges toward the middle on autonomous rollouts**: much better than the repo constant
      on successes (0.218 vs 0.447), worse on failures (0.133 vs 0.065) and timeouts (0.186 vs
      0.106); same on all 3 seeds
- [x] Synchronized video/value plots for 8–12 episodes chosen systematically (clean success,
      failure, recovery, intervention, typical error, worst error): both views, predicted value
      (raw; any smoothing labeled), MC target, intervention intervals, marked mistakes/recovery.
      Done for the step-2k baseline (9 episodes) and **the settled recipe** (`--run lowlr_8k`,
      11 episodes incl. a "drop failure"; PNGs in `reports/figures/stage4/lowlr_8k/`, videos
      in `data/review_videos/stage4/lowlr_8k/`). Human shading uses `source` = 1.
      Mistakes/recovery are not marked (no reviewed per-frame mistake labels). **Watched by
      Karen (2026-10-19): looks good, no data or label problems**
- [x] Note: the *targets* are straight countdowns and never drop at a mistake. A drop in the
      *prediction* is what we test for. Never force predictions to be monotonic.
- [~] Questions (`reports/baseline_results.md`, "Stage 4 questions"): **rises at seat: yes**
      (−0.29 at T−5 s → −0.02 at T); grasp: no visible step; **seated vs hovering: yes** (AUC
      1.0); **late rounds: yes** (beats repo constant on r02–r04, within-round AUC@10 s
      0.72–0.88); **fall at a drop: not testable** on val (5 VLM-labelled drops; the plotted one
      is a missed grasp); occlusion: the operator has no effect; **arm covering `side_1`: no level shift**
      (matched ΔV ≈ −0.01, all seeds; `vl.side_occlusion`, pixel-difference proxy), but the rare
      > 0.2 drops cluster when the arm leaves the view (cause open).
      Also: V jitters (frame-to-frame |ΔV| 30× the target slope); even 3 s smoothing stays
      worse than flat on shape
- [x] Assisted episodes analysed separately: human- vs robot-controlled frames have the same error
      (0.035 vs 0.031); only 1 assisted failure in val (c03/97, predicted as a confident success),
      so assisted failures cannot be analysed on this split
- [x] Operator visibility: matched (repo, outcome, 2 s bin) visible-minus-not ΔV +0.00…+0.06,
      ΔMAE ≤ 0 on all 3 seeds; sharp drops do not coincide with appearances (P 0.025–0.04 vs
      base 0.046; 0 of the drops > 0.2). Uses the 1 fps audit samples (not extended)
- [x] Timeout endpoints (val): soft-truncated MAE 0.174 (31 ep.), ended early 0.108 (5), cap
      0.283 (4). Truncated ones are **not** worse; capped are worst (long T → low targets).
      No label change needed; no need to ask Lars
- [x] Stall frames (heuristic: robot control, ≥ 3 s in, 3 s mean speed < 15 mm/s [ours]): 64% of
      timeout frames. V is flat in stalls (+0.02 vs target +0.145 per ~17 s segment), as expected;
      stall frames are not where timeout error concentrates (0.164 vs 0.224)
- [x] **Decide operator masking: no masking** (pre-set rule: V does not react to the operator).
      The Stage 5 masking ablation is dropped
- [~] Phase-one gate (Claude's assessment in `reports/baseline_results.md`; **Karen/Lars to
      sign off**): beats the trivial baselines overall (not on autonomous failures/timeouts
      alone); curves partly interpretable (seat, progress; jittery); failure cases mostly
      understood (stall ambiguity, repo prior hedging). Stalls are evidence that a short history
      or elapsed-time input is needed (Stage 5 "current frame vs short history")

## Stage 5 — Ablations and test (target: Tue 2026-10-20)
One change at a time, in this order:
- [x] 1. Cameras: wrist + side (= the `lowlr_8k` baseline, 3 seeds, done) vs **side only** vs
      **wrist only** (Karen, 2026-10-19), 3 seeds each, same recipe, compared on val. Also check
      which single-camera model still tells the round at frame 0 (where the repo cue lives) and
      whether the drops at arm retraction from the side view survive without `side_1`.
      Not adding `wrist_right`/`side_2` [ours]: `wrist_right` is the other lens of the same wrist
      camera (near-duplicate), and `side_2` is missing in r00–r02 (500 of 1,624 training-pool
      episodes), always shows the seated operator (a shortcut risk), and is black in r05-screen
      b02 eps 0–37 (test)
      Submitted 2026-10-19: jobs 17970458 (side, `side_s{0,1,2}`) and 17970459 (wrist,
      `wrist_s{0,1,2}`), one L40S each on humanoid1 (resubmitted from move), seeds run in sequence (`scripts/cameras.sbatch`)
      **Result (`reports/stage5_ablations.md`): keep both cameras.** Val MAE / CE over 3 seeds: both
      0.121 / 3.730, side only 0.130 / 3.799, wrist only 0.191 / 3.963 (worse than the repo-only
      constant 0.140). The side view carries most of the signal **and the round cue** (side-only
      frame-0 V has the baseline's round ordering; wrist-only's is reversed). The drops when the arm
      leaves the side view vanish in wrist only, so they come from `side_1`
- [x] 2. Images vs images + proprioception (end-effector coordinates; inspect fields and frames
      first; normalize with training statistics only). `vl.proprio`, `tests/test_proprio.py` (4)
      - [x] **Timing recheck (2026-10-20, `reports/proprio_sync.{md,png}`): video frame t = state row t
            (lag 0)** for both cameras. 220 train episodes, pooled lag curves for side~EE speed,
            wrist~EE speed, wrist~EE rotation, wrist fingers~gripper; all 12 peak at 0, sub-frame within
            −0.10…+0.26 frames. Teleop (no chunks) has a single peak at 0. The Stage 3 "+6" is the
            6-step action chunk (`n_action_steps: 6`; autonomous speed autocorrelation bumps at 6 and 12)
      - [x] Fields: `observation.state` = EE xyz (m) + Euler extrinsic `xyz` (rad; roll wraps at ±π) +
            gripper (= `cartesian_position` + `gripper_position` exactly). Reported cartesian velocity is
            ~1–2 rows ahead of position differences
      - [x] Candidate input [ours]: xyz (3) + 6D rotation (first 2 columns of R) + gripper (1) = 10,
            z-scored with train-split stats (`data/proprio/stats.json`, 373,947 frames). **Excluded:**
            velocities (not synchronous), actions (policy outputs, leak policy identity), `pen_*` /
            `holder_*` (per-episode constants = start layout, not proprio), telemetry wrench/torques
            (possible later variant: contact force may separate "pressing, stuck" from "seating"),
            `steps_to_go`/`success`/`done` (labels). Joint positions are an alternative to EE pose
      - [x] Review videos (`data/review_videos/stage5_proprio/`, 7; PNGs in
            `reports/figures/stage5/proprio/`): both crops + the 10 z-scored inputs as bars + curves
            with a cursor. Incl. the val worst-case pairs r04/101, r02/45 (timeouts held at the holder)
            and r03/68 (late seat), plus teleop and DAgger. **For Karen to watch**
      - Early look, 8 val mirror-pair episodes (`holder_z_compare.png`, by eye, not a test): during the
        hold, EE height is ~230–250 mm for both timeouts and successes; successes drop ~30 mm only in
        the last ~1 s (the seat, already visible in the images). So EE height may not separate stuck
        from about-to-seat. Gripper width at hold differs (0.77 vs 0.81) in 2 timeouts: probably grasp
        position on the marker; unchecked
      - [x] Model (2026-10-20, git `a39e42a`) [ours]: the 10 raw numbers are z-scored inside `forward`
            (train stats as buffers) and concatenated with the 2×512 image features into the same head
            (1034 → 256 → 201); no separate proprio encoder. +2,560 parameters. `--proprio` on
            `vl.train_baseline`; off by default, image-only items and checkpoints unchanged. Tests: item
            is row t, missing frame raises, normalization inside, proprio changes the output (63 pass).
            Real-data check: dataset proprio = parquet state to 3e-8; CPU smoke run trains and reloads
      - [x] Run: job 17996700 (move5 RTX PRO 6000, git `d12ed67`, `proprio_s{0,1,2}`, ~13 min/seed).
            **Result (`reports/stage5_ablations.md` §2): no gain.** Val CE 3.733 ± 0.021 vs 3.730 ± 0.002,
            MAE 0.125 ± 0.003 vs 0.121 ± 0.004, eval-repo MAE 0.197 vs 0.196, AUC 1/5/10 s 0.69/0.63/0.82 vs
            0.67/0.64/0.84. Hold-phase success-vs-stuck AUC 0.765 vs 0.778; worst-case stuck timeouts drift
            slightly toward fail within seed noise, late-seat successes unchanged. No new round shortcut.
            **Keep the image-only baseline**; move on to short history / elapsed time
- [ ] 3. Frozen vs fine-tuned encoder. Fine-tuned arm = `lowlr_8k` / `_s1` / `_s2` (backbone at
      `lr_backbone` 1e-5). Frozen arm: `--freeze-backbone` [ours], same `baseline_lowlr` recipe,
      cameras and head LR; ImageNet backbone gets no optimizer group and stays in eval mode, so
      BatchNorm statistics don't move either (`tests/test_model.py`). ~0.3M trainable params.
      Running 2026-10-19: job 17971269 (`frozen_s{0,1,2}`, move5 **RTX PRO 6000**, not the
      fine-tuned arm's L40S; seeds in sequence, git `2eeb32c`, `scripts/frozen.sbatch` with
      `--partition=move --gres=gpu:rtxpro6000:1 --cpus-per-task=32`; humanoid job 17971059
      cancelled while queued)
      **Result (2026-10-20, from `data/runs/frozen_s{0,1,2}/result.json`; not yet written into
      `reports/stage5_ablations.md`): frozen is much worse.** Val MAE 0.234 / 0.235 / 0.234, val CE
      4.081 / 4.088 / 4.083 (fine-tuned: 0.121 / 3.730), eval-repo MAE ~0.275 (worse than the
      repo-only constant 0.222), AUC 1/5/10 s ~0.62 / 0.54 / 0.59. Best step 7.5k, 8k, 8k = the end
      of training, so the frozen arm may also be under-trained. ImageNet features alone don't carry
      the signal; fine-tuning the encoder matters (relevant to the backbone experiment below)
- [ ] Image augmentation (Karen, 2026-10-19; plan: `reports/stage5_augmentation_plan.md`), aimed at
      the mentor's concern that V before the last ~10 s looks random. Distribution plots show two
      humps (success/failure) with V their average: real outcome uncertainty (not fixable by
      augmentation) plus flicker of the hump balance between near-identical frames (maybe fixable).
      - [x] Step 0 (seed 0; s1/s2 distributions run first inside job 17979141): `vl.flicker` ->
            `reports/flicker_stage5.md`, `vl.value_dist --plots` -> `reports/value_dist_lowlr_8k.md`.
            Pre-end (> 2 s before T): |dV| 0.0176/frame, |dp_succ| 0.031, TV 0.067, R2(hump flip) 0.97
            (frame jumps are almost all hump-balance flips; success hump tracks the target, MAE 0.031).
            A centred 0.5 s average cuts |dV| 4x (0.0042) at equal MAE and better shape MAE (0.042 vs
            0.046); 1 s: 0.0026. So frame flicker is free to remove; the rule's "beat 0.5 s smoothing"
            needs a ~76% cut. Slower wander |V(t)-V(t-1s)| 0.040 (target 0.008/s), smoothing only to
            0.026-0.030: this is the part augmentation could still help (report-only metric)
      - [x] Step 1: arms A1 photometric, A2 crop shift/scale, A3 both; 3 seeds each, val never augmented
            Implemented 2026-10-19 (`vl.dataset` augment, sections `baseline_lowlr_aug_{photo,crop,both}`,
            tests; real-frame check `reports/figures/stage5/augment_examples.jpg`: marker, gripper and
            holder stay in view). Humanoid jobs 17977330-2 cancelled unstarted (queue full);
            resubmitted 2026-10-20 on move rtxpro6000 (Karen: max 3 GPUs on move): jobs 17978977 (photo),
            17978978 (crop), 17979141 (both; replaced unstarted 17978979), runs `aug_<arm>_s{0,1,2}`, seeds in sequence, each seed
            followed by `vl.value_dist` (`DIST=1`) for the flicker metrics
            **All 9 runs finished (2026-10-20).** Val, best checkpoint, 3 seeds:

            | arm | val MAE | val CE | best step |
            |---|---|---|---|
            | baseline lowlr_8k | 0.121 ± 0.004 | 3.729 ± 0.002 | 2–3k |
            | A1 photometric | 0.118 ± 0.005 | 3.725 ± 0.003 | 2–4k |
            | A2 crop | 0.116 ± 0.003 | 3.656 ± 0.006 | 7–8k |
            | A3 both | 0.119 ± 0.001 | 3.665 ± 0.001 | 6.5–7.5k |

            MAE is the same for every arm within seed spread. Both crop arms lower CE by ~0.07 (well
            beyond noise) and peak later (7–8k vs 2–3k), i.e. they overfit later. Crop's best step is at
            or near the end of training, so 8k steps may be too few for it
      - [~] Step 2: pre-set rule: accuracy within 2 sd, pre-end |ΔV| and |Δp_succ| ≥ 25% lower and
            better than 0.5 s smoothing, real uncertainty kept (failure/stall plots stay two-humped)
            **Computed 2026-10-20, verdict: no arm passes; augmentation fails its goal.** Pre-end (> 2 s
            before T), 3 seeds:

            | arm | \|ΔV\| per frame | \|Δp_succ\| | 1 s wander |
            |---|---|---|---|
            | baseline | 0.0158 ± 0.0019 | 0.0274 ± 0.0040 | 0.0366 |
            | A1 photometric | 0.0155 (−2%) | 0.0269 | 0.0367 |
            | A2 crop | 0.0169 (+7%) | 0.0300 | 0.0390 |
            | A3 both | 0.0154 (−3%) | 0.0272 | 0.0366 |
            | baseline smoothed 0.5 s | 0.0042 (−73%) | 0.0074 | 0.0301 |

            Best arm cuts flicker 3% (rule: ≥ 25% and beat 0.5 s smoothing). The 1 s wander doesn't move
            either; two-hump structure unchanged (R² of hump flips ~0.97 in every arm). So the jitter
            is not overfitting to pixel-level appearance; 0.5 s smoothing at inference stays the cheap fix.
            **Still to do:** these numbers are only in the draft `data/runs/flicker_aug_draft.md` (copied
            from another session's scratchpad). Make them official by regenerating
            `reports/flicker_stage5.md` with the same `vl.flicker` command (the 3-seed baseline replaces
            the old seed-0-only rows), and write the Step 2 verdict into `reports/stage5_ablations.md`.
            **Decision for Karen:** crop's CE gain could justify adopting crop for accuracy, as a
            separate decision from flicker; if considered, first check whether crop keeps improving past
            8k steps
      - [ ] **Crop sweep (Karen, 2026-10-21): find the best crop, then reuse it on other backbones.**
            Why: crop is the only change that has lowered val CE (−0.07). Re-read of the A2 runs:
            (a) **cut off by the schedule, not by overfitting**: val CE still drifting down at 8k
            (s0 3.692 at 3k → 3.650 at 8k) with train CE 3.53 vs baseline 3.21, while the cosine LR
            reaches 0 at 8k; (b) **the gain is not uniform**: eval-repo MAE 0.180 (0.196) but shape MAE
            0.052 (0.046) and AUC 1 s 0.62 (0.67), so part of the CE gain may be wider, better-calibrated
            distributions rather than better tracking; (c) **the jitter was weaker than configured**: the
            box is clamped into the frame, so `wrist_left` (full height) gets no vertical shift at scale
            1.0, `side_1` (y0 = 0) never shifts up, and scale 0.9–1.0 only zooms in; (d) data-bound (54%
            data wait, 17 min per 8k on RTX PRO 6000).
            Code: `crop_pad: true` in `augment` [ours] lets the box leave the frame, filled with the
            ImageNet mean colour (0 after normalization), so shifts are not clamped and `crop_scale` > 1
            zooms out. Default (no `crop_pad`) checked byte-identical to `2df9f76`'s `jitter_box` /
            `crop_resize` (5,000 seeded boxes per camera, same crop pixels); +3 tests (73 pass).
            Extreme boxes of each arm on real frames: `reports/figures/stage5/crop_sweep_extremes.jpg`
            (marker, holder and gripper stay in view in every corner case; s72's side-view corners come
            closest to cutting the gripper top). Sections `baseline_lowlr_crop16k*` = `baseline_lowlr` +
            augment + 16k steps + eval every 1000 (500 would add ~15 min per run) [ours].
            - [x] Step 1, length: `baseline_lowlr_crop16k` = A2 unchanged (clamped) at 16k, 3 seeds:
                  `NAME=crop16k SECTION=baseline_lowlr_crop16k sbatch --gres=gpu:rtxpro6000:1
                  scripts/cameras.sbatch` -> `crop16k_s{0,1,2}`. Does crop keep improving past 8k?
            - [x] Step 2, strength screen (seed 0, padded, 16k): `p24` (A2 + pad: reference separating
                  padding from strength), `s48`, `s72` (shift, scale 0.9–1), `z75`, `z60` (scale 0.75–1 /
                  0.6–1, shift 24), `zout` (scale 1.0–1.15, shift 24), `strong` (shift 48, scale 0.75–1).
                  `ARMS="p24 s48 s72 z75" sbatch --gres=gpu:rtxpro6000:1 scripts/crop_sweep.sbatch` and
                  `ARMS="z60 zout strong" ...` -> `crop16k_<arm>_s0`.
            - [ ] ~~Step 3~~ (no arm qualified): 3 seeds for the top 1–2 arms. **Pre-set rule:** an arm wins only if val CE beats
                  `crop16k` (step 1) by > 2 sd with shape MAE (`offset_free_mae`) and eval-repo AUC 1 s no
                  worse than `crop16k`'s; otherwise it is choosing blur. Then carry the winner to DINOv2
                  (`EXTRA="--backbone dinov2_vits14_reg"`)
            - [ ] Free check: test-time crop averaging on existing checkpoints (no training)
            - Submitted 2026-10-21 (Karen: two A5000s on move3, run sequentially; the interactive TITAN RTX
              was not viable: 4 CPUs gave 103 samples/s, ~3.8 h per run, inside a 16 GB shared cgroup
              already at 15.8 GB). Code uncommitted on top of `2df9f76`, diff in
              `data/runs/crop16k_code_vs_2df9f76.diff`. `sbatch --gres=gpu:a5000:1 --cpus-per-task=24
              --mem=32G --time=1-00:00:00 --wrap ...` running the two scripts above:
              job **18004638**: `crop16k_s0`, then `p24 s48 s72 z75`; job **18004639**: `z60 zout strong`,
              then `crop16k_s1`, `crop16k_s2`. ~910 samples/s on A5000 (data wait 1%)
            - **Result (2026-10-21, both jobs completed, 28–36 min per run). Longer is worse, and no
              stronger crop beats A2; the step-1 hypothesis was wrong.** Best checkpoint, val:

              | run | best step | val CE | val MAE | shape MAE | eval-repo MAE | AUC 1 s | val CE at 16k |
              |---|---|---|---|---|---|---|---|
              | A2 crop 8k (3 seeds) | 7–8k | **3.656 ± 0.006** | 0.116 | 0.052 | 0.180 | 0.62 | (3.650–3.665 at 8k) |
              | crop16k (3 seeds) | 3–5k | 3.685 ± 0.006 | 0.119 | 0.050 | 0.185 | 0.64 ± 0.04 | 3.756–3.801 |
              | p24 (A2 + pad) | 3k | 3.701 | 0.126 | 0.047 | 0.194 | 0.62 | 3.755 |
              | s48 | 3k | 3.712 | 0.130 | 0.047 | 0.197 | 0.62 | 3.722 |
              | s72 | 7k | 3.713 | 0.117 | 0.058 | 0.175 | 0.64 | 3.736 |
              | z75 | 3k | 3.707 | 0.125 | 0.047 | 0.194 | 0.67 | 3.728 |
              | z60 | 10k | 3.698 | 0.118 | 0.058 | 0.181 | 0.61 | 3.703 |
              | zout | 7k | 3.723 | 0.114 | 0.056 | 0.180 | 0.63 | 3.766 |
              | strong | 3k | 3.720 | 0.131 | 0.050 | 0.195 | 0.65 | 3.730 |

              (1) **The 8k crop wasn't cut off by the schedule; the annealing is what helped.** Same crop:
              the 8k and 16k curves match to 4k (3.680 / 3.682), then the 8k run's LR goes to 0 (factor
              0.55 at 4k, 0 at 8k) and CE falls to 3.650, while the 16k run (LR factor 0.88 at 4k, 0.53
              at 8k) overfits instead: train CE 3.26 vs 3.53, val 3.72 at 8k. 16k steps = 1.37 passes
              over the train frames vs 0.68, so frames repeat. crop16k is 4.5 sd worse than A2.
              (2) **Padding alone hurts** (p24 +0.019 vs crop16k_s0, ~3 seed sd): vertical shifts of the
              full-height wrist box now add mean-colour strips. (3) **Stronger crop slows overfitting
              (z60 best at 10k, lowest CE at 16k, 3.703) but lowers no peak**; every padded arm is worse
              than crop16k_s0 (3.682) at its best. (4) The MAE/shape trade-off persists: arms with better
              MAE (s72, z60, zout: 0.114–0.118) have the worst shape MAE (0.056–0.058). AUC 1 s is too
              noisy for single seeds (crop16k seeds 0.62–0.68).
              **Rule verdict: no arm wins.** Caveat: the screen ran on the 16k schedule, which step 1
              shows is the wrong one, so strength is not ruled out at 8k (z60/s72 overfit slowest).
              Best recipe stays **A2 crop at 8k** (`baseline_lowlr_aug_crop`). Step 3 not run.
            - [x] Step 2 redo at 8k (Karen, 2026-10-21): same 7 padded arms as sections
                  `baseline_lowlr_crop8k_<arm>` = `baseline_lowlr_aug_crop` with only `augment` changed,
                  seed 0, compared against A2 (3.656 ± 0.006). Job **18009514** (`z60 s72 p24 z75 s48`) and
                  the tail of job **18009513** (`zout strong`) -> `crop8k_<arm>_s0`. Rotation (optional in
                  the plan) not run. New `DROP_LAST=1` in both sbatch scripts deletes each new run's
                  `last.pt` (final weights + optimizer, unused by the analysis; home had 2.5 GB free).
                  Diff `data/runs/crop8k_dinov2s_crop_code_vs_2df9f76.diff`
                  **Result (2026-10-21, both jobs completed 13:19 / 13:24): no arm beats A2; A2 stays the
                  crop.** Best checkpoint, val, seed 0:

                  | arm | best step | val CE | val MAE | shape MAE | eval-repo MAE | AUC 1 s |
                  |---|---|---|---|---|---|---|
                  | A2 crop (3 seeds) | 7–8k | **3.656 ± 0.006** | 0.116 | 0.052 | 0.180 | 0.62 |
                  | s48 | 8k | 3.660 | 0.119 | 0.053 | 0.182 | 0.62 |
                  | p24 (A2 + pad) | 8k | 3.663 | 0.117 | 0.052 | 0.181 | 0.65 |
                  | z75 | 8k | 3.668 | 0.120 | 0.054 | 0.183 | 0.65 |
                  | z60 | 8k | 3.671 | 0.122 | 0.055 | 0.182 | 0.63 |
                  | s72 | 8k | 3.672 | 0.119 | 0.052 | 0.184 | 0.63 |
                  | strong | 8k | 3.676 | 0.121 | 0.053 | 0.184 | 0.66 |
                  | zout | 8k | 3.683 | 0.117 | 0.054 | 0.181 | 0.61 |

                  (1) **Padding is harmless at 8k** (p24 +0.007, ~1 sd): the 16k padding penalty came from
                  the schedule, not the padding. (2) **Stronger crops still lose on the right schedule**:
                  s48 is closest (+0.004, within 1 sd, one seed); z60/s72/strong/zout are 2.5–4.5 sd worse
                  and none improves MAE, shape MAE or eval-repo MAE. Every arm peaks at the last step.
                  **Rule verdict: no arm qualifies for step 3.** Best recipe stays **A2 crop at 8k**.
            - [x] DINOv2 + A2 crop (Karen, 2026-10-21): job **18009513**, `NAME=dinov2s_crop
                  SECTION=baseline_lowlr_aug_crop EXTRA="--backbone dinov2_vits14_reg" DIST=1 DROP_LAST=1`
                  -> `dinov2s_crop_s{0,1,2}`. Compare against `dinov2s_s*` (same recipe, no crop: CE 3.795,
                  best step 1.5k) and A2 ResNet (3.656)
                  **Result (2026-10-21, 3 seeds): crop helps DINOv2 twice as much as ResNet, but DINOv2 +
                  crop only ties ResNet + crop.**

                  | run (3 seeds) | best step | val CE | val MAE | shape MAE | eval-repo MAE | AUC 1 s |
                  |---|---|---|---|---|---|---|
                  | DINOv2, no crop | 1.5k | 3.795 ± 0.030 | 0.122 | 0.050 | 0.195 | 0.61 |
                  | DINOv2 + crop | 3k / 4.5k / 5k | 3.652 ± 0.005 | 0.117 ± 0.001 | 0.049 | 0.188 ± 0.001 | 0.64 ± 0.01 |
                  | ResNet + crop (A2) | 7–8k | 3.656 ± 0.006 | 0.116 ± 0.003 | 0.052 | 0.180 ± 0.004 | 0.62 ± 0.02 |

                  Crop drops DINOv2's CE by 0.14 (ResNet: 0.07) and moves its best step from 1.5k to
                  3–5k, consistent with crop slowing DINOv2's overfitting. Against ResNet + crop: CE and
                  val MAE tie (< 1 sd); **eval-repo MAE is worse by 0.008 (~2 sd, every seed 0.187–0.189)**,
                  the metric that matters for the unseen test round; AUC 1 s slightly better, within seed
                  spread. Still peaks before ResNet. **Verdict: no reason to switch backbones**
      - [x] **Full image augmentation** (2026-10-21; standard settings suggested by Lars on Slack, the
            LeRobot ImageTransforms scheme reimplemented in torchvision, no LeRobot import): per image,
            exactly 3 of 6 transforms (uniform without replacement, applied in a fixed order):
            brightness 0.8–1.2, contrast 0.8–1.2, saturation 0.5–1.5, hue ±0.05, sharpness 0.5–1.5,
            affine (±5° rotation, ±5% translate, nearest, black fill). Each transform hits half the
            images. Differs from what we ran: sharpness and rotation never tried; saturation and hue
            stronger than A1; A1/A2 applied every transform to every image; no zoom. `augment.full`
            in `vl.dataset` (`full_augment`, `pick_subset`) applied after the 224 px crop; section
            `baseline_lowlr_aug_full` = `baseline_lowlr` + `augment` only (no crop jitter: the affine
            translate stands in for A2's shift). +4 tests (77 pass). Real frames:
            `reports/figures/stage5/full_aug_examples.jpg` (marker/holder/gripper in view; hue can
            turn the orange holder yellowish). 4 ms per image. Judge on eval-repo MAE as well as val
            CE: Lars's aim is generalization to the unseen test round, which val CE (held-out groups of
            the training repos) barely measures. Job **18009813** (after 18009514, A5000), 3 seeds,
            `DIST=1 DROP_LAST=1` -> `aug_full_s{0,1,2}`; diff `data/runs/aug_full_code_vs_2df9f76.diff`.
            Karen moved it forward: 18009813 cancelled unstarted, resubmitted as job **18011075**
            (`sbatch --gres=gpu:l40s:1 scripts/cameras.sbatch`, same env), move4 L40S, git `2df9f76` +
            2 dirty files, 12:49–13:45 (~19 min/seed, data wait 43–56% vs 1% for crop: the
            transforms are CPU-bound). Config checked against `lowlr_8k`: identical except `augment`.
            **Result (3 seeds): beats the baseline clearly and ties A2 crop; crop still ahead on CE.**

            | run (ResNet-18, 3 seeds) | best step | val CE | val MAE | shape MAE | eval-repo MAE | AUC 1 s |
            |---|---|---|---|---|---|---|
            | baseline `lowlr_8k` | 2–3k | 3.730 ± 0.002 | 0.121 ± 0.004 | 0.047 ± 0.004 | 0.196 ± 0.008 | 0.67 ± 0.02 |
            | full aug | 6.5k / 7.5k / 7.5k | 3.670 ± 0.009 | **0.113 ± 0.000** | 0.050 ± 0.001 | 0.181 ± 0.002 | 0.61 ± 0.01 |
            | A2 crop | 7–8k | **3.656 ± 0.006** | 0.116 ± 0.003 | 0.052 ± 0.001 | 0.180 ± 0.004 | 0.62 ± 0.02 |

            (1) vs baseline: CE −0.06 (crop −0.07), eval-repo MAE −0.015, best step 2.5k → ~7k, so like
            crop it slows overfitting. (2) vs crop: CE worse by 0.014 (~2 sd), best val MAE of any run
            (0.113, three seeds within 0.001), eval-repo MAE a tie, shape MAE slightly better (0.050 vs
            0.052). (3) **AUC 1 s drops with both augmentations** (0.61–0.62 vs 0.67, ~3 sd): better
            CE/MAE, worse short-horizon outcome separation. Check this before adopting either.
            Not yet run: `vl.flicker` on `aug_full_s*` (distributions are written), crop + full aug
            combined. Videos (seed 0, same 11 episodes as `lowlr_8k`):
            `vl.value_plots --run aug_full_s0 --dist --episodes-from lowlr_8k` ->
            `reports/figures/stage4/aug_full_s0/`, `data/review_videos/stage4/aug_full_s0/`
            (11 `_dist` mp4s, 6.4 MB). Episode MAE, `lowlr_8k` s0 → full aug s0: worst timeout r04/101
            0.484 → 0.221 (halved), other timeouts r02/79 0.120 → 0.101, r03/34 0.196 → 0.169; the
            late-seating successes get worse, r04/39 0.223 → 0.303, r01/103 0.353 → 0.401; c03/97
            unchanged (0.464, unpredictable topple); the rest within ±0.03. One seed, 11 episodes:
            illustrative only.
            **Why (checked on all val frames, 3 seeds per arm; script in the 2026-10-21 session
            scratchpad, `psucc_shift.py`):** augmentation lowers p_succ on failure and timeout frames
            and leaves success frames alone. Mean p_succ, baseline → full aug (crop): timeouts 0.31–0.34
            → 0.28–0.29 (0.24–0.26), failures 0.32–0.38 → 0.22–0.33 (0.21–0.31), successes unchanged
            within 0.03 in every time-to-end window (crop: 0.01–0.04 lower). The hump means (V_if_succ
            ≈ −0.01…−0.12, V_if_fail ≈ −0.60…−0.67) don't move in any arm, so the whole change is hump
            choice. Frame MAE by outcome: timeout 0.198 → 0.174 (crop 0.160), failure 0.153 → 0.138
            (0.127), success 0.086 → 0.084 (0.097); that is where the val MAE gain comes from, and why
            crop costs the successes and full aug doesn't. Success-vs-timeout separation of per-frame
            p_succ within each window is unchanged (AUC 0.93–0.99 in all arms): augmentation shifts
            non-success frames down, it does not teach the model to tell the two apart. Reading:
            regularization against the baseline's overconfident success calls (it overfits by 2–3k)
            [interpretation, not tested]. The "mirror pairs move together" reading of the 11 episodes
            doesn't hold across seeds: r04/101 halves in all 3 seeds with both augmentations (0.48 →
            0.19–0.22) and r04/39 worsens in all 3 (0.22–0.25 → 0.27–0.34), but r02/91 (timeout) gets
            worse, r03/132 (success) gets better, the rest are mixed within seed spread. Episode-level
            causes not looked at (would need the frames where p_succ changes).
            **Where the shift lands** (`psucc_bucket_rtm.py`; supersedes `psucc_by_bucket.py`, which
            bucketed on the 3-seed baseline mean and compared against that same mean, so extreme buckets
            carried regression to the mean, e.g. half of the "−0.20 on hard timeouts"). Frames bucketed
            by `lowlr_8k` s0's p_succ; full aug (3-seed mean) and crop (3-seed mean) compared against
            baseline s1/s2 (mean), which carries the same regression to the mean (up to ±0.17 in the
            extreme buckets). Change in p_succ vs baseline s1/s2:

            | baseline s0 p_succ | failure: full / crop | timeout: full / crop | success: full / crop |
            |---|---|---|---|
            | 0–0.3 | −0.10…−0.13 / −0.10…−0.13 | −0.08…−0.09 / −0.10 | −0.05 / −0.07 |
            | 0.3–0.5 | −0.09 / −0.13 | −0.05 / −0.06 | −0.04 / −0.06 |
            | 0.5–0.7 | −0.04 / −0.11 | +0.01 / −0.04 | +0.03 / −0.03 |
            | 0.7–0.9 | −0.00 / −0.05 | −0.10 / −0.13 | +0.05 / −0.00 |
            | 0.9–1.0 | +0.01 / −0.02 | −0.03 / −0.14 | +0.01 / −0.01 |

            **Full aug sharpens**: frames move further toward the side the baseline already leans, for
            every outcome, including hard successes (low-p success frames −0.04…−0.05: worse) and
            confident successes (+0.01…+0.05: better). Hard failures don't move (0.7–1.0: −0.00/+0.01);
            hard timeouts partly (0.7–0.9: −0.10). Net ~0 on success frames because the two success
            ends cancel. **Crop = the same sharpening on low-p frames plus a downward shift**: lower
            in all 18 cells (one at −0.001), including every success bucket (−0.00…−0.07), with larger fixes on hard
            non-success frames (timeouts 0.7–1.0: −0.13/−0.14; failures 0.5–0.9: −0.05…−0.11).
            Hypotheses, untested: (a) sharpening = longer effective training (both augmented arms peak
            ~7k vs 2–3k; CE training sharpens along learned features, and can't fix frames whose
            features point the wrong way); (b) crop's downward shift = losing absolute pixel position
            (fixed cameras, so position encodes arm pose near the holder), making near-holder states
            uncertain, and those frames are mostly from long non-success hovers. Consistent with crop's
            success cost sitting at 1–15 s and > 15 s to the end, not in the last 1 s (0.984 vs 0.982).
            Tests: (a) rerun one baseline seed saving per-frame predictions at 2.5k and 7k; (b) the
            listed test-time crop check: shift the baseline's and full aug's inputs by A2's range and see
            whether near-holder p_succ drops
      - [x] **Crop + full augmentation** (Karen, 2026-10-21): section `baseline_lowlr_aug_crop_full` =
            A2 crop jitter (shift ±24, scale 0.9–1.0, clamped) + the `full` block with **affine translate
            0** (rotation ±5° kept) [ours: crop already shifts the image, every stronger shift lost in the
            crop sweep, so a second shift would mostly re-test strength]. Checked: differs from
            `baseline_lowlr_aug_crop` and `baseline_lowlr_aug_full` only in `augment`; 77 tests pass.
            Job **18012739**, move5 RTX PRO 6000 (same GPU type as A2), `NAME=aug_crop_full
            SECTION=baseline_lowlr_aug_crop_full DIST=1 DROP_LAST=1 sbatch --gres=gpu:rtxpro6000:1
            scripts/cameras.sbatch` -> `aug_crop_full_s{0,1,2}`, started 15:43, ~2,100 samples/s, data
            wait 2%. Diff `data/runs/aug_crop_full_code_vs_2df9f76.diff`. Compare against A2 crop (CE
            3.656 ± 0.006) and full aug (3.670 ± 0.009) on val CE, val MAE, eval-repo MAE, AUC 1 s and
            per-outcome MAE (does it keep crop's hard-case fixes without crop's success cost?)
            **Result (completed 16:28, 3 seeds): no gain; it behaves like crop alone.**

            | run (ResNet-18, 3 seeds) | best step | val CE | val MAE | shape MAE | eval-repo MAE | AUC 1 / 5 / 10 s |
            |---|---|---|---|---|---|---|
            | baseline `lowlr_8k` | 2–3k | 3.730 ± 0.002 | 0.121 ± 0.004 | 0.047 | 0.196 ± 0.008 | 0.67 / 0.64 / 0.84 |
            | A2 crop | 7–8k | **3.656 ± 0.006** | 0.116 ± 0.003 | 0.052 | **0.180 ± 0.004** | 0.62 / 0.64 / 0.81 |
            | full aug | 6.5–7.5k | 3.670 ± 0.009 | **0.113 ± 0.000** | 0.050 | 0.181 ± 0.002 | 0.61 / 0.61 / 0.83 |
            | crop + full | 7.5–8k | 3.658 ± 0.005 | 0.118 ± 0.002 | 0.051 | 0.183 ± 0.003 | 0.60 / 0.64 / 0.82 |

            Frame MAE by outcome (failure / success / timeout): baseline 0.153 / 0.086 / 0.198, crop
            0.127 / 0.097 / 0.160, full 0.138 / 0.084 / 0.174, crop + full 0.129 / 0.098 / 0.164.
            Crop + full ties crop on CE (< 1 sd) and matches it outcome by outcome, including crop's
            success cost (0.098): full aug's success-preserving behaviour does not survive stacking.
            Eval-repo MAE (highest) and AUC 1 s (lowest) are the worst of the three augmented arms, but
            within ~1 sd.
            Val CE still edging down at 8k on s1/s2 (3.656 → 3.654). **Verdict: don't stack; A2 crop
            stays the best single recipe on CE, full aug on MAE.** Distributions written
            (`val_probs.npz`), flicker not run
      - [x] **Crop × full ensemble, offline** (Karen, 2026-10-22): average two runs' 201-bin val
            distributions per frame (`val_probs.npz`, renormalized from float16), recompute V and CE,
            score with `vl.metrics.score` (script `ensemble.py` in the 2026-10-21 session scratchpad).
            Check: single-run CE/MAE recomputed this way equal `result.json` to 4 decimals. Controls:
            two seeds of the same arm. Mean ± sd over pairs (crop+full: 6 pairs from 3 + 3 seeds, not
            independent):

            | ensemble | val CE | val MAE | shape MAE | eval-repo MAE | AUC 1 s | MAE success / failure / timeout |
            |---|---|---|---|---|---|---|
            | single crop | 3.656 ± 0.006 | 0.116 | 0.052 | 0.180 | 0.62 | 0.097 / 0.127 / 0.160 |
            | single full | 3.670 ± 0.009 | 0.113 | 0.050 | 0.181 | 0.61 | 0.084 / 0.138 / 0.174 |
            | crop + crop | **3.638 ± 0.004** | 0.116 | 0.050 | 0.180 | 0.63 | 0.096 / 0.126 / 0.160 |
            | full + full | 3.650 ± 0.006 | **0.112** | **0.048** | 0.181 | 0.61 | 0.084 / 0.138 / 0.173 |
            | crop + full | 3.641 ± 0.006 | 0.114 | 0.049 | 0.180 | 0.62 | 0.090 / 0.132 / 0.166 |

            **No combination gain: crop + full lands between crop + crop and full + full on every
            metric**, i.e. it interpolates the two arms' trade-off instead of keeping the best of each.
            The gain is all seed averaging: any 2-model ensemble lowers CE by ~0.02. Larger ensembles
            (single values): crop ×3 3.632, full ×3 3.643, all 6 3.627 (not comparable: 6 models);
            **baseline ×3 3.674** (single 3.730), so ensembling the un-augmented baseline recovers ~75%
            of crop's CE gain, keeps the baseline's AUC 1 s (0.67) and best shape MAE (0.044), but not
            its MAE (0.121) or eval-repo MAE (0.195). Reading: much of the CE gain from augmentation
            may be less over-confident distributions, which averaging also gives [interpretation].
            Implication for a crop/full *split* in training: likely the same interpolation
      - [x] **Test-time crop shift, hypothesis (b) check** (Karen, 2026-10-22; no training):
            `vl.shift_check` feeds each finished run's best.pt the val frames with one A2 box per frame
            and camera (shift ±24, scale 0.9–1.0, clamped; drawn once from seed 0, same pixels for every
            run), on both cameras, side_1 only, or wrist_left only. Runs: `lowlr_8k` ×3, `aug_full` ×3,
            `aug_crop` ×3. Job **18022006** (move3 A5000, 13 min; `sbatch --gres=gpu:a5000:1
            --cpus-per-task=24 --mem=48G scripts/shift_check.sbatch`; first try on L40S cancelled, move4
            unavailable). Output `data/runs/shift_check/frames.npz`, report `reports/shift_check.md`.
            Checks: fixed-crop pass matches each run's `val_predictions.parquet` (max |ΔV| ≤ 0.008, AMP);
            crop runs move ≤ 0.016 in p_succ under any shift (the control: they were trained on it).
            **Result: (b) as stated is wrong about the camera. Shifting side_1 makes the baseline *more*
            optimistic; shifting wrist_left reproduces crop's signature.** Baseline, 3 seeds:

            | baseline input | val CE | MAE success / failure / timeout | Δp_succ failure / timeout / success (1–15 s) |
            |---|---|---|---|
            | fixed | 3.729 | 0.086 / 0.153 / 0.198 | — |
            | side_1 shifted | 3.986 | 0.079 / 0.245 / 0.331 | +0.17…+0.18 / +0.24…+0.25 / +0.00…+0.01 |
            | wrist_left shifted | 3.759 | **0.097 / 0.141 / 0.172** | −0.04 / −0.07…−0.08 / −0.02…−0.03 |
            | (A2 crop model, fixed) | 3.656 | 0.097 / 0.127 / 0.160 | −0.07…−0.09 / −0.07…−0.08 / −0.01…−0.02 |

            Full aug (trained with ±5% translate) is in between: side shift +0.02 on non-success, −0.01…−0.02
            on success; wrist shift lowers all outcomes (−0.02…−0.07). Reading [interpretation]: the
            baseline leans heavily on the side camera's absolute position (CE +0.26 when shifted), but
            the success cost tracks the **wrist** crop: the wrist camera rides on the gripper, so where the
            holder sits in its image is the gripper–holder alignment ("about to seat"), and shifting it
            removes exactly that cue. Caveat: a test-time shift on an un-augmented model is
            out-of-distribution, so this shows where the models are position-sensitive, not what
            training with the shift does. That is step 2
      - [x] **Per-camera split** (Karen, 2026-10-22): A2 crop jitter on one camera, the unchanged
            `full` block on the other. `augment.per_camera` [ours] in `vl.dataset` (+1 test, 78 pass;
            real frames checked: each camera changed only by its own transform). Sections
            `baseline_lowlr_aug_split_cropside` (crop side_1, full wrist_left: the arm the shift check
            predicts keeps crop's fixes without its success cost) and `..._cropwrist` (the reverse,
            predicted to keep the success cost); each differs from `baseline_lowlr_aug_crop` only in
            `augment`. Jobs **18022316** (cropside) and **18022317** (cropwrist), A5000, `NAME=aug_split_<arm>
            SECTION=baseline_lowlr_aug_split_<arm> DIST=1 DROP_LAST=1 sbatch --gres=gpu:a5000:1
            --cpus-per-task=24 --mem=48G scripts/cameras.sbatch` -> `aug_split_<arm>_s{0,1,2}`. Diff
            `data/runs/aug_split_code_vs_2df9f76.diff`. Compare against A2 crop and full aug on val CE,
            eval-repo MAE, AUC 1 s and MAE by outcome (success cost: crop 0.097, full 0.084)
            **Result (move3 A5000; cropside 03:16–04:32, cropwrist 03:40–05:00; 3 seeds each): the
            prediction was backwards. Crop on the wrist + full on the side is the best single recipe so
            far; crop on the side carries crop's success cost.**

            | run (ResNet-18, 3 seeds) | best step | val CE | val MAE | shape MAE | eval-repo MAE | AUC 1 / 5 / 10 s | MAE success / failure / timeout |
            |---|---|---|---|---|---|---|---|
            | baseline `lowlr_8k` | 2–3k | 3.729 ± 0.002 | 0.121 ± 0.004 | 0.046 | 0.196 ± 0.008 | 0.67 / 0.64 / 0.84 | 0.086 / 0.153 / 0.198 |
            | A2 crop | 7–8k | 3.656 ± 0.006 | 0.116 ± 0.003 | 0.052 | 0.180 ± 0.004 | 0.62 / 0.64 / 0.81 | 0.097 / 0.127 / 0.160 |
            | full aug | 6.5–7.5k | 3.670 ± 0.009 | 0.113 ± 0.000 | 0.049 | 0.181 ± 0.002 | 0.61 / 0.61 / 0.83 | 0.084 / 0.138 / 0.174 |
            | crop + full (stacked) | 7.5–8k | 3.658 ± 0.005 | 0.118 ± 0.002 | 0.051 | 0.183 ± 0.003 | 0.60 / 0.64 / 0.82 | 0.098 / 0.129 / 0.164 |
            | split: crop side, full wrist | 7.5k | 3.670 ± 0.007 | 0.118 ± 0.003 | 0.054 | 0.181 ± 0.004 | 0.63 / 0.64 / 0.83 | 0.100 / **0.122** / 0.160 |
            | **split: crop wrist, full side** | 7–7.5k | **3.647 ± 0.008** | **0.112 ± 0.000** | 0.049 | **0.179 ± 0.001** | 0.62 / 0.62 / 0.82 | 0.088 / 0.133 / 0.164 |

            (1) **cropwrist keeps most of both arms' good sides in one model**: success MAE 0.088 (full
            0.084, crop 0.097: ~2 sd better than crop), timeout 0.164 (crop 0.160, full 0.174), val MAE
            0.112 (best of any single run, three seeds within 0.0005), eval-repo MAE 0.179 (tie, lowest
            spread). It matches the offline crop + full *ensemble* (3.641 / 0.114 / success 0.090) with
            one model. (2) CE vs A2 crop −0.009 is ~1.5 sd (seed ranges 3.639–3.654 vs 3.650–3.662
            overlap): suggestive, not established. (3) cropside has the largest success cost of any arm
            (0.100) and the best failure MAE (0.122): the success cost follows **crop on side_1** in
            training, the opposite of what the test-time shift check suggested (there, shifting
            wrist_left reproduced crop's profile). So the shift check's reading (wrist position encodes
            alignment) does not carry over to training: test-time position sensitivity was not a guide to
            what training-time shifts cost [lesson]. (4) AUC 1 s stays below the baseline in both arms
            (0.62–0.63 ± 0.03 vs 0.67), as for every augmented arm. Distributions written
            (`val_probs.npz`); flicker and shift check not run on the split arms. Videos (2026-10-22,
            seed 0, same 11 episodes as `lowlr_8k`, CPU 3 min): `vl.value_plots --run
            aug_split_cropwrist_s0 --dist --episodes-from lowlr_8k` ->
            `reports/figures/stage4/aug_split_cropwrist_s0/`, `data/review_videos/stage4/aug_split_cropwrist_s0/`
            (11 `_dist` mp4s, 6.5 MB). Episode MAE, `lowlr_8k` s0 → full aug s0 → cropwrist s0: r04/101
            0.484 → 0.221 → 0.268, r03/34 0.196 → 0.169 → 0.162, r02/79 0.120 → 0.101 → 0.097, r04/39
            0.223 → 0.303 → 0.295, r01/103 0.353 → 0.401 → 0.377, c03/97 0.464 → 0.464 → 0.465. One seed,
            11 episodes: illustrative only.
            **Candidate recipe: `baseline_lowlr_aug_split_cropwrist`.** Before adopting it over A2 crop on
            CE: more seeds (CE gap is within ~1.5 sd); AUC 1 s is still the open cost of augmentation
      - [x] **Temperature scaling + 3-seed ensembles, offline** (Karen, 2026-10-22; CPU, 2.5 min):
            `data/runs/temp_ensemble.py` (run from the repo root) -> `data/runs/temp_ensemble.csv`. From
            each run's `val_probs.npz` (renormalized): temperature = log p / T, T on a 0.01 grid in
            [0.5, 3] minimizing CE, **cross-fit** over val episode groups (130 `group_id`s split in half,
            seed 0: fit on one half, score the other, both ways) so T is never scored on its own frames
            [ours]. Ensemble = mean of the 3 seeds' distributions. Check: recomputed single-run CE/MAE match
            `result.json` to 4 decimals. Means ± sd over seeds; ensembles are one value each:

            | arm | variant | val CE | val MAE | eval-repo MAE | AUC 1 s | ECE p_succ | MAE success / failure / timeout |
            |---|---|---|---|---|---|---|---|
            | baseline | single | 3.729 | 0.121 | 0.196 | 0.67 | 0.034 | 0.086 / 0.153 / 0.198 |
            | baseline | + T (1.10–1.19) | 3.719 | 0.125 | 0.197 | 0.67 | 0.032 | 0.094 / 0.146 / 0.192 |
            | baseline | ensemble ×3 | 3.674 | 0.120 | 0.195 | 0.67 | 0.033 | 0.085 / 0.151 / 0.197 |
            | A2 crop | single | 3.656 | 0.116 | 0.180 | 0.62 | 0.025 | 0.097 / 0.127 / 0.160 |
            | A2 crop | ensemble ×3 | 3.632 | 0.115 | 0.180 | 0.63 | 0.021 | 0.096 / 0.126 / 0.159 |
            | full aug | single | 3.670 | 0.113 | 0.181 | 0.61 | 0.027 | 0.084 / 0.138 / 0.174 |
            | full aug | + T (1.14–1.21) | 3.655 | 0.116 | 0.183 | 0.61 | 0.017 | 0.094 / 0.130 / 0.168 |
            | full aug | ensemble ×3 | 3.643 | 0.112 | 0.180 | 0.60 | 0.023 | 0.084 / 0.138 / 0.173 |
            | crop wrist, full side | single | 3.647 | 0.112 | 0.179 | 0.62 | 0.023 | 0.088 / 0.133 / 0.164 |
            | crop wrist, full side | + T (1.11–1.17) | 3.639 | 0.115 | 0.181 | 0.62 | 0.015 | 0.094 / 0.128 / 0.160 |
            | **crop wrist, full side** | **ensemble ×3** | **3.622** | **0.111** | **0.178** | 0.61 | 0.017 | 0.087 / 0.132 / 0.163 |
            | crop wrist, full side | ensemble ×3 + T (1.08–1.10) | 3.619 | 0.113 | 0.179 | 0.61 | 0.013 | 0.092 / 0.128 / 0.160 |

            (Full table incl. crop + full and crop side arms in the CSV: same pattern.) (1) **Temperature is
            not the missing calibration.** Every model is mildly overconfident (T = 1.04–1.21, consistent
            across halves and seeds), but T buys only −0.004…−0.015 CE and costs +0.003 val MAE and
            +0.005…+0.010 success MAE in every arm (softening drains the success hump). On the baseline it
            recovers ~14% of crop's CE gain, vs ~75% from the 3-seed ensemble. So the 2026-10-22 reading
            "much of the augmentation CE gain may be less over-confident distributions" is **weakened**:
            what the ensemble fixes is seed-specific errors, not a uniform overconfidence one temperature
            can remove [interpretation]. (2) **Ensembling helps every arm by ~0.02–0.05 CE and keeps the
            ranking**: crop wrist ×3 3.622 < crop ×3 3.632 < full ×3 3.643; one ensemble per arm, so this is
            weak extra support for crop wrist's lead, not a seed test. (3) Neither fixes AUC 1 s (ensembles
            0.60–0.63, baseline 0.67). (4) T on top of an ensemble: ~0 (T ≈ 1.1, CE −0.003). **Decision
            input:** best available is the crop-wrist 3-seed ensemble, no temperature; temperature only
            if a calibrated p_succ is what Stage 6 needs (ECE 0.023 → 0.015), at a small MAE cost
      - [x] **Episode-level comparison: crop / full / crop + full / split (crop wrist)** (Karen,
            2026-10-22; CPU): `data/runs/aug_compare.py` -> `reports/figures/stage5/aug_compare/`
            (`01_metrics.png` seeds per metric, `02_spread_vs_noise.png`, `03_top_episodes.png`, `04`–`11`
            one per top episode with both camera crops at the frame of largest disagreement),
            `data/runs/aug_compare_episodes.csv`. Per frame, each arm's 3-seed-mean V; per episode, arm
            spread = mean (max − min over arms), seed spread = mean sd over seeds; top 8 by arm spread
            among episodes with arm spread > 2× seed spread [ours] (165 of 247 pass, so most differences
            are above seed noise, but the bulk sit near the 2× line; 8 stand clearly out).
            **The biggest differences are successful DAgger holds at the holder, and they split by whether
            side_1 is cropped.** c05/59, c04/85, c03/66, c04/76 (successes with human frames): crop and
            crop + full drop p_succ to 0.15–0.5 for seconds while the marker sits above the hole, while
            full and split stay at ~0.95 (c04/85 episode MAE: crop 0.14, crop + full 0.16, full 0.02,
            split 0.02). This is crop's success cost at episode level, and it follows the side-camera crop,
            matching the split result. The other top episodes are eval-repo stalls (r02/44, r01/48
            timeouts; r04/19 failure): full aug is the most over-confident (r02/44 p_succ ~0.85 vs crop
            ~0.5, MAE 0.34 vs 0.22), crop / crop + full best, split in between. Best arm per episode (MAE
            range > 0.02): successes full 28, split 17, crop + full 6, crop 5; failures + timeouts crop 12,
            crop + full 12, split 8, full 2. Episode-mean success MAE: DAgger crop / both 0.040–0.042 vs
            full / split 0.030; eval crop 0.241, both 0.241, full 0.222, split 0.230. Reading: split takes
            full's behaviour on confident holds and part of crop's caution on stalls; it doesn't tell
            stall from seat any better [interpretation; the stall-vs-seat ambiguity of the worst-case
            analysis remains]
- [x] **Worst-case analysis of `lowlr_8k` (2026-10-20, seed 0, val, `val_probs.npz`)**: why some
      episodes' predicted V is nowhere near the target. Per-episode table:
      `data/runs/lowlr_8k/worst_case_episodes.csv`; filmstrips with the distribution shaded:
      `reports/figures/stage4/lowlr_8k/mismatch/` (rendered with `vl.value_plots.episode_outputs`,
      ad-hoc script, not in `src/`). p_succ = probability mass above −0.5; "success hump" / "failure
      hump" = the mean of the distribution above / below −0.5.
      - **Most error is picking the wrong outcome, not a bad value inside it.** MAE 0.117 as predicted
        vs 0.045 if the correct hump's own mean were used (~62% of MAE is hump choice). By outcome,
        total vs within-hump: success 0.086 / 0.028, failure 0.141 / 0.089, timeout 0.186 / 0.072.
        Every top-30 episode shows sensible hump means (~−0.1 / ~−0.64) with p_succ on the wrong side
      - **The worst cases are mirror pairs at the holder.** Timeouts with p_succ ≈ 0.85–0.95 held for
        10–40 s, success mass at −0.02…−0.1 ("about to seat"): r02/44, r02/45, r01/48, r02/91, r04/101
        (worst, MAE 0.484). Successes with p_succ ≈ 0.2–0.35 that seat only in the last 1–2 s, then V
        jumps: r03/68, r02/42, r01/103, r03/132. The end of r02/45 (timeout) and 8–12 s of r03/68
        (success) look the same (marker tip under the hole). Same stall ambiguity as audit §5a; the bins
        show the model commits rather than hedges
      - **The commitment is a per-episode judgement with real signal, not flicker or a leak.** Hold
        phase (> 5 s in, > 2 s before T; 45 successes vs 45 `timeout_holding_marker` episodes): 72% of
        p_succ variance is between episodes; AUC(success vs stall) 0.77 from mean p_succ (episode length
        alone 0.65), 0.80 within the same start (only 25 pairs). p_succ clusters by start config
        (permutation p = 0.004) but so does the outcome itself (p = 0.03); no start is shared between
        train and val. So the worst cases are the confident tail of a useful but imperfect classifier
      - **Overconfident on the success side** (autonomous eval frames): p_succ ≈ 0.80 → 56% actually
        succeed; p_succ ≈ 0.96 → 75%. Low bins are calibrated (0.05 → 3.5%, 0.20 → 22%)
      - **The failure hump is time-blind.** Failure-hump mean vs target by time to T: 0–2 s −0.60 vs
        −0.51; 20–30 s −0.63 vs −0.70; 40–60 s −0.66 vs −0.89. corr(failure-hump mean, target) 0.32 vs
        0.69 for the success hump; min V anywhere −0.608. A stalled robot's time left is not in one
        frame, so long timeouts keep ~0.22 error even with the right hump
      - Early-failure modes are fine (missed grasp, pregrasp misalignment: MAE 0.04–0.09). The hard
        class is `timeout_holding_marker` (also when labelled failure) vs success
      - **c03/97** (DAgger failure, p_succ ≈ 1 throughout, MAE 0.464): label checked at full resolution
        (`mismatch/c03_97_end.jpg`) and it is **correct**: the marker topples in the last 2 frames during
        the takeover (frame 145–147). Not predictable earlier; DAgger failures are rare (10 train vs 469
        successes)
      - Repo-level p_succ roughly tracks each round's success rate (r00 0.24, r04 0.61)
- [x] **Q: would a harsher failure penalty fix the two-outcome problem? No** (reasoning, not run).
      The humps exist because the frame is ambiguous; C_fail only sets where the failure hump sits and
      adds no information. Hump gap = c_fail/(H + c_fail): 0.5 now, 0.75 with c_fail = 2700, so a wrong
      call costs more and blended V swings more. The countdown slope halves (1/1800 → 1/3600 per step;
      one bin ≈ 1.2 s instead of 0.6 s). Stage 6 advantages would mostly be p_succ changes × a larger gap.
      A smaller penalty only hides the problem. Cheap offline check if wanted: rescale existing
      predictions to the new target scale, no retraining. `c_fail_steps` is a label choice made with Lars
- **Mentor update (2026-10-20, for Lars)**, answering "from 10 s to the end looks reasonable, before
  that it seems random": he is right, for two reasons. (1) The model picks between two outcomes and
  wrong picks (at the holder) cause ~60% of error; the end looks good because insertion is visible.
  (2) Frame-to-frame jitter in the hump balance (~97% of V's jumps). Augmentation didn't reduce (2);
  0.5 s smoothing removes 73% for free. Next: proprio, short history / elapsed time, inference
  smoothing, maybe calibration, crop decision, then the rest of Stage 5. Suggested figures:
  `09_worst_error_r04-eval_ep0101_dist.png`, `10_worst_success_r01-eval_ep0103_dist.png`
- **RECAP (π*0.6, arXiv 2511.14759v2) vs ours: value-function training only (2026-10-20).** Full paper text:
  `data/refs/pi06_2511.14759v2.txt` (read §IV-A, §V-C, §V-D, Alg. 1, App. A-B…A-F). The paper has **no
  value-function hyperparameter table** and **no quantitative value metrics**. **No official code**:
  openpi has no value/RECAP code; issues #857, #860, #925, #791 unanswered. Community reimplementations:
  lerobot PR #3245 (WIP, unvalidated: PaliGemma 2B cut to 10 layers, mostly frozen; c_fail 500; 50 bins,
  hard one-hot via `bucketize`, CE; per-task max-length normalization clamped to [−1, 0]; lr 1e-4, batch
  4 × 4 accumulation), exla-ai/pi-openpi (SigLIP + "gemma_670m" + MLP head; 201 bins over raw steps 0–200,
  one-hot CE, no C_fail, no normalization), small forks (hzm8341/pi0.6, wpjNB/pi0.6, liudi159/pi05-recap;
  not inspected). None reproduces the web-data co-training or a 670M Gemma 3 value model.
  **Same as ours (targets already match):**
  - Reward −1/step, −C_fail at the end of a failure, success terminal 0 (Eq. 5); undiscounted Monte Carlo
    return = value of the behavior policy (§III, §IV-A Eq. 1)
  - Normalized to (−1, 0) per task by max episode length (§V-C); how C_fail fits inside is not specified
    (ours: divide by horizon 900 + c_fail 900). C_fail value not given ("a large constant")
  - Timeouts: success is defined with time limits (§VI-A), so running out of time = failure (inference)
  - Human interventions: whole episodes (autonomous parts + corrections) go into the data, nothing special
    in the value targets (§V-D)
  - 201 bins, cross-entropy, V = expectation over bins (§IV-A; distributional ref is C51 [72]). Target
    encoding (one-hot / two-hot / HL-Gauss) **not specified**; HL-Gauss is not cited
  - No history or timestep input; they assume o_t is Markovian (§III). **RECAP has the same stall blind
    spot.** Fig. 13 (App. A-B) shows "oscillating values" as its only acknowledged failure mode, the same
    thing the mentor flagged
  **Different from ours:**
  - Backbone: 670M VLM initialized from Gemma 3, "same architecture as the VLA policy" (§IV-A, §V-C);
    frozen/fine-tuned parts not specified. Ours: ImageNet ResNet-18, fine-tuned
  - Overfitting control: "co-train the value function on a small mixture of multi-modal web data" (§V-C).
    Ours: low LR only (overfit from ~0.09 passes)
  - Inputs: images (3 cameras on their robot) + robot state q (inference: shares the VLA's inputs) + the
    same language input as the VLA, task prompt + metadata (§V-A, §V-C). Ours: 2 images only
  - Training data and stages: V is first pre-trained multi-task on the demonstrations ("tens of thousands
    of hours", §IV step 2), then fine-tuned "on all of the data collected for the task so far" (§V-D).
    Ours: one stage, ~375k frames of one task, no value pre-training beyond ImageNet
  - Retrained from the pre-trained V each iteration "to avoid drift" (Alg. 1 line 8, §V-D). Ours: one run
  - Not specified in the paper (so ours can't be checked against it): optimizer, LR, batch size, steps,
    schedule, early stopping / checkpoint selection, validation split, web-data mixture ratio. Ours:
    AdamW, backbone 1e-5 / head 1e-4, batch 32, 8k steps cosine, checkpoint = lowest val CE
- [ ] **Next experiments, ranked (re-ordered by Karen 2026-10-20; each vs the 3-seed `lowlr_8k`
      baseline, one change at a time).** Replaces the old "Later" list (short history; categorical vs
      scalar head; other ways of combining the two images; larger backbone only if ResNet-18 underfits).
      V stays memoryless (single frame, like RECAP) for now.
      - No more training data: the 30 other repos in the HF collection are views of the 13 we use
        (checked 2026-10-20). `c0N-dagger-{baseline,mulligan,mulligan-no-cf}` are "training-view"
        repos whose parent is `c0N-dagger-mixed` (c01: the three views' `source_episode_index` cover
        112 of the 115 mixed episodes); per-policy `r0N-eval-b01-*-policy-rollouts` split up `r0N-eval`
        (r03, r04: frames sum exactly); `c00-teleop-{baseline,sobol,validation}` sum exactly to
        c00-mixed. Only r05 is unused, and it is the test set
      - [x] 1. **Proprioception** (= Stage 5 item 2; done, no gain, see item 2 above; RECAP's input set, inferred). Exact gripper/marker pose
            at the holder may separate "about to insert" from "stuck". First recheck video ↔ proprio timing
            (0 vs +6 frames, Stage 3 note)
      - [ ] 2. **Stronger pretrained backbone fine-tuned at low LR**: RECAP's biggest difference (670M VLM).
            Frozen ResNet (MAE 0.234 vs fine-tuned 0.121) says the image features matter. Not a bigger
            head: the model overfits rather than underfits. First arm: **DINOv2 ViT-S/14 with registers**
            (~21M params; strong fine-grained spatial features for "marker tip under the hole"; same
            ImageNet mean/std; LayerNorm, so the shared-backbone BatchNorm mixing goes away). Second arm
            if it helps: SigLIP ViT-B/16 (closest to RECAP's VLM encoder; needs mean/std 0.5). Judge on
            within-round AUC and hold-phase success-vs-stall AUC, not only MAE (a gain could be better
            round recognition)
            - [x] Set up (2026-10-20): `--backbone dinov2_vits14_reg` in `vl.train_baseline` (recorded as
                  `backbone` in the run's config.json; `vl.predict` / `vl.value_dist` rebuild it from
                  there, older runs default to resnet18). timm 1.0.20 `vit_small_patch14_reg4_dinov2`,
                  weights `timm/vit_small_patch14_reg4_dinov2.lvd142m` pinned to commit `c04b519` (loaded
                  from the file directly: timm's hub check fails with huggingface_hub 2.2). Readout per
                  camera [CLS, mean patch] = 768 (DINOv2 linear-eval readout) [ours]; head 1536 → 256 → 201.
                  22.1M params (ResNet arm 11.5M). Tests (`tests/test_model.py`, +6, 69 pass): shape, one
                  shared ViT, no BatchNorm, each camera matters, readout = CLS + mean patch, the pinned
                  file's weights are really loaded (pos embed resampled 37² → 16²), run_data_config.
                  Real batch at init: CE 5.354 (ResNet 5.380), V ≈ −0.50. CPU smoke: trains, evals, reloads
            - Fair comparison vs `lowlr_8k` / `_s1` / `_s2`, checked: same frame index (val 64,906 frames,
              bins identical, values within 1.1e-16; train 373,947), split, crops, 224², no augmentation,
              `baseline_lowlr` section (8k steps, batch 32, warmup 500 + cosine, AdamW wd 0.01, backbone
              1e-5 / head 1e-4, bf16, eval every 500 on full val, best.pt = lowest val CE), seeds 0/1/2.
              The ResNet code path for this config is unchanged since `8ada382` (later changes only add
              opt-in aug/proprio/freeze). Differences: the treatment (pretraining + architecture), 2×
              parameters, GPU (RTX PRO 6000 vs L40S; only nondeterminism), and **the recipe was tuned on
              ResNet** (the LR was chosen to stop ResNet overfitting), so it is not tuned for the ViT:
              a loss would not rule out DINOv2 at another LR. Code ran uncommitted on top of `4ff53f0`;
              exact diff in `data/runs/dinov2s_code_vs_4ff53f0.diff`
            - [x] Run: job 17998154 (`NAME=dinov2s EXTRA="--backbone dinov2_vits14_reg" DIST=1 sbatch
                  --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` -> `dinov2s_s{0,1,2}`, each followed by
                  `vl.value_dist`), move5 RTX PRO 6000, 45 min for all 3 seeds. **Result (from
                  `result.json`, 3 seeds): no gain with the ResNet recipe.** Val MAE 0.122 ± 0.003 (ResNet
                  0.121 ± 0.004), val CE 3.795 ± 0.030 (3.730), eval-repo MAE 0.195 (0.196), shape 0.050
                  (0.047), AUC 1/5/10 s 0.61/0.62/0.81 (0.67/0.64/0.84). **Overfits earlier:** best step
                  1,500 on all seeds (ResNet 2–3k); val CE passes the constant (4.53) by ~4.5k and ends at
                  4.76–4.96; train loss 2.54 at 8k (ResNet 3.23). Not yet conclusive about DINOv2 (recipe
                  untuned for it; next lever: lower backbone LR, as Decision 2(b) did for ResNet)
            - [x] LR arm (Karen, 2026-10-20): backbone LR 3e-6, everything else identical (section
                  `baseline_lowlr_bb3e6`; differs from `baseline_lowlr` only in `lr_backbone`). Job
                  17999270 (`NAME=dinov2s_bb3e6 SECTION=baseline_lowlr_bb3e6 EXTRA="--backbone
                  dinov2_vits14_reg" DIST=1 sbatch --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` ->
                  `dinov2s_bb3e6_s{0,1,2}`); code diff `data/runs/dinov2s_bb3e6_code_vs_4ff53f0.diff`. Success =
                  best val CE below ResNet's 3.730 and a later best step, with AUC/MAE at least equal. If it
                  still peaks early with no gain: layer-wise LR decay before ruling DINOv2 out.
                  **Result: fails on seeds 0–1; seed 2 died** (home quota full while saving its step-500
                  checkpoint; `last.pt` truncated). s0/s1: best step 1,500 / 2,000, val CE 3.822 / 3.786,
                  MAE 0.128 / 0.123, AUC 1/5/10 s 0.60/0.56/0.81 and 0.54/0.61/0.79 (ResNet s0/s1 3.727 /
                  3.732, 0.117 / 0.123). Overfits more slowly (val CE at 8k 4.29–4.46 vs 4.76–4.96) but
                  still peaks early. Seed 2 not rerun (Karen: go to layer decay instead)
            - [x] Layer-decay arm (Karen, 2026-10-20), set up: section `baseline_lowlr_llrd` = `baseline_lowlr`
                  + `layer_decay: 0.75` (only difference, checked). `vl.model.layer_decay_groups`: final
                  norm 1e-5, block 12 7.5e-6, ... block 1 3.2e-7, embeddings 2.4e-7; head 1e-4 as before
                  [ours]. Test: every backbone param in exactly one group, LRs as above, ResNet raises
                  (70 pass, run with a scratch copy of Python 3.11.16 because the venv's interpreter
                  was emptied, see Stage 0). Submitted 2026-10-20 after Karen freed space (all finished
                  runs' `last.pt` deleted; best.pt/predictions kept) and reinstalled Python: job 18001572,
                  git `2df9f76` (clean), `NAME=dinov2s_llrd SECTION=baseline_lowlr_llrd EXTRA="--backbone
                  dinov2_vits14_reg" DIST=1 sbatch --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` ->
                  `dinov2s_llrd_s{0,1,2}`. Same success rule as the 3e-6 arm
                  **Result (2026-10-21, `result.json`, 3 seeds): fails every part of the rule.** Val CE
                  3.783 ± 0.025 (ResNet 3.729 ± 0.002), val MAE 0.130 ± 0.002 (0.121 ± 0.004), eval-repo
                  MAE 0.201 ± 0.010 (0.196 ± 0.008), AUC 1/5/10 s 0.61/0.62/0.77 (0.67/0.64/0.84). Best
                  step 2,000 on all seeds (ResNet 2–3k). Overfits slowest of the DINOv2 arms (val CE at 8k
                  4.01–4.03 vs 4.29–4.46 at 3e-6 and 4.76–4.96 with the ResNet recipe) but still peaks
                  early and is worse at the peak. **DINOv2 ViT-S/14 ruled out** for this data: three
                  fine-tuned recipes (all `freeze_backbone: false`) never beat ResNet-18; lowering the
                  backbone LR only slows the overfitting
            - [ ] Still to do: `vl.value_plots --run dinov2s_s0 --episodes-from lowlr_8k --dist`
                  (same 11 episodes as `reports/figures/stage4/lowlr_8k/`; the episode rule is
                  run-dependent without `--episodes-from`), `vl.flicker`, hold-phase AUC
      - [ ] 3. **Condition V on the collecting policy** [ours; NOT RECAP: RECAP's V is the value of the
            mixed behavior data, and neither the paper nor the π0.6 model card lists metadata fields;
            π0.7's are speed/quality/mistake/control mode, none identify the policy]. Mainly a control for
            2: does a gain come from guessing the policy? Label = `producer_model_id` from the view
            sidecars (round alone mixes baseline-dp / mulligan-dp / mulligan-idql in r03–r04),
            `human` for teleop. Small learned embedding concatenated to the image features before
            the head; replace the label with `unknown` on ~20% of training samples, since r05 is an
            unseen policy. Report val with true label and with `unknown`; must beat repo + time constant
            (0.146). Expected gain small: a per-policy offset cancels in Stage 6's V(o_{t+n}) − V(o_t),
            and the main error (holder ambiguity) is within-policy
      - [ ] 4. **Target change, ask Lars now** (runs when he answers): count soft-truncated timeouts down
            to the 800-step cap instead of the reviewer's cut point (matches RECAP's fixed time limits,
            removes arbitrary endpoints). Changing C_fail won't help (see above); normalization already
            matches. Score on both old and new targets
      - [ ] 5. **Cheap substitute for web co-training:** HL-Gauss soft targets (label change: needs Karen's
            OK) or an auxiliary task-phase head. Web co-training itself probably not worth it at this scale
      - [ ] Deferred (Karen, 2026-10-20): **short frame history (2–4 frames over 1–2 s) or elapsed time**
            [ours, beyond RECAP]. Would target the stall ambiguity and the time-blind failure hump.
            Caveat: most timeouts end at a reviewer-chosen soft-truncation frame, so elapsed time does not
            give time left there
            - [x] **Frozen-feature probe (Karen, 2026-10-22, 1 h budget)** [ours]. `vl.history_feats` cached
                  the `aug_split_cropwrist_s0` backbone features (2 × 512) every 6th frame (2.5 Hz; 73,844
                  train+val frames; no GPU free on `move`, ran on the move2 interactive Titan RTX) to
                  `/move/u/karenvo/vl-cache/history_local/`. `vl.history_probe` trains only the head (same
                  1024·k → 256 → 201, CE, AdamW 1e-3, 20 epochs, best val-CE epoch, 3 head seeds) on
                  [current, past] features; first frame repeated near the start. Results in
                  `probe_{main,long,first}.csv` there. Val at 2.5 Hz (10,925 frames; hold phase 45 vs 45):
                  | arm | val CE | MAE | succ MAE | timeout MAE | hold AUC |
                  |---|---|---|---|---|---|
                  | checkpoint's own head | 3.641 | 0.112 | 0.088 | 0.166 | 0.805 |
                  | ctrl (current only) | 3.738 ± 0.004 | 0.110 | 0.084 | 0.163 ± 0.003 | 0.808 ± 0.009 |
                  | elapsed time | 3.740 | 0.110 | 0.084 | 0.163 | 0.810 |
                  | past 0.4 / 1.2 / 2 s | 3.777 / 3.774 / 3.767 | 0.110 / 0.110 / 0.109 | 0.085–0.086 | 0.163 / 0.160 / 0.158 | 0.80–0.81 |
                  | past 4 s | 3.758 | 0.108 | 0.086 | 0.155 | 0.819 |
                  | past 8 s | 3.733 ± 0.005 | 0.108 | 0.087 | **0.152** ± 0.002 | **0.829** |
                  | past 16 s | 3.717 ± 0.007 | 0.108 | 0.085 | 0.157 | 0.830 |
                  | past 24 s | **3.707** ± 0.003 | 0.109 | 0.085 | 0.159 | 0.827 |
                  | first frame of episode | 3.731 ± 0.002 | 0.109 | 0.084 | 0.161 | 0.821 |
                  | 4 frames over 0.4–8 s | 3.79–3.83 | 0.109–0.110 | 0.086–0.089 | 0.151–0.160 | 0.80–0.81 |
                  Reading: short spans (≤ 2 s) and extra frames within a span don't help (4-frame heads overfit:
                  CE worse). Long spans help: past 24 s passes the pre-set rule (CE −0.031 ≈ 8 sd of head-seed
                  noise, hold AUC +0.02, success MAE unchanged); past 8 s gives the best timeout MAE (−0.011)
                  but costs +0.003 success MAE. **Confound:** median val episode is 15 s, so a ≥ 8 s offset
                  clamps to the first frame on 44–90% of frames; the first-frame arm alone takes ~⅓ of the CE
                  gain and ~⅔ of the AUC gain (start-configuration information, starts are not shared across
                  train/val). past 24 s vs first differ only on the 10% of frames > 24 s in (long stalls), so
                  the remaining CE gain is from stall frames. Head-seed sd ignores episode sampling (90 hold
                  episodes). Proprio CPU check (`aug_split_cropwrist_s0` val predictions, 15 Hz, 5-fold by
                  start): hold-phase frame AUC p_succ 0.763 → 0.776 adding "seconds since gripper speed
                  < 1 cm/s" (elapsed time alone: 0.777): small. Verdict: **weak positive, worth one more
                  cheap step**: mask clamped frames (zero + "no past" flag) to split history from start
                  frame, then a 2-frame end-to-end run at an 8–16 s gap
            - [x] **Masked check (2026-10-22)**: `masked_<s>` = past features zeroed where the frame s back
                  doesn't exist + a "no past" flag (no first-frame fallback); `flag_<s>` = flag only.
                  Results `probe_masked.csv`. ctrl 3.738 / succ 0.084 / timeout 0.163 / AUC 0.808:
                  | arm | val CE | succ MAE | timeout MAE | hold AUC |
                  |---|---|---|---|---|
                  | flag 8 s / 24 s | 3.740 / 3.740 | 0.084 | 0.162 / 0.164 | 0.810 / 0.808 |
                  | masked 4 s | 3.754 ± 0.004 | 0.087 | 0.153 | 0.817 |
                  | masked 8 s | 3.739 ± 0.009 | 0.088 | 0.153 | 0.814 |
                  | masked 16 s | 3.719 ± 0.008 | 0.086 | 0.160 | 0.818 |
                  | masked 24 s | **3.711** ± 0.004 | 0.085 | 0.160 | 0.820 |
                  The flag carries nothing. **The CE gain survives masking** (24 s: −0.027 ≈ 7 sd; unmasked
                  −0.031), and masked 24 s has a real past frame on only the ~10% of val frames > 24 s in,
                  so the gain is on long stalls (inferred: per-frame CE not saved). The **hold-AUC gain was
                  mostly the first frame** (0.827 → 0.820, +0.012 over ctrl ≈ 1.3 sd). 4–8 s masked: timeout
                  MAE −0.010 but success MAE +0.003–0.004, no CE gain. Against the pre-set rule: CE passes;
                  AUC/timeout gains small; success MAE +0.001 (within ~1.5 sd). **Verdict: real but narrow:**
                  memory helps on long stalls only. Next: 2-frame end-to-end run at a 16–24 s gap (overnight)
            - [x] **Figures (2026-10-22)**: `vl.history_probe --save-preds` (per-frame val V/p_succ/CE, mean of 3
                  head seeds, `..._stride6_preds.parquet`) + `vl.history_plots` -> `reports/figures/history/`
                  (1 p_succ vs seconds gripper still, 2 per-episode scatter, 3 traces, 4 error by time; 95%
                  episode-bootstrap CIs). ctrl vs masked 24 s / 8 s. **The memory effect is mostly invisible
                  in MAE/p_succ:** traces overlap; hold-phase stalls already sit at p_succ ≈ 0.2 under ctrl
                  and only drop at > 32 s still (0.14 → 0.08, n = 4 episodes). Per episode, masked 24 s lowers
                  hold-phase p_succ for 64% of stalls but also 31% of successes, all by small amounts.
                  Frames > 24 s in: stall error 0.098 → 0.096 (24 s), 0.081 (8 s), but success error 0.109 →
                  0.121 / 0.140 (n = 14, CIs overlap). So the CE gain is distribution sharpening, not a visible
                  change in V. Worst stalls r02/45 (16 s) and r01/48 (20 s) never get a real past frame at 24 s
      - [ ] Also: 0.5 s centred smoothing as the default at inference; optional p_succ calibration
            (temperature on the success/failure split) for the overconfident timeouts; crop decision
            (Step 2 above)
            **Computed on R0 2026-10-26** (`data/runs/inference_choices.py` → `reports/inference_choices.md`,
            CPU; same 3 models post-processed, paired episode bootstrap). Ensemble ×3: CE −0.030
            [−0.033, −0.027]; + centred 0.5 s: −0.042 [−0.047, −0.036] (3.630), flicker −77%, shape
            0.044 vs 0.049, stall_auc +0.011; temperature ~0 on top and worse ECE; causal 0.5 s (live)
            ensemble 3.656. AUC 1 s unchanged, shape still above flat (0.042 vs 0.0365). **Proposed
            (awaiting Karen):** offline = ensemble ×3 + centred 0.5 s, no T; live = ensemble ×3 + causal 0.5 s
      Order: 1 → 2 → 3, with 4 inserted when Lars answers. If a stronger backbone wins, make it the
      new baseline and re-test proprio on top of it
- [ ] Optional diagnostic: timeouts excluded (baseline is timeout = failure)
- [x] Not needed (Stage 4: V does not react to the operator). Was: if Stage 4 shows V reacting to the operator: masked `wrist_left` bottom-left corner vs
      unmasked (see Stage 4 operator check)
- [ ] Freeze all choices (after the Stage 5.5 merge gate)
- [ ] Final test evaluation (r05-eval, r05-screen reported separately; check r05-screen
      provenance before treating it as equivalent; they share the same 50 starts)
- [ ] Progress package for Lars: what was built (architecture, inputs, targets, split), metrics
      + 3 representative plots/videos, failure examples, one comparison, next decision; with
      config, pinned revisions and checkpoint

## Stage 5.5 — Meeting follow-ups (Lars, 2026-10-23), before policy improvement
Source: Karen's notes from the 2026-10-23 meeting with Lars (transcript:
https://notes.granola.ai/t/11106840-5292-42c7-ab27-01d614f49fde). Inserted before Stage 5's
"Freeze all choices" / final test evaluation, which now wait for the 5.5 merge gate.

**How this stage runs.** Gate 0 comes first and is blocking: the crop/resize choice is the input
to every run after it, so changing it later would mean re-running everything. After Gate 0,
tracks A, B and C are independent and can run in parallel (different jobs, different
`NAME=` prefixes, one change at a time against the Gate 0 baseline). Each track ends in a
**You check** and a **Gate: Karen confirms** line; nothing that depends on a track starts
before its gate is ticked. Track A (pure visualization) can start alongside Gate 0.
Reference to beat until Gate 0 picks a new one: `aug_split_cropwrist_s{0,1,2}` (crop jitter on
wrist, full aug on side; val CE 3.647 ± 0.008, MAE 0.112), with `lowlr_8k` as the no-aug anchor.

### 5.5.0 Storage (done)
- [x] `data/` → `/iris/u/ankile/karenvo/vl-data` (2026-10-23; Stage 0). Home quota back to 9%

### 5.5.1 Gate 0 — crop alignment and resize (blocking; Karen, 2026-10-23)
Why: the crops (`configs/data.yaml`) were chosen by eye in Stage 3 on 640×480 frames; nobody has
checked, from the 224×224 tensors back, which pixels of each camera the model actually gets, or
whether the boxes sit right relative to each camera's view. Lars also raised other ways to
resize/scale the original frame.
- [ ] Back-projection sheet [ours]: for both cameras, the 224×224 model input (un-normalized,
      from the real loader path) next to the raw 640×480 frame with the box drawn on it, at
      approach / grasp / insertion / seat / stall / failure end, plus the initial-state extremes.
      One row per repo (r00–r04, c00–c02) to catch camera moves between rounds. Also the
      box under the A2 crop-jitter extremes (what augmentation does to the box)
      **Done 2026-10-23** (`vl.input_sheets gate0` → `reports/figures/stage5_5/gate0/
      backprojection_{side_1,wrist_left}.jpg`): 11 train repos × (success start / approach /
      insertion / seated T + a timeout stall at 3/4, else a failure at T, else success at 0.6 for
      the all-success c00 / c04), raw frame with box (cyan) + A2 extremes (yellow) next to the
      loader's 224 input
- [ ] Alignment numbers per repo and camera: where the gripper / holder / marker sit inside the
      224 input (pixel position and size; marker is ~5×2 px in `side_1` at 224), how much of the
      input is empty table, distance of the box centre from the region of interest. Check the
      resize (bilinear; antialias on or off) and that no aspect distortion happens
      **Done 2026-10-23** (`reports/figures/stage5_5/gate0/alignment.md`, `maps_{cam}.jpg`; 6 train
      episodes × 20 frame pairs per repo). Marker/gripper positions are not detected (no labels);
      motion maps stand in for the region of interest [ours]. Findings:
      - **No camera moved between rounds**: phase-correlation shift of every repo's mean frame onto
        the pooled mean is (0, 0) px for both cameras, all 11 repos. One box per camera is valid
      - **`side_1`: the scene sits in the top of the box.** 69–75% of the motion falls inside a box
        that is 57% of the frame (so motion is concentrated), but the motion centroid is 50–75 px
        above and 6–43 px left of the box centre (2 repos +10/+14 right); the box already starts at
        y = 0, so it cannot move up. The bottom ~40% of the input is empty table at every phase.
        The arm enters from the top and the approach in c01/c02/r03 passes the box's left edge
      - **`wrist_left`: motion is spread over the whole frame** (70–73% inside a box that is 75% of
        it; the camera moves with the gripper). Motion centroid 12–26 px above, 5–31 px left of centre.
        **Gripper sits right of centre** (Karen's observation, measured 2026-10-23: dark + static
        pixels, 4 eps × 12 frames per repo): gripper centroid x ≈ 430 of 640 (c01 363 – r00 441),
        12–21% of gripper pixels at x > 560 (cut by the current box, 0% in c01), 0–1% at x < 80.
        Same place in every repo (mount unchanged). But the left strip isn't blank: 29.5% of the
        wrist motion is at x < 160 (dropped by a right-flush box) vs 12.6% at x > 560. Box comparison
        `reports/figures/stage5_5/box/box_compare_wrist_left.jpg` (current vs (160, 0, 640, 480)):
        the shifted box shows the right finger, loses the yellow table edge and part of the holder
        when it is at the upper left during the approach (c01/c02). Trade-off → test as an arm (R1) Holder
        reaches the top edge in some insertion/stall frames (full-height box: only a whole-frame
        arm, R2/R3, adds anything vertically); the operator shows at the left edge in r00
      - Resize: `PIL.Image.resize(BILINEAR)` filters over the source footprint when downscaling
        (antialiased); boxes are square, so no aspect distortion. Every repo is the same 640×480
- **You check:** the sheet and the numbers; draw corrected boxes where the current ones are off
- [ ] Resize/scale arms, 3 seeds each (seed 0 first as a screen). **Recipe (Karen, 2026-10-23): full
      aug only (`baseline_lowlr_aug_full`, Lars's LeRobot set on both cameras), no crop jitter**, so
      every arm gets the same augmentation after its own resize and squash/letterbox need no jitter
      rule. Reference for this gate = `aug_full_s{0,1,2}` (val CE 3.670 ± 0.009, MAE 0.113), re-run as R0
      on the same code and GPU type:
      | arm | what goes in |
      |---|---|
      | R0 | current boxes → 224 (= reference) |
      | R1 | realigned boxes from the check above → 224 |
      | R2 | whole frame squashed 640×480 → 224×224 (nothing cut, aspect distorted) |
      | R3 | whole frame letterboxed (pad to 640×640 → 224; nothing cut, no distortion) |
      | R4 | tighter `side_1` box around the scene (bigger marker), `wrist_left` as R1 |
      | R5 | best of R1–R4 at a higher input size (e.g. 288 or 320) if the marker is still tiny |
      Judge on val CE, MAE, shape MAE, eval-repo MAE, AUC 1/5/10 s; crop jitter boxes are
      recomputed for each arm's base box
- [x] Code (2026-10-23) [ours]: `resize: {cam: crop|squash|letterbox}` in the data config
      (`vl.dataset.squash_resize` / `letterbox_resize`; letterbox = 224×168 + black bars, as openpi's
      `resize_with_pad`; crop jitter refused for squash/letterbox). `vl.train_baseline --crop CAM=box
      --resize CAM=mode`, recorded as `crops` / `resize` in the run's config.json and read back by
      `run_data_config`; runs without them get `STAGE3_CROPS` (the only crops data.yaml ever had), so
      a later data.yaml change can't silently re-crop old runs. Default path checked identical on 60
      real frames × 2 cameras with seeded full aug; +5 tests (83 pass); CPU smoke: squash trains and
      evaluates, letterbox/crop configs reload. Diff `data/runs/gate0_code_vs_2df9f76.diff`
- [x] Contact sheet of each arm's actual inputs: `vl.input_sheets arms` →
      `reports/figures/stage5_5/gate0/arms.jpg` (r04 approach/insertion, c01 approach, r01 stall)
- [x] **Runs submitted 2026-10-23** (`SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1 sbatch
      --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch`, 3 seeds each → `g0_<arm>_s{0,1,2}`). Arms as
      run (Karen, 2026-10-23, from Lars's transcript): R0 current (job 18058482); R1 wrist flush right
      (160, 0, 640, 480) (18058485); R1b wrist half shift (120, 0, 600, 480) (18058486); R2 squash both
      (18058487); R3 letterbox both (18058488); R4 side full-height box (80, 0, 560, 480), wrist
      unchanged, i.e. one change vs R0 (18058489). The tighter side box and higher resolution (old
      R4/R5) wait for these results. All 6 jobs COMPLETED (47–61 min each)
- [x] **Results (2026-10-23)**, mean ± seed sd over 3 seeds, best checkpoint per seed. Rule set before
      the runs [ours]: win = val CE below R0 by > 2 R0 sd (0.020) **and** shape MAE and AUC 1 s no
      worse. **No arm wins; R0 (current crops) stays.** R0 reproduces `aug_full` (3.670 ± 0.009).

      | arm | val CE | MAE | shape MAE | eval-repo MAE | AUC 1 s | AUC 5 s | AUC 10 s |
      |---|---|---|---|---|---|---|---|
      | R0 current | 3.672 ± 0.010 | 0.1136 | 0.0487 | 0.1826 | 0.621 ± 0.017 | 0.610 | 0.821 |
      | R1 wrist flush right | 3.664 ± 0.006 | 0.1128 | 0.0489 | 0.1813 | 0.627 ± 0.023 | 0.617 | 0.819 |
      | R1b wrist half shift | 3.669 ± 0.009 | 0.1123 | 0.0491 | 0.1805 | 0.618 ± 0.005 | 0.617 | 0.824 |
      | R2 squash both | 3.677 ± 0.008 | 0.1161 | 0.0470 | 0.1887 | 0.522 ± 0.033 | 0.657 | 0.815 |
      | R3 letterbox both | 3.658 ± 0.004 | 0.1174 | 0.0475 | 0.1911 | 0.537 ± 0.028 | 0.619 | 0.811 |
      | R4 side full height | 3.669 ± 0.009 | 0.1138 | 0.0476 | 0.1842 | 0.585 ± 0.018 | 0.659 | 0.825 |

      Reading: shifting the wrist box toward the gripper changes nothing beyond seed noise (the
      cut-off gripper pixels weren't costing anything measurable). Showing the whole frame (R2/R3, and
      R4 for the side camera) trades the same way each time: shape MAE a little better (−0.001 to
      −0.002), AUC 1 s clearly worse (−0.04 to −0.10, several sd), MAE / eval-repo MAE worse. R3 has the
      lowest CE (−0.014, short of the 0.020 bar) but the worst MAE. Likely cause: the gripper/holder
      gets fewer pixels when the whole frame is shrunk to 224, hurting short-range timing.
      `vl.input_sheets` unchanged; table: `.venv/bin/python scripts/gate0_table.py`
- **You check:** results table and the arm sheets
- [x] **Gate 0: Karen confirms the crop/resize arm** → written into `configs/data.yaml` as the
      new baseline input for tracks B and C (and Track A is regenerated on it)
      **Confirmed 2026-10-24: R0 (current boxes, crop → 224).** The `crops` values were already R0,
      so only a provenance comment changed in `configs/data.yaml`; Track A needs no regeneration
- [x] **R5 tighter `side_1` box** (Karen, 2026-10-24; the old R4 idea, held until the first 6 arms
      finished): `side_1` (160, 0, 520, 360), `wrist_left` unchanged, so one change vs R0. Box choice
      [ours], from all 11 train repos (8 eps × 12 frames each, scratch measurement): the red holder
      sits at x 356–453, y 155–204 (0.5–99.5%), so a 360 px box with its top at y = 0 still
      contains it, at 1.17× more pixels per object than the 420 px box. It keeps 72% of the
      frame-difference motion (R0: 78%) and 100% of the holder pixels, and drops the bottom of
      the table, where markers start in some episodes. The blue-pixel share falls 0.98 → 0.89,
      but that mask also catches the table's blue dots. Visual check
      `reports/figures/stage5_5/box/box_compare_side_1.jpg` (c01, c02, r01, r04: marker, holder
      and gripper in view at start / approach / insertion / end). Job **18065359** (`NAME=g0_r5
      SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1 EXTRA="--crop side_1=160,0,520,360"
      sbatch --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` → `g0_r5_s{0,1,2}`). Same win rule as
      the first 6 arms (val CE below R0 by > 0.020, shape MAE and AUC 1 s no worse)
- [x] **R6 higher input size** (Karen, 2026-10-24; the old R5): R0 boxes at **320 × 320** instead of
      224 (no arm beat R0, so the size change goes on R0). The choice of 320 is ours: it is the
      largest of the planned 288 / 320 that still downsamples both boxes (420 and 480 px), with 2.04×
      the pixels. The marker is ~5×2 px in `side_1` at 224, ~7×3 at 320. Code [ours]:
      `vl.train_baseline --image-size N` is recorded as `image_size` in config.json and read back by
      `run_data_config` (runs without it are 224). ResNet-18 pools adaptively, so the model needs no
      other change. +1 test (84 pass). CPU smoke at 320 trains, evaluates and reloads (3×320×320
      per camera). Diff `data/runs/gate0_r5r6_code_vs_2df9f76.diff`. Job **18065701** (`NAME=g0_r6
      SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1 EXTRA="--image-size 320" sbatch
      --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` → `g0_r6_s{0,1,2}`). Same win rule
- [x] **R5 / R6 results (2026-10-24; jobs completed 03:45 / 06:10, 65 / 76 min, no errors)**, 3 seeds,
      best checkpoint per seed, same table as above (`scripts/gate0_table.py`, now with R5/R6):

      | arm | val CE | MAE | shape MAE | eval-repo MAE | AUC 1 s | AUC 5 s | AUC 10 s |
      |---|---|---|---|---|---|---|---|
      | R0 current | 3.672 ± 0.010 | 0.1136 | 0.0487 | 0.1826 | 0.621 ± 0.017 | 0.610 | 0.821 |
      | R5 side 360 box | 3.671 ± 0.005 | 0.1135 | 0.0501 | 0.1825 | 0.611 ± 0.005 | 0.618 | 0.842 |
      | R6 320 px input | **3.648 ± 0.005** | 0.1127 | 0.0489 | 0.1811 | 0.584 ± 0.015 | 0.646 | 0.851 |

      **Rule verdict: neither wins; R0 stays.** R5 changes nothing (CE −0.001; shape MAE +0.0014, ~1 sd
      worse): a bigger side-view scene buys nothing. R6 is the first Gate 0 arm to clear the CE bar
      (−0.023 vs the 0.020 needed; every seed 3.644–3.653 below every R0 seed 3.663–3.682), shape MAE
      ties (+0.0002), MAE and eval-repo MAE slightly better, AUC 5/10 s better (+0.04 / +0.03). But it
      **fails on AUC 1 s (−0.037, ~2 R0 sd)**, the same short-range drop as R4 (−0.036) and the
      whole-frame arms. So more pixels give a real CE gain, and the 1 s ranking cost seen on every
      arm that changes the side input is not explained by object size. Best steps 6.5k / 7.5k / 8k:
      s2 peaked at the last step
- [x] **R6 at 16k steps** (Karen, 2026-10-24): does R6 keep improving past 8k? Same section, cosine
      stretched to 16k (`--steps 16000`, eval every 500 as at 8k). Prior: R6's val CE is already flat
      from 6.5k to 8k (±0.005 per seed), and `crop16k` lost to the 8k crop run (3.685 vs 3.656) because
      the longer schedule overfits before the LR anneals. Job **18071606** (`NAME=g0_r6_16k
      SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1 EXTRA="--image-size 320 --steps 16000" sbatch
      --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` → `g0_r6_16k_s{0,1,2}`). Compare against R6 8k
      and R0 on the same metrics, including AUC 1 s
      **Result (completed 14:49, 2 h 39 min, 3 seeds): it does not keep improving; 16k overfits, as
      `crop16k` did. R6 stays at 8k.**

      | run | best step | val CE | MAE | shape MAE | eval-repo MAE | AUC 1 s | AUC 5 s | AUC 10 s |
      |---|---|---|---|---|---|---|---|---|
      | R0 (8k) | 4–7.5k | 3.672 ± 0.010 | 0.1136 | 0.0487 | 0.1826 | 0.621 ± 0.017 | 0.610 | 0.821 |
      | R6 8k | 6.5–8k | **3.648 ± 0.005** | 0.1127 | 0.0489 | 0.1811 | 0.584 ± 0.015 | 0.646 | 0.851 |
      | R6 16k | 4–5k | 3.678 ± 0.002 | 0.1134 | 0.0478 | 0.1824 | 0.605 ± 0.003 | 0.641 | 0.850 |

      Val CE bottoms at 4–5k (3.676–3.695, while the LR is still high) and climbs to 3.78–3.80 by 16k;
      final train CE 3.28–3.36 vs 3.55–3.63 at 8k. The 8k run's later, lower best comes from the cosine
      annealing to 0 at 8k, not from more steps (same reading as `crop16k`). 16k's best checkpoints are
      worse on CE than R0 (+0.007); AUC 1 s is less bad than R6 8k (−0.016 vs R0, ~1 sd), but those are
      earlier, less-trained checkpoints, so this does not show the 1 s cost is fixable by schedule

### 5.5.2 Track A — visualize images after augmentation (can start now)
- [x] Contact sheet per augmentation arm (photometric, crop jitter, full, split crop-wrist /
      full-side): one frame × N random draws, both cameras, exactly the tensors the model trains
      on (un-normalized from the loader, not a re-implementation), plus each arm's extremes.
      First check that the existing `reports/figures/stage5/augment_examples.jpg` and
      `full_aug_examples.jpg` came from the loader path; regenerate if not
      **Done 2026-10-23**: no script in `src/` or `scripts/` produced the old sheets, so they were
      regenerated: `vl.input_sheets aug` → `reports/figures/stage5_5/aug/aug_{success_approach,
      success_insertion,stall}.jpg` (r04/135 approach and insertion, r04/62 stall; 5 arms × 2
      cameras; column 1 = no augmentation, then 8 seeded draws through `FrameDataset(augment=...)`).
      Crop-jitter extremes are drawn on the Gate 0 back-projection sheet (yellow boxes)
- [x] Same sheet on stall / insertion frames, where the task detail is smallest. What they show
      (Claude, to be checked): marker, holder and gripper stay in view in every draw. Things to
      look at: hue 0.05 in the full block turns the orange holder yellow/pale on some draws (the
      holder's colour is one of the few cues in `side_1`); affine rotation adds black corners
      (`fill=0`, not the ImageNet mean that `crop_pad` uses); A2 on the full-height wrist box only
      moves sideways (clamped), as intended
- [x] Lars's LeRobot config at its extremes (Karen, 2026-10-23): `vl.input_sheets lerobot` →
      `reports/figures/stage5_5/lerobot/{singles,combos_wrist_left,combos_side_1}.jpg`, `hue.md`.
      Asserts `baseline_lowlr_aug_full` = Lars's settings (3 of 6, fixed order, brightness/contrast
      0.8–1.2, saturation 0.5–1.5, hue ±0.05, sharpness 0.5–1.5, rotate ±5°, translate ±5%) and that
      the fixed-value ops reproduce `full_augment` pixel for pixel. singles = each transform at
      low / none / high; combos = all 20 three-transform subsets at all-low and all-high.
      Hue: ±0.05 = ±18° on the colour wheel, applied to the whole image. Natural spread across 11 repos
      (insertion-phase frames): holder 12–24°, yellow table edge 41–50°. The shift moves the holder
      to 0–36° (outside its natural range by up to 12°, short of the edge's 41°); within one image
      holder and edge stay 26° apart
- **You check:** nothing task-critical is cut or washed out; each camera gets the augmentation it
  is supposed to; colours stay plausible
- [x] **Gate A: Karen confirms** the augmentation is doing what it should (regenerate after Gate 0)
      **Confirmed 2026-10-24** (Karen). Gate 0 kept R0, so the sheets needed no regeneration

### 5.5.3 Track B — history / memory (after Gate 0)
Lars's framing: history should help detect **stalling** (no progress), not how much progress a
success has made. Closer frames catch stalls sooner and overfit less (more correlated with the
current frame); farther frames are less correlated, give a bigger fitting space and generalize
worse. Status so far: frozen-feature probe (Stage 5, 2026-10-22): ≤ 2 s no gain, ≥ 16–24 s gain,
partly from the first-frame fallback; masking keeps a narrow gain on long stalls. Current
approach per the meeting: a past frame sampled at random 2–20 s back; not yet trusted (subset
of data).
- [x] B1. Side-by-side series [Lars]: current frame + 1, 2, 3, 4 steps back, at step sizes
      0.5 / 1 / 2 / 4 / 8 s, both cameras. Episodes: hold-phase stalls, hold-phase successes
      (just before seating), failures, early-episode frames (where the past doesn't exist yet).
      Goal: see at which gap a stall becomes visible as "no change" while progress is still visible
      in successes
      **Done 2026-10-24** (`.venv/bin/python -m vl.history_series` → `reports/figures/stage5_5/history/`,
      ~10 min on 4 CPUs). Train split, eval repos, the model's 224 px inputs (no augmentation).
      13 sheets `series_<kind>_<repo>_ep<k>.jpg`: rows Δ = 0.5 / 1 / 2 / 4 / 8 s, columns t − 4Δ … t,
      both cameras, |Δpix| to the current frame under each past frame, grey "before start" cells.
      Kinds [ours; hold phase as in `vl.history_probe`: > 5 s in, > 2 s before T]: stall (machine label
      `timeout_holding_marker`, first frame with the gripper still ≥ 10 s by proprio) r00/64 r01/24
      r02/138 r03/117; success hold (T − 2.5 s) r00/70 r01/82 r02/14 r03/63; failure at T r00/37
      r01/79 r02/225; early (t = 2 s) r00/70 r01/82. Machine labels only pick examples.
      Companion curve [ours] `pixel_change_vs_gap.{png,csv}`: mean |Δpix| (0–255) between t and t − Δ,
      40 episodes × 3 frames per kind, 95% bootstrap over episodes:

      | wrist_left / side_1 | 0.5 s | 1 s | 2 s | 4 s | 8 s | 16 s |
      |---|---|---|---|---|---|---|
      | success, hold phase | 12.8 / 7.1 | 17.3 / 9.5 | 24.7 / 12.7 | 33.4 / 16.1 | 36.6 / 13.5 | 31.7 / 15.4 |
      | failure, hold phase | 13.4 / 6.5 | 17.4 / 9.1 | 23.7 / 12.3 | 30.9 / 15.2 | 33.1 / 13.7 | 30.1 / 14.7 |
      | stall episode, any hold frame | 7.4 / 4.4 | 9.9 / 5.0 | 12.7 / 6.8 | 18.6 / 9.1 | 25.4 / 10.3 | 27.0 / 10.9 |
      | stall, gripper still ≥ 10 s | 1.7 / 1.2 | 1.9 / 1.6 | 2.4 / 1.3 | 3.3 / 1.5 | 4.9 / 1.9 | 20.4 / 10.6 |

      What it shows (Claude, to be checked on the sheets): (1) once a stall has set in, both cameras are
      frozen at every gap up to the stall's length (|Δpix| 1–5, ~sensor noise), while hold-phase
      successes change 7–13 already at 0.5 s and keep growing to 4–8 s. The "still ≥ 10 s" row is
      flat up to 8 s **by selection** (picked on proprio stillness), so it shows what a stall looks like,
      not where it can be detected. (2) The non-circular row (any hold frame of a stall episode) is
      lower than successes by ~1.7–1.9× at 0.5–4 s and only 1.2–1.4× at 8–16 s: in pixels, short
      gaps separate stalls from successes best, longer gaps catch up with earlier motion (the
      approach). (3) Failures look like successes in pixel change; history won't separate those
      by motion. (4) Early frames: no past beyond t; at 2 s every gap ≥ 1 s back has holes, so
      longer gaps need the mask for the first 4Δ seconds of every episode. Caveats: pixel change is
      a proxy (lighting flicker, the operator in r00, camera noise all count); it is not what the
      model will learn to use. It conflicts with the frozen-feature probe (≥ 16–24 s helped, ≤ 2 s
      didn't), which also had the first-frame confound. Candidate gaps to consider for B2:
      **1–2 s** (largest pixel contrast), **4 s** (still separated, sees a stall sooner than 8 s), and
      the current random 2–20 s as the reference arm
- [x] B1 follow-up: how long do successes sit still? (Karen, 2026-10-24). Question: a gap only
      flags a stall if hold-phase successes rarely stay still that long, so measure their pauses.
      `.venv/bin/python -m vl.success_pauses` → `reports/success_pauses.md`,
      `reports/figures/stage5_5/history/long_pause_successes.jpg` (~5 min on 4 CPUs). Same hold-phase
      rule and stall definition as B1 [ours]; "still" = gripper < 1 cm/s (also 0.5 / 2 cm/s, same
      picture). **Result: no clean cutoff; many successes freeze like stalls.**

      | still < 1 cm/s, hold phase | median longest pause | 90th pct | episodes with pause ≥ 4 s | ≥ 8 s |
      |---|---|---|---|---|
      | success (277 train episodes) | 1.9 s | 21 s | 38% | 26% |
      | stall (257) | 11.4 s | 38 s | 74% | 60% |

      Share of hold *frames* still ≥ gap, stall / success: 2 s 64% / 45%, 4 s 55% / 37%, 8 s 42% / 27%:
      ~1.4–1.6× at every gap, so no gap stands out. The long success pauses are real (30 episodes:
      |Δpix| over 4 s median 5.7 wrist / 2.4 side vs stalls 2.3 / 1.3; the strips show the marker
      frozen beside the holder for 20–40 s, then seated in the last ~2 s; almost all end exactly at
      the hold-window edge, T − 2 s). 26% of stall episodes never pause ≥ 4 s (moving without progress).
      Targets and Gate 0 predictions (val split, `g0_r0_s{0,1,2}` mean, hold phase):

      | | episodes | time to go | target | V | V < −0.5 |
      |---|---|---|---|---|---|
      | success, never paused ≥ 2 s | 22 | 5.5 s | −0.046 | −0.314 | 16% |
      | success, paused ≥ 8 s, frames inside the pause | 13 | 12.1 s | −0.101 | −0.283 | 10% |
      | stall, frames still ≥ 4 s | 32 | 14.6 s | −0.622 | −0.472 | 58% |

      Reading (Claude, to be checked): frozen successes keep success-range targets (lower only because
      more time is left), and the image-only baseline already scores them like other successes, not
      like stalls. History cannot separate a frozen success from a stall while both are frozen (any
      gap inside the freeze shows "no change"), so it could pull frozen successes toward stall values.
      For B2 the gap choice matters less than thought; keep 2 s / 4 s + random 2–20 s. B3 should report
      frozen successes (paused ≥ 8 s) as their own group to catch that harm. n = 13 frozen-success val
      episodes, one model
- **You check:** pick 2–3 candidate gaps (or a range) from the series
      **Decided 2026-10-24 (Karen): gaps 2 s and 4 s, compared against no memory (R0). The random
      2–20 s arm is dropped: too many issues with that implementation to be a useful reference**
- [ ] B2. Full-data memory run, end-to-end (not the frozen probe): 2-frame input on the Gate 0
      baseline, arms = no memory (R0, `g0_r0_s{0,1,2}`) vs fixed gap 2 s vs fixed gap 4 s
      [Karen, 2026-10-24; was: random 2–20 s vs the B1 gaps]; missing past handled by
      masking + flag (no first-frame fallback, the start-configuration confound), 3 seeds, all
      train frames. Same metrics as Stage 5 plus a stall-vs-success split of hold-phase frames
      **Code (2026-10-24)** [ours unless tagged]: `vl.train_baseline --history-gap S` → data config
      `history: {gap_s: S}`, recorded as `history` in config.json and read back by `run_data_config`
      (older runs: none). `FrameDataset` adds `past = {images: {cam}, has_past}`: frame
      t − round(S · 15) of the same episode, same crop and the *same* augmentation draw as frame t
      (torch RNG state replayed per camera, so an unmoving robot gives identical inputs; independent
      draws would fake motion). t − gap < 0: has_past = 0, placeholder = copy of frame t (keeps
      BatchNorm on real images), masked in the model. `ValueModel(history=True)`: one shared ResNet-18
      across cameras *and* time [Lars: share across time steps], one backbone pass over
      [now, past]; head input [now feats (1024), has_past · past feats (1024), has_past] = 2049 → 256
      → 201 (R0: 1024). `model_inputs` now returns keyword arguments (`model(**model_inputs(b, dev))`;
      train_baseline, value_dist updated, history_feats refuses history models). +9 tests (past frame
      = t − gap and masked before the start, history off byte-identical, same augmentation draw on an
      identical frame, sub-frame gap refused; head = [now, past, flag], has_past = 0 ignores the
      placeholder per item, presence check, gradients through the past, config round trip); 100 pass.
      Real-data check (val episode, gap 2 s): frames 0 / 29 have_past 0 with |now − past| = 0,
      frame 30 has_past 1 with |Δpix| 21 wrist / 7 side; one CPU forward/backward, CE 5.27
      **Runs submitted 2026-10-24**, one change vs R0 (same section, flags, GPU type):
      `NAME=b2_gap<g> SEEDS=<s> SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1 EXTRA="--history-gap <g>"
      sbatch --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` → `b2_gap2_s{0,1,2}` jobs 18077080 / 81 / 82,
      `b2_gap4_s{0,1,2}` jobs 18077083 / 84 / 85. Two frames per camera doubles decoding (R0 already
      waits on data 53% of the time), so expect ~2× R0's 20 min per seed. Same win rule as Gate 0 (val
      CE below R0 by > 0.020, shape MAE and AUC 1 s no worse). Table:
      `.venv/bin/python scripts/compare_arms.py g0_r0 b2_gap2 b2_gap4`
      **Correction:** those six jobs were cancelled before starting. The runs went as two jobs, seeds
      back to back: `b2_gap2` job **18078017** (move3, **RTX A5000**, 2 h 16 min), `b2_gap4` job
      **18078018** (move4, **L40S**, 2 h 00 min), both COMPLETED 2026-10-24 23:55. Not the rtxpro6000
      of R0, so GPU type is a second difference vs R0
      **Result (3 seeds each): no win at either gap; history gives no measurable gain.**

      | arm | val CE | MAE | shape MAE | eval-repo MAE | AUC 1 s | AUC 5 s | AUC 10 s |
      |---|---|---|---|---|---|---|---|
      | R0 no memory | 3.672 ± 0.010 | 0.1136 | 0.0487 | 0.1826 | 0.621 | 0.610 | 0.821 |
      | gap 2 s | 3.667 ± 0.008 | 0.1110 | 0.0504 | 0.1776 | 0.606 | 0.634 | 0.843 |
      | gap 4 s | 3.672 ± 0.011 | 0.1125 | 0.0504 | 0.1792 | 0.598 | 0.637 | 0.834 |

      Paired test (new rule below), change vs R0 [95% interval]: gap 2 s CE −0.004 [−0.034, +0.025],
      shape +0.0017 [−0.0001, +0.0037], AUC 1 s −0.014 [−0.081, +0.048]; gap 4 s CE +0.001
      [−0.032, +0.031], shape +0.0017 [−0.0003, +0.0038], AUC 1 s −0.023 [−0.101, +0.033]. Every
      interval spans 0; episode-weighted CE is slightly *worse* (+0.005 / +0.014). gap 4 s s2 picked
      step 3.5k (others 7.5k)
- [x] **Win rule replaced (Karen, 2026-10-25)** [ours]: the old rule (CE below R0 by > 2 R0 sd, i.e.
      a bar from a 3-seed sd, and "no worse" with no tolerance) → `scripts/compare_arms.py` now also
      runs a paired hierarchical bootstrap on the saved best-checkpoint val predictions: resample the
      247 val episodes (shared by both arms) and, per arm, its seeds; metrics recomputed as in
      `vl.metrics.score` (checked: reproduces every run's result.json to 1e-6). Win = val CE interval
      entirely below 0, and neither shape MAE nor AUC 1 s credibly worse (interval entirely on the
      bad side). Re-run on all earlier gates: **no past verdict changes** (no Gate 0 / C1 arm won
      before or now); R6 320 px is closest, CE −0.023 [−0.050, +0.003]. Power caveat: the CE
      interval is ±0.025–0.03 and AUC 1 s ±0.07–0.09 on this val set, so the new rule is *stricter*
      on CE than the 0.020 bar, and the AUC 1 s guard almost never fires; effects below ~0.025 CE
      are not detectable with 247 episodes and 3 seeds
- [x] B3. Where the gain is: per-frame CE / p_succ on stall vs success hold frames and by time
      into the episode; check the gain isn't the first frame or episode length
      **Done 2026-10-25** (`.venv/bin/python -m vl.history_gain` → `reports/history_gain.md`, ~2 min CPU).
      Same groups as the B1 follow-up (hold phase, eval repos; stall = `timeout_holding_marker`; frozen
      success = longest pause ≥ 8 s), each split by "still ≥ 4 s now"; same bootstrap as compare_arms.
      Δp_succ (gap 2 s / gap 4 s) [95% interval]:

      | hold-phase group | episodes | R0 p_succ | Δp_succ gap 2 s | Δp_succ gap 4 s | ΔCE gap 2 s / 4 s |
      |---|---|---|---|---|---|
      | stall, still ≥ 4 s | 32 | 0.267 | −0.041 [−0.094, +0.001] | −0.050 [−0.108, +0.006] | −0.05 / −0.09 (n.s.) |
      | stall, moving | 45 | 0.301 | −0.002 | +0.012 | +0.08 / +0.08 (n.s.) |
      | other failure | 14 | 0.193 | **−0.069** [−0.128, −0.022] | **−0.071** [−0.139, −0.022] | −0.05 / −0.08 (n.s.) |
      | success, frozen, inside pause | 13 | 0.595 | −0.017 [−0.102, +0.084] | −0.064 [−0.154, +0.040] | −0.06 / +0.03 (n.s.) |
      | success, pause 2–8 s, moving | 10 | 0.633 | −0.014 | +0.021 | **+0.24** [+0.09, +0.42] / +0.09 |
      | success, pause < 2 s | 22 | 0.592 | +0.018 | +0.035 [−0.004, +0.080] | +0.06 / −0.08 (n.s.) |

      Time into the episode (all val frames): 0–2 s (no past, masked) ΔCE −0.016…+0.011, Δp_succ
      ±0.003, so no first-frame or start-configuration effect. The only credible shift is 20–40 s
      (64 episodes): Δp_succ −0.040 / −0.059. Episode length: long failures Δp_succ −0.036 (gap 2 s,
      credible); medium-length successes ΔCE −0.048 in both arms (credible, unexplained).
      Reading (Claude, to be checked): history does move p_succ the way Lars predicted, down on
      failures that have gone still (and on the 14 non-stall failures), but by only 0.04–0.07, and
      that never becomes a CE gain. Moving stalls get nothing, as expected (no "no change" signal).
      At 4 s it also pulls frozen successes down (−0.064, not credible, n = 13): the harm the B1
      follow-up predicted, the same size as the stall gain. At 2 s it is credibly worse on successes
      that paused 2–8 s and are moving again (10 episodes). No sign that a shorter gap would do better:
      2 s and 4 s show the same stall effect
- [x] B3 follow-up: upper bound for history (Karen, 2026-10-25). Question: if the model is *given* a
      perfect "how long has nothing moved" signal, does val CE improve? If not, image history cannot
      either and Track B stops; if it does, the B2 arms failed at reading the images, not the idea.
      **Code** [ours]: `vl.train_baseline --still-time` → data config `still_time.enabled`, recorded
      as `still_time` in config.json, read back by `run_data_config`. `FrameDataset` adds
      `still` = seconds the gripper has been still at frame t (`vl.dataset.still_seconds`: speed over
      the last 7 frames < 1 cm/s, the `vl.history_plots.stall_seconds` rule of B1 / B3, causal, from
      `data/proprio/frame_proprio.parquet`); the model appends log1p(still) to the head input
      (1024 + 1 → 256 → 201), nothing else changes vs R0. Not image-derived and longer-reaching than
      any one gap (it knows the pause length up to 46 s), so it bounds what history can give.
      +4 tests (rule counts up from the last motion and is causal, row order independent; item =
      seconds at frame t; head = [feats, log1p s]; presence check; config round trip); 104 pass.
      Real data: matches `stall_seconds` to 2e-6 on all train / val frames; > 0 on 41% / 40% of
      frames, p90 13.8 / 11.2 s, max 46 s. CPU forward/backward on 16 val frames: CE 5.31, gradient
      reaches the still column
      **Run submitted 2026-10-25**, one change vs R0, same section, flags and GPU type (rtxpro6000, so
      no GPU confound unlike B2): `NAME=b3_still SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1
      EXTRA="--still-time" sbatch --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` → job **18092151**
      (seeds back to back) → `b3_still_s{0,1,2}`. Read with
      `.venv/bin/python scripts/compare_arms.py g0_r0 b3_still` and
      `.venv/bin/python -m vl.history_gain --arms b2_gap2 b2_gap4 b3_still --report reports/still_time_gain.md`
      **Targeted metrics, pre-registered 2026-10-25 before b3_still finished** (Karen) [ours]: whole-val
      CE dilutes effects that live on a few hold frames (B3: p_succ shifts of 0.04–0.07 on stall
      groups, no CE change), so a null CE from the upper bound would not tell "stillness doesn't
      help" from "diluted". Two primary metrics, frame-level AUC of p_succ, success = positive, hold
      phase, groups of `vl.history_gain.hold_groups` (now shared, B3 report reproduces byte-identical):
      **stall_auc** = all success frames vs stall frames still ≥ 4 s; **frozen_auc** = frozen-success
      frames (still ≥ 4 s, longest pause ≥ 8 s) vs stall frames still ≥ 4 s. Pooled over eval repos;
      in `scripts/compare_arms.py` with the same paired bootstrap (frames weighted by episode draw
      count; weighted AUC checked = pair AUC at unit weights), own "targeted" verdict (credible = interval
      entirely above / below 0); the CE win rule is unchanged (B2 CE columns reproduce exactly).
      Sizes: stall_auc 8114 success frames / 45 episodes vs 5895 / 32; frozen_auc 2731 / 13 vs 5895 / 32.
      Scored on the B2 arms (Δ vs R0, 95% interval):

      | arm | stall_auc | frozen_auc | targeted |
      |---|---|---|---|
      | R0 (point value) | 0.800 | 0.820 | |
      | gap 2 s | +0.023 [−0.022, +0.073] | +0.009 [−0.067, +0.079] | no change |
      | gap 4 s | +0.033 [−0.024, +0.089] | −0.011 [−0.096, +0.058] | no change |

      Reading (Claude, to be checked): history leans the right way on stall_auc but is not credible;
      the interval is ±0.05 (stall) and ±0.08 (frozen), so these metrics resolve shifts of about
      0.05 / 0.08 AUC with 13–45 episodes per class, a power limit like the CE one. The image-only
      baseline already separates frozen successes from frozen stalls (0.82), so "frozen" is not
      the blind spot B1 suggested for R0 itself
      **Result (2026-10-25, job 18092151 COMPLETED, move5 rtxpro6000, 59 min, best steps in
      `b3_still_s*/result.json`): no win; even a perfect stillness signal gives no credible gain.**
      `compare_arms.py g0_r0 b3_still b2_gap2 b2_gap4`, Δ vs R0 [95%]:

      | arm | val CE | shape MAE | AUC 1 s | stall_auc | frozen_auc |
      |---|---|---|---|---|---|
      | still time (upper bound) | −0.016 [−0.041, +0.006] | +0.0017 [+0.0000, +0.0033]* | −0.008 [−0.065, +0.050] | +0.034 [−0.007, +0.084] | +0.013 [−0.044, +0.075] |
      | gap 2 s | −0.004 [−0.034, +0.027] | +0.0017 [−0.0001, +0.0037] | −0.014 | +0.023 [−0.025, +0.072] | +0.009 |
      | gap 4 s | +0.001 [−0.031, +0.031] | +0.0017 [−0.0005, +0.0038] | −0.023 | +0.033 [−0.020, +0.090] | −0.011 |

      (B2 intervals differ in the 3rd decimal from the B2 entry: more arms change the bootstrap draws.)
      Per-seed val CE 3.659 / 3.640 / 3.669 (R0 3.670 / 3.663 / 3.682). Shape MAE credibly worse by the
      same +0.0017 as both B2 arms (lower bound exactly 0). Breakdown `reports/still_time_gain.md`: still
      stalls Δp_succ −0.040 [−0.089, −0.002] (= gap 2 s), still other failures −0.052; *no* harm on frozen
      successes (−0.012 vs gap 4 s −0.064) or on paused-then-moving successes (ΔCE −0.018 vs gap 2 s
      +0.236). Does the model use it? Best checkpoints, 32 val frames still ≥ 8 s: real still time vs
      0 lowers V by 0.024 / 0.030 / 0.031 (seeds 0–2); fixed 2 → 30 s lowers V by ~0.02. Is the
      information there? Hold frames by still time: success share 0.47 (moving) → 0.27–0.32 (4–16 s) →
      0.19 (> 16 s), mean target −0.39 → −0.47…−0.51; R0 is optimistic there by +0.05–0.06 and b3_still
      removes about a third of that (+0.03–0.04 left).
      Reading (Claude, to be checked): stillness is weakly informative on this data because a quarter
      of successes also freeze ≥ 8 s, so a perfect signal shifts targets by only ~0.1 and the model
      learns ~0.03 of it. The best a "how long nothing moved" memory can buy here is about −0.016 CE
      and +0.03 stall_auc, both below what 247 val episodes × 3 seeds resolve. Image history (B2)
      reaches the same stall effect but adds harm on paused successes. Answer to the question: the
      limit is the signal, not how the images were combined; Track B as stall detection can stop.
      Not ruled out: memory carrying other information than stillness (e.g. direction of motion,
      the "other failure" retries), which this bound does not cover
- [x] Caveat checks before closing Track B (Karen, 2026-10-26) [ours], CPU only, saved predictions.
      **1a. Where the interval width comes from** (`.venv/bin/python scripts/power_split.py g0_r0
      b3_still b2_gap2 b2_gap4`, ~3 min): the compare_arms bootstrap run with episodes only / seeds only.
      b3_still val CE: 76% of the variance is seeds (sd 0.0105 vs episodes 0.0059); projected half-width
      0.024 / 0.019 / 0.016 / 0.014 at 3 / 6 / 10 / 20 seeds per arm, so ~10–20 seeds of R0 and b3_still
      would resolve its −0.016 if real. B2 arms: episodes dominate (67–75%), half-width stays ≥ 0.026 even
      at 20 seeds, so image-history effects of this size cannot be resolved on this val set. stall_auc:
      ±0.03 at best. Seed sd comes from 3 seeds per arm, so the projection is rough
      **2a. Residual probe** (`.venv/bin/python scripts/residual_probe.py` → `reports/residual_probe.md`,
      ~15 min): R0's val residual (target − mean V of 3 seeds) regressed on causal proprio-history groups
      (still time; elapsed; path / displacement / dz / d gripper over 2–20 s; retries = gripper toggles,
      z dips, path so far; current proprio as control), additive ridge on binned features, base = bins of
      V, 5 episode folds, λ by inner CV (a fixed λ = 1 overfit episode offsets: every group scored worse
      than predicting 0). **No group explains more than ~2.5% of residual MSE in any subset.** Best,
      hold-phase stall frames (45 episodes): memory −1.3%, motion −1.1% vs base, credible vs still but
      ~0.4 × 1e-3 MSE; other failure (14 episodes) still + motion −5.5%, not credible. Sensitivity caveat:
      still time itself, which b3_still showed carries a little signal, registers only +0.4% here, so the
      probe rules out large missing information, not effects the size of b3_still's
- [ ] **1b. More seeds for the still-time upper bound** (Karen, 2026-10-26) [ours]: is b3_still's
      −0.016 CE real? Seeds 3–9 for R0 and b3_still (10 per arm, 1a projects ±0.016), same section /
      flags / GPU type as seeds 0–2 (config.json of g0_r0_s0 vs b3_still_s0 differs only in
      `still_time`; code since then only adds branches that are off by default: C3 attention head
      gated on `fusion`, 127 tests pass). `NAME=g0_r0|b3_still SEEDS="3 4 5 6"|"7 8 9"
      SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1 [EXTRA=--still-time] sbatch
      --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch` → jobs **18107218 / 18107219** (R0 3–6 / 7–9),
      **18107220 / 18107221** (b3_still 3–6 / 7–9). `seeds()` in compare_arms / power_split /
      history_gain now finds every `<arm>_s<n>`, not just 0–2 (3-seed table reproduces exactly).
      Read: `.venv/bin/python scripts/compare_arms.py g0_r0 b3_still` (+ `power_split.py` for the
      new seed share). **Reading fixed before results:** same win rule as B2/B3 (CE interval
      excludes 0, AUC 1 s guard). If CE still does not resolve at 10 seeds, the stillness bound is
      below what this val set can detect and Track B closes with no history; if it does, history
      carries a small real stall signal (≈0.016 CE upper bound) and Gate B weighs that against cost
- **You check:** B2 table and B3 breakdown
- [ ] **Gate B: Karen confirms** whether history goes into the baseline, and at which gap

### 5.5.4 Track C — architecture: combining images (after Gate 0; C3 uses Gate B's gap)
Lars: share a backbone **across time steps, not across cameras**; wrist and side need different
features. Each arm is one change vs the Gate 0 baseline, 3 seeds, same fair-comparison checks as
Stage 5 (frame index, split, recipe, seeds; diff recorded under `data/runs/`).
- [x] C1. Two ResNet-18s, one per camera (paper's Insert Marker critic), concat → same head.
      Removes the shared BatchNorm statistics. Tests: weights distinct per camera; changing one
      camera's input changes only that backbone's features. ~22M params (= DINOv2-S arm)
      **Code (2026-10-24)** [ours]: `ValueModel(per_camera_backbone=True)` / `vl.train_baseline
      --per-camera-backbone` → `self.backbone` is an `nn.ModuleDict` {camera: ResNet-18}, both from
      the same ImageNet weights, features concatenated in camera order → the same 1024 → 256 → 201
      head. Recorded as `per_camera_backbone` in config.json and read back by `run_data_config`
      (older runs: shared). Same optimizer group and LR (1e-5) for both backbones. 22,353,024 backbone +
      314,057 head = 22.67M params (shared: 11.49M). `layer_decay_groups` refuses per-camera,
      `vl.history_feats` asserts a shared backbone. +7 tests (`tests/test_model.py`: distinct
      params and BN buffers, identical pretrained init, changing one camera changes only its
      backbone's feature, forward = head(concat), BN stats per camera, config round trip); 91 pass.
      CPU smoke on a real augmented batch: CE 5.37 → 3.56 in 3 steps, checkpoint reloads through
      `run_data_config` bit for bit. Diff `data/runs/c1_code_vs_2df9f76.diff` (also contains the
      concurrent `--image-size` change, default 224, a no-op here)
      **Runs submitted 2026-10-24**, one change vs the Gate 0 baseline R0 (`g0_r0_s{0,1,2}`, val CE
      3.672 ± 0.010, MAE 0.1136, shape 0.0487, eval-repo 0.1826, AUC 1/5/10 s 0.621/0.610/0.821),
      same section, GPU type and flags: `NAME=c1_percam SEEDS=<s> SECTION=baseline_lowlr_aug_full
      DIST=1 DROP_LAST=1 EXTRA="--per-camera-backbone" sbatch --gres=gpu:rtxpro6000:1
      scripts/cameras.sbatch`, jobs 18065698 / 18065699 / 18065700 (seeds 0 / 1 / 2) →
      `c1_percam_s{0,1,2}`. Same win rule as Gate 0 (val CE below R0 by > 0.020, shape MAE and
      AUC 1 s no worse). Table: `.venv/bin/python scripts/compare_arms.py g0_r0 c1_percam`
      **Result (2026-10-24, 3 seeds, all COMPLETED on move5 rtxpro6000, ~22 min each): no win; ties R0.**

      | arm | best step | val CE | MAE | shape MAE | eval-repo MAE | AUC 1 s | AUC 5 s | AUC 10 s |
      |---|---|---|---|---|---|---|---|---|
      | R0 shared ResNet-18 | 6.5k / 7.5k / 4k | 3.672 ± 0.010 | 0.1136 | 0.0487 | 0.1826 | 0.621 ± 0.017 | 0.610 ± 0.017 | 0.821 |
      | C1 per-camera ResNet-18s | 6.5k / 7.5k / 3.5k | 3.670 ± 0.003 | 0.1138 | 0.0506 ± 0.002 | 0.1817 | 0.583 ± 0.039 | 0.655 ± 0.025 | 0.823 |

      Val CE −0.002 (bar −0.020), MAE and eval-repo MAE unchanged. Shape MAE +0.002 (~2 R0 sd, worse).
      AUC 1 s −0.04 (per seed 0.61 / 0.54 / 0.60 vs 0.63 / 0.60 / 0.64; seed 1 drives most of it),
      AUC 5 s +0.045 (~2.6 R0 sd): mixed and noisy, not a consistent timing gain. C1 fits a bit
      faster (val CE at 2k 3.70–3.73 vs 3.75–3.76; train loss at 8k 3.50–3.53 vs 3.55–3.60) but
      reaches the same best val CE, with no extra overfitting by 8k (3.68–3.69 in both). Reading:
      separate BN statistics / features per camera don't matter at this data size and recipe;
      twice the parameters buy faster fitting, not better generalization. Stays on the list as the
      per-camera trunk C3 would use (Lars: share across time, not cameras), but it is not a gain by itself
- [ ] C2. Channel stacking for history [Lars, alternative]: current + past RGB stacked as 6
      channels into one per-camera ResNet (first conv widened, pretrained weights copied to both
      halves and scaled ½). Needs Gate B's gap
- [ ] C3. Transformer fusion [Lars]: each image (per camera, per time step) → one token
      (ResNet feature + learned camera and time embeddings), a learnable readout/query token
      attends to all image tokens (1–2 small transformer layers), readout → same 201-bin head.
      Variants: cameras only (vs C1's concat), then cameras + history. Per-camera backbones
      shared across time
      **Code, cameras-only variant (2026-10-25)** [Lars: one token per image, a learnable readout;
      sizes and block design ours]: `ValueModel(fusion="attention")` / `vl.train_baseline --fusion
      attention` → `model.head` is a `TokenFusionHead` with the same `head(f)` input as the concat MLP
      (so `features`, the head optimizer group at LR 1e-4, checkpoints and `history_feats` are
      unchanged). Each camera's 512-d feature → Linear → 256-d token + learned camera embedding (+ time
      embedding, 1 step for now); a learned readout token goes through 2 pre-norm cross-attention blocks
      (4 heads, FF 512, GELU, no dropout; only the readout is a query, camera tokens are not updated, so
      its attention is one weight per camera per frame, summing to 1); readout → LayerNorm → 256 → ReLU
      → 201. Head 1.31M params vs C1's 0.31M (backbones 22.35M, same as C1): a second difference vs C1,
      on top of the fusion rule. `key_padding_mask` and `n_time` are in the head for the history variant
      but not wired; `fusion="attention"` with history / still time / proprio raises. Recorded as
      `fusion` in config.json, read back by `run_data_config` (older runs: concat). `vl.attention --run
      <run>` → `val_attn.parquet` (same rows as val_predictions, `attn_<cam>_l<layer>`), run per seed by
      `cameras.sbatch ATTN=1`. +14 tests (`tests/test_model.py` head: shape, attention is a distribution,
      camera identity via embedding, attention depends on the frame, masked tokens ignored, fully masked
      row refused, gradients to every parameter; model: token head used, each camera matters, fits one
      batch, unbuilt inputs refused, config round trip; `tests/test_attention.py`); 118 pass. GPU smoke
      (TITAN RTX, bf16 AMP, 3 steps, 2 val episodes): finite val CE 5.297, attention layer sums 1.0000,
      checkpoint reloads strict, attention rows = val_predictions rows. Diff
      `data/runs/c3_code_vs_2df9f76.diff` (also contains the concurrent still-time / C1 / B2 changes)
      **Run submitted 2026-10-25**: `NAME=c3_attn SECTION=baseline_lowlr_aug_full DIST=1 ATTN=1 DROP_LAST=1
      EXTRA="--per-camera-backbone --fusion attention" sbatch --job-name=vl-c3 --gres=gpu:rtxpro6000:1
      scripts/cameras.sbatch` → job **18092717**, seeds 0–2 back to back → `c3_attn_s{0,1,2}`. Primary
      reference C1 (`compare_arms.py c1_percam c3_attn`: one change, concat → attention), secondary R0
      (`compare_arms.py g0_r0 c3_attn`). Same win rule (paired bootstrap)
      **Provenance (review, 2026-10-25)** [ours]: seeds of one job start up to ~1.5 h apart and read the
      code on disk then, while another session edits `src/`. `vl.train_baseline` now writes each run's
      uncommitted code (`git diff HEAD` + untracked files under src/configs/scripts) to
      `data/runs/<run>/code.diff` and its sha256 as `code_sha256` in config.json (+1 test; 119 pass).
      Check before reading C3: the three seeds' `code_sha256` match, and config.json says
      `section baseline_lowlr_aug_full`, `fusion attention`, `per_camera_backbone true`.
      **C4 caveat (review)**: the readout attention weights are not camera contributions (the values are
      learned projections that can scale a low-weight camera up, the readout keeps a residual from block 1,
      and the dump averages heads). Read `val_attn.parquet` next to a contribution check, e.g. ΔV when one
      camera token is masked, before claiming "wrist weight rises at insertion"
      **Result (2026-10-25, job 18092717, move5 RTX PRO 6000, 3 seeds ~15 min each; all three seeds
      `code_sha256` 7d213a87df2c, section baseline_lowlr_aug_full): no win; slightly worse and noisier.**

      | arm | best step | val CE | MAE | shape MAE | eval-repo MAE | AUC 1 s | AUC 5 s | AUC 10 s |
      |---|---|---|---|---|---|---|---|---|
      | C1 concat | 6.5k / 7.5k / 3.5k | 3.670 ± 0.003 | 0.1138 | 0.0506 | 0.1817 | 0.583 | 0.655 | 0.823 |
      | C3 attention | 3.5k / 3.5k / 4k | 3.689 ± 0.030 | 0.1138 | 0.0541 | 0.1821 | 0.566 | 0.600 | 0.833 |

      Paired vs C1 [95%]: CE +0.019 [−0.032, +0.073], shape MAE **+0.0035 [+0.0002, +0.0066]** (credibly
      worse), AUC 1 s −0.017 [−0.113, +0.091], stall_auc +0.013, frozen_auc +0.022 (both n.s.). Vs R0: CE
      +0.018 [−0.036, +0.072], shape **+0.0054 [+0.0029, +0.0079]**. Per-seed CE 3.696 / 3.657 / 3.715
      (C1 3.666–3.672): seed sd 10× C1's. The attention head fits faster and overfits: best checkpoints at
      3.5–4k (C1 mostly 6.5–7.5k), train loss 3.36 at 8k vs C1 3.50–3.53, s2 val CE 3.72 → 3.84 by 8k.
      Attention (last block, mean over heads; weights, not contributions, see caveat): wrist 0.52–0.58
      overall; last 5 s before T of successes 0.69 / 0.71 / 0.70 vs 0.47–0.54 earlier and 0.47–0.63 on
      failures, the same in all 3 seeds. Reading (Claude, to be checked): the fusion rule does not help at
      this data size; the bigger from-scratch head overfits sooner. The attention does move the way Lars
      expected (more wrist near insertion of successes); C4 should confirm it with a masking check
      **Regularised C3, second try (2026-10-25)** [ours]: the first C3 overfit (best at 3.5–4k, train loss
      below C1's) and had a 4× bigger head than C1. Two arms, one change each vs `c3_attn`:
      `c3_small` = head shrunk to C1's size (`--fusion-cfg {"d":128,"n_layers":1,"ff":256}`: 0.28M params vs
      C1 0.31M), and `c3_small_do` = the same + dropout 0.1 (image tokens, attention weights, both residual
      branches). Code: `TokenFusionHead(dropout=)`, `ValueModel(fusion_cfg=)`, `vl.train_baseline
      --fusion-cfg JSON`, recorded as `fusion_cfg` in config.json ({} = the c3_attn defaults). +5 tests
      (small head within ±20% of the concat head, dropout on in train / off in eval, cfg reaches the head,
      concat refuses a cfg, config round trip, default key names unchanged so c3_attn_s{0,1,2} reload strict —
      a first version put Dropout inside the FF Sequential and renamed `ff.2`; caught by a reload check,
      fixed); 124 pass. GPU smoke (TITAN RTX) ok. Jobs **18095315** (`c3_small_s{0,1,2}`) and **18095316**
      (`c3_small_do_s{0,1,2}`), same command as C3 plus `--fusion-cfg` with space-free JSON (cameras.sbatch
      word-splits EXTRA; the first submission, 18095300/01, quoted the JSON and was cancelled before
      starting). Compare: `compare_arms.py c1_percam c3_attn c3_small c3_small_do`
      **Result (2026-10-25, move5 RTX PRO 6000, 3 seeds each, all `code_sha256` 4a08ca2fc8a6, configs
      checked): regularising does not rescue C3; both arms are worse than the first C3.**

      | arm | head params | best step | val CE | shape MAE | AUC 1 s | AUC 5 s | train loss 8k (s0) |
      |---|---|---|---|---|---|---|---|
      | C1 concat | 0.31M | 6.5k / 7.5k / 3.5k | 3.670 ± 0.003 | 0.0506 | 0.583 | 0.655 | 3.50–3.53 |
      | C3 attention | 1.31M | 3.5k / 3.5k / 4k | 3.689 ± 0.030 | 0.0541 | 0.566 | 0.600 | 3.40 |
      | C3 small | 0.28M | 3.5k × 3 | 3.704 ± 0.008 | 0.0530 | 0.627 | 0.612 | 3.37 |
      | C3 small + dropout 0.1 | 0.28M | 3.5k × 3 | 3.725 ± 0.020 | 0.0527 | 0.611 | 0.619 | 3.47 |

      Paired vs C1 [95%]: small CE +0.034 [−0.007, +0.078]; small + dropout CE **+0.055 [+0.009, +0.108]**
      (credibly worse); shape, AUC, stall_auc, frozen_auc all n.s. Reading (Claude, to be checked): the early
      peak (every seed best at 3.5k) and the low train loss survive a 4.6× smaller head and dropout, so the
      overfitting is not head capacity. Likely the attention path itself: through the readout, each
      backbone's gradient depends on the other camera, so the per-camera ResNets fit the train set faster
      (untested). Dropout lowers fit a little (train 3.47) but costs val CE. Three C3 variants, none at or
      below C1 or R0: attention fusion of two cameras gives no gain here
- [x] C4. Attention check for C3: log readout attention per token; expected [Lars]: wrist weight
      rises at insertion, history weight low outside stalls. Plots by task phase and outcome
      **Done 2026-10-26 as a masking check** (the C4 caveat: weights are not contributions) [ours]:
      `vl.camera_contrib` (new; +3 tests, 127 pass): backbone features of every val frame once, then the
      head re-run with one camera removed, `drop` = its token masked out of the readout attention (C3
      only), `mean` = its features replaced by their val mean (same intervention for concat heads).
      Job **18100831** (move3 A5000, 11 min for 9 runs: `c3_attn`, `g0_r0`, `c1_percam` × 3 seeds;
      full-model CE reproduces each run's training-time val CE to 1e-4) → `data/runs/<run>/val_cam_mask.parquet`,
      `.venv/bin/python -m vl.camera_contrib --report c3_attn g0_r0 c1_percam` → `reports/camera_contrib.md`.
      Phases [ours]: success / not success × first 5 s, middle, last 5 s before T. Means over 3 seeds:

      | phase | C3 attn wrist | C3 drop: wrist share | C3 drop ΔCE wrist / side | R0 mean: wrist share | R0 ΔCE wrist / side | C1 ΔCE wrist / side |
      |---|---|---|---|---|---|---|
      | success, first 5 s | 0.36 | 0.25 | +0.82 / +0.54 | 0.20 | +0.43 / +0.18 | +0.36 / +0.14 |
      | success, middle | 0.57 | 0.38 | +0.58 / +0.69 | 0.34 | +0.03 / −0.00 | −0.02 / −0.01 |
      | **success, last 5 s** | **0.70** | **0.66** | **+2.95 / +0.04** | **0.63** | **+2.56 / +0.15** | **+2.24 / +0.20** |
      | not success, first 5 s | 0.38 | 0.33 | −0.09 / +0.71 | 0.17 | −0.02 / +1.22 | −0.06 / +1.50 |
      | not success, middle | 0.59 | 0.40 | −0.40 / +1.08 | 0.26 | −0.17 / +1.23 | −0.13 / +1.28 |
      | not success, last 5 s | 0.55 | 0.43 | −0.20 / +0.60 | 0.29 | −0.15 / +0.80 | −0.04 / +0.73 |

      (ΔCE = val CE with the camera removed − full, nats per frame; seed sd ≤ 0.2 except C3 / R0 side ≤ 0.35.)
      Reading (Claude, to be checked): (1) **Lars's expectation holds as a contribution, not only as a
      weight**: in the last 5 s of successes the wrist dominates in all three architectures (share
      0.59–0.71), removing it costs +2.2–3.0 nats and lowers V by 0.09–0.13 (the wrist sees the seat);
      the side camera is nearly irrelevant there (+0.04–0.20). (2) **On failures the side camera carries V
      and the wrist hurts**: removing the side camera costs +0.6–1.9 nats and *raises* V by 0.1–0.28;
      removing the wrist mostly *lowers* CE (−0.02 … −0.40), so the wrist view makes failures look more
      promising: a candidate source of V's optimism on stalls at the holder. (3) It is the same in R0 and
      C1, so it is a property of the data, not something attention fusion learned. (4) C3's attention
      weights track contributions in direction (per-frame rank correlation with the drop share
      0.40 ± 0.12) but overstate the wrist on failures (weight 0.55–0.59 vs share 0.40–0.43). History
      weight: not applicable (no history token in C3)
- **You check:** C1–C3 results, C4 attention plots
- [ ] **Gate C: Karen confirms** the architecture

### 5.5.5 Merge gate
- [ ] Combine the confirmed winners of Gate 0 + A + B + C into one recipe; 3 seeds; check the
      combination is no worse than each winner alone (Stage 5 showed crop + full aug didn't stack)
- [ ] **Merge gate: Karen confirms the new baseline** → then Stage 5's "Freeze all choices",
      final test evaluation and the progress package for Lars

## Stage 6 — Policy improvement (Phase 2; only after the Stage 4 gate and the Stage 5.5 merge gate pass)
Advantage conditioning happens here, not in this week's deliverable.
Context (Lars, 2026-10-23): the lab's goal is a π₀.₅ DROID policy that improves itself with no
human in the loop. Two streams already run: Astra (slow, general; gets tasks from 0% to non-zero)
and Rohan (rolls out the policy and keeps successes; 50% → 80–90%). The missing piece is value
learning + RL, which is this project; likely done alongside Rohan.
- [ ] 6.0 Setup with Lars/Rohan: π₀.₅ DROID checkpoint, compute (IRIS), their rollout data
      format and success labels; which task(s) to start on. **Gate: Karen confirms** the setup
- [ ] 6.1 Value head on π₀.₅ [Lars]: a value head that cross-attends into the π₀.₅ DROID
      model's tokens (frozen VLA first, then low-LR fine-tune), same 201-bin targets and Stage 5.5
      inputs. Compare on the same val set against the Stage 5.5 baseline
- [ ] 6.2 Does it learn something meaningful? Same Stage 4/5 checks (MAE, shape, within-round
      AUC, stall vs success, eval repos) + trace plots. **Gate: Karen confirms** the VLA value
      function is good enough to make advantages from (else fall back to the Stage 5.5 V)
- [ ] 6.3 Then the RECAP steps below (advantage labels → thresholds → advantage-conditioned
      training). Advantage-conditioned RECAP has only been shown in sim so far; a real-robot
      version is the contribution
- [ ] Advantage labels: Â_t = Σ_{k=t}^{t+n−1} r̃_k + V̂(o_{t+n}) − V̂(o_t), rewards and V in the
      same normalization, terminals handled explicitly, lookahead n chosen in seconds then
      converted to frames (do not copy RECAP's frame counts)
- [ ] Threshold advantages into positive/negative indicators (as in RECAP)
- [ ] Inspect advantage distributions and positive/negative examples first. A rising value is
      not automatically a positive advantage (progress must beat the cost of elapsed time).
      An overfit V gives ~zero advantages on its training episodes: consider out-of-fold V.
- [ ] π₀.₅ advantage-conditioned fine-tuning; compare with vs without V (IRIS cluster). Lars:
      conceptually simpler than learning V but needs more compute.
- [ ] Fair comparison: matched initialization, budget, preprocessing and data mixture; include a
      strong imitation baseline; separate the effect of conditioning from the effect of which
      data reaches the actor
- [ ] Real policy improvement needs robot rollouts (or an execution environment); offline
      imitation loss alone cannot show it. Ask Lars what evaluation setup is available.

## Stage 7 — LWD comparison (later)
- [ ] LWD as a comparison: does bootstrapped value learning give better improvement signals?
      Matched comparison; trade-offs between the two (both are SOTA VLA post-training)

## Stage 8 — Autonomous improvement loop and throughput (after Stage 6; Lars, 2026-10-23)
Final system [Lars]: π₀.₅ policy (core) + learning module (BC now, full RL the goal) + success
classifier (labels rollouts good/bad) + System 2 model (Astra or equivalent, slow fallback for
tasks at 0%). Main emphasis: **throughput, not only success rate** (89% success alone isn't
economically enough; e.g. 1 box/min → 3–4), at 1× speed looking reliable (not sped-up
videos). Each step ends with **Gate: Karen confirms** before the next.
- [ ] 8.1 Plug V into Rohan's rollout stream: score live rollouts, compare V's end-of-episode
      judgement against the success classifier's labels (agreement, disagreements reviewed by
      eye). **Gate**
- [ ] 8.2 Throughput as the objective: the per-step cost already makes V prefer shorter
      episodes; check advantage conditioning actually speeds the policy up (time-to-success and
      completions/min at 1× speed, not only success rate). **Gate**
- [ ] 8.3 Iterate: rollouts → V/advantages → retrain → rollouts, no human labels; track success
      rate and throughput per iteration; decide when V must be re-trained on new rollouts. **Gate**
- [ ] 8.4 Interface with System 2 (Astra) for tasks at 0%: which tasks hand over and when
      (out of scope to build; agree the hand-off with Lars)
