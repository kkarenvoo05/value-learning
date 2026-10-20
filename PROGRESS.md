# Project progress

Legend: `[x]` done and verified · `[~]` partly done · `[ ]` not started.
Details for finished items are in `reports/dataset_audit.md`.

## Stage 0 — Setup
- [x] Project skeleton, `pyproject.toml`, `.venv` (uv), `.gitignore` (data/videos/checkpoints excluded)
- [x] `README.md` and `CLAUDE.md` for future sessions
- [x] `git init`
- [~] Storage. Lars: training data goes on NFS storage under Karen's name, **not** under `/move/`.
      **Temporary solution (2026-10-17):** `data/` in the project is a symlink to
      `/sailhome/karenvo/vl-data` (NFS home, 20 GB quota, ~19 GB free after moving
      `~/.cursor-server` to `/move/u/karenvo/.cursor-server` behind a symlink). Fits the
      ~12 GB training cameras + data with a few GB to spare; the 3.3 GB test videos make it
      tight, so fetch those later. Home filling up breaks the shell, so check `df -h ~` before
      each download.
      **Permanent:** `/iris/u/karenvo` does not exist and Karen is not in the `finnlab` group that
      owns `/iris/u`. Ask Lars to get it created, then move `vl-data` there and re-point `data`.
      No other NFS mount on move2 is writable for this account (`/scr` is a local move2 disk).
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
- [~] Q3: storage → NFS under Karen's name, not under `/move/` (Lars). Temporary: home directory
      (`/sailhome/karenvo/vl-data`); permanent `/iris/u/karenvo` pending (see Stage 0).
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
      Video ↔ proprio timing is ambiguous at 0 or +6 frames (probably 6-frame action chunks):
      recheck before the Stage 5 proprio ablation
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
- [ ] 2. Images vs images + proprioception (end-effector coordinates; inspect fields and frames
      first; normalize with training statistics only)
- [ ] 3. Frozen vs fine-tuned encoder. Fine-tuned arm = `lowlr_8k` / `_s1` / `_s2` (backbone at
      `lr_backbone` 1e-5). Frozen arm: `--freeze-backbone` [ours], same `baseline_lowlr` recipe,
      cameras and head LR; ImageNet backbone gets no optimizer group and stays in eval mode, so
      BatchNorm statistics don't move either (`tests/test_model.py`). ~0.3M trainable params.
      Running 2026-10-19: job 17971269 (`frozen_s{0,1,2}`, move5 **RTX PRO 6000**, not the
      fine-tuned arm's L40S; seeds in sequence, git `2eeb32c`, `scripts/frozen.sbatch` with
      `--partition=move --gres=gpu:rtxpro6000:1 --cpus-per-task=32`; humanoid job 17971059
      cancelled while queued)
- [ ] Image augmentation (Karen, 2026-10-19; plan: `reports/stage5_augmentation_plan.md`), aimed at
      the mentor's concern that V before the last ~10 s looks random. Distribution plots show two
      humps (success/failure) with V their average: real outcome uncertainty (not fixable by
      augmentation) plus flicker of the hump balance between near-identical frames (maybe fixable).
      - [ ] Step 0: pre-end flicker metrics on the baseline (needs full-val distributions, job 17973560)
      - [ ] Step 1: arms A1 photometric, A2 crop shift/scale, A3 both; 3 seeds each, val never augmented
            Implemented 2026-10-19 (`vl.dataset` augment, sections `baseline_lowlr_aug_{photo,crop,both}`,
            tests; real-frame check `reports/figures/stage5/augment_examples.jpg`: marker, gripper and
            holder stay in view). Submitted on humanoid: jobs 17977330 (photo), 17977331 (crop),
            17977332 (both), runs `aug_<arm>_s{0,1,2}`, seeds in sequence
      - [ ] Step 2: pre-set rule: accuracy within 2 sd, pre-end |ΔV| and |Δp_succ| ≥ 25% lower and
            better than 0.5 s smoothing, real uncertainty kept (failure/stall plots stay two-humped)
- [ ] Later: current frame vs short history; categorical vs scalar head; other ways of combining
      the two images; larger backbone only if ResNet-18 underfits
- [ ] Optional diagnostic: timeouts excluded (baseline is timeout = failure)
- [x] Not needed (Stage 4: V does not react to the operator). Was: if Stage 4 shows V reacting to the operator: masked `wrist_left` bottom-left corner vs
      unmasked (see Stage 4 operator check)
- [ ] Freeze all choices
- [ ] Final test evaluation (r05-eval, r05-screen reported separately; check r05-screen
      provenance before treating it as equivalent; they share the same 50 starts)
- [ ] Progress package for Lars: what was built (architecture, inputs, targets, split), metrics
      + 3 representative plots/videos, failure examples, one comparison, next decision; with
      config, pinned revisions and checkpoint

## Stage 6 — Policy improvement (Phase 2; only after the Stage 4 gate passes)
Advantage conditioning happens here, not in this week's deliverable.
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
