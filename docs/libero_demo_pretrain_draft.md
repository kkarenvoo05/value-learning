# DRAFT: RECAP-style value pre-training on LIBERO demos (2026-11-01, not started)

Status (2026-11-02 08:40): table + ResNet stage 1 done (inner CE 2.638); ResNet stage 2 submitted (18265882-84);
LoRA smoke failed on warmup > steps (mine), resubmitted 18265885 → stage 1 18265886. Moka demos KEPT in stage 1
(default; `--exclude-task` exists). LoRA stage 2 waits for 18265886. Details: docs/libero_plan.md 2026-11-02 08:40.

- Download: `/iris/u/ankile/karenvo/libero/lerobot/physical-intelligence_libero` (33 GB, 1,693 episodes, 40 tasks).
- Table: `scripts/libero_demo_table.py` (openpi env) → `/iris/u/ankile/karenvo/libero/tables/libero_demos/`
  (raw .npz as the rollouts, 224 px via openpi_client resize_with_pad; task → suite from the .bddl file names as
  LIBERO's `grab_language_from_filename`; roles fit / inner, 10% per task, rng 0); targets `python -m
  vl.libero_demos --dataset <table>` (per-suite scale 2 × max_steps; libero_10 = moka's 1040); tests
  `tests/test_libero_demos.py`. Smoke (20 episodes): images equal a PIL bilinear resize of the source (max diff 0),
  state equal, targets equal hand values. ~128 KB/frame on disk → ~35 GB.
- Trainer changes: `Frames(multi_prompt=)` (per-episode prompt) in `scripts/pi05_value_expert.py`; a `role` column in
  episodes.csv switches `scripts/pi05_value_lora.py` and `vl.libero_resnet` to fit / inner from the table (val =
  inner in stage 1); `--init-from` in both (weights only, fresh optimiser). Moka tables have no `role`: unchanged.
  182 tests pass.
- Jobs: table 18260656 (CPU) → ResNet stage 1 18260657 (8,000 steps, as its moka runs); LoRA smoke 18260658
  (after the table and frozen B fixed + fresh s2 18237900, for the GPU limit) → LoRA stage 1 18260659 (lr 1e-4,
  5,400 steps, accum 4, as 18237894).
Source for the recipe: RECAP / π\*0.6 (arXiv 2511.14759) §V-D, summarised in `docs/libero_plan.md` (2026-11-01).

## Question
Does pre-training V on success demonstrations (steps-to-success) before the moka fine-tune improve
progress tracking (change SNR 2 s, first-pot milestone), and does it help pi0.5 more than the ResNet?

## Demo data: what exists
- **Not on disk.** The LIBERO config's `datasets:` dir (`/iris/u/ankile/karenvo/libero/datasets`) does not exist;
  no LIBERO demo hdf5 or `physical-intelligence/libero` copy under `~/.cache/huggingface`, `/iris/u/ankile`,
  `/iris/u/am208`, `/iris/u/rbhowmik` (find depth ≤ 6, timed out at 100 s, so not fully exhaustive).
- **Option 1 (recommended): `physical-intelligence/libero` (HF, LeRobot).** The dataset `pi05_libero` was trained
  on (openpi `config.py`, `repo_id="physical-intelligence/libero"`): libero_spatial / object / goal / 10, no-noop
  demos, built by openpi's `examples/libero/convert_libero_data_to_lerobot.py`. Same features as our moka set
  (image, wrist_image, state 8, actions 7, 10 fps), so the existing `--dataset` loaders should read it after an
  `episodes.csv` is written. Size: check before download; `/iris/u/ankile` is 97% full (204 GB free).
- **Option 2: libero_90 demos** (LIBERO hdf5 via `third_party/libero/benchmark_scripts/download_libero_datasets.py`).
  New to pi0.5 (not in its policy training), so neither model has seen them: removes pi0.5's head start. Costs
  more: hdf5 → re-render at the policy's resolution/flip → LeRobot. Worth it only as a second arm.

## Targets [ours unless marked]
- Every demo is a success, so V target = −(steps to success) [RECAP §V-C], **normalised per task by that task's
  maximum episode length** [RECAP §V-C: "we normalize the values per task based on the maximum episode length of
  the task"; corrects the earlier draft's single 1040 normaliser]. Moka stays on its existing scale.
- **Stage 2 data = moka rollouts + the 50 moka demos** [RECAP Alg. 1 line 3: D_ℓ is initialised with the task's
  demonstrations, then rollouts are added; V_ℓ is always trained from V_pre, lines 4 / 8].
- **Scale gap (record in any write-up):** RECAP's V_pre sees "tens of thousands of hours of demonstrations from
  numerous tasks and a variety of different robots" (§IV-C); ours would see 1,693 demos ≈ 7.6 h (273,465 frames at
  10 fps), one robot, 40 tasks. Plus web-data co-training against overfitting (§V-C), which we do not have.
- Outcome head: no failures in demos → outcome loss weight 0 in stage 1.
- **Moka demos (libero_10 task "put both moka pots on the stove")**: keep (they are human demos, not our
  rollouts; RECAP pre-trains on all tasks) — Karen to confirm. Check: no val init states / layouts depend on them.
- Hold out ~10% of demo episodes for stage-1 selection (steps-to-success MAE).

## Arms (one seed each first, go / no-go, then seeds 1–2)
Stage 2 data for all arms = fixed + fresh (825 fit episodes), same inner/val split and selection as existing arms.

| arm | stage 1 (demos) | stage 2 (moka) | exists? |
|---|---|---|---|
| ResNet | – | fixed + fresh | yes, 3 seeds |
| ResNet + demos | full fine-tune, same steps as below | fixed + fresh | new |
| pi0.5 LoRA | – | fixed + fresh | **new control** (current LoRA run is fresh-only) |
| pi0.5 LoRA + demos | LoRA lr 1e-4 + expert | fixed + fresh | new |

Order: (1) ResNet + demos (hours; tells us whether demos help progress tracking at all); (2) LoRA fixed + fresh
control and LoRA + demos in parallel (~1 day each at 4.3 frames/s); stay within 2–3 GPUs.
Go / no-go: change SNR and milestone vs the arm's no-demo control; final comparison as the 3b rule.

## Code needed
1. Demo table: `episodes.csv` (episode_index, task, T, success=1) for the downloaded LeRobot set; reuse
   `vl.libero_targets` for `frame_targets.parquet`. Test: hand-computed targets for one short demo.
2. `--init-from <ckpt>` for `vl.libero_resnet` and `scripts/pi05_value_lora.py` (load stage-1 weights, fresh
   optimiser). Test: loaded weights equal saved ones; step 0 predictions equal stage-1 predictions.
3. Outcome-loss weight 0 flag for stage 1 (LoRA script already has `outcome weight`; check ResNet).
4. Prompt: demos carry their own task strings; pass them (pi0.5 is language-conditioned), not the moka prompt.

## Open decisions for Karen
- Download `physical-intelligence/libero` (size TBD, shared disk at 97%)?
- Keep the moka demos in stage 1?
- Stage-1 step budget (proposal: equal number of frames seen for ResNet and LoRA).
- libero_90 arm later, to remove pi0.5's head start?
