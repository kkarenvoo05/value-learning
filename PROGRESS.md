# Project progress

Legend: `[x]` done and verified · `[~]` partly done · `[ ]` not started.
Details for finished items are in `reports/dataset_audit.md`.

## Stage 0 — Setup
- [x] Project skeleton, `pyproject.toml`, `.venv` (uv), `.gitignore` (data/videos/checkpoints excluded)
- [x] `README.md` and `CLAUDE.md` for future sessions
- [x] `git init` (nothing committed yet)
- [ ] Get compute + storage (mentor's workstation over SSH, ~50–100 GB disk, GPU)
- [ ] First commit

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
- [ ] Inspect odd episodes by video: c02/82 (7.3 s "timeout"), r03/5, 34, 35, 50 (over the cap)
- [ ] Crop check on mid-episode frames (gripper near top edge during insertion)

## Open questions (block Stage 2)
- [ ] Q1: timeouts — treat as failure (RECAP) or as censored (Mulligan critic)? Ask Lars.
- [ ] Q2: cameras — confirm `wrist_left` + `side_1`. Ask Lars.
- [ ] Q3: where training data lives (see Stage 0)

## Stage 2 — Targets
- [~] Target design proposed (seconds, C_fail = H = 60 s, v = R/120 s, 201 bins on [−1, 0])
- [ ] Freeze failure penalty, normalization, support, timeout rule
- [ ] Implement target construction; report out-of-support targets
- [ ] Tests on hand-computable synthetic episodes

## Stage 3 — Data and model
- [ ] Download pinned training-pool `side_1`, `wrist_left`, data on the workstation
- [ ] Frame loader (valid frames only, original timing kept when subsampling)
- [ ] Fixed per-camera crops chosen
- [ ] Model: shared ImageNet ResNet-18 → concat 2×512 → MLP → 201 logits; cross-entropy
- [ ] Tiny-subset overfitting check

## Stage 4 — Baseline training and evaluation
- [ ] Train baseline; select checkpoint on validation
- [ ] Metrics: CE, expected-value error, per-episode/source breakdowns vs trivial baseline
- [ ] Synchronized video/value plots (predictions, targets, intervention markers)
- [ ] Failure-case analysis: what the model learns

## Stage 5 — Ablations and test
- [ ] One camera vs two
- [ ] Proprioception (if useful)
- [ ] Timeout-handling sensitivity (if Q1 leaves it open)
- [ ] Freeze all choices
- [ ] Final test evaluation (r05-eval, r05-screen reported separately)

## Stage 6 — Toward policy improvement
- [ ] Check advantage labels (consistent units, lookahead defined in time)
- [ ] π₀.₅ advantage-conditioned fine-tuning
- [ ] Compare against LWD-style value learning
