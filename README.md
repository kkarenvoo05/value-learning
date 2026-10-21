# value-learning

Does a small RECAP-style **state-value function V(o)** learn useful task progress from a
real-robot manipulation dataset (Mulligan **Insert Marker**)?

If V proves useful, it can serve as a signal for downstream policy improvement.

## References

- RECAP / π*0.6 — https://arxiv.org/abs/2511.14759 (reward −1/step, failure penalty, values
  normalized per task to (−1, 0), 201-bin categorical value, advantage-conditioned policy)
- Mulligan — https://arxiv.org/abs/2610.05882 (the data source; release docs live in each HF
  repo's README and `meta/` sidecars)
- LWD — https://arxiv.org/abs/2605.00416 (related value-learning approach)
- Data — https://huggingface.co/collections/mulligan/real-insert-marker

Read the HF release docs, not only the papers: outcome/endpoint semantics live there.

## Ground rules

- Correctness and understanding beat complexity. Small readable PyTorch, no big frameworks.
- Separate **paper-specified** details from **our choices**; never claim exact reproduction.
- The train/test split is fixed in `configs/dataset.yaml`; validation (~15% of episode
  *groups*, fixed seed) is our choice. Test repos are for final evaluation only: metadata
  integrity checks now, no tuning on test performance.
- Labels come from reviewed outcomes and endpoints, with unknowns flagged, never turned into
  failures. Machine (VLM) stage labels are diagnostics, not truth.
- Never commit data, caches, checkpoints, videos, or credentials (`.gitignore`).
- Record configs, seeds, dataset commits, commands, and results. Only report what actually ran.

## Layout

```
configs/dataset.yaml       repos, split, validation seed, metadata patterns
configs/targets.yaml       target constants (horizon, failure penalty, bins, timeout rule)
configs/data.yaml          cameras, crops, image size, normalization
configs/train.yaml         optimizer, learning rates, AMP, Check C subset
src/vl/
  config.py                YAML config loading
  fetch_metadata.py        pin repos to commits, download metadata only
  frame_summary.py         per-episode summaries of trajectory parquet (remote column reads)
  manifest.py              episode manifest + label/endpoint precedence rules
  splits.py                group-aware train/val/test assignment
  audit_tables.py          regenerates reports/audit_tables.md
  targets.py               per-frame value targets (return, normalized value, bin)
  fetch_videos.py          download pinned camera videos -> data/videos/
  frames.py                frame index (one row per training example) + exact frame decoding
  check_data.py            integrity of downloaded videos/parquet vs pins and manifest
  dataset.py               PyTorch Dataset/DataLoader (images only as model input), crops
  data_figures.py          crop sheets, Check B, sync checks, review videos
  preview.py               labeled multi-camera contact sheets from remote videos
  operator_visibility.py   person-segmenter audit of operator visibility per camera
  model.py                 ValueModel: shared ResNet-18 over cameras -> 201-bin value head
  train.py                 training loop (Check C tiny-subset overfit)
  train_baseline.py        baseline: full train set, val-CE checkpointing
  metrics.py               validation metrics; re-scores a finished run's saved predictions
  predict.py               per-frame predictions of a run's best checkpoint on train episodes
  value_plots.py           synchronized camera / V-vs-target plots and videos for val episodes
  analysis.py              breakdowns (outcome/kind/repo, timeouts, stalls, operator, shape)
  side_occlusion.py        occlusion check: arm covering side camera vs V (pixel-difference proxy)
  proprio.py               Stage 5 proprio: video<->state timing, 10-d input + train stats, review videos
scripts/*.sbatch           SLURM jobs (Check C, baseline, camera ablation, predict)
tests/                     synthetic unit tests + integrity checks on the built manifest
reports/                   audit findings, data checks, targets summary, run analyses, figures
data/                      (gitignored) pinned metadata, manifest, targets, videos, runs
```

## Setup and commands

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev,video]"

.venv/bin/python -m vl.fetch_metadata     # pins + metadata  -> data/meta/   (~100 MB)
.venv/bin/python -m vl.frame_summary      # remote summaries -> data/manifest/frame_summary/
.venv/bin/python -m vl.manifest           # -> data/manifest/episodes_unsplit.parquet
.venv/bin/python -m vl.splits             # -> data/manifest/episodes.{parquet,csv}
.venv/bin/python -m vl.audit_tables       # -> reports/audit_tables.md
.venv/bin/python -m vl.targets            # -> data/targets/frame_targets.parquet, reports/targets_summary.md
.venv/bin/python -m vl.fetch_videos --cams wrist_left side_1 --data   # ~11 GB -> data/videos/
.venv/bin/python -m vl.frames             # -> data/frames/frame_index.parquet
.venv/bin/python -m vl.check_data         # -> reports/data_integrity.md (hashes all files, ~10 min)
.venv/bin/python -m vl.dataset --workers 4 --batches 100   # loader throughput + sampled mix
.venv/bin/python -m vl.data_figures       # -> reports/figures/stage3/, data/review_videos/
.venv/bin/python -m vl.model              # parameter count + initialization check on a real batch
sbatch scripts/check_c.sbatch             # Check C on a GPU -> data/runs/check_c/
sbatch scripts/baseline.sbatch            # baseline -> data/runs/baseline/
NAME=side CAMS=side_1 sbatch scripts/cameras.sbatch   # camera ablation, seeds 0-2 -> data/runs/side_s*/
NAME=proprio EXTRA=--proprio DIST=1 sbatch scripts/cameras.sbatch   # images + proprio -> data/runs/proprio_s*/
RUN=<run> sbatch scripts/predict.sbatch   # train-episode predictions (train/val gap)
.venv/bin/python -m vl.metrics --run baseline       # -> data/runs/<run>/metrics.json
.venv/bin/python -m vl.value_plots --run baseline   # -> reports/figures/stage4/, data/review_videos/stage4/
.venv/bin/python -m vl.proprio --sync --per-repo 20   # -> reports/proprio_sync.{md,png}
.venv/bin/python -m vl.proprio --stats --table --videos   # -> data/proprio/{stats.json,frame_proprio.parquet}, review videos
.venv/bin/python -m pytest -q
.venv/bin/python -m vl.preview --repo real-marker-d2-r04-eval --episodes 0 1   # needs ffmpeg w/ AV1
```

Only `vl.fetch_videos` downloads videos. The manifest (`data/manifest/episodes.parquet`) is the single
source of truth for episode identity, outcome, valid endpoint `T`, group, and split.

## Status

The checklist is in `PROGRESS.md`; the current stage and next steps are in
`reports/dataset_audit.md`. Planned progression: audit → target construction + tests →
tiny-subset overfit → baseline (shared ImageNet ResNet-18 over cameras → MLP → 201 bins,
cross-entropy) → validation-based checkpointing and breakdowns vs a trivial baseline →
video/value plots → a few ablations (1 vs 2 cameras, then proprioception) → freeze → test →
advantage-label sanity check.

## Environment notes

- The full dataset is ~27 GB; the training cameras alone are ~12 GB. Keep data on storage with
  enough quota; see `PROGRESS.md` Stage 0 for the current location.
- HF repos are public; no token needed.
- GPU jobs are submitted via SLURM (`scripts/*.sbatch`); adjust partition and paths for your
  cluster.
