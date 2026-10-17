# value-learning

Does a small RECAP-style **state-value function V(o)** learn useful task progress from the
Mulligan **Insert Marker** real-robot dataset? Karen Vo with Lars Ankile (Stanford IRIS).

If V turns out useful, later stages use it for advantage-conditioned π₀.₅ policy improvement and
compare against LWD-style value learning.

## References

- RECAP / π*0.6 — https://arxiv.org/abs/2511.14759 (reward −1/step, failure penalty, values
  normalized per task to (−1, 0), 201-bin categorical value, advantage-conditioned policy)
- Mulligan — https://arxiv.org/abs/2610.05882 (the data source; HG-DAgger + initial-state
  selection; release docs live in each HF repo's README and `meta/` sidecars)
- LWD — https://arxiv.org/abs/2605.00416
- Data — https://huggingface.co/collections/mulligan/real-insert-marker

Read the HF release docs, not only the papers: outcome/endpoint semantics live there.

## Ground rules

- Correctness and understanding beat complexity. Small readable PyTorch, no big frameworks.
- Separate **paper-specified** details from **our choices**; never claim exact reproduction.
- Split comes from Lars (see `configs/dataset.yaml`); validation (~15% of episode *groups*,
  fixed seed) is our choice. Test repos (r05-eval, r05-screen) are for final evaluation only:
  metadata integrity checks now, no tuning on test performance.
- Labels come from reviewed outcomes and endpoints, with unknowns flagged, never turned into
  failures. Machine (VLM) stage labels are diagnostics, not truth.
- Never commit data, caches, checkpoints, videos, or credentials (`.gitignore`).
- Record configs, seeds, dataset commits, commands, and results. Only report what actually ran.

## Layout

```
configs/dataset.yaml       repos, split, validation seed, metadata patterns
src/vl/
  fetch_metadata.py        pin repos to commits, download metadata only
  frame_summary.py         per-episode summaries of trajectory parquet (remote column reads)
  manifest.py              episode manifest + label/endpoint precedence rules
  splits.py                group-aware train/val/test assignment
  audit_tables.py          regenerates reports/audit_tables.md
  preview.py               labeled multi-camera contact sheets from remote videos
tests/                     synthetic unit tests + integrity checks on the built manifest
reports/dataset_audit.md   audit findings, open questions, target-construction proposal
data/                      (gitignored) pinned metadata, summaries, manifest
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
.venv/bin/python -m pytest -q
.venv/bin/python -m vl.preview --repo real-marker-d2-r04-eval --episodes 0 1   # needs ffmpeg w/ AV1
```

Nothing above downloads videos. The manifest (`data/manifest/episodes.parquet`) is the single
source of truth for episode identity, outcome, valid endpoint `T`, group, and split.

## Status

Current stage and next steps are in `reports/dataset_audit.md` (Sections 8–9). Planned
progression: audit → target construction + tests → tiny-subset overfit → baseline
(shared ImageNet ResNet-18 over two cameras → MLP → 201 bins, cross-entropy) → validation-based
checkpointing and breakdowns vs a trivial baseline → video/value plots → a few ablations
(1 vs 2 cameras, then proprioception) → freeze → test → advantage-label sanity check.

## Environment notes

- Developed on an 8 GB Apple M2 laptop whose disk was nearly full (~2–3 GB free). The full
  dataset is ~27 GB; the training cameras alone are ~12 GB. Plan data storage before training.
- HF repos are public; no token needed.
