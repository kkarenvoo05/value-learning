# Dataset audit — Mulligan Insert Marker (`real-marker-d2`)

Date: 2026-10-17. All numbers below come from `data/manifest/episodes.parquet` and
`reports/audit_tables.md` (regenerate with the commands in the README). Revisions: every repo
pinned to its `release-1` tag commit (= `main` at audit time), listed in `audit_tables.md`.

Legend: **[paper]** stated in a paper/release doc · **[data]** verified in the data ·
**[ours]** our choice · **[open]** unresolved.

## 1. Format and schema

- LeRobot `v3.0`, Franka, **15 fps** everywhere, AV1 640×480 videos, one task (`marker_d2`). [data]
- Schema is consistent across the 13 repos except: r00–r02 have only `wrist_left` + `side_1`
  (no `wrist_right`/`side_2`); some repos add extra telemetry columns. [data]
- Episode-level outcomes are **not** in LeRobot's episode table; they live in Mulligan sidecars:
  `meta/episode_provenance.parquet` (eval repos), `.label_history.jsonl`,
  `.outcome_edit_progress.json`, `results.json`/`results_eval_time.json`, and the collection
  ledgers (`meta/protocol_quota_ledger.jsonl`, `meta/teleop_manifest_ledger.jsonl`). [data]
- Multi-session eval repos (r02, r05-screen) keep per-session review records under
  `meta/sessions/bNN/`, keyed by **session-local** `source_episode_index`. [data]

## 2. Outcomes and endpoints

### Precedence rule (implemented in `src/vl/manifest.py`)
1. Eval repos: `episode_provenance.parquet` `outcome` and `num_steps` — "the reviewed values used
   by the paper" [paper: repo README]. Valid final frame **T = num_steps − 1**.
2. Collection repos: latest **human** outcome event in `.label_history.jsonl` — "the latest human
   label wins" [paper: repo README]. T = `outcome_frame`.
3. Human `{"action":"skip"}` → fall back to the auto (collection-results) event, marked
   `auto_after_human_skip`.
4. Nothing usable → `unknown`, excluded from targets (never turned into failure).
- VLM `stage` events (gemini cascade) are carried as `machine_stage`/`machine_failure_mode`
  diagnostics only. [ours, following README: "not human-adjudicated"]

### Verification [data]
- For **all 2,095 known-outcome episodes**: provenance == latest human label (outcome and T);
  per-frame `done` first fires exactly at T for success/failure; for soft-truncated episodes
  `is_valid` covers exactly frames 0..T.
- Reviewed T is **not** the last frame: median 13 frames trimmed in collection repos; in eval repos
  mean 63 and up to 560 frames are recorded after T (robot retracts, stalls, or the operator
  resets). The last row is always `is_valid=0`, and in the c01 sheets it already shows the
  **scene reset for the next trial**, so post-T frames are not even temporally continuous.
- Per-frame `steps_to_go` counts down to the **raw recording end**, not to T → must not be used
  for targets. Per-frame `round_id` is the trial index within a session, not the model round.
- Timestamps equal `frame_index/15` exactly (synthetic), so they cannot reveal wall-clock gaps.

### Outcome classes
- `success`: marker released and fully seated (S7). [paper §C.3]
- `failure`: operator-recorded unrecoverable state. [paper §C.3]
- `timeout`: not success and not judged unrecoverable. Eval cap is 800 steps (53 s) [paper],
  but only 79/253 training-pool eval timeout *recordings* reach it; 158 are **soft-truncated** at a stall
  frame chosen by the reviewer (`soft_truncate=true`, the rest is `is_valid=0`). VLM stages put
  most timeouts at stage 3–4 (at holder / tip engaged), failures spread over stages 0–3 —
  diagnostic only. [data]
- The Mulligan critic gives **timeouts no terminal anchor** (bootstraps through them) [paper
  Table 17]; RECAP counts running out of time as failure [paper π*0.6 §VI]. → **[open] Q1.**

### Anomalies [data]
- 4 training-pool episodes are `unknown` (human `skip`, no auto event): c01/53, c03/60, c03/68,
  c04/42. Excluded. Their collection-time ledger outcomes exist (success, failure, failure,
  timeout) but endpoints were never reviewed.
- 4 r03 episodes have valid length 826–840 > the 800-step cap (r03/5, 34, 35, 50). Kept, flagged
  in `audit_tables.md`; they matter for the normalization constant.
- c02/82 is labeled `timeout` after only 7.3 s with human frames, in an uncapped collection
  repo — worth a look.
- r00/35 `wrist_left` video is one frame short; only the post-T last frame is missing (T covered).

## 3. Counts

| split | episodes | groups | failure | success | timeout | unknown | valid hours |
|---|---|---|---|---|---|---|---|
| train | 1377 | 748 | 201 | 958 | 214 | 4 | 6.92 |
| val | 247 | 130 | 35 | 172 | 40 | 0 | 1.20 |
| test_r05_eval | 150 | 50 | 25 | 91 | 34 | 0 | 1.01 |
| test_r05_screen | 200 | 50 | 24 | 119 | 57 | 0 | 1.52 |

Data sources in the training pool: 250 teleop (c00, all success), 574 HG-DAgger collection
(c01–c05: 558 success; 444 with human frames; 59 counterfactual all-human demos), 800 autonomous
eval rollouts (r00–r04: **zero** human frames or interventions). Full per-repo tables, durations,
and val shares (11.8–20.3% per repo) are in `audit_tables.md`.

## 4. Overlap and grouping

Groups = connected components over shared initial state (5-D, 1e-4 quantized), identical content
fingerprint, or identical (release block, episode) identity. [ours]

- **Structural cross-repo links** (by design of the Mulligan protocol): each collection increment
  c(N+1) re-collects at the failed starts of eval round r(N) ("promoted failures"): c01↔r00,
  c02↔r01, c03↔r02, c04↔r03, c05↔r04; c00's 50 `eval_heldout` teleop demos sit on the r00 starts.
  Counterfactual demos and retries share starts within a c-repo. Splitting by episode would leak.
- Groups have 1–7 episodes; 545 singletons.
- **Train pool vs test: 0 shared starts, 0 content duplicates, 0 identity collisions.** [data]
- **r05-eval and r05-screen share the same 50 starts** (same Sobol seed, start indices 0–49)
  [paper README + data]. Both are test; they are not independent samples of starts.
- Audit-only repos: `c00-teleop-validation` = content-identical copy of c00's 50 eval_heldout
  episodes; `r02-eval-b03` = the raw b03 session (100/125 episodes content-identical to r02-eval
  b03; the other 25 share its starts). Neither is in the split.
- Bug found and fixed during the audit: r02 and r05-screen sessions reuse one `source_repo`
  string with session-local indices; keying identity on `source_repo` falsely merged 100 pairs of
  distinct recordings. Identity now uses the release block.
- Release metadata marks eval rounds `trainable: false` for *their* models; Lars's split
  intentionally uses r00–r04 as value-model training data. [paper README + lars]

## 5. Cameras

- Paper: Insert Marker policies/critics use **2 cameras**, one ResNet-18 each, 224×224 after a
  fixed per-camera crop; hardware ZED stereo. [paper Table 16/21]
- `wrist_left`/`wrist_right` are the two lenses of the same wrist ZED (near-identical images).
- `side_1` and `side_2` are different viewpoints. `side_1`: clean third-person table view.
  `side_2`: opposite view that always shows the seated operator and lab clutter, and during
  human control their hands/body. r05-screen b02 recorded black `side_2` frames (eps 0–37)
  [paper README].
- **`side_1` is the only side camera in r00–r02** (500 of 1,624 training-pool episodes), so it is
  almost certainly the paper's second camera.
- Recommendation: `wrist_left + side_1`. **[open] Q2** — confirm with Lars.
- Crop check (f=0, initial-state extremes, `figures/initial_state_extremes.jpg`): marker and
  holder stay inside a standard 224 center crop for `side_1` and `wrist_left`, but in `side_1`
  the marker is ~15×5 px at 640×480 (~5×2 px at 224). Crop choice must also be checked on
  mid-episode frames (the gripper rises to the top edge during insertion).

## 6. Synchronization

- Every episode: contiguous `frame_index`, `n_rows == length`, timestamps = frame_index/15.
- Every camera's `[from, to]` span equals the frame count (one exception above); ffprobe on
  sampled per-episode files confirms frame counts equal row counts exactly.
- Contact sheets sample the same frame index across cameras; events (grasp, takeover, seating)
  appear simultaneously in all views (`reports/figures/`, regenerate with `vl.preview`).

## 7. Storage

All 13 repos: 27.4 GB. Training pool `side_1 + wrist_left + data`: ~11.9 GB; test: ~3.3 GB.
The local disk had ~2.6 GB free, so nothing was bulk-downloaded. Options: train elsewhere, free
local space, or pre-decode only valid frames at reduced resolution/rate.

## 8. Recommended target construction (to be frozen before training)

Per episode with valid final frame T (0-indexed), for t = 0..T, in **seconds** (Δ = 1/15 s):

    R_t = −(T − t)·Δ − C_fail·1[failure]

- No per-episode normalization; one task-wide constant. Proposed **[ours]**: horizon H = 60 s
  (above the longest observed valid episode, 56.0 s / 840 frames) and **C_fail = H = 60 s**, so
  `v = R / (H + C_fail) = R / 120 s`, giving successes in (−0.5, 0] and failures in [−1, −0.5].
- Support: 201 uniform bins on [−1, 0] (bin width 0.6 s), as in RECAP's B = 201 [paper]. Count
  and report any target outside [−1, 0]; never clip silently.
- Terminal: the frame at T gets R_T = 0 (success) or −C_fail (failure); frames > T are never
  sampled. Subsampling keeps each frame's original index t, so targets use true time.
- Units in seconds keep later advantage lookahead defined in time, not frames.
- **Timeouts — [open] Q1.** Proposed default: treat as failure terminal at T (RECAP-style,
  simplest Monte-Carlo target), keep `outcome` 3-valued in the manifest, and run the
  "exclude/censor timeouts" variant as an early sensitivity check, since that is how the Mulligan
  critic treats them and timeouts are 32% of eval rollouts.
- Unknown-outcome episodes are excluded (4). Outcomes, T, and remaining time are label inputs
  only, never model inputs.
- Test on hand-computable synthetic episodes before training.

## 9. Remaining uncertainties / next actions

1. Q1 timeout semantics (labels), Q2 camera choice — ask Lars.
2. Inspect c02/82 and the four over-cap r03 episodes by video.
3. Decide the per-camera crop from mid-episode frames of both cameras.
4. Choose where training data will live (disk constraint), then download pinned `side_1`,
   `wrist_left`, and data for the training pool only.
5. Implement and test target construction (Section 8), then a tiny-subset overfitting check.
