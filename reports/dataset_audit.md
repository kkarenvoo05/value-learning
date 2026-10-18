# Dataset audit — Mulligan Insert Marker (`real-marker-d2`)

Date: 2026-10-17. All numbers below come from `data/manifest/episodes.parquet` and
`reports/audit_tables.md` (regenerate with the commands in the README). Revisions: every repo
pinned to its `release-1` tag commit (= `main` at audit time), listed in `audit_tables.md`.

Legend: **[paper]** stated in a paper/release doc · **[data]** verified in the data ·
**[ours]** our choice · **[Lars]** decided by Lars · **[open]** unresolved.

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
  Table 17]; RECAP counts running out of time as failure [paper π*0.6 §VI]. → **Q1 decided: timeout =
  failure, as in RECAP [Lars, 2026-10-17].**

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
- **Q2 decided: `wrist_left + side_1` [Lars].** Lars's meeting notes: of the 4 cameras only 2 are
  used for policies, the left wrist view and the side-on view; `side_1` is that side view. Camera
  set and proprioception remain ablations (PROGRESS Stage 5).
- Crop check (f=0, initial-state extremes, `figures/initial_state_extremes.jpg`): marker and
  holder stay inside a standard 224 center crop for `side_1` and `wrist_left`, but in `side_1`
  the marker is ~15×5 px at 640×480 (~5×2 px at 224).
- Mid-episode crop check (16 episodes, Section 5a, yellow box = Resize(256)+CenterCrop(224),
  i.e. x 110–530, y 30–450 of 640×480): the marker, gripper and holder opening stay inside the
  box at the insertion moment and at T in all 16. In `wrist_left` the holder's upper part
  leaves the top edge during the approach (it is ~2× marker length tall), and in `side_1` the
  whole scene of interest is a small region in the upper middle. A center crop is adequate for
  the baseline; a tighter `side_1` crop around the table is a possible Stage 3 improvement. [data]
- **The operator is also visible in `wrist_left`, not only `side_2`.** A hand/forearm appears at
  the bottom-left edge of the wrist view (partly inside the crop) during *autonomous* r03
  rollouts (eps 5, 34, 35, 50, all with zero human frames) and an arm crosses the top-left in
  r04/132. Shortcut risk: the operator's posture may correlate with outcome (e.g. reaching in
  before a reset). [data]
  **Measured 2026-10-18** (`reports/operator_visibility.md`): visible in ~20% of valid
  frames (21–24% after correcting with hand labels), 13% inside the crop, in 58% of
  episodes, in every repo. It follows the task phase (mostly while the arm is at the holder),
  not the outcome: success and timeout differ by ≤5 points at matched time. [data]

## 5a. Video review (Stage 1, 2026-10-17)

16 episodes rendered as 10-column filmstrips of `wrist_left`, `side_1`, `side_2` (evenly spaced
frames 0..T, plus the first human frame and the last raw frame after T):
`.venv/bin/python -m vl.preview --repo <repo> --episodes <eps> --cols 10 --crop-box --cams
wrist_left side_1 side_2 --out reports/figures/stage1_review`. Ten were drawn with
`random_state=0` from the training split, one per category; six are the audit's odd cases.
Reviewed by Claude from still frames, **not yet watched as video by a person**.

| episode | category | label, T | what the frames show |
|---|---|---|---|
| c00/244 | teleop success | success, 9.5 s | grasp → seat; seated at T |
| r04/135 | autonomous success | success, 16.3 s | seated and released at T |
| r04/132 | autonomous failure | failure, 5.9 s | marker slipped, lies by the holder at T |
| r01/70 | autonomous failure | failure, 5.9 s | grasp never closes on the marker (pregrasp misalignment) |
| r00/53 | timeout, soft-truncated | timeout, 28.2 s | marker held at holder opening from 10.6 s, never seats |
| r00/23 | timeout, *neither capped nor soft-truncated* | timeout, 19.0 s | same stall; recording just ends (46 such train timeouts) |
| r02/213 | timeout at 800-step cap | timeout, 53.3 s | static stall at the holder from ~13 s to 53 s |
| c05/94 | intervention → success | success, 14.7 s | robot stalls at holder; ~0.8 s human nudge at 10.9 s; seated |
| c02/113 | intervention → failure | failure, 8.9 s | human takes over at 8.3 s; marker falls beside holder |
| c04/23 | counterfactual all-human demo | success, 10.5 s | clean human demo, seated at T |
| c05/25 | collection success, no human | success, 13.9 s | clean autonomous seat |
| c02/82 | odd: 7.3 s "timeout" | timeout, 7.3 s | grasp, human takeover at 4.6 s, marker never brought to holder; looks aborted, not out of time |
| r03/5, 34, 50 | odd: over the 800 cap | timeout, 55–56 s | ~40 s static stall at the holder opening |
| r03/35 | odd: over the cap | success, 55.0 s | same ~40 s stall, then seats at 55 s |

Findings [data]:
- **Labels and endpoints agree with the frames in all 16.** Successes are seated at T; failures
  show the marker dropped or never grasped at T; timeouts show a stall. Takeover frames line up
  across all three cameras (sync holds).
- **Frames after T are a different scene**: the last raw frame often already shows the marker
  back on the table for the next trial (c00/244 within 0.8 s of T). Confirms never sampling > T.
- **Stalls dominate timeouts and look like near-successes.** The robot holds the marker at the
  holder opening for tens of seconds with an almost unchanged image. Two consequences for V:
  (1) during a stall the target rises by up to ~0.35 while the image barely changes, so an
  image-only model *cannot* fit those frames (irreducible error, not a bug);
  (2) the same stalled image appears in timeouts (target ≈ −0.9…−0.5) and in r03/35, which
  eventually succeeds (≈ −0.46…0). Expect V to be uncertain/bimodal at the holder.
- c02/82 is better described as an aborted trial than a timeout; under timeout = failure the
  target is identical, so it stays as labeled.
- The four over-cap r03 episodes are genuine (≈55 s of real activity, no reset inside the valid
  range); they fit inside H = 60 s and stay in.

## 6. Synchronization

- Every episode: contiguous `frame_index`, `n_rows == length`, timestamps = frame_index/15.
- Every camera's `[from, to]` span equals the frame count (one exception above); ffprobe on
  sampled per-episode files confirms frame counts equal row counts exactly.
- Contact sheets sample the same frame index across cameras; events (grasp, takeover, seating)
  appear simultaneously in all views (`reports/figures/`, regenerate with `vl.preview`).

## 7. Storage

All 13 repos: 27.4 GB. Training pool `side_1 + wrist_left + data`: ~11.9 GB; test: ~3.3 GB.
The original laptop disk had ~2.6 GB free, so nothing was bulk-downloaded. Work has since moved
to the move2 machine. **[Lars]** Training data goes on NFS storage under Karen's name, not under
`/move/`. **Temporary (2026-10-17):** `data/` → `/sailhome/karenvo/vl-data` (home, 20 GB quota,
~19 GB free). Permanent: `/iris/u/karenvo`, pending `finnlab` access. Pre-decoding only valid
frames at reduced resolution/rate would also cut the footprint.

## 8. Target construction (frozen 2026-10-17; implemented in `src/vl/targets.py`)

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
- **Timeouts — decided [Lars]:** failure terminal at T (RECAP-style Monte-Carlo target), same
  −C_fail as other failures. `outcome` stays 3-valued in the manifest so timeouts can still be
  broken out in evaluation; excluding them is an optional diagnostic only (timeouts are 32% of
  eval rollouts and are censored by the Mulligan critic).
- Unknown-outcome episodes are excluded (4). Outcomes, T, and remaining time are label inputs
  only, never model inputs.
- Bin assignment **[ours]**: hard one-hot to the nearest of 201 centers spaced evenly on [−1, 0]
  inclusive, so v = 0 and v = −1 are exact centers. Out of support = more than half a bin
  (0.3 s) beyond an edge → `bin = −1`, counted in `reports/targets_summary.md`.
- Tested on hand-computable synthetic episodes (`tests/test_targets.py`).

### Result on the real manifest (`python -m vl.targets`, 2026-10-17) [data]
- Manifest rebuilt on move2 from the same 15 pinned commits (all SHAs identical to the original
  audit); split counts identical (train 1,377 / val 247 / test 150 + 200). Only diff in
  `audit_tables.md`: two tied rows swapped order.
- 575,406 target frames from 1,970 episodes (1,974 in split − 4 unknown). **0 out of support.**
  Independently recomputed first/last value and length of every episode from the manifest:
  exact match; no frame index > T. Table: `reports/targets_summary.md`; example curves:
  `reports/figures/target_examples.png`.
- Train frames: 62.4% success, 27.3% timeout, 10.4% failure → 37.6% of train frames are ≤ −0.5.
  Timeouts contribute 2.6× the frames of failures because they are long.
- Train bins used: successes 111–200, failures/timeouts 7–100 (184 of 201 bins occupied). The
  lowest train value, −0.966, is r03/50 (56 s timeout).

## 9. Remaining uncertainties / next actions

1. ~~Q1 timeout semantics, Q2 camera choice~~ — decided by Lars (timeout = failure;
   `wrist_left + side_1`).
2. ~~Inspect c02/82 and the four over-cap r03 episodes~~ (Section 5a; still frames, a person
   should watch them as video).
3. ~~Mid-episode crop check~~ (center crop adequate; final crop decided in Stage 3).
4. Permanent NFS storage (`/iris/u/karenvo`); temporary home-directory storage in use.
5. ~~Implement, test and run target construction~~ (done). Next stage: data download, loader,
   alignment check (Check B), model, tiny-subset overfit (Check C).
6. Open for review: stall frames are unfittable from images alone (Section 5a). Operator in
   `wrist_left` is measured (Section 5, `operator_visibility.md`); recommendation is no
   masking for the baseline and a visible/not-visible breakdown in Stage 4 (decision pending).
7. `vl.preview` frames were one frame late until 2026-10-18 (fixed; filmstrips regenerated).
