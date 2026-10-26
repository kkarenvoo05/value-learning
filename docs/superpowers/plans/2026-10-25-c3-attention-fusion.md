# C3 Attention Fusion (cameras only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap C1's "concatenate the two camera features → MLP" for a small attention readout that picks, per frame, how much to rely on each camera. Train it with 3 seeds and compare it to C1 and R0.

**Architecture:** Each camera's backbone feature (512-d, per-camera ResNet-18s as in C1) is projected to a 256-d token and gets a learned camera embedding. A learned readout token runs 2 pre-norm cross-attention blocks over the camera tokens (the queries come only from the readout, so the camera tokens are never updated). The readout then goes to a 256 → 201 MLP. The whole thing replaces `model.head` and keeps its signature (`head(f)` with `f` = concatenated features in camera order). The optimizer group (head LR 1e-4), `model.features`, checkpointing and `history_feats` need no change. Time embeddings and a padding mask are built into the head now (`n_time`, `key_padding_mask`). The history variant is not wired up in this plan.

**Tech Stack:** PyTorch (`nn.MultiheadAttention`), the existing `vl.model` / `vl.train_baseline` / `scripts/cameras.sbatch` / `scripts/compare_arms.py`.

**Spec:** `PROGRESS.md` §5.5.4, items C3 (cameras-only variant) and C4 (attention logging only; plots come later).

## Global Constraints

- One change vs the reference. The primary reference is **C1 `c1_percam_s{0,1,2}`** (same per-camera backbones, concat head). R0 `g0_r0_s{0,1,2}` is secondary.
- Same recipe as C1: `SECTION=baseline_lowlr_aug_full`, seeds 0 1 2, `DIST=1 DROP_LAST=1`, `--gres=gpu:rtxpro6000:1`, flags `--per-camera-backbone --fusion attention`.
- Win rule: `scripts/compare_arms.py` paired bootstrap. The val CE interval must lie entirely below 0, and shape MAE and AUC 1 s must not be credibly worse.
- **No git commits.** The repo has had uncommitted work since `2df9f76`, and another session is editing `model.py` / `dataset.py` / `train_baseline.py` (still-time upper bound, targeted metric). Re-read each file right before editing it, and don't touch the `still_time` code or `compare_arms.py`. Record code as a diff: `data/runs/c3_code_vs_2df9f76.diff`.
- Tag design choices `[ours]` / `[Lars]` in docstrings and PROGRESS. Only report what actually ran.
- Tests: `.venv/bin/python -m pytest -q`. Everything that passed before must still pass.

## Review Focus

1. **Camera identity.** Without the camera embedding, attention is order-blind, so swapping wrist and side features would give the same output. → test in Task 1.
2. **Fully masked rows** (future history variant: if every token were masked, softmax would give NaN). The current-frame tokens must never be maskable. → forward raises on a fully masked row, test in Task 1.
3. **Old runs and the concat path stay byte-identical.** Runs without `fusion` in config.json must load as `concat`, and the concat model's state_dict keys must not change. → tests in Task 2.
4. **Unsupported combinations fail loudly.** Attention + history / proprio / still_time is not built yet, so it must raise, not silently ignore the extra input. → test in Task 2.
5. **AMP dtype.** Attention under bf16/fp16 autocast must give finite logits, and the attention weights must still sum to 1. → covered by the GPU smoke in Task 4 (finite loss, attention-dump sums checked in Task 3).

---

### Task 1: `TokenFusionHead`

**Files:**
- Modify: `src/vl/model.py` (new class after `layer_decay_groups`, plus a docstring bullet "Attention fusion (Stage 5.5 C3)")
- Test: `tests/test_model.py` (new section at the end)

**Interfaces:**
- Produces: `TokenFusionHead(n_tokens_per_time: int, feat: int, n_bins: int, n_time: int = 1, d: int = 256, n_heads: int = 4, n_layers: int = 2, ff: int = 512)`.
  `forward(f: Tensor (B, n_time * n_tokens_per_time * feat), key_padding_mask: BoolTensor (B, n_time * n_tokens_per_time) | None = None, return_attn: bool = False) -> logits (B, n_bins)` or `(logits, attn)`, with `attn: (B, n_layers, n_tokens)` = the readout's attention averaged over heads.
  Token order: time-major, cameras in model order within each time step (`f` is laid out the same way: `[now cams..., past cams...]`).
  Learned params: `proj` Linear(feat, d), `cam_emb` (n_tokens_per_time, d), `time_emb` (n_time, d), `readout` (1, 1, d), all embeddings init N(0, 0.02). Each block: LN → MHA(query = readout, key/value = LN(tokens)) → residual, then LN → FF(d→ff→d, GELU) → residual, dropout 0. Output: LN → Linear(d, 256) → ReLU → Linear(256, n_bins) [ours; same 256-hidden head shape as concat].

- [ ] **Step 1: Write the failing tests** (`fusion_head()` helper: `TokenFusionHead(2, 512, 201).eval()`, `torch.manual_seed(0)`)
  - `test_fusion_head_output_shape`: input (4, 1024) → logits (4, 201).
  - `test_fusion_attention_is_a_distribution_over_image_tokens`: `return_attn=True` → attn shape (4, 2, 2), `attn.sum(-1)` ≈ 1 (atol 1e-5), all ≥ 0.
  - `test_fusion_head_knows_which_camera_is_which`: swapping the two 512 halves of `f` changes the logits (`not allclose`).
  - `test_fusion_attention_depends_on_the_frame`: two different random `f` give different attn (it is not a fixed weighting).
  - `test_fusion_masked_token_is_ignored`: `TokenFusionHead(2, 512, 201, n_time=2)`, mask the 2 past tokens; changing the past half of `f` leaves logits unchanged (allclose) and the masked attn entries are 0.
  - `test_fusion_fully_masked_row_is_refused`: all-True mask on one row → `ValueError`.
  - `test_fusion_gradients_reach_every_parameter`: one backward from `logits.sum()`, every parameter's `.grad` is not None and nonzero.

- [ ] **Step 2: Run and see them fail**
  Run: `.venv/bin/python -m pytest -q tests/test_model.py -k fusion`
  Expected: FAIL (ImportError: `TokenFusionHead`)

- [ ] **Step 3: Implement `TokenFusionHead` in `src/vl/model.py`**
  Use `nn.MultiheadAttention(d, n_heads, batch_first=True)` with `need_weights=True, average_attn_weights=True`. Keep the per-layer weights only when `return_attn` is set.

- [ ] **Step 4: Run and see them pass**
  Run: `.venv/bin/python -m pytest -q tests/test_model.py`
  Expected: all pass

### Task 2: Wire `fusion` through the model, config and training

**Files:**
- Modify: `src/vl/model.py` (`ValueModel.__init__`, `build_model`, `run_data_config`, module docstring)
- Modify: `src/vl/train_baseline.py` (argparse, `dcfg`, `meta`, usage line at the top)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `TokenFusionHead` (Task 1).
- Produces: `ValueModel(..., fusion: str = "concat")` with `fusion in ("concat", "attention")`, stored as `self.fusion`. For `"attention"`, `self.head = TokenFusionHead(len(cams), feat, n_bins)`. `build_model` reads `data_cfg.get("fusion", "concat")`. `run_data_config` returns `fusion=meta.get("fusion", "concat")`. CLI: `vl.train_baseline --fusion {concat,attention}` (default concat), and `meta["fusion"]` is written to config.json.

- [ ] **Step 1: Write the failing tests**
  - `test_attention_fusion_model_uses_the_token_head`: `ValueModel(CAMS, pretrained=False, per_camera_backbone=True, fusion="attention")`, `isinstance(m.head, TokenFusionHead)`, uint8 batch → logits (B, 201).
  - `test_attention_fusion_each_camera_changes_the_output` (parametrize over cams, same pattern as `test_changing_one_camera_changes_the_output`).
  - `test_attention_fusion_fits_one_batch`: same pattern as `test_a_few_steps_fit_one_batch`, but the loss must decrease over the steps.
  - `test_attention_fusion_refuses_unbuilt_inputs`: `fusion="attention"` with `history=True`, with `still_time=True`, and with proprio mean/std each → `ValueError`. Unknown `fusion="sum"` → `ValueError`.
  - `test_run_data_config_reads_fusion` (tmp_path): config.json `{"fusion": "attention"}` → `"attention"`; `{}` → `"concat"`.
  - The existing `test_image_only_state_dict_unchanged` must still pass untouched (concat keys unchanged).

- [ ] **Step 2: Run and see them fail**
  Run: `.venv/bin/python -m pytest -q tests/test_model.py -k "attention_fusion or reads_fusion"`
  Expected: FAIL (unexpected keyword `fusion`)

- [ ] **Step 3: Implement.** Keep the `n_img` / `nn.Sequential` concat path exactly as it is. Add `--fusion` to `train_baseline.py` next to `--per-camera-backbone`.

- [ ] **Step 4: Run the full suite**
  Run: `.venv/bin/python -m pytest -q`
  Expected: all pass, including every test that passed before

### Task 3: Readout attention dump (for C4)

**Files:**
- Create: `src/vl/attention.py`
- Test: `tests/test_model.py` (or `tests/test_attention.py` if it needs dataset fixtures)

**Interfaces:**
- Consumes: `run_data_config`, `build_model`, `FrameDataset`, `load_split`, `model_inputs`, `TokenFusionHead.forward(..., return_attn=True)`.
- Produces: `.venv/bin/python -m vl.attention --run <run>` → `data/runs/<run>/val_attn.parquet`, one row per val frame, in the same row order and with the same identifying columns as `val_predictions.parquet`, plus `attn_<cam>_l<layer>` (floats). It refuses a run whose `fusion != "attention"`.
  Add a pure helper `attention_table(attn: np.ndarray (N, L, T), cams: list[str]) -> pd.DataFrame` with columns `attn_<cam>_l<l>`.

- [ ] **Step 1: Write the failing test** `test_attention_table_columns_and_sums`: random attn normalized over T, cams `["wrist_left", "side_1"]`, L = 2 → columns `attn_wrist_left_l0, attn_side_1_l0, attn_wrist_left_l1, attn_side_1_l1`, and each layer's pair sums to 1.
- [ ] **Step 2: Run, see it fail; implement; run, see it pass.** The CLI loads `best.pt` the way `vl.value_dist` does (eval mode, no augmentation, same eval batch size).

### Task 4: Smoke, diff, submit

- [ ] **Step 1: CPU smoke**
  Run: `.venv/bin/python -u -m vl.train_baseline --run c3_smoke --section baseline_lowlr_aug_full --per-camera-backbone --fusion attention --steps 3 --val-limit 2`
  Expected: finite train loss, `best.pt` + `result.json` written. Then `run_data_config(…, data/runs/c3_smoke)["fusion"] == "attention"`, and the checkpoint reloads into `build_model` with `strict=True`. Run `vl.attention --run c3_smoke` and check every layer's attention sums to 1. Record the parameter count (expected ≈ 22.35M backbone + ≈ 1.3M head, vs C1's 0.31M head) and delete `data/runs/c3_smoke`.
- [ ] **Step 2: Record the code**
  Run: `git diff 2df9f76 -- src/vl/model.py src/vl/train_baseline.py src/vl/attention.py tests/test_model.py > data/runs/c3_code_vs_2df9f76.diff` (`src/vl/attention.py` is untracked; append it with `git diff --no-index /dev/null src/vl/attention.py >> …`). Note in PROGRESS that the diff also contains the other session's concurrent changes.
- [ ] **Step 3: Submit (3 seeds, one job, on the same GPU type as C1/R0)**
  Run: `NAME=c3_attn SECTION=baseline_lowlr_aug_full DIST=1 DROP_LAST=1 EXTRA="--per-camera-backbone --fusion attention" sbatch --gres=gpu:rtxpro6000:1 scripts/cameras.sbatch`
  Expected: one job id, ~25 min per seed (C1 took ~22). Check that `config.json` of `c3_attn_s0` says device RTX PRO 6000 and `"fusion": "attention"`.
- [ ] **Step 4: PROGRESS.md §5.5.4 C3.** Add a "Code (2026-10-25)" entry (design, [ours]/[Lars] tags, param count, tests added, smoke numbers, diff path) and a "Runs submitted" entry (command, job id).

### Task 5: Result

- [ ] **Step 1:** `.venv/bin/python scripts/compare_arms.py c1_percam c3_attn` (primary) and `.venv/bin/python scripts/compare_arms.py g0_r0 c3_attn`.
- [ ] **Step 2:** `.venv/bin/python -m vl.attention --run c3_attn_s{0,1,2}`, then one summary table: mean wrist attention (last layer) by hold phase vs approach, success vs failure. C4 plots proper come later.
- [ ] **Step 3:** If the other session's targeted hold-phase metric has landed in `compare_arms.py`, report it too. Don't build it here.
- [ ] **Step 4:** Add a result block under C3 in PROGRESS: table (same columns as the C1 table), paired intervals, verdict by the win rule, attention summary, and a reading marked "(Claude, to be checked)". Leave C3 and Gate C unticked for Karen.
