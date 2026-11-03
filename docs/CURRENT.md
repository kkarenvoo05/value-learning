# Current state

**Updated 2026-11-03 09:50 PST.** Overwrite this file at the end of every session; do not append. History lives
in `docs/libero_plan.md` (LIBERO) and `PROGRESS.md` (Mulligan stages). Where to find things: `docs/INDEX.md`.

## Running and queued jobs

GPU cap: **≤ 4 GPUs including any interactive session.** A new interactive session should be CPU-only until
18285014 ends.

| job | what | state | depends on | output |
|---|---|---|---|---|
| 18274106 | E2 frozen B + subtask, seed 2 | running (move3) | — | `runs/value_expert_fixedfresh_w256_sub01_s2_18274106` |
| 18275463 → 64–66 | π₀.₅ out-of-fold folds (other session) | running / chained | — | see plan 2026-11-02 23:00 |
| 18269064 | arm A policy fine-tune (other session) | pending, priority (est. 11:30) | — | — |
| 18285014 | LoRA + subtask **eval-only** from ckpt_5400 (96 GB) | pending | afterany 18277015 | `runs/value_lora_fixedfresh_lr1e4_sub01_s0_18274115` |
| 18279695 | P1c per-layer hidden states h0–h18 | pending | afterany 18285014 | `/move/u/karenvo/libero_features/p1c_hidden_18279695/` |

(`runs/` = `/iris/u/ankile/karenvo/libero/runs/`; logs `/iris/u/ankile/karenvo/libero/logs/slurm-<job>.out`.)

## Latest verdicts

- P1/P1b/P1c (`reports/2026-11-02-frozen-encoders/report.md`): two separate noise sources, each about halving head
  SNR: the frozen LLM layers and SigLIP fine-tuning. Layout variety, not episode count, drives the frozen
  encoders' SNR.
- E2 subtask head: no SNR change (ResNet control; frozen B s0 0.152 → 0.155, s1 0.152 → 0.152).

## Open decision (Karen)

SigLIP-only value path with gentle fine-tuning vs adapting the LLM. Waits on 18285014 (LoRA + subtask) and P1c.

## Resume steps

1. When 18285014 ends: score `val_predictions.parquet` with `score(..., "ev_expert")` (scripts/libero_head_swap.py)
   against LoRA baseline 18258391 (SNR 0.133, milestone +0.052, MAE 0.155). Report SNR, milestone, MAE, MAE succ>20 s,
   AUC 5/10 s, subtask acc. Bar: clearly above 0.133, ideally above frozen B's 0.166.
2. When 18274106 ends: add E2 seed 2 to the table in the plan (baseline s2 18237900: SNR 0.194).
3. When 18279695 ends: `PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze --out
   /move/u/karenvo/libero_features/p1c_hidden_18279695 --fit-subset <fit_fixed.txt | fit_fresh375_s0.txt |
   fit_fixed_fresh.txt> --tag <tag> --encoders llm_h0 llm_h1 llm_h2 llm_h3 llm_h4 llm_h5 llm_h7 llm_h9 llm_h11
   llm_h13 llm_h15 llm_h17 llm_h18 llm_hfinal` (CPU, run as sbatch if the session has a 16 GiB cap). Check
   llm_h0 ≈ siglip_pi05. If h1–h3 fall below h0 by > ~2 sd: `ABLATE=1 sbatch scripts/pi05_layer_hidden.sbatch`
   (Karen pre-approved; stay ≤ 4 GPUs).
4. Update `reports/2026-11-02-frozen-encoders/report.md`, `docs/findings.md`, and this file.

## Traps

- π₀.₅ does not load in a 16 GiB interactive session (OOM). Always sbatch π₀.₅ jobs.
- `/iris/u/ankile` is 99% full: large features go to `/move/u/karenvo/libero_features/`.
- Never `scancel` other sessions' jobs; re-chain only your own.
