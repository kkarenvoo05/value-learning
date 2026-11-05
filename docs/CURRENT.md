# Current state

**Updated 2026-11-04 18:30 PST.** Overwrite this file at the end of every session; do not append. History lives
in `docs/libero_plan.md` (LIBERO) and `PROGRESS.md` (Mulligan stages). Where to find things: `docs/INDEX.md`.
Progress site: https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/index.html. Week 7 =
everything before the 2026-11-03 mentor meeting (frozen). **New work goes on Week 8** (`reports/site/content/week08.json`,
built by `scripts/build_site.py`).

## Running and queued jobs

GPU cap (Karen, 2026-11-04): **≤ 3 batch GPUs on top of the interactive session.** Karen counts jobs as they
appear in `squeue`, so keep CPU jobs chained too (one running at a time). TITANs (move1/move2): π₀.₅/JAX policy
fine-tunes OOM at 24 GB; layer probes need ≥ 48 GB (exclude move1–3).

| job | what | state | output |
|---|---|---|---|
| **lane 1** 18310224 → 18310225 | oracle-tag fine-tune (tag = rollout outcome) → its tag-usage check | running | `runs/policy_armC_oracle_s0_18310224`, `runs/tag_usage_18310225` |
| lane 1 → 18310775 → 18310776 → 18310777 | recipe R3: arm A original recipe, `--save-every 1000` → evals at step 1000, 3000 (does the drop grow with training?) | queued | `runs/policy_armA_ckpts_s0_18310775`, `runs/eval_armA_ckpt{1000,3000}_*` |
| lane 1 → 18309643 → 18309644 | steering pre-check v2, noise scale 1.5 then 2.0 | queued | `runs/steer_v2_scale{1p5,2p0}_libero_10_*` |
| **lane 2** 18315373 → 18315374 | recipe R2: arm A, LoRA lr 1e-5 / rest 2.5e-6 (10× lower), save every 1000 → eval final. **Resumed** on L40S from step 4000 of 18310773 (move5, cancelled 21:42 after saving `trainable_4000`); weights + lr schedule + data position restored, Adam moments restart at 0 | queued | `runs/policy_armA_lowlr_s0_18315373` (steps 1000–4000 in `…_18310773`), `runs/eval_armA_lowlr_*` |
| lane 2 → 18309646 (now afterany:18315374) | A2 N=4 seed 2 | queued | `runs/value_expert_trunc4_fixedfresh_w256_s2_18309646` |
| **lane 3** 18309647 | A3: truncated N=4 + LoRA value expert, seed 0 | running | `runs/value_lora_trunc4_fixedfresh_lr1e4_s0_18309647` |
| lane 3 → 18310771 → 18310772 | recipe R1: arm A, `--freeze-siglip` (new flag; only LoRA + action projections + time MLP train), save every 1000 → eval final | queued | `runs/policy_armA_frozensiglip_s0_18310771`, `runs/eval_armA_frozensiglip_*` |
| lane 3 → 18313821 → 18313822 | **B1** subtask-scoped ResNet V (`vl.libero_resnet --subtask-scoped`), fixed+fresh, seeds 0, 1 (~1.5 h each) | queued | `runs/resnet18_fixedfresh_scoped_s{0,1}_*` |

Recipe bar: a variant "does not hurt" if its eval is within ~5 pts of base 0.564 on the same 330 trials (CI of the
paired difference covering 0). New flags: `pi05_policy_finetune.py --freeze-siglip` (load_finetuned reads it from
results.json), `libero_eval.sbatch FT_STEP=<n>`.

Done today: per-layer analysis 18306065 (375 fixed), A2 rescores 18305874 (N=9) / 18305875 (N=4), per-layer analysis 18305897
(fixed+fresh), π₀.₅ OOF labels 18306000 (CPU). `runs/` = `/iris/u/ankile/karenvo/libero/runs/`;
logs `/iris/u/ankile/karenvo/libero/logs/slurm-<job>.out`.

## Latest verdicts (2026-11-04)

- **R2 low-lr recipe passes the bar**: arm A 0.521 vs base 0.564, −0.042 [−0.115, +0.027] (18315374).
- **Tag predictability** (`reports/2026-11-05-tag-predictability/`): the action carries 3.2% of the current tag's
  information beyond the state (≈ the ignored oracle's 2.3%); RECAP's 30–40% positive → 7.4–8.1%, + value-binned
  calibration → 8.5%. **Value-binned calibration** (`reports/2026-11-05-value-calibration/`): κ +0.05–0.06, no
  subtask offset in our A. Candidate next policy run [to decide]: R2 recipe + value-binned 40% positive + 30% tag
  dropout + moka demos tagged positive, eval β = 1 / 1.5 / 2.5.

- **Literature review** (`reports/2026-11-05-literature-advantage-conditioning/`): RECAP uses ~30–40% positive,
  30% tag dropout, demos tagged positive, and learns the tag in pre-training; **arm C was 70% positive, no dropout,
  no demos**. Theory (RCSL): a tag not predictable from the action is ignored. Dissecting (2609.28161): value-binned
  calibration + continuous weights work best on 4 real tasks. Open decision: offline tag-informativeness test +
  calibration first, then a RECAP-faithful tag run and/or a weighting run.

- **Label denoising** (`reports/2026-11-05-label-denoise/`): current labels agree across ResNet / expert at κ 0.20.
  Smoothing V (k = 9) is the only remedy that lowers noise and keeps timing (flips 0.96 → 0.34/s); n = 80/160 lifts κ
  to 0.39/0.54 but smears credit 4/8 s before events; margins filter, do not denoise. Open with the rule decision.

- **Label rules** (`reports/2026-11-04-label-rules/`, no training): dropping the time cost (R3) = current rule (constant
  shift under a percentile). A > 0 (R1) / top 30% (R2) turn stalls negative (78–97% vs 26–38% now), but ordinary
  progress frames carry A noise ±0.04 around 0, so R1 labels them ~54/46. **Open decision (Karen):** which rule, or
  a noise-aware variant (smoothed V, longer n, margin / no-tag middle), for the next policy run. Viewer has a rule
  selector: `reports/2026-11-04-advantage-examples/viewer/`.

- **Advantage examples** (`reports/2026-11-04-advantage-examples/`): the ResNet's 201-bin V is bimodal (success /
  failure bands); A tracks the 20-step (1 s) change in its P(success) (r 0.89–1.00 on 5 episodes). Labels before a pot falls
  76–92% negative, before a successful first placement 19% (base 30%); expert near base at both. Exceptions:
  placement labelled negative (ep 139), band flicker (ep 87). Flat stretches are positive by default. The oracle
  tag (18310225, ignored: tag/noise 0.021) was per-episode, so it did not test per-frame conditioning. Tag-usage
  CI bug fixed (`boot_ratio_ci`). Karen to review before choosing the better tag tests.

- **A3 truncated N=4 + LoRA** (18309647, seed 0): SNR **0.169** / MAE 0.166 / AUC10 0.837 / milestone 0.007. Between
  LoRA-full (0.133 / 0.155) and frozen N=4 (0.180–0.182 / 0.169–0.170): truncation lifts LoRA's steadiness, LoRA
  gives back some of it for accuracy. Not steadier than frozen N=4; still far below ResNet 0.247. Milestone gone.
- **GPU placement (Karen, 2026-11-04 20:20): no RTX PRO 6000 (move5).** π₀.₅ jobs pinned to `gpu:l40s:1` (24 GB
  cards OOM); ResNet B1 jobs `gpu:1` excluding move5 (L40S / A5000 / TITAN). 18310224 and 18310773 were already
  running on move5 when this was asked. 18310224 finished 20:45; 18310773 was let run to step 4000 (Karen) and
  resumed on L40S as 18315373 (eval 18315374; 18309646 re-chained). Nothing on move5 now.

- **A2 N=4 seed 1** (18309645): SNR 0.182 / MAE 0.170, replicates seed 0 (0.180 / 0.169); seed 2 queued.

- **Fine-tuning hurts the policy** (`reports/2026-11-04-base-vs-finetuned/`, 18309278): base π₀.₅ 186/330 =
  **0.564** on the arms' exact trials; success-only −0.158 [−0.224, −0.091], + tag −0.155, − tag −0.133. Drops
  largest on easy scenes. Labels untested until a non-harmful recipe exists. Proposed: arm A recipe variants
  (frozen SigLIP; 10× lower lr; evaluate intermediate checkpoints) before any further label comparison.

- **The tagged policy ignores its tag** (`reports/2026-11-04-tag-usage/`, 18309419): + vs − tag moves arm C's
  action by 2% of its sampling noise (arm A, never tagged: 4%); flow-matching loss with correct vs flipped tag
  identical (0.00689 / 0.00689). No tag learned, so the eval says nothing about label quality yet. Proposed next:
  oracle-tag fine-tune (tag = rollout outcome) to separate "labels too weak" from "conditioning too weak".
- **PaliGemma vs π₀.₅ per layer** (18308029): PaliGemma SNR drifts 0.146 → 0.113 with depth; π₀.₅ flat ~0.15
  to h15 then 0.122 / 0.107. Robot training raised accuracy at every layer (MAE ~0.20 → ~0.18, AUC ~0.69 → ~0.81)
  and mid-layer steadiness (h11 0.119 → 0.159), and removed the top layers' milestone response (+0.05 → ~0).
  Top-layer noise is inherited, not caused by robot training.

- **Policy results: no gain from the advantage tag.** Success-only 18291522 134/330 = 0.406; advantage-tagged
  (ResNet OOF labels) + tag 0.409, − tag (control) 0.430. One seed, 330 held-out trials each.
- **π₀.₅ OOF labels are worse than the ResNet's** (labels/oof_expert_n20_p30): OOF MAE 0.182 vs 0.195, but
  within-scene AUC 0.68 vs 0.74; interior positive success/failure 72/66% (ResNet 73/66%); agreement 65% (chance
  57%), advantage corr 0.27. **No π₀.₅-label policy fine-tune** (decided 2026-11-04; reopen only if asked).
- **B2 steering stopped at step 2** (`reports/2026-11-04-steer-probe/`): π₀.₅'s 8 candidates end 1.9 mm apart
  (arm moves 22.7 mm per chunk); cross-validated oracle gain +0.006 [−0.028, 0.042].
- **Per-layer probe (fixed+fresh, 3 seeds):** SNR flat 0.14–0.16 for h0–h15, drops at h17 0.122 / h18 0.107 /
  final 0.116; milestone vanishes at h17+; MAE ~0.175 at every depth; AUC10 best mid-stack (h13 0.837). h1–h3
  within 2 sd of h0 → image-only-attention ablation not triggered. On 375 fixed (18306065): h0 0.107 ± 0.023,
  h1–h17 0.079–0.094, final 0.055 — P1b's "fall from layer 1" was ~1 sd noise on the low-variety set.
- **A2 truncated value expert, seed 0** (`reports/2026-11-04-truncated-value-expert/`): N=18 SNR 0.152 / MAE
  0.160; N=9 0.158 / 0.162; N=4 **0.180** / 0.169 (noise 0.101 → 0.092 → 0.085). N=4 is the steadiest π₀.₅ V
  so far but inside the 0.166 ± 0.025 bar on one seed; ResNet still 0.247.
- P1c self-check fixed 2026-11-04 (pooled rel L2 gate); PaliGemma runs skip the Stage 3a pool comparison (it
  compares with π₀.₅-weight pools, 2.67 by design).
- Earlier: ResNet vs π₀.₅ trade-off (π₀.₅ more accurate via scene difficulty, ResNet 1.5–2× steadier); two noise
  sources (frozen LLM layers, SigLIP fine-tuning); SigLIP-only V closed; E2 subtask head no SNR change.

## Open decisions for Karen

1. A2 seeds 1–2 for N=4 (and N=9): 4 GPU jobs × ~3–5 h (`pi05_value_expert.sbatch`, `--mem=96G` or scoring
   OOMs; ARGS in the A2 report). Two lanes fit the cap.
2. Policy "no gain → diagnose" (meeting step 5): labels are weak for both backbones (~6–7 pt success/failure gap).
   Options: tag usage at test time, a stronger label rule (longer lookahead / stricter percentile), or train-all
   labels (paused).

## Mentor meeting 2026-11-03: agreed next steps

Notes: Granola transcript (link in the meeting chat). The notes' "CQ" = SeeQ, "SmallVLA" = SmolVLA.

1. **DONE: verify episode length in environment steps** (≈20 s real: LIBERO steps at 20 Hz, videos are written at 10 fps; every "seconds" label in our scripts is 2× too long, metrics unchanged; Week 8 card). Original notes: What we know: `frame_index` runs to 516–519 and `T` ≤ 519, which
   matches `MAX_STEPS["libero_10"] = 520` in `scripts/libero_rollouts.py:44`, so one frame = one policy/env step.
   Every "seconds" number assumes **10 Hz** (`libero_oof_labels.py --n 20` = "2 s"; change SNR lag 5 grid frames of 4 =
   "2 s"; episodes "~52 s"). robosuite's default `control_freq` is 20 Hz. To check: print `env.env.control_freq` (or
   the LIBERO env's `control_freq` kwarg) inside `libero_rollouts.py` once; also check the recorded video fps. If it
   is 20 Hz, every time label halves (2 s → 1 s); the metrics themselves are unchanged, only their units.
2. **(Paused) the "train on all / label all" advantage run** (comparison for the 5-fold out-of-fold labels; not a priority while the out-of-fold labels run).
   - Train a ResNet V on all 900 train rollouts (same recipe as the fixed+fresh ResNet, `vl.libero_resnet`, no held-out
     fold; keep the 5 inner states for early stopping), predict every train frame, then label with the same rule
     (`scripts/libero_oof_labels.py`: n = 20, 30th percentile → 70% positive). Easiest: add a `--source resnet_all`
     pattern to `PATTERN` in that script.
   - Before any policy fine-tune, compare the two label sets: positive fraction, within-scene AUC (OOF: 0.74), and
     agreement. **Degeneracy check:** if in-sample V has memorised outcomes, advantages collapse (nearly all positive
     on successes, negative on failures; AUC near 1.0 against outcome with little within-episode variation).
   - Then an advantage-tagged fine-tune with those labels (third policy arm). Needs one large GPU (L40S / RTX PRO
     6000); start it when the current advantage-tagged fine-tune ends to stay ≤ 4.
   - Mentor's three options for reference: (1) train all / label all, simplest, risk of degenerate advantages;
     (2) new held-out rollout set, cleanest, 2× data; (3) 5-fold out-of-fold, most principled (running).
3. **DONE on fixed+fresh (fixed-only analysis 18306065 running): Revisit the LLM final-layer ablation details** (P1b, `scripts/libero_p1b_prepare.py`,
   `reports/2026-11-02-frozen-encoders/report.md`). What was measured: same small head on frozen features, 375
   fixed-layout episodes, 3 seeds: SigLIP 0.122; LLM layer 1 / 5 / 9 / 13 / 17: 0.093 / 0.098 / 0.089 / 0.080 / 0.060;
   final 0.059. Points to be ready to explain: (a) the 375-fixed set has the least layout variety, so absolute SNR is
   low everywhere (frozen SigLIP 0.194 on 375 fresh); compare within the set only; (b) the drop starts at layer 1
   (−25%) and is steepest after layer 13; (c) SNR only, not accuracy per layer. 18298946 redoes all 19 layers on
   fixed+fresh: analyze with the command in "Resume steps", and report **MAE / AUC per layer next to SNR** (the
   point is to find a layer with π₀.₅'s accuracy and less flicker).
4. **DONE: scene-calibration scatter plot** (Week 8: all backbones rank scenes alike, r ≈ 0.66; π₀.₅'s early V spreads ~2× wider, slope vs target 0.49–0.57 vs 0.23, which is its accuracy edge). Original plan: (the "π₀.₅ recognises hard scenes" claim). Val = 120 episodes over 10 scenes
   (12 rollouts each). Per scene: x = actual success rate (of 12), y = mean predicted V over the first 2 s (and
   optionally the whole episode); one panel or colour per backbone (ResNet 18237895 `ev_resnet`, frozen π₀.₅ 18237898
   `ev_expert`, LoRA 18258391 `ev_expert`); report Pearson/Spearman r per backbone. Only 10 points per backbone:
   pool seeds 0–2 for the 3-seed backbones and show seed spread. CPU only. Add to the Week 8 site.
5. **DONE (no gain, see verdicts): Milestone:** rollout results for the ResNet advantage-tagged policy vs the success-only baseline. If no gain,
   diagnose (label quality, tag usage at test time, negative-tag control) before scaling up.

## New directions to explore

### A. Which language-model layers does a value function need? (SmolVLA / SmolVLM)

Papers: SmolVLM (arXiv 2504.05299), small VLMs with aggressive image-token compression; SmolVLA (arXiv 2506.01844)
builds a VLA on SmolVLM-2 and **drops the top half of the language model**: the action expert reads only the first
N = L/2 layers. Their LIBERO ablation (Table 8, avg success): N = 8: 75.0, 16: 78.5 (chosen), 24: 79.5, 32: 80.3;
skip every 2nd layer 75.5; smaller 256M VLM 75.8. They also cut images to 64 visual tokens/frame (pixel shuffle, no
tiling) and keep the VLM frozen. Mentor's point: going from SmolVLA back to SmolVLM, the upper LM layers mostly serve
language generation, and control (or value) may not need them. That fits our finding that steadiness falls through
π₀.₅'s layers.

How our model maps on: `scripts/pi05_value_expert.py` grafts an 18-layer value expert that attends to **every**
layer's K/V of frozen π₀.₅'s PaliGemma (the same way π₀.₅'s action expert does), so the value sees the noisy top
layers too.

Experiments, cheapest first:
- **A1 (done, see verdicts):** the per-layer probe 18298946, with MAE/AUC per layer (step 3 above). Tells which depth to cut at.
- **A2 (seed 0 done, see verdicts): truncated π₀.₅ value expert** (code in place: `pi05_value_expert.py --train --n-layers N`; slices the K/V cache, exact by causality, checked in `--selftest`). Run the prefix through only the first N PaliGemma layers and give the value
  expert N layers (it must match the backbone's layer count layer for layer). N ∈ {4, 9, 18 = current}, frozen,
  fixed+fresh 825 episodes, seed 0 first; score with `score(p, "ev_expert")`. Bar: SNR clearly above frozen π₀.₅'s
  0.166 ± 0.025 while MAE stays ≤ ~0.175. Also ~2× cheaper per step at N = 9. Code: a `--n-layers` flag that slices
  the `gemma.Module` config depth for both the backbone pass in `backbone_kv` and the expert.
- **A3: truncated + LoRA.** If A2 helps, repeat the LoRA run (18258391 recipe, `scripts/pi05_value_lora.py`) on the
  truncated model: LoRA had the best accuracy (0.155) but SNR 0.133.
- **A4: a small VLM as the value backbone.** SmolVLM-256M/500M (64 tokens/frame), first half of its LM, small value
  head, fully fine-tunable on one GPU. Tests "smaller models + limited data" directly and is a clean comparison with
  the ResNet and with SeeQ's finding that a VLM without pre-training ≈ ResNet.
- **A5 (optional): fewer visual tokens.** Pool π₀.₅'s 256 SigLIP tokens per image down to 64 before the LM.
  Speculative; only if A2 points at the image tokens.
- Check against 18286806 (same per-layer probe on pre-robot PaliGemma): did robot training make the upper layers
  noisy, or were they noisy already?

### B. SeeQ (arXiv 2609.22085): subtask-scoped Q-functions

SeeQ learns a Q-function for the **currently active subtask** (shorter horizon, easier TD backups), predicts the
subtask in text first and then its value, pre-trains on broad robot data, and uses it for best-of-N policy steering.
Ours differs: a Monte-Carlo V over the whole task, used for advantage conditioning. Our quick check found the subtask
is easy to read (~97%) but an auxiliary subtask head did not change steadiness. Still worth doing in full (mentor):
- **B1 (Karen: go 2026-11-04; queued 18313821/22, adjust after steering v2):** subtask-scoped V: the value of
  finishing the *current* subtask (place the first pot, then the second), conditioned on the subtask.
  - Code: `vl.libero_subtask.scoped_targets` (+ `tests/test_libero_scoped.py`), `vl.libero_resnet --subtask-scoped`.
    Boundary = the milestone frame m (`vl.libero_milestone.milestone`: first image with a pot on the stove that stays
    10 steps; monotone). Subtask-0 frames of episodes that reach m get −(m − t)/1040 (subtask done, no failure
    penalty); every other frame keeps the whole-task target (an episode that never reaches m failed subtask 0 and
    the task alike). Head = 2 × 201 logits, the true scope's half is read (SeeQ predicts the subtask; we have it
    exactly from the sim). [ours: Monte Carlo, not SeeQ's TD; SeeQ: subtask scope + conditioning.] On 30 episodes:
    28 reach m, 58% of frames in subtask 0, 51% retargeted.
  - Step 1 (queued): fixed+fresh, seeds 0–1, compare with ResNet ff 18237895–97. Metrics on the scoped target are
    not directly comparable to whole-task ones: compare change SNR / MAE within subtask 1 (same target in both) and
    the steadiness of subtask 0 against its own target; check the value switch at m is sharp.
  - Step 2 (if steadier): 5-fold OOF scoped V → scoped advantage labels (A within a subtask; when the boundary falls
    inside the n-step lookahead, bootstrap with 0 = subtask done, as SeeQ's no-cross-boundary backup) → compare
    with the ResNet labels (within-scene AUC 0.74). **Changing the labels: ask Karen first.**
- **B2: SeeQ-style Q-function + best-of-N steering** (Karen approved 2026-11-03; the notes' "CQ differences: uses
  Q-function, different policy extraction, likely a stronger method"). SeeQ does not train the value into the policy;
  at each replan it samples N candidate action chunks from the base policy, scores each with Q(o, subtask, chunk),
  and executes the best (policy weights unchanged). Our version:
  - action = one π₀.₅ chunk (the 5 executed actions; raw npz already stores `actions` + `new_chunk`, no new data);
  - Q input = frame + subtask + chunk (ResNet: image features + small action MLP into the same 201-bin head; π₀.₅
    later: chunk tokens next to the value expert's query); target = B1's subtask-scoped target, TD/SARSA with the
    next executed chunk (all 900 rollouts are π₀.₅'s own, so this is Q of π₀.₅ and best-of-N = one
    policy-improvement step);
  - extraction = a steering policy server wrapping the base server; same client, same 330 held-out trials.
  Steps (stop early if a check fails):
  **STOPPED at step 2 (2026-11-04, oracle gain ≈ 0).**
  1. **Candidate spread (job 18296012):** do π₀.₅'s N = 8 samples at a replan point differ? Arm actions (per-dim
     std units) and end-effector position after 5 steps (mm) vs how far the arm moves in 5 steps; gripper split.
  2. **Simulator headroom (same job):** at policy steps 40/130/220/310 of 10 base episodes, each of 8 candidates is
     executed from the saved state and the base policy continues to the end, 2× per candidate. Between-candidate
     variance of success (binomial noise removed) and a cross-validated oracle selection gain (pick on repeat 1,
     score on repeat 2). If the gain ≈ 0, a learned Q cannot steer → stop B2. After-chunk frames are saved so the
     existing ResNet V can be scored offline (can it rank candidates?).
     Analyze: `.venv/bin/python scripts/libero_steer_analyze.py --run /iris/u/ankile/karenvo/libero/runs/steer_probe_libero_10_18296012`.
     Code: `scripts/libero_steer_probe.py` (+ `.sbatch`), smoke-tested with `--args.fake-policy`. Restore is exact
     (3e-17) only because it also restores the Panda gripper's integrated command, the solver warm start and
     robosuite's step counter (without the gripper: 0.011); any steering code that branches the sim must do the same.
     Expected run time ~3–4 h (~25 sim steps/s).
  3. ResNet Q with the subtask target (TITAN OK); check Q changes when the action is shuffled across frames.
  4. Steering server + eval, N = 1 / 4 / 8 on the same 330 trials; later steer the advantage-tagged policy too.
  5. π₀.₅ Q, only if the ResNet version steers.
  Costs: N× inference per replan; offline Q with one action per state may ignore the action (step 3 check).
- **Try later (Karen, 2026-11-04): subtask-scoped Q trained on noisy-candidate data.** Only if steering v2's
  oracle gain is real (≥ ~+5 pts). Normal-noise rollouts have ~one action per state, so a Q would likely learn to
  ignore the action; rollouts that execute noise-scaled chunks (v2's best scale) give the action variety a Q needs.
  Then B1's target with a TD/SARSA backup, ResNet Q (image + chunk MLP), shuffled-action check, steering eval on
  the 330 trials. Cost: new rollouts (~900 episodes, GPU hours) + Q training; noisier candidates are also worse on
  average, so the gain must beat that.
- Pre-training (SeeQ/RECAP style) stays on hold: infra is being arranged by the mentor and is not the bottleneck;
  keep pushing with small models and limited data.

## Resume steps

1. When 18305873 ends: `PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze --out
   /move/u/karenvo/libero_features/step0_paligemma_hidden_18286806 --fit-subset fit_fixed_fresh.txt --tag fixedfresh
   --encoders llm_h0 ... llm_hfinal` via `FEAT=... FIT=fit_fixed_fresh.txt TAG=fixedfresh sbatch
   scripts/layer_analyze_cpu.sbatch` (CPU). Plot π₀.₅ vs PaliGemma per-layer SNR / MAE together: did robot
   training make layers 17–18 noisy?
3. Karen's open decisions above.
4. Update `reports/2026-11-02-frozen-encoders/report.md`, `docs/findings.md`, the Week 8 site, and this file.

## Traps

- π₀.₅ does not load in a 16 GiB interactive session (OOM). Always sbatch π₀.₅ jobs. A one-batch π₀.₅ check can run
  as a **CPU** sbatch (`JAX_PLATFORMS=cpu`, 96G, ~7 min; `scripts/debug/*.sbatch`) without using a GPU.
- `/iris/u/ankile` is 99% full: large features go to `/move/u/karenvo/libero_features/`.
- Never `scancel` other sessions' jobs; re-chain only your own.
- Site rebuild: `.venv/bin/python scripts/build_site.py`; upload with
  `/iris/u/ankile/karenvo/google-cloud-sdk/bin/gcloud storage cp -r --cache-control=no-cache reports/site/value-learning/* gs://zima-robot/reports/karenvo/value-learning/`.
