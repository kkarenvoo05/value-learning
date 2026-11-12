# Directions and mentor next-step lists

Moved out of `docs/CURRENT.md` on 2026-11-11 (CURRENT is overwritten each session; these are longer-lived). Status
notes inside are as of when each was written; check `docs/libero_plan.md` `## Log` for what happened since.

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
- **Try later (Karen, 2026-11-08): AdaRMS + discrete, with the tag in both places.** Not run. As the code stands,
  `pi05_tag_adarms.py` strips the tag from the prompt (`vl.tag_route`) and puts it only into the action expert's
  AdaRMS conditioning. The discrete action-token loss (`--discrete-weight`) trains the backbone to predict the actions
  from the prompt, so combined as is it would never see the tag and would learn untagged action tokens. A real
  version keeps the tag text in the prompt (for the backbone + discrete loss) and also adds it to AdaRMS (for the
  expert): a small change in the tag-routing patch. Low priority: every tag check so far (prompt, AdaRMS,
  prompt + discrete, RECAP-complete, speed stage 1) shows the real tag moving the actions by only 2-3% of the
  sampling noise, so a second route for the same weak tag is unlikely to change that.
- Pre-training (SeeQ/RECAP style) stays on hold: infra is being arranged by the mentor and is not the bottleneck;
  keep pushing with small models and limited data.

