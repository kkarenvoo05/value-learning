# Findings

Settled conclusions, grouped by topic. **Corrected in place**: when a finding is overturned, edit or strike it
here and note the date; do not append a contradiction. Each item cites its evidence. Observations and
interpretations are labelled. Live status is in `docs/CURRENT.md`, chronology in `docs/libero_plan.md`.

## LIBERO (libero_10 task 8, moka pots)

### Data
- Stage 2: 570 frozen-π₀.₅ rollouts (450 on 30 train init states, 120 on 10 val states); open-loop replay of
  the saved actions is exact. Success differs by init state from 0 to 100%. [plan, Stage 2 / 2026-10-28]
- Layout variety matters more than episode count for value heads on frozen features: at 375 episodes, fresh
  layouts give SigLIP head SNR 0.194 vs 0.122 on the 30 fixed layouts (ResNet 0.193 vs 0.155), and adding the fixed
  episodes to fresh (825) lowers both. *Interpretation:* repeated fixed layouts invite a layout shortcut.
  [`reports/2026-11-02-frozen-encoders/`, 2026-11-03]
  - *Corrected 2026-11-03:* "frozen SigLIP needs more data" (P1b) was wrong; it needs more layouts.

### Where π₀.₅'s value noise comes from
- Frozen π₀.₅ encodes "a pot is on the stove" (probe AUC 0.96–0.99), but value heads do not use it well.
  [plan 2026-10-30, progress probe]
- B's jittery V is per-frame flicker of the success belief, not error within a value hump (~98% of 2 s ΔV
  variance). [plan 2026-10-31; P2, plan 2026-11-02]
- Robot (VLA) training did not make SigLIP noisy: frozen π₀.₅ SigLIP ≥ web SigLIP on every metric, and ≈ an
  ImageNet ResNet at 825 episodes (head SNR 0.172 vs 0.168). [`reports/2026-11-02-frozen-encoders/`]
- Two separate noise sources, each about halving head SNR [same report]:
  - the frozen LLM layers: SigLIP 0.122 → LLM final hidden 0.059 (matched head, 375 fixed episodes);
  - fine-tuning SigLIP: 0.172 → 0.084 (825), 0.122 → 0.078 (375). Fine-tuning the ResNet costs nothing.
    A 10× lower SigLIP lr (1e-6) does not remove it: full trainer, noise 0.097 vs frozen 0.057 (1 seed).
    [`reports/2026-11-03-siglip-direct/`]
- Skipping the LLM (V straight on SigLIP) is not a fix, 1 seed: frozen SigLIP is steady (SNR 0.221) but misses the
  pot landing (milestone +0.017) and is least accurate (MAE 0.215); B's accuracy edge (MAE 0.160) comes via the LLM.
  [`reports/2026-11-03-siglip-direct/`]
  - *Open:* where in the LLM the drop starts (P1c step 2, job 18279695).
- The value head architecture never matters (± 0.01 SNR in head swaps). [plan 2026-10-30, head swap]

### What did not help
- Auxiliary subtask head (label = pot count): no SNR change on the ResNet and on frozen B (2 of 3 seeds); it
  learns pot state at 96% accuracy. [plan 2026-11-02 E2, 2026-11-03 08:45]
- Removing the outcome loss: SNR 0.073, same as B. [plan 2026-10-31]
- Head-side regularisation (aux targets, feature dropout, bottleneck). [plan 2026-10-30]
- Smoothing V raises SNR but does not close B's gap to the ResNet at matched smoothing. [`reports/libero_smoothing.md`]

### Policy fine-tuning (Stage 6; task 8, 330 test trials per arm, base π₀.₅ 186/330 = 0.564)
- Fine-tuning π₀.₅ on the moka rollouts to the full 5,000 steps lowers success: success-only arm A 0.406, arm C
  + tag 0.409, step 3,000 0.433; faithful RECAP 0.40–0.45 in every eval mode. *Observation.*
  [`reports/2026-11-04-base-vs-finetuned/`; 18310777; 18338861-4]
- **Stopping early is not a fix** (*corrected 2026-11-06*; the step-1000 result had been read as one): arm A,
  original recipe, step 1,000: seed 0 0.639 (18310776), **seed 1 0.518** (18340452, same 5,000-step lr schedule,
  `--stop-after 1000`). Paired by episode: seed 1 − seed 0 −0.121 ± 0.074; seed 1 − base −0.045 ± 0.072; seed 0
  step 2,000 − base +0.024 ± 0.073 (18338526). The two seeds also succeed on different init states (state 48: 0.88
  vs 0.36). *Interpretation:* steps 1,000–2,000 are roughly neutral, not a gain; at a fixed step the training seed
  moves success by ~0.1, the size of every effect tested so far. The low-lr recipe R2 (0.521, −0.042 [−0.115,
  +0.027]) is also a single seed. [plan 2026-11-05 / 2026-11-06]
- **Tagged policies ignore the tag**, every label set so far: + vs − tag moves the executed action by 2.2–2.3% of
  the policy's sampling noise for faithful RECAP (18338865) and subtask-scoped labels (18352443), no more than arm A,
  which never saw a tag (2.6%); flipped-tag training loss = correct-tag loss. RECAP evals: + tag 0.442, − tag 0.397
  (+0.045 ± 0.063), guidance β 1.5 / 2.5 0.409 / 0.448. Not a pipeline bug (tags reach training at the configured
  40% / 30% dropout; tagged prompts tokenize distinctly, untruncated). [`reports/2026-11-04-tag-usage/`,
  `reports/2026-11-06-tag-deep-dive/`]
  - *Settled 2026-11-08:* the conditioning learns (planted tags of δ 0.02 / 0.1 are learned on every route); the
    labels are weak, and **not because of V**: labels from what actually happened (hindsight, realised return − V)
    carry less action information than the current 1 s labels (ideal shift 0.0091 vs 0.0151), and longer
    lookaheads score lower. The bottleneck is the data (one policy, little action variety at the same states).
    [`reports/2026-11-08-coach-test/`]
  - *Earlier open question:* labels too weak vs conditioning that does not learn. The data supports a tag-dependent action
    difference of ~0.015 raw (mostly gripper timing) against the models' 0.0008, an upper bound since our state
    features are weaker than π₀.₅'s. Injected-shift positive controls running (18352944 / 18352946).
  - The policy does read its prompt (2026-11-06, 18355145): swapping the instruction moves the action ~1 noise-width
    (base 0.97–1.38×, RECAP-tagged 0.68–0.98×), ~30–45× the tag's effect. The tag-trained model is less
    text-sensitive than base (tag suffix 0.022 vs base 0.044). [`reports/2026-11-06-tag-deep-dive/`]
- Steering by resampling the start noise (best of N candidates, cross-validated): no gain over a normal sample at
  noise ×1.5 (+0.026 [−0.014, +0.068]; `runs/steer_v2_scale1p5_libero_10_18329175/analysis`) or ×2.0 (+0.010
  [−0.049, +0.077]; `runs/steer_v2_scale2p0_libero_10_18329176/analysis`), although ×2.0 doubles gripper
  disagreement between candidates (0.10 → 0.22). [`reports/2026-11-04-steer-probe/` for v1]
- **Structured end-effector offsets do create choices worth ranking** (offset G0, 2026-11-21, sim oracle): picking
  among the plain chunk and the chunk + ~2 cm along ±x / ±y / ±z (selected on one repeat, scored on the other 3)
  beats the plain chunk by +12.5 [+5.6, +19.8] points (offset on one chunk) and +12.1 [+4.9, +19.3] (held 4 chunks),
  vs +0.6 for noise candidates. No single direction helps in both probes (z+, the height hypothesis: −1.9 / +2.5):
  the useful offset depends on the state. Upper bound for a learned ranker (uses the simulator's held-out outcome);
  20 base episodes. [`reports/2026-11-20-offset-g0/`]

### Policy improvement (2026-11-09 to 11-11; same 330 val trials, matched noise seed 1, base 0.524)
All single training seed; *observations* unless marked. Chronology in `docs/libero_plan.md` log 11-10 to 11-11.
- **Amplifying the learned tag difference at inference beats base** (Advantage Guidance, arXiv 2610.19783, Eq. 4,
  v = v_none + w (v_pos − v_neg)). On the AdaRMS vb40 checkpoint (the one real-label run whose tag moves actions,
  13.6% of noise): w 8 0.658, +0.133 [+0.070, +0.197] paired; w 0 / 2 / 4 / −4 0.385 / 0.348 / 0.497 / 0.230
  (dose-response, and the sign control falls). Shuffled-label null control (same recipe, w 8): 0.363 on 289/330,
  −0.304 vs real labels, so the gain comes from the labels, not from guidance itself. *Interpretation:* the tag
  carries usable signal; it was too small to matter at w = 1, which refines "tagged policies ignore the tag" above.
  [`reports/2026-11-11-advantage-guidance/`]
- **Residual parameterisation is better still**: frozen base π₀.₅ + zero-init AdaLN head on the action-expert
  features, ±1 tags, at the action shift matched to AdaRMS w 8 (shift 0.99 × noise, w 9.26, picked offline):
  **0.754, +0.230 [+0.164, +0.300]**; +0.097 [+0.030, +0.167] vs AdaRMS w 8 on the same trials. Shift 0.57 0.697,
  w 16 0.658: 0.99 is the peak. Base vs guided videos: guidance removes the "closed freeze" (gripper closed on a
  handle, no lift ≥ 3 s: base 12/30 wrist-view episodes, guided 0/30); guided failures are pots left on their side
  and open-hand hovers. [log 11-11 11:55, 17:00; `reports/site/value-learning/gallery-residual/`]
- **DSRL** (noise-space SAC on base π₀.₅, online): seed 0 held-out 0.861, +0.106 [+0.048, +0.167] vs the
  residual (*corrected 2026-11-20*: this agent trained on **1,357** of its own episodes, not 1,800; the run restarted
  at row 443 and the checkpoint `ep1800` counts rows). *Interpretation:* π₀.₅'s noise space holds a lot of
  outcome-relevant headroom on this task; online RL is the stronger reference. [`reports/2026-11-10-dsrl-online/`]
- **Iterated filtered BC does not climb**: round 2 (round 1 + round 2 successes, from base) 0.442 / 0.506 (s0 / s1)
  vs round 1 0.555. [log 11-11 15:30]
- **Advantages vs simulator ground truth**: corr(A, A*) 0.04–0.14 for every value / label source (ceiling 0.69);
  corr(V, V*) 0.22–0.29 (ceiling 0.97). The labels are weak per frame, yet guidance turns them into +0.13 to +0.23.
  [`reports/2026-11-10-value-vs-gt/`]

### Plateau, distillation, noisy judge (2026-11-13 to 11-20; same 330 trials, one seed per policy)
*Observations* unless marked. Numbers and paired CIs: [`reports/2026-11-20-plateau-and-matched-budget/`];
chronology in the `docs/libero_plan.md` log.
- **Both methods level off at 85–90%.** AG rounds r0–r7 (900 base rollouts + 100 per round): 75.5, 79.1, 74.5,
  80.9, 79.4, 87.6, 77.3, 83.0, with no trend after round 0. DSRL seed 1: 87.0 / 89.1 / 90.0 / 86.1 / 88.2 at
  1,300 / 1,800 / 2,400 / 3,000 / 3,300 episodes; best DSRL − AG r5 +2.4 [−2.4, +7.3].
- **Guidance strength is not the limit**: push 0.8–1.2 is flat on r5 / r7; push 1.8 costs 9.7 / 11.5 points; fixed
  w = 1 gives 63–64% on r5–r7.
- **Moving the guidance reference does not help**: anchored to the data's average action −9.7 (r5) / −28.2 (r7);
  learned no-tag branch, best w, −7.9 vs standard guidance; stacked on the previous round's policy −7.3 / −0.9 / −1.2
  (r0 / r5 / r6). IQL labels are worse than value-binned labels (−4.2 on r7 data, −15.8 at 100 base episodes).
- **Dropping the 29 demos helps round 0**: +5.8 [−0.6, +11.8] at 900 rollouts, +27.9 [+20.9, +35.2] at 50.
- **Distillation keeps the teacher's success**: LoRA fine-tune of π₀.₅ (filtered BC round 1 recipe) on teacher
  successes: DSRL 0.882 (−0.9 vs DSRL s1 @1,800), AG 0.891 on 266/330 (+0.8 vs AG r5), hybrid 0.809, base 0.497
  (executed targets 0.503). *Interpretation:* the gain transfers into π₀.₅ weights, so either method has a deployable
  form; the fine-tune itself no longer costs points (contrast the −13 to −16 under "Policy fine-tuning", a different
  recipe).
- **Noisy judge** (25% symmetric, independent outcome flips): AG round 0 +0.6 [−5.5, +6.7]; DSRL −36.7 at 400 and
  −47.9 at 600 episodes (ends below base). *Interpretation:* with a real VLM judge AG is the safer method; a
  realistic, lopsided error model is untested.
- **Small fixed-budget chains drop at round 1, for chains with ~50 base episodes per task** (corrected 2026-11-21):
  b50cv val 44.8 → 11.5 / 10.0 (training seeds 0 / 1), moka cont 51.0 → 15.3 / 14.7; two-task mt100cv moka val
  65.8 → 32.1 / 14.5 (seed 0 on 305 trials; the first eval logged 280 client errors as failures and read 4.5).
  b100's drop (79.7 → 53.0) was training-seed variance: seed 1 on the same data scored 77.0 (moka cont 78.0).
  mt100cv seed 1 on task 64 fresh layouts: 73.3 (296/300) vs seed 0 33.3, r0 77.3, so even a reproduced drop can
  differ by task within one policy. Labels treat base and guided episodes alike in every chain
  (`reports/2026-11-21-r1-factorial/`). Cause still open (`reports/2026-11-20-mt-r1-drop/`); these chains all train
  with demos. *Method trap:* single retrains on the same data vary by up to ~25 points; check `episodes.csv`
  `error` before reading any eval.
- **Matched budget, per state**: no val state is solved by one method and unsolved by the other (worst state AG r5
  24/33, DSRL 15–19/33, base 4/33); per-state gaps are 3–5 trials of 33 except one DSRL checkpoint.

### Method traps
- One training seed per recipe is not evidence: same recipe and step, two seeds 0.639 vs 0.518 (above). Train
  ≥ 2–3 seeds before reading a policy-recipe effect of ≤ 0.1. Eval noise alone is ±0.054 per arm at 330 trials
  (±0.06–0.075 paired by episode); `MATCHED_NOISE=1` (`vl/libero_noise.py`, 2026-11-06) tightens the paired
  comparison but not the seed spread. [plan 2026-11-06]
- Matched noise does not make evals repeatable trial by trial: two evals of the same or a near-identical policy
  agree on only 72–75% of trials (r0 control rerun 75%, 14% identical step counts; r6 push 0.8 vs 0.99 72%), and a
  second eval noise seed moves r5 / r7 by −2.7 / −3.0. Treat single-eval differences below ~6 points as noise.
  [`reports/2026-11-20-plateau-and-matched-budget/`]
- Milestone and dip effects are not stable across seeds (no 3a probe arm has a positive milestone excess on all 3
  seeds); report ≥ 3 seeds before reading them. [`reports/libero_seeds.md`, plan 2026-10-29]
- Inner-CE early stopping stops soft-target arms at epoch 0; select on inner MAE instead. [plan 2026-10-31]
- π₀.₅ cannot be loaded in a 16 GiB interactive session; use sbatch. [plan 2026-11-03]

## Mulligan Insert Marker (Stages 1–5.5)

To fill from `PROGRESS.md` (Stages 4–5.5) and `reports/*.md`; not yet consolidated here.
