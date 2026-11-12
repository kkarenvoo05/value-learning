# How others handle value noise and advantage conditioning (literature review, 2026-11-05)

**Question (Karen).** Before designing our own fixes: how do RECAP and similar implementations deal with the
problems we hit, and what has been shown to work?

Our problems (from `reports/2026-11-04-*`, `reports/2026-11-05-label-denoise/`):
1. The advantage-tagged policy ignores its tag (+/− moves actions by 2% of sampling noise; correct vs flipped tag
   loss identical); the outcome-oracle tag is ignored too.
2. Fine-tuning on our own rollouts lowers success by ~15 points, tag or not.
3. Labels are mostly value noise: ResNet vs expert agree at κ 0.20; stalls default to positive; A ≈ ΔP(success).
4. A value function makes confident systematic mistakes at a subtask boundary (ep 139's first placement).

Sources read in full or in large part (quotes checked by the reader; items marked *partial* came from the
unread remainder of a paper and need checking before citing): RECAP / π\*0.6 (arXiv 2511.14759 v2, full);
CFGRL (2505.23458, Sections 1–6 + App. A); ESPER (2205.15967, first ~40% verbatim); Brandfonbrener et al. RCSL
(2206.01079, first ~40%); RLinf RECAP docs (full); OpenPIE README; Dissecting Advantage-Guided Post-Training
(2609.28161, full); RAPolicy (2609.22888, *partial*); RePO-VLA (2605.09410, *partial*); CRR (2006.15134,
search summaries only); Stop Regressing / HL-Gauss (2403.03950, search summaries only). SeeQ is summarised in
`docs/libero_plan.md` 2026-11-02.

## 1. Our arm C against RECAP and the open reimplementations

| | RECAP / π\*0.6 | RLinf RECAP (π0.5, LIBERO-10 task 0) | CFGRL | **ours (arm C)** |
|---|---|---|---|---|
| value | 670M VLM, 201-bin MC, pre-trained on all robot data + web co-training | SigLIP2 + Gemma3-270M, 201 bins, failure reward −300 | IQL Q, V | ResNet-18 OOF, 201-bin MC, one-hot CE |
| advantage | n-step, N = 50 at 50 Hz (~1 s); MC (N = T) in pre-training, "higher variance" | n-step, N = 10, γ = 1 | Q(s,a) − V(s) | n-step, 20 steps at 20 Hz (1 s) |
| positive share | **~30%** in pre-training, **~40%** in fine-tuning, 10% for one task (App. A-F) | **30%** (`positive_quantile: 0.3`) | A ≥ 0 | **70%** |
| tag dropout | **30%** | 10%; negatives always untagged ("positive-only conditional") | 10% | **0%** |
| guidance at test | β = 1 by default, β ∈ [1.5, 2.5] "where useful" | CFG scale 1.0 | w swept 1–3; improves with w | none possible (no dropout) |
| demonstrations | SFT stage on demos with tag fixed positive; corrections forced positive | SFT demos in the data | — | **not used** |
| where the tag is learned | **in large multi-task pre-training**, then task fine-tune from the pre-trained checkpoint each iteration | task fine-tune, 30k steps, batch 512, lr 1e-5, 4,096 rollouts | task training | task fine-tune, 5k steps, batch 128, LoRA lr 1e-4 / 2.5e-5, 900 rollouts |
| reported result | throughput ×2, failures ÷2 (real robot) | 48.8% → 66.5% (one run, few-shot SFT base) | beats AWR on most offline-RL tasks | −15 pts vs base |

Quotes: RECAP App. A-F "During pre-training we select the threshold for each task such that approximately 30% of
the demonstration data has positive advantage"; "During fine-tuning we generally set the threshold such that
approximately 40% of the evaluation rollouts in each iteration have positive advantage"; "we randomly drop out the
conditioning on the advantage indicator 30% of the time". Footnote 2: prior work "uniformly chose ε=0 and tuned β
at test time. We found it easier to obtain good results by instead using the threshold ε_ℓ". Sec. V-B: the tag
text "Advantage: positive" / "Advantage: negative" sits after the (predicted) subtask text and before the actions.

**Correction to our own record.** `docs/libero_plan.md` (Stage 5 checklist) planned the "RECAP 30% threshold";
`scripts/libero_oof_labels.py` implemented it as the 30th percentile of A, i.e. 70% positive, and noted "[paper
ambiguous]". Sec. V-D's wording ("30% percentile of values") is ambiguous, but App. A-F states the positive share
directly: ~30% (pre-training) and ~40% (fine-tuning). Our labels have the opposite positive share.

### How the tag reaches the actions: π*0.6 vs our π₀.₅ fine-tune (added 2026-11-06)

Source: arXiv 2511.14759v2 (HTML), read 2026-11-06. Ours: openpi `Pi0.compute_loss` (`src/openpi/models/pi0.py`),
which returns only the flow-matching velocity MSE.

| | π*0.6 (RECAP) | ours (`pi05_policy_finetune.py`) |
|---|---|---|
| tag form and position | text, "after ℓ̂ but before the (discretized and continuous) actions" (V-B) | text, `"{prompt}. Advantage: {tag}"` |
| discrete action tokens | **yes**: "trains the entire model end-to-end on continuous actions and discretized tokens (including actions discretized via FAST)" (V-A); CE on FAST tokens, conditioned on I_t (Eqs. 4, 9) | **none** |
| flow-matching action expert | 860M, conditioned on I_t (f_θ(·, I_t, …), Eq. 4); attends to the backbone | same structure, the only loss |
| gradient from action expert into backbone | **blocked**: "a stop gradient to prevent the flow-matching action expert from impacting the rest of the model" (V-A, Knowledge Insulation) | **open** (no stop-gradient); LoRA backbone trained only by the flow loss |
| where the tag is first learned | pre-training (V-D), then task fine-tunes from the pre-trained checkpoint; first fine-tune stage fixes I_t = True (SFT) | a new 5k-step single-task fine-tune |
| dropout / guidance | 30% (A-F); β = 1 default, CFG applied to flow sampling (Eq. 13); CFG "would not affect the autoregressive part" (fn. 2) | 30%; β 1.5 / 2.5 tested |

*Interpretation (ours, not stated in the paper).* In π*0.6 the backbone learns what the tag means through
next-token cross-entropy on discrete action tokens placed directly after it: to predict those tokens the backbone
must carry the tag into the representation of the actions. The action expert then reads a backbone that already
encodes the tag, and its own loss cannot alter that representation. Our recipe has no discrete action loss, so the
only signal that can teach the tag is the flow loss through the action expert's attention to the prompt tokens, a
weaker and indirect path. This fits our measurements (tag effect 2% of noise; the fine-tuned model reads text less
than base, `reports/2026-11-06-tag-deep-dive/`). Options it suggests: (a) a route that feeds the tag to the action
expert directly (AdaRMS, running); (b) add a discrete-action (FAST) CE loss on the tokens after the tag, as π*0.6
does. (b) changes the training objective, so it needs Karen's go.

## 2. Problem by problem

### 2a. The policy ignores the tag

- **RECAP's tag is learned in pre-training, not at task fine-tuning.** The policy sees the indicator across
  "tens of thousands of hours" of multi-task data before any task fine-tune. The Dissecting paper skipped RECAP as
  a condition for this reason: "its advantage conditioning is introduced during pre-training ... and has been
  reported to yield more modest gains over supervised fine-tuning." We introduce a new text token in a 5k-step,
  single-task fine-tune. No source shows a tag being learned from scratch at our scale; RLinf (30k steps, batch
  512, 4,096 rollouts) is the closest and reports one run.
- **Theory says a tag is ignored when it is not predictable from the action.** Brandfonbrener et al.: the
  conditioned policy is π(a|s) = β(a|s) · P(tag | s, a) / P(tag | s). If the tag barely depends on the action given
  the state, the policy "simply defaults to the behavior" (*partial*). ESPER: conditioning on a realised outcome
  conditions on luck; the conditioning statistic must be independent of environment stochasticity (Theorem 2.1),
  i.e. an expected value. Our oracle-outcome tag is the realised outcome; our ResNet tag is mostly V noise. Both
  carry little information about the action taken, which predicts exactly what we measured.
- **Dropout + guidance.** CFGRL proves conditioning alone (w = 1) improves on the data policy only with exact
  advantages, and that w > 1 improves further ("performance generally improves as the CFG weight increases").
  RECAP and RLinf train with dropout so guidance is available. Guidance amplifies the conditional–unconditional
  difference; with our 2% difference it is a cheap test, not a cure.
- **Positive-only conditioning (RLinf).** Negatives are trained unconditionally, positives with the tag: the policy
  learns "top 30%" against "all data" rather than two tokens.
- **Diagnostic others use**: RePO-VLA swaps the inference token and reports success 46.7% → 73.3% (*partial*);
  ESPER plots target vs achieved return. Ours (action shift, loss flip) is stricter and already says "unused".

### 2b. Fine-tuning lowers success

No source reports this; every pipeline anchors to demonstrations or a strong initial checkpoint. RECAP retrains
the policy "from the pre-trained checkpoint, rather than ... from the last iteration ... useful for avoiding drift",
on all task data plus demos (tag fixed positive). RAPolicy samples batches 50:50 from a demo buffer and the online
buffer (*partial*). RLinf uses lr 1e-5 with 5,000 warm-up steps. Dissecting starts from an SFT checkpoint on ~40 h
of demos. Our arm A / C train only on our own rollouts, without the 50 moka demos, at LoRA lr 1e-4. The running
R2 (10× lower lr) and R3 (checkpoints) recipes test the learning-rate part; nothing we run tests demo mixing.

### 2c. Noisy labels

- **Estimator.** Dissecting compares critics on offline ordering checks (do human-takeover windows score below the
  preceding policy window; do "advancing" segments score above "stalled" ones): n-step TD beat Q − V for every
  critic (R_prog 0.81–0.95), and the MC critic (RECAP's choice) and the Q − V variants were "generally closer to
  chance". RECAP itself validates its value function only qualitatively (Fig. 4: "correctly identifies mistakes ...
  as well as the speed of progress"). Nobody reports inter-critic agreement like our κ.
- **Calibration.** Dissecting normalises advantages within bins of V ("Value" calibration: z = (A − μ_bin) /
  max(σ_bin, 0.1σ)), which cuts the dependence of A on the subtask (η² 0.092 → 0.026) and, in ablation, helps the
  robot. Their stated reason: "stalls, and recoveries cause temporal bins to mix different progress levels,
  whereas state value more closely tracks task progress."
- **Binary vs continuous.** Dissecting (4 real tasks, 20 trials each): continuous weights exp(clip(z, −2, 2)) 74%
  success vs hard filter at the median 50% vs SFT init 11%. RAPolicy also weights (exp(clip(A/0.5, −3, 3)), twin-Q
  min). RECAP argues against weighting: AWR "leads to much slower policies"; CRR found binary ≈ exponential on easy
  manipulation tasks. Weighting has no tag to ignore.
- **Smoothing V** is not used by any source; GAE-style multi-step averaging is the standard variance reduction.
  Our smoothing result (flips 0.96 → 0.34/s at k = 9, timing kept) is ours and untested elsewhere.

### 2d. Confident systematic value errors

- **Subtask structure.** SeeQ: task-level MC values are "noisy, potentially reflecting spurious correlations";
  subtask-scoped values put the largest value change at the subtask switch (without it, drops lag 4–24 frames).
  Ep 139's error sits at exactly that switch; our B1 subtask-scoped ResNet (18313821 / 22) is queued.
- **Value-binned calibration** (above) also removes systematic per-stage offsets.
- **Pessimism / ensembles.** RAPolicy uses the minimum over twin Q heads; averaging ResNet and expert V halved
  ep 139's negative share here (69% → 38%).
- **Distributional loss.** HL-Gauss (Stop Regressing) spreads each target over neighbouring bins instead of a
  one-hot bin; reported gains in scaling, not calibration specifically. Ours is one-hot CE.
- **Pre-training V on large robot data** (RECAP, SeeQ) is the main lever both papers use; our demo pre-training was
  1,693 demos.

## 3. What this suggests, in order of cost [ours, to decide]

Offline, no policy training:
1. **Is the tag predictable from the action?** Train a small classifier for the tag from (state) and from
   (state, action chunk) on train frames. If the action adds nothing, no tagged fine-tune can use it
   (Brandfonbrener Eq. 3). Run for each label set (current, 30% / 40% positive, smoothed, value-calibrated).
2. **Value-binned calibration** (Dissecting) on our A, scored with our noise metrics (κ, stall separation, flips) and
   their ordering checks (stall vs advancing, drop windows).
3. **Critic acceptance tests** before any policy run, as Dissecting does: stall-vs-advancing ordering accuracy and
   drop-window ordering for each value function (ResNet, expert, mean, B1 when done).

Policy runs (one GPU each; after R2/R3 say which recipe does not hurt):
4. **RECAP-faithful tag run**: ~30–40% positive, 30% tag dropout, the 50 moka demos with tag positive, eval at β = 1
   and with guidance β ∈ {1.5, 2.5} (one training run, several evals). Control: arm A with demos, same recipe.
5. **Advantage weighting instead of a tag** (Dissecting / RAPolicy): weights exp(clip(z, −2, 2)) from value-binned
   z, same recipe and data. Sidesteps tag-ignoring entirely.
6. Value improvements feed both: B1 subtask-scoped V, a ResNet seed ensemble (+ expert), HL-Gauss targets.

What the literature does not settle: whether a tag can be learned at our data scale without RECAP-style
pre-training; how binary tags compare with weights under label noise; any threshold sweep (RECAP reports only
its per-task choices).

## Sources

- π\*0.6 / RECAP: https://arxiv.org/abs/2511.14759 (HTML v2 https://arxiv.org/html/2511.14759v2)
- CFGRL: https://arxiv.org/abs/2505.23458, code https://github.com/kvfrans/cfgrl
- ESPER: https://arxiv.org/abs/2205.15967
- RCSL: https://arxiv.org/abs/2206.01079
- RLinf RECAP: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html
- OpenPIE: https://github.com/qrafty-ai/pi-openpi
- Dissecting Advantage-Guided Post-Training: https://arxiv.org/abs/2609.28161
- RAPolicy: https://arxiv.org/abs/2609.22888
- RePO-VLA: https://arxiv.org/abs/2605.09410
- CRR: https://arxiv.org/abs/2006.15134
- Stop Regressing (HL-Gauss): https://arxiv.org/abs/2403.03950
- SeeQ: arXiv 2609.22085 (summary in docs/libero_plan.md, 2026-11-02)
