# Advantage Guidance (Zhou & Levine, arXiv 2610.19783): what it is, what transfers, what we run

**Status 2026-11-11 06:00.** The zero-training test (contrastive guidance on the AdaRMS vb40 checkpoint) gives the first arm
that clearly beats base: **w 8: 0.658 vs 0.524, +0.133 [+0.070, +0.197] paired, all 330 trials**. Queued: a null
control for it (episode-shuffled labels, same recipe, w 8), the residual parameterisation (built, checks 1-4 pass),
and w 16. Action imagination: plan only, behind a gate (it needs a Q(s, a) that ranks pi0.5's own samples).

## 1. Zero-training test: contrastive guidance on the tag-dropout checkpoint

v = v_none + w (v_pos - v_neg) [AG Eq. 4]; v_none = the dropout branch of the same network [ours]. Run AdaRMS vb40
18355677 (rollouts + demos, tag dropout 0.3). 330 val trials (10 val states x 33, env seed 7), matched noise seed 1, so
every trial is paired with the same trial of base (18355478, 0.524). Paired bootstrap over trials (5000).

| sampling | job | success | vs base (paired) | action shift / noise (offline check) |
|---|---|---|---|---|
| no tag (w 0) | 18409259 | 0.385 | -0.139 [-0.206, -0.076] | 0 |
| plain + tag | 18355679 | 0.409 | -0.115 [-0.185, -0.048] | 0.12 (+ vs -) |
| w 2 | 18409260 | 0.348 | -0.176 [-0.245, -0.109] | 0.30 |
| w 4 | 18409262 | 0.497 | -0.027 [-0.091, +0.033] | 0.57 |
| **w 8** | 18432563 | **0.658** | **+0.133 [+0.070, +0.197]** | 0.99 |
| w -4 (sign control) | 18409263 | 0.230 | -0.294 [-0.355, -0.230] | 0.44 |

Read:
- Dose-response from w 2 up, and the sign flip hurts. A tag that was nearly ignored when sampled plainly (+ vs - =
  12% of the sampling noise) becomes useful when amplified about 8x.
- The checkpoint itself sits 14 points below base (w 0). So w 8 is worth about +30 points relative to its own
  unconditional branch.
- w 8 was +0.164 at 214 of 330 trials; the full 330 settle at +0.133. The first 92 replayed the failed attempt
  18409261 exactly (same outcome and step count on every pair: matched-noise evals are deterministic).
- Per state (w 8 vs base, 33 each): the gain is uneven. 42: 0.64 vs 0.12, 17: 0.97 vs 0.61, 13: 0.33 vs 0.18,
  25 / 32 / 48 up 0.15-0.21; 40 and 12 flat; 44: 0.45 vs 0.70 and 9: 0.55 vs 0.64 go down.
- w 1 dropped (cancelled 18434522, Karen 2026-11-11): it sits between w 0 and w 2 and would not change the read.

Open: is it the advantage labels, or just "amplify any learned + vs - difference"? Positives include the demos, and
vb40 labels are 52% positive in successful episodes vs 28% in failed ones. **Control queued:** the same recipe
with `control_shuffled_vb40` labels (vb40 label sequences permuted across episodes, derangement, seed 20261111).
It keeps the 41% positive share and the temporal clumping but has no relation to the episode's own states or outcome
(positive share 0.42 in successes vs 0.41 in failures; demos still positive). Train 18438131 -> w 8 eval 18438132.
If the control also gains, the effect is amplification of a demo/rollout difference, not advantage.

## 2. The paper, in the detail that matters for implementation

Setting: offline RL, **state-based** OGBench (20 tasks, 1M transitions, play data) + OCBench (6 tasks, 10-30k
transitions, near-expert demos), action chunks H 5 / 3, small MLPs (actor 512x4 / 1024x8), 0.5-1M gradient steps,
10 seeds. No images, no pre-trained policy. Our moka data (one policy's own rollouts, narrow) is closest to OCBench.
Code: github.com/zhouzypaul/advantage-guidance (`agents/advantage_guidance.py`, `utils/networks.py`; read at f110819).

### 2a. Why naive advantage conditioning fails (Secs. 4-5)
- **CFG on a continuous advantage matches a value; it does not maximise one.** With Y | s, a ~ N(A, σ²), CFG tilts by
  exp(-w (y - A)² / 2σ²): it penalises actions above the query as well as below (Eq. 3). Pure conditioning is the
  w = 1 case.
- **"Advantage alignment problem":** the naive conditional policy (RCP-A, FiLM) barely responds to y. Two
  diagnostics:
  - (i) cosine similarity of the loss gradient under true vs random labels is about 1;
  - (ii) E‖π(A90, ε) − π(A10, ε)‖² / E‖π(A50, ε) − π(A50, ε′)‖² < 0.5 for naive, > 1 for AG (squared ratio).
  - **We built both independently:** the true-vs-random-label check and the tag shift / noise ratio. Ours, un-squared:
    plain + vs - 0.12, i.e. about 0.015 squared, far below their naive policies.
- A transformer instead of FiLM does not fix it (A.4: output varies a bit more, gradients are still label-invariant).
  This matters because pi0.5 is a transformer, and our AdaRMS route (= AdaLN) is their conditioning choice already.
- The root cause (Sec. 6.3): about one action per state, so the conditional loss is minimised by the plain policy and
  gives no counterfactual supervision on how actions should change with y.

### 2b. Contrastive guidance (Sec. 6.1, Thm. 1)
- v = v_ref + w [v(y+) − v(y−)]. If p(y | A) is exponential family, h(y) exp(T(y) A − F(A)), the clean-action score
  is exactly that of π* ∝ π̂ exp(A/β) with **1/β = w [T(y+) − T(y−)]**. This is advantage maximisation under a KL
  constraint, not value matching.
- Conditions: π(a | s, y) must be the Bayes posterior of the **same** π̂ whose score is v_ref. The paper extends the
  result to intermediate denoising steps "as a practical approximation", as CFG does.
  - Our dropout branch = π̂ trained on the same data, so it fits.
  - Mixing base π0.5 as v_ref with a fine-tune's v± would not.
- Gaussian labels: T(y) = y / σ², so β = σ² / (w (y+ − y−)). Only the product matters: they fix y± = ±1 and σ_y
  (0.1 OGBench, 0.01 OCBench; σ = 0 also works) and tune **w only**.
  - Tuned w: OGBench 1-2.5, OCBench 1.5-4. Grids {0, 0.75, ..., 4} and {0, 0.5, ..., 4}.
  - Too large a w degrades performance, which they list as an open problem.
- **Binary tags are covered:** a Bernoulli with P(+ | A) = sigmoid(A/τ) is exponential family with T(y) = y/τ, so
  β = τ/w [ours]. Our labels relate to the true A* only weakly (corr 0.10, Part 4), i.e. a large effective τ.
  A larger w is therefore needed for the same tilt, consistent with w 8 being the first weight that helps.
- Pure conditioning on a high y collapses on narrow demo data (A.5, OCBench). There contrastive guidance is "crucial"
  (Fig. 8: +, versus equal on OGBench).
- Efficient form (code): h(y+) − h(y−) with the residual, so v_ref cancels in the guidance term. They compute the
  difference in feature space with one output projection.

### 2c. Residual parameterisation (Sec. 6.2; `ConditionalAndUncondFlowField`)
- v_cond(x, t, s, y) = **stop_grad(v_uncond(x, t, s))** + h(x, t, s, y).
- h is a **separate full-size network** (same 512 x 4 as the reference), not a small head. Its input is concat(s, x_t,
  sincos(t)). y enters through AdaLN on every layer: LayerNorm without affine, then scale/shift from an MLP(128, 128)
  of the scalar y. The output Dense uses the default init, not zero.
- v_uncond is trained **jointly** on dataset actions with loss weight 0.1 (`uncond_loss_weight`). The conditional loss
  covers dataset + imagined actions.
- Flow-time sampling: discrete uniform over {0, 1/T, ..., 1}, T = 10 (pi0.5 uses Beta(1.5, 1)). 10 Euler steps.
  Actions are clipped to [-1, 1] after sampling.
- Effect (Fig. 6-7): improves both diagnostics and success alone; better combined with imagination.

### 2d. Action imagination (Sec. 6.3, A.2)
- Per dataset state, K = 7 extra chunks are sampled from the **current unconditional policy**, clipped and
  stop-gradded. a_0 = the dataset action. K = 3 hurts; gains plateau after about K = 7.
- Labels: A = Q(s, a) − V(s) from **IQL**:
  - 2 Q's, min; target critic; expectile 0.9 OGBench / 0.5 OCBench;
  - chunked n-step backup;
  - **small critic 2x128** (4x512 overfits, Fig. 16).
- **Per-state normalisation:** Ã = (A − mean over the K+1) / (std + 1e-8). This beats batch- and dataset-level
  normalisation by a wide margin (Fig. 10, at K = 31). Plus Gaussian label noise of std 0.01.
- So y is the within-state rank of the action under Q. Only the critic's **within-state ranking** of the policy's own
  samples matters.
- The paper treats robustness to noisy advantages as future work (Sec. 8).

### 2e. Other results worth citing
- AG beats FQL / EDP / QGF / CFGRL / FAWR / RCP / CRR overall. Largest margins are on the hardest tasks: puzzle-3x3
  79 vs 64, OCBench block-double 91 vs 84. CFGRL is weak (binary filtering discards information).
- Best-of-N with Q: strong on play data at large N, but beaten by AG on demo data at any N (Fig. 5). This is
  consistent with our steer probe: best-of-8 oracle gain on moka +0.006 [−0.028, +0.042].

## 3. What is useful for us (ranked)

1. **Contrastive guidance: the most useful piece, already run, zero training.** It is the only arm that has beaten base.
   Next: the null control (queued), the peak of the w curve (w 16 queued), and a second seed of the checkpoint.
2. **Residual parameterisation: cheap and worth one test, but expect less than in the paper.**
   - Their mechanism: v_ref absorbs the y-independent part, so the conditional branch's only learnable signal is
     y-dependent.
   - Our naive fine-tunes already start from base, which already explains most of the y-independent part. A
     LoRA-on-pi0.5 "residual" initialised at the base would be the same function, init and loss as our naive tag runs.
   - So the faithful test is a **frozen** reference plus a **separately initialised** conditional net: built (§4).
3. **Action imagination: the paper's biggest lever, but its premise is the weakest on our task.** It needs Q to rank
   pi0.5's own K samples per state. Our evidence:
   - pi0.5's samples end 1.9 mm apart.
   - The simulator's own cross-validated best-of-8 gain is +0.006, also null at noise x1.5 / x2.0.
   - The per-decision A* is mostly noise (split-half 0.31).
   - Our best learned advantage reaches corr 0.10-0.14 with A* against a 0.69 ceiling.
   - Per-state normalisation rescales whatever within-state spread Q produces to unit variance. With a non-ranking Q,
     imagination trains on noise that looks like signal.
   - The w 8 result points elsewhere: a **consistent** shift of about one noise SD applied at every decision helps,
     even though a single chunk choice does not (oracle null). The useful variation sits in a policy-level direction
     outside base's sample spread, which imagination with π_p = base cannot propose.
4. **Continuous advantages with per-state normalisation:** only meaningful with imagination (otherwise there is no
   per-state group), so it waits with item 3.
5. **Reproducing their OGBench numbers:** their code runs as-is (JAX, state-based), but it tells us nothing about
   pi0.5. Not worth GPU time unless the mentor wants a sanity anchor.

## 4. Residual parameterisation: built and queued

`scripts/pi05_ag_residual.py` [ours, adaptation]:
- **Reference:** v = sg(v_ref) + h_θ, with v_ref = **frozen base pi05_libero**. Base is the behaviour policy of every
  rollout in `moka_fixed_fresh`, i.e. exactly π̂ of the theory. The paper trains v_ref instead.
- **Head h_θ:** a 512 x 4 AdaLN MLP as the paper's (LayerNorm without affine, MLP(128, 128) of y), applied per action
  token to [frozen action-expert output feature (1024; carries image, language, x_t and t), x_t token, sincos(t)].
  - Final layer zero-initialised [ours], so step 0 = base exactly.
  - About 2 M trainable parameters; nothing in pi0.5 trains, so there is no backward pass through Gemma.
- **Labels:** y = +1 / −1 from the vb40 tags, with y± = ±1 (paper defaults). Tag dropout 0 (the reference is
  separate).
- **Data:** same as AdaRMS 18355677 (rollouts + demos, vb40), so against §1 only the parameterisation changes.
- **Hyperparameters:** lr 3e-4 (the paper's actor lr), 5000 steps, batch 32.
- Same velocity-mix interface as TagPi0, so `pi05_contrast_check.py` and the eval sbatch work unchanged
  (`PY_SCRIPT=pi05_ag_residual.py` in `pi05_tag_adarms.sbatch`).
- Selftest:
  - step 0 = base for every tag (loss and sampling);
  - only `res_head` trains;
  - a random head moves + vs − but not none or w 0;
  - gradients, a fit check and a save/load round trip.
- **Comparison at matched action shift, not matched w** (w is not comparable across parameterisations).
  `FT_SHIFT_TARGET` in the eval sbatch sets w = target / slope from the run's contrast check. For AdaRMS this
  gives 7.7 for 0.99, close to 8.

Chain (`scripts/ag_residual_chain.sh`, one GPU slot used sequentially):

| step | job | what |
|---|---|---|
| selftest | 18441001 | `MODE=selftest PY_SCRIPT=pi05_ag_residual.py` (4th attempt, see below) |
| train | 18438185 | afterok selftest |
| check | 18438186 | shift / noise at w 1 2 4 8 16 (alignment diagnostic: the paper's AG > 1 squared) |
| eval 1 | 18438187 | residual at the shift of AdaRMS w 8 (0.99) |
| eval 2 | 18438188 | AdaRMS 18355677 at w 16 |
| eval 3 | 18438189 | residual at the shift of AdaRMS w 4 (0.57; AdaRMS there −0.03) |

Selftest attempts:
- 18438161: `nnx.LayerNorm(use_scale=False, use_bias=False)` stores `Param(None)` (flax 0.10.2), which broke the
  param count. Replaced with a functional LayerNorm.
- 18439043: iris10 GPU has uncorrectable ECC errors (JAX sees no CUDA device). iris10 is now in the default exclude.
- 18439175 (iris6): checks 1-4 pass (step-0 losses equal for +, −, none, and within 3e-4 of plain pi0.5; only
  `res_head` trains, 1.92 M params; step-0 sampling identical; random head moves + vs − by 0.12 RMS and leaves none
  and w 0 at base). Then host OOM (91 of 96 GB) in check 5: the gradient function closed over the frozen weights, so
  each compile embedded about 6 GB of constants. Fixed by passing them as arguments, as the trainer does.
  Resubmitted as 18441001: **all 6 checks pass** (iris5, peak host RAM 7.5 GB; grad norm + 5.2, − 6.5, untagged 0;
  40 Adam steps on one + batch 0.604 -> 0.010; save / load exact). Training 18438185 released.

Read:
- Residual beats AdaRMS at the same shift → a better-aligned direction (the paper's claim) on our data.
- Equal → the direction comes from the labels, not the parameterisation, and the label work (Part 4) is the lever.
- Also check the paper's diagnostic: does the residual's plain + vs − response exceed AdaRMS's 0.12?

## 5. Action imagination: plan, gated

Do not build it until G1 passes. Order and cost:
- **G0 (done, fails for π_p = base):** steer probe, 40 branch points x 8 candidates x 2 continuations. Between-
  candidate success SD 0.08; cross-validated oracle gain +0.006 [−0.028, +0.042]. The simulator cannot rank base's
  samples at one decision, so no learned Q can.
- **G1 (about 1 day):** a chunk-conditioned critic Q(s, a_{t:t+5}).
  - Architecture: ResNet-18 features as our V, the action chunk, a small 2x128 head (the paper's overfitting finding).
  - Training: IQL expectile 0.7, 2 heads with min, discount 0.999⁵ per decision.
  - Data: the base rollouts **plus the exploration rollouts (noise x1.5 / x2.0, wider actions) and every branching
    continuation** (many actions from the same states: the only within-state supervision we have), plus round 2 once
    collected.
  - Gate: within-state rank correlation of Q with continuation success at held-out branch points (K = 12 runs), CI
    above 0, scored next to the Part 4 ceiling.
- **G2 (if G1 passes, about 3 GPU-hours):** precompute K = 7 imagined chunks per training frame once.
  - Proposal: base, and separately the w 8-guided policy as a wider proposal [ours; the paper allows any π_p].
  - Score with Q; per-state normalisation with a floor on σ_s [ours], so near-identical samples do not get unit-
    variance labels.
- **G3:** train the residual head on continuous ỹ over dataset + imagined actions, contrastive guidance y± = ±1.
  Eval at matched shift as in §4.

Risk, stated up front: G0 already says one-decision differences between base's samples are not resolvable. Imagination
can only help if Q learns a ranking that the simulator's 2-continuation estimate misses, or if the wider proposal
adds real spread.

## 6. Queue changes tonight (GPU cap ≤ 6 incl. interactive)
- Cancelled 18434522 (w 1).
- Held 18433699: round-2 lane 3, `fbc_r1_s1` (round-1 recipe, seed 1), and with it its evals 18433700/01.
  Release with `scontrol release 18433699`.
- New: control 18438131 -> 18438132; residual chain 18441001 (selftest), 18438185-89.
- Round-2 lane 1 (`fbc_r2_s0`, 18433693) died on iris10 (ECC). Resubmitted as 18441002; its eval 18433694 now
  waits on it.
- New label set: `tables/moka_fixed_fresh/labels/control_shuffled_vb40` (README inside).
