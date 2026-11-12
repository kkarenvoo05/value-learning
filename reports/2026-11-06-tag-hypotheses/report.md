# Why the policy ignores its tag: gaps and new hypotheses (2026-11-06)

**Question (Karen).** The other session's tag deep dive (positive controls 18352944 / 18352946, prompt test 18352820)
is testing whether a tag can be learned at all. What else could be wrong that those tests do not cover? Checked
against the papers (RECAP / π\*0.6, SeeQ, CFGRL, knowledge insulation, open RECAP reproductions, prompt-vs-AdaRMS
conditioning) and our own data. No training was run. Builds on `reports/2026-11-05-literature-advantage-conditioning/`
(not repeated here).

Labels: **[data]** computed here from our files; **[S]** stated by a source; **[I]** inference. Sources marked
*agent-read* were read by a literature sub-agent and not re-checked line by line.

## Corrections and updates (2026-11-06, later the same day)

From the deep-dive session:

- **1c applies only to the old arm-C labels.** The labels of the latest runs do tag stalls negative:

  | labels | stall frames positive | other frames positive |
  |---|---|---|
  | old arm C (70%) | 73% | 70% |
  | faithful RECAP (value-binned 40%) | 6% | 45% |
  | scoped value-binned 40% | 14% | 44% |

  Those models still ignored the tag. So "labels point the wrong way" (H2) does not explain the latest runs, which
  makes a conditioning failure more likely.
- **H7 is done:** seed 1 of the step-1000 recipe scored 0.518 (seed 0 0.639, base 0.564), so "early stopping is
  the fix" does not replicate. Still open: re-evaluating base π₀.₅ with a new sampling seed (`MATCHED_NOISE=1`).
- **1a is consistent with steering ×2.0:** between-candidate spread 0.083 (×1.5: 0.065), CV gain +0.015
  [−0.011, +0.049]. Too wide to rule a gain in or out.
- **1d:** the injected-shift positive controls (18352944 / 18352946) test learnability of a per-frame tag. The
  per-frame truth from E1 is still the test of *real* advantages.

New, found while building E1 [data]:

- **Sim restore is exact for physics but not for observations.** `libero_steer_probe.Sim.restore` goes through
  LIBERO's `set_init_state`, which forces every robosuite observable to resample. That resets their sampling timers.
  The MuJoCo state stays bit-identical, but the robot-state observation drifts from an unbranched run: up to 0.011
  within 30 steps on ep 165.
  - The steer probe's fidelity check compared two restored runs with each other, so it could not see this.
  - Its candidate comparisons are still like-for-like, since every candidate is restored the same way. But
    "restore exact (3e-17)" holds only for the physics.
  - `ExactSim` in `scripts/libero_branch_value.py` also saves and restores the observables (timers, delays, last
    values, observation cache) and copies MjData field by field. The replay through two random branches then matches
    the recording to 4e-15.
- **Rendering differs by node.** On the interactive TITAN RTX (iris3), replayed images differ from the recorded ones
  by up to ~200 grey levels on ~5% of pixels, even without branching. The original replay check on the recording
  nodes found them exact. E1 logs the pixel difference at every branch point to check this on its node.
- **Failures look like gripper chatter plus stalls.** In 4 failed val episodes the gripper switched up to 15 times
  within ~35 steps, with 30–210 stall frames. In 2 successful ones: 3–4 clean switches, 0 stall frames.

Jobs submitted (one GPU lane, chained):

- **E4: 18353550** `scripts/libero_tag_keyframes.py`. Tag usage on 495 key frames (277 grip, 98 stall, 120 moving)
  from 60 val episodes. Models: faithful RECAP, scoped, and arm A frozen-SigLIP / original as floors.
- **E1: 18353523** `scripts/libero_branch_value.py`. Train episodes 165 350 425 321 336 106: 6 fixed states with
  success rates of 0.40–0.67, 3 successes and 3 failures. K = 6 continuations every 20 steps.
  - Analysis: `scripts/libero_branch_analyze.py --run <run dir>`.
  - 18353523 failed after 90 s. `ExactSim` also copied the constraint-solver arrays (`efc_*`, `island_efcind`),
    whose size follows the number of active constraints (nefc 99–183 over an episode), so a restore after the contact
    count had changed could not copy them. These arrays are rebuilt from qpos/qvel at every step; the warm start is
    `qacc_warmstart`, which has a fixed size and is still copied. They are now skipped.
  - Check: a full 520-step replay of ep 165 with 52 random-action branches matches the recorded robot state to
    1.4e-11, and the outcome matches. Resubmitted as **18355241** (same episodes and settings).

### Idea 3 built: the tag through AdaRMS (`scripts/pi05_tag_adarms.py`), not yet trained

- **Route.** `adarms_cond = time_emb + 40 · tag_emb[tag]`, where `tag_emb` is (2, 1024) with rows −/+ and adds
  nothing when there is no tag.
  - Zero init, so step 0 is exactly π₀.₅ for every tag [ours].
  - The ×40 gives this one parameter an effective Adam lr of 1e-3 while the recipe's two lr groups stay unchanged
    [ours].
  - The tag is removed from the prompt (`src/vl/tag_route.py`), so it reaches the model only through adaRMS.
- **Same recipe.** Everything else is `pi05_policy_finetune.py`, run unchanged through three patched functions:
  data, labels, dropout, demos, LoRA, checkpoints, resume. Against faithful RECAP, only the route changes.
- **Guidance.** CFG needs one K/V cache, since the prefix no longer depends on the tag.
- **Tools.** Existing tag tools work on these runs through `load_adarms`: tagged prompts are parsed at the input
  transform.
- **Checks.**
  - GPU selftest 18355558, 7 checks: step-0 equality, prompt stripping, sampling, random embedding moves only
    tagged samples, guidance at β = 0/1, gradient rows, save/load.
  - CPU tests: `tests/test_tag_route.py`.
- **Launch commands** are in `scripts/pi05_tag_adarms.sbatch`. MODE=train with the faithful RECAP args; MODE=eval
  for the 330 trials.
  - Cheapest first run: the injected-shift positive control through this route (same args as 18352944), to compare
    learnability with the prompt route.

### E4 result: the tag is ignored at key frames too (18353550) [data]

Run: `/iris/u/ankile/karenvo/libero/runs/tag_keyframes_18353550/` (`results.json`, `frames.parquet`). The
measures:
- tag / noise = RMS(a(+) − a(−)) / RMS between two noise draws, over the 5 executed steps × 7 dims;
- gripper flips = share of executed steps where the two disagree on open/close;
- earlier switch > 0 means the + tag switches the gripper earlier (range ±2).

Values are ratios with a 95% CI, bootstrapped over episodes.

| frames | measure | faithful RECAP | scoped | floor: arm A frozen-SigLIP | floor: arm A original |
|---|---|---|---|---|---|
| grip (277) | tag / noise | 0.009 [0.006, 0.012] | 0.009 [0.006, 0.012] | 0.016 [0.011, 0.021] | 0.008 [0.005, 0.012] |
| grip | gripper flips: tag vs noise | 0.2% vs 33.2% | 0.2% vs 33.1% | 0.2% vs 31.1% | 0.1% vs 31.6% |
| grip | earlier switch (noise \|·\| ≈ 0.18) | −0.0004 [−0.0019, 0.0008] | −0.0009 [−0.0022, 0.0010] | +0.0007 [−0.0015, 0.0038] | −0.0016 [−0.0033, −0.0003] |
| stall (98, 33 eps) | tag / noise | 0.041 [0.024, 0.056] | 0.023 [0.015, 0.038] | 0.055 [0.021, 0.110] | 0.053 [0.021, 0.107] |
| moving (120) | tag / noise | 0.021 [0.015, 0.032] | 0.023 [0.017, 0.034] | 0.045 [0.026, 0.076] | 0.050 [0.019, 0.099] |

Observations:
- At the moments the outcome is decided, the policy's own noise flips the gripper in a third of the executed steps.
  Swapping the tag flips it in 0.2%. That is the same rate as models that never saw a tag.
- The earlier-switch effect is about 1/200 of its noise scale. Its CIs cover 0 for both RECAP models.
- On stall and moving frames, the tag moves the actions less in the tagged models than in the never-tagged floors.
  On moving frames that is about half the floor's effect.

Interpretations [I]:
- "Ignored on average but used where it matters" is ruled out. The tag is ignored everywhere, so the earlier null
  result was not a dilution artefact.
- In the untagged models, the suffix text is a small out-of-distribution perturbation. The tagged models were trained
  with the suffix present or dropped (30% dropout) on the same frames. They may have learned to make their actions
  invariant to it, since nothing in the flow loss rewarded reading it (H1 / H3 in §3).
- The positive controls (18352944 / 18352946) are now the deciding test. If even an injected per-frame shift is not
  learned, the conditioning path is the problem rather than the labels.

## 1. Findings from our own data that change current conclusions

### 1a. The steer probe could not have detected the headroom it was looking for [data]

`reports/2026-11-04-steer-probe/analysis.json`: between-candidate SD of success (binomial noise removed) = **0.079**
(var 0.00625, 40 branch points × 8 candidates × 2 repeats). If a scorer could rank 8 candidates with that spread,
the expected best-of-8 gain is ≈ 1.42 × 0.079 ≈ **+0.11 per decision point**.

The cross-validated oracle (`cv_gain`) picks on **one 0/1 repeat** and scores on the other. The correlation of a
single binary repeat with a candidate's true success rate is about 0.079 / √(0.079² + 0.25) ≈ 0.16, so the expected
CV gain is ≈ 0.11 × 0.16 ≈ **+0.02**. The measured +0.006 [−0.028, +0.042] is consistent with that. So the probe could
not tell "no headroom" from "+0.1 headroom". **The B2 stop rule ("oracle gain ≈ 0 → stop") was triggered by low
power, not by evidence of no headroom.** The point estimate of var_between is itself noisy (40 points), so this
reopens the question rather than answering it.

The same quantity bounds RECAP [I]. CFGRL-style conditioning reweights π(a|s) by p(positive | s, a). The available
improvement is the variation of Q(s, a) across the actions π already samples: the same variance the steer probe
measures.

### 1b. Most outcome variance is within an init state, and the sim is deterministic [data]

`tables/moka_fixed_fresh/episodes.csv`, 450 train rollouts on 30 fixed states (15 each): success 0.598; per-state
rates span 0–1.0, but **72% of the outcome variance is within a state**. Restore is exact (steer probe, 5e-17), so
given the init state, success depends only on the policy's sampling noise. Outcomes are therefore causally tied to
actions. The open question is *where* in the episode they are decided. The steer probe sampled fixed steps 40 / 130
/ 220 / 310, not the moments when outcomes diverge.

### 1c. Every failure is a timeout, and the current labels call stalls positive [data]

- All **384/384** train failures are timeouts. `goals_end` shows that most end with one goal predicate unmet
  (205 "011", 96 "101", 83 "001").
- Successes take a median of 409 of the 520 allowed steps.
- So the task is mostly lost by being **slow or stuck**, not by a catastrophic event.

This is RECAP's headline regime: throughput ×2 [S]. But `reports/2026-11-04-label-rules/` shows the current rule
(R0) labels stalls only 26–38% negative, so **most stall frames are tagged positive**. For a timeout-dominated task
that is the frame type the tag most needs to call negative. *To check:* how the faithful-RECAP labels (40% positive,
value-binned) treat stalls.

### 1d. The oracle-tag run does not show that "conditioning is too weak" [data / I]

That oracle tag was **per-episode** (rollout outcome), as noted in `docs/CURRENT.md`. Within an episode it is
constant, so at a fixed state it carries information about the future noise, not about the current action. A
well-fit model *should* ignore it [I, RCSL]. The positive controls test learnability of a synthetic shift. Nothing
yet tests a **correct per-frame** advantage.

### 1e. Training budget is ~100× below the working references [data / S]

- Ours: batch 32 × 5000 steps = 160k samples (`pi05_policy_finetune.py` defaults), ≈ 0.4 epoch of the arm-C frames.
- RLinf RECAP: 30k steps × batch 512 = 15.4M samples, 5k warmup, lr 1e-5, full model [code, *agent-read*].
- The open LIBERO reproduction (theblock1226/pi05-recap-libero-repro) measured a tag effect only at step 30k (task 0:
  plain 25/50 vs "Advantage: positive" 32/50, single runs). At 6k–9k it was not better [S, *agent-read*].
- The two small-budget reproductions found got worse than SFT: GordonLuQwQ/PI0.5-RECAP (LoRA, batch 8, 5k steps)
  and BurningDawn8888/lerobot-pi05-acp (round 1 60.5% vs 77.8% baseline) [S, *agent-read*].

[I] So a null at step 5000 is the expected result for a weak conditional signal, even if everything else is right.
This collides with our other result that longer training hurts the policy. **The two fixes conflict**: early
stopping keeps the policy good, while tag learning needs ~100× more samples.

## 2. Mechanism: how the tag reaches the actions

| | π\*0.6 RECAP [S] | RLinf RECAP [code] | ours |
|---|---|---|---|
| tag route | text in prefix, after subtask | text after task | text after task ("…. Advantage: x, State: …") |
| loss that teaches the VLM the tag | **FAST-token CE** (discrete action loss) + flow; KI stop-grad on prefix K/V for flow | flow MSE only, full model | flow MSE only, LoRA r16 on frozen Gemma |
| tag seen before task fine-tune | yes, in pre-training | no (`pi05_base`) | no (`pi05_libero`) |
| negatives | tagged "negative" | **never tagged** (positive vs plain) | tagged "negative" |
| dropout | 30% | 10% | 30% (faithful run) |

- **[S] π\*0.6:** the VLM learns what the tag means through the FAST cross-entropy, a per-token discrete action loss.
  The action expert then reads an already action-relevant K/V. openpi has no FAST head for π₀.₅, so only the noisy
  flow gradient can teach the VLM what the tag means.
  - Checked in openpi `pi0.py`: there is no stop-gradient, so the LoRA does receive the gradient.
- **[S] ALIGN (arXiv 2610.26455, *agent-read*):** this paper compares injection points on a π₀.₅-style flow model at
  5k steps, batch 256.

  | injection point | ARX bowl | Franka key |
  |---|---|---|
  | prompt suffix | 70.0% | 40.0% |
  | AdaRMS fused with the flow-time embedding | **90.0%** | **56.7%** |
  | outcome token after the prefix | 76.7% | 53.3% |

  Their conditional-velocity-field separation correlates with success (r 0.93–0.95). They also report thresholded
  V-difference labels under the prompt interface at 50.0%, against 70.0% for direct labels.
- **[S] LIBERO-Plus / LIBERO-PRO:** LIBERO-fine-tuned VLAs largely ignore their instruction (blank or corrupted
  instructions barely change behaviour). π₀.₅ was not in their tables. The queued prompt test 18352820 covers this
  for us.
- **[S] CFGRL:** conditions through a learned embedding, not text; gains need w > 1.

## 3. Hypotheses, ranked, with the test for each

| # | hypothesis | covered by the deep dive? | cheapest test |
|---|---|---|---|
| H1 | Little per-chunk advantage exists at most frames; it is concentrated at a few decision moments (1a, 1b) | no | sim-branching oracle V (E1) |
| H2 | Labels point the wrong way on the frames that matter (stalls positive, 1c) | partly (label-rules report) | stall-label table for the faithful labels; E1 as ground truth |
| H3 | Budget ~100× too small for a weak tag (1e) | partly (controls at 1k/3k/5k) | E3: train the tag path long with the base frozen |
| H4 | The prompt route is weak in this model (frozen LLM + LoRA, LIBERO ignores text) | yes (prompt test) | E2: AdaRMS / time-embedding tag |
| H5 | Positive-vs-negative is a worse contrast than positive-vs-plain | no | RLinf's positive-only conditioning (label-free change) |
| H6 | The tag-usage metric dilutes a local effect (averaged over all frames, at ~2% of noise) | no | recompute tag usage on decision frames only (CPU, existing checkpoints) |
| H7 | The R3 0.639 is eval noise | no | base π₀.₅ re-eval with a new sampling seed; 2nd seed of step 1000 |

## 4. Proposed experiments [ours, to decide]

- **E1. Sim-branching oracle value (~2–3 GPU-h).**
  - Setup: on ~10 train episodes (mixed outcomes), restore the saved state every 40 steps and run K = 6 base-policy
    continuations to the end. P̂(success | s_t) gives the true V(s_t) (SE ≈ 0.2 per point).
  - It yields:
    1. where outcomes are decided (the jumps in P̂), so the steer probe can be rerun at those frames with ≥ 8 repeats;
    2. a ground truth to score the ResNet / B1 / faithful labels against (agreement with the truth instead of κ
       between two critics);
    3. if extended, a correct **per-frame oracle tag** for the policy: the test that 1d says is still missing.
  - Uses the existing restore code in `scripts/libero_steer_probe.py`.
- **E2. AdaRMS tag injection.**
  - Change: a learned 2-way embedding (positive / negative-or-none), zero-initialised and added to the action
    expert's flow-time conditioning. Step 0 is exactly the base model, and CFG works as with the prompt tag.
  - Comparison: same labels and recipe as faithful RECAP, so only the route differs.
  - Code change in our openpi wrapper, plus the selftest.
- **E3. Long tag training without drift.**
  - Change: freeze the base (or the arm-A step-1000 checkpoint) and train only the tag path (E2's embedding plus the
    action expert's LoRA) for 20–30k steps. Optionally add an L2 term toward the base weights.
  - Goal: resolve the 1e conflict.
- **E4. CPU re-analyses, no new jobs:**
  - tag usage restricted to gripper-transition and stall frames (H6);
  - a power calculation for the steer probe's headroom design (repeats needed to detect +0.05).
- **E5.** Positive-only conditioning (H5) as a variant of the next tag run.

Not recommended yet: building the SeeQ Q-function. E1 says first whether, and where, choosing between candidates can
matter.

## Caveats

- 1a's +0.11 is an upper bound for a perfect scorer, taken from a noisy variance estimate over 40 points.
- ALIGN, the reproductions and the RLinf code were read by a sub-agent.
- The deep-dive report (`reports/2026-11-06-tag-deep-dive/`) was not in this checkout, so the comparison relies on
  its summary.

## Sources

- π\*0.6 / RECAP: https://arxiv.org/html/2511.14759v2 (Sec. V-A/B, App. A-F)
- Knowledge insulation: https://arxiv.org/html/2505.23705
- CFGRL: https://arxiv.org/html/2505.23458
- SeeQ: https://arxiv.org/html/2609.22085, model card https://huggingface.co/seeq-iclr/SeeQ-3B
- ALIGN: https://arxiv.org/html/2610.26455
- LIBERO-Plus: https://arxiv.org/html/2510.13626v1; LIBERO-PRO: arXiv 2510.03827
- RLinf RECAP: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html, `rlinf/models/embodiment/openpi_cfg/`
- Reproductions: github.com/theblock1226/pi05-recap-libero-repro, github.com/GordonLuQwQ/PI0.5-RECAP,
  github.com/BurningDawn8888/lerobot-pi05-acp
- ForesightFlow (advantage-weighted flow matching alternative): arXiv 2606.04968
