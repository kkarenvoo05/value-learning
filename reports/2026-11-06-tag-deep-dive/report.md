# Why does the policy ignore its advantage tag? (2026-11-06, deep dive, part 1)

**Question (Karen).** Every tagged fine-tune ignores its tag: faithful RECAP (18338571; value-binned 40% labels,
30% tag dropout, demos tagged positive, frozen SigLIP) and the subtask-scoped version (18338529) both move their
actions between the + and − tag by ~2% of their own sampling noise. Is this because (a) the labels are too weak
to learn from, or (b) the model cannot learn a per-frame text tag in this recipe at all?

## Part 1: what is already ruled out (no new jobs)

| check | result |
|---|---|
| tags reach training | `tags_seen` 18338571: 9,153 pos / 13,188 neg / 9,659 none (resumed segment); 18338529: 47,581 / 64,453 / 47,966. 40% positive, 30% dropout as configured |
| prompt tokenisation | `pi05_policy_finetune.py --selftest` (2): +, −, none prompts tokenize differently, none truncated |
| tag usage, RECAP | tag/noise 0.022 [0.018, 0.027]; flipped-tag loss − correct +1.0e-5 [−0.5e-5, +3.0e-5] (18338865) |
| tag usage, scoped | tag/noise 0.023 [0.019, 0.028]; flipped − correct +0.4e-5 [−0.1e-5, +1.0e-5] (18352443) |
| **floor: arm A, never trained with a tag** | tag/noise **0.026** [0.020, 0.033] (same job) |
| label flicker | value-binned 40%: 0.97 flips/s; scoped: 1.04 flips/s (policy replans every 0.25 s, chunk = 0.5 s) |

Both tagged runs respond to the tag **no more than a model that never saw one**: nothing was learned.

## Part 2: how far *should* the action move with the tag? (`scripts/libero_tag_ideal_shift.py`)

**Setup.** Interactive TITAN RTX, 8 min. 900 train rollouts on the label grid (104,049 frames). An MLP regresses the
executed chunk (5 steps × 7 dims, raw units, as the tag-usage action test) on the tag-predictability state S
(ImageNet ResNet-18 features of both cameras, robot state, t, pots on the stove, both value functions' V) plus the
tag (±1); 5 folds by episode. **Ideal shift** = out-of-fold RMS of f(S, +1) − f(S, −1): the move a learner that fits
the data exactly would make between tags at the same state (the analogue of the action test's `tag_pos_vs_neg`).
Tag-free fit R² 0.905.

### Observations

| labels | ideal shift (raw) | arm dims | gripper | action info beyond state (gain) |
|---|---|---|---|---|
| RECAP faithful (value-binned 40%) | **0.0151** [0.0149, 0.0154] | 0.0057 | 0.0344 | 8.1% [7.3, 8.8] |
| scoped value-binned 40% | 0.0150 [0.0147, 0.0152] | 0.0059 | 0.0334 | 7.4% [6.8, 8.0] |
| oracle outcome (per episode) | 0.0112 | 0.0037 | 0.0265 | (2.3%, earlier report) |
| control+: stall flag | 0.0296 | 0.0144 | 0.0629 | (12.9%) |
| control−: random 40% (estimator floor) | 0.0039 | 0.0018 | 0.0087 | — |

What the trained policies actually do (raw `tag_pos_vs_neg`, val frames): RECAP **0.00075**, scoped **0.00079**,
never-tagged arm A **0.00102** (18338865, 18352443).

1. The data supports a tag-dependent action difference of 0.015 raw, about **20× what the tagged models show**,
   and clearly above the estimator floor (0.004).
2. Even if an oracle-sized part of it (0.011) were state confounding rather than action, the rest is
   √(0.0151² − 0.0112²) ≈ 0.010, still ~13× the models' move.
3. The tag-associated difference is mostly the **gripper** (0.034 vs 0.006 on the arm): in the data, + and − frames
   at the same state differ mainly in when the gripper opens or closes.
4. Scoping the labels to the subtask changes neither the information (7–8%) nor the ideal shift.

### Interpretation

- Within the limits below, (a) alone does not explain it: the labels are weak, but the models learned *none* of
  even this weak signal, not a small part of it. That points to (b), the conditioning failing to learn a tag of
  this size in 5,000 steps.
- Caveat that keeps this from being decisive: S is weaker than what π₀.₅ sees (the residual of our fit is 2.5×
  π₀.₅'s own sampling spread: 0.106 vs 0.043). Part of what the tag "predicts" here could be state that π₀.₅
  already sees, and a tag predictable from what the policy sees is ignored by design (RCSL). So the ideal shift is an
  upper bound for a π₀.₅-strength learner, by an unknown factor.
- That caveat also sank my first positive-control design (tag = sign of the action's deviation from S's
  prediction): π₀.₅ could largely predict that deviation from the image. Its label sets were deleted unused.

## Part 3 (running): decisive controls

1. **Injected-shift positive controls** (`--tag-action-shift`, new): random 50% tags (`labels/control_random_p50`),
   and on tagged samples action dim 0 (x) of the whole training chunk is moved by +δ (positive) / −δ (negative).
   The tag is then informative by construction, independent of the state. Same recipe as faithful RECAP (frozen
   SigLIP, 30% tag dropout, 5,000 steps, save every 1,000), no demos.
   - strong, δ = 0.1 (0.5 std of dim 0): a fully learned tag moves the action by 2δ√(5/35) = **0.076** raw
     (1.8× the policy's own noise). 18352944 → tag check at steps 1000 / 3000 / 5000: 18352945.
   - RECAP-sized, δ = 0.02: a fully learned tag moves it **0.015** raw, the RECAP labels' ideal shift.
     18352946 → 18352947.
   Reading: strong learned + RECAP-sized learned → the conditioning works and the RECAP signal is mostly state
   π₀.₅ already sees (labels are the problem). Strong learned, RECAP-sized not → a tag this weak is below what this
   recipe can pick up in 5,000 steps (need a stronger tag, longer training or a different conditioning path).
   Neither → the text-tag pathway itself does not train (recipe/architecture problem).
2. **Prompt test** (18352820, `libero_tag_usage.py --prompt-test --base`): does π₀.₅-LIBERO change its action when
   the task instruction is swapped (same objects / unrelated task) or removed? The tag is prompt text, so a policy
   that barely reads its prompt cannot be expected to read the tag.

### Prompt test result (18355145; 18352820 crashed on a `libero_tag_usage.py` refactor bug, fixed and re-run)

Same 320 val frames and noise draws (k 4); effect = RMS action change vs the true instruction, over the same-prompt
sampling noise (normalised space; raw in brackets).

| model | near (same objects) | far (other task) | empty prompt | + vs − tag |
|---|---|---|---|---|
| base π₀.₅-LIBERO | 0.97 (0.72) | 1.38 (1.17) | 1.10 (1.04) | 0.044 (0.037) |
| RECAP tagged (18338571) | 0.68 (0.48) | 0.98 (0.79) | 0.83 (0.74) | 0.022 (0.018) |

*Observations.* (1) Both models read the prompt: swapping the whole instruction moves the action by about one
noise-width, ~30–45× the tag's effect. The gripper almost never flips (≤ 0.4%): the instruction steers the arm path,
not the grasp timing. (2) The tag-trained model is *less* sensitive to text than base, both to the instruction
(~0.7×) and to the tag suffix (0.022 vs base 0.044, which never saw a tag). *Interpretation:* the prompt pathway
works, so "π₀.₅ ignores text" is ruled out; fine-tuning on one task with a meaningless-to-it suffix taught it to
attend to text less, not more. Single checkpoint, one seed.

## Files

`ideal_shift_table.csv`, `ideal_shift_summary.json` (its `labels_written` lists the deleted residual-PC1 sets),
`ideal_shift.log`; code `scripts/libero_tag_ideal_shift.py`; control option `fd.sample(..., action_shift)` /
`--tag-action-shift` (test `test_tag_action_shift_moves_only_tagged_dim`); `libero_tag_usage.py --steps`,
`--prompt-test`, `--base`.
