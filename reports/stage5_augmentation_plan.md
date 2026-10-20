# Stage 5 plan: image augmentation vs. "the readings before the end look random" (2026-10-19)

**The concern (mentor):** from about 10 s to the end, V looks clean and reasonable. Before that, the
readings look random or arbitrary (e.g. `reports/figures/stage4/lowlr_8k/01_*_dist.png`, `10_*_dist.png`).

## What the distribution plots already tell us

Before the end, the model predicts **two humps**: success (above −0.5) and failure (below −0.5). The
success hump usually sits on the target line, and V is the average of the two humps. So the
"randomness" has two separate parts, and augmentation can only address one of them:

| part | example | correct behaviour? | can augmentation help? |
|---|---|---|---|
| (a) **Real outcome uncertainty**: both humps present because one frame can't show whether the run will succeed | failure r04/20 stays two-humped for all 18 s; frame-0 V ≈ the round's average target | **yes**, it is the honest answer | **no**. Needs more information (short history, elapsed time), not invariance |
| (b) **Flicker**: the balance between the humps jumps between neighbouring frames that look almost the same | r04/39 at 7 s and 8.4 s: the failure hump darkens for a few frames, V drops to −0.5 and back | **no**, nothing in the scene changed that much | **maybe**. If V reacts to small pixel changes (glare, slight motion), augmentation teaches it to ignore them |

So the goal is narrow: **reduce (b) without hiding (a)**, and without losing accuracy.

## Step 0: measure the baseline (no training; needs the full-val distributions)

Job 17973560 (`vl.value_dist --run lowlr_8k`, all val frames) is queued on humanoid. When it lands,
compute, on frames **more than 2 s before T** (the "before the end" region; the last 2 s are the
clean part):

1. **Flicker of V:** mean |ΔV| between consecutive frames. Reference: target slope 0.00056 per
   frame; baseline ≈ 0.022–0.025 (measured from val_predictions).
2. **Flicker of the outcome belief:** mean |Δp_succ| per frame, p_succ = mass above −0.5.
3. **Distribution change per frame:** mean total-variation distance ½·Σ|p_t − p_{t−1}|.
4. **How much of V's flicker is (b):** R² of ΔV ≈ Δp_succ × (V_if_succ − V_if_fail) (already in
   `vl.value_dist --plots`). High R² means the jumps are hump-balance flips.
5. **Within-hump accuracy:** on success episodes, |V_if_succ − v| (does the success hump track the
   target?), to check that (a) is just averaging, not a wrong success estimate.
6. **Smoothing reference:** the same metrics for V smoothed over 0.5 s and 1 s. Augmentation is only
   worth it if it beats what a moving average gives for free, at the same accuracy.

If metric 4 shows the jumps are *not* hump flips (low R²), the flicker is inside a hump and the plan
still applies. If flicker is already small relative to (a), augmentation is low priority and the
history ablation goes first.

## Step 1: augmentation arms (training only; validation never augmented)

Same recipe as the baseline (`baseline_lowlr`, 8k steps, both cameras), 3 seeds each, one change per arm [ours]:

| arm | what | why |
|---|---|---|
| A0 | none (= `lowlr_8k`, done) | reference |
| A1 **photometric** | per camera, independently: brightness ±0.3, contrast ±0.3, saturation ±0.3, hue ±0.03 | glare and lighting vary between rounds and frames; also may weaken the round cue in `side_1` |
| A2 **geometric** | random shift of the fixed crop box by up to ±24 px and scale 0.9–1.0 before the resize, independently per camera | stops V keying on exact pixel positions; small enough that the marker, gripper and holder opening stay inside (checked on the Stage 3 crop sheets before running) |
| A3 **both** | A1 + A2 | if both help, check they add up |

Not used [ours]: horizontal flips (left vs right matters for the wrist camera and the side view),
large rotations, cutout/erasing (could hide the marker tip, which decides "seated").

Implementation: an `augment` block in `configs/train.yaml` sections, applied in `FrameDataset` on the
uint8 crop in the workers (the crop shift is applied to the box before `crop_resize`, so decoding stays
exact). Tests: augmentation off ⇒ byte-identical to today; on ⇒ shape/dtype unchanged; the shifted box
stays inside the 640×480 frame; validation dataset never augments. Cost: 9 runs × ~20–30 min on L40S.

If an arm's val CE is still improving at 8k steps (augmentation regularizes, so it may need longer),
rerun that arm once at 16k and report both.

## Step 2: decision rule (set before seeing results)

Adopt an augmentation if, averaged over 3 seeds:
1. **Accuracy not worse:** val CE and MAE within 2 sd of the baseline (CE 3.730 ± 0.002, MAE 0.121 ± 0.004),
   and within-round AUC at 1/5/10 s not lower.
2. **Less flicker before the end:** pre-end |ΔV| and |Δp_succ| at least **25% lower** than the
   baseline, and lower than the 0.5 s smoothed baseline at equal or better shape MAE.
3. **Uncertainty kept where it is real:** the failure and stall episodes keep both humps (look at
   the same 11 `_dist` plots), and the seated-vs-hovering AUC stays ≈ 1.0.

Report also: the frame-0 round cue (does A1 weaken it?) and the drops when the arm leaves the side view.

## What this will and won't answer

- **Will:** whether the jumps between similar frames are fixable by making V less sensitive to small
  image changes.
- **Won't:** the two-hump uncertainty itself. That is the expected answer from one frame. The Stage 5
  "current frame vs short history" ablation (and an elapsed-time input) is the test for that part,
  together with reading p_succ next to V. A temporal-consistency loss between neighbouring frames
  would target flicker more directly but changes the loss, so only with Karen's and Lars's OK.
