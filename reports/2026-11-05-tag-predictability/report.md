# Is the advantage tag predictable from the action? (2026-11-05)

**Question (Karen).** The tagged policy ignores its tag. Theory (Brandfonbrener et al. 2022, arXiv 2206.01079,
Eq. 3): π(a | s, tag) ∝ π_β(a | s) · P(tag | s, a) / P(tag | s), so a tag that the action does not predict, given
the state, cannot change the policy. How much information about each candidate tag does the action carry beyond
the state?

**Setup.** `scripts/libero_tag_predictability.py`, interactive A5000, 11 min. 900 train rollouts, 104,049 value
frames. MLP classifiers (2 × 256, dropout 0.2, early stopping on 10% of the training episodes), 5 outer folds by
episode. State S = frozen ImageNet ResNet-18 features of both cameras + robot state + time + pots on the stove +
both value functions' V at t; S+A = S + the recorded next 20 actions (the advantage's lookahead); A = actions only.
Gain = (log-loss_S − log-loss_S+A) / H(tag): the share of the tag's entropy the action explains beyond the state;
95% bootstrap CI over episodes. Controls: the stall flag (defined from the next 20 steps of motion; must be
action-predictable) and iid random labels (must be 0).

## Observations

| labels | positive | AUC S | AUC S+A | state explains | gain from actions (95% CI) | gain on ordinary frames |
|---|---|---|---|---|---|---|
| current (arm C, ResNet 70%) | 70% | 0.768 | 0.784 | 14.7% | **3.2%** [2.6, 3.9] | 3.0% |
| expert, 70% | 70% | 0.741 | 0.756 | 12.2% | 2.7% [2.2, 3.2] | — |
| ResNet, 40% | 40% | 0.771 | 0.821 | 15.7% | **8.1%** [7.3, 8.8] | 7.4% |
| ResNet, 30% (RECAP pre-train share) | 30% | 0.778 | 0.828 | 17.1% | **7.4%** [6.6, 8.1] | 6.5% |
| value-binned, 40% | 40% | 0.783 | 0.836 | 18.3% | **8.5%** [7.8, 9.1] | 7.6% |
| smooth 9 + value-binned, 40% | 40% | 0.786 | 0.829 | 18.4% | 6.6% [5.7, 7.4] | — |
| oracle outcome | 52% | 0.843 | 0.854 | 27.9% | 2.3% [1.2, 3.4] | 2.1% |
| control: stall flag | 13% | 0.977 | 0.991 | 65.8% | 12.9% [11.4, 14.3] | — |
| control: random | 70% | 0.502 | 0.499 | −0.3% | −0.03% [−0.08, 0.01] | — |

Ordinary frames = not stalled, no event (`gain_by_frame_type.csv`). Stall frames (13% of frames) carry 12–18% of
each tag's gain.

1. Controls behave: random 0, stall flag 12.9%.
2. **The current tag carries 3.2% action information**, about as little as the outcome oracle (2.3%), which the
   oracle-tag fine-tune also ignored (18310225).
3. **RECAP's positive shares (30–40%) more than double it** (7.4–8.1%); value-binned calibration at 40% is highest
   (8.5%). The increase holds on ordinary frames (3.0% → 6.5–7.6%), so it is not only "stalls are now negative".
4. Smoothing V lowers action information (8.5% → 6.6%): the smoothed tag depends on frames outside the 20-step
   action window.
5. Most of every tag is explained by neither: S+A explains 15–30% of the entropy; the rest is unpredictable from
   (state, next actions) with these features.

## Interpretation

- The ignored tag is what the theory predicts for labels this weakly tied to the action. The current 70%-positive
  rule puts the cut where the action matters least; 30–40% positive (RECAP's shares) is the cheapest change that
  raises it.
- 8% is still small; a tagged policy may still learn little. These are lower bounds (frozen ImageNet features,
  small MLP), comparable across label sets, not an absolute ceiling.

## Caveats

Recorded actions are those of the data policy; "action information" here = information in the executed chunk
about the tag, as P(tag | s, a) in the theory. Features frozen; a stronger state model would lower "gain" for all
label sets alike.

## Files

`predictability_table.csv`, `gain_by_frame_type.csv`, `oof_predictions.parquet`, `summary.json`, `run.log`.
