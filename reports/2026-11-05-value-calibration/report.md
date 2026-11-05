# Value-binned advantage calibration (2026-11-05)

**Question (Karen).** Does the Dissecting paper's value-binned calibration (arXiv 2609.28161, Eq. 4) reduce the
noise in our advantage labels? (Literature review: `reports/2026-11-05-literature-advantage-conditioning/`.)

**Setup.** `scripts/libero_value_calibration.py`, 900 train rollouts, out-of-fold V of the ResNet and the π₀.₅
expert, n = 20-step advantage. z = (A − μ_b) / max(σ_b, 0.1 σ), 8 equal-width bins of V over [−1, 0], bins under
32 frames use global statistics [paper; bins over our V range ours]. Compared with global standardisation (= the
uncalibrated tag), with and without smoothing V (k = 9), at 70% (current), 40% (RECAP fine-tune) and 30% (RECAP
pre-train) positive. Noise metrics as `reports/2026-11-05-label-denoise/` (ResNet–expert r(A), κ; flips/s); the
paper's subtask dependence η²; ordering accuracies [ours, after the paper's]: R_prog = P(moving frame's A > stall
frame's A) within episode, R_drop = P(20–60 steps before a pot falls > the last 20 steps), R_place = P(last 20 steps
before a successful first placement > 20–60 steps before).

## Observations (ResNet tags)

| variant | r(A_R, A_E) | κ | flips/s | η² subtask | R_prog | R_drop | R_place | neg stall | neg ordinary |
|---|---|---|---|---|---|---|---|---|---|
| raw, 70% pos (current) | 0.27 | 0.20 | 0.96 | 0.000 | 0.58 | 0.69 | 0.66 | 27% | 27% |
| raw, 40% pos | 0.27 | 0.23 | 1.02 | 0.000 | 0.58 | 0.69 | 0.66 | 92% | 53% |
| raw, 30% pos | 0.27 | 0.21 | 0.90 | 0.000 | 0.58 | 0.69 | 0.66 | 97% | 65% |
| value-binned, 70% | 0.30 | 0.25 | 0.90 | 0.004 | 0.59 | 0.71 | 0.66 | 39% | 24% |
| value-binned, 40% | 0.30 | 0.29 | 0.97 | 0.004 | 0.59 | 0.71 | 0.66 | 94% | 51% |
| smooth 9 + value-binned, 40% | 0.38 | 0.30 | 0.39 | 0.000 | 0.60 | 0.72 | 0.61 | 92% | 53% |

1. **Our advantages have no subtask offset to remove**: η² of A by subtask is 0.00002, although V itself depends on
   it (η² 0.44). The difference V(t+20) − V(t) cancels the per-stage level; the paper's raw advantages had 0.06–0.13.
2. **What binning does correct**: A depends on the current V (mean +0.067 in the lowest V bin, −0.013 in the
   highest: an unluckily low V tends to bounce back) and its spread varies (std 0.067 → 0.10 in the middle bins,
   between the success and failure bands). Removing both raises agreement modestly: κ 0.20 → 0.25 (70% pos),
   0.23 → 0.29 (40% pos), r 0.27 → 0.30.
3. **Smoothing adds to it** (r 0.38, κ 0.30, flips 0.39/s) but lowers R_place 0.66 → 0.61.
4. **Ordering accuracies are low for every variant**: R_prog 0.58–0.60 (stall vs moving), R_drop 0.69–0.72. The
   paper's selected critics reach R_prog 0.81–0.95 on its (different, human-defined) segments.

## Interpretation

Calibration is a small, safe improvement here (+0.05–0.06 κ); it does not touch the main noise source. Our critics
order stalls below moving frames only 58–60% of the time, far from the paper's selected critics; the value
function, not the calibration, is the limit.

## Caveats

R_prog / R_drop are our analogues on automatic stall / drop definitions, not the paper's human-takeover or
VLM-segmented windows; numbers are not comparable across papers. ResNet tags only in the table (expert in CSV).

## Files

`calibration_table.csv`, `calibrated_labels.parquet` (tags per variant; z and weights exp(clip(z, −2, 2)) for a
weighting run), `summary.json`.
