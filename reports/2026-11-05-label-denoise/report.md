# Denoising the advantage labels: smoothing, longer lookahead, margin (2026-11-05)

**Question (Karen).** The label-rules comparison (`reports/2026-11-04-label-rules/`) showed the tag is mostly a cut
through value-model noise. Try three remedies and describe how each changes the noise.

**Setup.** `scripts/libero_label_denoise.py`, 900 train rollouts, out-of-fold V of the ResNet and the π₀.₅ value
expert. Variants [ours]: smooth V over a centred window of k value frames (k·4 steps) before differencing; lookahead
n = 40 / 80 / 160 steps (2 / 4 / 8 s) instead of 20; margin = negative below p30 as now, positive above p50 or p70,
the middle untagged; combo = smooth 5 + n = 80 + p30/p70. Smooth and n variants keep 70% positive (p30) so only the
noise changes. The unsmoothed n = 20 row reproduces the trained labels (asserted).

**Noise metric.** The two value functions were trained independently, so the part of A they share is treated as
signal and the part that differs as noise: r(A_ResNet, A_expert) on interior frames, and Cohen's κ of their tags.
Also label flips per second, and the stall-vs-ordinary separation: (median A on stalls − median A on ordinary
progress) / IQR of ordinary progress (ResNet). Ordinary progress = success, hand moving, no event, interior.

## Observations (ResNet labels unless stated)

| variant | r(A_R, A_E) | κ | flips/s | tagged | stall − ordinary, in IQRs | neg on stalls (lift) | neg before drop (lift) | neg on ordinary |
|---|---|---|---|---|---|---|---|---|
| current (n = 20) | 0.27 | 0.20 | 0.96 | 100% | −0.29 | 27% (0.9×) | 78% (2.6×) | 27% |
| smooth 3 (12 steps) | 0.30 | 0.21 | 0.59 | 100% | −0.34 | 31% (1.0×) | 81% (2.7×) | 26% |
| smooth 5 (20 steps) | 0.32 | 0.22 | 0.49 | 100% | −0.39 | 34% (1.1×) | 82% (2.7×) | 25% |
| smooth 9 (36 steps) | 0.38 | 0.25 | 0.34 | 100% | −0.53 | 44% (1.5×) | 84% (2.8×) | 22% |
| n = 40 (2 s) | 0.35 | 0.25 | 0.70 | 100% | −0.47 | 40% (1.3×) | 80% (2.7×) | 23% |
| n = 80 (4 s) | 0.45 | 0.39 | 0.43 | 100% | −0.68 | 50% (1.7×) | 75% (2.5×) | 18% |
| n = 160 (8 s) | 0.55 | 0.54 | 0.27 | 100% | −0.86 | 41% (1.4×) | 62% (2.1×) | 15% |
| margin p30/p50 | 0.27 | 0.26 | 0.94 | 80% | −0.29 | 61% (1.6×) | 85% (2.3×) | 32% |
| margin p30/p70 | 0.27 | 0.31 | 0.98 | 60% | −0.29 | 89% (1.8×) | 91% (1.8×) | 44% |
| combo | 0.49 | 0.65 | 0.18 | 60% | −0.76 | 78% (1.6×) | 95% (1.9×) | 29% |

How far back the negative tag reaches before an event (`event_timing.csv`; negative share by steps before):

| | drop 0–20 | 20–40 | 40–80 | 80–160 | successful 1st place 0–20 | 20–40 | 40–80 |
|---|---|---|---|---|---|---|---|
| current | 0.78 | 0.57 | 0.36 | 0.33 | 0.19 | 0.38 | 0.24 |
| smooth 9 | 0.84 | 0.70 | 0.35 | 0.27 | 0.17 | 0.27 | 0.14 |
| n = 80 | 0.75 | 0.81 | 0.75 | 0.25 | 0.17 | 0.18 | 0.13 |
| n = 160 | 0.62 | 0.70 | 0.70 | 0.71 | 0.13 | 0.16 | 0.14 |
| combo | 0.89 | 0.91 | 0.89 | 0.49 | 0.23 | 0.24 | 0.21 |

1. **The current labels barely agree across value functions**: κ = 0.20, r(A) = 0.27. Most of each model's tag is its
   own noise.
2. **Smoothing removes frame-to-frame noise and keeps the timing.** Flips/s fall 0.96 → 0.34 (k = 9), r(A) rises
   0.27 → 0.38, stalls separate from ordinary progress almost twice as well (−0.29 → −0.53 IQRs), and the penalty
   before a drop stays concentrated in the last 40 steps (0.84 / 0.70, then 0.35).
3. **A longer lookahead raises agreement most but blurs where the credit goes.** n = 160 gives r 0.55, κ 0.54,
   separation −0.86, but the negative tag before a drop is flat (0.62–0.71) over the whole 8 s before it, so good
   actions seconds earlier share the blame; drop lift falls 2.6× → 2.1×. n = 80 sits between (flat over 4 s).
   Longer n also puts more frames in the terminal window, where the outcome enters A directly (4% → 17% at n = 80).
4. **A margin does not change the noise in A** (r unchanged at 0.27); it only drops the ambiguous frames. κ rises
   0.20 → 0.31 on the frames still tagged; stalls become mostly untagged (69% at p30/p70) or negative when tagged
   (89%). Ordinary progress is 38% untagged and 44% negative of the tagged.
5. **The combo is the cleanest by the noise metrics** (κ 0.65, 0.18 flips/s) but inherits n = 80's 4 s smear
   before drops (0.89–0.91 over 0–80 steps).

## Interpretation

- Smoothing is the only remedy that lowers the noise without changing what a label means in time. It is the
  natural first step.
- Longer n buys agreement by asking a coarser question (did the next 4–8 s go well?). That may still be what the
  policy can use, but credit is no longer local to the action.
- A margin is a filter, not a denoiser; worth combining with smoothing so the untagged middle is narrower.
- Agreement between two value functions is a proxy: errors they share (same data, same blind spots) count as
  signal here. The oracle-outcome tag had perfect agreement and the policy still ignored it.

## Caveats

Noise proxy as above; stall threshold ours; no policy trained; ResNet-only signal checks.

## Next

Karen to choose; a candidate is smooth 9 (or 5) + margin p30/p50, which needs one more row before training.
