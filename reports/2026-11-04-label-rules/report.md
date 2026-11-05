# Label threshold rules: what flips (2026-11-04)

**Question (Karen).** In ep 1099 the arm pauses (steps 272–324) and both advantages are negative (ResNet ≈ −0.02),
yet the labels stay positive. Compare three alternative rules for turning A into the binary tag before the next
policy run.

**Setup.** `scripts/libero_label_rules.py`. Same out-of-fold V and n = 20-step advantage as the arm C labels; only
the rule changes (R0 reproduces the trained labels exactly, asserted). 900 train rollouts, 104,049 value frames.
Rules: R0 current A > p30 (70% positive); R1 A > 0; R2 A > p70 (30% positive); R3 drop the time cost
(A + n/1040, terminal A + (T−t)/1040), then > p30 [ours: R1–R3]. Stall [ours] = end effector moves < 1.5 cm in the
next 20 steps and gripper opening changes < 5 mm (normal motion 2–18 cm per 20 steps): 12.8% of frames. Events:
20 steps before a debounced pots-on-stove change. Viewer: `reports/2026-11-04-advantage-examples/viewer/`
(rule selector; strips for every rule + stall).

## Observations (ResNet V; expert in `rules_table.csv`)

| rule | threshold | positive | neg before pot falls | neg before 1st place (success) | neg on stalls (succ / fail) | neg on moving frames | within-scene AUC | flips/s |
|---|---|---|---|---|---|---|---|---|
| R0 now | −0.026 | 70% | 78% | 19% | 38% / 26% | 31% | 0.80 | 0.96 |
| R1 A > 0 | 0 | 47% | 87% | 28% | 78% / 88% | 49% | 0.98 | 1.06 |
| R2 top 30% | +0.026 | 30% | 92% | 40% | 92% / 97% | 67% | 0.95 | 0.90 |
| R3 no cost | −0.007 on A′ | 70% | 78% | 19% | 38% / 26% | 31% | 0.81 | 0.96 |

1. **R3 equals R0** (0.05% of frames change): the time cost is the same constant on every interior frame, so
   removing it shifts all A equally and a percentile threshold does not move relative to them.
2. **R1 and R2 make stalls negative** (78–97% vs 26–38% under R0), including ep 1099's pause.
3. **On ordinary progress frames A is noise around 0.** Success episodes, moving, no event, interior (87% of
   success frames): median A +0.004, IQR [−0.031, +0.046]. A stall costs −0.019, a quarter of that spread. Positive
   share on these frames: R0 73%, R1 54%, R2 35%.
4. **Event contrast relative to each rule's base rate is similar** (pot falls / base: R0 2.6×, R1 1.6×, R2 1.3×;
   successful placement / base: R0 0.62×, R1 0.53×, R2 0.58×). Moving the threshold mainly changes how many frames
   are negative, not how sharply events stand out.
5. **Within-scene AUC rises under R1/R2** (0.80 → 0.95–0.98): failures accumulate long flat (stalled / stuck)
   stretches that become negative, so the per-episode share tracks the outcome more closely.

## Interpretation

- The labels are mostly a cut through value-model noise; the threshold decides which side of the noise most
  frames fall on. R0 lets stalls through as positive; R1 catches stalls but labels ordinary progress near a coin
  flip; R2 calls most ordinary progress negative.
- The higher AUC under R1/R2 is partly outcome information (long stuck stretches in failures), not finer
  action-level credit; the oracle-outcome tag had AUC 1.0 and the policy ignored it.
- A rule alone cannot make the per-frame signal cleaner than V's frame-to-frame noise. Options that attack the noise
  [to decide]: smooth V before differencing; longer n; label only frames with |A| beyond a margin (drop the
  ambiguous middle from the tag, i.e. "no tag").

## Caveats

Stall threshold is ours; ResNet OOF V only for the headline table; expert in the CSV. No policy was trained.

## Next

Karen to compare in the viewer and choose a rule (or a noise-aware variant) for the next policy run.
