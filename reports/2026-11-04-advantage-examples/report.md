# What advantages did the value functions assign? (2026-11-04)

**Question (Karen).** Before designing better tag tests: look at the per-frame advantages the value functions
gave on real train rollouts, with the ResNet's full 201-bin value distribution, and at episodes where they disagree.

**Time base.** LIBERO steps at 20 Hz (env `control_freq`; docs/libero_plan.md 2026-11-03), so seconds below = steps / 20. (A first version of this page used 10 Hz and showed every time 2× too long; corrected 2026-11-04.)

**Setup.** `scripts/libero_advantage_examples.py`. Values, advantages and labels are read from the label tables
arm C was trained on (`labels/oof_{resnet,expert}_n20_p30`, `labels/oracle_outcome`). A_t = V(t+20 steps) − V(t) −
20/1040 [paper: RECAP n-step, no discount; n = 20 ours]; negative = below the 30th percentile (ResNet −0.026,
expert −0.024). The 201-bin probabilities are recomputed by re-running each episode's out-of-fold ResNet
(`resnet18_oof_f<fold>_s0_*`); p·centres reproduces the stored V to ≤ 6e-4 on every episode. The expert's OOF run
saved E[V] and its P(success) head only, so it has lines, no heatmap. P(success) for the ResNet = probability mass
above −0.5 (success targets lie in (−0.5, 0], failure targets in [−1, −0.5]). Pots on the stove from the sim goal
predicates, debounced (changes shorter than 10 steps (0.5 s) removed: the predicate flickers while a pot settles) [ours].
Episode picks [ours]: success with the most negative frames; a typical success (median); failure with the most
positive frames; a sustained pot drop (longest time off the stove); most ResNet-vs-expert disagreement.

Scroll-through viewer: `viewer/index.html` (`scripts/libero_advantage_viewer.py`): the 5 picks + 8 random successes + 8 random failures (seed 0); 20 fps video of every step synced to the per-value-frame values, distribution, A and labels.

## Observations

| episode | picked as | outcome | ResNet neg | expert neg | ResNet≠expert | corr(A, ΔP(success)) |
|---|---|---|---|---|---|---|
| 139 | typical success | success 20.1 s | 26% | 41% | 43% | 0.99 |
| 87 | success, most negative | success 22.0 s | 46% | 32% | 45% | 0.98 |
| 1265 | failure, most positive | failure | 11% | 13% | 13% | 0.89 |
| 1099 | pot falls | failure | 37% | 29% | 31% | 0.98 |
| 1176 | ResNet vs expert | failure | 47% | 32% | 58% | 1.00 |

All 900 train episodes: negative share median 26% in successes (p10–p90 15–35%), 35% in failures (25–43%);
ResNet and expert labels disagree on 33% of frames on average.

Labels in the 20 steps (1 s) before an event (`event_labels.csv`; base rate 30% negative):

| event | n | ResNet neg | expert neg |
|---|---|---|---|
| pot falls off the stove, failed episode | 28 | 76% | 53% |
| pot falls off the stove, successful episode | 5 | 92% | 48% |
| first pot placed, successful episode | 521 | 19% | 32% |
| first pot placed, failed episode | 325 | 41% | 48% |

1. **The ResNet's distribution is bimodal.** One band follows the success targets, one the failure targets;
   little mass in between. E[V] is a blend of the two, so it moves mostly when weight shifts between bands.
2. **The advantage is the change in predicted P(success).** Within the five episodes the ResNet's interior A
   correlates 0.89–1.00 with the 20-step (1 s) change of its own P(success). Speed of progress (the −1/1040 per step
   within a band) is invisible next to it.
3. **At events the ResNet labels are mostly sensible**: 76–92% negative before a pot falls, 19% negative before a
   successful first placement. The expert is near the base rate at both.
4. **Systematic exceptions exist.** Ep 139 (typical success): right after the first pot is placed (9.4 s, step 188) the
   ResNet's P(success) collapses 0.8 → ~0 and stays there ~7.5 s; the placement frames are labelled negative.
   Ep 87: after the first pot (from ~13.5 s) the ResNet oscillates between bands; labels flip every 0.5–1 s.
5. **Flat stretches are positive by default.** When nothing changes, A ≈ −20/1040 = −0.019, above the −0.026
   threshold. Ep 1265 (fails without placing a pot) is 89% positive; ep 1099 after its pot falls is ~15 s of
   mostly positive labels.
6. **Terminal frames**: in the last 20 steps (1 s) the outcome enters A directly, so successes end positive and failures that
   were "close" (ep 1099, pot placed at 25.4 s) end strongly negative.

## Interpretation

- What the labels encode is "did the value model's success odds go up or down over 1 s (20 steps)", noisy frame to frame.
  The action-relevant part (credit at drops, at placements) is there but carried by a minority of frames; most
  frames are labelled by flicker between modes or by the default-positive flat regime.
- This changes what the oracle-outcome run tested (see `reports/2026-11-04-tag-usage/`): the per-episode tag is
  constant within an episode, while these labels flip within episodes; a policy ignoring one says little about
  the other.
- A per-frame oracle with the same formula from exact sim progress would remove the flicker but keep the
  default-positive flat regime unless the threshold changes [to decide].

## Caveats

- Five hand-picked episodes; the event table covers all 900 but uses a 20-step (1 s) window and a debounced predicate.
- The expert has no 201-bin heatmap (OOF run saved only E[V] and P(success)).
- corr(A, ΔP) is computed on these five episodes only.

## Next

Karen to look; then decide the better tag tests (per-frame oracle from sim progress; injected-shift conditioning
test).

## Addendum 2026-11-05: ep 139's first placement (Karen's question)

- ResNet P(success) 0.93 (step 160) → 0.06 (step 240) across the first placement (step 188); A −0.20 to −0.39 at
  steps 176–192, negative under every rule and denoising variant (a sustained V drop, not noise). Expert dips
  0.87 → 0.01 (steps 196–204) and recovers to 0.79 by step 236.
- Not typical (`first_placement.csv`): over 516 successes ResNet V rises across the first placement (−0.44 → −0.30);
  it crosses from the success to the failure track in 4.8% (expert 2.7%). Ep 139 was "typical" by negative share only.
- Placement quality is real signal: over 415 fixed-layout placements, distance from the scene's median placement
  predicts failure (farthest quartile ~4 cm: 44% success vs 75% closest) and smaller V gains (ResNet +0.05 vs +0.14).
  Ep 139's offset (2.1 cm, 68th percentile) does not explain its drop: a ResNet error.
- Ep 139 steps 172–232 negative share: ResNet 69%, expert 31%, mean of both 38%. Denoising removes random noise,
  not a value function's systematic mistakes; averaging value functions dilutes them.
