# Failure taxonomy: base pi0.5 vs AG residual (2026-11-12)

Scripts: `scripts/libero_failure_curves.py` (front-camera IQL progress curves), `scripts/libero_failure_taxonomy.py`
(pot detector, types, paired comparison), `scripts/libero_failure_media.py` (side-by-side videos, index.html).
Evals: base 18355478 vs residual 18438187, 330 matched trials each (val states x 33, noise seed 1).
Page: https://storage.googleapis.com/zima-robot/reports/karenvo/moka-failure-taxonomy-2026-11-12/index.html

## Tools and their checks
- Evals saved only the agentview video, so every model here reads agentview frames (frozen ImageNet ResNet-18).
- IQL progress curve (tau 0.9, success reward, + t): corr with V* +0.57 at the 424 branch points, +0.44 within
  episode (5-fold OOF). A logistic success classifier on the same input: +0.34 / +0.13 (dropped). The IQL value is
  time-aware, so every failure ends near 0: it dates *when* an episode went wrong, not how far it got.
- Pot detector [ours]: MLP on agentview features -> the two "pot on stove" goal predicates recorded in the raw
  rollouts. Held-out-episode frame accuracy 0.98; on stage2 val (120 base rollouts on the val states, video frames
  vs recorded flags) frame accuracy 0.94, failures' end state 0.93 (all 44 one-pot failures right; 4 of 13 no-pot
  failures read as one pot). Its groups line up with the sim-state failure types of scripts/libero_failure_types.py
  (stuck_pot1 -> no_pot 5/5; stuck_pot2, stuck_placing_pot2, near_miss, slow, stalled_earlier -> one_pot).

## Results
| outcome | base | residual |
|---|---|---|
| success | 173 | 249 |
| one pot on the stove at the end | 113 | 75 |
| a placed pot came off | 18 | 4 |
| no pot placed | 26 | 2 |

- Fixed (base failed, residual succeeded): 113 = 84 one-pot + 18 no-pot + 11 lost-pot.
- Broken (base succeeded, residual failed): 37, of which 35 end with one pot on.
- The residual's remaining failures are 93% one-pot: the first pot is solved, the second is what is left.
- Faster: successes median 409 -> 397 steps; on the 136 trials both solve, -12.3 steps [-20.6, -3.9].
  First pot on the stove: median step 208 -> 192, and base's tail of late first pots (steps 270-520) is gone.
- IQL drop time of failures: median step 288 (base) vs 316 (residual): the residual's failures go wrong later.

## Next
- Look at the residual's one-pot failures (videos "still failing" / "broken"): stuck on the second pot, pressing it
  against the first, or slow? The sim-state types need raw saving in the eval (`--args.save-raw`).
