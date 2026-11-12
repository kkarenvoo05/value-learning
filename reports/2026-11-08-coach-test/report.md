# Coach test: is the value function the bottleneck? (2026-11-08)

**Question.** Our policy labels ("+"/"−" per frame) come from a value function V guessing how each rollout will
turn out. If V were the problem, labels built from what *actually* happened next should tie much more closely to
the actions. Do they?

**Setup** (job 18387077, `scripts/libero_tag_ideal_shift.py --candidates`, no gate; labels from
`scripts/libero_td_labels.py hindsight`, `vl.libero_td.hindsight_labels`). Same 900 train rollouts, same label grid
(104,049 frames), same rule (value-binned, 40% positive). Measure = **ideal shift**: an MLP predicts the executed
action from the state S plus the tag, out of fold by episode; ideal shift = how far its prediction moves between
+ and − at the same state (raw action units). Also `gain_action_over_state`: how much better the tag is predicted
from state + action than from state alone. [ours]

| labels | what judges each frame | ideal shift [95% CI] | action info beyond state |
|---|---|---|---|
| current (`resnet_valuebinned_p40`) | V's guess 1 s ahead | **0.0151** [0.0149, 0.0154] | 8.1% |
| look 4 s (`look_n80_vb40`) | V's guess 4 s ahead | 0.0102 [0.0101, 0.0104] | 3.8% |
| look 8 s (`look_n160_vb40`) | V's guess 8 s ahead | 0.0094 [0.0092, 0.0095] | 2.0% |
| hindsight (`hind_v0_vb40`) | what actually happened (realised return − V0) | 0.0091 | 2.6% |
| hindsight, ensemble baseline (`hind_ens_vb40`) | same, other baseline | 0.0093 | 1.0% |
| outcome only (`oracle_outcome`) | episode succeeded? | 0.0112 | 1.0% |
| random 40% (floor) | — | 0.0039 | — |
| planted control (11-06) | tag written into the action | 0.0485 | — |

**Verdict.**
- *Observation:* perfect knowledge of the future (hindsight) gives labels that are **less** tied to the action
  than the current 1 s labels (0.0091 vs 0.0151), and every longer lookahead is lower than 1 s. The baseline
  choice does not matter (0.0091 vs 0.0093).
- *Interpretation (reading rule set before the result):* "about equal or below → V is not the bottleneck; the
  data is." A better value function cannot make these labels more informative about actions. What the action
  explains is local: the 1 s labels score highest because the next second's change in V is caused by the action
  just taken; the further the label looks, the more it reflects later luck (a pot slipping) that the current
  action did not choose. With one policy doing nearly the same thing in the same spots, there is little choice in
  the data for any label to point at.

**Caveats.** S (frozen ImageNet features + robot state + both V's) sees less than π₀.₅, so all shifts are upper
bounds and the comparison is relative. Hindsight labels are mostly the outcome (1% of failure frames +, 76% of
success frames +): they are the ceiling for *knowing the future*, not for assigning credit to the right moment.
One test, no policy training.

**Next.** Make the data carry choices: rollouts with action variety at the same states (noise-scaled chunks as in
steering v2, or a mix of policies), then re-run this test on them; if the ideal shift rises, train on that data.
Cheap side check: V trained on half the data vs all (does more data change the labels?).
