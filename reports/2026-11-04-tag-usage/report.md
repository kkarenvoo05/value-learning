# Does the advantage-tagged policy use its tag? (job 18309419)

**Question.** Arm C (π₀.₅ fine-tuned with ResNet OOF advantage tags) scored 0.409 with the positive tag, 0.430 with
the negative tag, success-only arm A 0.406. Does arm C condition on the tag at all?

**Setup** (`scripts/libero_tag_usage.py`, [ours]). Final checkpoints of arm C (policy_armC_resnet_s0_18275460) and
arm A (policy_armA_s0_18288138, never saw a tag = control).
- Action test: 320 val frames (40 val episodes × 8), 4 noise draws each; action chunk sampled with tag positive /
  negative / none at the *same* start noise. Tag effect = RMS(a(+) − a(−)); noise effect = RMS between two noise
  draws at the same tag; over the 5 executed steps × 7 action dims, model-normalised space (raw units similar).
- Loss test: 512 train frames (64 episodes, 256 positive / 256 negative by the training labels); flow-matching loss
  of the recorded chunk with the correct vs the flipped tag, same noise and time, 4 repeats; episode bootstrap.

| | arm C (tagged) | arm A (control) |
|---|---|---|
| tag effect (+ vs −), normalised | **0.0013** | 0.0024 |
| tag effect (+ vs no tag) | 0.0106 | 0.0153 |
| noise effect (same tag) | 0.057 | 0.067 |
| ratio tag / noise | **0.023** | 0.036 |
| frames with tag effect > noise effect | 0 / 320 | 0 / 320 |
| gripper sign flips: tag / noise | 0.0% / 2.6% | 0.03% / 3.5% |
| loss correct / flipped | 0.00689 / 0.00689 | 0.00694 / 0.00694 |
| flipped − correct (95% CI) | −0.0000 [−1.0e-5, 3.1e-6] | −0.0000 [−1.7e-5, 1.0e-5] |

(`ratio_ci95` in results.json is the bootstrap of per-episode ratios, a slightly different estimator than the
pooled ratio; both are ~0.02–0.05.) **Correction 2026-11-04:** that mismatch was a bug: the interval did not
bootstrap the reported statistic and can exclude it (oracle run 18310225: 0.021 vs [0.024, 0.033]).
`scripts/libero_tag_usage.py` now bootstraps the pooled ratio (`boot_ratio_ci`); this run's intervals need a rerun
(per-frame values were not saved). The verdict (tag ≪ noise) does not depend on it.

**Observations.** Flipping the tag moves arm C's action by ~2% of its sampling noise, less than for the control that
never saw a tag (the tag is only prompt text there). Arm C fits its own training actions no better with the correct
tag than with the flipped one.

**Interpretation.** Arm C did not learn to condition on the advantage tag; the eval's "no tag effect" is "no tag
learned", so it says nothing yet about label quality vs the policy improvement method. Two candidate causes not
separable here: (1) the labels carry too little information about the actions (ResNet labels: 73% vs 66% positive
in successes vs failures); (2) prompt-text conditioning through the frozen LLM + LoRA at 5000 steps is too weak.

**Next.** Oracle-tag fine-tune (tag = rollout outcome) + this check: learns it → labels are the bottleneck;
does not → conditioning (more steps, tag dropout / CFG-style, a learned tag embedding).
