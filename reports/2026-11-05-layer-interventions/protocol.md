# Layer intervention diagnostic — protocol

2026-11-05. Our diagnostic design; not a reproduction of a paper. No training labels, dataset splits,
policy-test scenes, or checkpoint-selection rules are changed. Existing value-validation episodes are used
for diagnosis, not as a new final evaluation. Training and checkpoint weights stay fixed during interventions.

## Questions and interpretation rules

1. Recompute the 4- and 18-layer results using all three seeds and identical episode/frame keys. Report
   MAE, signal, residual noise, and their ratio separately. The legacy `change_snr_2s` is a **1-second**
   difference: five cached grid frames × four environment steps / 20 Hz. Also measure adjacent (0.2 s)
   changes. Residual variation is not automatically erroneous variation.
2. Quantify uncertainty with a paired hierarchical bootstrap: resample scenes, then episodes within each
   drawn scene; use the same episode weights for both models, and resample training seeds independently
   per model. Average metrics over models, not predictions over seeds. Three seeds and ten scenes limit
   inference. Do not claim an accuracy penalty, benefit, or noise reduction when its interval covers zero.
3. Choose adjacent frame pairs using full-expert seed 0, with up to two jump pairs per episode. Candidate
   suspicious jumps require no goal-predicate transition within 10 steps of either endpoint and at most
   5 mm pot-position change during the pair. These filters do not establish unchanged true value or
   unchanged visibility. Compare against low-jump controls matched within the same episode, random
   no-event pairs, and sustained placement events chosen without looking at predictions. Report behavior
   on exactly the same pairs for seeds 1 and 2; do not reselect their worst pairs.
4. Apply between-frame K/V substitutions at the **same layer and token position** of the existing expert.
   Measure both directions, selected layer bands and individual layers, and camera vs text/state tokens.
   Do not interpret sums of individual effects as an additive decomposition. Require compatible masks
   for partial text/cache patches; report exclusions. Unchanged and full-cache substitutions are numerical
   controls. Reproduce saved predictions before interpreting interventions.
5. An intervention implicates the expert's dependence on that cache component, not the backbone layer
   that originally created the information. Later-layer caches can carry earlier-layer effects. Inspect
   interventions on real-event and steady controls before proposing removal of a layer. Attention weights
   or feature-distance correlations alone do not establish a mechanism.

## Initial scope

- Frozen full expert: seeds 0–2, jobs 18237898/18237899/18237900. Truncated 4: seeds 0–2,
  jobs 18291640/18309645/18309646. Truncated 9: seed 0 only, descriptive.
- Default 12 pairs per cohort (jump, matched steady, random no-event, placement); fewer if eligible data
  are insufficient. The pilot can use fewer pairs to validate plumbing; label pilot outputs accordingly.
- K/V bands: 0–3, 4–8, 9–15, 16–17, and individual layers. Endpoints use 0-based block indices.
- Diagnostic inputs/results stay local. No automatic public upload.
- Respect the existing three-batch-GPU cap and exclude move5. Any GPU diagnostic waits until existing
  work leaves capacity; existing jobs are not cancelled or modified.

## Follow-up gate

Only after the diagnostic identifies a reproducible effect, consider a training comparison that holds
expert depth fixed while restricting access to backbone layers. Existing 4/9/18 runs change both accessible
backbone features and expert depth (about 7.9/17.8/35.5 million trainable parameters).
