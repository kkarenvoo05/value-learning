# value-learning

Learned value functions as the improvement signal for **self-improving vision-language-action (VLA) policies**.

The long-term loop: a VLA (π₀.₅) collects rollouts → a critic scores progress and success → the scores become
training or selection signal → the improved policy collects the next round. This repo builds and stress-tests each
link of that loop, small models and limited data first:

1. **Value function.** A distributional (201-bin) Monte-Carlo state value V(o), RECAP / π\*0.6 style, on real-robot
   data (Mulligan Insert Marker) and on π₀.₅'s own simulated rollouts (LIBERO).
2. **Value on the VLA's own representations.** A value expert grafted onto π₀.₅'s PaliGemma backbone, reading its
   per-layer key/value cache the way the action expert does; frozen, LoRA, truncated-depth (SmolVLA-style) and
   demo-pretrained variants, benchmarked against a fine-tuned ResNet-18 critic.
3. **Policy improvement.** RECAP-style advantage-conditioned fine-tuning of π₀.₅ (flow matching with an advantage
   indicator in the prompt, labels from out-of-fold critics) vs success-only imitation, with negative-indicator
   and oracle-label controls; SeeQ-style Q-guided best-of-N action selection as the test-time alternative.
4. **Real robot.** Port the working recipe to π₀.₅-DROID on the lab's Franka.

## Testbeds

| | Mulligan Insert Marker (real) | LIBERO moka pots (sim) |
|---|---|---|
| data | ~1.4k teleop / policy episodes, Franka, public HF release | 900+ rollouts of frozen `pi05_libero` on libero_10 task 8 ("put both moka pots on the stove", ~58% success), fixed + fresh layouts |
| labels | reviewed outcomes and endpoints, unknowns flagged | exact sim outcomes; per-step goal predicates (subtask / milestone) |
| splits | fixed train/test in `configs/dataset.yaml`, ~15% val episode *groups* | by init state: 30 train / 10 value-val / 10 policy-test (`configs/libero_moka.yaml`) |
| critics | ResNet-18 over cameras (+ proprio, history) | ResNet-18; π₀.₅ value expert (frozen / LoRA / truncated); probes on SigLIP and per-layer LLM states |
| policy | — | π₀.₅ LoRA fine-tunes (openpi), evaluated on 330 held-out trials per arm |

LIBERO is where the whole loop runs end to end: every rollout stores the executed actions and MuJoCo states, so
episodes replay exactly and the simulator can be branched from any step (used to measure best-of-N headroom).

## What we have learned so far (2026-11-04)

Settled conclusions with evidence: `docs/findings.md`. Headlines:

- π₀.₅'s features *know* task progress (pot-on-stove probe AUC 0.96–0.99) and give more accurate V than the ResNet,
  but a far noisier one within an episode. The noise has two separate sources: the upper LLM layers (inherited
  from PaliGemma, not caused by robot training) and fine-tuning SigLIP. Layout variety matters more than episode count.
- Reading only the first 4 of 18 backbone layers (truncated value expert) gives the steadiest π₀.₅ critic so far,
  consistent across seeds 0–1.
- Advantage-conditioned fine-tuning has **not** yet beaten success-only imitation (0.409 vs 0.406, 330 trials); a
  tag-usage diagnostic shows the policy ignores the tag, so the next test separates weak labels from weak conditioning.
- Best-of-N steering of base π₀.₅ has ~no headroom on this task (candidate chunks end ~2 mm apart).
- Mulligan: critics latch onto collection-session shortcuts; early-fusion frame stacking is the only memory variant
  that improved stall-vs-success separation.

## Where to look

| need | read |
|---|---|
| What is running, what to do next | `docs/CURRENT.md` |
| Settled conclusions | `docs/findings.md` |
| Find the document for a question | `docs/INDEX.md` |
| LIBERO stages, gates, dated lab notebook | `docs/libero_plan.md` |
| Mulligan stage checklist | `PROGRESS.md` |
| One experiment | `reports/YYYY-MM-DD-<slug>/report.md` (older: `reports/*.md`) |
| Progress site (public) | https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/index.html |

## References

- RECAP / π\*0.6 — https://arxiv.org/abs/2511.14759 (−1/step reward, failure penalty, per-task values in (−1, 0),
  201-bin value, advantage-conditioned policy, value pre-training)
- SeeQ — https://arxiv.org/abs/2609.22085 (subtask-scoped Q-functions, best-of-N policy steering)
- SmolVLA — https://arxiv.org/abs/2506.01844 (action expert reads only the first half of the VLM);
  SmolVLM — https://arxiv.org/abs/2504.05299
- π₀.₅ / openpi — https://github.com/Physical-Intelligence/openpi
- Mulligan — https://arxiv.org/abs/2610.05882; data https://huggingface.co/collections/mulligan/real-insert-marker
  (read the HF release docs: outcome/endpoint semantics live there)
- LWD — https://arxiv.org/abs/2605.00416 (related value-learning approach)

## Ground rules

- Correctness and understanding beat complexity. Small readable code, no big frameworks beyond openpi.
- Separate **paper-specified** details from **our choices**; never claim exact reproduction.
- Splits are fixed (above). Held-out test data (Mulligan test repos, LIBERO policy-test init states) is for final
  evaluation only; never tune on it.
- Labels come from reviewed outcomes and endpoints, with unknowns flagged, never turned into failures. Machine
  (VLM) stage labels are diagnostics, not truth.
- Report ≥ 3 seeds before reading small effects (milestone / dip effects flip by seed); pre-register the reading
  rule; compare arms with the paired episode × seed bootstrap (`scripts/compare_arms.py`).
- Never commit data, caches, checkpoints, videos, or credentials (`.gitignore`).
- Record configs, seeds, dataset commits, commands, and results. Only report what actually ran.

## Layout

```
configs/                   dataset / targets / data / train YAML; libero_moka.yaml (task, init-state split)
src/vl/                    library code (PyTorch)
  manifest, splits, targets, frames, dataset, model, train*, metrics, ...   Mulligan pipeline (Stages 0–5.5)
  libero_targets, libero_resnet, libero_probe, libero_subtask              LIBERO critics and targets
  libero_advantages, libero_oof, libero_finetune_data                      advantages, out-of-fold labels, policy data
scripts/
  libero_rollouts.py, libero_replay.py, libero_to_lerobot.py   π₀.₅ rollouts (LIBERO client env), exact replay, LeRobot
  pi05_value_expert.py, pi05_value_lora.py                     value expert on π₀.₅ (frozen / LoRA, --n-layers N)
  libero_frozen_encoders.py, pi05_layer_hidden.py, libero_layer_value.py   encoder and per-layer value probes
  libero_oof_labels.py                                         RECAP advantage labels from out-of-fold V
  pi05_policy_finetune.py, serve_pi05_finetuned.py             advantage-conditioned / imitation fine-tunes, serving
  libero_tag_usage.py                                          does the fine-tuned policy use its tag?
  libero_steer_probe.py, libero_steer_analyze.py               best-of-N headroom by branching the simulator
  compare_arms.py                                              paired bootstrap win rule
  build_site.py, build_week_report.py                          progress site / weekly pages
  *.sbatch                                                     SLURM jobs (move partition)
tests/                     unit tests (targets, splits, advantages, data) + manifest integrity checks
reports/                   per-experiment reports, figures, site sources
docs/                      CURRENT, findings, INDEX, libero_plan
```

## Environments

Three, because π₀.₅ (JAX) and LIBERO (Python 3.8) cannot share one:

| env | where | used for |
|---|---|---|
| this repo, `.venv` (Python 3.11, PyTorch) | `uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -e ".[dev,video]"` | Mulligan pipeline, ResNet critics, analysis, tests |
| openpi (215abfb), its `.venv` (JAX) | `/iris/u/ankile/karenvo/pi05/openpi` | π₀.₅ value experts, policy fine-tunes, policy server |
| LIBERO client (Python 3.8) | `/iris/u/ankile/karenvo/libero/.venv` | rollouts, replay, steering probe |

`pi05_libero` checkpoint (read-only): `/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero`.
LIBERO runs and logs: `/iris/u/ankile/karenvo/libero/{runs,logs}/`; large feature caches: `/move/u/karenvo/libero_features/`.

## Common commands

```bash
# tests (after any pipeline change)
.venv/bin/python -m pytest -q

# LIBERO: rollouts / eval of base or fine-tuned π₀.₅ (server + client on one node)
SUITE=libero_10 TASKS=8 REPEATS=2 SEED=8 NAME=moka sbatch scripts/libero_eval.sbatch
FT_RUN=<runs/policy_*> FT_TAG=positive|negative|none ... sbatch scripts/libero_eval.sbatch

# π₀.₅ value expert (frozen; --n-layers N for the truncated expert)
ARGS="--train ..." NAME=<name> sbatch scripts/pi05_value_expert.sbatch

# advantage labels, then a policy fine-tune per arm
.venv/bin/python scripts/libero_oof_labels.py --source resnet
NAME=armA_s0 ARGS="--train --arm imitation --seed 0" sbatch scripts/pi05_policy_finetune.sbatch
NAME=armC_s0 ARGS="--train --arm advantage --labels <parquet> --seed 0" sbatch scripts/pi05_policy_finetune.sbatch

# Mulligan pipeline (metadata → manifest → splits → targets → videos → frames → baseline)
.venv/bin/python -m vl.fetch_metadata && .venv/bin/python -m vl.manifest && .venv/bin/python -m vl.splits
.venv/bin/python -m vl.targets && .venv/bin/python -m vl.fetch_videos --cams wrist_left side_1 --data
.venv/bin/python -m vl.frames && sbatch scripts/baseline.sbatch
.venv/bin/python -m vl.metrics --run baseline

# progress site
.venv/bin/python scripts/build_site.py      # then publish (public bucket; review first)
```

Each script's docstring has its full usage. The Mulligan manifest (`data/manifest/episodes.parquet`) is the single
source of truth for Mulligan episode identity, outcome, endpoint, group and split.
