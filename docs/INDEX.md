# Docs index

Start with [CURRENT](CURRENT.md) (live jobs, open decision, resume steps), then [findings](findings.md) (settled
conclusions). Open only what the question needs. Dated reports and log entries are **evidence, not live status**.

## Layout

| where | holds |
|---|---|
| `docs/CURRENT.md` | Live state; overwritten each session |
| `docs/findings.md` | Settled conclusions by topic; corrected in place |
| `docs/libero_plan.md` | LIBERO plan (stages, gates) + dated lab notebook (`## Log`), append-only |
| `PROGRESS.md` | Mulligan stage checklist |
| `reports/YYYY-MM-DD-<slug>/` | One folder per experiment from 2026-11-02 on: `report.md` + data (`*.json`) + figures |
| `reports/*.md`, `reports/*.json`, `reports/figures/` | Older per-experiment reports (before the folder layout) |
| `reports/site/<week>/` | Built weekly pages (`scripts/build_week_report.py`); published copy in the bucket |
| `/iris/u/ankile/karenvo/libero/` | LIBERO data: `runs/`, `features/`, `logs/`, `tables/` (not in git) |
| `/move/u/karenvo/libero_features/` | Large feature caches (iris is full) |

## Need → read

| need | read |
|---|---|
| What is running, what to do next | [CURRENT](CURRENT.md) |
| LIBERO goal, stages, gates, paths | [libero_plan.md](libero_plan.md) top sections |
| Is π₀.₅'s value noise in SigLIP, the LLM layers, or fine-tuning? | [frozen encoders report](../reports/2026-11-02-frozen-encoders/report.md) |
| Does reading V straight from SigLIP (no LLM) beat π₀.₅ / the ResNet? | [SigLIP-direct report](../reports/2026-11-03-siglip-direct/report.md) |
| Does layout variety or episode count matter? | same report, P1c |
| Seed spread of LIBERO value arms (3a) | [libero_seeds.md](../reports/libero_seeds.md) |
| Does frozen π₀.₅ encode pot-on-stove? | [libero_progress.md](../reports/libero_progress.md) |
| Milestone (pot placement) response of V | [libero_milestone.md](../reports/libero_milestone.md), [libero_milestone_probe.md](../reports/libero_milestone_probe.md) |
| Does smoothing close B's gap? | [libero_smoothing.md](../reports/libero_smoothing.md) |
| LIBERO advantage labels and checks | [libero_advantages.md](../reports/libero_advantages.md) |
| Does fine-tuning hurt or help the policy? Does stopping early fix it? | [findings](findings.md) "Policy fine-tuning", [base-vs-finetuned report](../reports/2026-11-04-base-vs-finetuned/report.md) |
| Is the value function the bottleneck for the labels? (hindsight labels) | [coach test](../reports/2026-11-08-coach-test/report.md) |
| Does exploration noise add action variety without ruining episodes? | [explore pilot](../reports/2026-11-08-explore-pilot/report.md) |
| Why the tagged policy ignores its tag (labels vs conditioning, positive controls, prompt test) | [tag deep dive](../reports/2026-11-06-tag-deep-dive/report.md), [tag predictability](../reports/2026-11-05-tag-predictability/report.md), [tag usage](../reports/2026-11-04-tag-usage/) |
| E2 subtask head, LoRA, arm C status | [libero_plan.md](libero_plan.md) log, 2026-11-02 E2 → 2026-11-03 08:45 |
| Mulligan dataset audit, splits, targets | [dataset_audit.md](../reports/dataset_audit.md), [audit_tables.md](../reports/audit_tables.md), [targets_summary.md](../reports/targets_summary.md) |
| Mulligan baseline and ablations | [baseline_results.md](../reports/baseline_results.md), [stage5_ablations.md](../reports/stage5_ablations.md) |
| Mulligan history / camera / still-time studies | [history_gain.md](../reports/history_gain.md), [camera_contrib.md](../reports/camera_contrib.md), [still_time_gain.md](../reports/still_time_gain.md) |
| Package for Lars | [lars_package_draft.md](../reports/lars_package_draft.md) |
