# Current state

**Updated 2026-11-11 22:30 PST.** Overwrite this file at the end of every session; do not append. History lives
in `docs/libero_plan.md` `## Log` (LIBERO) and `PROGRESS.md` (Mulligan stages). Where to find things:
`docs/INDEX.md`. Longer-lived direction lists (mentor 2026-11-03 steps, SmolVLA layers, SeeQ): `docs/directions.md`.
Progress site: https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/index.html. New work goes on
Week 9 (`reports/site/content/week09.json`, `["since"]` section; built by `scripts/build_site.py`).

Comparable evals: libero_10 task 8, val states 9 12 13 17 25 32 40 42 44 48 × 33 = 330 trials, env seed 7, matched
noise seed 1 (`MATCHED_NOISE=1`). Base under these settings = **0.524** (18355478). `runs/` =
`/iris/u/ankile/karenvo/libero/runs/`. Intervals are paired-by-trial 95% CIs vs base unless stated.

## Where things stand (one line each; details in the log entries named)

| arm | success | vs base | status / log entry |
|---|---|---|---|
| Residual Advantage Guidance, shift 0.99 (w 9.26), frozen base + AdaLN head | **0.754** | +0.230 [+0.164, +0.300] | final; 11-11 11:55 |
| Residual AG, shift 0.57 / w 16 | 0.697 / 0.658 | — | final; 0.99 is the peak; 11-11 21:45 |
| AdaRMS vb40 + contrastive guidance w 8 | 0.658 | +0.133 [+0.070, +0.197] | final; 11-11 06:00 |
| AdaRMS vb40 w 16 | 0.742 on 264/330 | — | **partial** (18445301 TIMEOUT) |
| Null control: episode-shuffled labels, AdaRMS w 8 | 0.363 on 289/330 | −0.304 vs real-label w 8 | **partial** (18438132 TIMEOUT) |
| DSRL online (noise-space SAC on base), final agent ep 1800 | **0.861** | +0.106 [+0.048, +0.167] vs residual | final, held-out (18452292); 1 seed |
| Filtered BC round 2 (s0 / s1, final) | 0.442 / 0.506 | below round 1 (0.555) | final; no climb; 11-11 15:30 / 21:45 |

Value vs simulator ground truth (`reports/2026-11-10-value-vs-gt/`): corr(A, A*) 0.04–0.14 for every source
against a ceiling of 0.69; the advantage labels carry little action information. Gallery of base vs residual
rollouts: `reports/site/value-learning/gallery-residual/` (guidance removes the "closed freeze" failure: base 12/30,
guided 0/30 on the wrist-view reruns; 11-11 17:00).

Every policy number above is **one training seed**. Single-seed recipe effects of ≤ 0.1 have flipped before
(`docs/findings.md` "Method traps").

## Running and queued jobs (iris; mentor 11-11: rollout rounds with the best policy, 5–10 rounds)

**Residual AG rollout rounds** (`scripts/ag_rounds.sh`, log 11-11 18:30 and 21:45). Per round: collect 100 episodes
with the previous round's residual policy at shift 0.99 (30 train states × 2 + 40 fresh) → table + labels (OOF V,
vb40 within each round; `libero_round_labels.py`) → residual head retrained from zero init on all rounds → contrast
check → 330-trial eval. Next round's collection runs beside the eval.

| round | collect | label | train | check | eval |
|---|---|---|---|---|---|
| 1 | 18460302 / 03 | 18460304 | 18460305 | 18460306 | 18460307 |
| 2 | 18460308 / 09 | 18460310 | 18460311 | 18460312 | 18460313 |
| 3 | 18460314 / 15 | 18460316 | 18460317 | 18460318 | 18460319 |
| 4–9 | submitted by the feeder as queue space frees (QOS limit 25 jobs per user) | | | | |

- **Feeder**: `scripts/ag_rounds_feed.sh 4 9 18460317 18460318`, running in the background on the interactive
  session **18453948 (iris2, 24 h limit, started ~17:40 11-11)**. It checks every 10 min, submits round R when
  ≤ 19 non-interactive jobs are queued, and stops on any failed or never-satisfiable round job. Log:
  `/iris/u/ankile/karenvo/libero/logs/ag_rounds_feed.log`. **If the interactive session ends, the feeder dies**;
  rounds 4–9 will likely outlast it (see Resume steps).
- **Held**: 18433699 (filtered BC round-1 recipe, seed 1) → evals 18433700 (afterok) / 18433701 (afterany). Held
  for the GPU cap. Release with `scontrol release 18433699`, or cancel: round 2 did not climb, so this only adds a
  second seed to round 1's 0.555.

## Resume steps

1. `tail /iris/u/ankile/karenvo/libero/logs/ag_rounds_feed.log`; `squeue -u karenvo`. If the feeder stopped
   because the interactive job ended, restart it from the next unsubmitted round R with the previous round's train /
   check ids: `scripts/ag_rounds_feed.sh R 9 <train id> <check id>` (in a new interactive session or under `nohup`).
2. As each round's eval lands: paired vs base 18355478 and vs round 0 (residual shift 0.99, 0.754) on the same 330
   trials. Read: rising round over round → self-improvement works with guidance; flat → the residual head is
   saturated on this data.
3. Add each round to the week-9 page (`week09.json["since"]`, `scripts/site_since_data.py`), rebuild, publish.
4. Write `reports/2026-11-11-ag-rounds/` (report.md + summary.json) when the rounds finish; update
   `docs/findings.md`.

## Open decisions for Karen

1. **IQL in parallel** (mentor 11-11): not started. Needs a choice of critic inputs (π₀.₅ features vs ResNet) and
   whether it uses the rounds' data.
2. **Seeds**: residual AG and DSRL are one seed each. A second seed of residual shift 0.99 (and of the shuffled-label
   residual, already on the week-9 page as a future row) would make the +0.23 a claim rather than an observation.
3. Rerun the two partial evals (AdaRMS w 16, shuffled control) to 330/330 with a longer time limit, or leave partial.
4. 18433699: release or cancel (above).

## Traps

- π₀.₅ does not load in a 16 GiB interactive session (OOM while restoring the checkpoint; `sacct` MaxRSS misses the
  spike). Always sbatch π₀.₅ jobs, evals included. A one-batch π₀.₅ check can run as a **CPU** sbatch
  (`JAX_PLATFORMS=cpu`, 96G, ~7 min; `scripts/debug/*.sbatch`) without using a GPU.
- Bad nodes: iris8 (`nvidia-smi` hangs, GPU driver stuck), iris9 (JAX finds no CUDA device), iris10 (uncorrectable
  ECC). Excluded in the sbatch defaults; Turing nodes iris[1-4] and iris-hp-z8 fail the bf16 self-check.
- iris QOS: at most 25 submitted jobs per user; a long chain has to be fed in pieces (`ag_rounds_feed.sh`).
- Chains that re-point to a rerun: check every downstream job names the rerun's run dir, not the dead one (round 2 s0
  evals once pointed at the failed 18433693).
- DSRL is not resumable (replay buffer lives in memory): a requeue restarts with a fresh agent.
- Matched-noise evals are deterministic, but `--args.save-raw` reruns are not bit-reproducible.
- `/iris/u/ankile` is 99% full: large features go to `/move/u/karenvo/libero_features/`.
- Never `scancel` other sessions' jobs; re-chain only your own.
- Site rebuild: `.venv/bin/python scripts/build_site.py`; upload with
  `/iris/u/ankile/karenvo/google-cloud-sdk/bin/gcloud storage cp -r --cache-control=no-cache reports/site/value-learning/* gs://zima-robot/reports/karenvo/value-learning/`.
- The repo has a second, stale clone at `/iris/u/karenvo/value-learning` (last touched 2026-11-05). Work here, in
  `/move/u/karenvo/Projects/value-learning`, which every sbatch script points at.
