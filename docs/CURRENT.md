# Current state

**Updated 2026-11-21 14:00 PST** (mentor session: Karen's 10:45 state below, plus the rows marked *mentor*; Karen's 12:50 push-direction work is in the plan log). Overwrite this file at the end of every session; do not append. History lives
in `docs/libero_plan.md` `## Log` (LIBERO) and `PROGRESS.md` (Mulligan stages). Where to find things:
`docs/INDEX.md`. Longer-lived direction lists: `docs/directions.md`.
Progress site: https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/index.html.

Comparable evals: moka = libero_10 task 8, val 330 trials (10 states x 33, env seed 7, `MATCHED_NOISE=1`), base
**52.4%**; moka_cont100 (300 fresh-layout continuation starts, base ~55); t64_cont100 = libero_90 task 64, base
**60.7%**. All residual AG evals at push 0.99 unless stated. `runs/` = `/iris/u/ankile/karenvo/libero/runs/`.
**Single-seed retrains vary by up to ~25 points** (b100 r1: seed 0 53.0, seed 1 77.0): attribute nothing to one seed.

## Results that matter now

| what | result | where |
|---|---|---|
| 900 chain (frozen 900-ep V), r0 -> r7 | 75.5, 79.1, 74.5, 80.9, 79.4, 87.6, 77.3, 83.0 | 11-1x |
| **Round-1 drop vs training seed** (*mentor*: finals) | b100: seed 0 53.0, **seed 1 77.0** (r0 79.7) = noise. b50cv val 11.5 / **10.0**, moka cont 15.3 / **14.7** (r0 44.8 / 51.0). mt100cv val 32.1 / **14.5**, moka cont 33.3 / **12.3** (227/300), **t64 cont 33.3 / 73.3** (296/300; r0 77.3): seed 1 holds on task 64 | reports/2026-11-21-nov21-update |
| *mentor*: **eval correction** | mt100cv r1 s0 val eval 18598985 has 280 client errors (read 4.5%); valid rerun 18604938 = 32.1% (305). Nov 20 report / findings / page fixed | same |
| *mentor*: push direction, 10 of 17 policies | dropping r1s are not more aligned with r0 (mt r1 s0 cos -0.11, beta 0.37; mtos91 0.28 / 0.56; mtos77 0.31 / 0.54) -> against double push | same |
| Base share of r1 frames (mt100cv, same 100 base eps) | ~52%: 32 / 33 / 33; 77%: 24 / 31 / 43; 91%: 64 / 54 / 46 (val / moka cont / t64 cont; r0 65 / 64 / 77) | same |
| r1 push sweep | no push makes r1 beat base (moka r1 @0.3/0.6/0.99: 52/48/34, base 55) | same |
| **Offset G0** | gate passes both probes: cv_gain_vs_cand0 +0.125 [+0.06, +0.20] / +0.121 [+0.05, +0.19]; no direction replicates | reports/2026-11-20-offset-g0 |
| Demos hurt (moka) | nd900 81.2 vs 75.5; ndb50 72.7 vs 44.8 | 11-19 / 11-20 |

## Open question: why the ~50-base-per-task chains drop at round 1
Reading so far (not proven): the guided share a chain tolerates grows with its base episodes per task; oversampling
the same 100 base eps to 91% fixes val but not fresh layouts. **Label check (11-21): labels treat base and guided
episodes alike in every chain, dropping or not -> not a label problem.** Factorial running:
`reports/2026-11-21-r1-factorial/report.md` (arms A-D + r0 controls on subsets of `tables/moka_agr_r7`, frozen 900 V,
no demos, 2 seeds; pre-registered reading rule there). Score each with the usual paired val / moka cont comparison.

## Running / queued (gpu_cap priority `.gpu_cap_priority` in the libero logs dir)
1. Seed-1 r1: b50cv val / moka cont and mt100cv val done (14:00, *mentor*); mt100cv moka / t64 cont 18615179 / 80 at
   227 / 296 of 300, client logs "Still waiting for server" since 10:40 (check the server side).
2. b200cv r1: train 18580523 (step ~4250/5000), then check / eval / ceval 18580524-26 (held).
3. b100 rounds 3-9: r3 train done; **check resubmitted 18620264** (old 18607030 admin-held after the train ended
   PREEMPTED), eval / ceval 18607031 / 32 re-pointed to it. Feeder host 18620302 (rounds 4-9, prev check 18620264).
4. **Round-1 factorial** (`agr_fact*`, 40 jobs held, priority after b200cv): trains s0 18620353 / 57 / 62 / 67 / 71 / 75
   (A B C D R100 R700), s1 18620379 / 83 / 87 / 91 (A-D); log `logs/fact_r1.log`. ~95 GPU-h, about a day.
5. Paused (`.gpu_cap_pause`): mtbv (t64_base, bigv_label, agr_mtbv_*), agr_mid_*, agr_b50v900_*.

## Infrastructure
- iris7 hung 11-20 ~02:20: excluded everywhere (`iris[1-4,7-10]`); undo when fixed. GPU nodes: iris5 / iris6.
- gpu_cap host 18622088 (<= 6 GPU jobs; 1-day limit: restart before 2026-11-22 12:42, the previous host timed out 10:08 today and nothing was released until 12:42). Held jobs sit on iris-hi.
- **A train that finishes but ends PREEMPTED leaves its afterok check JobHeldAdmin forever** (users cannot release
  it). Resubmit the check, re-point the evals' dependency, and fix the chain's feeder (its next round depends on the
  old check id; the host resumes from the feeder log's last "round R submitted (" block).
- Eval harness counts client-errored episodes as failures: check `episodes.csv` `error`.
- gcloud needs re-auth; value videos (reports/2026-11-19-t64-value-videos) not uploaded.

## Waiting on Karen
- r1-vs-r0 push-direction check (GPU job + new script) worth writing?
- Offset G0 passed: start G1 on offset proposals?
- Drop `--demos` from the chains?
- Rebalancing (rbag / rbfail) still a dry run; retry iris10?
