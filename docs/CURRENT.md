# Current state

**Updated 2026-11-20 11:30 PST** (mentor session: Karen's working tree committed, Nov 13-20 summary report, Nov 20 site page; Karen's 09:00 state below unchanged). Overwrite this file at the end of every session; do not append. History lives
in `docs/libero_plan.md` `## Log` (LIBERO) and `PROGRESS.md` (Mulligan stages). Where to find things:
`docs/INDEX.md`. Longer-lived direction lists: `docs/directions.md`.
Progress site: https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/index.html.

Comparable evals: moka = libero_10 task 8, val 330 trials (10 states x 33, env seed 7, `MATCHED_NOISE=1`), base
**52.4%**; moka_cont100 (300 fresh-layout continuation starts); t64_cont100 = libero_90 task 64 ("stack the right
bowl on the left bowl and place them in the tray"), base **60.7%**. All residual AG evals at push 0.99 unless stated.
`runs/` = `/iris/u/ankile/karenvo/libero/runs/`. Every number is one seed; round-to-round noise ~3 pts (r6 77.3 vs
rerun rs6 80.3).

## Results that matter now

| what | result | log |
|---|---|---|
| 900 chain (frozen 900-ep V), r0 -> r7 | 75.5, 79.1, 74.5, 80.9, 79.4, 87.6, 77.3, 83.0 | 11-1x |
| **Demos hurt** (moka, no --demos, same labels) | nd900 81.2 vs 75.5 with; ndb50 72.7 vs 44.8 (cont 60.7 vs 51.0) | 11-19 22:45, 11-20 01:25 |
| Small chains dip at round 1 | b100 79.7 -> 53.0 (r2 69.7); b100iql 63.9 -> 55.5; mt100cv moka cont 63.7 -> 33.3, t64 cont 77.3 -> 31.4 (207/300) | 11-20 08:10 |
| Two-task r0 (mt100cv, moka demos only) on task 64 | 77.3 (combined w) / 75.3 (task-64 w) vs base 60.7 | 11-19 23:10 |
| Task-64 demos | t64d50 0% (demo strategy = stack first; pi0.5 = tray first) | reports/2026-11-19-t64-demo-collapse |
| Fresh-layout sweep of base pi0.5, 4 trained suites | 94-99%; only moka (75%) in 50-80% -> no new second task | 11-19 22:45 |
| 700-step cap (r5) | 87.9 vs 85.5 by step 520: failures are stalls, not lack of time | 11-19 19:00 |
| DSRL s1 run long | 87.0 / 89.1 / 90.0 / 86.1 / 88.2 at 1.3k / 1.8k / 2.4k / 3.0k / 3.3k episodes: saturated | reports/2026-11-20-plateau-and-matched-budget |
| Distillation into π₀.₅ (LoRA, teacher successes) | DSRL 88.2, AG 89.1 (266/330), hybrid 80.9, base 49.7 | same |
| Noisy judge, 25% symmetric flips | AG r0 76.1 (clean 75.5); DSRL 43.0 / 35.2 at 400 / 600 | same |
| Push sweep r5-r7 | 0.8-1.2 flat; 1.8: r5 77.9, r7 71.5; w=1: 63-64 | same |

## Open question: why the small chains drop at round 1
`reports/2026-11-20-mt-r1-drop/report.md`. Ruled out: iris7, labels (40% per (round, task), sensible per stage), V
quality (b100 uses the same frozen 900 V as the 900 chain and still dips), calibration, training, serving. Failures
stall at the second stage and time out. Not the guided-data share alone (900 chain reached 42% by r7 without a drop);
remaining: one-step jump to ~50% guided data from one policy, only 100 base episodes, seed noise.

## Running / queued (gpu_cap priority order, `logs/.gpu_cap_priority`)
1. `scripts/mt_r1_tests_submit.sh` (logs/mt_r1_tests.log): **overs** = mt100cv r1 retrained with the 100 base eps x9
   (90.8% of rollout frames) 18605557-61; **sweep** = r1 at push 0.3 / 0.6 on moka / t64 cont rep0 18605562-65.
2. b100cv r1 eval / ceval 18580609/11, b50cv r1 eval / ceval 18581090/91 (two more small chains at r1).
3. b100 r1 seed-1 retrain 18593303 (critic_tests b100s1; seed test of the dip).
4. mt100cv chain: round 2 re-fed by host 18604928 (`logs/feeders_mt/mt100cv.env`); r1 val eval rerun 18604938; r1
   t64 ceval 18598987 requeued (resumes at 207/300).
5. mtbv (one ResNet V on 450 moka + 450 task-64 base rollouts, no task-64 demos): t64_base E 18604936 / F 18604924
   -> bigv_label 18604937 -> train 18599010 -> checks / evals.
6. b50v900 (b50 r0 with the frozen 900 V labels) check / evals; mid-task r6 check 18604917 / eval 18604923.

## Infrastructure
- **iris7 hung 11-20 ~02:20** (GPU jobs silent, unkillable). Excluded in every pending job and the active scripts
  (`iris[1-4,7-10]`); undo when fixed. Usable GPU nodes now iris5 / iris6 only.
- `gpu_cap` batch job 18593806 (`scripts/gpu_cap.sh` v3): <= 6 GPU jobs (`logs/.gpu_cap`), releases in priority
  order, offers each released job to `iris-hi,iris`. Held jobs sit on iris-hi only (iris QOS MaxSubmitPU 25 counts
  any job listing iris). Jobs preempted on iris sometimes end PREEMPTED without requeue: resubmit by hand.
- Feeders: host 18581098 (b100 / b100cv / b100iql / b200cv / b50cv), host 18604928 (mt100cv). `ag_rounds_feed.sh`
  counts only iris-QOS jobs (FEED_MAX).
- Eval harness counts client-errored episodes as failures: check `episodes.csv` `error` before trusting a number.
- gcloud needs re-auth (`! /iris/u/ankile/karenvo/google-cloud-sdk/bin/gcloud auth login`); value videos
  (reports/2026-11-19-t64-value-videos) not uploaded yet.

## Waiting on Karen
- Drop `--demos` from the chains (demos hurt in both no-demo tests)?
- Rebalancing (rbag / rbfail, `SUBMIT=1 PARTS=rebal scripts/plateau_tests_submit.sh`) still a dry run.
- Retry iris10 for more GPUs (bad card history)?
