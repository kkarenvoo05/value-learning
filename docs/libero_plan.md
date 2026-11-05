# LIBERO plan: value expert on frozen π₀.₅, then advantage-conditioned fine-tuning

Source: Karen's next-steps notes (2026-10-27). Plan drafted by Claude, agreed with Karen 2026-10-27.
Legend as in `PROGRESS.md`: `[x]` done and verified · `[~]` partly done · `[ ]` not started.
Every stage ends with a **Check** and a **Gate: Karen confirms**; nothing downstream starts before its gate.

Goal: use the pre-trained π₀.₅ as the starting point for a value function, in sim first (LIBERO), then move
to the real robot early. Pipeline: frozen `pi05_libero` → rollouts on one task with 50–80% success → value
expert grafted in place of the action expert (one output token, 201-bin head, backbone frozen) → value
checks → RECAP advantages → three fine-tuning arms (imitation on successes, Rohan's data, advantage-conditioned).

Main risk: `pi05_libero` is reported at ~97% average (openpi README: spatial 98.8, object 98.2, goal 98.0,
libero_10 92.4), so few tasks sit in 50–80%. Stage 1 measures per-task success first, with fallbacks.

## Paths and environment

| what | where |
| --- | --- |
| checkpoint (read-only, lab cache) | `/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero` |
| openpi clone (215abfb) + its env (policy server) | `/iris/u/ankile/karenvo/pi05/openpi`, `.venv` |
| LIBERO client env (Python 3.8, new, ours) | `/iris/u/ankile/karenvo/libero/.venv` |
| LIBERO source | `openpi/third_party/libero` (submodule) |
| outputs (eval videos, logs, rollouts) | `/iris/u/ankile/karenvo/libero/` |
| GPU | L40S on `move` (Karen, 2026-10-27); vague job names |

## Stage 0 — Evaluation pipeline reproduces a published number
- [x] Separate Python 3.8 env for the LIBERO client (openpi `examples/libero/requirements.txt` +
      `third_party/libero/requirements.txt` + `openpi-client` + editable libero), headless `MUJOCO_GL=egl`
- [x] One sbatch job on one L40S: policy server (`scripts/serve_policy.py`, `pi05_libero` config, cached
      checkpoint) and the client (`examples/libero/main.py`) on the same node
- [x] Smoke: one task, a few trials, videos saved and watched
- [x] Reproduce one suite (`libero_spatial`, 50 init states × 1 trial = 500 episodes); record seeds,
      commits, command
- **Check:** success within a few points of the published 98.8% (if not: image flip/resize, chunking,
  `replan_steps`, or env versions are wrong, and every later number would be too)
- **Gate: Karen confirms** the evaluation matches the reference

## Stage 1 — Pick the task
- [~] Per-task success: libero_spatial (Stage 0, 500 episodes) and libero_10 (20 trials/task) measured;
      libero_object and libero_goal skipped (published ~98% averages leave room for at most ~one task near 80%)
- [x] Chose **libero_10 task 8, "put both moka pots on the stove"** (the only measured task in 50–80%), more
      than one failure mode (failure types below). Fallbacks if none [ours]:
      (a) LIBERO-PRO perturbed variants (vendored in `ASPIRE/aspire/sim/cap/third_party/LIBERO-PRO`),
      (b) the weakest `libero_10` task, (c) a briefly fine-tuned `pi05_base`
- [x] Split by **init state** (50 fixed per task): 30 train, 10 value-val, 10 policy-test; never train on
      the held-out ones (rollouts from one init state start from the identical scene). Lists in
      `configs/libero_moka.yaml`, drawn as `np.random.default_rng(0).permutation(50)` [ours]
- **Check:** passed. 58/100 [48, 67] on fresh trials; Karen watched successes and failures (2026-10-28)
- **Gate: confirmed by Karen 2026-10-28** (task, split; keep the success / failure labels as they are)
- [x] Track failure types (Karen, 2026-10-28): near misses vs truly stuck, per run, in the log. Labels stay
      success / failure; the types are for analysis only (e.g. does V or advantage conditioning help near
      misses more than stalls)

## Stage 2 — Rollout data from the frozen policy
- [x] ~300–500 episodes on the train init states, failures kept (450: 15 per train state), plus 120 on the
      val init states (12 per state) so V is checked on scenes it never saw [ours]. π₀.₅ samples noise, so
      rollouts from one init state differ. The noise is **not** seeded per episode (the server seeds
      `jax.random.key(0)` once and splits it per call; per-episode seeds need a server change, deferred,
      Karen 2026-10-28). Instead each episode stores its **executed actions** and MuJoCo states (init, after
      the wait steps, end), which is enough to replay it exactly. All episodes run in one job with one
      server, so the noise stream never restarts
- [x] LeRobot format: agentview + wrist images exactly as the policy saw them, state, executed actions,
      prompt, success, endpoint T, timeout flag. Per-step goal predicates and object positions kept in the raw
      files, for failure types
- [x] Targets with `vl.targets` unchanged: −1/step, failure penalty, timeout = failure, horizon = the
      suite's `max_steps`, 201 bins over (−1, 0). Sim labels are exact, so no "unknown" outcomes
- **Check:** open-loop replay of the saved actions from the saved init state reaches the same final state
  and outcome (`scripts/libero_replay.py`); data success rate matches Stage 1; Check A (target arithmetic
  tests, `tests/test_libero_targets.py`); image/frame-index sync (the replay re-renders every frame and
  compares it with the stored one)
- **Gate: Karen confirms** the data

## Stage 3 — Value head on frozen π₀.₅
- [x] 3a. Probe: pooled final-layer prefix tokens (as in `scripts/pi05_smoke.py`) → linear/MLP V. Cheap
      answer to "do the backbone features carry value at all?" Done 2026-10-29 (3 seeds: `reports/libero_seeds.md`)
- [ ] 3b. Grafted value expert: same depth and attention-head layout as the action expert (so it can read
      the backbone's per-layer K/V), narrower width, **one query token**, 201-bin CE head, no flow matching;
      backbone frozen, only the expert + head train [ours: width]
- [ ] Storage decision before caching: full tokens ~2.3 MB/frame (~350 GB for 150k frames) → subsample
      frames or run the backbone forward during training
- **Check:** Check C (tiny-subset overfit) first. Controls on the same val init states: clock-only baseline
  (mean target per second) and this repo's ResNet-18 V trained on the same rollouts. If π₀.₅ does not beat
  the ResNet, the graft isn't earning its cost

## Stage 4 — Is V meaningful?
- [ ] `vl.metrics` (CE, MAE, shape MAE), early-time success-vs-failure AUC, `vl.value_plots` traces, by eye
- **Gate: Karen confirms** the value plots look good (the decision point in the notes)

## Stage 5 — Advantage labels
- [ ] `vl.libero_advantages`, lookahead n in seconds, RECAP 30% threshold. Pipeline written and dry-run on val
      with the ResNet control V (2026-10-29: the bar for the π₀.₅ V, not the fine-tuning labels); the real run
      waits on 3b and the Stage 4 gate
- [ ] **Out-of-fold V** (k-fold over train init states): a V trained on the same rollouts gives ~0
      advantages on them
- **Check:** successes get more positives than failures, time-matched; not a clock artefact; decide on
  failure-ending frames (on Mulligan 72% came out positive)
- **Gate: Karen confirms** the label scheme

## Stage 6 — Policy fine-tuning, three arms
- [ ] Same init (`pi05_libero`), steps, preprocessing and openpi recipe (LoRA if memory requires):
      **A** imitation on successful rollouts · **B** Rohan's data (definition open, see below) ·
      **C** advantage-conditioned on all rollouts (indicator in the prompt, positive at inference)
- **Check:** C run with the *negative* indicator must be worse than with positive; otherwise C's gain comes
  from which data it saw, not from conditioning. Power: 65% vs 75% at α = 0.05, 80% power ≈ **330 trials
  per arm**, on held-out init states; 50 trials cannot separate arms
- **Gate: Karen confirms** the result

## Stage 7 — Real robot (start early)
- [ ] Port the 3b graft to `pi05_droid` on Mulligan (= `PROGRESS.md` Stage 6.1), then Rohan's stream (8.1).
      Start as soon as Stage 4 passes, not after Stage 6

## Open questions
- Arm B "Rohan's data": in sim, Rohan's method (roll out, keep successes) equals arm A. His real DROID
  rollouts, or a second sim round of filtered imitation?
- Standard LIBERO only, or LIBERO-PRO if Stage 1 finds nothing in range?

## Log
- 2026-10-27 Stage 0: client env built at `/iris/u/ankile/karenvo/libero/.venv` (Python 3.8, torch 1.11+cu113,
  robosuite 1.4.1, per openpi `examples/libero` README). **Trap:** the global `~/.libero/config.yaml` points at
  ASPIRE's LIBERO-PRO files, which would silently swap in perturbed benchmark files; this env uses its own
  `LIBERO_CONFIG_PATH=/iris/u/ankile/karenvo/libero/libero_cfg` → openpi's `third_party/libero` (stock).
  Job script `scripts/libero_eval.sbatch` (server + client on one L40S, per-job port). Smoke = libero_spatial,
  1 trial per task (10 episodes), seed 7: job 18139437. Known limit: `main.py` names videos per task and
  outcome, so multiple trials of one task overwrite each other's videos (fine for the reproduction)
- 2026-10-27 smoke done (job 18139437, move4 L40S): **10/10 success**. Server load 6 min (checkpoint over NFS),
  ~12 s per episode. Video frames checked by eye: upright, bowl from stove onto plate. The EGL
  `EGL_NOT_INITIALIZED` tracebacks at exit come from robosuite's `__del__` cleanup, after the results, harmless.
  Full reproduction submitted: libero_spatial × 50 trials/task = 500 episodes, seed 7, job 18139650 (~1.7 h)
- 2026-10-27 reproduction done (job 18139650, move4 L40S, 44:48): **libero_spatial 489/500 = 97.8%**
  (Wilson 95% CI [96.1, 98.8]) vs published 98.8%, so the published number sits at the CI's top edge: consistent.
  Per task 47–50/50 (lowest: bowl in the top drawer 47/50). Seed 7, openpi 215abfb, 50 trials × 10 tasks.
  Results: `/iris/u/ankile/karenvo/libero/runs/repro_libero_spatial_18139650/`. Stage 0 check passes; awaiting gate.
  For Stage 1: no spatial task is anywhere near 50–80%, as expected; libero_10 (published 92.4%) is the place to look
- 2026-10-27 Karen: go ahead with Stage 1. Sweep `libero_10` × 20 trials/task (200 episodes), seed 7, job 18141750.
  These trials only *choose* the task; the Stage 1 check re-measures the chosen task on fresh trials (new seed)
- 2026-10-27 libero_10 sweep done (job 18141750, 32:52): 190/200 = 95.0% (published 92.4%). Only one task in range:
  **"put both moka pots on the stove" 14/20 = 70%** (Wilson 95% CI [48, 86]); the other nine are 19–20/20.
  The one surviving failure video (others overwritten) is a real failure, not an eval artefact: the first pot is
  placed on the stove, then the gripper hovers at the second pot without grasping it until the 520-step
  timeout (a stall, like Mulligan's `timeout_holding_marker`). Candidate task; not yet confirmed (Stage 1 check)
- 2026-10-28 own client `scripts/libero_rollouts.py` (copy of openpi `examples/libero/main.py` 215abfb, same loop)
  [ours]: `--args.task-ids` filter, one video per episode (`epNNNN_task{t}_init{i}_r{r}_{outcome}.mp4`), and
  `episodes.csv` (task, init state, repeat, env seed, success, policy steps, error). `libero_eval.sbatch` uses it
  when `TASKS` is set. Found while reading main.py: (1) the policy's sampling noise is **not** seedable per episode:
  the server seeds `jax.random.key(0)` once and splits it on every infer call, so `--seed` only seeds the env;
  a per-episode noise seed for Stage 2 needs a server-side change. (2) main.py maps trial i → init state i, so the
  20-trial sweep covered moka init states 0–19 only.
  Stage 1 check run: moka pots (libero_10 task 8) × 50 init states × 2 repeats = 100 episodes, env seed 8,
  job 18143532
- 2026-10-28 Stage 1 check done (job 18143532, move4 L40S, 26:24, ~14 s/episode, 0 errors): moka pots **58/100 = 58.0%**
  (Wilson 95% CI [48.2, 67.2]); 29/50 in each repeat. Lower than the sweep's 14/20, but on the same init states 0–19
  this run gets 26/40 = 65%, so the gap is mostly sampling noise; both CIs overlap and the task is in the 50–80% window.
  All 42 failures are 520-step timeouts (no early termination); successes take 354–505 steps (median 412).
  **Failures do not cluster on init states:** per init state 0/2 = 7, 1/2 = 28, 2/2 = 15; repeats agree on 44% of init
  states vs 51% under a shared-p permutation null (one-sided p = 0.91); 7 init states at 0/2 vs 8.8 expected. So with
  2 trials per state there is no evidence that some init states are hard; outcome looks driven by policy noise
  (**wrong, corrected by Stage 2**: with 15 trials per state they differ strongly; see the 2026-10-28 correction).
  0/2 states: 2, 13, 21, 41, 42, 44, 47. Mid/last frames of 10 failures (every 4th) look like real failures with more
  than one mode: one pot on the stove and the gripper at or holding the second pot (the sweep's stall), and neither
  pot placed with the gripper still at the first pot. Videos not yet watched in full.
  Results: `/iris/u/ankile/karenvo/libero/runs/moka_libero_10_18143532/` (`episodes.csv`, 100 videos)
- 2026-10-28 **Stage 1 gate confirmed by Karen**: she watched the videos, task and init-state split are confirmed,
  labels stay success / failure, failure types to be tracked here (near misses vs truly stuck). Per-episode noise
  seeds deferred: Stage 2 stores executed actions and MuJoCo states and checks by open-loop replay instead.
  Split fixed in `configs/libero_moka.yaml` before any Stage 2 rollout: train 0 1 2 3 4 6 8 10 11 16 18 19 20 21 22
  23 24 26 27 28 30 34 35 36 37 38 43 45 46 47 · val 9 12 13 17 25 32 40 42 44 48 · test 5 7 14 15 29 31 33 39 41 49
- 2026-10-28 **Failure types, Stage 1 check (job 18143532, all 42 failures, all 520-step timeouts).** By eye from
  frames at policy steps 200 / 300 / 400 / 460 / 519 of each video [ours: categories and thresholds]:

  | type | what the last frames show | n | episodes |
  | --- | --- | --- | --- |
  | **near miss** | pot 2 carried to or being lowered onto the stove at the timeout | 7 | 9 19 29 47 70 89 96 |
  | **slow** | pot 1 placed late; at timeout only reaching or just lifting pot 2, still progressing | 10 | 5 14 22 26 38 41 52 73 80 92 |
  | **stuck: pot 2 grasp** | at pot 2 for ≥ ~12 s (steps 400–519), never lifts it | 9 | 2 12 24 35 44 60 66 71 87 |
  | **stuck: placing pot 2** | pot 2 at the stove pressed against pot 1 for ≥ ~6–12 s, goal not met | 8 | 25 33 42 54 56 90 94 97 |
  | **stuck: pot 1** | never gets pot 1 onto the stove, or stays at pot 1 and never goes to pot 2 | 5 | 13 21 63 84 91 |
  | **knocked a pot over** | a pot lies on its side; the task can no longer succeed | 3 | 28 65 99 |

  So **17/42 (40%) near miss or slow** (still making progress when time ran out), **22/42 (52%) truly stuck**
  (no progress for the last ≥ 6–12 s), 3/42 unrecoverable. Caveats: five frames per episode, not full videos;
  borderline cases (ep 2 stuck vs slow, ep 70 near miss vs stuck placing). "Stuck: placing pot 2" may include
  pots that are on the stove but fail the `On(..., cook_region)` predicate; Stage 2 records the goal predicates
  per step, so there the types can be computed from sim state instead of by eye
- 2026-10-28 **Stage 2 pipeline** [ours]. `scripts/libero_rollouts.py --args.save-raw` writes one `raw/epNNNN.npz`
  per episode: both images (224², exactly as sent to the server), state, executed actions (original float64),
  a new-chunk flag, the three goal predicates (`On(moka_pot_1, cook_region)`, `On(moka_pot_2, ...)`,
  `Turnon(stove)`; the stove starts on) and object positions after every step, MuJoCo states (init, after the
  10 wait steps, end); ~42 MB per episode. `scripts/libero_stage2.sbatch`: one server, train then val
  states, then `scripts/libero_replay.py` on the first 50 episodes of each split. `scripts/libero_to_lerobot.py`
  (openpi env, lerobot 0.1.0): openpi's LIBERO features and fps (image, wrist_image, state 8, actions 7, 10 fps)
  but 224² policy-input images, plus `episodes.csv` (split, init state, repeat, success, timeout, final goals).
  `python -m vl.libero_targets`: `vl.targets` unchanged with the `targets:` block of `configs/libero_moka.yaml`
  (horizon 520 = libero_10 max_steps, failure penalty 520; T = policy steps − 1). Check A:
  `tests/test_libero_targets.py` (hand-computed: 412-step success starts at −411/1040, bin 121; timeout
  starts at −1039/1040, bin 0, ends at −0.5); full suite 160 passed
- 2026-10-28 Smoke (4 episodes). First attempt (job 18155098) stored actions as float32: the replay matched the
  outcome and goals but drifted ~1e-7 in the sim state. Fixed (raw arrays keep float64); smoke 2 (job 18155725):
  **sim states (start and end), robot state, goal predicates and outcome bit-identical on replay**; images
  identical except one wrist frame per train episode off by 1 grey level (renderer noise, the sim state being
  identical). LeRobot round trip: images, actions and frame indices equal the raw arrays (0 mismatches).
  Throughput ~14 s per episode, replay ~10 s per episode. Full run submitted: job 18155849 (450 train + 120 val)
- 2026-10-28 **Stage 2 rollouts done** (job 18155849, move4 L40S, 3 h 38 min of rollouts + 13 min replay, env seed 9,
  one server): **train 269/450 = 59.8% [55.2, 64.2]**, **val 63/120 = 52.5% [43.6, 61.2]**, all 332/570 = 58.2%
  [54.2, 62.2], matching the Stage 1 check (58/100); train vs val z = 1.44, p = 0.15. 0 client errors; every
  failure is a 520-step timeout; successes take 343–520 steps (median 412). **Replay check: 50/50 train and
  50/50 val episodes replay exactly** (MuJoCo start and end states, robot state, goal predicates every step,
  outcome); images identical in 44/100 episodes and within 1 grey level in all 100 (renderer noise). 24 GB raw
  in `/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849/{train,val}/` (`episodes.csv`, `raw/`, `videos/`,
  `replay.csv`, `failure_types.csv`, `stalls.csv`)
- 2026-10-28 **Correction to the Stage 1 check: init states do differ in difficulty.** With 15 trials per train state
  (12 per val state), per-state success runs from 0/15 (state 2) to 15/15 (state 46); far more spread than a
  shared rate gives (dispersion χ² = 126.7 on 29 df for train, 49.5 on 9 df for val, both p < 0.001). Stage 1's
  "no clustering" came from 2 trials per state, too few to see it; its own 0/2 states are the hard ones here
  (states at 0/2, 1/2, 2/2 in Stage 1 succeed 25%, 60%, 70% in Stage 2). Consequences: (1) the init-state split
  matters more, not less: part of what V can learn is "this scene is hard", and only held-out states test
  whether it reads that from the image; (2) Stage 6 must evaluate every arm on the same test states and treat
  the state as a cluster (paired by state) when computing CIs; the 330-trials-per-arm power estimate assumed
  independent trials. Per state, train (successes/15): 2:0 21:1 38:2 4:4 26:5 47:5 20:5 0:6 8:7 30:8 3:8 34:8
  6:9 23:10 43:10 37:10 24:10 35:11 10:11 22:11 19:11 45:12 16:12 18:12 28:12 1:13 36:13 11:14 27:14 46:15;
  val (/12): 12:1 13:1 17:3 42:3 9:7 48:7 44:9 40:9 25:11 32:12
- 2026-10-28 **Failure types, Stage 2** (`scripts/libero_failure_types.py`, from sim state, not by eye; rules in its
  docstring) [ours]. "Stuck" = what the robot is doing when time runs out. Stall cut-off = 214 steps (21 s), the
  95th percentile of the longest no-pot-moves stretch over the 269 train successes (successes: median 146, max 258;
  the policy often hesitates ~15–25 s at a grasp even when it succeeds). A first version mislabelled two groups
  (recovered early stalls counted as "stuck at pot 1"; jams at the stove counted as near misses, the jam distance
  of 8 cm being below two touching pots); fixed and re-checked against frames of 4 random episodes per type:

  | type | train | val | all |
  | --- | --- | --- | --- |
  | near miss: pot 2 lifted and within 20 cm of pot 1 at the end | 26 | 7 | 33 |
  | slow: no stall over the cut-off, just not done | 19 | 7 | 26 |
  | stalled earlier: a > 21 s stall, then progressed again, ran out of time | 57 | 11 | 68 |
  | stuck at pot 2: final stall > 21 s with one pot on the stove (never lifts pot 2) | 24 | 11 | 35 |
  | stuck placing pot 2: pot 2 pressed against pot 1 for ≥ 6 s at the end | 24 | 9 | 33 |
  | stuck at pot 1: final stall > 21 s with no pot on the stove | 13 | 5 | 18 |
  | knocked over: a pot tilted > 45° at the end | 18 | 7 | 25 |
  | total failures | 181 | 57 | 238 |

  So **127/238 (53%) still progressing when time ran out** (near miss + slow + stalled earlier), **86/238 (36%)
  stuck at the end**, 25/238 (11%) unrecoverable. Close to the Stage 1 by-eye split (40% / 52% / 7%) given the
  different definition of "slow" there. Edge case seen: a pot visibly on the stove can fail `On(pot,
  cook_region)` (e.g. train ep 422, placed at the back edge)
- 2026-10-28 **Stage 2 data built and checked.** LeRobot dataset `/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2`
  (570 episodes, 261,897 frames, 10 fps; job 18161755, 34 min, 21 GB peak; a first attempt inside the interactive
  session was killed at 416/570 by its 16 GiB cap, hence `scripts/libero_to_lerobot.sbatch`). Checks: episode
  count, lengths and frame total equal the rollout CSVs (sum of policy steps 261,897); episode table matches
  source episodes, outcomes and init states; no init state in both splits; 100 frames from 25 random episodes
  identical to the raw arrays (both images, state, actions, frame and episode index). Targets
  (`frame_targets.parquet`, one row per frame): successes −0.499 to 0, timeouts −0.999 to −0.5, 0 out of support.
  **All Stage 2 checks pass; awaiting Karen's gate**
- 2026-10-28 **Stage 2 gate confirmed by Karen**: data approved; failure penalty stays 520 steps (= horizon, so any
  success scores above any failure). Starting Stage 3 with the 3a probe
- 2026-10-28 **Stage 3a set-up** [ours]. `scripts/pi05_libero_features.py` (+ `.sbatch`): frozen `pi05_libero`, each
  saved frame through openpi's own LIBERO input transform, prefix run through the full LLM (final norm, as
  `scripts/pi05_smoke.py`): 968 tokens, 522 valid (2 × 256 image + 10 prompt/state; the 3rd image slot masked).
  Saved per frame, fp16: mean pools over all valid / agentview / wrist / text tokens. Every 4th frame (~65k
  frames). Smoke (job 18163025, A5000): 482 frames, finite, ~12 frames/s after the 51 s load. `python -m
  vl.libero_probe`: linear and MLP (512, GELU) heads, 201-bin CE, early stopping on 5 of the 30 train states held
  out inside train (no tuning on val); val metrics CE, MAE of E[V], success AUC at 5/10/20/30 s; control = clock
  (train bin histogram per frame index). Tests `tests/test_libero_probe.py` (4). **Check C passes** on 2 train
  episodes: CE 0.019 (linear) / 0.009 (MLP) nats, MAE 0.0012 (below half a bin). Full extraction: job 18163142
- 2026-10-28 **Stage 3a results** (features job 18163142, A5000, 94 min, 65,602 frames = every 4th frame of all 570
  episodes; probes jobs 18166231 / 18166319 / 18166593; ResNet job 18166232 + eval 18166594). Val = the 10 value-val
  init states (120 episodes, 13,944 frames); early stopping on 5 inner train states (10 16 22 26 35).

  | model | CE | MAE | AUC 5 s | 10 s | 20 s | 30 s |
  | --- | --- | --- | --- | --- | --- | --- |
  | clock (bin histogram per frame index) | 2.513 | 0.276 | 0.50 | 0.50 | 0.50 | 0.51 |
  | clock by outcome, constant P(success) | 2.315 | 0.288 | 0.50 | 0.50 | 0.50 | 0.51 |
  | π₀.₅ all-token pool, MLP, 201 bins | 3.492 | 0.196 | 0.74 | 0.78 | 0.81 | 0.86 |
  | same, + temperature (1.5, fitted on inner states) | 3.456 | 0.204 | 0.74 | 0.78 | 0.81 | 0.86 |
  | **π₀.₅ all-token pool, MLP, P(success) × clock** | **2.109** | 0.197 | **0.79** | **0.83** | 0.79 | 0.84 |
  | ResNet-18 (`ValueModel`, baseline_lowlr, best step 1500 / 8000) | 3.540 | 0.215 | 0.65 | 0.58 | 0.73 | 0.87 |
  | ResNet-18 + temperature (1.06) | 3.539 | 0.217 | 0.65 | 0.58 | 0.73 | 0.87 |

  Other poolings (agentview, wrist, text, concat) and linear heads land within ~0.03 CE and ~0.05 AUC of the row
  shown (`probe_results.json`); text pooling is not text-only (the prefix attends bidirectionally to the images).
  Findings: (1) **frozen π₀.₅ features carry value information**: success vs failure on unseen scenes from 5 s on,
  and the predicted value tracks scene difficulty (per-state mean E[V] vs state success rate r = 0.78). (2) **They
  beat the ResNet-18 control** on early AUC (0.74–0.79 vs 0.58–0.65 at 5–10 s) and MAE (0.196 vs 0.215), equal at
  30 s; the ResNet overfits fast (train loss 2.2 vs inner CE 3.9). (3) **The plain 201-bin CE is dominated by
  timing**: all failures are 520-step timeouts, so at frame t every failure has the same bin and the clock puts 31%
  of its mass on it (clock CE 1.18 on failure frames, 4.03 on successes). Images give time only roughly, so 201-bin
  probes lose to the clock on CE (3.5 vs 2.5) while winning MAE and AUC; temperature does not fix it. (4) Given the
  same clock (`outcome+clock`: P(success | o) from the features, bins from the per-outcome clock), **the images
  lower CE by 0.206 nats [0.074, 0.352]** (95% bootstrap over the 10 val scenes) vs the constant-P(success) clock,
  29% of the way to a perfect-outcome oracle (CE 1.596). P(success) is calibrated: predicted 0–0.2 / 0.2–0.4 /
  0.4–0.6 / 0.6–0.8 / 0.8–1 → actual 0.10 / 0.32 / 0.51 / 0.63 / 0.81. **Stage 3 check passes** (beats clock on
  CE given the clock, beats ResNet-18). Open for 3b: a V without an explicit clock has to read time from images;
  the 3b expert could take elapsed time as an input [ours, to decide], or the target could be split into outcome
  and time-to-go. Not yet run: outcome+clock for the ResNet (needs a binary-head retrain)
- 2026-10-29 **Correction / refinement of "π₀.₅ beats the ResNet"** (Karen, from the value plots). Per val episode
  (120): π₀.₅ all-token MLP probe MAE 0.195 vs ResNet-18 0.211, lower on 73/120 episodes and 7/10 scenes, and on
  every failure type (stuck at pot 2 0.168 vs 0.237, stalled earlier 0.162 vs 0.220). But the **ResNet is better on
  successes mid-to-late** (MAE at 20–30 s 0.131 vs 0.216, 30–40 s 0.092 vs 0.141; whole successes 0.179 vs 0.193) and
  separates outcomes better at 24–36 s (AUC 0.78–0.96 vs 0.74–0.92); π₀.₅ is better in the first 20 s (AUC at 8–20 s
  0.70–0.81 vs 0.60–0.73). Reading [ours]: the end-to-end ResNet keeps spatial detail (pot in the gripper above the
  stove); the probe's mean pool averages it away. Bias on failures: ResNet +0.240, probe +0.186 (both optimistic).
  For 3b: the expert reads all tokens per layer, so it should close this; **check it at least matches the ResNet on
  success frames after 20 s**. All models here are single-frame (no history)
- 2026-10-29 **Milestone check** (`python -m vl.libero_milestone`, report `reports/libero_milestone.md`, figure
  `reports/figures/libero_milestone/aligned.png`) [ours: definitions in the module docstring]. Milestone = first frame
  with either pot On(cook_region) held ≥ 1 s; 107/120 val episodes reach it with a full ±5 s window (median 20.4 s),
  8 never place a pot. Statistic: mean V over the 5 s after minus the 5 s before, minus the same on time-matched val
  episodes with no pot placed in the window; 95% bootstrap over the 10 val scenes. Reference oracle (mean train target
  per 2 s bin × milestone reached) = the jump a V reading only clock + milestone would show: **+0.343** in V,
  **+0.550** in P(success). **ResNet-18 +0.064 [+0.019, +0.096]** (19% of oracle, positive on 81% of episodes;
  successes +0.100, failures +0.012). **π₀.₅ probes do not jump**: 201-bin −0.018 [−0.061, +0.021], outcome+clock
  −0.040 [−0.076, −0.009]; their P(success) *falls* −0.080 [−0.150, −0.023] for ~8 s after placement, in successes
  (−0.05) and failures (−0.13), then recovers by +10 s. Reading [ours, not yet checked]: the pooled features follow
  arm phase (the move to pot 2 resembles the stuck-at-pot-2 failures) rather than whether a pot is on the stove. So
  the probe's AUC comes from scene difficulty and timing, not task progress; within-episode changes in V, which the
  Stage 5 advantages are built from, are where it fails. The 3b expert needs to pass this check, not just AUC
- 2026-10-29 **Why the probe dips: follow-up** (`python -m vl.libero_milestone_probe`, report
  `reports/libero_milestone_probe.md`, figures `reports/figures/libero_milestone_probe/`; CPU only, cached features).
  (1) **The mean-pooled features know a pot is on the stove**: a probe for "pot placed yet" (same heads / early
  stopping as `vl.libero_probe`, trained on train, scored on val) gets time-matched AUC 0.93–0.95 for every pooling
  (clock 0.50) and 0.99 [0.99, 1.00] in the ±5 s window (`all` MLP); P(placed) steps from ~0.05 to ~0.95 within ~1 s
  of placement. So it is not the averaging: the value head has the information and does not use it. The grid test
  is not expected to remove the dip [ours]. (2) **The dip is the empty-arm trip between pots**: P(success) drops when
  pot 1 is released, is lowest ~4 s later, and comes back when the gripper reaches pot 2 (+0.15 in the near-pot-2
  aligned curve) and rises further while pot 2 is lifted; no single event locks the minimum tighter than the others
  (MAD 2.7–3.8 s). Nearest train neighbours of dip frames are mostly train frames in the same phase (78%), and only
  slightly more often failures than before placement (0.44 vs 0.39; stuck-at-pot-2 0.17 vs 0.10); for val successes
  the neighbours' failure share does not change (0.33 → 0.34), so the head's drop in successes (0.66 → 0.60) is not
  what the neighbours say. (3) **Correction to the milestone check**: aligned on a stopping time, a calibrated
  P(success) should have the same mean before and after; the oracle's +0.34 jump does not separate outcomes (AUC 0.52),
  so "% of oracle" is not a measure of progress-reading. The probe is calibrated before placement (0.58–0.61 vs actual
  0.59) and from +10 s (0.62 vs 0.64) but too low in the 8 s after (0.48–0.51 vs 0.59). What matters for advantages is
  whether the change separates outcomes: success − failure jump ResNet +0.077 [+0.005, +0.149] (AUC 0.63), probe V
  +0.042 [−0.025, +0.107] (0.58), probe P(success) +0.046 [−0.071, +0.162] (0.53); 107 episodes / 10 scenes, so the
  models are not clearly different. Proposed check for 3b and the grid / history runs [ours]: (a) phase calibration
  (this table) and (b) jump-separates-outcome AUC, instead of the raw excess jump
- 2026-10-29 **Dip checks on the grid and history models** (`scripts/libero_dip_compare.py`; the other session's
  `feats_grid4_every4_18167104_attn`, probe `hist20` / `hist20_48_100`, ResNet `c2a20` / `hist20_48_100`; one seed
  each). **Correction to the entry above: the grid probe removes the dip.** Its P(success) rises at placement instead of
  falling (window means 0.51 → 0.59 → 0.62 vs actual 0.59; mean-pool 0.58 → 0.51 → 0.48), transit − before +0.16 vs
  −0.07, and V's time-matched excess jump is +0.048 [+0.006, +0.086] (mean-pool −0.018, ResNet +0.064). So "placed" being
  decodable from the mean pool was not enough: the value head trained on outcome does not pick it up there, and does
  from the grid tokens. Confound: the grid run also changes the head (attention vs MLP). But the grid probe's change is
  still common to successes and failures (AUC jump → success 0.47 for P, 0.58 for V; mean-pool 0.53 / 0.58), and it is
  now too low before placement (0.51–0.55 vs 0.59). **History does not fix the probe dip** (P excess −0.13 / −0.12,
  deeper than without history). Best on outcome separation are the ResNets with history: c2a 2 s AUC 0.69, success −
  failure +0.082 [+0.039, +0.117]; 2/4.8/10 s 0.67; ResNet without history 0.63. Intervals overlap throughout
  (107 episodes, 10 scenes)
- 2026-10-29 **Grid confound resolved: it is the features, not the head** (`scripts/libero_attn_avg.py`, out
  `/iris/u/ankile/karenvo/libero/features/feats_every4_18163142_attnavg`; the grid run's `AttnProbe` and training, seed 0,
  on the averaged features; scored with `scripts/libero_dip_compare.py`). Attention over the one all-token mean (same
  input as the MLP probe) or over the four means still dips: P(success) excess −0.066 [−0.131, −0.021] / −0.068
  [−0.165, +0.012], V excess −0.023 / −0.014 (MLP on averages −0.080 / −0.018; attention on grid +0.045 / +0.048).
  AUC / CE are unchanged (AUC 10 s 0.81 / 0.78; outcome+clock CE 2.115 / 2.125 vs MLP 2.109). So keeping *where* things
  are in the image is what removes the dip. Change at placement still does not predict success (AUC 0.44–0.52).
  One seed per arm
- 2026-10-29 **First LIBERO advantage labels, ResNet V, val only** (`python -m vl.libero_advantages`, report
  `reports/libero_advantages.md`, figures + `val_labels.parquet` in `reports/figures/libero_advantages/`). RECAP
  n-step advantage on the every-4th-frame grid, n = 1.2 / 2 / 4.8 s, thresholds 70% / 30% positive; V = 3-seed
  ResNet ensemble, seed 0 alone, 3-seed 2 s-history ensemble, and an image-free clock control. Val predictions are
  out-of-sample (trained on train init states only), so no out-of-fold V is needed for this check. Time-matched (same
  scene, same frame), mid-episode: successes +5.4 points more positive [+2.3, +8.6] at 2 s (AUC 0.74), +8.6
  [+5.0, +12.5] at 4.8 s (AUC 0.81); clock exactly 0. Similar to the station R0 (+6.1 at 2 s). Raw differences are
  confounded by episode length (clock alone: raw AUC 1.000). No clear difference between the ensemble, one seed,
  or history. Failure-ending frames 61% positive at the 70% threshold (open decision, as on Mulligan). This is a dry
  run with the control V, run ahead of the Stage 4 gate: the plan's fine-tuning labels come from the 3b π₀.₅ V,
  which must at least match these numbers. ResNet labels for fine-tuning would be a plan change (Karen to decide)
- 2026-10-29 **Seeds: grid / history tests at 3 seeds** (`reports/libero_seeds.md`; tables
  `scripts/libero_seed_table.py reports/libero_seeds_arms.json` and `scripts/libero_dip_seeds.py`). New in
  `vl.libero_probe` [ours]: `--seed` (head init and batch order; inner split fixed), `--pooling grid2 grid1` averaged
  exactly from the grid4 features, `--history` on grid tokens (past tokens appended along the token axis + one
  has-past flag token per lag); `scripts/libero_resnet.sbatch` takes `SEED`. Tests +2 (170 pass). Seed 0 reproduces
  the earlier runs exactly. Selection on inner CE only. Results, mean ± sd over 3 seeds: (1) **inner CE: grid and
  history stack**: grid 1/2/4 3.450 / 3.410 / 3.382; probe + past 2/4.8/10 s 3.401 (single 3.440); **grid4 + past
  2/4.8/10 s 3.307 ± 0.022, best arm**; grid4 + 1 past 2 s 3.387 (no gain over grid4). ResNet: single 3.530, one past
  frame 1.2 / 2 / 4.8 s 3.489 / 3.473 / 3.446, three 3.364. (2) **None of it is outcome**: outcome+clock CE 2.10–2.14
  for every probe arm, so the gains are timing. (3) **No probe reads progress**: no probe arm has a positive milestone
  excess on 3/3 seeds (grid4 +0.013 ± 0.033: +0.048 / +0.011 / −0.019), jump → success AUC 0.46–0.57, success MAE after
  20 s 0.155–0.178; every ResNet arm is positive on 3/3 (+0.057 … +0.079, AUC 0.62–0.71, success MAE > 20 s
  0.119–0.128). (4) **Correction to the two dip entries above**: both rest on seed 0, the lucky seed. The probe outcome
  head at placement is not stable across seeds (grid4 P dip +0.159 / −0.101 / +0.003; mean-pool P dip −0.074 /
  +0.134 / +0.081; grid1 = attention on per-image means, P excess −0.060 / −0.163 / −0.069), so neither "the
  mean-pool probe dips" nor "the grid removes it" nor "features, not head" holds at 3 seeds. **3a is done**; for 3b
  [ours, to decide]: spatial tokens come with per-layer reading; history (2/4.8/10 s) as a second arm after a
  single-frame expert (it multiplies backbone compute); the clock needs a decision since every CE gain is timing; 3b
  passes only if it beats the ResNet on success MAE after 20 s and has a positive milestone excess on every seed
- 2026-10-29 **Stage 1 revisited: task sweep for a one-step task** (Karen: run libero_90 + object + goal, 20 trials per
  task, 2 A5000). Reason [ours]: moka is two-step with every failure a 520-step timeout, so timing dominates CE and the
  mid-episode trip between pots is ambiguous; a one-step task at 50–80% would give a cleaner first test for 3b, with
  moka kept as the harder case. `pi05_libero` was trained on spatial / object / goal / libero_10 only (openpi
  `convert_libero_data_to_lerobot.py`), so libero_90 is new to it and its success rates should spread more.
  `scripts/libero_sweep.sbatch` (one server, suites in sequence, `scripts/libero_rollouts.py`, init states 0–19 ×
  1 trial, env seed 7 as the libero_10 sweep): job 18177252 = libero_90 tasks 0–49; job 18177253 = libero_90 50–89,
  then libero_object, libero_goal (2,200 episodes). These trials only choose candidates; a chosen task gets a fresh
  check on all 50 init states before any gate, as moka did
- 2026-10-29 **Task sweep done** (jobs 18177252, 4:32, and 18177253, 4:24; 0 client errors). Results:
  `/iris/u/ankile/karenvo/libero/runs/sweep90{a_18177252,b_18177253}/<suite>_<range>/episodes.csv`.
  libero_90 0–49: 260/1000 = 26% (28 tasks at 0/20, 6 at 20/20); libero_90 50–89: 259/800 = 32% (21 at 0/20, 6 at
  20/20); libero_goal 192/200 = 96% (lowest: wine bottle on the rack 17/20); libero_object 200/200. So goal / object
  are saturated and libero_90 is mostly all-or-nothing. libero_90 tasks at 8–16/20 (all failures 400-step timeouts;
  success steps = when successful episodes finish):

  | task | instruction | success | success steps |
  | --- | --- | --- | --- |
  | 60 | pick up the black bowl on the left and put it in the tray | 10/20 | 105–125 |
  | 12 | put the black bowl at the back on the plate | 14/20 | 82–203 (median 95) |
  | 29 | put the black bowl in the top drawer of the cabinet | 14/20 | 111–274 |
  | 2 | put the black bowl in the top drawer of the cabinet (other scene) | 12/20 | 154–392 |
  | 33 | close the microwave | 15/20 | 244–354 |
  | 70 | put the chocolate pudding to the right of the plate | 15/20 | 80–329 |
  | 30 | put the black bowl on the plate | 16/20 | 112–400 |
  | 38 | put the right moka pot on the stove | 10/20 | 354–398 |
  | 64 | stack the right bowl on the left bowl and place them in the tray | 9/20 | 235–383 |
  | 59 | pick up the tomato sauce and put it in the tray | 8/20 | 153–395 |

  Pick [ours]: **task 60** (one step, successes end far before the cap, so the outcome is not mostly timing);
  task 12 next. Avoid 38: successes end at the cap, the moka timing problem again. 20 trials → CIs ≈ ±20 points.
  Karen: log and run the 50-state check on task 60. Check run: libero_90 task 60 × 50 init states × 2 repeats
  = 100 episodes, env seed 8 (as moka), job 18185238
- 2026-10-29 **Stage 3b set-up and Check C** (`scripts/pi05_value_expert.py` + `.sbatch`, openpi env). Graft as in
  `Pi0.sample_actions`: the frozen PaliGemma LLM runs the prefix once (both images + prompt + tokenized state, openpi's
  LIBERO input transform) and returns every layer's K/V; the value expert is expert 1 of openpi's own two-expert
  `gemma.Module` (PaliGemma = expert 0, never run, so no parameters) and its single learned query token, at position =
  number of valid prefix tokens, attends to the valid prefix + itself layer by layer. 18 layers, 8 query heads, 1 KV
  head, head dim 256 (shared with the backbone), **width 512, MLP 2048 [ours] → 99.2 M parameters**; no adaRMS, no flow
  matching. Heads [ours]: 201-bin value (CE on the target bin) + 2-way P(success) (CE on the outcome, for CE given the
  clock as in 3a); loss = sum. Single frame, no elapsed time (Karen). Backbone frozen by construction (its K/V come
  from a separate jitted function). `--selftest` (CPU, dummy sizes): the cached path equals one joint forward of prefix
  + query through the two-expert module (max |diff| 0), padded prefix K/V do not change the output, valid ones do.
  **Check C passes** (job 18177581, A5000, 5.5 min; first attempt 18177018 died of host OOM because the jitted step
  captured the 4 GB K/V cache as a constant, fixed by passing it as an argument): first 2 train episodes, every 4th
  frame (222 frames), full batch, AdamW 3e-4 cosine, 1000 steps → train CE 0.0001 nats, MAE 0.0012 (binning floor;
  3a probes 0.009–0.019 / 0.0012), outcome accuracy 1.0. Expert step on 222 frames ~0.07 s, so a full run is bound by the
  backbone forward (~10–20 frames/s per GPU); storage decision: no K/V cache (630 GB at every 4th frame vs 430 GB free
  on /iris/u/ankile), backbone run during training
- 2026-10-29 **Stage 3b full single-frame runs, 3 seeds: fails the 3b pass rule** (jobs 18178421 / 18178422 /
  18178423, A5000, ~4.2 h each; smoke 18178077; `/iris/u/ankile/karenvo/libero/runs/value_expert_single_s{0,1,2}_*`).
  Moka, same splits as 3a (fit 375 episodes / 43,087 frames, inner 5 states, val 10 unseen states, every 4th frame),
  width 512 / MLP 2048 (99.2 M), batch 32, AdamW 1e-4 (wd 0.01, 500 warmup, cosine), 5,400 steps, early stopping on
  inner CE; **no image augmentation** (`backbone_kv` calls `preprocess_observation(train=False)`; the 3a probes and
  the LIBERO ResNet had none either). Scored with `scripts/libero_seed_table.py` (same metrics as
  `reports/libero_seeds.md`), mean ± sd over 3 seeds:

  | arm | inner CE | MAE | success MAE > 20 s | AUC 5 / 10 / 20 / 30 s | milestone excess (seeds > 0) |
  |---|---|---|---|---|---|
  | 3b expert, E[V] | 3.556 ± 0.027 | 0.188 ± 0.009 | 0.159 ± 0.019 | 0.78 / 0.77 / 0.82 / 0.84 | +0.020 ± 0.026 (2/3) |
  | 3b expert, P(success) | | | | 0.79 / 0.78 / 0.83 / 0.81 | +0.033 ± 0.012 (3/3) |
  | 3a probe single | 3.440 ± 0.004 | 0.192 | 0.170 | 0.76 / 0.79 / 0.82 / 0.86 | −0.011 (1/3) |
  | ResNet-18 single | 3.530 ± 0.016 | 0.211 | **0.128** | 0.73 / 0.62 / 0.75 / 0.87 | **+0.063 (3/3)** |

  Per seed E[V]: success MAE > 20 s 0.145 / 0.180 / 0.152, milestone +0.042 / −0.009 / +0.028. Val CE 3.508 / 3.533 /
  3.511 (temperature 1.20–1.28). (1) **Fails**: loses to the ResNet on success MAE > 20 s (0.159 vs 0.128) and the
  milestone excess is not positive on every seed. (2) Slightly better than the 3a probe at progress (success MAE 0.159
  vs 0.170, milestone +0.020 vs −0.011) but still far below the oracle +0.343: mostly reading time. (3) **Overfits
  early**: best inner CE at step 1000 / 1000 / 2000 (32k–64k frames seen, ~1–1.5 passes over fit), then rises to 4.46
  / 4.54 / 4.47 at 5,400 while train CE falls to ~1.6. With 375 episodes and neighbouring frames near-identical, the
  expert learns the episodes. (4) The P(success) head's milestone is positive on 3/3 (the 3a probe P head flipped sign
  across seeds), small. Next [ours, to decide]: regularise before judging the graft (augmentation, narrower expert,
  stronger weight decay; arms not yet chosen); a cleaner one-step task (libero_90 task 60) is running separately
- 2026-10-29 **Stage 3b regularisation screen, seed 0** (Karen: run the three arms). `--augment` added to
  `scripts/pi05_value_expert.py`: openpi's own train-time augmentation (`preprocess_observation(train=True)`, as pi05
  was trained: main camera 95% random crop + resize + rotate ±5°, all cameras ColorJitter brightness 0.3 / contrast
  0.4 / saturation 0.5) on fit batches only, never on inner/val [ours: openpi's scheme, not the LeRobot one used for
  the Mulligan ResNet, so the frozen backbone sees its training distribution]; `--weight-decay` added (default 0.01
  unchanged). Check (`/iris/u/ankile/karenvo/libero/checks/augcheck.py`, job 18186578): rng = None gives K/V
  identical to before (diff 0.0); augmented K differ by 0.24 (|K| 1.12), two keys differ, masks unchanged. Smoke
  18186653 (all three flags) runs end to end; width 256 / MLP 1024 = 35.5 M parameters. Arms, all = the 3b recipe +
  `--augment`, 5,400 steps, 6 h limit: **A** width 512 (job 18186727), **B** width 256 / MLP 1024 (18186728),
  **C** weight decay 0.1 (18186729, after A to stay within 3 GPUs). Compare with 3b seed 0 (success MAE > 20 s
  0.145, milestone +0.042, best step 1000); the seed sd is ~0.02, so a single-seed gap < ~0.03 is noise. The winner
  gets seeds 1–2; the pass rule is unchanged. If augmentation helps, the ResNet bar needs an augmented ResNet too.
  C was moved (Karen) to an rtxpro6000 on move5 without the dependency: job 18186729 cancelled, rerun as **18188578**.
  **Arm A result** (augmentation only, width 512; job 18186727, move4 L40S, 2 h 35 min, 24.9 frames/s; scored with
  `scripts/libero_seed_table.py`): **no change from 3b**. Inner CE 3.557 (3b s0 3.547), val CE 3.494 (3.508),
  MAE 0.192 (0.189), success MAE > 20 s 0.149 (0.145), AUC 5 / 10 / 20 / 30 s 0.80 / 0.78 / 0.80 / 0.86
  (0.80 / 0.78 / 0.82 / 0.87), milestone excess E[V] +0.003 (+0.042; the 3b seed range is −0.009 … +0.042), P(success)
  milestone −0.052 (3b: positive on 3/3). Best step still 1000; augmentation only slows the later rise (inner CE at
  5,400: 4.215 vs 4.456). All gaps are within seed noise, and A still misses the ResNet bar (0.128, +0.063 on 3/3).
  Reading [ours]: at this capacity, augmentation does not let the expert extract more value signal from the 375
  episodes; it only memorises more slowly. B and C pending.
  **Arm B result** (augmentation + width 256 / MLP 1024, 35.5 M; job 18186728, move4 L40S, 2 h 46 min): **also flat
  on the pass metrics**. Inner CE 3.567, val CE 3.549 (3b s0 3.508), MAE 0.177 (0.189; lowest of any 3b run, 3b seeds
  0.178–0.196), success MAE > 20 s 0.148 (0.145), AUC 0.78 / 0.79 / 0.82 / 0.85, milestone excess E[V] +0.049 (+0.042),
  P(success) milestone +0.101 (3b +0.020 … +0.044; largest so far, one seed). Best step 2500 (3b 1000) and a slower
  rise (inner CE at 5,400: 3.941 vs 4.456), so a third of the capacity overfits later without losing anything, but
  success MAE > 20 s still misses the ResNet's 0.128 by ~0.02. Reading [ours]: the narrower expert is at least as good
  for a third of the parameters and should be the default for later 3b runs; the P(success) milestone is the one
  number worth re-checking on seeds 1–2. C pending. B seeds 1 and 2 (Karen): jobs 18189370 / 18189371, move4 L40S,
  same flags with `--seed 1` / `--seed 2` (`value_expert_aug_w256_s{1,2}_*`).
  **Arm C result** (augmentation + weight decay 0.1, width 512; job 18188578, move5 rtxpro6000, 2 h 00 min, 35.3
  frames/s vs 24.9 on L40S): **flat, like A**. Inner CE 3.555, val CE 3.526 (3b s0 3.508), MAE 0.194 (0.189), success
  MAE > 20 s 0.151 (0.145), AUC 0.78 / 0.79 / 0.81 / 0.84, milestone excess E[V] +0.026 (+0.042), P(success) milestone
  +0.022. Best step 1000; inner CE at 5,400 4.153 (A 4.215, 3b 4.456), so wd 0.1 slows the rise about as much as
  augmentation alone and no more. **Screen verdict [ours]:** at seed 0 none of A / B / C moves success MAE > 20 s
  (0.148–0.151 vs 3b 0.145, bar 0.128) or the E[V] milestone beyond seed noise. Augmentation and weight decay only
  slow memorisation; only the narrower expert (B) moves the best step (2500) and the overall MAE (0.177). Weight
  decay 0.1 is not worth carrying forward; B (width 256 + augmentation) is the candidate, pending seeds 1–2.
  **Arm B, 3 seeds** (seeds 1 / 2: jobs 18189370 / 18189371, 2 h 34 / 2 h 32 min; best steps 2500 / 2500 / 2000),
  mean ± sd vs 3b: inner CE 3.531 ± 0.037 (3.556 ± 0.027), val CE 3.549 / 3.508 / 3.478, MAE 0.180 ± 0.005
  (0.188 ± 0.009), **success MAE > 20 s 0.146 ± 0.003** (0.159 ± 0.019; per seed 0.148 / 0.142 / 0.147), AUC 5 / 10 /
  20 / 30 s 0.77 / 0.78 / 0.81 / 0.85, **E[V] milestone excess +0.027 ± 0.019, positive on 3/3** (+0.049 / +0.017 /
  +0.014; 3b +0.020, 2/3), P(success) milestone +0.040 ± 0.064 (+0.101 / +0.046 / −0.027: the seed-0 value did not
  hold). **Still fails the 3b pass rule:** the milestone half now holds (positive on every seed, though less than half
  the ResNet's +0.063 ± 0.023 and far below the oracle +0.343), but success MAE > 20 s misses the ResNet's 0.128 ±
  0.011 by 0.018. Reading [ours]: B is a real but small improvement over 3b, mostly in stability (3b's seed-1 outlier
  0.180 is gone) at a third of the parameters; it does not close the gap to the ResNet. Next: the progress probe
  (does the frozen backbone encode "pot on the stove" at all, vs the ResNet)
- 2026-10-30 **Progress probe set-up** (Karen: set it up). Question [ours]: every pi05 model (3a, 3b, A/B/C) reads
  time but barely reacts when a pot lands; the losing models share the frozen backbone, the ResNet is fine-tuned. Does
  the frozen backbone encode "a pot is on the stove" at all? If not, no head can fix it (→ adapt the backbone, e.g.
  LoRA); if it does, the value heads are not using it (→ remove shortcuts: clock input, state tokens). Code:
  `scripts/pi05_progress_features.py` + `.sbatch` (openpi env): per-layer prefix **values** V (what the 3b expert
  attends to; keys skipped, they carry RoPE positions) at layers 1 / 5 / 9 / 13 / 17, grid-4 pooled as in 3a (33
  tokens × 256), plus the final query output h of the 3b s0 and B s0 experts; `vl.libero_progress` + `.sbatch`:
  label y = at least one pot On(cook_region) in image t (goals[t − 1, :2], the milestone convention), probes = 3a
  heads with 2-way CE (linear / MLP on vectors, attention head on token sets), same frames / inner split / val.
  Sources: pi05 final-layer mean pools (all / agentview / wrist, the cached 3a features), pi05 V per layer and all
  layers, expert h, frozen ImageNet ResNet-18, the 3a ResNet V's fine-tuned backbone, object positions (positive
  control, should be ~1), clock. **Metric = time-matched AUC** (AUC within 2 s bins, pair-weighted; a clock scores
  0.5 by construction), 95% CI by bootstrap over the 10 val init states; plain AUC alongside. A within-episode
  before/after-milestone AUC was considered and dropped: inside an episode "after" also means "later", so a clock
  scores ~1 on it. Label check on the real episodes: y = 1 on 42.4% of fit / 42.1% of val frames, 112/120 val
  episodes reach the milestone, val frames show 0 or 1 pot (2 only at the final step, off the 4-frame grid); pots
  also touch or leave the stove outside the held milestone in 111/530 episodes, so y is the current image, not
  "ever placed"; **clock on val: time-matched AUC 0.500, plain AUC 0.885** (plain AUC would reward the clock).
  +4 tests (`tests/test_libero_progress.py`; full suite 174 pass). Jobs (moved from rtxpro6000 to A5000 while queued, Karen): extraction **18191428** →
  probe **18191429** (after it); out `/iris/u/ankile/karenvo/libero/features/progress_18191428`,
  `reports/libero_progress.{md,json}`. Reading rule [ours, set before the result]: the object-position control must
  reach ≥ 0.95 or the set-up is broken; "frozen pi05 lacks progress" if the best pi05 source's time-matched AUC is
  below both ResNets' with non-overlapping CIs; "present but unused" if a pi05 V source matches the ResNets
- 2026-10-29 **Task 60 50-state check done** (job 18185238, move4 L40S, 18:09, 0 errors; libero_90 task 60 × 50 init
  states × 2 repeats, env seed 8; `/iris/u/ankile/karenvo/libero/runs/bowltray_libero_90_18185238/`):
  **51/100 = 51.0%** (Wilson 95% CI [41.3, 60.6]); repeats 28/50 and 23/50; init states 0–19 give 18/40 vs the
  sweep's 10/20. In the window, at its bottom edge. All 49 failures are 400-step timeouts; successes take 101–389 steps
  (median 115). **Outcome is almost fixed by the init state:** per state 0/2 = 21, 1/2 = 7, 2/2 = 22 vs 12 / 25 / 13
  under a shared rate, i.e. the two repeats disagree on 7/50 states (14%). Unlike moka (Stage 1 showed no clustering
  at 2 repeats), here it shows already at 2. Consequence [ours]: a V that only recognises the init layout from the
  first frame would score well on train states, so value-val / test must stay on unseen init states (the split does
  this) and the within-episode signal has little to explain; the 0/2 states could also be a near-deterministic
  "policy can't do this layout" rather than a recoverable failure. 0/2 states: 4 6 7 10 12 13 15 16 19 22 25 28 33 34
  35 36 39 40 42 47 49. Videos not yet watched. Next: task 12 check, same design (job 18187311 cancelled while
  waiting for a GPU slot; rerun on an A5000 as job 18188382, Karen)
- 2026-10-29 **Task 12 50-state check done** (job 18188382, move3 A5000, 29:41, 0 errors; libero_90 task 12 × 50 init
  states × 2 repeats, env seed 8; `/iris/u/ankile/karenvo/libero/runs/bowlplate_libero_90_18188382/`):
  **37/100 = 37.0%** (Wilson 95% CI [28.2, 46.8]), **below the 50–80% window**; repeats 19/50 and 18/50. States 0–19
  give 20/40 (sweep 14/20); states 20–49 only 17/60. All 63 failures are 400-step timeouts; successes take 80–188 steps
  (median 92). Init state fixes the outcome even more than task 60: per state 0/2 = 30, 1/2 = 3, 2/2 = 17 vs 19.8 /
  23.3 / 6.8 under a shared rate; repeats agree on 94% of states (task 60 86%, moka Stage 1 44%). Verdict [ours]:
  task 12 is out, task 60 not better than moka; **moka stays the task** (pending Karen). 0/2 states: 0 2 3 5 7 10 11
  14 16 21 22 23 24 26 27 30 31 32 33 34 38 39 41 42 43 44 45 46 47 49
- 2026-10-29 Karen: 50-state checks on the remaining in-window one-step libero_90 candidates, tasks 29 2 33 70
  (skipped [ours]: 30 at the window's top edge, 38 successes at the cap, 64 / 59 already below 50%). Same design,
  400 episodes in one job on an A5000, env seed 8: job 18189388, queued after value-expert job 18188578 to stay
  within the GPU limit. Switch only if a task lands in 50–80% *and* its repeats often disagree (moka: agree 44%)
- 2026-10-30 **Tasks 29 / 2 / 33 / 70 50-state checks done** (job 18189388, move3 A5000, 1:36:22, 400 episodes, 0 errors,
  env seed 8; `/iris/u/ankile/karenvo/libero/runs/check4_libero_90_18189388/`). All failures are 400-step timeouts.
  "Agree" = share of init states whose 2 repeats match; "null" = expected agreement under one shared rate p² + (1−p)²:

  | task | success [Wilson 95%] | repeats | agree (null) | per state 0/2, 1/2, 2/2 | success steps (median) |
  | --- | --- | --- | --- | --- | --- |
  | 2 black bowl → top drawer | 60/100 [50.2, 69.1] | 28, 32 | 64% (52%) | 11, 18, 21 | 87–394 (288) |
  | 29 black bowl → top drawer | 65/100 [55.3, 73.6] | 31, 34 | 78% (55%) | 12, 11, 27 | 111–319 (125) |
  | 33 close the microwave | 91/100 [83.8, 95.2] | 48, 43 | 86% (84%) | 1, 7, 42 | 222–392 (261) |
  | 70 chocolate pudding right of plate | 78/100 [68.9, 85.0] | 39, 39 | 80% (66%) | 6, 10, 34 | 82–400 (219) |

  For comparison: moka 58%, agree 44% (null 51%); task 60 51%, 86% (50%); task 12 37%, 94% (53%).
  Reading [ours]: 33 is out (91%); 70 is at the window's top edge with a clear layout effect. **Task 2 is the only
  libero_90 task close to moka on layout dependence** (+12 points over null vs moka −7, task 29 +23, task 60 +36), but
  its successes come late (median 288 of 400, cf. moka 412 of 520), so timing still matters. 29 has early successes
  but a stronger layout effect. Neither clearly beats moka; recommendation stays moka, with task 2 as the fallback /
  second task if Karen wants one. Videos not yet watched
- 2026-10-30 **Progress probe done** (extraction 18191428, 1:36 on move3 A5000; probe 18191429, 7:37;
  `reports/libero_progress.{md,json}`). Time-matched AUC for "≥ 1 pot on the stove in image t" on val, with 95% CI over
  the 10 val init states. Best head per source:

  | source | auc_time [95% CI] | plain auc |
  | --- | --- | --- |
  | clock | 0.500 | 0.885 |
  | object positions (control, MLP) | 0.955 [0.923, 0.980] | 0.985 |
  | pi05 final-layer pool: all / agentview / wrist | 0.950 / 0.963 / 0.934 | 0.98–0.99 |
  | pi05 V layer 1 / 5 / 9 / 13 / 17 (attn head) | 0.984 / **0.987 [0.978, 0.993]** / 0.972 / 0.976 / 0.959 | 0.988–0.995 |
  | pi05 V all layers (attn head) | 0.974 [0.957, 0.984] | 0.992 |
  | expert h, 3b s0 (w512) / B s0 (w256 + aug) | 0.954 [0.928, 0.978] / 0.951 [0.917, 0.982] | 0.985 |
  | ResNet ImageNet (frozen) / ResNet V (3a, fine-tuned) | 0.977 [0.962, 0.988] / 0.980 [0.965, 0.990] | 0.993 |

  Control: 0.955 clears the ≥ 0.95 bar, but only narrowly and *below* the image sources, so it is not the ceiling it
  was meant to be. Cause checked [ours]: `object_pos` has the two pots and the pot→gripper vectors but **no stove
  position**. On val, a pot that is on the stove sits in a tight band (y sd 1.5 cm, z sd 1.8 mm), so positions carry
  the label. The likely remaining gap is that the predicate uses contact, not position, around placements; this is
  not verified. Label and set-up are otherwise sane: the clock gets 0.500 exactly, and every image source is ≥ 0.92.
  **Reading (pre-set rule): progress is present in the frozen backbone.** The best pi05 source (V at layer 5) matches
  the ResNets with overlapping CIs, so it is not "lacks progress". It is strongest early (layers 1–5), falls toward
  the final layer (0.959 at 17, final pool 0.95), and the experts' own h carries it at the final-layer level (0.95,
  with an overlapping CI vs the ResNet, marginally). Two caveats [ours]:
  (1) every source is near the ceiling: 0.95 vs 0.98 is about 5% vs 2% of mis-ordered same-time pairs. So the
  milestone is **not** where the 3b success-MAE gap comes from (0.146 vs 0.128). That gap has to come from timing
  within phases or from overfitting (best step 1000–2500).
  (2) LoRA / backbone fine-tuning is therefore not the next step.
  **Next arms, queued** (the "present" branch of the plan): two new flags in `scripts/pi05_value_expert.py`, both on
  the B recipe (w256 / MLP 1024, aug, lr 1e-4, 5,400 steps).
  - `--blank-state`: robot state → zeros in fit, inner and val, so the state prompt tokens are constant. This tests
    the proprioception shortcut, and it puts pi05 on the same inputs as the ResNet V, which sees only the two images.
  - `--time-input`: Fourier features of t / 520 are projected and added to the query token. The ResNet V has no time
    input, so a win here needs a ResNet + time control before it counts.
  CPU selftest passes; a time-input check shows h changes only for the row whose t changed; suite 174 pass.
  Jobs:
  - smoke 18193275 (both flags, 2 episodes, 20 steps). The first smoke, 18193233, failed on my arguments (20 steps <
    the 500-step warmup) after the model had built; the chain was resubmitted;
  - D = blank-state, on A5000: s0 18193276 → s1 18193278 → s2 18193280;
  - E = time-input, on L40S: s0 18193277 → s1 18193279 → s2 18193281.
  s0 runs only after the smoke passes, and each later seed after the previous one, so at most 2 GPUs are in use.
  Out: `/iris/u/ankile/karenvo/libero/runs/value_expert_{blankstate,timein}_w256_s{0,1,2}_<job>`. Same pass rule as
  3b: success MAE > 20 s below the ResNet's 0.128±0.011, and the milestone positive on every seed. B for comparison:
  0.146±0.003, +0.027±0.019. **Off-ramp [ours]:** if neither D nor E passes, Stage 3b closes negative and Stage 5
  uses the ResNet V.
- 2026-10-30 **Arm E (time-input) seed 0 result; arm D seed 0 hit the time limit.**
  **E s0** (job 18193277, move4 L40S, 2 h 32 min; best step 2500; scored with `scripts/libero_seed_table.py`, E[V]
  head), vs B 3 seeds / ResNet bar:

  | arm | inner CE | MAE | success MAE > 20 s | AUC 5 / 10 / 20 / 30 s | milestone excess |
  | --- | --- | --- | --- | --- | --- |
  | B (w256 + aug, 3 seeds) | 3.531 ± 0.037 | 0.180 | 0.146 ± 0.003 | 0.77 / 0.78 / 0.81 / 0.85 | +0.027 ± 0.019 |
  | **E s0** (B + time input) | 2.380 | 0.170 | 0.139 | 0.77 / 0.77 / 0.80 / 0.86 | +0.016 |
  | ResNet-18 single | 3.530 | 0.211 | 0.128 ± 0.011 | 0.73 / 0.62 / 0.75 / 0.87 | +0.063 (3/3) |

  Val CE 1.896 (B 3.478–3.549), temperature 1.13. P(success) head: AUC 0.79 / 0.81 / 0.82 / 0.87, milestone +0.026.
  Reading [ours]: (1) **does not pass at seed 0**: success MAE > 20 s 0.139 is ~0.007 better than B, but still above
  the ResNet's 0.128 (inside its ±0.011 sd, so seeds 1–2 decide); the milestone (+0.016) is positive but at the low end
  of B's range. (2) Most of the inner-CE drop (3.53 → 2.38) is the clock itself: the target is −t/T for successes, so
  an explicit t sharpens the bin; it is not evidence of more progress reading. AUCs are unchanged. (3) Even a pass
  here needs the ResNet + time control before it counts (set-up entry above). E s1 18193279 running (L40S), s2
  18193281 after it.
  **D s0** (blank-state, job 18193276) was cancelled at the 4 h limit at step ~5,050 of 5,400: on the A5000 it ran at
  13.1 frames/s (L40S ~25), so 5,400 × 32 frames + evals do not fit. `best.msgpack` (step 2500, inner CE 3.564; at
  5,000 3.857) was saved, but no val scoring. D s1 / s2 (18193278 / 18193280) are on A5000s too and will hit the same
  limit. Fix [ours, Karen chose this over resubmitting]: `--eval-only LOG` in `scripts/pi05_value_expert.py` skips
  training, loads `--out`'s `best.msgpack`, reads the inner-CE history from the run's slurm log, then the same
  temperature / val scoring as a full run (no-op if `results.json` exists, so a seed that does finish is not
  overwritten). The best step lies well before the cut, so selection is unchanged (it would differ only if the last
  ~350 steps had beaten 2500, which the rising inner CE rules out for s0). Eval jobs, 1 h limit, A5000: D s0
  **18195673**; D s1 18195674 (afterany 18193278); D s2 18195675 (afterany 18193280). Suite 174 pass.
- 2026-10-30 **Arm E seed 1: E fails the 3b pass rule** (job 18193279, move4 L40S, 2 h 33 min, best step 3000).
  s1: success MAE > 20 s 0.135, milestone excess **−0.024**, AUC 0.79 / 0.74 / 0.83 / 0.87, val CE 1.933. E over s0–s1:
  success MAE > 20 s 0.137 ± 0.003 (B 0.146; ResNet 0.128 ± 0.011), milestone −0.004 ± 0.028. The milestone half needs
  a positive value on every seed, so E fails whatever s2 (18193281, running) gives. Reading [ours]: the clock buys
  ~0.01 of success MAE and nothing on progress; it moves the expert towards reading time, not towards reacting
  to the pot. D (blank state) pending its eval-only jobs.
- 2026-10-30 **Half-data check set-up** (Karen: run it for both models). Question [ours]: is 375 fit episodes the
  bottleneck, and specifically π₀.₅'s? Every 3b arm and the ResNet control overfit within 1–1.5 passes, but the ResNet
  wins on the same data, so more rollouts would help both. Retrain both on half the fit data: if π₀.₅'s gap to the
  ResNet **widens** at half data, data is what π₀.₅ lacks and more rollouts are worth collecting; if the gap stays
  flat, more data probably will not change the 3b verdict. Subset [ours]: per fit init state 7 or 8 of its 15
  repeats (alternating over sorted states, rng seed 0) = **187 / 375 episodes, all 25 states**, 21,368 / 43,087
  ResNet frames; success 0.61 (full fit 0.584), mean length 456 steps (458). Repeats rather than states are halved
  because LIBERO's init states are fixed, so more data would mean more repeats per layout. Inner / val unchanged.
  File `/iris/u/ankile/karenvo/libero/subsets/fit_half_s0.txt`; new flag `--fit-episodes FILE` in
  `scripts/pi05_value_expert.py` and `vl.libero_resnet` (+ `FIT` / `TAG` in `scripts/libero_resnet.sbatch`); subset
  ⊆ fit is asserted. Recipes unchanged: ResNet single as 3a (8,000 steps), π₀.₅ = arm B (w256 / MLP 1024, aug, lr
  1e-4, 5,400 steps, L40S), both early-stopped on inner CE, 3 seeds each. One dependency chain, so 1 GPU, starting
  after E s2 (18193281): ResNet s0 / s1 / s2 **18196780 / 18196781 / 18196782** → expert s0 / s1 / s2 **18196783 /
  18196784 / 18196785**. Out `runs/resnet18_half_s{0,1,2}_<job>`, `runs/value_expert_half_w256_s{0,1,2}_<job>`.
  Compare against the full-data rows: ResNet 0.128 ± 0.011 / +0.063, B 0.146 ± 0.003 / +0.027 (success MAE > 20 s /
  milestone). Suite 174 pass.
- 2026-10-30 **Arm D seed 0 (blank state), scored with `--eval-only`** (job 18195673, move3 A5000, 32 min; the
  path works end to end: 10 inner evals parsed from the cut-off log, best step 2500). Inner CE 3.564, val CE 3.587
  (3.535 at T 1.20), MAE 0.179, **success MAE > 20 s 0.153** (B s0 0.148, B 0.146 ± 0.003; ResNet 0.128), AUC 5 / 10
  / 20 / 30 s 0.78 / 0.78 / 0.81 / 0.87, **milestone excess +0.042** (B s0 +0.049). Reading [ours, one seed]: blanking
  the robot state changes nothing beyond seed noise, so proprioception is not the shortcut keeping π₀.₅ behind the
  ResNet; D is on course to fail the late-MAE half. s1 / s2 pending (training 18193278 / 18193280, eval 18195674 /
  18195675).
- 2026-10-30 **E complete (3 seeds) and D seed 1: both fail the 3b pass rule.** E s2 (18193281, L40S, best step
  3500): success MAE > 20 s 0.132, milestone −0.004. **E over 3 seeds:** success MAE > 20 s 0.135 ± 0.003 (ResNet
  0.128 ± 0.011), milestone −0.004 ± 0.020 (positive 1/3). D s1 (training 18193278 cut at the 4 h limit; scored with
  `--eval-only`, job 18195674, 9 inner evals, best step 2000): success MAE > 20 s 0.151, milestone **−0.025**. **D over
  s0–s1:** 0.152 ± 0.001, milestone +0.009 ± 0.048; negative on s1, so D fails whatever s2 gives. Reading [ours]:
  neither shortcut removal moves π₀.₅ past the ResNet. The clock buys ~0.01 of late MAE and costs the milestone;
  blanking the state changes nothing. The off-ramp now depends on the half-data check.
  Scheduling (Karen: use another A5000 if needed): half-data expert s1 moved off the serial L40S chain. 18196784
  was cancelled and resubmitted as **18199177** (A5000, 6 h limit, since A5000 runs at ~13 frames/s ≈ 4.6 h with no
  dependency); s2 18196785 now waits on s0 18196783. The chain ends ~19:00 instead of ~21:30.
- 2026-10-30 **Stage 4 value plots, B vs ResNet** (Karen: make them; `scripts/libero_b_vs_resnet_plots.py` →
  `reports/figures/libero_stage4_b_vs_resnet/`, 3 seeds each, no GPU). Episodes by a fixed rule set before looking:
  per kind, the median-MAE episode with MAE averaged over both models, plus B's and the ResNet's biggest wins.
  `overview.png` shows mean V by failure type; `aligned.png` shows V − V(milestone) ±10 s around the first pot on
  the stove. What they show [ours]:
  (1) **B's early edge is layout recognition.** At 0–5 s, B's mean V by val init state correlates 0.82 with the
  state's success rate (ResNet 0.68), and the early AUC is B 0.80 / ResNet 0.74. **Within a layout** (V minus the
  layout's mean, 108 episodes in mixed-outcome states), both models are at chance (B 0.53, ResNet 0.49). B's two
  biggest wins are failures from state 13 (8% success), where B is low from frame 0 and stays near the target. Its
  biggest loss is a rare success from state 42 (25%), where B stays pessimistic until the end. These are unseen
  states, so this is genuine generalisation, but it carries no within-episode signal, which is what Stage 5
  advantages use.
  (2) **Within episodes the ResNet reacts more sharply**, e.g. ep 542: +0.3 when pot 2 lands, then back down when
  the grasp of pot 1 fails, while B moves ~0.15. It overshoots in both directions (MAE 0.213 vs B 0.056 on that
  episode). Averaged over episodes (`aligned.png`), the two rise by similar amounts around the milestone; the
  ResNet's rise is steeper in the last ~2 s before contact, and B's starts slightly earlier.
  (3) In the overview, B separates success from failure types from t = 0, and the ResNet only from ~10–15 s.
  Reading [ours]: B is the better outcome predictor (from the layout); the ResNet is the more reactive progress
  tracker. The 3b pass rule measures the second on purpose. Gate pending Karen's look at the plots.
- 2026-10-30 **Stage 5 advantage comparison, B vs ResNet, val only** (Karen: run it). `vl.libero_advantages` with B
  (3-seed ensemble) and B s0 added as value functions (`ENSEMBLES` / `EV_COL` / `SOURCES`); report addendum in
  `reports/libero_advantages.md`, forest plot `reports/figures/libero_advantages/01_checks.png`. The old report and
  figures were backed up before the re-run, and the ResNet rows reproduce exactly. Main row (n = 2 s, 70% positive,
  mid-episode, time-matched): **B +2.4 points [−1.0, +6.1], AUC within scene 0.57; ResNet +5.4 [+2.3, +8.6], 0.74.**
  Paired scene bootstrap B − ResNet: −2.7 / −3.0 / −3.2 points at n = 1.2 / 2 / 4.8 s, all CIs below 0 (upper ends
  −0.5 / −0.1 / −0.4). B's labels agree with the ResNet's on 64% of interior frames. Reading [ours]: B's better
  overall MAE / early AUC does not carry over to advantages, because it comes from the layout and gives a flat
  within-episode V. The ResNet gives the better Stage 5 labels, which agrees with the 3b pass rule.
- 2026-10-30 **Head swap: B's weaker advantage labels come from its features, not its value head**
  (`scripts/libero_head_swap.py` → `reports/libero_head_swap.{json,val_predictions.parquet}`, CPU, 3 seeds per cell)
  [ours]. Question (Karen): do the head differences (B: linear h → 201 + a 2-way outcome head, loss summed; ResNet:
  1024 → 256 → ReLU → 201, value CE only) explain why B's Stage 5 labels are worse? First, the native predictions:
  within-episode change in V, time-matched, interior frames (≤ 34 s), signal = mean change successes − failures,
  noise = sd around it. At 2 s: **B signal +0.72 / noise 9.8 points, SNR 0.07; ResNet +1.70 / 6.3, SNR 0.27** (3 seeds
  each, tight); same ratio at 0.4 s and 4.8 s. So B has about half the signal and 1.6× the noise; it is not flat, it
  jitters. Then every cached feature set (`features/progress_18191428`: B s0's h, the 3a ResNet V's fine-tuned
  backbone, frozen ImageNet ResNet) × four heads (linear, linear + outcome, the ResNet's MLP, MLP + outcome), trained
  on fit, early-stopped on inner value CE. SNR at 2 s / milestone excess / success MAE > 20 s, mean over seeds:

  | features | linear | MLP (ResNet head) | MLP + outcome |
  | --- | --- | --- | --- |
  | B h | 0.091 / +0.046 / 0.152 | 0.082 / +0.039 / 0.150 | 0.086 / +0.037 / 0.151 |
  | ResNet V fine-tuned | 0.189 / +0.100 / 0.131 | 0.196 / +0.090 / 0.130 | 0.195 / +0.086 / 0.130 |
  | ResNet ImageNet frozen | 0.151 / +0.020 / 0.166 | 0.157 / +0.024 / 0.162 | 0.170 / +0.024 / 0.157 |

  Reading: (1) within any row the head changes nothing beyond seed noise (SNR ± 0.01), and the outcome loss on a shared
  hidden layer neither helps nor hurts; across rows the features move SNR 2×. (2) Even frozen ImageNet ResNet features
  give a higher SNR than B's h (0.15–0.17 vs 0.08–0.09; lower noise), so it is not only frozen vs fine-tuned. (3) The
  linear + outcome cell equals linear exactly: with fixed features and no shared layer the outcome gradient cannot reach
  the value head, so that cell tests nothing (MLP + outcome is the real test). Caveat: h comes from an expert trained
  with the outcome loss, so what that loss did *inside* the expert is untested here; that needs a GPU retrain of B
  without it. Next candidate [ours]: the same SNR on value heads over pi05's own per-layer V (layer 5 had the best
  pot-on-stove AUC, 0.987), to place the noise in the backbone vs the expert's readout
- 2026-10-30 **Arm D stopped; two follow-ups to the head swap** (Karen: run both; keep only the data runs otherwise).
  D s2 (18193280, at step ~2,450) and its eval (18195675) cancelled: D already fails (milestone −0.025 on s1), so s2
  cannot change the verdict. Queue now = the half-data check (ResNet half s0 18196780 → s1 18196781 → s2 18196782 →
  expert half s0 18196783 → s2 18196785; expert half s1 18199177 on an A5000) plus the two follow-ups:
  (1) **B without the outcome loss** (`--outcome-weight W` in `scripts/pi05_value_expert.py`, default 1.0 = every
  earlier run; 0 = value CE only, like the ResNet V; the outcome head is then untrained, so its P(success) is not
  read). Arm B recipe otherwise (w256 / MLP 1024, aug, lr 1e-4, 5,400 steps), L40S, seeds 0 / 1 / 2 chained:
  **18200394 → 18200395 → 18200396**, out `runs/value_expert_noout_w256_s{0,1,2}_<job>`. Tests whether the
  episode-constant outcome label makes the expert's own layers noisy, which the head swap could not test. Compare with
  B: change SNR 2 s 0.07 (native), milestone +0.027 ± 0.019, success MAE > 20 s 0.146 ± 0.003. Selftest passes, suite
  174 pass.
  (2) **Value heads straight on pi05's per-layer values** (`scripts/libero_layer_value.py`, CPU in the session
  because a CPU-only slurm job would not start before 2026-10-31 19:00): AttnProbe on layers 1 / 5 / 9 / 13 / 17 (33
  grid tokens), value CE only, 10 epochs, all layers at seed 0 first, then seeds 1 / 2 (~25 min per fit). Reading rule
  in the docstring, set before the result. Out `reports/libero_layer_value.{json,val_predictions.parquet}`.
  `scripts/libero_head_swap.py` gained `change_snr` (the SNR above), shared by both
- 2026-10-30 **Smoothing check: smoothing does not close B's gap** (Karen: run it; `scripts/libero_smoothing_check.py`
  → `reports/libero_smoothing.{md,json}`, CPU, minutes). V per val episode smoothed with a moving average over w grid
  frames (0.4 s each), centred or trailing, for each seed and the 3-seed ensembles of B and the ResNet. Scored with
  `change_snr` (`scripts/libero_head_swap.py`) and the Stage 5 time-matched label check (`vl.libero_advantages.checks`,
  n = 2 s, 70% positive). The raw rows reproduce the earlier numbers (SNR B 0.073 / ResNet 0.272; labels +2.4 / +5.4).

  | V, smoothing | change SNR 2 s (3 seeds) | signal / noise | Stage 5 label, time-matched [95% CI] | AUC |
  | --- | --- | --- | --- | --- |
  | B raw | 0.073 ± 0.009 | 0.007 / 0.098 | +2.4 [−1.0, +6.4] | 0.57 |
  | B centred ±1.2 s | 0.111 ± 0.016 | 0.007 / 0.063 | +4.3 [+0.6, +8.8] | 0.71 |
  | B centred ±2 s | 0.147 ± 0.027 | 0.007 / 0.048 | +6.0 [+2.1, +11.1] | 0.76 |
  | ResNet raw | 0.272 ± 0.023 | 0.017 / 0.063 | +5.4 [+2.2, +8.5] | 0.74 |
  | ResNet centred ±1.2 s | 0.394 ± 0.033 | 0.017 / 0.043 | +6.9 [+3.8, +10.8] | 0.81 |
  | ResNet centred ±2 s | 0.490 ± 0.042 | 0.016 / 0.033 | +9.0 [+5.4, +13.0] | 0.80 |

  Trailing (past-only) windows give the same pattern, slightly weaker (B ±2 s-equivalent 0.122, ResNet 0.428).
  Reading (pre-set rule) [ours]: (1) **B's deficit is half signal, half noise.** Smoothing removes noise for both
  models (B 0.098 → 0.048, ResNet 0.063 → 0.033) but cannot create signal, and B's signal is 2.4× smaller at every
  window (0.007 vs 0.017). Even at ±2 s, B's SNR (0.147) stays below the *unsmoothed* ResNet's (0.272). So by the
  pre-set rule the gap is not just fast jitter. (2) On labels, heavily smoothed B (+6.0) reaches the unsmoothed
  ResNet (+5.4), but the ResNet smoothed the same way stays ahead (+9.0). The fair comparison is at the same
  smoothing. Not formally tested as a paired difference; the CIs overlap, the SNR gap does not. Caveat: a ±2 s
  centred window around a 2 s lookahead lengthens the effective horizon and peeks ahead, which helps on its own
  (cf. n = 4.8 s: B +5.3, ResNet +8.6). (3) **Smoothing is worth keeping for Stage 5 whichever V is used**: ResNet
  ±1.2 s centred raises the label gap from +5.4 to +6.9 and the AUC from 0.74 to 0.81, with label flips 0.48 → 0.19 per
  s. Window choice [ours, to decide], and the peeking issue for labels needs a decision too (labels are offline, so
  centred is allowed, but trailing is the conservative choice).
  **(2) result: the noise is already in frozen pi05's features** (CPU, 36 min; best epoch 1–3 so each fit took
  ~2.4 min; the queued A5000 job 18200937 was cancelled unused). Value heads straight on the per-layer values, 3 seeds,
  mean ± sd:

  | layer | change SNR 2 s | signal / noise (points) | milestone excess | success MAE > 20 s | AUC 10 s |
  | --- | --- | --- | --- | --- | --- |
  | 1 | 0.104 ± 0.010 | 0.9 / 8.7 | −0.031 | 0.181 | 0.74 |
  | 5 | 0.102 ± 0.014 | 1.0 / 9.7 | −0.031 | 0.179 | 0.76 |
  | 9 | 0.104 ± 0.007 | 1.0 / 9.5 | −0.046 | 0.181 | 0.75 |
  | 13 | 0.076 ± 0.007 | 0.7 / 9.7 | −0.041 | 0.176 | 0.77 |
  | 17 | 0.106 ± 0.027 | 1.1 / 10.1 | −0.011 | 0.176 | 0.66 |

  For comparison (head swap): B's h 0.08–0.09, frozen ImageNet ResNet 0.15–0.17, fine-tuned ResNet 0.19–0.20. By the
  rule set before the result, **every layer stays near B's level, so the noise comes from the frozen backbone, not the
  expert's readout**; the milestone excess is negative on every layer and seed. The same features decode "pot on the
  stove" at time-matched AUC 0.96–0.99 (progress probe), so the information is there but a value head trained on
  per-frame CE does not pick it up from frozen pi05 features, while it does from frozen ImageNet ResNet features.
  Caveat: grid-4 pooled tokens (33), coarser than the 522 the expert reads; the expert on all tokens is no better
  (0.07). Revises the progress-probe entry's "LoRA / backbone fine-tuning is not the next step" [ours]: adapting the
  backbone is now the main remaining route for a pi05 V, pending the no-outcome retrain (18200394–96)
- 2026-10-30 **Oracle "pot on the stove" bit** (`scripts/libero_oracle_bit.py` → `reports/libero_oracle_bit.{json,
  val_predictions.parquet}`; run with `srun --overlap` on the TITAN RTX of half-data ResNet job 18196781). The true bit
  y[t] (progress-probe label) given to the value head next to the features, value CE only, 3 seeds:

  | arm | change SNR 2 s | milestone excess (seeds > 0) | success MAE > 20 s | AUC 10 s |
  | --- | --- | --- | --- | --- |
  | pi05 L5 | 0.119 ± 0.019 | −0.010 (1/3) | 0.177 | 0.77 |
  | pi05 L5 + bit | 0.098 ± 0.014 | −0.012 (2/3) | 0.182 | 0.77 |
  | bit + clock, no image | **0.222 ± 0.012** | **+0.325 (3/3)** | 0.169 | 0.50 |
  | B h / B h + bit | 0.082 / 0.084 | +0.039 / +0.045 | 0.150 / 0.151 | 0.79 |
  | ResNet ft / ResNet ft + bit | 0.189 / 0.206 | +0.088 / +0.092 | 0.132 / 0.131 | 0.63 / 0.62 |

  Reading [ours]: (1) the bit is useful for value: on its own with a clock it gives the cleanest within-episode V of
  any model (SNR 0.22, milestone +0.33 ≈ the oracle's +0.343), though it knows nothing of the layout (AUC 0.50).
  (2) **Next to image features the head ignores it**: pi05 L5, B's h and the ResNet all move by ≤ 0.02 SNR with the bit
  added. With rich features the per-frame CE is fitted from what identifies the scene and episode (best epoch 2), so a
  perfect, explicit progress cue is left unused. (3) Correction to the reading rule set before the result: it assumed
  the head would use the bit; "L5 + bit ≈ L5" therefore does not show the pot cue is unhelpful, it shows the training
  set-up does not make the head use progress information even when it is handed over. So the per-layer result
  ("noise is in the frozen backbone") is better read as: frozen pi05 features make the layout shortcut easy, and
  value-CE training takes it. Candidate fixes are in the objective, not only the backbone (e.g. a progress /
  within-episode auxiliary target, feature dropout, a bottleneck), alongside LoRA
- 2026-10-30 **Shortcut fixes on frozen pi05 L5: none works** (`scripts/libero_shortcut_fixes.py` →
  `reports/libero_shortcut_fixes.{json,val_predictions.parquet}`; this session's TITAN RTX, 34 min). Same input and
  head as the oracle's "pi05 L5", only the training changed, 3 seeds. Change SNR 2 s (per seed) / milestone excess
  (seeds > 0):

  | arm | SNR | milestone |
  | --- | --- | --- |
  | base | 0.109 ± 0.015 (0.121 / 0.092 / 0.115) | −0.024 (0/3) |
  | aux (predict the bit too) | 0.128 ± 0.015 (0.135 / 0.112 / 0.138) | −0.012 (1/3) |
  | aux → value | 0.129 ± 0.014 | −0.005 (1/3) |
  | token dropout 0.3 / 0.6 | 0.115 / 0.114 | −0.018 / −0.023 (1/3 each) |
  | bottleneck 16 / 4 | 0.105 / 0.118 | −0.039 (0/3) / +0.010 (2/3) |
  | bottleneck 4 + true bit (diagnostic) | 0.149 ± 0.021 | −0.006 (0/3) |

  By the rule set before the result (SNR > base + 2 sd and milestone positive 3/3), no arm works; all stay below the
  fine-tuned ResNet (0.19, +0.09). Small paired effect: the auxiliary bit target raises SNR on every seed (+0.014 /
  +0.020 / +0.023), but V still does not jump at placement. Even squeezed to 4 numbers and handed the true bit, the head
  barely uses it (milestone ≈ 0; the bit alone + clock gives +0.33). Reading [ours]: on 375 episodes from 25 layouts,
  per-frame value CE on frozen pi05 features is fitted from scene identity, and head-side regularisation does not
  change that. Remaining routes: history (third image slot / velocity), more layouts or data (half-data check
  pending), backbone adaptation (LoRA), or a target / head that forces V through progress (e.g. V from a predicted
  progress state + clock only, with layout as a small correction) [ours, to decide]
- 2026-10-30 **Progress-routed value head on frozen pi05 L5 (option 2)** (`scripts/libero_progress_routed.py` →
  `reports/libero_progress_routed.{json,val_predictions.parquet}`; this session's TITAN RTX, 20 min). The readout
  predicts the bit (BCE; val accuracy 0.95); the value MLP sees that prediction **detached**, plus the arm's inputs.
  3 seeds, mean ± sd (per-seed SNR):

  | arm | change SNR 2 s | milestone (seeds > 0) | success MAE > 20 s | MAE | AUC 10 s |
  | --- | --- | --- | --- | --- | --- |
  | **P + scene2 (no clock; the fair one)** | 0.168 ± 0.046 (0.147 / 0.136 / 0.220) | **+0.077 (3/3)** | 0.184 | 0.217 | 0.62 |
  | P + clock | 0.151 ± 0.007 | +0.270 (3/3) | 0.197 | 0.245 | 0.47 |
  | P + clock + scene2 | 0.130 ± 0.033 | +0.035 (3/3) | **0.123** | **0.167** | 0.75 |
  | ResNet ft + clock (control) | 0.134 ± 0.005 | +0.063 (3/3) | 0.115 | 0.180 | 0.61 |
  | true bit + clock (ceiling) | 0.222 ± 0.012 | +0.325 (3/3) | 0.169 | 0.231 | 0.50 |

  Reading [ours]: (1) **first single-frame frozen-pi05 head whose V jumps at placement on every seed with a
  ResNet-size effect** (P + scene2 +0.077 vs the ResNet V's +0.063 native / +0.088 in the head swap), and its SNR is
  up from 0.11 to 0.17; by the rule set before the result it passes the milestone half but not the SNR bar (0.19,
  reached on 1/3 seeds). So the progress information in frozen pi05 is usable once value CE cannot route around it.
  (2) The price is everything the shortcut was good for: AUC 10 s 0.77 → 0.62 (the layout edge is gone), success MAE
  > 20 s 0.184 (ResNet 0.128). (3) Giving the head the clock and a 2-d scene path brings the shortcut back (milestone
  +0.035) but gives the best MAE of any model so far (0.167; success MAE > 20 s 0.123, below the ResNet's 0.128). The
  clock arms beat ResNet ft + clock on milestone (+0.27 vs +0.06). Next candidate [ours, to decide]: keep the scene
  path but make it carry layout only, e.g. a training-time penalty on its change within an episode (inference stays
  single-frame), so outcome knowledge comes back without the shortcut
- 2026-10-30 **Layout-only scene code** (`scripts/libero_layout_code.py` → `reports/libero_layout_code.{json,
  val_predictions.parquet}`; this session's TITAN RTX). Option 2's routed head with a training-time penalty on the
  scene code changing within an episode (within / total variance, scale-free; batches 32 episodes × 8 frames). The
  penalty works mechanically: within-episode share of the code's variance on val 0.97 (λ 0) → 0.26 (λ 1) → 0.13 (λ 10).
  3 seeds, no clock:

  | arm | change SNR 2 s | milestone (seeds > 0) | AUC 5 / 10 s | success MAE > 20 s |
  | --- | --- | --- | --- | --- |
  | scene2, λ 0 | 0.152 ± 0.043 | +0.069 (3/3) | 0.66 / 0.68 | 0.160 |
  | scene2, λ 1 | 0.147 ± 0.018 | +0.130 (3/3) | 0.64 / 0.55 | 0.167 |
  | scene2, λ 10 | 0.166 ± 0.035 | **+0.148 (3/3)** | 0.65 / 0.57 | 0.188 |
  | scene8, λ 10 | 0.126 ± 0.028 | +0.035 (3/3) | **0.80 / 0.74** | 0.150 |

  References: ResNet V SNR 0.19–0.27, milestone +0.06–0.09, AUC 10 s 0.63, success MAE > 20 s 0.128; B 0.07 / +0.027 /
  0.78 / 0.146. Reading [ours]: by the rule set before the result, not a clean win: a 2-d layout code makes V react
  ~2× the ResNet at placement but the code is not used for outcome (AUC falls); 8 dims bring the layout edge back
  (AUC 5 s 0.80, B-level) but the jump shrinks to +0.035 and SNR to 0.13. It is a trade-off between knowing the
  scene and reacting to progress, set by how much scene information the head gets, and no setting beats the ResNet
  on both. Late-success accuracy stays worse than the ResNet in every arm (0.150–0.188 vs 0.128)
- 2026-10-30 **Fresh-layout data, option A** (Karen: go; start with a 100-episode success check). Reason [ours]: with 30
  fixed train layouts the value heads can fit layout identity → outcome (oracle / shortcut / layout-code entries); a new
  layout per episode removes that. `scripts/libero_rollouts.py --args.fresh-layouts N`: N episodes per task, each from
  the layout `env.reset()` samples from the task's BDDL placement regions (as the 50 saved init states were drawn)
  instead of `set_init_state`; CSV init_state −1, raw `sim_init` = the sampled state, so replay is unchanged.
  `scripts/libero_eval.sbatch` gained `CLIENT_ARGS` (extra client arguments). Check of the sampler (moka, 6 resets):
  6/6 distinct layouts, every pot position inside the range of the 50 saved states, 2.4–3.9 cm from the nearest saved
  layout. Val / test stay on the 10 + 10 fixed held-out states, so all numbers remain comparable. Check run: moka × 100
  fresh layouts, env seed 8, `--save-raw`, A5000, job **18202119** (after 18199177, to stay within 3 GPUs). Go on to
  ~450 train episodes only if success lands in 50–80%
- 2026-10-31 **Half-data check: result** (jobs 18196780–82 ResNet, 18196783 / 18199177 / 18196785 arm B; scored with
  `scripts/libero_seed_table.py`, full-data rows reproduce the reference values exactly). 187 / 375 fit episodes, all
  25 states, 3 seeds, mean ± sd:

  | arm | inner CE | success MAE > 20 s | milestone (seeds > 0) | AUC 10 s | best step |
  | --- | --- | --- | --- | --- | --- |
  | ResNet full | 3.530 ± 0.016 | 0.128 ± 0.011 | +0.063 ± 0.023 (3/3) | 0.62 | 1.5–2k |
  | ResNet half | 3.561 ± 0.015 | 0.165 ± 0.014 | +0.049 ± 0.013 (3/3) | 0.61 | 1.5k × 3 |
  | B full | 3.531 ± 0.037 | 0.146 ± 0.003 | +0.027 ± 0.019 (3/3) | 0.78 | |
  | B half | 3.648 ± 0.026 | 0.178 ± 0.024 | −0.010 ± 0.029 (1/3) | 0.76 | 1.5k / 1.5k / 1k |

  Reading by the rule set before the result [ours]: **the gap does not widen on the main metric.** B − ResNet on
  success MAE > 20 s is +0.018 at full data and +0.013 at half; halving costs both about the same (ResNet +0.037, B
  +0.032). Weak hints the other way, within seed noise: the milestone gap grows 0.036 → 0.059 (sd ~0.03; B half
  negative on 2/3 seeds), and B loses ~4× more inner CE (+0.117 vs +0.031). So more repeats of the same 25 layouts
  help both models equally but are unlikely to close π₀.₅'s gap; this points to new layouts (option A), not more
  repeats.
- 2026-10-31 **Fresh-layout check: result** (job 18202119, move3 A5000, 51 min): moka × 100 fresh layouts, **55/100 =
  0.55** success (fixed-layout fit 0.584), inside the 50–80% go band. No errors; all 45 failures are 520-step
  timeouts; raw `sim_init` + video for every episode (4.2 GB raw). Out
  `/iris/u/ankile/karenvo/libero/runs/freshcheck_libero_10_18202119/`.
- 2026-10-31 **Fresh-layout train collection launched** (Karen: go). moka × 450 fresh layouts, **env seed 9** (seed 8
  would replay the check's layouts; the policy noise is the server's `jax.random.key(0)` stream as in every run),
  `--save-raw`, A5000, job **18212530** (~4 h at the check's rate, ~19 GB raw): `SUITE=libero_10 TASKS=8 SEED=9
  NAME=fresh450 CLIENT_ARGS="--args.fresh-layouts 450 --args.save-raw" sbatch --gres=gpu:a5000:1
  scripts/libero_eval.sbatch`. Size [ours]: 450 = the current fixed-layout train pool (30 states × 15 = 375 fit + 75
  inner), so fresh-only vs fixed-only is a matched comparison of layout diversity at equal size; fixed + fresh (900)
  is the more-data arm. For reference, the Mulligan marker V trains on 1,373 episodes / 6.9 h at 15 fps (~18 s each);
  LIBERO's 450 train episodes are ~5.7 h at 10 Hz (~46 s each), so LIBERO already has about Mulligan's hours with a
  third of its episodes. Val / test unchanged (10 + 10 fixed held-out states). To check on the first episodes:
  layouts differ from the seed-8 check's.
- 2026-10-31 **B without the outcome loss: result** (jobs 18200394–96, L40S, `--outcome-weight 0`; scored with
  `score()` of `scripts/libero_head_swap.py` via `scripts/libero_noout_score.py`, whose B / ResNet rows reproduce the
  logged SNR 0.073 / 0.272). 3 seeds, mean ± sd; half-data rows added for SNR:

  | arm | change SNR 2 s (signal / noise) | milestone (seeds > 0) | success MAE > 20 s | MAE | AUC 5 / 10 s |
  | --- | --- | --- | --- | --- | --- |
  | ResNet full | 0.272 ± 0.023 (0.017 / 0.063) | +0.063 (3/3) | 0.128 | 0.211 | 0.73 / 0.62 |
  | ResNet half | 0.255 ± 0.019 (0.016 / 0.063) | +0.049 (3/3) | 0.165 | 0.217 | 0.69 / 0.61 |
  | B full | 0.073 ± 0.009 (0.007 / 0.098) | +0.027 (3/3) | 0.146 | 0.180 | 0.77 / 0.78 |
  | B half | 0.065 ± 0.010 (0.007 / 0.106) | −0.010 (1/3) | 0.178 | 0.198 | 0.75 / 0.76 |
  | **B no outcome loss** | **0.073 ± 0.025** (0.007 / 0.093) | +0.028 (3/3) | 0.162 ± 0.009 | 0.185 | 0.79 / 0.78 |

  Reading [ours]: **the outcome loss is not what makes B noisy.** Without it SNR is unchanged (0.073, per seed 0.105 /
  0.068 / 0.045), noise drops only 0.098 → 0.093 and the milestone is the same; the one clear change is worse
  late-success accuracy (0.146 → 0.162, ~2 sd), so the outcome head helps a little and stays in. The noise in B's V comes
  from the expert / features themselves, not from the episode-constant label. Half data also leaves the SNR ratio flat
  (B / ResNet 0.27 → 0.25), consistent with the half-data reading above.
- 2026-10-31 **Value objective on frozen pi05 L5: bootstrapped targets lift SNR to the ResNet's level, milestone stays ~0**
  (`scripts/libero_value_objective.py` → `reports/libero_value_objective{,_selmae}.{json,val_predictions.parquet}`;
  this session's TITAN RTX, ~10 min per run). Motivation [ours]: in arm B's val probabilities (`val_probs.npy`, 3 seeds
  + the no-outcome retrain) **97–99% of the variance of the 2 s change in E[V] is P(success) moving between the success
  and failure humps**, so V's noise is a flickering per-frame outcome classifier. Hypothesis: within a layout the
  outcome is mostly policy noise, so per-frame Monte Carlo labels are near-random episode draws that rich features
  memorise. Same input / head / scoring as `libero_shortcut_fixes.py` "base"; only the target or loss changes. Arms:
  consistency penalty on E[V](t + 2 s) − E[V](t) − 20/1040 (λ 10 / 100); `oof` = two-hot of an out-of-fold V
  (5 folds over the 25 fit layouts); `td n` = two-hot of −4n/1040 + V_oof(t + n) (Monte Carlo past the episode end).
  Reading rule in the docstring, set before the result. Base reproduces `libero_shortcut_fixes` per seed (0.121 / 0.092).
  **Selection confound found in the first run:** under the shared rule (inner Monte Carlo value CE) every soft-target
  arm stopped at epoch 0 (inner CE 5.4–7.9: unimodal predictions on bimodal one-hot labels), so the run was repeated
  with `--select mae` (inner MAE of E[V], same rule for every arm and for the V_oof folds). 3 seeds, mean ± sd:

  | arm | SNR 2 s, CE sel. | SNR 2 s, MAE sel. | same, smoothed ±1.2 s (CE / MAE) | milestone (seeds > 0), CE / MAE | success MAE > 20 s, CE / MAE | AUC 10 s (MAE) | hump share |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | base | 0.109 ± 0.015 | 0.100 ± 0.026 | 0.160 / 0.164 | −0.024 (0/3) / +0.007 (2/3) | 0.184 / 0.160 | 0.79 | 1.0 |
  | consist 10 | 0.121 ± 0.009 | 0.116 ± 0.022 | | −0.005 (1/3) / +0.018 (3/3) | 0.181 / 0.164 | 0.80 | 1.0 |
  | consist 100 | 0.106 ± 0.040 | 0.079 ± 0.023 | | +0.011 (3/3) / +0.010 (3/3) | 0.219 / 0.184 | 0.76 | 1.2 |
  | oof | 0.159 ± 0.027 | 0.152 ± 0.018 | 0.231 / 0.229 | −0.012 (1/3) / −0.011 (1/3) | 0.188 / 0.160 | 0.62 | 0.3 |
  | td 2 s | 0.168 ± 0.013 | 0.153 ± 0.020 | | −0.016 (0/3) / −0.014 (0/3) | 0.199 / 0.163 | 0.65 | 0.35 |
  | **td 10 s** | **0.198 ± 0.004** | 0.162 ± 0.025 | **0.290 / 0.255** | −0.009 (1/3) / +0.008 (3/3) | 0.183 / **0.155** | 0.56 | 0.5 |

  References: ResNet V raw 0.272, smoothed ±1.2 s 0.394, milestone +0.063, success MAE > 20 s 0.128; progress-routed
  head 0.168 / +0.077. Reading [ours]: (1) **the consistency penalty is only smoothing** (noise and signal fall
  together; λ 100 flattens V); fails. (2) **Bootstrapped targets work on the noise side**: td 10 s removes half the hump
  flicker (share 1.0 → 0.5) and at matched smoothing beats base by 0.09–0.13 SNR (0.26–0.29 vs 0.16), reaching the raw
  ResNet; it passes the SNR half of the rule (CE sel. clearly, MAE sel. at ~2 sd). (3) **It does not create a jump at
  placement**: milestone ≈ 0 under both selections (+0.008 on 3/3 with MAE selection is ~1/8 of the ResNet's), so by
  the rule it does not work. Expected in hindsight: the bootstrap source V_oof is itself a Monte Carlo head on the same
  features with no placement jump, and TD can only propagate what its source knows. (4) The early layout edge is lost
  (AUC 10 s 0.79 → 0.56–0.65), as with the progress-routed head; unexplained, since V_oof does generalise across
  layouts. (5) **Selection matters for base too**: MAE selection trains to epoch 7 instead of 1 and lowers base's
  success MAE > 20 s from 0.184 to 0.160, so inner-CE early stopping (every 3b run) favours a timing / layout fit.
  Next candidate [ours, to decide]: bootstrap from a source that does react to placement (the progress-routed head)
  instead of the Monte Carlo V_oof, i.e. combine option 2's routing with td 10 s targets; and re-select 3b runs on
  inner MAE
- 2026-10-31 **Next 3b arm: LoRA on the pi05 backbone (Karen: go)** (`scripts/pi05_value_lora.py`,
  `scripts/pi05_value_lora.sbatch`). Rationale: (1) everything on *frozen* pi05 features fails the same way (head
  swap, per-layer heads, shortcut fixes, objective changes), while the ResNet V it is compared with fine-tunes its
  backbone (lr 1e-5); on Mulligan frozen vs fine-tuned ResNet was MAE 0.234 vs 0.121, and on LIBERO frozen ImageNet
  features give SNR 0.15–0.17 vs fine-tuned 0.19–0.20. So 3b so far compares a fine-tuned ResNet with a frozen pi05.
  (2) RECAP / π\*0.6 (arXiv 2511.14759) trains its value function as a VLM "initialized from" a pretrained VLM
  (§V, §V-C: 670M, Gemma 3) with Monte Carlo 201-bin CE (§IV-A) and "finetune[s]" it each iteration (§V-D); it never
  describes a frozen backbone, and it trains V without TD (TD enters only through the n-step advantage, §III), so
  the td arms of the value-objective entry are off-recipe and are dropped. Design = arm B unchanged (width 256 /
  MLP 1024 expert, one query token, value + outcome CE, openpi augmentation, 5,400 steps, batch 32, expert lr 1e-4)
  plus openpi's `gemma_2b_lora` adapters (rank 16, attention + FFN, every layer) trained at lr 1e-5. [ours: pi05's
  own VLM rather than a separate smaller one as in RECAP; LoRA rather than full fine-tuning; SigLIP frozen; lora_b
  zero-initialised so step 0 is exactly B's backbone; LoRA lr = expert lr / 10 as the ResNet's backbone / head ratio.]
  Selection: inner CE and inner MAE recorded at every eval, trainable weights saved at every eval, val predictions
  for both the best-CE and the best-MAE step. Comparison caveats: the ResNet V has no augmentation (B does) and
  pi05 is ~100× larger, so a win reads "fine-tuned pi05 is the better value network", not "better features at equal
  budget". Same pass rule as 3b (beat the ResNet's success MAE > 20 s 0.128 and milestone > 0 on every seed), plus
  change SNR vs the ResNet's 0.272.
- 2026-10-31 **Fresh-layout collection done; fresh-layout arms submitted** (Karen: all 12 jobs, at most 3 GPUs, run
  sequentially, A5000 or L40S). Collection 18212530 COMPLETED (move3 A5000, 3 h 48 min): 450 / 450 episodes, no errors.
  Data [ours]: `scripts/libero_fresh_table.py` writes one episode table for fixed + fresh
  (`/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/`): the 570 Stage 2 episodes with their ids, `raw` made
  absolute; the fresh episodes as split train, init_state −1, ids 1000+; targets via `vl.libero_targets` (asserted equal
  to Stage 2's on Stage 2 episodes); fit lists `fit_fixed` (375, = every earlier full-data run), `fit_fresh375_s0`
  (375 of the 450, rng 0: matched in size), `fit_fixed_fresh` (375 + 450 = 825). **Inner (5 fixed states) and val (10
  fixed states) unchanged for every arm**; fresh episodes are never held out. No LeRobot conversion (the trainers read
  raw files). Code: `--dataset` in `scripts/pi05_value_expert.py` (absolute `raw` accepted, `inner_states` ignores
  init_state −1); `vl.libero_resnet --every 4 --dataset T` builds frames from the table (`table_meta`; on Stage 2 it
  equals the features-file frames exactly, row order included) and `--pack-only`; `vl.libero_probe.inner_split`
  ignores −1 (unchanged on Stage 2); ResNet results.json now records `setup`; sbatch options `PREP_FRESH_RUN`,
  `WAIT_FOR`, `EVERY` / `DATASET`. +2 tests (`tests/test_libero_fresh.py`), 176 pass. Checks: dry run of the table on
  the 100-episode check run; packed frames == raw for fixed and fresh episodes; expert data loading on fresh episodes
  (smoke; the training-loop smoke was killed by the session's 16 GB cap, so the first real expert job is the smoke).
  Recipes unchanged. `scripts/libero_fresh_arms.sh` → two chains (another session holds a third GPU), each job on an
  A5000 or L40S (`--gres=gpu:1 --exclude=move1,move2,move5`):
  chain 1: ResNet fresh s0 (+ table + frame cache) **18219696** → s1 18219697 → s2 18219698 → expert fresh s0 18219699 →
  s1 18219700 → s2 18219701; chain 2 (starts with 18219696): expert fixed+fresh s0 18219702 → s1 18219703 → s2
  18219704 (96 GB, 8 h) → ResNet fixed+fresh s0 18219705 → s1 18219706 → s2 18219707. Out `runs/resnet18_{fresh,
  fixedfresh}_s{0,1,2}_<job>`, `runs/value_expert_{fresh,fixedfresh}_w256_s{0,1,2}_<job>`. Compare against fixed-only
  (ResNet 0.128 / +0.063 / SNR 0.27; B 0.146 / +0.027 / 0.073). ~18–20 h wall if both chains get GPUs promptly.
- 2026-10-31 **Cut to 6 jobs: fresh-only first; fixed + fresh deferred** (Karen). Chain 2 (18219702–07) cancelled
  before it started. The 6 fresh-only jobs now run as two chains (2 GPUs; the other session's job is the 3rd): ResNet
  **18219696** (+ table + frame cache) → 18219697 → 18219698, may also use a TITAN RTX (40 GB; the half-data ResNet ran
  ~1 h per seed on move2's TITANs); expert **18219699** (starts with 18219696, waits for the table) → 18219700 →
  18219701 on an A5000 / L40S. Why TITAN for the ResNet only [ours]: Turing has no native bf16 and the expert has never
  been timed there (A5000 already takes ~4.6 h), so it stays on A5000 / L40S. At 18:30 all 6 were pending: move3's
  A5000s had GPUs free but 0 of 96 CPUs, and higher-priority jobs were ahead on A5000 / L40S / TITAN.
  **Next (to resume, in order):**
  1. When the 6 finish: score with `scripts/libero_noout_score.py`-style rows (change SNR, milestone, success MAE
     > 20 s, AUC) for `resnet18_fresh_s{0,1,2}_*` and `value_expert_fresh_w256_s{0,1,2}_*` against fixed-only (ResNet
     0.128 / +0.063 / SNR 0.27; B 0.146 / +0.027 / 0.073). Question: does π₀.₅'s gap to the ResNet shrink when the
     layouts are fresh (no layout → outcome shortcut)?
  2. **Deferred fixed + fresh arm (6 jobs, 825 fit episodes):**
     `scripts/libero_fresh_arms.sh /iris/u/ankile/karenvo/libero/runs/fresh450_libero_10_18212530 fixedfresh`
     (ResNet ×3 chain + expert ×3 chain, expert 96 GB / 8 h; needs the table and frame cache that 18219696 builds).
     Run it if fresh-only helps π₀.₅, or to test whether more data on top of fresh layouts adds more.
  3. Spare data: the 100 fresh episodes of the check run (18202119, env seed 8, layouts distinct from seed 9) are not
     in the table; add them if a bigger fresh arm is wanted.
- 2026-10-31 **Table-job failure and fix.** 18219696 started on a move1 TITAN at 16:21 and failed in 2 s on the table
  builder's check `split == "fresh"`: fresh450's `split` column is empty (no `--args.split-name` at launch; the check
  run had "fresh"). Data fine (450 × init_state −1, env seed 9, 0 errors, success 0.549). The check now accepts empty
  or "fresh". Its afterany / after dependants were held before they could start without a table. Table built directly
  (1 min, no GPU): `/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/` (1,020 episodes, 471,105 target frames,
  Stage 2 targets reproduced exactly; fresh375 success 0.555, fresh 450 0.549, fixed fit 0.598 incl. inner) + DONE.
  Current jobs: ResNet fresh s0 **18220907** (packs the frame cache) → s1 18219697 → s2 18219698; expert fresh s0
  **18219699** → s1 18219700 → s2 18219701.
  **Checks + launch (same day).** Selftest (job 18219453, A5000): with lora_b = 0 the LoRA backbone's K/V equal frozen
  pi05 exactly (max |diff| 0); a non-zero LoRA moves K; gradients reach LoRA (28.0 M params) and the expert. Check C
  (18220055, batch 8: batch 32 runs out of memory on 24 GB): CE 6.19 → 0.028 by step 700, MAE 0.0037 (half-bin bar
  0.0025; the run's "FAILS" read the step-700 log, not step 1000, a logging bug since fixed), so the gradient path
  trains. Smoke (18221031): train / eval / checkpoints / both selections end to end. Full runs [ours: `--accum 4`,
  micro-batch 8, effective batch 32 as B]: ~4.4 frames/s on an A5000, ~11–12 h per seed; seed 1 = job 18221358,
  seed 0 = 18221385, seed 2 = 18221386 (after seed 0; partition limit 24 h). Out
  `/iris/u/ankile/karenvo/libero/runs/value_lora_r16_s<k>_<job>/`.
- 2026-11-01 **Fresh-layout and LoRA arms: interim result (2 of 3 seeds for B fresh and LoRA; ResNet fresh 3/3)**.
  Scored with `score()` of `scripts/libero_head_swap.py` as in the no-outcome entry; the reference rows reproduce
  exactly (ResNet 0.272 / 0.128, B 0.073 / 0.146). Still running: B fresh s2 18219701, LoRA s2 18221386; re-score
  when they finish. Val = the same 10 fixed held-out states for every arm. Mean ± sd:

  | arm | change SNR 2 s | milestone (seeds > 0) | success MAE > 20 s | MAE | AUC 5 / 10 s | best step |
  | --- | --- | --- | --- | --- | --- | --- |
  | ResNet fixed (ref) | 0.272 ± 0.023 | +0.063 (3/3) | **0.128 ± 0.009** | 0.211 | 0.73 / 0.62 | 1.5–2k |
  | B fixed (ref) | 0.073 ± 0.009 | +0.027 (3/3) | 0.146 ± 0.003 | 0.180 | 0.77 / 0.78 | 2–2.5k |
  | ResNet fresh | 0.258 ± 0.037 | **+0.085 ± 0.005 (3/3)** | 0.148 ± 0.015 | 0.215 | 0.59 / 0.64 | 2–2.5k |
  | **B fresh** | **0.172 ± 0.006** | +0.051 ± 0.011 (2/2) | 0.151 ± 0.013 | 0.188 | 0.57 / 0.80 | 2–2.5k |
  | LoRA r16, inner-CE sel. | 0.096 ± 0.001 | +0.011 ± 0.041 (1/2) | 0.158 ± 0.013 | 0.183 | 0.78 / 0.77 | 2–2.5k |
  | LoRA r16, inner-MAE sel. | 0.055 ± 0.003 | +0.023 ± 0.009 (2/2) | 0.144 ± 0.003 | **0.173** | 0.79 / 0.80 | 5–5.4k |

  Reading [ours]: (1) **Fresh layouts are the first change that clearly helps frozen π₀.₅**: SNR 0.073 → 0.172
  (2.4×, sd 0.006) and the milestone jump ~doubles (+0.027 → +0.051), while the ResNet's SNR is unchanged. (2) On the
  main metric the gap closes (0.151 vs 0.148) **because the ResNet gets worse** (0.128 → 0.148), not because B gets
  better (0.146 → 0.151): part of the ResNet's fixed-layout lead was the layout → outcome fit, as is its early AUC
  (0.73 → 0.59). By the 3b rule (beat 0.128 and milestone > 0 on every seed) B fresh does not pass, and the ResNet still
  leads on SNR and milestone. (3) **LoRA r16 as run changes almost nothing** vs frozen B (fixed layouts). Check of the
  trained adapters (seed 0, `ckpt_{500,2500,5400}.msgpack`): rms lora_b 0.00016 / 0.00067 / 0.00080 (max 0.0065), lora_a
  stays at its init 0.010, so the weight update ΔW = a·b is ~√16 · 0.010 · 0.0008 ≈ 3e-5 per element, ~0.1% of
  typical Gemma weight scale (estimate; base rms not measured). With lora_b zero-initialised and lr 1e-5 the backbone
  barely moved, most of all at the CE-selected step 2–2.5k. So this run is close to frozen B and is **not a test of
  fine-tuning π₀.₅**; it does not refute the RECAP-style arm.
- 2026-11-01 **RECAP / π\*0.6 value-function recipe, rechecked against our LoRA arm** (arXiv 2511.14759, HTML).
  What the paper states: V "uses the same architecture as the VLA policy, but with a smaller VLM backbone" (§IV-A);
  "a smaller 670M parameter VLM backbone that is also initialized from Gemma 3" (§V-C, Fig. 3: "a separate value
  function initialized from a pre-trained but smaller VLM"); same language inputs as the VLA (§V-C); observations =
  images + robot configuration (§V-A); 201 bins, CE on empirical returns (§IV-A, Eq. 1); "co-train ... on a small
  mixture of multi-modal web data" against overfitting (§V-C); **"During pre-training, we first train the value
  function on the same dataset"** (the multi-task demonstration data, steps-to-success targets), then "finetune the
  value function on all of the data collected for the task so far", and "both the value function and policy are
  finetuned from the pre-trained checkpoint" each iteration (§V-D, Alg. 1 lines 4, 8). It never says which parts are
  frozen; "finetune" with no freezing stated reads as the whole VLM [ours: inference, not stated]. The KI stop-gradient
  (§V-A) is for the VLA's action expert, not V. Readout token / head, optimiser, lr, steps, selection: not given.
  Differences in our LoRA arm, in order of how much they plausibly matter [ours]:
  1. **Amount of adaptation**: RECAP fine-tunes the VLM; ours trains 28 M rank-16 adapters at lr 1e-5 from lora_b = 0,
     which moved the weights by ~0.1% (above). Our lr choice copied the ResNet's full-fine-tune backbone lr; LoRA
     usually needs ~10× more (LoRA lr ~1e-4, i.e. = the expert lr), since lora_b starts at 0.
  2. **No value pre-training stage**: RECAP first trains V on the large multi-task demo set (steps-to-success), then
     fine-tunes on task rollouts. Ours goes straight from the `pi05_libero` *policy* checkpoint to 375 moka episodes.
     Closest LIBERO analogue: pre-train V on the LIBERO demonstrations of all suites (all successes, so the
     steps-to-success target is defined; the outcome head gets no signal there), then fine-tune on moka rollouts.
  3. **Training data**: RECAP's per-task fine-tune uses all data collected so far, with diverse real-world states;
     ours used the 30 fixed layouts. The fresh-layout result says this matters for π₀.₅.
  4. **Backbone identity**: RECAP's V is a separate, smaller VLM (Gemma 3, 670M) read out by its own head; ours reuses the
     policy's 2B PaliGemma (trained for LIBERO actions) and reads it with a cross-attending expert. Not something to
     copy for the "is π₀.₅ a better value network" question; a separate VLM would be a different arm.
  5. SigLIP frozen in ours (openpi's LoRA default); not stated in the paper.
  6. No web-data co-training in ours. Not practical here; our substitute is low lr + early stopping.
  Same as ours: Monte Carlo 201-bin CE, (−1, 0) normalisation, single frame with no time input, language + state
  inputs through the VLA's input transform.
- 2026-11-01 **Proposed order of next runs [ours, to decide; ≤ 3 GPUs]:**
  1. Let B fresh s2 / LoRA s2 finish and re-score (no extra cost). Cancelling LoRA s2 (~8 h left) is an option, since
     the arm is near-frozen by construction.
  2. **LoRA that actually adapts, on fresh layouts** (combines the one lever that worked with the RECAP difference
     that was not really tested). LoRA lr 1e-4 (= expert lr), everything else as B fresh. First a cheap check that
     the backbone moves (rms lora_b and the K/V change after ~1k steps vs this run's 0.0008) and, as Check C, still
     trains; then 1 seed (~12 h) as go / no-go before seeds 1–2. Running LoRA on fresh layouts at the old lr 1e-5
     would only re-measure B fresh.
  3. **Fixed + fresh (825 episodes)** for B and the ResNet (`scripts/libero_fresh_arms.sh ... fixedfresh`): tests
     whether more data on top of layout diversity keeps helping π₀.₅; can run beside item 2. If item 2 wins, its
     second-round data arm should be LoRA + fixed + fresh.
  4. **Value pre-training on LIBERO demos, then moka fine-tune** (RECAP's two-stage recipe): only if item 2 shows
     adaptation helps; it needs a demo table + steps-to-success targets (new code).
  Not now: full fine-tuning of the 2B LLM (does not fit our 24–48 GB GPUs with this batch without more engineering),
  a separate 670M Gemma 3 V (a different question), web-data co-training.
- 2026-11-01 **Launched items 2 + 3; LoRA s2 cancelled** (Karen: go; up to 4 GPUs). B fresh s2 (18219701) finished
  first, so B fresh now has 3 seeds (re-score pending). LoRA r16 s2 (18221386) cancelled at 4 h 40 min.
  `scripts/pi05_value_lora.py` gained `--dataset` / `--fit-episodes` (as `pi05_value_expert.py`; sets that module's
  `DATASET` too, since `Frames` reads it) and logs rms lora_b at every eval (also in `history`); 176 tests pass.
  **Item 2, LoRA lr 1e-4 on fresh layouts** (everything else as the r16 run: rank 16, accum 4 / micro-batch 8,
  augmentation, 5,400 steps, both selections): smoke **18237893** (2 fresh fit episodes, 20 steps) → seed 0
  **18237894** (afterok), out `runs/value_lora_fresh_lr1e4_s0_<job>/`. Go / no-go on the weights moving: compare
  rms lora_b at step 500 against the lr 1e-5 run's 1.6e-4 (2,500: 6.7e-4); ~10× expected. Seeds 1–2 only if seed 0
  beats B fresh. **Item 3, fixed + fresh (825 fit episodes)** via `scripts/libero_fresh_arms.sh ... fixedfresh`:
  ResNet 18237895 → 18237896 → 18237897; expert 18237898 → 18237899 → 18237900 (96 GB, 8 h each). With the other
  session's TITAN job that is 4 GPUs at most.
- 2026-11-01 **B fresh, 3 seeds (final; s2 = 18219701, best step 2,500)**: change SNR **0.157 ± 0.022** (per seed
  0.166 / 0.178 / 0.126), milestone +0.044 ± 0.013 (3/3 > 0), success MAE > 20 s 0.148 ± 0.012, MAE 0.187, AUC 5 /
  10 s 0.60 / 0.80. Same reading as the 2-seed entry, slightly weaker: SNR still ~2.2× fixed-layout B (0.073) but
  below the ResNet's 0.26–0.27; late-success MAE now equal to ResNet fresh (0.148 vs 0.148); does not pass 3b.
- 2026-11-01 **Check-in: LoRA lr 1e-4 moves the backbone (go); ResNet fixed + fresh, 2 of 3 seeds.**
  LoRA fresh lr 1e-4 seed 0 (18237894, smoke 18237893 passed): rms lora_b **1.84e-3 at step 500** (lr 1e-5 run:
  1.6e-4, **11.5×**) and 3.46e-3 at step 1,000 (≈ 5× the old run's step 2,500). Inner CE / MAE at 500: 3.834 / 0.235
  (old 3.921 / 0.231); at 1,000: 3.562 / 0.202 (old 3.609 / 0.199). Not a like-for-like comparison (fit = fresh 375
  vs fixed 375). ResNet fixed + fresh (825 fit episodes; s0 18237895, s1 18237896; s2 18237897 pending on priority):

  | arm | change SNR 2 s | milestone | success MAE > 20 s | MAE | AUC 5 / 10 s | best step |
  | --- | --- | --- | --- | --- | --- | --- |
  | ResNet fixed (ref) | 0.272 ± 0.023 | +0.063 | 0.128 ± 0.009 | 0.211 | 0.73 / 0.62 | 1.5–2k |
  | ResNet fresh | 0.258 ± 0.037 | +0.085 | 0.148 ± 0.015 | 0.215 | 0.59 / 0.64 | 2–2.5k |
  | ResNet fixed + fresh (2 seeds) | 0.239 ± 0.021 | **+0.087** (2/2) | **0.128 ± 0.009** | **0.191** | 0.72 / 0.73 | 3k / 4.5k |

  Reading [ours, interim]: more data helps the ResNet: best overall MAE yet (0.191 vs 0.211), the fresh-layout
  milestone (+0.087) and the fixed-layout late-success accuracy (0.128) together, later best steps (less
  overfitting). So the bar for π₀.₅ fixed + fresh (18237898 running) is this row, not ResNet fixed alone.
- 2026-11-01 **Check-in: ResNet fixed + fresh final (3 seeds); B fixed + fresh seed 0; LoRA lr 1e-4 at step 2,000.**

  | arm | change SNR 2 s | milestone | success MAE > 20 s | MAE | AUC 5 / 10 s | best step |
  | --- | --- | --- | --- | --- | --- | --- |
  | ResNet fixed + fresh (3 seeds) | **0.247 ± 0.020** | **+0.094 ± 0.012** (3/3) | 0.119 ± 0.015 | 0.189 | 0.70 / 0.74 | 3k–4.5k |
  | B fixed + fresh (seed 0, 18237898) | 0.152 | +0.042 | 0.119 | **0.160** | 0.70 / 0.80 | 5,000 of 5,400 |

  Reading [ours, interim, B is 1 seed]: (1) B on 825 episodes ties the ResNet on late-success accuracy (0.119 vs
  0.119 ± 0.015) and has the lowest overall MAE of any arm so far (0.160 vs 0.189), but its noise / milestone
  are where B fresh left them (SNR 0.15 vs 0.25, milestone +0.042 vs +0.094): more data improves its level, not
  its progress-tracking. (2) B's best step is 5,000 of 5,400 (every earlier B run: 2–2.5k), so the 5,400-step
  budget may now be too short for 825 episodes; worth a longer run if seeds 1–2 agree [ours]. Still fails 3b
  (milestone well below the ResNet's; late-success MAE only ties).
  LoRA lr 1e-4 fresh s0 (18237894): step 1,500 / 2,000 inner CE 3.502 / 3.460, MAE 0.186 / 0.171, rms lora_b
  4.6e-3 / 5.3e-3. Matched-data comparison with frozen B fresh s0 (18219699) inner CE: 3.503 / 3.472 — **no
  difference yet**. (vs the lr 1e-5 fixed-layout run, 3.632 / 0.192 at 2,000, different fit data.)
- 2026-11-01 **Check-in: B fixed + fresh seed 1** (18237899, best step 4,500 of 5,400; seed 2 18237900 pending on
  priority). Per seed 0 / 1: SNR 0.152 / 0.152, milestone +0.042 / +0.025, success MAE > 20 s 0.119 / 0.135, MAE
  0.160 / 0.166. 2-seed mean: SNR 0.152, milestone +0.033 ± 0.008, success MAE > 20 s **0.127 ± 0.008** (ResNet
  fixed + fresh 0.119 ± 0.015), MAE **0.163** (ResNet 0.189), AUC 5 / 10 s 0.73 / 0.79. Reading [ours]: seed 1
  weakens seed 0's tie on late-success accuracy (B now ~0.008 behind, within the ResNet's seed sd); overall MAE stays
  the best of any arm; progress-tracking unchanged (milestone ~⅓ of the ResNet's +0.094). Best step again late (4.5k),
  so the step-budget question stands.
- 2026-11-01 **LoRA lr 1e-4 fresh s0 final (18237894, 12 h 36 min, move3 A5000) + LoRA fixed + fresh submitted.**
  Scored with `score()` of `scripts/libero_head_swap.py` (Claude session; not yet in `libero_fresh_lora_score.py`):

  | arm (fresh layouts) | change SNR 2 s | milestone | success MAE > 20 s | MAE | AUC 5 / 10 s |
  | --- | --- | --- | --- | --- | --- |
  | LoRA 1e-4, CE-sel. (step 2,000) | 0.145 | +0.044 | 0.125 | 0.179 | 0.66 / 0.72 |
  | LoRA 1e-4, MAE-sel. (step 5,000) | 0.104 | +0.043 | 0.114 | 0.176 | 0.61 / 0.69 |
  | B fresh (3 seeds) | 0.157 | +0.043 | 0.148 | 0.187 | 0.60 / 0.80 |
  | ResNet fresh (3 seeds) | 0.258 | +0.085 | 0.148 | 0.215 | 0.59 / 0.64 |

  Reading [ours, 1 seed]: the adapters move (rms lora_b 6.4e-3) and train CE falls faster than frozen B (1.75 vs ~2.1
  at 5,400), but inner CE is the same at the best step (3.460 vs 3.472) and worse later: the extra capacity
  memorises the 375 fit episodes. Longer training buys level (late-success MAE 0.114, best of any arm) at the cost
  of noise (SNR 0.104); the milestone stays at the frozen level at both selections. Does not pass 3b.
  Next (Karen: go): **LoRA lr 1e-4 on fixed + fresh (825 fit episodes)**, everything else as 18237894, seed 0, job
  **18258391** (`NAME=fixedfresh_lr1e4`, `--accum 4` as 18237894, `--mem=96G`, A5000 / L40S / rtxpro6000; first
  submission 18257306 OOMed on an A5000 at step 1 because `--accum` was left at its default 1, micro-batch 32) → `runs/value_lora_fixedfresh_lr1e4_s0_<job>/`;
  also the no-pretraining control for the demo pre-training draft (`docs/libero_demo_pretrain_draft.md`).
  Queue note: 18237900 (moved to move5 rtxpro6000, started 22:55) / 18257306 pended because move3 and move4 have all CPUs allocated (GPUs free: 4 A5000, 3 L40S).
- 2026-11-02 **Demo pre-training (RECAP two-stage), stage 1 submitted** (Karen: go, without waiting for the LoRA fixed +
  fresh result, since stage 1 does not depend on it). Details, data checks and code changes:
  `docs/libero_demo_pretrain_draft.md`. Jobs: table 18260656 → ResNet stage 1 18260657; LoRA smoke 18260658 →
  LoRA stage 1 18260659. Moka demos kept in stage 1 (our default).

## 2026-11-02 08:40: stage 1 results, stage 2 submitted, frozen B fixed + fresh complete
- **Demo table 18260656 done** (01:31): 55 GB at `/iris/u/ankile/karenvo/libero/tables/libero_demos/`; ResNet frame
  cache 68,991 frames (every 4th).
- **ResNet stage 1 18260657 done** (6.6 min on an A5000, cached frames): best step 7500, inner (held-out demos) CE
  2.638, MAE 0.0156 → `runs/resnet18_demos_stage1_18260657/best.pt`.
- **LoRA smoke 18260658 failed (my error)**: `--steps 20` with the default `--warmup 500` gives a negative cosine
  decay (optax ValueError) before step 0; so LoRA stage 1 18260659 never ran (cancelled). Resubmitted with
  `--warmup 5 --eval-every 10` for the smoke.
- **`--outcome-weight 0` added to the LoRA stage-1 args** [ours, as the draft planned]: every demo is a success, so
  the outcome loss has nothing to learn. Not in the cancelled 18260659's args. The ResNet has no outcome head.
- `scripts/libero_resnet.sbatch`: new `INIT=<best.pt>` → `--init-from`.
- **Stage 2 (ResNet + demos), fixed + fresh, same settings as resnet18_fixedfresh_s* except INIT**: s0 18265882
  (now), s1 18265883 (after the LoRA fixed + fresh control 18258391), s2 18265884 (after s1). GPU limit counts the
  interactive session: 18258391 + one new job + the session = 3.
- Stage 2 data is fixed + fresh only, **without the 50 moka demos** RECAP's Alg. 1 adds [ours: matches the no-demo
  controls exactly, so the only difference is the initialisation; the moka demos were seen in stage 1].
- **LoRA chain**: smoke 18265885 (after ResNet s0) → stage 1 18265886 (5,400 steps, lr 1e-4, accum 4, outcome
  weight 0). LoRA stage 2 to submit when 18265886 finishes (~13 h after start).
- **Frozen B fixed + fresh, 3 seeds (s2 18237900 done; best step 2500)**, `score()` of scripts/libero_head_swap.py:

| arm | SNR 2 s | milestone | success MAE > 20 s | MAE | AUC 5 / 10 s |
|---|---|---|---|---|---|
| B ff s0 / s1 / s2 | 0.152 / 0.152 / 0.194 | +0.042 / +0.025 / +0.012 | 0.119 / 0.135 / 0.159 | 0.160 / 0.166 / 0.185 | |
| B ff mean | 0.166 | +0.026 | 0.138 | 0.170 | 0.74 / 0.80 |
| ResNet ff mean (3 seeds) | 0.247 | +0.094 | 0.119 | 0.189 | 0.70 / 0.74 |

  Seed 2 is the weakest of the three (MAE 0.185, milestone +0.012), so the 2-seed B means quoted before (MAE 0.163)
  were optimistic. B still has the lower overall MAE and higher AUC; the ResNet still tracks progress better.
- 08:55: Karen allowed one more GPU (4 incl. the interactive session). Used for the LoRA chain (critical path):
  smoke 18265885's dependency on ResNet s0 removed, so it runs alongside 18258391 and ResNet s0 18265882. It is
  pending on queue priority; the chain to stage 1 18265886 is unchanged.
- 2026-11-02 ~10:10: **LoRA smoke 18265885 failed at the results write** (UnboundLocalError: `inner` only set in
  the moka branch of `scripts/pi05_value_lora.py`; the demo branch never defined it). 20 training steps ran fine
  (inner CE 5.140 → 5.115). Fix: `inner = set()` in the demo (`role`) branch. 18265886 (DependencyNeverSatisfied)
  cancelled; resubmitted, same args as 18265885 / 18265886: smoke **18266858** (`NAME=demos_smoke3`) → stage 1
  **18266859** (afterok, `NAME=demos_stage1_lr1e4`, 5,400 steps). ResNet stage 2 s0 18265882 done (MAE 0.185,
  AUC 5 / 10 s 0.65 / 0.81, best step 2500); not yet re-scored with `score()`.
- 2026-11-02 **ResNet stage 2 (demo pre-trained) s0 scored** (`score()` of scripts/libero_head_swap.py, `ev_resnet`; the
  no-pretrain seeds re-scored in the same call reproduce the logged means):

  | arm | SNR 2 s | milestone | success MAE > 20 s | MAE | AUC 5 / 10 s |
  |---|---|---|---|---|---|
  | ResNet ff s0 / s1 / s2 | 0.261 / 0.218 / 0.261 | +0.095 / +0.079 / +0.107 | 0.120 / 0.137 / 0.101 | 0.192 / 0.190 / 0.186 | |
  | ResNet ff mean ± sd | 0.247 ± 0.025 | +0.094 ± 0.014 | 0.119 ± 0.018 | 0.189 ± 0.003 | 0.70 / 0.74 |
  | **ResNet ff + demo pre-train s0** (18265882) | 0.221 | **+0.054** | 0.111 | 0.185 | 0.65 / 0.81 |

  Reading [ours, 1 seed]: pre-training slightly improves level (MAE 0.185, AUC 10 s 0.81, both edge of the no-pretrain
  range) and **weakens the milestone** (+0.054, below all three no-pretrain seeds, ~3 sd under the mean); SNR within
  range. Same direction as π₀.₅ (level up, progress-tracking not). Plausible cause [ours, untested]: demo targets are
  pure steps-to-success (a clock), so stage 1 teaches a time trend the moka fine-tune keeps. Seeds 1–2 (18265883/84) decide.
- 2026-11-02 **Stage 5 check on the two Stage 6 candidates (ResNet ff, B ff; Karen: go, both in parallel).**
  `vl.libero_advantages` with two new 3-seed ensembles, `ResNet ff` (resnet18_fixedfresh_s{0,1,2}) and `B ff`
  (value_expert_fixedfresh_w256_s{0,1,2}); val only (out-of-sample, so no out-of-fold V yet); the 18 old rows reproduce
  exactly (backup of the old report / csv in `/iris/u/ankile/karenvo/libero/backup_*_20261102.*`). Main row: n = 2 s,
  70% positive, mid-episode, time-matched successes − failures (95% scene CI) / time-matched within-scene AUC:

  | V | 1.2 s | **2 s** | 4.8 s | AUC 2 s | failure-ending frames positive |
  |---|---|---|---|---|---|
  | ResNet (fixed, ref) | +3.9 [+1.4, +7.1] | +5.4 [+2.3, +8.6] | +8.6 [+5.0, +12.5] | 0.74 | 61% |
  | B (fixed, ref) | +1.3 [−1.4, +4.1] | +2.4 [−1.0, +6.1] | +5.3 [+1.4, +10.5] | 0.57 | 72% |
  | **ResNet ff** | +2.7 [+0.4, +5.1] | **+4.2 [+1.7, +7.1]** | +8.1 [+4.5, +12.5] | **0.70** | 64% |
  | **B ff** | +2.4 [−0.3, +5.3] | **+3.3 [+0.1, +7.0]** | +6.6 [+2.1, +12.3] | 0.57 | **82%** |

  Paired scene bootstrap B ff − ResNet ff (same draws for both): −0.4 [−2.2, +1.4] / −0.9 [−2.9, +0.9] / −1.5
  [−4.1, +0.9] points at 1.2 / 2 / 4.8 s. Reading [ours]: (1) both pass the basic check (successes get more positives,
  time-matched, CI above 0 at 2 s), B ff for the first time; (2) the B − ResNet gap shrank from −3.0 (all CIs below 0,
  fixed layouts) to −0.9 with a CI over 0, partly because the ResNet's lead drops with fixed + fresh (5.4 → 4.2), as its
  milestone / layout fit did; (3) the ResNet still sorts episodes better within a scene (AUC 0.70 vs 0.57); (4) **B ff
  calls 82% of failure-ending frames positive** (ResNet ff 64%): its V is too low just before a failure ends, so
  "−0.5 arrived" looks better than expected. That is the open failure-ending decision of Stage 5 and matters more
  for B ff's labels. Both remain usable candidates; the ResNet labels are the safer arm.
- 2026-11-02 **Stage 6 fine-tuning code written (not yet run on a full arm).** `src/vl/libero_finetune_data.py` (arms,
  held-out-state assert, action chunks, per-frame labels, prompt tag `"{prompt}. Advantage: positive|negative"` [ours:
  RECAP puts the indicator in the prompt; wording ours], episode streaming from the raw npz in groups of 48 since
  a frame cache of 900 episodes would be ~120 GB with the disk at 98%); `scripts/pi05_policy_finetune.py`
  (`--selftest`, `--overfit` = Check C, `--train --arm imitation|advantage [--labels]`); `scripts/serve_pi05_finetuned.py`
  (serves a run to the unchanged client, rewrites the prompt with the tag; `--tag negative` for the plan's check);
  `scripts/libero_eval.sbatch` takes `FT_RUN` / `FT_TAG`. Recipe = openpi's LoRA fine-tune (`pi0_libero_low_mem_finetune`)
  on pi05_libero: LoRA on both LLMs, **openpi's freeze filter also trains SigLIP and the action projections in full**
  (the plan's "LoRA" undersold that), batch 32, AdamW b2 0.95, clip 1, no EMA, train-time augmentation. Ours: lora_b = 0
  (step 0 = pi05_libero exactly), LoRA lr 1e-4 / other 2.5e-5, warmup 500 + cosine, 5,000 steps default, no val
  selection (final weights; val loss logged as a drift check), saves trainable weights only (~1.7 GB). +7 tests (189
  pass). GPU jobs: selftest 18267914 → Check C 18267915.
  **Still needed before a real arm C:** out-of-fold V labels for the 900 train rollouts (step 2; none of the current
  V's can label their own fit episodes), the failure-ending-frame decision, and arm A's step budget timed (~frames/s
  from Check C).
- 2026-11-02 13:05: **LoRA stage 2 queued: 18268508** (afterok stage 1 18266859, `NAME=fixedfresh_demos_lr1e4`), args
  identical to the no-pretrain control 18258391 (fixed + fresh fit, lr 1e-4, accum 4, 5,400 steps, outcome weight 1)
  except the init. `scripts/pi05_value_lora.sbatch` gained `INIT_RUN=<stage-1 run>`: at job start it reads
  `results.json` `select_ce.best_step` and adds `--init-from <run>/ckpt_<step>.msgpack` [ours: inner-CE selection, as
  the ResNet stage 2 used stage 1's `best.pt`]. Stage 1 at step 500: inner CE 2.991, MAE 0.031.
- 2026-11-02 **LoRA stage 2 without the outcome loss queued: 18268888** (Karen: go), as 18268508 except `--outcome-weight 0`;
  afterok stage 1 18266859, so it runs beside 18268508. Question [ours]: with a trainable backbone, does the
  episode-constant outcome loss reshape the features against progress tracking? (On frozen B it did not, 2026-10-31.)
  Compare with 18268508 (same init, outcome weight 1).
- 2026-11-02 **Stage 6 code checks passed.** Selftest 18267997 (A5000; the first try 18267914 ran out of memory in
  check 4 only, which held two models at once): (1) LoRA(b = 0) loss 0.01527 vs pi05_libero 0.01526 on the same batch
  / rng; (2) tagged prompts 14 tokens vs 10 untagged, positive ≠ negative, no truncation (max 200); (3) action chunk →
  input transform → output transform = raw executed actions, max |err| 0; (4) save → `load_finetuned` exact.
  Check C: 18267998 ran out of memory at the first gradient step on a 24 GB A5000 (gradients of 4 micro-batches were
  held in a list; now a running sum, and the sbatch excludes move3). **18268408 (L40S, 4 CPUs) PASS**: loss on one
  fixed batch 0.0050 → 0.0000 in 300 steps, lora_b rms 0 → 1.4e-3 (the adapters move), val loss 0.0104 → 0.0118
  (memorising one batch, as expected). Speed **6.7 frames/s on an L40S** at micro-batch 8 → 5,000 steps × 32 ≈ 6.6 h
  per arm before data-loading costs (Check C reuses one batch; the real run streams raw npz). Note: pi05_libero's loss
  on its own moka rollouts is already low (~0.005-0.015), as expected for on-policy data.
- 2026-11-02 **Out-of-fold ResNet V + arm A submitted** (Karen: go). `vl.libero_resnet --oof-fold k` (`oof_split`,
  +1 test, 190 pass) [ours]: 5 folds over the 900 train rollouts; the 30 fixed train states shuffled (rng 0) and dealt
  6 per fold (whole scenes), the 450 fresh episodes dealt 90 per fold; fold k is held out (180 episodes, predicted by
  the best checkpoint → `oof_predictions.parquet`), early stopping on 5 fixed states drawn from the other folds (rng
  1 + k, 75 episodes), fit = the remaining **645 episodes** (vs 825 for ResNet ff: the cost of labelling every episode
  out of fold). One seed per fold (the 10-29 advantage check found no clear ensemble gain). Recipe = ResNet ff
  (every 4th frame, 8,000 steps, inner-CE selection). Checked on the real table: every train episode held out
  exactly once; held-out success 0.53-0.64 per fold. Jobs (one GPU slot, GPU limit 4 incl. the session): folds
  18269059 → 60 → 61 → 62 → 63 (~1 h each), then **arm A** 18269064 (imitation on the 516 successful train
  rollouts, seed 0, 5,000 steps, L40S, ~7 h). Folds first because they are on arm C's critical path.
  Seen in the queue: two `pi05_value_lora.sbatch` jobs (18268508, 18268888) from another session, both afterok
  18266859 (LoRA demo stage 1); they will start together when it ends.
- 2026-11-02 14:10 **LoRA lr 1e-4 fixed + fresh s0 final (18258391, 13 h 52 min, move3 A5000)**; both selections picked
  step 5,400 (the last; inner CE 3.273, so the budget may be short), so CE- and MAE-selected rows are identical.
  `score()` of scripts/libero_head_swap.py, frozen B / ResNet ff re-scored in the same call (reproduce the logged means):

  | arm (fixed + fresh, 825 ep) | SNR 2 s | milestone | success MAE > 20 s | MAE | AUC 5 / 10 s |
  |---|---|---|---|---|---|
  | ResNet (3 seeds) | 0.247 ± 0.025 | +0.094 ± 0.014 | 0.119 ± 0.018 | 0.189 | 0.70 / 0.74 |
  | frozen B (3 seeds) | 0.166 ± 0.025 | +0.026 ± 0.015 | 0.138 ± 0.020 | 0.170 | 0.74 / 0.80 |
  | **LoRA 1e-4 (1 seed)** | 0.133 | +0.052 | 0.110 | **0.155** | 0.66 / 0.79 |

  Reading [ours, 1 seed]: same pattern as LoRA on fresh layouts. Best overall MAE and late-success MAE of any model;
  noisier than frozen B (0.133, below all three frozen seeds); milestone above every frozen seed (+0.052 vs
  0.012–0.042) but still ~half the ResNet's and below its lowest seed (0.079). Adapting the backbone buys level and
  some milestone, not steadiness. This is also the no-pretrain control for LoRA stage 2 (18268508 / 18268888).
- 2026-11-02 **Vision-only V from pi05's own SigLIP encoder** (Karen: go). Question [ours]: is pi05's noise in its image
  features or in the LLM / action layers? `scripts/export_pi05_siglip.py` (CPU job 18269707) exports pi05_libero's
  PaliGemma/img (So400m/14, 412.4 M; the 1152 -> 2048 LLM projection dropped [ours]) to timm
  `vit_so400m_patch14_siglip_224` names; `vl.model.SiglipPi05Features` (tanh GELU as flax, inputs [-1, 1] as openpi,
  mean of the 256 encoded tokens -> 1152-d [ours]). Port check vs openpi's JAX module on 4 real frames: max |diff|
  1.05e-4 at |encoded| max 72 (`tests/test_siglip_pi05.py`; 192 tests pass). Run = ResNet fixed + fresh exactly
  (vl.libero_resnet: every 4th frame, fit_fixed_fresh 825 episodes, 8,000 steps, backbone lr 1e-5 / head 1e-4,
  inner-CE selection, no augmentation) with `--backbone siglip_pi05` and gradient checkpointing. Smoke **18269858**
  (20 steps, after 18265884 for the GPU limit) -> seed 0 **18269859** (afterok).
  Smoke 18269858 passed (A5000, 6 min 51 s: 20 steps + 2 inner evals + val predictions written; untrained, val CE
  5.34 ≈ ln 201, MAE 0.274, as expected at 20 steps). Seed 0 18269859 started 15:04 (move3 A5000). A TITAN RTX copy of
  the smoke run in the interactive session used 17.5 GB and ~10× the time (no bf16 tensor cores); stopped once
  18269858 passed.

### 2026-11-02 — ResNet demo pre-train → fixed+fresh, 3 seeds (18265882/83/84): pre-training halves the milestone

All three stage-2 runs initialised from `resnet18_demos_stage1_18260657/best.pt` (log line "initialised from ..."), same recipe as the
no-pretrain ResNet ff seeds (18237895/96/97); scored with `score(p, "ev_resnet")` on `val_predictions.parquet` (825-ep fixed+fresh split, 120 val eps).

| | SNR 2 s | milestone | MAE succ>20 s | MAE | AUC 5/10 s | inner CE (best step) |
|---|---|---|---|---|---|---|
| ResNet ff (no pretrain) | 0.247±0.025 | **+0.094±0.014** (0.079–0.107) | 0.119 | 0.189 | 0.70 / 0.74 | 3.34–3.38 (3000–4000) |
| ResNet ff + demo pretrain | 0.203±0.019 | **+0.045±0.010** (0.035–0.054) | 0.118 | 0.184 | 0.67 / 0.77 | 3.45–3.46 (2500–3000) |

- Milestone drop is real: every pre-trained seed is below every no-pretrain seed. SNR also lower (ranges barely touch: 0.182–0.221 vs 0.218–0.261).
- Accuracy is a wash (MAE −0.005, late-success MAE equal); inner CE is *worse* and selection picks an earlier step.
- Reading [ours]: demo pre-training on success-only demos teaches a smooth time-to-go prior that washes out the pot-landing jump; it doesn't buy accuracy here. Not adopting it for the ResNet V used for advantages.

### 2026-11-02 — SigLIP (π₀.₅'s own encoder) fine-tuned, fixed+fresh s0 (18269859): overfits by step 1,500, noisy

Same recipe as ResNet ff (8,000 steps, backbone lr 1e-5, head 1e-4, inner-CE selection), backbone = `siglip_pi05`.
Inner CE 3.763 / 3.646 / **3.584** (step 1,500, selected) then rises every eval to 5.305 at 8,000 while train loss
falls to 1.3: memorises. (Log prints "resnet18 val" for any backbone: cosmetic label in `vl.libero_resnet`.)

| s0 | SNR 2 s | milestone | MAE succ>20 s | MAE | AUC 5/10 s |
|---|---|---|---|---|---|
| ResNet ff (3 seeds) | 0.247±0.025 | +0.094±0.014 | 0.119 | 0.189 | 0.70 / 0.74 |
| frozen B ff (3 seeds) | 0.166±0.025 | +0.026±0.015 | 0.138 | 0.170 | 0.74 / 0.80 |
| LoRA ff s0 | 0.133 | +0.052 | 0.110 | 0.155 | 0.66 / 0.79 |
| **SigLIP-π₀.₅ ff s0** | **0.129** | **+0.060** | 0.223 | 0.207 | 0.66 / 0.72 |

Reading [ours, 1 seed]: as noisy as the π₀.₅ arms and the least accurate model. It does not cleanly say "image
features are the problem": a 400 M encoder fully fine-tuned on 825 episodes overfits after ~1.5 epochs-equivalent,
and the selected checkpoint is barely trained. It matches SeeQ's no-robot-pretraining ablation (below): a large
VLM vision stack fine-tuned directly on target data loses to a small ImageNet CNN.

### 2026-11-02 — RECAP vs SeeQ value functions, and what to try next [proposal, to decide]

**SeeQ** (Singh, Hu, Sobol Mark, Yu, Erickson, Kumar; arXiv 2609.22085 v2, Sep 2026), read from the HTML:
- Backbone: "a pretrained 3B PaliGemma vision-language backbone" — a VLM, separate from the π₀.₅ policy. No frozen parts
  stated. 3 cameras at 224 px, **no proprioception** ("We do not use proprioceptive state as an input to any of the
  value functions trained"). Q(s, a, l, l̃): a candidate action chunk is linearly projected in; a learned value token
  at the end of the sequence attends to everything, linear readout to a **scalar** Q.
- Objective: TD with a best-of-N backup (8 π₀.₅ candidates scored by an EMA target net, τ 0.005), γ 0.999.
  **Subtask-scoped**: reward 1 when the active subtask ends, no bootstrapping across subtask boundaries. The model
  first decodes the active subtask in text (next-token CE, weight 0.1, teacher-forced), then predicts its value
  conditioned on it. Subtask labels from the datasets ("when unavailable, ... generated by prompting VLMs").
- Pre-training: **the Q-function itself** is pre-trained on RoboCOIN bimanual (131 tasks, 40,054 episodes, 31.3 M
  steps) for 230 k steps, batch 256, cosine lr 1e-5 → 1e-6; then fine-tuned per task (97–473 episodes) for 20 k steps,
  batch 128, lr 5e-6 → 5e-7. AdamW wd 1e-6, β2 0.95, clip 1.
- Use: best-of-8 steering of π₀.₅ (shirt-hang 10 → 22 / 24, lid 10 → 15, grocery 9 → 17, LEGO 5 → 10).
- Ablations relevant here: **Table 3 (shirt-hang): PaliGemma without robot pre-training 8/24 < ResNet-50 + MLP 16/24 <
  SeeQ 22/24.** Their hypothesis: "Direct target-task finetuning may therefore overfit to image features"; median
  image-gradient norm 0.093 (SeeQ) vs 0.323 (no pre-train) vs 0.494 (ResNet). Table 2: task-level MC 9/24, task TD 12,
  subtask SARSA 13, SeeQ 22. Table 4: no subtask 8/24, subtask prediction as an auxiliary only 15/24, full 22/24.
  "MC predictions are noisy, potentially reflecting spurious correlations" (Fig. 4, Fig. 8); with subtask
  conditioning the largest value drop coincides with the subtask switch, without it drops lag 4–24 frames (A.2.1).
  No RECAP-style or VLA-backbone critic baseline; no stated reason for a separate VLM beyond "vision-language priors".

**RECAP / π\*0.6** (2511.14759; details in the 2026-11-01 entry): separate 670 M VLM from Gemma 3, images + robot
state + language, 201-bin distributional **Monte Carlo** V(s) (no action input, no TD), pre-trained on the policy's
multi-task pre-training data, co-trained on web data, fine-tuned per iteration on all task data; used for
advantage conditioning.

**Where our results sit [ours]:**
1. Both papers use a VLM that is *not* the policy, and both pre-train **the value function itself** on large
   multi-task robot data before the task fine-tune. We have tested neither: our π₀.₅ arms are a VLA, and our
   pre-training is 1,693 success demos (~1/20 of SeeQ's episodes) — for the ResNet it halved the milestone.
2. SeeQ's Table 3 is our pattern: big VL backbone trained directly on task data < ResNet, on progress / steering,
   while (in ours) average error is fine. Their fix is value pre-training, not a different backbone.
3. Our weak point is exactly the subtask boundary (milestone = first pot lands), and SeeQ's subtask mechanism targets
   it. LIBERO gives subtask labels for free: the BDDL goal is a conjunction of predicates (moka: On(pot1, stove),
   On(pot2, stove)), already evaluated per step for the milestone (`vl.libero_milestone`). No VLM labelling needed.
4. RECAP's distributional MC is what we use; SeeQ finds task-level MC the noisiest objective. The objective is a
   lever we dropped on 10-31 as off-RECAP.

**Proposed experiments, in order:**
- **E0 (free, ~2 pm 11-03):** LoRA stage 2 (18268508 / 18268888) = value pre-training on a VLA. Compare with 18258391.
- **E1 VLM backbone ± value pre-training (Karen's proposal; SeeQ Table 3 on LIBERO).** Same LoRA value-expert
  pipeline, backbone initialised from **PaliGemma 3B pt_224** (SeeQ's backbone, and π₀.₅'s own init before robot
  training: same architecture, so VLM vs VLA is the only difference). 2 × 2 with the existing π₀.₅ runs:

  | | no value pre-training | LIBERO value pre-training → ff |
  |---|---|---|
  | π₀.₅ (VLA) | 18258391 (done) | 18268508 (running) |
  | PaliGemma (VLM) | **new** | **new** (stage 1 on the same 1,693-demo table) |

  Prediction from SeeQ: VLM no-pretrain worst; pre-training helps the VLM more than the VLA. Cost ≈ 14 h per ff run
  + ~13 h stage 1 (1 seed each, A5000). Code: a PaliGemma weight loader in `scripts/pi05_value_lora.py` (openpi's
  `PaliGemmaWeightLoader`, pt_224.npz not cached locally; download needed); prompt/state format unchanged. [ours:
  LoRA, not full fine-tuning, for 24–48 GB GPUs; the 1,693-demo corpus is small and success-only — if E1 shows
  pre-training matters, scale it with the libero_90 demos (4,500 episodes) and π₀.₅ rollouts with failures.]
- **E2 subtask auxiliary (SeeQ Table 4, "prediction without conditioning" 8 → 15/24), cheap.** Add a head predicting
  the number of satisfied goal predicates (moka: 0 / 1 / 2 pots on stove) from the same frame, CE weight 0.1; V target
  unchanged. ResNet ff and frozen B ff, 3 seeds each (ResNet ~1–1.5 h per seed). Asks: does it raise π₀.₅'s
  milestone toward the ResNet's, and does it calm the flicker? This adds a training label (needs Karen's ok per
  CLAUDE.md). Subtask-*scoped* values (SeeQ's full recipe) change the value target itself and the advantage
  definition, so a later step only if E2 helps.

### 2026-11-02 — E2 launched: subtask auxiliary head on frozen π₀.₅, LoRA π₀.₅ and the ResNet (Karen: go)

Karen: cancel the LoRA demo pre-training chain (stage 1 18266859 at step ~3,250 / 5,400, best inner CE at step 1,500
and rising; stage 2 18268508 / 18268888 never started; `ckpt_*.msgpack` up to 3000 kept) and run E2 on frozen π₀.₅
(3 seeds) + LoRA π₀.₅ (1 seed), with ResNet s0 as the label-leakage control.

- **Label** (`src/vl/libero_subtask.py`, new; Karen ok'd the new training label): pots on the stove in image t =
  count of goals[t − 1, :2] (`goals` = per-step BDDL predicates in the raw files, the ones `vl.libero_milestone`
  uses), image 0 from goals[0] [ours]. True state, no HOLD smoothing [ours]. Checked on all 1,020 fixed+fresh
  episodes: goals[0] count is 0 in every episode; class 2 is 0.12% of steps and **0.000 of the every-4 fit frames**
  (both pots on = success = the last step, never shown in an image), so in practice the label is binary "first pot
  landed". 308 episodes have a count drop: 459 drops, 67% recover within 10 steps (≤ 1 s; predicate flicker or a nudge),
  46 never recover, 144 episodes have a drop > 10 steps.
- **Heads** [ours]: SeeQ weight 0.1 on the subtask CE, value target / selection (inner value CE) / val metrics
  unchanged. ResNet: `ValueModel(aux_classes=3)`, a linear head on the value head's 256-d hidden layer
  (`return_aux=True`). Frozen B and LoRA: a third Dense on the expert's final h, next to value and outcome heads
  (`ValueExpert(subtask=True)`); in LoRA its gradient also reaches the adapters. Val predictions gain `subtask` and
  `p_subtask{0,1,2}`; inner subtask accuracy logged at every eval. Flag `--subtask-weight` in `vl.libero_resnet`,
  `scripts/pi05_value_expert.py`, `scripts/pi05_value_lora.py` (default 0 = no head, old behaviour). Raw-file
  `Frames.rows` gained an 8th column (subtask, −1 without goals). Tests: `tests/test_libero_subtask.py`; 196 pass;
  B `--selftest` ok.
- **Caveat**: the label is the milestone event, so a milestone gain is partly "trained on the metric". ResNet + head
  (whose milestone is already +0.094) measures how much the head alone moves it; SNR, MAE, AUC are not affected.
- Jobs (same recipes as the no-head baselines, `--subtask-weight 0.1` added):
  - ResNet ff sub s0 **18273830** (local smoke passed: subtask CE 1.03 → 1.00 in 20 steps, columns written;
    s1 / s2 18273831 / 32 cancelled per plan). Baseline 18237895.
  - Frozen B ff sub s0 / s1 / s2 **18274104 → 18274105 → 18274106** (after smoke 18273914 ok and 18273830 ends; any GPU
    but TITAN). Baselines 18237898 / 99 / 900.
  - LoRA ff sub s0 **18274115** (after smoke 18274114 ok). Baseline 18258391.

### 2026-11-02 — Offline check: averaging ResNet and π₀.₅ predictions (no training)

Per-frame mean of E[V] on the same val frames, `score()` as above, seed-matched pairs (mean ± sd over 3 pairs):

| | SNR 2 s | milestone | MAE succ>20 s | MAE | AUC 5/10 s |
|---|---|---|---|---|---|
| ResNet ff | 0.247±0.025 | +0.094±0.014 | 0.119 | 0.189 | 0.70 / 0.74 |
| frozen B ff | 0.166±0.025 | +0.026±0.015 | 0.138 | 0.170 | 0.74 / 0.80 |
| avg ResNet + B | 0.262±0.027 | +0.060±0.008 | 0.128 | 0.179 | 0.74 / 0.80 |
| avg 2 ResNet seeds (control) | 0.268±0.011 | +0.094±0.007 | 0.119 | 0.189 | 0.71 / 0.76 |
| avg 2 B seeds (control) | 0.195±0.017 | +0.026±0.007 | 0.137 | 0.170 | 0.74 / 0.80 |
| avg ResNet s0 + LoRA s0 | 0.238 | +0.073 | 0.114 | 0.172 | 0.70 / 0.80 |

Reading [ours]: the SNR gain from averaging is generic noise averaging (two ResNet seeds do as well). π₀.₅ adds level
(AUC 10 s 0.80, MAE) but no event signal: mixing it in dilutes the milestone in proportion. The two models are
complementary on *what* they get right (level vs change), not on noise.

### 2026-11-02 — P2: π₀.₅'s flicker is the success belief switching, upstream of the heads (`scripts/libero_flicker_modes.py`)

Exact split of each 2 s change of E[V] (as `change_snr_2s`) into probability moving between the success lump
(bins > −0.5) and the failure lump, and movement inside the lumps; noise = change minus its (frame, outcome) mean.

| | noise sd | between lumps | within lumps | 2 cov | SNR (= logged) | SNR outcome head | SNR P(success lump) |
|---|---|---|---|---|---|---|---|
| frozen B ff s0 | 0.101 | 0.960 | 0.026 | 0.014 | 0.152 | 0.150 | 0.139 |
| frozen B ff s1 | 0.102 | 0.959 | 0.021 | 0.020 | 0.152 | 0.142 | 0.140 |
| frozen B ff s2 | 0.074 | 0.988 | 0.027 | −0.014 | 0.194 | 0.178 | 0.170 |
| LoRA ff s0 | 0.112 | 0.958 | 0.029 | 0.013 | 0.133 | 0.130 | 0.124 |

- ~96% of the noise variance (and ~93% of the signal) is P(success) moving; sd of its 2 s noise change 0.13–0.19.
- The separate outcome head is just as unsteady (SNR 0.13–0.18) as the value distribution's success mass: both read
  the expert's h, so the flicker is in h, not in the 201-bin readout. A head-side fix is not expected to help.
- ResNet / SigLIP-ft equivalents need their probabilities (not saved by `vl.libero_resnet`); P1's extraction saves them.

### 2026-11-02 — Arm C labels from out-of-fold ResNet V; π₀.₅ out-of-fold copies queued (Karen: go)

Mentor (Slack): run the full advantage-conditioned pipeline on pi05 now, push on value learning in parallel. Choice of
V for arm C: **ResNet ff, no pretrain** (steadiest and largest milestone of every V tried; LoRA most accurate but
noisiest; demo pretraining halves the milestone; averaging dilutes it). Second arm C with **frozen π₀.₅ (B ff)**
labels, because π₀.₅ V is the real-robot plan: does its flicker cost task success? Terminal frames left as computed,
not forced negative (Karen).

- ResNet folds 18269059–63 all done (oof MAE 0.197 / 0.203 / 0.206 / 0.189 / 0.182; folds 1–4 ran in ~6 min each on
  a warm cache, all 8,000 steps, best steps 2000–3000).
- `oof_split` moved to `src/vl/libero_oof.py` (numpy only; `vl.libero_resnet` re-exports it) so the openpi script can
  use it. `scripts/pi05_value_expert.py --oof-fold k`: the ResNet's folds exactly (checked: per-fold held-out sets
  identical to the ResNet oof outputs, fit 645 / inner 75 / held 180, every train episode once), writes
  `oof_predictions.parquet` (ev_expert, ps_expert, fold).
- `scripts/libero_oof_labels.py --source resnet|expert`: concatenates folds, `vl.libero_advantages.advantage` (n = 20,
  v_T 0 / −0.5), threshold = 30th percentile of the **train** advantages [ours] → `tables/moka_fixed_fresh/labels/
  oof_<source>_n20_p30/labels.parquet`. ResNet: 104,049 frames, 900 episodes, threshold −0.0260, positive 70.0%;
  interior positive success 72.8% / failure 65.6%; terminal success 100% / **failure 63.8%** (val: 64%); within-scene
  AUC of interior share **0.74** (28 fixed scenes; val ResNet ff 0.70). Loads through `select_episodes("advantage")` +
  `load_episode` (900 episodes, all labelled).
- Jobs (one GPU slot, serial): arm A 18269064 (gres widened to any non-A5000/TITAN GPU) → **arm C ResNet 18275460**
  (afterany) → π₀.₅ copies **18275462–66** (folds 0–4, 18237898's recipe with `--oof-fold`, 5,400 steps, ~4.6 h each).
  Smoke **18275461** (fold 0, 4 episodes, 20 steps) runs now; fold 0 needs it afterok. Then: build `--source expert`
  labels → arm C π₀.₅.

### 2026-11-02 — E2 ResNet leakage control (18273830): no change

ResNet ff + subtask head (`--subtask-weight 0.1`, s0) vs the three ResNet ff seeds (18237895–97), `score(…, "ev_resnet")`:

| | SNR 2 s | milestone | MAE succ>20 s | MAE | AUC 5 s | AUC 10 s |
|---|---|---|---|---|---|---|
| ResNet ff s0–s2 (mean ± sd) | 0.247 ± 0.025 | 0.094 ± 0.014 | 0.119 ± 0.018 | 0.189 ± 0.003 | 0.697 ± 0.041 | 0.740 ± 0.033 |
| + subtask s0 | 0.258 | 0.104 | 0.129 | 0.195 | 0.727 | 0.777 |

- Val subtask accuracy 0.965, best step 3000 of 8000 (as the baseline). Every metric is within about 1 sd of the baseline
  seeds. One seed, so the conclusion is only "no large effect".
- This is the leakage control: the label equals the milestone event, yet the ResNet's milestone barely moves. So a
  jump in frozen B / LoRA (18274104–06, 18274115, both running) would not come trivially from the label. It would
  mean the head pushes π₀.₅'s features to carry pot state, which the ResNet already does.

### 2026-11-02 22:30 — RESUME HERE: P1 encoder comparison + E2 status

**P1 (frozen encoders: are π₀.₅'s image features noisy in themselves?)** — `scripts/libero_frozen_encoders.py`
- `extract`: DONE (120 min on the move2 TITAN). In `/iris/u/ankile/karenvo/libero/features/frozen_encoders/`:
  `{resnet_in, siglip_pi05, siglip_web, resnet_ft, siglip_pi05_ft}.npy` (117,993 frames × 2 cams, from
  `tables/moka_fixed_fresh/frames_every4.npy`), `probs_{resnet_ft,siglip_pi05_ft}_val.npy`, `meta.parquet`, `state.npy`.
  - `resnet_in`: ImageNet ResNet-18.
  - `siglip_pi05`: π₀.₅'s SigLIP.
  - `siglip_web`: timm `vit_so400m_patch14_siglip_224.webli`.
  - `*_ft`: the fine-tuned checkpoints `resnet18_fixedfresh_s0_18237895` and `siglip_pi05_fixedfresh_s0_18269859`.
- `analyze` (CPU): job **18276839** (`sbatch -A move -p move --cpus-per-task=16 --mem=96G`, log
  `logs/p1_analyze_18276839.out`, output `.../frozen_encoders/analysis.json`). It was PENDING (Priority) at 22:30.
  - Bug fixed before submit: the subset-file comment line was split into words (line 191, now `splitlines()`).
  - If it failed or was lost, rerun with `PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze`.
    Not in the interactive session: about 4 min per encoder on 4 CPUs.
- Measures per encoder:
  - Robot-state R² (top-64 PCs, ridge).
  - Logistic pot-on-stove probe: val acc, event SNR, flip rate.
  - The same MLP value head (`libero_head_swap.fit`, 3 seeds) scored with `score()`.
  - P2 lump split for the two fine-tuned models.
- First encoder (local smoke, matches the earlier frozen-ImageNet head SNR of 0.15–0.17). resnet_in:
  - Robot-state R² 0.648; pot probe acc 0.957, event SNR 4.93, flip rate 0.042.
  - Head: SNR 0.168 ± 0.004, milestone 0.045 ± 0.014, MAE 0.186, AUC 10 s 0.756.
- Decision rule (agreed with Karen):
  - π₀.₅ SigLIP noisier than web SigLIP → VLA training made the features noisier → supports a VLM backbone with
    pre-training (E1).
  - Both SigLIPs alike and worse than the ResNet → a SigLIP property → E1 unlikely to pay off.
  - Frozen SigLIP steady (the noise appears only after fine-tuning) → overfitting → try gentler SigLIP fine-tuning
    (lower lr / freeze early blocks).
  - Then log the table here.

**E2 (auxiliary subtask head, weight 0.1)** — status at 22:30:
- ResNet s0 18273830: DONE, no effect (entry above).
- Frozen B ff + subtask, 3 seeds, `NAME=fixedfresh_w256_sub01_s{N}`, chained. Baselines 18237898 / 99 / 900.
  - s0 **18274104**: RUNNING on move3 (~55 min in, 8 h limit).
  - s1 **18274105** → s2 **18274106**: pending (dependency).
- LoRA + subtask s0 **18274115** (`NAME=fixedfresh_lr1e4_sub01`): RUNNING on move3 (~2.8 h in). Baseline 18258391.
- To score: `score(pd.read_parquet(run/"val_predictions.parquet"), "ev_expert")` against the baselines.
  - Report milestone, SNR, MAE succ>20 s, MAE, AUC 5/10 s, and `subtask_acc` from `results.json`.
  - Caveat: the label is the milestone event, so read any milestone gain together with the ResNet control.
- Jobs from the other session (arm C / π₀.₅ out-of-fold copies): 18269064, 18275460–66.
  - Keep total GPU jobs ≤ 4, including any interactive session.

**Not yet approved:** E1 (PaliGemma VLM ± robot pre-training), gated on P1.

### 2026-11-02 23:00 — π₀.₅ oof fold 0 decoupled from arm C ResNet (Karen: go)
- 18275462 dependency changed to `afterok:18275461` only (was also `afterany:18275460`), so the π₀.₅ folds start
  after the smoke test instead of waiting for arm A → arm C ResNet. Folds 1–4 still chain serially. GPU cap 4
  (incl. interactive).

### 2026-11-02 23:15 — P1 result: frozen SigLIP is steady; the noise comes from fine-tuning

Run locally on CPU (move2, 4 CPUs, ~50 min), log `/iris/u/ankile/karenvo/libero/logs/p1_analyze_local.out`,
output `/iris/u/ankile/karenvo/libero/features/frozen_encoders/analysis.json`. Head = same MLP, 3 seeds.

| encoder | state R² | probe acc | probe event SNR | flip | head SNR 2 s | milestone | MAE | MAE succ>20 s | AUC 10 s | inner CE |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet_in (frozen ImageNet) | 0.648 | 0.957 | 4.93 | 0.042 | 0.168 ± 0.004 | 0.045 ± 0.014 | 0.186 | 0.152 | 0.756 | 3.323 |
| siglip_pi05 (frozen) | 0.660 | 0.972 | **6.95** | 0.025 | **0.172 ± 0.011** | 0.056 ± 0.025 | **0.171** | 0.145 | **0.826** | 3.308 |
| siglip_web (frozen) | 0.588 | 0.963 | 5.10 | 0.038 | 0.160 ± 0.015 | 0.057 ± 0.027 | 0.186 | 0.155 | 0.698 | 3.431 |
| resnet_ft | 0.629 | 0.962 | 5.84 | 0.028 | 0.169 ± 0.005 | 0.072 ± 0.013 | 0.188 | 0.153 | 0.762 | 3.509 |
| siglip_pi05_ft | 0.554 | 0.970 | 6.30 | 0.030 | **0.084 ± 0.003** | 0.058 ± 0.014 | 0.197 | 0.194 | 0.726 | 3.827 |

P2 lump split (fine-tuned models' own probabilities): p2 SNR resnet_ft 0.262 vs siglip_pi05_ft 0.129.

- π₀.₅'s SigLIP is **not** noisier than web SigLIP: it is equal or better on every row (head SNR 0.172 vs 0.160, probe
  event SNR 6.9 vs 5.1, AUC 0.83 vs 0.70). VLA training did not hurt the vision features. Branches 1 and 2 are out.
- All three frozen encoders give head SNR 0.16–0.17. Frozen SigLIP alone already matches the ResNet.
- Fine-tuning SigLIP halves head SNR (0.172 → 0.084) and its own p2 SNR (0.129 vs ResNet-ft 0.262), drops state R²
  (0.66 → 0.55) and raises late-success MAE (0.145 → 0.194). The pot probe stays strong (6.3), so pot info survives
  but the features gain per-frame jitter. Fine-tuning the ResNet costs nothing (0.168 → 0.169).
- **Decision (branch 3):** the noise is overfitting from SigLIP fine-tuning → try gentler fine-tuning (lower lr, freeze
  early blocks). E1's rationale (VLA pre-training made features noisy) is not supported at the SigLIP level.
- Note: earlier heads on frozen π₀.₅ *LLM-layer* features (`libero_layer_value.py`) gave 0.08–0.11, so inside
  frozen π₀.₅ the jitter enters after SigLIP, in the language-model layers.

### 2026-11-02 23:55 — P1b set-up: frozen π₀.₅ language-model rows under P1's exact protocol

Question: does the head-SNR drop from frozen SigLIP (0.172) to frozen π₀.₅ LLM features (0.08–0.11, earlier
`libero_layer_value.py` with an AttnProbe on 33 grid tokens) survive a matched head? It decides whether the LLM is a
second noise source besides SigLIP fine-tuning.
- `scripts/libero_p1b_prepare.py` → `/iris/u/ankile/karenvo/libero/features/frozen_encoders_p1b/`. The cached LLM
  features cover only the 570 Stage 2 episodes (65,602 frames; all 13,944 val frames), so every encoder is re-scored
  on that subset with `fit = subsets/fit_fixed.txt` [ours]. Rows compare with each other, not with the P1 table
  (fit there was 825 episodes). Join on (split, init_state, episode, frame_index), checked T and success.
  - `llm_final`: Stage 3a final-layer prefix hidden states (`feats_every4_18163142.npz`, agentview / wrist mean
    pools), 2 × 2048.
  - `llm_v{1,5,9,13,17}`: per-layer attention values V (`progress_18191428/v_layers.npy`), the 16 grid-4 cells per
    camera averaged (= mean over the 256 tokens), 2 × 256. Note V is a 256-d projection, not the 2048-d hidden state.
  - P1's five encoders, rows subset.
- `scripts/libero_frozen_encoders.py analyze` gained `--fit-subset` (default unchanged, `fit_fixed_fresh.txt`).
  Suite 196 pass.
- Run: CPU, local (move2), log `/iris/u/ankile/karenvo/libero/logs/p1b_analyze_local.out`, output
  `.../frozen_encoders_p1b/analysis.json`.
- Reading rule [ours, set before the result]: if `llm_final` and the V layers stay within ~0.02 of `siglip_pi05`'s
  head SNR, the earlier 0.08–0.11 was the probe, and fine-tuning is the main noise source; if they drop to ≤ 0.12
  (vs ~0.17), the LLM adds jitter on its own, and a fix for B must cover the LLM (gentler LoRA, or reading V from
  SigLIP / early layers). The layer at which SNR falls says where.

### 2026-11-03 — P1b result: the frozen π₀.₅ LLM halves head SNR on its own

Fit 43,087 frames (fit_fixed, 375 episodes), inner 8,571, val 13,944; same MLP head, 3 seeds.

| encoder | dim | state R² | probe acc | probe event SNR | head SNR 2 s | milestone | MAE | MAE succ>20 s | AUC 10 s |
|---|---|---|---|---|---|---|---|---|---|
| resnet_in (frozen) | 1024 | 0.645 | 0.947 | 4.44 | 0.155 ± 0.011 | 0.024 | 0.196 | 0.162 | 0.731 |
| siglip_pi05 (frozen) | 2304 | 0.653 | 0.967 | 6.90 | **0.122 ± 0.004** | 0.002 | 0.181 | 0.165 | 0.799 |
| siglip_web (frozen) | 2304 | 0.582 | 0.955 | 4.49 | 0.135 ± 0.022 | −0.003 | 0.195 | 0.146 | 0.663 |
| llm_v1 | 512 | 0.602 | 0.947 | 5.52 | 0.093 ± 0.013 | −0.001 | 0.192 | 0.176 | 0.785 |
| llm_v5 | 512 | 0.658 | 0.938 | 5.12 | 0.098 ± 0.013 | −0.016 | 0.192 | 0.177 | 0.805 |
| llm_v9 | 512 | 0.693 | 0.941 | 5.04 | 0.089 ± 0.017 | −0.013 | 0.186 | 0.171 | 0.803 |
| llm_v13 | 512 | 0.658 | 0.946 | 5.14 | 0.080 ± 0.003 | −0.005 | 0.193 | 0.174 | 0.814 |
| llm_v17 | 512 | 0.787 | 0.911 | 4.58 | 0.060 ± 0.006 | 0.006 | 0.201 | 0.181 | 0.756 |
| llm_final (hidden) | 4096 | 0.684 | 0.942 | 5.05 | **0.059 ± 0.009** | 0.010 | 0.191 | 0.180 | 0.761 |
| siglip_pi05_ft | 2304 | 0.544 | 0.964 | 6.43 | 0.078 ± 0.012 | 0.063 | 0.195 | 0.193 | 0.713 |
| resnet_ft | 1024 | 0.619 | 0.965 | 5.47 | 0.163 ± 0.005 | 0.090 | 0.189 | 0.152 | 0.756 |

- Matched head, matched frames: frozen SigLIP 0.122 → frozen LLM final hidden state 0.059 (−52%), about B's native
  0.07. The earlier 0.08–0.11 was not a probe artefact. Per reading rule: the LLM adds jitter on its own. (The rule's
  absolute "≤ 0.12" assumed SigLIP at ~0.17; judged against SigLIP's own 0.122 on this fit set, the drop is far
  beyond the 0.02 band.)
- Where: V already lower at layer 1 (0.093; V is a 256-d projection, so part of that may be the projection), flat to
  layer 9, then falls to 0.060 at layer 17. The final hidden state matches layer 17.
- Pot info stays decodable through the LLM (probe acc 0.94, event SNR ~5) but below SigLIP (6.9).
- Fit size matters for SigLIP, not the ResNet: frozen SigLIP 0.172 (825 episodes, P1) → 0.122 (375); frozen ResNet
  0.168 → 0.155. So "frozen SigLIP ≈ ResNet" holds only at 825 episodes.
- Two separate noise sources, each about halving SNR: the LLM layers (0.122 → 0.059 frozen) and SigLIP fine-tuning
  (0.122 → 0.078 here; 0.172 → 0.084 in P1). Gentler SigLIP fine-tuning alone cannot fix B, whose value reads the LLM
  prefix. Next decision (Karen): SigLIP-only value path with gentle fine-tuning vs adapting the LLM (see LoRA +
  subtask 18274115).

### 2026-11-03 01:30 — Arm A held behind LoRA + subtask (Karen: go)
- 18269064 now `afterany:18274115`, to keep ≤ 4 GPUs (E2 frozen chain, LoRA, π₀.₅ oof folds, interactive). Arm A →
  arm C ResNet 18275460 unchanged after that.

### 2026-11-03 — P1c: layout variety, not size; PCA control; per-layer hidden-state job queued

**Data (frozen encoders, P1 protocol, `--fit-subset`, `analysis_<tag>.json` in `features/frozen_encoders/`):**

| fit set | episodes | resnet_in | siglip_pi05 | siglip_web |
|---|---|---|---|---|
| fit_fixed (30 fixed layouts) | 375 | 0.155 ± 0.011 | 0.122 ± 0.004 | 0.135 ± 0.022 |
| fit_fresh375_s0 (fresh layouts) | 375 | 0.193 ± 0.006 | **0.194 ± 0.012** | 0.160 ± 0.008 |
| fit_fixed_fresh (P1) | 825 | 0.168 ± 0.004 | 0.172 ± 0.011 | 0.160 ± 0.015 |

(head SNR 2 s, 3 seeds; the fixed row reproduces P1b's rows exactly.)
- It is layout variety, not episode count: at equal size, fresh layouts lift frozen SigLIP 0.122 → 0.194 and the
  ResNet 0.155 → 0.193; adding the 375 fixed episodes to fresh (825) *lowers* both (0.172 / 0.168). SigLIP is the
  more layout-sensitive encoder. Consistent with the layout/episode shortcut seen before. Fresh-only pays in level:
  MAE 0.194 vs 0.181 (SigLIP), AUC 10 s 0.755 vs 0.799. So P1b's "SigLIP needs more data" is retracted: it needs
  more *layouts*.
- Implication: the 0.122 → 0.059 LLM comparison (P1b) was on the shortcut-prone fixed fit; the hidden-state job below
  re-scores the LLM rows on fresh375 / 825 too.

**PCA-256 control (`scripts/libero_p1c_pca.py`, fit_fixed, per camera, variance kept 96–99.6%):**
- siglip_pi05 → 256-d: 0.115 ± 0.004 (vs 0.122 full). V at layer 1: 0.093 ± 0.013. So the 256-d size costs ~0.007;
  ~0.02 of the layer-1 gap remains (~1.6 sd). Suggestive, not decisive: V is a learned projection, not PCA.
- llm_final → 256-d: 0.079 ± 0.001 (vs 0.059 full, V17 0.060): compression *helps* at the top (drops low-variance
  jitter dims), and the LLM still sits below SigLIP at matched size (0.079 vs 0.115).

**Per-layer hidden states (step 2):** `scripts/pi05_layer_hidden.py` + `.sbatch` [ours]: frozen pi05 image-token
residual stream h_l, mean-pooled per camera, at l = 0 (projector output) 1 2 3 4 5 7 9 11 13 15 17 18 + final norm,
on all 117,993 P1 frames (1,020 episodes); P1-format dirs under `/move/u/karenvo/libero_features/` (iris is 99%
full). Self-checks abort on the first batch: layer loop vs openpi `llm()` final (< 2e-2), llm_hfinal vs Stage 3a
(< 3e-2). `ABLATE=1` = step 3, image tokens cannot attend to prompt / state tokens.
- In-session smoke OOM-killed loading pi05 (16 GiB session cgroup) → batch job **18279695**, dependency
  `afterany:18274115` (keeps GPUs at 4). Out `/move/u/karenvo/libero_features/p1c_hidden_18279695/`.
- Then: `PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze --out <dir>
  --fit-subset <fit_fixed.txt | fit_fresh375_s0.txt | fit_fixed_fresh.txt> --tag <tag> --encoders llm_h0 llm_h1 ...
  llm_hfinal`. Reading: llm_h0 must ≈ siglip_pi05 (linear projector; sanity). An early drop = h1–h3 below h0 by
  more than ~2 sd. If so (Karen, 2026-11-03: go), run step 3 `ABLATE=1 sbatch scripts/pi05_layer_hidden.sbatch`
  (reuse the freed GPU; stay ≤ 4) and compare its h1–h5 with step 2's.

### 2026-11-03 08:45 — E2 status; LoRA + subtask OOM at the end; GPU chain re-ordered

- Frozen B + subtask (fixedfresh_w256_sub01), scored with `score(..., "ev_expert")` vs the same-seed baselines:

  | seed | SNR 2 s base → sub | milestone base → sub | MAE succ>20 s base → sub | MAE base → sub | AUC 10 s base → sub | subtask acc |
  |---|---|---|---|---|---|---|
  | s0 (18237898 → 18274104) | 0.152 → 0.155 | 0.042 → 0.043 | 0.119 → 0.124 | 0.160 → 0.161 | 0.797 → 0.809 | 0.960 |
  | s1 (18237899 → 18274105) | 0.152 → 0.152 | 0.025 → 0.040 | 0.135 → 0.136 | 0.166 → 0.165 | 0.791 → 0.796 | 0.960 |
  | s2 (18237900 → 18274106) | 0.194 → running | 0.012 → | | | | |

  No SNR change on 2/2 seeds; milestone +0.001 / +0.015. Same as the ResNet control: the subtask head does not
  steady V.
- LoRA + subtask s0 **18274115**: trained to step 5,400 (inner CE 3.269, MAE 0.1568, subtask acc 0.963; baseline
  18258391 at 5,400: 3.273 / 0.1585), then OOM-killed (MaxRSS 67 GB > 64 GB; the baseline had `--mem=96G`) while
  writing val predictions. All checkpoints saved. Recovery: `pi05_value_lora.py --eval-only STEP` [ours] (skips
  training, both selections = STEP; the log shows best CE and MAE both at 5,400) → job **18285014** (96 GB, after
  the interactive session 18277015 ends, ~10:40), writes into the 18274115 run dir. Suite passes.
- GPU cap: arm A 18269064 (other session) was also `afterany:18274115`, so it and 18279695 would both have started
  (5 GPUs). Now: 18269064 takes the freed slot; 18285014 takes the interactive session's slot; 18279695 (P1c
  hidden states) is `afterany:18285014`.
- Order confirmed by Karen (2026-11-03 08:55): arm A 18269064 → freed slot; LoRA eval 18285014 after 18277015; P1c 18279695 after 18285014. A new interactive session should be CPU-only until 18285014 ends (≤ 4 GPUs).

### 2026-11-03 11:20 — Arm C ResNet runs beside arm A; P1c held (Karen: go)
- 18279695 (P1c) held (`scontrol hold`; `scontrol release 18279695` to resume).
- 18275460 (arm C ResNet) dependency `afterany:18269064` → `afterany:18274106` (takes E2 s2's GPU, ends ≤ 16:14), so
  arm C trains in parallel with arm A 18269064 instead of after it.
- π₀.₅ oof fold 1 (18275463) COMPLETED 11:06 (5 h 34 min). Interactive 18277015 timed out 10:38.
- 18275464 (π₀.₅ oof fold 2) dependency set to `afterany:18285014` (Karen: go), so it starts when the LoRA eval ends and
  the batch jobs + interactive 18286104 stay ≤ 4 GPUs. Folds 3–4 (18275465–66) still chain after it.
- 11:35: TITAN RTXs don't count against the GPU cap (Karen). P1c 18279695 released and pinned to `gpu:titanrtx:1`
  with no dependency (forward-only π₀.₅, fits 24 GB; Turing has no native bf16, so expect it slower). Policy
  fine-tunes (OOM on 24 GB) and the LoRA eval stay off TITANs; π₀.₅ oof folds stay off too (never timed on Turing).

### 2026-11-03 11:45 — Step 0 (PaliGemma gate) prepared, SigLIP-direct arm queued (Karen: go)
- **SigLIP-direct** [ours]: V read from SigLIP's pooled image tokens, no LLM (same model / recipe as 18269859:
  fixed+fresh 825 episodes, every 4th frame, 8,000 steps, head lr 1e-4, inner-CE selection). New
  `vl.libero_resnet` flags `--freeze-backbone` and `--backbone-lr` (default 1e-5 = unchanged). Job **18286678**
  (seed 0, sequential): (a) `siglip_pi05_fixedfresh_frozen_s0_18286678` frozen SigLIP; (b)
  `siglip_pi05_fixedfresh_lr1e6_s0_18286678` SigLIP lr 1e-6 (10× gentler than 18269859). Runs as a 5th GPU
  (Karen's choice); excludes move1/move2 (TITAN RTX: no bf16, ~10× slower). Compare with ResNet ff (SNR 0.247,
  milestone +0.094), frozen B (0.166), SigLIP-π₀.₅ ft 18269859 (0.129).
- **Step 0** [ours]: `scripts/pi05_layer_hidden.py --paligemma <npz>` (`PALIGEMMA=<npz> sbatch
  scripts/pi05_layer_hidden.sbatch`) swaps pi05's SigLIP + projector + Gemma expert 0 + embedder for the
  big_vision PaliGemma 3B pt_224 weights (π₀.₅'s pre-robot initialisation), same prefix, same per-layer pooled
  hidden states as P1c. Asserts every non-expert tensor is replaced and differs from pi05; layer loop vs `llm()`
  check kept. openpi's GCS path returns 403; weights from HF `google/paligemma-3b-pt-224-jax` (gated, Karen accepted)
  → `/iris/u/ankile/karenvo/pi05/cache/paligemma/paligemma-3b-pt-224.npz` (11.7 GB f32, 32 tensors). Smoke
  **18286805** (256 frames → `/move/u/karenvo/libero_features/step0_paligemma_smoke`), full **18286806**
  (`afterok:18286805,afterany:18286678`, Karen: stay at 5 GPUs) → `/move/u/karenvo/libero_features/step0_paligemma_hidden_18286806`.
  Then analyze as P1c (same `--encoders llm_h0 … llm_hfinal`) and compare per layer with 18279695.

### 2026-11-03 13:40 — Arm A crashed in the data loader (fixed, resubmitted); P1c fails on TITAN; E2 s2 + LoRA eval scored
- **Arm A 18269064 FAILED after 8 min** (move5, step 0 val 0.0104): `ValueError: generator already executing`. The
  prefetch pool had 2 threads calling `next()` on the same `EpisodeStream` iterator. Check C never hit it (fixed batch).
  Fix in `scripts/pi05_policy_finetune.py` [ours]: `next(raw_it)` on its own 1-worker executor (serial, so batch
  order is deterministic), `batch_arrays` still on 2 workers. Toy check: old pattern raises the same error, new gives
  all batches in order. 196 tests pass. Resubmitted **18288138** (`NAME=armA_s0 ARGS="--train --arm imitation --seed 0"`,
  14 h, any non-TITAN/A5000 GPU). Arm C 18275460 reads the script at start, so it gets the fix.
- **P1c 18279695 FAILED on the TITAN** (move2): layer-loop vs `llm()` self-check max rel err 0.353 (> 2e-2) on the
  first batch. Same code passes at 0.00 on move5 (PaliGemma smoke 18286805), so treat this as Turing numerics (no
  native bf16), not a code bug. Not resubmitted: π₀.₅ forward jobs must stay off TITANs.
- **LoRA + subtask (18285014 eval of 18274115)**, `score(..., "ev_expert")` vs LoRA 18258391: SNR 0.133 → 0.140,
  milestone +0.052 → +0.047, MAE 0.1553 → 0.1564, MAE succ>20 s 0.110 → 0.107, AUC 5 s 0.664 → 0.681, AUC 10 s 0.788 →
  0.783, subtask acc 0.965. Below the bar (clearly above 0.133, ideally above frozen B's 0.166): no real change.
- **E2 frozen B + subtask s2 (18274106)** vs 18237900: SNR 0.194 → 0.137, milestone +0.012 → +0.029, MAE 0.185 →
  0.169, MAE succ>20 s 0.159 → 0.132, AUC 5 s 0.772 → 0.741, AUC 10 s 0.802 → 0.779, inner subtask acc 0.919. E2 over
  3 seeds: SNR 0.152/0.152/0.194 → 0.155/0.152/0.137 (s2's baseline was the high seed). Subtask head does not steady V.
- 13:55: **P1c resubmitted as 18288157** (Karen: after arm A and arm C), `--exclude=move1,move2` (no TITANs),
  `afterany:18288138:18275460`, 64 GB as before → `/move/u/karenvo/libero_features/p1c_hidden_18288157/`. Analyze per the
  P1c resume step with that dir.

### 2026-11-03 14:00 — Progress site (hub + weeks 5, 6, 7) built for the mentor
- `scripts/build_site.py` builds `reports/site/value-learning/` (hub `index.html`, `week-05/`, `week-06/`, `week-07/`) from
  `reports/site/content/{week07,earlier,jobs,episodes}.json`; the week-7 in-progress table reads job states live from
  sacct at build time. `scripts/site_episode_assets.py` makes the episode-player data (val eps 454 / 534 / 469, seed 0
  of ResNet ff 18237895, B ff 18237898, LoRA ff 18258391, SigLIP ft 18269859; per-episode MAE, flicker = sd of the
  0.4 s change, jump at the first pot placement) and camera sprites.
- All backbone numbers recomputed with `score()` from each run's val_predictions.parquet; means match the log within
  0.001 (some log sds were population sds; the site uses sample sds).
- New result seen while compiling: **SigLIP-direct frozen** (18286678 part a,
  `siglip_pi05_fixedfresh_frozen_s0_18286678`): SNR 0.221, milestone +0.017, MAE 0.215, MAE succ>20 s 0.194 (1 seed).
  Steady but barely reacts at the milestone. Part b (SigLIP lr 1e-6) still running.
- Publish (public): `gcloud storage cp -r --cache-control=no-cache reports/site/value-learning/* gs://zima-robot/reports/karenvo/value-learning/`

**2026-11-03 16:45 — SigLIP-direct part (b), gentle fine-tune (lr 1e-6, job 18286678, seed 0, fixed+fresh 825).**
Run `/iris/u/ankile/karenvo/libero/runs/siglip_pi05_fixedfresh_lr1e6_s0_18286678`, best step 5000/8000 (inner CE 3.389).
`score(val_predictions.parquet, "ev_resnet")`: MAE 0.187, MAE succ>20 s 0.150, change SNR 2 s 0.107, milestone +0.093,
AUC 5/10 s 0.712/0.776, val CE 3.517. Accuracy matches the ResNet (0.189) but steadiness is the lowest of every backbone
(ResNet 0.247, frozen SigLIP 0.221, frozen π₀.₅ 0.166, LoRA 0.133, SigLIP lr 1e-5 0.129). Fails the bar. SigLIP-only
branch closed: fine-tuning SigLIP adds noise at every lr tried, independent of the LLM. Site card updated + uploaded.

**2026-11-03 19:30 — Mentor meeting; Week 8 opened.** Agreed next steps (episode length in env steps, train-all /
label-all advantage labels as a comparison to the 5-fold labels, revisit the LLM layer ablation, scene-calibration
scatter) and new directions (SmolVLA-style truncated π₀.₅ value expert, small-VLM backbone, SeeQ subtask-scoped V)
written into docs/CURRENT.md with how-to steps. Site: Week 7 frozen as the pre-meeting snapshot; new work goes on
week-08 (`reports/site/content/week08.json`).

### 2026-11-03 17:45 — SigLIP-direct scored (18286678): steady-but-blind frozen, reactive-but-noisy gentle
- Frozen SigLIP: SNR 0.221 (noise 0.057, lowest of all), milestone +0.017, MAE 0.215. lr 1e-6: SNR 0.107 (noise
  0.097), milestone +0.093, MAE 0.187. Refs s0: ResNet 0.261 / +0.095 / 0.192; frozen B 0.152 / +0.042 / 0.160.
  Not more useful than the ResNet; B's accuracy comes via the LLM. Report `reports/2026-11-03-siglip-direct/`.

**2026-11-03 20:30 — Episode length, scene calibration, train-all labels queued.**
- Episode length: LIBERO env `control_freq=20` (openpi third_party/libero env_wrapper.py:27; our rollout script keeps
  the default), but videos are written at fps=10 and every "seconds" label in our scripts assumed 10 Hz (FPS = 10 in
  vl.libero_compare). Median success 404 steps = 20 s real (video 40 s). All "2 s" lags / lookaheads (20 steps) are
  1 s real; metrics unchanged (computed in steps). Week 8 site states times in steps + real seconds.
- Scene calibration (10 val scenes × 12, mean V over frame_index < 40): r with scene success ResNet 0.66 / frozen
  π₀.₅ 0.68 / LoRA 0.65 (same); slope vs target early V 0.23 (per seed 0.19–0.27) vs 0.49 (0.45–0.53) / 0.57;
  per-scene error 0.150 vs 0.116 / 0.120. π₀.₅'s accuracy edge = committing to scene difficulty, not ranking it
  better. Data: reports/site/content/scene_calibration.json; Week 8 site.
- Train-all / label-all: `vl.libero_resnet --predict-train` (new flag, writes train_predictions.parquet for all
  900 train episodes) on 18237895's best.pt (fit 825 + 75 inner early-stop episodes = the full train set), then
  `libero_oof_labels.py --source resnet_all` (new source; same n=20, 30th pct rule) → labels/trainall_resnet_n20_p30.
  Job 18291440 (A5000). pytest: 196 passed.

**2026-11-03 21:00 — Reprioritised (Karen): train-all paused; evals, layer probe and A2 queued.** 18291440 cancelled
(pending, never ran). Policy evals queued: 18291520 (advantage-tagged, positive; afterok 18275460), 18291521
(negative-tag control; afterany 18291520), 18291522 (success-only; afterok 18288138); each 33 × 10 val states
(9 12 13 17 25 32 40 42 44 48), env seed 7. P1c 18288157 dependency released (runs next); PaliGemma 18286806 now
afterany 18288157. A2: `pi05_value_expert.py --n-layers N` (slices the K/V cache to the first N layers; exact by
causality in depth, selftest diff 0.0) — 18291639 (N=9) → 18291640 (N=4), frozen-π₀.₅ recipe (w256, lr 1e-4, 5400
steps, augment, fixed+fresh), afterany 18286806. Peak GPU use 4.

**2026-11-03 21:20 — GPU over-use fixed; layer probes resubmitted.** P1c 18288157 (self-check rel err 0.319 on an
A5000) and PaliGemma 18286806 (OOM on an A5000) both failed in 6 min, so 18291639 (A2, N=9) started early as a 4th
batch GPU next to Karen's interactive session; Karen: too many. 18291639 requeued (held, then chained). Now three
lanes, ≤ 3 batch GPUs: 18275460 → 18291520 → 18291521 → P1c 18292149 → PaliGemma 18292150 (both --exclude
move1,move2,move3, OUT dirs reused); 18288138 → 18291522 → 18291639 (N=9) → 18291640 (N=4); folds 18275464 → 65 → 66.

**2026-11-03 22:00 — B2 (SeeQ-style Q + best-of-N steering) approved; pre-checks queued.** Plan in docs/CURRENT.md
§B2. New `scripts/libero_steer_probe.py` (+ `.sbatch`, analysis `scripts/libero_steer_analyze.py`): base π₀.₅,
10 val states, 8 candidate chunks at every replan point (spread), and at policy steps 40/130/220/310 each candidate
run to the end 2× from the saved state (headroom: between-candidate success variance, cross-validated oracle gain;
estimators checked on synthetic data: ≈ 0 under the null, recover the true variance). Sim restore had to include
the Panda gripper's integrated command (else 0.011 state drift after 5 steps; now 3e-17). Job 18296012, chained
afterany 18291521; P1c 18292149 re-chained behind it (still ≤ 3 batch GPUs). Report folder when done:
reports/2026-11-0X-steer-probe/.

Reordered (Karen): P1c 18292149 → steering probe 18296012 → PaliGemma 18292150, all afterany, behind 18291521.

**2026-11-04 00:20 — Job check; P1c self-check was a bad metric, not the GPU.** Finished cleanly: success-only
fine-tune 18288138 (5000/5000 steps, val loss 0.0091), evals 18291520 (advantage-tagged, + tag) **135/330 = 0.409**
and 18291521 (− tag, control) **142/330 = 0.430**, fold 3 18275465 (val MAE 0.180). P1c 18292149 failed its
self-check on an L40S (0.449), so the "A5000/TITAN numerics" story is wrong: with π₀.₅ weights it failed on TITAN /
A5000 / L40S and on **CPU** (0.335; `scripts/debug/selfcheck_diag.py`, job 18298452). The only earlier pass was the
PaliGemma-weights smoke. Diagnosis: layer 0's k matches exactly, then the unrolled loop and openpi's scanned/remat
`llm()` drift by bf16 rounding (k rel L2 3e-3 at layer 1 → ~1e-2 by layer 17, smooth, no jump), while the gate used
max single-element |diff| / max |ref| over all 968 tokens (one element of 30 at |ref| 90). The saved quantity, per-camera
pools, differs by 0.31% / 0.46%. Fix [our choice]: gate on pooled rel L2 < 2e-2; valid-token rel L2 and the old
metric are logged for info only. CPU smoke 18298721 (64 frames): pooled 4.9e-3, Stage 3a pool check 9.8e-3, done.
Resubmitted P1c as **18298946** (afterany 18296012, OUT reused); PaliGemma 18292150 re-chained afterany 18298946.

**2026-11-04 11:15 — Overnight jobs; B2 stopped; π₀.₅ OOF labels weaker than ResNet's.**
- Success-only eval 18291522: **134/330 = 0.406** (vs advantage-tagged + tag 0.409, − tag 0.430): no effect of
  the ResNet advantage labels.
- π₀.₅ OOF fold 4 18275466 done (OOF MAE 0.176). Labels `--source expert` → labels/oof_expert_n20_p30 (CPU job
  18306000): OOF MAE 0.182 (ResNet 0.195) but within-scene AUC **0.68** (ResNet 0.74), interior positive success
  vs failure 72/66% (ResNet 73/66%), terminal-failure positive 70% (ResNet 64%). Frame agreement with ResNet labels
  65% interior (chance 57%), advantage corr 0.27. More accurate V, worse labels (V flicker enters A directly).
  No π₀.₅-label policy fine-tune queued.
- B2 steering probe 18296012 (reports/2026-11-04-steer-probe/): candidate EEF spread 1.9 mm vs 22.7 mm motion;
  cross-validated oracle gain **+0.006 [−0.028, 0.042]** → stop rule met, **B2 stopped at step 2**.
- P1c 18298946 done (all 19 layers). CPU analysis 18305897 (fit_fixed_fresh) / 18305898 (fit_fixed),
  `scripts/layer_analyze_cpu.sbatch`.
- PaliGemma 18292150 failed only the Stage 3a pool check, which compares with π₀.₅-weight pools (2.67 by design;
  the PaliGemma layer-loop check passed, 3.4e-3). Check now skipped with --paligemma; resubmitted **18305873**.
- A2 N=9 18291639 / N=4 18291640 trained 5400 steps, best.msgpack saved, host-OOM (64G) in scoring;
  `--eval-only` at 96G: **18305874** / **18305875**. GPUs: 3 batch + interactive.

**2026-11-04 12:45 — A2 seed 0 scored; per-layer probe on fixed+fresh.** A2 `--eval-only` at 96G (18305874 /
18305875): N=18 (18237898 s0) SNR 0.152 / MAE 0.160, N=9 0.158 / 0.162, N=4 0.180 / 0.169 (noise 0.101 → 0.092 →
0.085); bar not cleared on one seed (reports/2026-11-04-truncated-value-expert/). Per-layer analysis 18305897
(fit_fixed_fresh, 3 seeds): SNR 0.14–0.16 for h0–h15, h17 0.122, h18 0.107, final 0.116; milestone ~0 at h17+;
MAE ~0.175 throughout; AUC10 max h13 0.837. h1–h3 within 2 sd of h0 → no ABLATE run. CPU analysis now
`scripts/layer_analyze_cpu.sbatch`. GPU rule restated by Karen: ≤ 3 batch jobs visible besides the interactive
session (CPU jobs chained). Week 8 site updated and uploaded.
- 13:00: per-layer analysis on fit_fixed (18306065, 375 fixed, 3 seeds): h0 0.107 ± 0.023, h1–h17 0.079–0.094,
  h18 0.094, final 0.055 (MAE 0.175–0.198). Reproduces P1b (L1 0.093 / L13 0.080 / final 0.059); the drop after
  layer 0 is ~1 sd. Both sets: final layer worst, h0–h15 about equal.

**2026-11-04 14:45 — Diagnosis + research lanes queued (Karen: go).** Base π₀.₅ eval on the arms' exact 330 trials
(18309278): earlier base val rollouts were 63/120 = 0.525 vs fine-tunes ~0.41, so the fine-tune itself may hurt.
Tag-usage check (18309419, new `scripts/libero_tag_usage.py`): action change + vs − tag at fixed noise vs change
across noise draws; flow-matching loss with correct vs flipped tag on labelled train frames; arm A as control.
Steering v2 [ours]: per-request `noise_scale` in `serve_pi05_finetuned.py --base`; `libero_steer_probe.py
--args.cand-noise-scale` scales candidates 1..N-1 only; analysis adds `cv_gain_vs_cand0` (picked candidate vs a
normal sample, both on held-out repeats; v1 run: +0.003). Jobs 18309643 (1.5) → 18309644 (2.0). A2 N=4 seeds 1–2
(18309645 → 18309646). A3: `pi05_value_lora.py --n-layers` (K/V sliced as in A2; LoRA of unused layers gets no
gradient), N=4 seed 0, 18309647. Lanes keep ≤ 3 batch GPUs.
- 15:00: tag-usage 18309419 → arm C ignores its tag (reports/2026-11-04-tag-usage/). PaliGemma per-layer analysis
  18308029: top-layer noise already present pre-robot; robot training improves accuracy everywhere and mid-layer
  steadiness, removes top-layer milestone response. A3 18309647 released (its dependency ended).
- 15:35 (Karen: go, prioritise): oracle-tag fine-tune 18310224 (arm C recipe, seed 0; labels/oracle_outcome =
  positive iff the rollout succeeded, 52% of frames; diagnostic, not a deployable label) → tag check 18310225
  (`libero_tag_usage.py --run --labels`, new flags). Steering v2 18309643 re-chained after 18310225.
  Reading: oracle tag learned (tag effect ≫ noise / loss correct < flipped) → ResNet labels too weak; not learned →
  prompt-tag conditioning too weak for this recipe.
- 16:00: base π₀.₅ on the arms' 330 trials 0.564 (18309278): all fine-tunes −13 to −16 pts (reports/2026-11-04-base-vs-finetuned/). Oracle fine-tune 18310224 started (tag-learning check still valid; its eval would inherit the recipe loss).
- 16:20 (Karen: go, ahead of steering): recipe variants on arm A — R1 frozen SigLIP 18310771 → eval 18310772 (lane 3); R2 10× lower lr 18310773 → eval 18310774 (lane 2, A2 s2 re-chained after); R3 original recipe + checkpoints 18310775 → evals @1000 18310776 / @3000 18310777 (lane 1, steering re-chained after). New: --freeze-siglip, FT_STEP.
- 17:45: A2 N=4 seed 1 18309645: SNR 0.182, MAE 0.170 (seed 0: 0.180 / 0.169). R2 18310773 started.

### 2026-11-04 18:30 — B1 subtask-scoped ResNet V queued (Karen: go)
`vl.libero_resnet --subtask-scoped` (target `vl.libero_subtask.scoped_targets`, test `tests/test_libero_scoped.py`): value of finishing the current subtask (first pot, then second; boundary = milestone frame), head 2 × 201 read by the true scope [ours: MC, SeeQ uses TD]. Jobs 18313821 / 18313822 (seeds 0, 1, fixed+fresh) chained after R1 eval 18310772. Subtask-scoped Q on noisy-candidate rollouts listed as "try later" in docs/CURRENT.md, only if steering v2 shows headroom.

### 2026-11-04 20:25 — A3 scored; jobs moved off RTX PRO
A3 (truncated N=4 + LoRA, 18309647): change SNR 0.169, MAE 0.166 (frozen N=4 0.180/0.169, LoRA-full 0.133/0.155). Pending jobs re-pinned to L40S (π₀.₅) or L40S/A5000/TITAN (ResNet B1) at Karen's request.

### 2026-11-04 22:40 — R2 moved off move5; oracle tag check; CI fix; advantage examples
R2 (low lr) 18310773 cancelled at step 4000 on move5 (Karen) and resumed on L40S as 18315373 (val 0.0075 = original; step 4050 loss/grad identical; Adam moments restart, b2 0.95) → eval 18315374 → 18309646. Oracle-outcome tag check 18310225: tag/noise 0.021, loss correct = flipped — the outcome tag is ignored too, but it is a per-episode tag (constant within an episode), so it does not test per-frame advantage conditioning. Bug: `libero_tag_usage.py` bootstrapped mean-of-ratios against a pooled-ratio point estimate (CI excluded it); fixed (`boot_ratio_ci`), old runs' intervals need a rerun. Advantage examples (`reports/2026-11-04-advantage-examples/`): ResNet V bimodal, A ≈ ΔP(success) over 20 steps (1 s; r 0.89–1.00 on 5 episodes); 76–92% negative before pot drops, 19% before successful first placements (base 30%); flat stretches default positive (A −0.019 > threshold −0.026).

### 2026-11-05 00:10 — Label threshold rules compared (no training)
`scripts/libero_label_rules.py` → `reports/2026-11-04-label-rules/`. R0 (p30, as trained) / R1 A>0 / R2 p70 / R3 no time cost. R3 ≡ R0 (constant cost under a percentile). R1/R2 make stalls negative (78–97% vs 26–38%) but ordinary success progress is A noise around 0 (IQR ±0.04 vs stall −0.019): R1 labels it 54% positive, R2 35%. Event lift vs base shrinks (drops 2.6× → 1.6× / 1.3×). Viewer gained a rule selector + stall strip; 21 episodes, 20 fps video (LIBERO 20 Hz — earlier figure times were 2× too long, fixed). Karen to choose.

### 2026-11-05 — Label denoising compared (no training)
`scripts/libero_label_denoise.py` → `reports/2026-11-05-label-denoise/`. Noise proxy = ResNet vs expert agreement (κ 0.20, r(A) 0.27 for the current labels). Smoothing V (k = 9 value frames) cuts flips 0.96 → 0.34/s, r → 0.38, stall/ordinary separation ×1.8, timing kept. Longer n (80/160) raises κ to 0.39/0.54 but spreads each event's penalty over 4/8 s. Margin (untag middle) leaves A noise unchanged, filters ambiguous frames. Karen to choose.

### 2026-11-05 — Literature review: advantage conditioning and value noise
`reports/2026-11-05-literature-advantage-conditioning/`. RECAP App. A-F: ~30% positive (pre-training), ~40% (fine-tuning), 30% tag dropout, demos tagged positive, tag learned in multi-task pre-training. **Our arm C labels were 70% positive with no dropout and no demos** (the "RECAP 30% threshold" was implemented as the 30th percentile of A). RCSL theory explains an ignored tag when P(tag | s, a) ≈ P(tag | s). Dissecting (2609.28161): n-step TD critic, value-binned calibration, continuous weights (74% vs filter 50% success). Proposed: offline tag-informativeness test + value-binned calibration, then a RECAP-faithful tag run and a weighting run (Karen to decide).

### 2026-11-05 — R2 passes; value-binned calibration; tag predictability
- **R2 (arm A, 10× lower lr; 18310773 → resumed 18315373 → eval 18315374): 172/330 = 0.521 vs base 0.564, −0.042
  [−0.115, +0.027]** (scene-stratified, as the base-vs-finetuned report): within the "does not hurt" bar. Original
  recipe −0.158.
- Value-binned calibration (`reports/2026-11-05-value-calibration/`): no subtask offset in our A (η² 0.00002); binning
  removes a V-dependent mean (regression to the mean) and spread; κ 0.20 → 0.25 (70%) / 0.29 (40%). R_prog 0.58–0.60.
- Tag predictability (`reports/2026-11-05-tag-predictability/`, interactive A5000): the action explains 3.2% of the
  current tag's entropy beyond the state (oracle 2.3%, stall control 12.9%, random 0); 30–40% positive → 7.4–8.1%;
  value-binned 40% → 8.5%; holds on ordinary frames.
