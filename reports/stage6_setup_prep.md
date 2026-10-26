# Stage 6.0 prep: what to settle with Lars and Rohan (draft, 2026-10-26)

Draft for Karen, written by Claude from `PROGRESS.md` (Stage 6–8, Open questions) and the Stage 4–5.5 results.
Nothing here has been asked yet. Stage 6 formally starts after the Stage 5.5 merge gate; this is the setup
conversation only, which has the longest lead time.

## What we bring (one paragraph for the conversation)

An image-only RECAP-style V(o) on Mulligan Insert Marker: per-frame `wrist_left` + `side_1`, a fine-tuned
ImageNet ResNet-18 shared over both cameras, a 201-bin categorical head over (−1, 0), −1 per step plus a
failure penalty, timeouts = failure. On 247 held-out val episodes it beats the repo-only constant
(MAE 0.113 vs 0.140; eval rounds 0.183 vs 0.222) and separates success from failure within a round from
~10 s on (AUC 0.82). Since Gate 0, nothing beat it: per-camera backbones, attention fusion, 2/4 s image
history and even a perfect "how long has the robot been still" input all tie or lose. Known limits: within
an episode V wiggles more than the straight-line target (shape error 0.049 vs 0.037 for a flat line), V at
the holder can't tell a stall from a pause before a seat, and part of V's skill is recognising the round.

## A. Must-know before 6.1 (the value head on π₀.₅)

1. **Checkpoint.** Which π₀.₅ DROID checkpoint (openpi `pi05_droid`, or a lab fine-tune)? Where does it live
   on IRIS, and which code base runs it (openpi JAX vs a PyTorch port)? 6.1 needs access to intermediate
   tokens (Lars: the value head cross-attends into them), so: which layer(s), and is there a hook for them?
2. **Inputs it expects.** π₀.₅ DROID is trained on DROID camera views and a language prompt. Does it take
   our `wrist_left` + `side_1` at 15 fps as-is, at what resolution, and with what prompt for Insert Marker?
   Does the Mulligan setup *already* run π₀.₅ (were the r00–r05 eval policies π₀.₅-based)?
3. **Compute.** IRIS GPU type, count, memory, queue limits. Size estimate [ours, rough]: Mulligan train is
   373,947 frames (+ 64,906 val). A frozen-VLA token cache means one π₀.₅ forward per frame; at an assumed
   ~10–30 frames/s per GPU that is ~4–10 GPU-hours, plus storage for the cached tokens (depends on which
   tokens; decide before caching). Low-LR fine-tuning of the VLA (the second step of 6.1) costs much more.
4. **Phase 2 storage.** Our data already sits under `/iris/u/ankile/karenvo/vl-data`; confirm the quota is
   enough for VLA token caches.

## B. Rohan's rollout stream (6.3, Stage 8)

5. **Data format.** LeRobot like Mulligan? Fields, cameras (the same wrist + side view?), fps, episode
   boundaries, proprio.
6. **Are failures kept?** Rohan "keeps successes". V needs failures and timeouts to learn the failure hump;
   if failures are discarded at collection time, that has to change before any V can be trained on the stream.
7. **Success labels.** Who labels (human, script, VLM)? Binary only, or with an endpoint T? Are stalls
   soft-truncated like Mulligan's reviewer edits (Q7 below)? Our targets need a valid final frame T per episode.
8. **Task.** Same Insert Marker task and scene as Mulligan, or new tasks? V is task-specific (the target scale
   uses a 900-step horizon = the 60 s cap). A new task means new targets (horizon, failure penalty) and
   training data from that task.
9. **Volume.** Episodes per day, and how many failures among them.

## C. Policy improvement and evaluation (6.3–6.4, Stage 8)

10. **Advantage-conditioned training of π₀.₅.** Whose training code (openpi fine-tuning?), who runs it, budget.
    RECAP's advantage conditioning has been shown in sim only; the real-robot version is the contribution.
11. **Evaluation.** How much robot time, which success measure, how many trials per arm for a decision? (Our
    val set resolves only ~0.025 CE with 247 episodes; policy success rates will have the same power problem.)
    Lars's throughput objective (Stage 8.2): completions per minute at 1× speed.
12. **Fair comparison.** Matched init, budget, preprocessing and data mixture; strong imitation baseline
    (filtered BC on Rohan's successes is the natural one).
13. **Which V.** Lars's plan is the π₀.₅ value head (6.1). The Stage 5.5 ResNet V is the fallback (6.2 gate).
    Out-of-fold V for advantages on V's own training episodes (an overfit V gives ~0 advantages there)?

## D. Still-open data questions (from PROGRESS.md "Open questions"; good to clear in the same meeting)

- **Q4** Does Lars have a loader that applies reviewed outcomes and endpoint trimming? (we wrote our own)
- **Q5** Is `r05-screen` a separate diagnostic distribution? It and `r05-eval` share the same 50 starts.
- **Q6** Compute and policy-evaluation setup (= A3 / C11).
- **Q7** Soft truncation: what rule picked the stall frame (210 human `soft_truncate` edits)? Why do 47
  training timeouts end before the cap, and why do 4 r03 rollouts run 826–840 steps past `max_steps: 800`?
- **Q8** Does a marker seated in *any* of the holder's 4 holes count as success?
- **Round cue** (Stage 4): V tells rounds apart from frame 0 (mean V −0.55 r00–r02 … −0.40 r04). Were rounds
  recorded on different days / lighting / camera setups? Matters for Rohan's stream: V may key on session
  appearance rather than task state.

## E. Decisions for Karen to take into the meeting

- Whether to propose the ResNet V as the first advantage source (ready now) while the π₀.₅ head is built.
- Which inference recipe V ships with (ensemble, smoothing, temperature): see `reports/lars_package_draft.md`,
  "Inference recipe".
