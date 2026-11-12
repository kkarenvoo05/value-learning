# Current state

**Updated 2026-11-08 14:00 PST.** Overwrite this file at the end of every session; do not append. History lives
in `docs/libero_plan.md` (LIBERO) and `PROGRESS.md` (Mulligan stages). Where to find things: `docs/INDEX.md`.
Progress site: https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/index.html. New work goes on
Week 8 (`reports/site/content/week08.json`, built by `scripts/build_site.py`).

## Running and queued jobs (iris only; Karen 2026-11-07: max 6 GPUs incl. the interactive session; 18359363 timed out 18:15, now 18377277)

Comparable evals: libero_10 task 8, val states 9 12 13 17 25 32 40 42 44 48 × 33, env seed 7, matched noise seed 1.
Base under these settings = **0.524** (18355478). `runs/` = `/iris/u/ankile/karenvo/libero/runs/`.

| lane | jobs | what (2026-11-07 log entry "items 1-4") |
|---|---|---|
| speed pre-training | 18363753 (stage 1, preempted 6×, resumes) → 18363754 check → 18363755 stage 2 → 18363756 check → 18364109 eval | prompt route + our 127-symbol discrete loss, tag pre-learned on the speed tag of all LIBERO demos |
| 1 (item 1) | 18376075 eval R1 arm A (matched noise) **done: 0.555** → 18376076 **untagged all-rollout control** (preempted at step 0 20:03, requeued) (929 eps + demos, `--tag-dropout 1.0`) → 18376077 eval | is the −8 to −12 pt drop of the tagged runs the failure rollouts? |
| 2 (item 2) | 18376078 **weighted arm** (`resnet_valuebinned_weights`, exp(clip(z,±2))) → 18376079 eval | advantage weighting instead of a tag |
| 3 (item 3) | 18376106-10 TD-target OOF ResNet folds (TITAN) → 18376129 labels + offline gate → 18376130 AdaRMS run on the chosen labels → 18376131 check → 18376132 eval | better labels: fitted n-step TD critic / V ensembles; the chain stops at the gate (exit 3) if no candidate beats the current labels' action information |
| 4 (item 4) | 18376189 selftest **passed** (parity 256/256, 0 truncated; FAST CE at init 16.6 vs uniform 7.6) → 18376190 (running, step 1400/5000 at 20:25) **RECAP-complete run** (prompt + real FAST BPE + KI + subtask text + vb40 + dropout 0.3 + demos) → 18376191 check → 18376192 eval (`FT_SUBTASK=1`) | RECAP's missing pieces; round 2 built (`scripts/recap_round2.sh`), not run |

## Contrastive guidance on the AdaRMS checkpoint (queued 2026-11-09 ~18:00, Karen: set up + test)

Question: does amplifying the learned tag difference at inference move success? Advantage Guidance (Zhou & Levine,
arXiv 2610.19783) Eq. 4: v = v_none + w (v_pos - v_neg) [paper]; v_none = the tag-dropout branch of the same net [ours;
paper uses a separate reference]. Run = AdaRMS vb40 18355677 (the only real-label run whose tag moves actions: 13.6%
of noise), trained on all rollouts + demos; reference + tag 0.409 (18355679), same settings (matched noise seed 1).
Code: `TagPi0.mix` + `set_mix` / `contrastive_mix` / `--contrast` in `scripts/pi05_tag_adarms.py`, `FT_CONTRAST` and
`MODE=contrastcheck` in its sbatch, offline check `scripts/pi05_contrast_check.py` (self-check: one-hot mixes = plain
+/-/none sampling; then shift/noise per w). 251 tests pass.
- 18409258 check (w 1 2 4 8 -4; `runs/contrast_check_adarms_vb40`), afterany peer noise-sweep 18408589
- lane A (afterok check): 18409259 none (w = 0) -> 18409260 w 2 -> 18409261 w 8
- lane B (afterok check, afterany 18408591): 18409262 w 4 -> 18409263 w -4 (sign control: should not beat w 0)
- **Check done (19:26): self-check exact (err 0.0 for +/−/none).** Shift/noise vs none: w 1 0.137, 2 0.295, 4 0.574,
  8 0.992, −4 0.438 (linear, ~0.14 per w; plain + vs − 0.120). No blow-up (max |raw| ≤ 1.13 vs 1.01 none); gripper flips
  vs none 0.4/1.0/2.2/3.8% at w 1/2/4/8 (below the 4.3% between two noise draws). So w 8 moves actions by one noise SD.
  Evals 18409259 / 18409262 pending on Priority at 19:30 (4 GPUs in use, these make 6).
- Read: paired vs none 18409259 and vs + tag 18355679. A dose-response (w 2 < 4 < 8 rising, -4 falling) = the tag carries
  usable signal; flat = the learned difference is noise. Caveat: this checkpoint sits ~11 pts below base (failure data).

## Morning read 2026-11-09 (overnight results)

Paired CIs vs base 18355478 (pairs = state × repeat, bootstrap 5000).
- **Grid @2000 (done):** d50 orig 18396164 **0.430** (142/330), −0.094 [−0.164, −0.024]; d75 orig 18396168 **0.415**
  (137/330), −0.109 [−0.176, −0.042]. Both a bit above ref 0.391 but still below base. Partial at ~09:00: d75 low @2000
  18396176 161/299 = 0.54 (the first arm near base); d50 orig @final 90/187 = 0.48; d75 orig @final 40/79.
  d50 low 18396170 and d02 low 18396975 still training.
- **Weighted success-only eval 18387308: 0.567** (187/330), +0.042 [−0.030, +0.115] vs base (unread until now).
- **Coach, 4 label rules (18394844):** rules barely differ (shift 0.016–0.018 round 1, 0.0185–0.0194 explore; mse
  gain 0.003–0.015). Explore is above round 1 (vb40 0.0192 [0.0188, 0.0197] vs 0.0175 [0.0170, 0.0180]) but PC1 rose
  as much (0.050 → 0.059): explore actions are noisier, so label/PC1 stays ≈ 0.33–0.35. Read: variety helps only marginally.
- **Explore curve (18394908), pooled 112/225/450/900:** shift over random 0.0144 / 0.0158 / 0.0139 / 0.0136 (flat);
  action info 0.060 / 0.064 / 0.083 / 0.079 (flat); mse gain over random 0.011 / 0.015 / 0.016 / 0.026 (vb40;
  single draw at 900, sd ≈ 0.01 at 450). With the old-data curve: **flat → more of the same data won't raise the tag signal.**
- **Test 2, failure branching (18395412/13/14, 12 eps, K 12, 299 points):** A* split-half 0.02 / 0.43 / 0.47 (all < the
  0.5 bar), V* 0.90 / 0.75 / 0.92. vb40 labels vs A* corr 0.03 / 0.27 / 0.02; + share at A*>0 vs A*<0: 0.40 vs 0.33,
  0.24 vs 0.11, 0.48 vs 0.29 (right direction, weak). OOF V vs V* 0.09 / −0.05 / 0.31. Replay outcome = recorded in
  only 9/12 eps (state diff up to 0.098), and 3 failure eps have V* = 1 at the last point. **Read: A* still mostly
  noise at K 12 → branching corrections unlikely to give a usable tag signal.** Pooled analysis not built.
- **Test 1 rollouts done:** more_plain a 113/225, b 130/225 (0.54), 0 errors. Not tabled; the curve no longer needs it.
- **d75 low @2000 (18396176) done: 0.536, +0.012 [−0.048, +0.076] → passes** (first recipe to hold base).
- **Follow-up auto-queued (Karen 2026-11-09: set it up):** decision job **18401829** (CPU, afterany the d50-low, d75-low
  final and d02-low evals) runs `scripts/step1_followup.py`: picks the best passing low-lr checkpoint (rule above) and
  submits 3 lanes on its recipe, 5000 steps, evals @2000 + @final: `fu_<arm>_seed1_s1` (untagged, seed 1),
  `fu_<arm>_tag_s0` (tag dropout 0.3, eval tag positive, + tag negative at the winner's step),
  `fu_<arm>_wsucc_s0` (weighted, 516 train successes, `resnet_valuebinned_weights_succ`, demos weight 1).
  No pass → nothing submitted. Job ids → `reports/2026-11-09-step1-followup/decision.json`.
- **Finals in:** d50 orig **0.515** (−0.009 [−0.076, +0.061], passes); d75 low 0.470 (−0.055 [−0.118, +0.012]);
  d75 orig 0.455 (−0.070 [−0.133, −0.006]). Original-lr arms added to the decision's candidates (rule is per recipe).
- **Released early (checkpoints already saved):** d50 low @2000 18396172, d02 low @2000 18396976, ref armU @2000
  18396978 (dependencies cleared). Finals re-pointed to their training: 18396173 afterok 18396170, 18396977 afterok
  18396975. Peak = 6 GPUs incl. the interactive session.
- **The two early @2000 evals failed (my bug, fixed 12:10):** 18396172 / 18396976 died in 1.4 min, "target dict contains
  keys {'img'}": `load_finetuned` read freeze_siglip from results.json, which is only written at the end of training,
  so a mid-training eval assumed an unfrozen SigLIP (arm/tag default and seed were also lost). Fix: trainer writes
  `args.json` at start; `pi05_policy_finetune.run_args` (results.json, else args.json, else a clear error) used by
  `load_finetuned` and `serve_pi05_finetuned.res_seed / uses_subtask`. 251 tests pass. Resubmitted: d02 low @2000
  **18404064**, d50 low @2000 **18404065** (afterok 18396170); decision 18401829 now afterany 18404064:18404065:
  18396173:18396977; `step1_followup.py` skips eval dirs without episodes.csv.
- **Week 9 page built** (`reports/site/content/week09.json` + `week09_data.json` from `scripts/site_week09_data.py`;
  `build_site.py week9`; charts.js `ciPlot` / `lineChart`). Rebuild after evals finish: run the data script, then
  `build_site.py`. **Not uploaded yet: gcloud has no active account in this session.**
- **Discrepancies found building it:** (1) AdaRMS real-label run 18355677 *does* respond to its tag: 0.0060 raw
  (13.6% of noise, 40% of the 0.015 ideal shift), loss 1% lower with the correct tag; prompt routes 0.0008. Never
  evaluated with the − tag. (2) Data-curve rule used the ideal shift, which is flat even for PC1; the right metric is
  action information (labels plateau ~450 eps, PC1 keeps rising). (3) Prompt-route models move more when the tag is
  removed (0.0045) than flipped (0.0008). (4) 25% of branch points restored inexactly (state diff > 1e-3); results
  unchanged without them (A* split-half 0.33, label corr 0.14). Pooled branching: A* split-half 0.33 (0.50 full K).

## 2026-11-08 20:45 (Karen: go on step 1 + build A)

- **Speed pre-training eval 18364109: 157/330 = 0.476** (base 0.524, −4.8 pts); tag check already showed the tag unused.
- **Step 1, RECAP label rules:** 18390934 cancelled and replaced by **18394844** (same afterok 18390932:18390933),
  `scripts/coach_explore_rules.sh`: labels `explore_vb40` (ours), `explore_thr40` (one threshold on A, RECAP Eq. 5),
  `explore_vb40_round` / `explore_thr40_round` (40% cut within each round; thr40_round = RECAP exact) → coach test per set
  → `reports/2026-11-08-explore-pilot/coach_explore*/`. Code: `labels_from_v(binned=, groups=)`,
  `libero_td_labels.py oof --rule thr --share-by round` (+1 test).
- **A, data curve:** `libero_tag_ideal_shift.py --table ... --curve N.. --curve-seeds k` (episodes spread evenly over
  start states; new column `tag_mse_gain` = held-out action-error drop from knowing the true tag, read against random
  labels, because the ideal shift alone can be inflated by overfitting on small data). `scripts/coach_curve.sh`:
  **18394907** old 450 fixed (N 56/112/225/450 × 5) → `reports/2026-11-08-coach-curve/old_fixed/`;
  **18394908** (afterok 18394844) explore table, per round and pooled to 900, labels explore_vb40 + explore_thr40_round.
  Reading [set before]: label excess over random (shift and mse gain) still rising from 225 → 450 (→ 900 pooled) =
  data-limited, more data should help; flat = more of the same data won't.
- V training data confirmed rollouts only (moka tables hold no demos).
- Folds reach best inner-val CE at step 2000–3000 of 8000; cap future refits at 4000 steps (halves cost).

## Step-1 recipe grid queued 2026-11-08 21:45 (Karen: set it up) + first A result

- **A, old data (18394907, done):** label excess over random, by episodes 56/112/225/450: shift 0.0122 / 0.0140 /
  0.0132 / 0.0120; held-out error drop 0.002 / 0.015 / 0.008 / 0.016; action info 0.040 / 0.062 / 0.070 / 0.074.
  The PC1 control keeps growing (action info 0.029 → 0.171), so the test can see data growth when the signal
  is real. **Reading [set before]: flat → more of the same data won't raise the label signal.** V fixed (trained
  on all 900), curve only to 450; the pooled-to-900 explore curve (18394908) is the check. `reports/2026-11-08-coach-curve/old_fixed/`.
- **Grid** (`scripts/step1_grid.sh`, all untagged: `--arm advantage --tag-dropout 1.0`, 900 rollouts + 29 moka demos,
  frozen SigLIP, 5000 steps, seed 0; ref 18376076 = 2% demos, original lr: 0.391). New `--demo-share p`
  (`fd.oversample_demos`: each demo listed k times; 0.5 → k 35, 0.75 → k 106; +1 test, 251 pass). Low lr = R2 (LoRA
  1e-5, rest 2.5e-6). Each lane: train → eval @2000 → @final (FT_TAG none, matched noise 1, val states × 33).
  @1000 evals dropped (Karen 23:15; 18396163/67/71/75 cancelled, @2000 evals re-pointed afterok:train).
  - d50 orig 18396162 → 64 / 65 (after 18395411); d75 orig 18396166 → 68 / 69 (after 18395412);
    d50 low 18396170 → 72 / 73 (after 18395413); d75 low 18396174 → 76 / 77 (after 18395414);
    d02 low (each demo once + low lr) 18396975 → 18396976 / 77 (after 18394908); reference 18376076 @2000: 18396978.
    (The last arm went in at 23:21 via `scripts/step1_grid_rest.sh` after the iris QOS limit, 25 queued / user.)
  - `--exclude` now also drops `iris-hgx-[1-2]` in libero_eval.sbatch and libero_branch_value.sbatch (and on the
    queued grid evals): neither needs the shared H100/H200s (branching runs on 20 GB A4500s, evals on 24 GB cards).
  - Pass [set before]: a recipe whose paired CI vs base 0.524 includes 0 or is above it; then repeat it with seed 1.

## Overnight 2026-11-08 21:00 (Karen: queue tests 1 and 2), chained to stay at <= 6 GPUs

- **E1 analysed at last** (18355241, Nov 6; `runs/branch_value_libero_10_18355241/analysis.json`): 6 eps, K 6, 143
  points. V* split-half reliability 0.84, A* (true n=20 advantage) only 0.30; labels vs A*: corr 0.08 (vb40), OOF A
  vs A* −0.01; OOF V vs V* 0.17 (vs G* 0.54). Failures decided in jumps (−0.83 at t 300 ep 165, −0.5 at t 140 ep 336).
  With the steer probe (best-of-8 chunk oracle gain +0.006, also null at x1.5 / x2.0): branching corrections only
  make sense at decision moments, with many continuations.
- **Test 1, more plain rollouts:** 18395410 (states 0–22, noise seed 11) + 18395411 (states 23–47, seed 12,
  afterany 18394907); 30 fixed train states x 15 = 450 eps, base via BASE_OURS, env seed 23, split `more`, raw saved,
  run dirs `runs/more_plain_{a,b}_libero_10_<job>`. Use: next point of the A curve (450 → 900 same-kind fixed eps;
  needs a table + 5-fold OOF V refit at 4000 steps) and training data if A is still rising.
- **Test 2, failure branching (E1, larger):** K 12, stride 20, 12 failure eps (one per state, state success
  0.33–0.80, all 520-step timeouts): 18395412 (189 341 22 349, afterany 18390932), 18395413 (76 21 29 183, afterany
  18387308), 18395414 (200 7 227 280, afterany 18395410). Run dirs `runs/branch_fail_{a,b,c}_libero_10_<job>`.
  `libero_branch_analyze.py` takes one run: analyse each, then pool (merge not built). Reading [set before]: A*
  reliable (split-half >= ~0.5) and large at the V* drops → build branching corrections there; A* still noise →
  branching unlikely to add a usable tag signal.

## Queued 2026-11-08 11:30 (Karen: queue both): success-only diagnostics, NOT RECAP

Same 516 train successes as R1 arm A (`subsets/train_successes.txt`), R1 recipe (frozen SigLIP, original lr), no
demos, seed 0; only change vs arm A = labels used as weights or tag. Compare with R1 arm A 0.555 (18376075).
- 18387031 **weighted, successes only** (`labels/resnet_valuebinned_weights_succ`: the all-data weights rescaled to
  mean 1 on these frames, ×1/1.146 [ours]) → eval 18387032 (tag none)
- 18387033 **tagged, successes only** (prompt route, vb40 labels: 51.5% of these frames positive, dropout 0.3) →
  eval 18387034 (+ tag)
Reading: either beats arm A → the labels carry usable information and the failure data is the obstacle; neither →
the labels are the bottleneck.

## Data-variety pilot queued 2026-11-08 (Karen: option A without B, pilot first)

**Result (13:55): ×2.0 qualifies; awaiting Karen's go on the 450-episode collection.** Success control 0.644,
×1.5 0.489 (−0.156, CI [−0.27, −0.04]), ×2.0 0.567 (−0.078, CI [−0.20, +0.06]); early arm spread ratio 1.23 / 1.64.
Server applies the scale (spread rises with it; ×2.0 noisy chunks flip the gripper 0.081 vs 0.046 per chunk).
Non-monotone success is likely n=90 noise. Report `reports/2026-11-08-explore-pilot/`.
**Collection queued 14:40 (Karen: go at ×2.0):** 18390881 (states 0–16) + 18390883 (states 18–47), 15 states ×
15 repeats each = 450 episodes, p 0.5, ×2.0, env seed 22, explore seed 1 (fresh vs the pilot), split name
`explore`, run dirs `runs/explore_x2p0_{a,b}_libero_10_<job>`. Next: table → 5-fold OOF ResNet V → vb40 labels →
coach test. **Collection done 16:25:** 450/450 episodes, success 0.609 (a 0.542, b 0.676), no client errors. **Chain queued (afterok):** table + pack 18390928 (`tables/moka_explore`, round 2 = explore) →
OOF ResNet V folds 18390929/30 → 31/32 → 33 (2 lanes) → labels `explore_vb40` + coach test 18390934 →
`reports/2026-11-08-explore-pilot/coach_explore/` (`--table` mode, groups = round within the fixed layout: old 450
fixed-state rollouts vs the 450 explore ones, same V and rule). Old-data reference in the same mode:
`coach_ref_old_fixed/` = 0.0160 [0.0156, 0.0164] (random 0.0040, PC1 control 0.0491). **Reading [set before]:**
explore ideal shift CI clearly above the round-1 group's → the variety helps, consider policy training (ask
Karen); about equal → noise doesn't add useful variety, consider option C (replay from saved sim states).
Speed stage 2 (18363755) was requeued and restarted 13:20, so its eval 18364109 lands later than ~15:00.

Base pi05_libero, 30 fixed train states x 3 repeats, env seed 21, raw saved, all arms on `serve_pi05_finetuned.py
--base` (BASE_OURS=1): control 18388271 → ×1.5 18388272 → ×2.0 18388273 (one GPU, chained, ~30 min each).
Exploration = each new chunk with p 0.5 sampled from start noise × s (`libero_rollouts.py --args.explore-prob /
--args.explore-scale / --args.explore-seed`, raw `noisy`, run-dir explore.json; +3 tests, 248 pass). Analysis:
`.venv/bin/python scripts/libero_explore_pilot.py --runs control=<dir> x1.5=<dir> x2.0=<dir> --out
reports/2026-11-08-explore-pilot`. Rule [set before]: largest scale with success ≥ control − 10 pts (point) and
early (first 60 steps) arm-action spread across repeats ≥ 1.5× control; none → stop, consider option C (replay
from saved sim states). Then (Karen's go): 450 noisy episodes on the 30 fixed states → table + 5-fold OOF ResNet V +
vb40 labels → coach test on the new table (needs `libero_tag_ideal_shift.py --table`, not built) → policy runs only
if the ideal shift clearly rises above 0.0151.

## Coach test queued 2026-11-08 11:45 (Karen: set it up) — is the value function the bottleneck?

**Result (18387077, 11:13): V is not the bottleneck.** Hindsight 0.0091 (ensemble baseline 0.0093), 4 s 0.0102,
8 s 0.0094, outcome 0.0112, all below the current 1 s labels 0.0151. Report `reports/2026-11-08-coach-test/`.
Success-only runs: first submission (18387031/33) failed at start (comment line in the episode list, my bug, fixed);
resubmitted as weighted 18387307 → eval 18387308, tagged 18387309 → eval 18387310 (dead evals cancelled).
**Tagged success-only eval (18387310, 18:17): 0.561** (185/330); vs base +0.036 [−0.027, +0.097], vs R1 arm A
+0.006 [−0.052, +0.070]: adding the tag to success-only training changes nothing (the tag is ignored, as before).
**Speed stage 2 tag check (18363756): tag still not used** (action shift 0.015× noise, loss correct = flipped);
eval 18364109 running.

18387077 (TITAN, ~15 min): `libero_tag_ideal_shift.py --candidates` (no gate) → `reports/2026-11-08-coach-test/ideal_shift/`.
Candidates (`scripts/libero_td_labels.py hindsight`, `vl.libero_td.hindsight_labels`, test in `tests/test_item3_item4.py`):
`hind_v0_vb40` A = realised return G_t − V0(t) (what actually happened replaces V's guess; 1% of failure frames +,
76% of success frames +; κ vs current 0.21), `hind_ens_vb40` (ensemble baseline), `look_n80_vb40` / `look_n160_vb40`
(current rule at 4 s / 8 s), `oracle_outcome` (per episode; earlier 0.0112). Current labels 0.0151, random 0.0039,
planted positive control 0.0485 (25% flipped 0.0203). **Reading set before the result:** hindsight clearly above
0.0151 (CI above) → a better V could help, up to that ceiling; about equal or below → V is not the bottleneck,
the data (one policy, little action variety) is. Caveat: hindsight spreads credit over the whole episode (it is
mostly the outcome), so it is the ceiling for "knowing the future", not for credit assignment; n80/n160 show the
local versions.

## Latest verdicts (2026-11-07)

- **2026-11-08 01:40.** Item 2 weighted arm (18376079): 173/330 = **0.524 = base 0.524**, +0.000 [−0.064, +0.061];
  vs R1 arm A (0.555) −0.030. Weighting with the same labels removes the tagged runs' loss but adds no gain.
  Per scene base → W: 9 .64/.73, 12 .12/.27, 13 .18/.36, 17 .61/.30, 25 .79/.88, 32 .61/.52, 40 .91/.70, 42 .12/.30,
  44 .70/.42, 48 .58/.76.
- **2026-11-08 06:15. Item 1 untagged all-rollout control** (18376077, 330/330, resumed after preemption): 129/330 =
  **0.391** vs base 0.524, −0.133 [−0.197, −0.076]; vs R1 arm A 0.555 −0.164 [−0.221, −0.106]. Training on all 929
  rollouts (incl. 384 failures) + demos with no tag costs ~13 pts: the failure data, not the tag, is the drop. Against
  this control, every tagged run is at or above it (AdaRMS 0.409, prompt + discrete 0.445, RECAP-complete 0.458:
  +0.067 [+0.003, +0.130]), and weighting 0.524 is +0.133 [+0.076, +0.191]. Reading (to check, one seed each): the
  tag recovers part of the cost of the failure data despite the small tag effect at test time; weighting recovers
  all of it but no more than success-only R1.
- **2026-11-08 02:40.** Item 4 RECAP-complete eval (18376192, + tag, oracle subtask): 151/330 = **0.458** vs base
  0.524, −0.067 [−0.133, 0.000]; vs R1 arm A 0.555 −0.097 [−0.161, −0.033]. Between AdaRMS 0.409 and prompt +
  discrete 0.445 tagged runs: FAST + subtask text did not change the picture. Scene 9 .64 → .24. Round 2 not
  warranted (gate: match base). Item 1 run 18376076 preempted again, resumed at step 4000 (eval ~05:45).
- Item 3 gate (18376129) **failed (exit 3)**: ideal shift current vb40 0.0151 [0.0149, 0.0154]; td_vb40 0.0131,
  ens_vb40 0.0121, td_ens_vb40 0.0112 (random 40% 0.0039). TD V corr with V0 0.91. No new labels carry more action
  information; 18376130-32 left pending with DependencyNeverSatisfied (cancel needs Karen's OK).
- Item 4 tag check (18376191): tag pos vs neg 0.0011, 1.8% of noise at final (2.5% @1000); loss correct ≈ flipped.
  The RECAP-complete run ignores its tag too. Eval 18376192 running.
- Speed stage 1 tag check (18363754, speed_p30 labels on demos): pos vs neg 0.0009, 3.3% of noise [2.7, 4.0];
  loss correct ≈ flipped. **The speed tag was not learned in stage 1** (δ 0.02 control: 0.34× noise), so stage 2
  (18363755, running) starts without a pre-learned tag.

- **R1 arm A, matched noise** (18376075): 183/330 = 0.555 vs base 0.524, +0.030 [−0.036, +0.088] (scene-stratified
  paired bootstrap, 4000). Success-only R1 does not hurt under the current eval settings, so the tagged runs' drop
  (0.409 / 0.445) is not the recipe; lane 1's untagged all-rollout control tests whether it is the failure data.
  Per scene base → R1: 9 .64/.88, 12 .12/.24, 13 .18/.42, 17 .61/.42, 25 .79/.97, 32 .61/.64, 40 .91/.61, 42 .12/.52,
  44 .70/.27, 48 .58/.58 (large per-scene swings both ways).
- Lane 4 selftest 18376189: the official-FAST parity holds; pi05_libero has no prior FAST knowledge (CE at init
  16.6, above uniform 7.6), so RECAP's "tokenizer the model already knows" is not reproduced by switching to FAST.

- Real RECAP labels (value-binned 40%), R1 recipe, + tag at eval: **AdaRMS 0.409** (−0.115 [−0.185, −0.045] vs base
  paired) and **prompt + discrete 0.445** (−0.079 [−0.148, −0.009]). Both worse than base.
- Planted-tag controls: prompt, AdaRMS and prompt + discrete all carry δ = 0.1 (1.5× noise); prompt and AdaRMS learn
  δ = 0.02 fully (0.020-0.022, 0.34× noise). Real labels: AdaRMS 0.007, discrete 0.001. The routes work; the labels
  carry little action information (ideal-shift upper bound 0.015, mostly gripper).
- Correction (2026-11-07): the tagged runs used R1 (frozen SigLIP, original lr), which passed the bar for arm A
  (0.521 vs 0.564, seed-0 noise). The difference from arm A is the data (all rollouts incl. 384 failures + demos),
  not the lr: lane 1 tests this.

## Resume steps

1. Lane 3 gate result: `reports/2026-11-07-td-labels/ideal_shift/gate.json` (+ `ideal_shift_table.csv`). If it
   exits 3, 18376130-32 are cancelled by the dependency: report the candidates' numbers.
2. Lane 4 selftest 18376189: check (9) parity, (10) truncation, and (11) CE at init (does pi05_libero know FAST?).
3. When the evals land: paired comparison vs base 18355478 (same script as 2026-11-07 log), and vs each other.
4. Round 2 (`scripts/recap_round2.sh <run>`, dry run by default) only for a round-1 run that matches base.
5. Write `reports/2026-11-07-items-1-4/` when the lanes finish; update docs/findings.md.

## Earlier verdicts (2026-11-05)

- **R1 frozen-SigLIP recipe also passes the bar** (18310772): arm A 172/330 = 0.521 vs base 0.564, −0.042
  [−0.106, +0.024] (scene-stratified paired bootstrap, 4000). Same total as R2 but a different per-episode pattern
  (per scene R1/R2: 9 .91/.79, 12 .12/.21, 17 .30/.42, 25 .85/.67, 40 .67/.85, 42 .48/.21). Week 8 card `recipe-fixes`.
- **A2 N=4 seed 2** (18309646): SNR **0.191** / MAE 0.171 / AUC10 0.797. **3 seeds: SNR 0.184 ± 0.006, MAE 0.170**
  vs N=18 0.152 / 0.160 → steadier on every seed; still far below ResNet 0.247. N=9 has seed 0 only (0.158 / 0.162).
  *Corrected 2026-11-05 (reports/2026-11-05-layer-interventions/):* 0.152 / 0.160 is N=18 seed 0; its 3-seed mean is
  SNR 0.166 / MAE 0.170, and N=18 seed 2 (0.194) beats N=4 seed 2. N=4 − N=18 SNR +0.018, 95% CI [−0.020, +0.050]:
  not resolved. "change_snr_2s" is a 1 s change. **Offline 3-seed ensembles (mean of V, no new training):** N=18
  SNR 0.208 (noise 0.101/0.102/0.074 → 0.072), MAE 0.169; N=4 SNR 0.214, MAE 0.170. Much of the flicker is
  seed-specific (learned expert), not the shared backbone. **ResNet fixed+fresh 3-seed average** (18237895-7): SNR
  0.247 mean of singles → 0.278, noise 0.076 → 0.067, MAE 0.189, AUC10 0.770. After averaging, the noise is about
  equal (0.067 vs 0.072); ResNet's remaining SNR lead comes from a larger signal (≈0.019 vs 0.015), and π₀.₅ stays
  more accurate (MAE 0.169, AUC10 0.803). Point estimates, no CI yet.
  **Scene check (offline, 3-seed-avg V, val only):** a no-image "scene + clock" lookup (leave-one-episode-out mean
  target of the same val init state at the same frame) gets MAE 0.180 / AUC 0.802, better MAE than ResNet (0.189).
  V frozen at frame 0 + mean trend: π₀.₅ AUC 0.782 vs ResNet 0.698 (π₀.₅'s first look is much more informative);
  freezing costs both models MAE (π₀.₅ 0.169 → 0.228, ResNet 0.189 → 0.256). π₀.₅ AUC5 actual 0.741 < frozen 0.782.
  Reading: π₀.₅'s accuracy lead over ResNet is mostly setup recognition, which cancels in advantages (tags); ResNet
  reacts more to events. Script: scratchpad scene_check.py (not in repo).
- **Which cut the per-layer probe points to (not run):** in `pi05_layer_hidden.py`, h_l = the input to block l. The
  probe's single-layer head is flat 0.14–0.16 for h0–h15 and drops at h17 (0.122) / h18 (0.107). h16 was not
  extracted (probe: h0–5, odd layers to h17, h18). A truncated expert at N reads the K/V of blocks 0..N−1, i.e. h0..h_{N−1},
  so the probe's cut is **N = 16–17** (drop only the last 1–2 blocks). The 4/9 cuts came from SmolVLA's
  "first half" before the probe. The probe doesn't explain why N=4 > N=9 (single-layer heads: h3 0.153, h4 0.157, h9 0.144, all within ~1 sd):
  the truncation gain is in how the expert reads several layers, not in one steadier layer.
  Accuracy: "flat with depth" holds only for the single-layer probe head (MAE ~0.175 vs SigLIP 0.171). In the full
  value expert the LM adds accuracy: SigLIP-direct 0.215 (1 seed) → N=4 0.170 → N=9 0.162 → N=18 0.160.
- **R2 low-lr recipe passes the bar**: arm A 0.521 vs base 0.564, −0.042 [−0.112, +0.027] (18315374).
- **Tag predictability** (`reports/2026-11-05-tag-predictability/`): the action carries 3.2% of the current tag's
  information beyond the state (≈ the ignored oracle's 2.3%); RECAP's 30–40% positive → 7.4–8.1%, + value-binned
  calibration → 8.5%. **Value-binned calibration** (`reports/2026-11-05-value-calibration/`): κ +0.05–0.06, no
  subtask offset in our A. Candidate next policy run [to decide]: R2 recipe + value-binned 40% positive + 30% tag
  dropout + moka demos tagged positive, eval β = 1 / 1.5 / 2.5.

- **Literature review** (`reports/2026-11-05-literature-advantage-conditioning/`): RECAP uses ~30–40% positive,
  30% tag dropout, demos tagged positive, and learns the tag in pre-training; **arm C was 70% positive, no dropout,
  no demos**. Theory (RCSL): a tag not predictable from the action is ignored. Dissecting (2609.28161): value-binned
  calibration + continuous weights work best on 4 real tasks. Open decision: offline tag-informativeness test +
  calibration first, then a RECAP-faithful tag run and/or a weighting run.

- **Label denoising** (`reports/2026-11-05-label-denoise/`): current labels agree across ResNet / expert at κ 0.20.
  Smoothing V (k = 9) is the only remedy that lowers noise and keeps timing (flips 0.96 → 0.34/s); n = 80/160 lifts κ
  to 0.39/0.54 but smears credit 4/8 s before events; margins filter, do not denoise. Open with the rule decision.

- **Label rules** (`reports/2026-11-04-label-rules/`, no training): dropping the time cost (R3) = current rule (constant
  shift under a percentile). A > 0 (R1) / top 30% (R2) turn stalls negative (78–97% vs 26–38% now), but ordinary
  progress frames carry A noise ±0.04 around 0, so R1 labels them ~54/46. **Open decision (Karen):** which rule, or
  a noise-aware variant (smoothed V, longer n, margin / no-tag middle), for the next policy run. Viewer has a rule
  selector: `reports/2026-11-04-advantage-examples/viewer/`.

- **Advantage examples** (`reports/2026-11-04-advantage-examples/`): the ResNet's 201-bin V is bimodal (success /
  failure bands); A tracks the 20-step (1 s) change in its P(success) (r 0.89–1.00 on 5 episodes). Labels before a pot falls
  76–92% negative, before a successful first placement 19% (base 30%); expert near base at both. Exceptions:
  placement labelled negative (ep 139), band flicker (ep 87). Flat stretches are positive by default. The oracle
  tag (18310225, ignored: tag/noise 0.021) was per-episode, so it did not test per-frame conditioning. Tag-usage
  CI bug fixed (`boot_ratio_ci`). Karen to review before choosing the better tag tests.

- **A3 truncated N=4 + LoRA** (18309647, seed 0): SNR **0.169** / MAE 0.166 / AUC10 0.837 / milestone 0.007. Between
  LoRA-full (0.133 / 0.155) and frozen N=4 (0.180–0.182 / 0.169–0.170): truncation lifts LoRA's steadiness, LoRA
  gives back some of it for accuracy. Not steadier than frozen N=4; still far below ResNet 0.247. Milestone gone.
- **GPU placement (Karen, 2026-11-04 20:20): no RTX PRO 6000 (move5).** π₀.₅ jobs pinned to `gpu:l40s:1` (24 GB
  cards OOM); ResNet B1 jobs `gpu:1` excluding move5 (L40S / A5000 / TITAN). 18310224 and 18310773 were already
  running on move5 when this was asked. 18310224 finished 20:45; 18310773 was let run to step 4000 (Karen) and
  resumed on L40S as 18315373 (eval 18315374; 18309646 re-chained). Nothing on move5 now.

- **A2 N=4 seed 1** (18309645): SNR 0.182 / MAE 0.170, replicates seed 0 (0.180 / 0.169).

- **Fine-tuning hurts the policy** (`reports/2026-11-04-base-vs-finetuned/`, 18309278): base π₀.₅ 186/330 =
  **0.564** on the arms' exact trials; success-only −0.158 [−0.224, −0.091], + tag −0.155, − tag −0.133. Drops
  largest on easy scenes. Labels untested until a non-harmful recipe exists. Proposed: arm A recipe variants
  (frozen SigLIP; 10× lower lr; evaluate intermediate checkpoints) before any further label comparison.

- **The tagged policy ignores its tag** (`reports/2026-11-04-tag-usage/`, 18309419): + vs − tag moves arm C's
  action by 2% of its sampling noise (arm A, never tagged: 4%); flow-matching loss with correct vs flipped tag
  identical (0.00689 / 0.00689). No tag learned, so the eval says nothing about label quality yet. Proposed next:
  oracle-tag fine-tune (tag = rollout outcome) to separate "labels too weak" from "conditioning too weak".
- **PaliGemma vs π₀.₅ per layer** (18308029): PaliGemma SNR drifts 0.146 → 0.113 with depth; π₀.₅ flat ~0.15
  to h15 then 0.122 / 0.107. Robot training raised accuracy at every layer (MAE ~0.20 → ~0.18, AUC ~0.69 → ~0.81)
  and mid-layer steadiness (h11 0.119 → 0.159), and removed the top layers' milestone response (+0.05 → ~0).
  Top-layer noise is inherited, not caused by robot training.

- **Policy results: no gain from the advantage tag.** Success-only 18291522 134/330 = 0.406; advantage-tagged
  (ResNet OOF labels) + tag 0.409, − tag (control) 0.430. One seed, 330 held-out trials each.
- **π₀.₅ OOF labels are worse than the ResNet's** (labels/oof_expert_n20_p30): OOF MAE 0.182 vs 0.195, but
  within-scene AUC 0.68 vs 0.74; interior positive success/failure 72/66% (ResNet 73/66%); agreement 65% (chance
  57%), advantage corr 0.27. **No π₀.₅-label policy fine-tune** (decided 2026-11-04; reopen only if asked).
- **B2 steering stopped at step 2** (`reports/2026-11-04-steer-probe/`): π₀.₅'s 8 candidates end 1.9 mm apart
  (arm moves 22.7 mm per chunk); cross-validated oracle gain +0.006 [−0.028, 0.042].
- **Per-layer probe (fixed+fresh, 3 seeds):** SNR flat 0.14–0.16 for h0–h15, drops at h17 0.122 / h18 0.107 /
  final 0.116; milestone vanishes at h17+; MAE ~0.175 at every depth; AUC10 best mid-stack (h13 0.837). h1–h3
  within 2 sd of h0 → image-only-attention ablation not triggered. On 375 fixed (18306065): h0 0.107 ± 0.023,
  h1–h17 0.079–0.094, final 0.055 — P1b's "fall from layer 1" was ~1 sd noise on the low-variety set.
- **A2 truncated value expert, seed 0** (`reports/2026-11-04-truncated-value-expert/`): N=18 SNR 0.152 / MAE
  0.160; N=9 0.158 / 0.162; N=4 **0.180** / 0.169 (noise 0.101 → 0.092 → 0.085). N=4 is the steadiest π₀.₅ V
  so far but inside the 0.166 ± 0.025 bar on one seed; ResNet still 0.247.
- P1c self-check fixed 2026-11-04 (pooled rel L2 gate); PaliGemma runs skip the Stage 3a pool comparison (it
  compares with π₀.₅-weight pools, 2.67 by design).
- Earlier: ResNet vs π₀.₅ trade-off (π₀.₅ more accurate via scene difficulty, ResNet 1.5–2× steadier); two noise
  sources (frozen LLM layers, SigLIP fine-tuning); SigLIP-only V closed; E2 subtask head no SNR change.

## Open decisions for Karen

1. A2 follow-ups (N=4 seeds 1–2 done): N=9 seeds 1–2 (not run), and/or the probe-implied **N = 16–17** cut
   (`pi05_value_expert.sbatch`, `--mem=96G`, ~3 h each on L40S). Tests whether dropping just the top blocks
   recovers N=18's accuracy (MAE 0.160) with less flicker.
2. Policy "no gain → diagnose" (meeting step 5): labels are weak for both backbones (~6–7 pt success/failure gap).
   Options: tag usage at test time, a stronger label rule (longer lookahead / stricter percentile), or train-all
   labels (paused).

## Mentor meeting 2026-11-03: agreed next steps

Notes: Granola transcript (link in the meeting chat). The notes' "CQ" = SeeQ, "SmallVLA" = SmolVLA.

1. **DONE: verify episode length in environment steps** (≈20 s real: LIBERO steps at 20 Hz, videos are written at 10 fps; every "seconds" label in our scripts is 2× too long, metrics unchanged; Week 8 card). Original notes: What we know: `frame_index` runs to 516–519 and `T` ≤ 519, which
   matches `MAX_STEPS["libero_10"] = 520` in `scripts/libero_rollouts.py:44`, so one frame = one policy/env step.
   Every "seconds" number assumes **10 Hz** (`libero_oof_labels.py --n 20` = "2 s"; change SNR lag 5 grid frames of 4 =
   "2 s"; episodes "~52 s"). robosuite's default `control_freq` is 20 Hz. To check: print `env.env.control_freq` (or
   the LIBERO env's `control_freq` kwarg) inside `libero_rollouts.py` once; also check the recorded video fps. If it
   is 20 Hz, every time label halves (2 s → 1 s); the metrics themselves are unchanged, only their units.
2. **(Paused) the "train on all / label all" advantage run** (comparison for the 5-fold out-of-fold labels; not a priority while the out-of-fold labels run).
   - Train a ResNet V on all 900 train rollouts (same recipe as the fixed+fresh ResNet, `vl.libero_resnet`, no held-out
     fold; keep the 5 inner states for early stopping), predict every train frame, then label with the same rule
     (`scripts/libero_oof_labels.py`: n = 20, 30th percentile → 70% positive). Easiest: add a `--source resnet_all`
     pattern to `PATTERN` in that script.
   - Before any policy fine-tune, compare the two label sets: positive fraction, within-scene AUC (OOF: 0.74), and
     agreement. **Degeneracy check:** if in-sample V has memorised outcomes, advantages collapse (nearly all positive
     on successes, negative on failures; AUC near 1.0 against outcome with little within-episode variation).
   - Then an advantage-tagged fine-tune with those labels (third policy arm). Needs one large GPU (L40S / RTX PRO
     6000); start it when the current advantage-tagged fine-tune ends to stay ≤ 4.
   - Mentor's three options for reference: (1) train all / label all, simplest, risk of degenerate advantages;
     (2) new held-out rollout set, cleanest, 2× data; (3) 5-fold out-of-fold, most principled (running).
3. **DONE on fixed+fresh (fixed-only analysis 18306065 running): Revisit the LLM final-layer ablation details** (P1b, `scripts/libero_p1b_prepare.py`,
   `reports/2026-11-02-frozen-encoders/report.md`). What was measured: same small head on frozen features, 375
   fixed-layout episodes, 3 seeds: SigLIP 0.122; LLM layer 1 / 5 / 9 / 13 / 17: 0.093 / 0.098 / 0.089 / 0.080 / 0.060;
   final 0.059. Points to be ready to explain: (a) the 375-fixed set has the least layout variety, so absolute SNR is
   low everywhere (frozen SigLIP 0.194 on 375 fresh); compare within the set only; (b) the drop starts at layer 1
   (−25%) and is steepest after layer 13; (c) SNR only, not accuracy per layer. 18298946 redoes all 19 layers on
   fixed+fresh: analyze with the command in "Resume steps", and report **MAE / AUC per layer next to SNR** (the
   point is to find a layer with π₀.₅'s accuracy and less flicker).
4. **DONE: scene-calibration scatter plot** (Week 8: all backbones rank scenes alike, r ≈ 0.66; π₀.₅'s early V spreads ~2× wider, slope vs target 0.49–0.57 vs 0.23, which is its accuracy edge). Original plan: (the "π₀.₅ recognises hard scenes" claim). Val = 120 episodes over 10 scenes
   (12 rollouts each). Per scene: x = actual success rate (of 12), y = mean predicted V over the first 2 s (and
   optionally the whole episode); one panel or colour per backbone (ResNet 18237895 `ev_resnet`, frozen π₀.₅ 18237898
   `ev_expert`, LoRA 18258391 `ev_expert`); report Pearson/Spearman r per backbone. Only 10 points per backbone:
   pool seeds 0–2 for the 3-seed backbones and show seed spread. CPU only. Add to the Week 8 site.
5. **DONE (no gain, see verdicts): Milestone:** rollout results for the ResNet advantage-tagged policy vs the success-only baseline. If no gain,
   diagnose (label quality, tag usage at test time, negative-tag control) before scaling up.

## New directions to explore

### A. Which language-model layers does a value function need? (SmolVLA / SmolVLM)

Papers: SmolVLM (arXiv 2504.05299), small VLMs with aggressive image-token compression; SmolVLA (arXiv 2506.01844)
builds a VLA on SmolVLM-2 and **drops the top half of the language model**: the action expert reads only the first
N = L/2 layers. Their LIBERO ablation (Table 8, avg success): N = 8: 75.0, 16: 78.5 (chosen), 24: 79.5, 32: 80.3;
skip every 2nd layer 75.5; smaller 256M VLM 75.8. They also cut images to 64 visual tokens/frame (pixel shuffle, no
tiling) and keep the VLM frozen. Mentor's point: going from SmolVLA back to SmolVLM, the upper LM layers mostly serve
language generation, and control (or value) may not need them. That fits our finding that steadiness falls through
π₀.₅'s layers.

How our model maps on: `scripts/pi05_value_expert.py` grafts an 18-layer value expert that attends to **every**
layer's K/V of frozen π₀.₅'s PaliGemma (the same way π₀.₅'s action expert does), so the value sees the noisy top
layers too.

Experiments, cheapest first:
- **A1 (done, see verdicts):** the per-layer probe 18298946, with MAE/AUC per layer (step 3 above). Tells which depth to cut at.
- **A2 (seed 0 done, see verdicts): truncated π₀.₅ value expert** (code in place: `pi05_value_expert.py --train --n-layers N`; slices the K/V cache, exact by causality, checked in `--selftest`). Run the prefix through only the first N PaliGemma layers and give the value
  expert N layers (it must match the backbone's layer count layer for layer). N ∈ {4, 9, 18 = current}, frozen,
  fixed+fresh 825 episodes, seed 0 first; score with `score(p, "ev_expert")`. Bar: SNR clearly above frozen π₀.₅'s
  0.166 ± 0.025 while MAE stays ≤ ~0.175. Also ~2× cheaper per step at N = 9. Code: a `--n-layers` flag that slices
  the `gemma.Module` config depth for both the backbone pass in `backbone_kv` and the expert.
- **A3: truncated + LoRA.** If A2 helps, repeat the LoRA run (18258391 recipe, `scripts/pi05_value_lora.py`) on the
  truncated model: LoRA had the best accuracy (0.155) but SNR 0.133.
- **A4: a small VLM as the value backbone.** SmolVLM-256M/500M (64 tokens/frame), first half of its LM, small value
  head, fully fine-tunable on one GPU. Tests "smaller models + limited data" directly and is a clean comparison with
  the ResNet and with SeeQ's finding that a VLM without pre-training ≈ ResNet.
- **A5 (optional): fewer visual tokens.** Pool π₀.₅'s 256 SigLIP tokens per image down to 64 before the LM.
  Speculative; only if A2 points at the image tokens.
- Check against 18286806 (same per-layer probe on pre-robot PaliGemma): did robot training make the upper layers
  noisy, or were they noisy already?

### B. SeeQ (arXiv 2609.22085): subtask-scoped Q-functions

SeeQ learns a Q-function for the **currently active subtask** (shorter horizon, easier TD backups), predicts the
subtask in text first and then its value, pre-trains on broad robot data, and uses it for best-of-N policy steering.
Ours differs: a Monte-Carlo V over the whole task, used for advantage conditioning. Our quick check found the subtask
is easy to read (~97%) but an auxiliary subtask head did not change steadiness. Still worth doing in full (mentor):
- **B1 (Karen: go 2026-11-04; queued 18313821/22, adjust after steering v2):** subtask-scoped V: the value of
  finishing the *current* subtask (place the first pot, then the second), conditioned on the subtask.
  - Code: `vl.libero_subtask.scoped_targets` (+ `tests/test_libero_scoped.py`), `vl.libero_resnet --subtask-scoped`.
    Boundary = the milestone frame m (`vl.libero_milestone.milestone`: first image with a pot on the stove that stays
    10 steps; monotone). Subtask-0 frames of episodes that reach m get −(m − t)/1040 (subtask done, no failure
    penalty); every other frame keeps the whole-task target (an episode that never reaches m failed subtask 0 and
    the task alike). Head = 2 × 201 logits, the true scope's half is read (SeeQ predicts the subtask; we have it
    exactly from the sim). [ours: Monte Carlo, not SeeQ's TD; SeeQ: subtask scope + conditioning.] On 30 episodes:
    28 reach m, 58% of frames in subtask 0, 51% retargeted.
  - Step 1 (queued): fixed+fresh, seeds 0–1, compare with ResNet ff 18237895–97. Metrics on the scoped target are
    not directly comparable to whole-task ones: compare change SNR / MAE within subtask 1 (same target in both) and
    the steadiness of subtask 0 against its own target; check the value switch at m is sharp.
  - Step 2 (if steadier): 5-fold OOF scoped V → scoped advantage labels (A within a subtask; when the boundary falls
    inside the n-step lookahead, bootstrap with 0 = subtask done, as SeeQ's no-cross-boundary backup) → compare
    with the ResNet labels (within-scene AUC 0.74). **Changing the labels: ask Karen first.**
- **B2: SeeQ-style Q-function + best-of-N steering** (Karen approved 2026-11-03; the notes' "CQ differences: uses
  Q-function, different policy extraction, likely a stronger method"). SeeQ does not train the value into the policy;
  at each replan it samples N candidate action chunks from the base policy, scores each with Q(o, subtask, chunk),
  and executes the best (policy weights unchanged). Our version:
  - action = one π₀.₅ chunk (the 5 executed actions; raw npz already stores `actions` + `new_chunk`, no new data);
  - Q input = frame + subtask + chunk (ResNet: image features + small action MLP into the same 201-bin head; π₀.₅
    later: chunk tokens next to the value expert's query); target = B1's subtask-scoped target, TD/SARSA with the
    next executed chunk (all 900 rollouts are π₀.₅'s own, so this is Q of π₀.₅ and best-of-N = one
    policy-improvement step);
  - extraction = a steering policy server wrapping the base server; same client, same 330 held-out trials.
  Steps (stop early if a check fails):
  **STOPPED at step 2 (2026-11-04, oracle gain ≈ 0).**
  1. **Candidate spread (job 18296012):** do π₀.₅'s N = 8 samples at a replan point differ? Arm actions (per-dim
     std units) and end-effector position after 5 steps (mm) vs how far the arm moves in 5 steps; gripper split.
  2. **Simulator headroom (same job):** at policy steps 40/130/220/310 of 10 base episodes, each of 8 candidates is
     executed from the saved state and the base policy continues to the end, 2× per candidate. Between-candidate
     variance of success (binomial noise removed) and a cross-validated oracle selection gain (pick on repeat 1,
     score on repeat 2). If the gain ≈ 0, a learned Q cannot steer → stop B2. After-chunk frames are saved so the
     existing ResNet V can be scored offline (can it rank candidates?).
     Analyze: `.venv/bin/python scripts/libero_steer_analyze.py --run /iris/u/ankile/karenvo/libero/runs/steer_probe_libero_10_18296012`.
     Code: `scripts/libero_steer_probe.py` (+ `.sbatch`), smoke-tested with `--args.fake-policy`. Restore is exact
     (3e-17) only because it also restores the Panda gripper's integrated command, the solver warm start and
     robosuite's step counter (without the gripper: 0.011); any steering code that branches the sim must do the same.
     Expected run time ~3–4 h (~25 sim steps/s).
  3. ResNet Q with the subtask target (TITAN OK); check Q changes when the action is shuffled across frames.
  4. Steering server + eval, N = 1 / 4 / 8 on the same 330 trials; later steer the advantage-tagged policy too.
  5. π₀.₅ Q, only if the ResNet version steers.
  Costs: N× inference per replan; offline Q with one action per state may ignore the action (step 3 check).
- **Try later (Karen, 2026-11-04): subtask-scoped Q trained on noisy-candidate data.** Only if steering v2's
  oracle gain is real (≥ ~+5 pts). Normal-noise rollouts have ~one action per state, so a Q would likely learn to
  ignore the action; rollouts that execute noise-scaled chunks (v2's best scale) give the action variety a Q needs.
  Then B1's target with a TD/SARSA backup, ResNet Q (image + chunk MLP), shuffled-action check, steering eval on
  the 330 trials. Cost: new rollouts (~900 episodes, GPU hours) + Q training; noisier candidates are also worse on
  average, so the gain must beat that.
- **Try later (Karen, 2026-11-08): AdaRMS + discrete, with the tag in both places.** Not run. As the code stands,
  `pi05_tag_adarms.py` strips the tag from the prompt (`vl.tag_route`) and puts it only into the action expert's
  AdaRMS conditioning. The discrete action-token loss (`--discrete-weight`) trains the backbone to predict the actions
  from the prompt, so combined as is it would never see the tag and would learn untagged action tokens. A real
  version keeps the tag text in the prompt (for the backbone + discrete loss) and also adds it to AdaRMS (for the
  expert): a small change in the tag-routing patch. Low priority: every tag check so far (prompt, AdaRMS,
  prompt + discrete, RECAP-complete, speed stage 1) shows the real tag moving the actions by only 2-3% of the
  sampling noise, so a second route for the same weak tag is unlikely to change that.
- Pre-training (SeeQ/RECAP style) stays on hold: infra is being arranged by the mentor and is not the bottleneck;
  keep pushing with small models and limited data.

## Traps

- π₀.₅ does not load in a 16 GiB interactive session (OOM). Always sbatch π₀.₅ jobs. A one-batch π₀.₅ check can run
  as a **CPU** sbatch (`JAX_PLATFORMS=cpu`, 96G, ~7 min; `scripts/debug/*.sbatch`) without using a GPU.
- `/iris/u/ankile` is 99% full: large features go to `/move/u/karenvo/libero_features/`.
- Never `scancel` other sessions' jobs; re-chain only your own.
- Site rebuild: `.venv/bin/python scripts/build_site.py`; upload with
  `/iris/u/ankile/karenvo/google-cloud-sdk/bin/gcloud storage cp -r --cache-control=no-cache reports/site/value-learning/* gs://zima-robot/reports/karenvo/value-learning/`.

## Noise sweep queued 2026-11-09 (mentor: "maybe 2x is still not enough"; Karen: set it up)

Pilot design reused (control 18388271 + x2.0 p0.5 18388273 as-is): base on our server, 30 train states x 3 repeats,
env seed 21, explore seed 0, raw saved. `scripts/noise_sweep.sh`: lane A x3.0 p0.5 **18408588** -> x8.0 p0.5
**18408589**; lane B x5.0 p0.5 **18408590** -> x2.0 every chunk **18408591**; analysis **18408592** (CPU, afterany)
-> `reports/2026-11-09-noise-sweep/`. Rule [set before, the pilot's]: success >= control - 10 pts (point) and early
arm spread >= 1.5x control; choice = largest spread among those; -10 to -15 pts flagged borderline. A qualifying
setting beyond x2.0 -> (Karen's go) 450 episodes -> table -> OOF V -> vb40 -> coach test, read by action info.
`libero_explore_pilot.py`: choice now by spread (name parse broke on x2.0_all), + borderline column. 251 tests pass.
Mentor feedback 2026-11-09 answered from existing data: base success per init state 12-91% (val, 33 each) and
13-93% (30 train states, 15 each), no state at 0; best-of-16 per state 0.99. Next after the sweep: rounds of
collect -> keep successes -> fine-tune on the step-1 winning recipe (needs Karen's go).
GPU chaining (session karenvo-15, 2026-11-09): its contrast check 18409258 waits afterany:18408589, then evals
18409259 -> 60 -> 61; 18409262 -> 63 wait afterany:18408591. New GPU jobs from this session go after 18409261 or
18409263. It edited scripts/pi05_tag_adarms.py / .sbatch (+ `--contrast`, MODE=contrastcheck) and added
scripts/pi05_contrast_check.py; the sweep and step-1 follow-ups don't use those files.
