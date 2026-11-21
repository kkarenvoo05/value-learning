# Round-1 factorial: what makes the small chains drop at round 1? (2026-11-21)

**Status 2026-11-21 10:40: label check done (below); 40 jobs submitted, held for gpu_cap. Reading rule written before
any result.** Karen: "run the free label check first, then build arms A–D".
Background: `reports/2026-11-20-mt-r1-drop/report.md` (the drop repeats across seeds for the ~50-base-per-task chains
b50cv / mt100cv, not for b100; oversampling base to 91% fixes val only).

## Question
Three explanations for the drop: (1) a one-step jump to ~50% guided data from **one** policy; (2) too few distinct
base episodes anchoring the guidance; (3) too little data overall. The existing chains change base count, V, demo
share and the guiding policy together, so they cannot separate these.

## Label check (free, offline): do labels treat base and guided episodes differently?
`label_by_source.py` → `label_by_source.txt`: each chain's round-1 table (and the 900 chain's r1 / r7), train split,
split by source (round 1 = base pi0.5, later rounds = guided).

| chain | source | eps | frame share | ep success | positive | pos on success / fail | negatives from successes | mean A |
|---|---|---|---|---|---|---|---|---|
| 900 r1 | base / guided | 900 / 100 | 0.90 / 0.10 | 0.57 / 0.63 | 0.40 / 0.40 | 0.52 / 0.27 · 0.49 / 0.29 | 0.42 / 0.48 | 0.001 / 0.003 |
| 900 r7 (no drop) | base / guided | 900 / 700 | 0.59 / 0.42 | 0.57 / 0.77 | 0.40 / 0.40 | 0.52 / 0.27 · 0.45 / 0.27 | 0.42 / 0.65 | 0.001 / 0.008 |
| b100 r1 | base / guided | 100 / 100 | 0.53 / 0.47 | 0.59 / 0.83 | 0.40 / 0.40 | 0.51 / 0.27 · 0.44 / 0.27 | 0.44 / 0.74 | 0.001 / 0.009 |
| b100cv r1 | base / guided | 100 / 100 | 0.53 / 0.47 | 0.59 / 0.76 | 0.40 / 0.40 | 0.50 / 0.29 · 0.45 / 0.29 | 0.45 / 0.64 | −0.002 / 0.004 |
| **b50cv r1** (drops) | base / guided | 50 / 50 | 0.52 / 0.49 | 0.54 / 0.70 | 0.40 / 0.40 | 0.50 / 0.30 · 0.45 / 0.31 | 0.40 / 0.59 | −0.002 / 0.003 |
| **mt100cv r1 moka** (drops) | base / guided | 50 / 50 | 0.51 / 0.49 | 0.54 / 0.66 | 0.40 / 0.40 | 0.52 / 0.29 · 0.48 / 0.28 | 0.39 / 0.53 | 0.000 / 0.003 |
| **mt100cv r1 t64** (drops) | base / guided | 50 / 50 | 0.54 / 0.46 | 0.66 / 0.80 | 0.40 / 0.40 | 0.50 / 0.28 · 0.46 / 0.26 | 0.46 / 0.62 | −0.001 / 0.006 |

(b100iql in the txt: same picture.)

- Observation: positives are 40% in both sources by construction (thresholded per round), the mean advantage is ~0
  for both, and successful vs failed episodes are separated alike in both sources. Guided episodes succeed more, so
  more of their negatives come from successful episodes (0.53–0.74).
- Observation: the chains that drop (b50cv, mt100cv) look no different from the ones that do not (900 r7 at 42%
  guided, b100 seed 1). If anything the non-dropping 900 r7 and b100 have the most negatives from successes.
- Interpretation: the labels do not encode "which policy collected this", and nothing in them predicts the drop. The
  cause is in what the residual learns from the data (state / action distribution), not in the labels. A label-level
  fix (e.g. rethresholding guided episodes) is not supported by this check.
- Not done (needs a GPU job and a new script): whether r1's push points the same way as r0's on the same frames
  (the guided data re-teaching r0's push, applied twice).

## Setup (arms)
All arms: round-1 retrain on a subset of `tables/moka_agr_r7` (900 base + 100 guided from each of the 900 chain's
r0..r6), its frozen-900-V labels `agr_vb40_r7` (r1 and r7 label files agree on 99.3% of shared frames), **no demos**
[ours: demos hurt, and their share would differ by arm], 5000 steps as every chain, seeds 0 and 1 (controls seed 0).
Lists `tables/moka_agr_r7/rebal/fact_*.txt` from `rebal/fact_lists.py` (rng 20261121; `fact_lists.json` here).
base100 = the b100 chain's 100 r0 base episodes (fresh layouts), so A = b100 r1 with the 900 chain's r0 as guide.

| arm | base eps | guided eps | guided frame share | frames | compares |
|---|---|---|---|---|---|
| A | 100 | 100 from r0 | 49% | 90k | drop from the data mix alone? (vs R100) |
| B | 100 | 100, 14–15 from each of r0..r6 | 48% | 89k | A vs B: one guiding policy vs seven (1) |
| C | 700 | 700 from r0..r6 | 48% | 619k | B vs C: more base and more data, same share |
| D | 100 ×7 (52.2% of frames) | 700 from r0..r6 | 48% | 616k effective | C vs D: distinct base (2) vs volume (3) |
| R100 / R700 | 100 / 700 | 0 | 0 | 46k / 324k | r0 controls |

Reference (with demos, so not exactly comparable): 900 chain r1 (900 + 100 from r0) val 79.1; nd900 (900 base, no
demos) 81.2. Check on the r7 table (its 120 base val episodes, as every moka table). Evals at push 0.99: moka val 330
(matched noise 1), moka cont 300. Submit: `scripts/fact_r1_submit.sh` (log `logs/fact_r1.log`).

## Reading rule (pre-registered)
Per arm: mean of the 2 seeds; a difference counts only if both seeds agree in sign and the mean gap is ≥ 10 points on
both val and moka cont (single retrains vary by up to ~25 points; two-seed means by roughly half that).
- **A ≈ R100 (no drop):** the data mix alone does not cause it; the drop needs the chain's own weak r0 as guide and /
  or its own V (b50cv / mt100cv) → next: A with b50cv's guided episodes, or with chain V.
- **A drops:** reproduced in a clean setting, then:
  - **B ≈ R100** (many policies fine, one policy drops): explanation (1).
  - **B drops, C fine, D drops:** distinct base episodes (2).
  - **B drops, C fine, D fine:** data volume (3).
  - **C drops too:** the 50% share itself hurts at any scale; the 900 chain survived because its share grew slowly
    (7 rounds) → (1) in its "sudden" form.
- R700 vs R100 gives the base-count effect without guided data (expected small: b100 79.7, b200 73.6, 900 75.5).

## Jobs (submitted 10:35, held; gpu_cap priority line `agr_fact.*`, after b200cv, before the b100 chain)
| arm | seed 0 train / check / eval / ceval | seed 1 |
|---|---|---|
| A | 18620353 / 54 / 55 / 56 | 18620379 / 80 / 81 / 82 |
| B | 18620357 / 58 / 59 / 61 | 18620383 / 84 / 85 / 86 |
| C | 18620362 / 64 / 65 / 66 | 18620387 / 88 / 89 / 90 |
| D | 18620367 / 68 / 69 / 70 | 18620391 / 92 / 93 / 94 |
| R100 | 18620371 / 72 / 73 / 74 | – |
| R700 | 18620375 / 76 / 77 / 78 | – |

Cost: 10 trains (~3.2 h) + 10 checks (~50 min) + 20 evals (~2.5–3 h) ≈ 95 GPU-h; at ~5 GPUs about a day.

## Caveats (known before running)
- One guiding pool (the 900 chain's policies, all strong: 75–88%); the small chains' own r0 were weaker (b50cv 45%).
- Fixed 5000 steps: C / D see each frame ~4× less often than A / B (as the real chains: every round trains 5000 steps).
- Guided episodes: 60 fixed-state + 40 fresh per round; base100 is all fresh layouts.
- No demos, unlike every chain reference number.
