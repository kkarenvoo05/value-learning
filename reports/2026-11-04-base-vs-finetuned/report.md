# Base π₀.₅ vs the fine-tuned policies, same 330 trials (job 18309278)

**Question.** Advantage-tagged and success-only fine-tunes scored 0.41–0.43. Is that above or below the policy they
started from? Earlier base rollouts on the same 10 val states (different env seeds, 12 per state) gave 63/120 = 0.525.

**Setup.** pi05_libero checkpoint unchanged (`scripts/libero_eval.sbatch`, openpi server), libero_10 task 8, 33 × 10
val init states (9 12 13 17 25 32 40 42 44 48), env seed 7, same client and episode order as the arm evals
(18291520/21/22). CI: scene-stratified bootstrap of the per-episode paired difference (4000 resamples).

| policy | success | vs base (95% CI) |
|---|---|---|
| base π₀.₅ | **186/330 = 0.564** | — |
| arm A success-only (18288138) | 134/330 = 0.406 | −0.158 [−0.224, −0.091] |
| arm C ResNet tags, + tag (18275460) | 135/330 = 0.409 | −0.155 [−0.218, −0.088] |
| arm C, − tag | 142/330 = 0.430 | −0.133 [−0.200, −0.067] |

Per scene (success rate of 33): base / success-only / tag+ / tag−
9: .64/.39/.61/.67 · 12: .21/.03/.09/.12 · 13: .15/.55/.39/.45 · 17: .48/.18/.18/.15 · 25: .61/.73/.67/.70 ·
32: .76/.48/.79/.58 · 40: .85/.61/.33/.58 · 42: .24/.12/.24/.21 · 44: .91/.52/.36/.48 · 48: .79/.45/.42/.36

**Observations.** Every fine-tune is 13–16 points below base; largest drops on the easiest scenes (44, 40, 48);
two scenes improve under fine-tuning (13, 25).

**Interpretation.** The Stage 6 fine-tune recipe (openpi LoRA recipe on pi05_libero, LoRA 1e-4 / rest 2.5e-5 incl.
SigLIP, 5000 steps, 900 own rollouts) degrades the policy; together with the tag-usage check (tag not learned),
the label comparison has not been tested yet.

**Next.** Recipe fix on arm A first (frozen SigLIP; 10× lower lr; intermediate checkpoints evaluated), then labels.
