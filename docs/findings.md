# Findings

Settled conclusions, grouped by topic. **Corrected in place**: when a finding is overturned, edit or strike it
here and note the date; do not append a contradiction. Each item cites its evidence. Observations and
interpretations are labelled. Live status is in `docs/CURRENT.md`, chronology in `docs/libero_plan.md`.

## LIBERO (libero_10 task 8, moka pots)

### Data
- Stage 2: 570 frozen-π₀.₅ rollouts (450 on 30 train init states, 120 on 10 val states); open-loop replay of
  the saved actions is exact. Success differs by init state from 0 to 100%. [plan, Stage 2 / 2026-10-28]
- Layout variety matters more than episode count for value heads on frozen features: at 375 episodes, fresh
  layouts give SigLIP head SNR 0.194 vs 0.122 on the 30 fixed layouts (ResNet 0.193 vs 0.155), and adding the fixed
  episodes to fresh (825) lowers both. *Interpretation:* repeated fixed layouts invite a layout shortcut.
  [`reports/2026-11-02-frozen-encoders/`, 2026-11-03]
  - *Corrected 2026-11-03:* "frozen SigLIP needs more data" (P1b) was wrong; it needs more layouts.

### Where π₀.₅'s value noise comes from
- Frozen π₀.₅ encodes "a pot is on the stove" (probe AUC 0.96–0.99), but value heads do not use it well.
  [plan 2026-10-30, progress probe]
- B's jittery V is per-frame flicker of the success belief, not error within a value hump (~98% of 2 s ΔV
  variance). [plan 2026-10-31; P2, plan 2026-11-02]
- Robot (VLA) training did not make SigLIP noisy: frozen π₀.₅ SigLIP ≥ web SigLIP on every metric, and ≈ an
  ImageNet ResNet at 825 episodes (head SNR 0.172 vs 0.168). [`reports/2026-11-02-frozen-encoders/`]
- Two separate noise sources, each about halving head SNR [same report]:
  - the frozen LLM layers: SigLIP 0.122 → LLM final hidden 0.059 (matched head, 375 fixed episodes);
  - fine-tuning SigLIP: 0.172 → 0.084 (825), 0.122 → 0.078 (375). Fine-tuning the ResNet costs nothing.
  - *Open:* where in the LLM the drop starts (P1c step 2, job 18279695).
- The value head architecture never matters (± 0.01 SNR in head swaps). [plan 2026-10-30, head swap]

### What did not help
- Auxiliary subtask head (label = pot count): no SNR change on the ResNet and on frozen B (2 of 3 seeds); it
  learns pot state at 96% accuracy. [plan 2026-11-02 E2, 2026-11-03 08:45]
- Removing the outcome loss: SNR 0.073, same as B. [plan 2026-10-31]
- Head-side regularisation (aux targets, feature dropout, bottleneck). [plan 2026-10-30]
- Smoothing V raises SNR but does not close B's gap to the ResNet at matched smoothing. [`reports/libero_smoothing.md`]

### Method traps
- Milestone and dip effects are not stable across seeds (no 3a probe arm has a positive milestone excess on all 3
  seeds); report ≥ 3 seeds before reading them. [`reports/libero_seeds.md`, plan 2026-10-29]
- Inner-CE early stopping stops soft-target arms at epoch 0; select on inner MAE instead. [plan 2026-10-31]
- π₀.₅ cannot be loaded in a 16 GiB interactive session; use sbatch. [plan 2026-11-03]

## Mulligan Insert Marker (Stages 1–5.5)

To fill from `PROGRESS.md` (Stages 4–5.5) and `reports/*.md`; not yet consolidated here.
