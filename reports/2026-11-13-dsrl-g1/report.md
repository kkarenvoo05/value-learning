# Action-imagination gate G1 with DSRL's critic

**Status 2026-11-13: smoke test passed; probes running (18493812); results pending.** Written before any result.

## Question

Can DSRL's noise-space critic Q(s, w) rank candidate π₀.₅ chunks *within one state*? Action imagination (Advantage
Guidance §6.3) needs exactly that. G1 on 10 offline critics failed: within-state Spearman −0.07 to +0.11 against a
ceiling of 0.37 (`reports/2026-11-12-iql-g1/`). DSRL's critic is the one critic we have that was trained on actions it
varied itself (online SAC, 1,357 own episodes, 0.861 held-out), and its action is π₀.₅'s start noise. So it scores
π₀.₅ samples directly.

## Setup

- **Critic:** `dsrl_s0` `agent_final_ep1800.pkl`, the agent evaluated at 0.861 in 18452292. Earlier checkpoints
  (ep 1000 / 500 / 100) are scored as a learning-curve diagnostic.
- **Probe:** `scripts/libero_steer_probe.py`, the same simulator branching as G0/G1, against a DSRL eval server
  (`dsrl_pi05.py --mode eval`; `libero_steer_probe.sbatch DSRL_AGENT=...`).
  - States: the 10 held-out val init states, never used in DSRL training.
  - Trunk episode: the DSRL policy. Branch steps 40 100 160 220 280 [ours]: DSRL successes last a median 331
    steps (p10 311), so G1's 310 would sit near the end.
  - At each branch: N = 8 candidate noise vectors. Each becomes a chunk via frozen π₀.₅; its first 5 actions run
    from the saved MuJoCo state, then the DSRL policy continues **R = 4** times [ours: G1 used 2, and the 2026-11-12
    literature review found 2 underpowered]. p_c = success share.
  - Two probes, reported separately, never pooled:
    - **actor:** w = the DSRL actor's samples, i.e. the critic's own policy (what best-of-N on DSRL would rank).
    - **gauss:** w ~ N(0, I) clipped to [−1, 1] (the actor's support) [ours]. This is base-like spread inside DSRL's
      action space, the imagination premise. Some of these w are off the actor's distribution, which is part of
      the test.
- **Consistency check:** the server returns Q(s, w) for every candidate. The offline re-score
  (`scripts/libero_dsrl_g1.py`) must reproduce it (max |diff| < 1e-3), so image, state and w are known to match.
- **Statistics:** as `scripts/libero_iql_g1.py`: within-state Spearman over points where p varies; best-of-8 pick
  gain; bootstrap over the trunk episodes; ceiling from the probe's between-candidate variance with R = 4.
  Diagnostics: min over heads, the actor's own log-density of w, a random-score null.

## Reading rule (pre-registered)

- **Gate (plan §5):** q:final within-state Spearman with a 95% CI above 0, read next to the ceiling. Applied per
  probe.
- **actor passes:** DSRL's critic ranks its own policy's samples, so best-of-N on DSRL can help. This is not yet
  imagination.
- **gauss passes:** the premise of imagination holds in noise space. Next is G2: imagined w per training frame,
  scored by this Q.
- **Neither passes, and the ceiling is close to G1's 0.37 or lower:** even an online, interventional critic cannot
  rank single decisions on this task. Imagination is closed.
- **Pick gains are reported but are not the gate:** with 8 candidates and R = 4 they are noisy.

## Run

| step | job | what |
|---|---|---|
| GPU smoke | 18493792 | val state 9, 80 steps, 3 gauss candidates, branch 20 / 40, R 2: protocol + served-vs-offline Q. **Done (3 min, iris7):** restore fidelity 1.7e-17; offline Q = served Q to 3.8e-6; 30% of gauss entries clipped; no success within the 80-step cap, as expected |
| probes actor, then gauss | 18493812 | afterok smoke; one job, one server, `runs/dsrl_g1_libero_10_<job>/{actor,gauss}/`; 24 h limit |

Score when done (openpi env):
`.venv/bin/python <vl>/scripts/libero_dsrl_g1.py --probes <run>/actor <run>/gauss --out <vl>/reports/2026-11-13-dsrl-g1`
