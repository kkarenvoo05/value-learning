# DSRL online on moka: calibration baseline (mentor direction 1) — setup

**Status 2026-11-10 (setup, no result yet).** Code built and checked; smoke run on the interactive L40S works (~6 ms per SAC update, ~12 s of updates + ~23 s rollout per episode → ~18 h for 1800 episodes). Training job 18434524, queued after the round-2 collectors.

## Question

What does an established RL method get on this exact task, starting from the same base π₀.₅ (0.524 on the 330 val
trials)? And does π₀.₅'s noise space contain outcome-relevant variation (headroom) at all?
- DSRL climbs well above base → headroom exists; our offline pipeline (RECAP tag, weighting, filtered BC) is limited
  by its signal, and online / iterated data is the way forward.
- DSRL stalls near base → "the policy's own variation leaves little to select from" holds for an online learner too.

## Method (Wagenmaker, Nakamoto et al., CoRL 2025, arXiv 2506.15799; code nakamotoo/dsrl_pi0 @ 7f48937)

π₀.₅ stays frozen. A small pixel SAC agent picks the flow-matching **start noise**: one 32-d vector w ∈ [−1, 1]
(tanh-Gaussian), repeated over the action horizon. Everything else is the base policy. Reward −1 per decision,
0 + terminal at success. Hyperparameters = the paper's LIBERO column / `run_libero.sh` (listed in `src/vl/dsrl.py`):
64×64 agent view + 8-d state, small CNN, 10 Q heads (mean), UTD 20, batch 256, discount 0.999 per env step, warmup
with w ~ N(0, I).

Paper's LIBERO result, for scale: one LIBERO-90 task held out from π₀'s fine-tuning, ≈20% → ≈100% "after only
≈10,000 online samples" (Sec. 5.5, 3 seeds). Moka differs: it is in π₀.₅'s training set and base is already 52%,
so headroom is less obvious. The paper's own limitation: a concentrated action distribution "might not provide
enough options" (our steer probe: 8 samples end 1.9 mm apart).

## Our choices [ours]

| choice | value | why |
|---|---|---|
| decision | one π₀.₅ chunk = 5 executed steps (≈104 per episode) | our replan rate; paper's π₀ executes 20 of 50 |
| per-decision discount | 0.999⁵ | paper's 0.999 per env step |
| colour jitter | off | official learner applies it by default; fixed sim renders |
| training states | the 30 train init states, cycled (REPEATS=60) | same states as the rollout data of the offline arms |
| budget | 1800 episodes | = round 1 (900) + round 2 (900) of filtered BC / RECAP: matched env interaction |
| checkpoints | every 100 episodes | eval at matched budgets (300 / 900 / 1800 first) |
| eval | 10 val states × 33, env seed 7 (as every arm); actor sampled stochastically (as the official eval) | matched noise does not apply (the noise is the agent's) |

## Code

- `src/vl/dsrl.py`: SAC port (networks, inits, losses as jaxrl2; distrax replaced by closed-form tanh-Gaussian,
  checked against scipy to 1e-7); runs in the openpi env, no new packages.
- `scripts/dsrl_pi05.py`: policy server (train / eval / `--selftest` with a fake π₀.₅: protocol, rewards and
  terminals, warmup → updates, save/load, noise bound; passes).
- `scripts/libero_rollouts.py --args.dsrl`: client sends `{"dsrl_end", success, final obs}` after each episode.
- `scripts/libero_eval.sbatch`: `DSRL_MODE=train|eval`, `DSRL_AGENT`, `DSRL_SEED`, `DSRL_ARGS`.

Train: `DSRL_MODE=train SUITE=libero_10 TASKS=8 REPEATS=60 SEED=51 NAME=dsrl_s0 CLIENT_ARGS="--args.init-states <30 train states>" sbatch --time=24:00:00 ... scripts/libero_eval.sbatch`
Eval a checkpoint: `DSRL_MODE=eval DSRL_AGENT=<run>/agent_900.pkl SUITE=libero_10 TASKS=8 REPEATS=33 SEED=7 NAME=eval_dsrl_s0_ep900 CLIENT_ARGS="--args.init-states 9 12 13 17 25 32 40 42 44 48" sbatch scripts/libero_eval.sbatch`

## Caveats

- Not resumable: a preempted training job restarts with a fresh agent (in-memory buffer). Checkpoints survive.
- One seed first; the paper averages 3.
- The training curve in `dsrl_log.jsonl` (exploration episodes, with actor noise) is not the eval number.
