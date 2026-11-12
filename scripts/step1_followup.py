"""Step-1 follow-up [2026-11-09, Karen: "if one holds base, run seed 1 of it, then put the tag or the success-only
weighting back on top of that recipe"]. Runs as a CPU slurm job after the low-lr grid evals (step1_grid.sh).

Decision [rule set 2026-11-08, before these evals were read]: a grid arm checkpoint (@2000 or @final) passes when its
paired 95% CI vs base 18355478 includes 0 or lies above it (pairs = init state x repeat, matched noise seed 1, 5000
bootstrap draws). Winner = the passing checkpoint with the highest success (ties -> @final). Only complete evals
(330 episodes) count. No pass -> nothing is submitted.

On a winner (its recipe = its demo share + low lr), three lanes, each 5000 steps, evals @2000 and @final as the grid:
  seed1   the winner's recipe untagged, seed 1                       (is it a seed fluke?)
  tag     + the tag: arm advantage, vb40 labels, tag dropout 0.3, demos "+", seed 0; eval tag positive, plus
          tag negative at the winner's step (does the tag change behaviour?)
  wsucc   + success-only weighting: arm weighted on the 516 train successes (weights resnet_valuebinned_weights_succ,
          demos weight 1), seed 0 (as 18387307 = 0.567, but on the winner's recipe)
Writes reports/2026-11-09-step1-followup/decision.json.  --dry-run: decide and print the sbatch lines only.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/move/u/karenvo/Projects/value-learning")
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
TAB = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
LAB = TAB / "labels/resnet_valuebinned_p40/labels.parquet"
WSUCC = TAB / "labels/resnet_valuebinned_weights_succ/weights.parquet"
SUCC = TAB / "subsets/train_successes.txt"
BASE_EVAL = RUNS / "eval_base_noise1_libero_10_18355478"
LOW = "--lora-lr 1e-5 --lr 2.5e-6"
# grid arms: name -> (train job, recipe args beyond the shared ones). The original-lr arms were added 2026-11-09
# after d50 orig @final passed (0.515, CI [-0.076, +0.061]): the rule is about any grid recipe, not only low lr.
ARMS = {"grid_d50_orig": (18396162, "--demo-share 0.5"),
        "grid_d75_orig": (18396166, "--demo-share 0.75"),
        "grid_d50_low": (18396170, f"--demo-share 0.5 {LOW}"),
        "grid_d75_low": (18396174, f"--demo-share 0.75 {LOW}"),
        "grid_d02_low": (18396975, LOW)}
COMMON = "--train --demos --freeze-siglip --save-every 1000"
TRAIN_SB = ["sbatch", "--parsable", "-A", "iris", "-p", "iris",
            "--exclude=iris[1-4,8-9],iris-hgx-[1-2],iris-hp-z8"]
STATES = "--args.init-states 9 12 13 17 25 32 40 42 44 48"
OUT = REPO / "reports/2026-11-09-step1-followup"


def load(d):
    e = pd.read_csv(d / "episodes.csv")
    return e.set_index(["init_state", "repeat"]).success


def paired(d, base, rng):
    a = load(d)
    j = a.index.intersection(base.index)
    x = a[j].values - base[j].values
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=int(len(a)), success=float(a.mean()), diff=float(x.mean()), ci=[float(lo), float(hi)])


def sb(cmd, dry):
    print(" ".join(cmd), flush=True)
    if dry:
        return "DRY"
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO).stdout.strip().split(";")[0]


def evalj(dep, run, step, name, tag, dry):
    env = (f"ALL,SUITE=libero_10,TASKS=8,REPEATS=33,SEED=7,MATCHED_NOISE=1,NOISE_SEED=1,FT_RUN={run},FT_TAG={tag},"
           f"FT_STEP={step},NAME=eval_{name}_at{step}_{tag}_noise1,CLIENT_ARGS={STATES}")
    return sb(["sbatch", "--parsable", "-d", dep, f"--export={env}", "scripts/libero_eval.sbatch"], dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    base = load(BASE_EVAL)
    scores, notes = [], []
    for arm, (job, _) in ARMS.items():
        for step in ("2000", "final"):
            hits = sorted(h for h in RUNS.glob(f"eval_{arm}_s0_at{step}_noise1_libero_10_*") if (h / "episodes.csv").exists())
            if not hits:
                notes.append(f"{arm} @{step}: no eval dir")
                continue
            r = paired(hits[-1], base, rng)
            r.update(arm=arm, step=step, eval=hits[-1].name, passed=bool(r["n"] == 330 and r["ci"][1] >= 0))
            if r["n"] != 330:
                notes.append(f"{arm} @{step}: incomplete ({r['n']}/330), not counted")
            scores.append(r)
    passing = [r for r in scores if r["passed"]]
    win = max(passing, key=lambda r: (r["success"], r["step"] == "final")) if passing else None
    dec = dict(rule="paired 95% CI vs base 18355478 includes 0 or above; best passing success wins",
               scores=scores, notes=notes, winner=win, submitted={})
    for r in scores:
        print(f"{r['arm']:14s} @{r['step']:5s} n {r['n']} success {r['success']:.3f} diff {r['diff']:+.3f} "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {'PASS' if r['passed'] else 'fail'}", flush=True)
    if win:
        arm, extra = win["arm"], ARMS[win["arm"]][1]
        short = arm.replace("grid_", "")
        lanes = {
            f"fu_{short}_seed1_s1": (f"{COMMON} --arm advantage --labels {LAB} --tag-dropout 1.0 {extra} --seed 1",
                                     ["none"]),
            f"fu_{short}_tag_s0": (f"{COMMON} --arm advantage --labels {LAB} --tag-dropout 0.3 {extra} --seed 0",
                                   ["positive"]),
            f"fu_{short}_wsucc_s0": (f"{COMMON} --arm weighted --episodes {SUCC} --weights {WSUCC} {extra} --seed 0",
                                     ["none"]),
        }
        for name, (args, tags) in lanes.items():
            t = sb(TRAIN_SB + [f"--export=ALL,NAME={name},ARGS={args}", "scripts/pi05_policy_finetune.sbatch"], a.dry_run)
            run = RUNS / f"policy_{name}_{t}"
            e2 = evalj(f"afterok:{t}", run, "2000", name, tags[0], a.dry_run)
            e3 = evalj(f"afterany:{e2}", run, "final", name, tags[0], a.dry_run)
            jobs = dict(train=t, run=str(run), eval_2000=e2, eval_final=e3)
            if "_tag_" in name:  # tag usage: same checkpoint as the winner, tag negative
                jobs["eval_negative"] = evalj(f"afterany:{e3}", run, win["step"], name, "negative", a.dry_run)
            dec["submitted"][name] = jobs
            print(name, jobs, flush=True)
    else:
        print("no low-lr arm holds base: nothing submitted", flush=True)
    if not a.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "decision.json").write_text(json.dumps(dec, indent=1))
        print("wrote", OUT / "decision.json")


if __name__ == "__main__":
    main()
