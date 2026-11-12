"""Data-variety pilot (2026-11-08, Karen: option A, pilot first): does exploration noise make the robot act
differently in the same spots without ruining the episodes?

Why: the coach test (reports/2026-11-08-coach-test/) found labels from what actually happened carry less action
information than the current 1 s labels, so the bottleneck is the data: one policy doing nearly the same thing in
the same spots. Option A: the same base pi05_libero, but each new action chunk is, with probability 0.5, sampled
from start noise x s (scripts/libero_rollouts.py --args.explore-prob / --args.explore-scale).

Runs (scripts/libero_eval.sbatch, BASE_OURS=1 so all arms use serve_pi05_finetuned.py --base): 30 fixed train
states x 3 repeats, env seed 21, raw saved; arms control (p 0), x1.5 and x2.0 (p 0.5).

Measures [ours]:
  success     rate per arm; difference vs control, paired by init state (bootstrap over states)
  steps       mean policy steps of successes (speed)
  spread      how differently the robot acts in the same spot: for each init state and step t < EARLY (states still
              nearly identical across repeats), the std over the 3 repeats of the executed action, RMS over arm dims
              (0-5) and separately the gripper (6); mean over (state, t). Control = the policy's own variety
  chunk stats in explore arms, noisy vs normal chunks: mean |arm action|, gripper sign flips per chunk
Reading rule [set before the result]: choose the largest scale whose success is at most ~10 pts below control
(point estimate) and whose early arm spread is clearly above control (>= 1.5x; the gripper is reported but not
used: it is almost always open in the first 3 s, so its ratio has a near-zero base); if none qualifies, report and
stop (option C: replay from saved sim states). Caveat: spread is measured early only, not at grasps / placements.

Noise sweep [2026-11-09]: the same rule; the choice is the qualifying setting with the largest spread, and settings
10-15 pts below control (with the spread) are flagged borderline. scripts/noise_sweep.sh.

Usage: .venv/bin/python scripts/libero_explore_pilot.py --runs control=<dir> x1.5=<dir> x2.0=<dir> --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EARLY = 60  # policy steps (3 s at 20 Hz) [ours]: before the repeats' states drift far apart
B = 4000


def load(run: Path) -> tuple[pd.DataFrame, dict]:
    eps = pd.read_csv(run / "episodes.csv")
    raws = {}
    for r in eps.itertuples():
        if isinstance(r.raw, str) and r.raw:
            z = np.load(run / "raw" / r.raw)
            raws[r.episode] = {k: z[k] for k in ("actions", "new_chunk") + (("noisy",) if "noisy" in z.files else ())}
    return eps, raws


def paired_diff(a: pd.DataFrame, c: pd.DataFrame, rng) -> tuple[float, list[float]]:
    pa, pc = a.groupby("init_state").success.mean(), c.groupby("init_state").success.mean()
    s = pa.index.intersection(pc.index)
    d = (pa[s] - pc[s]).to_numpy()
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def spread(eps: pd.DataFrame, raws: dict) -> dict:
    arm, grip = [], []
    for _, g in eps.groupby("init_state"):
        acts = [raws[e]["actions"] for e in g.episode if e in raws]
        if len(acts) < 2:
            continue
        T = min(EARLY, *(len(a) for a in acts))
        A = np.stack([a[:T] for a in acts])  # (repeats, T, 7)
        sd = A.std(0)  # (T, 7)
        arm.append(np.sqrt((sd[:, :6] ** 2).mean(1)))
        grip.append(sd[:, 6])
    return {"arm": float(np.concatenate(arm).mean()), "gripper": float(np.concatenate(grip).mean())}


def chunk_stats(raws: dict) -> dict:
    out = {}
    for flag in (False, True):
        mag, flips, n = [], 0, 0
        for r in raws.values():
            if "noisy" not in r:
                continue
            starts = np.flatnonzero(r["new_chunk"])
            for i, s in enumerate(starts):
                e = starts[i + 1] if i + 1 < len(starts) else len(r["actions"])
                if bool(r["noisy"][s]) != flag:
                    continue
                a = r["actions"][s:e]
                mag.append(np.abs(a[:, :6]).mean())
                flips += int((np.diff(np.sign(a[:, 6])) != 0).sum())
                n += 1
        if n:
            out["noisy" if flag else "normal"] = {"chunks": n, "mean_abs_arm": float(np.mean(mag)),
                                                  "gripper_flips_per_chunk": flips / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir; the first is the control")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.runs)
    names = list(runs)
    data = {n: load(Path(d)) for n, d in runs.items()}
    rng = np.random.default_rng(0)
    ctrl = data[names[0]][0]
    res = []
    for n in names:
        eps, raws = data[n]
        ok = eps[eps.success == 1]
        row = {"arm": n, "episodes": len(eps), "success": float(eps.success.mean()),
               "steps_success": float(ok.policy_steps.mean()) if len(ok) else None, **{
                   f"spread_{k}": v for k, v in spread(eps, raws).items()}}
        if n != names[0]:
            d, ci = paired_diff(eps, ctrl, rng)
            row.update(success_vs_control=d, success_vs_control_ci95=ci)
        cs = chunk_stats(raws)
        if cs:
            row["chunks"] = cs
            row["noisy_share"] = cs.get("noisy", {}).get("chunks", 0) / sum(v["chunks"] for v in cs.values())
        res.append(row)
    c = res[0]
    for r in res[1:]:
        r["spread_ratio_arm"] = r["spread_arm"] / c["spread_arm"]
        r["spread_ratio_gripper"] = r["spread_gripper"] / c["spread_gripper"]
        r["qualifies"] = bool(r["success_vs_control"] >= -0.10 and
                              r["spread_ratio_arm"] >= 1.5)
        r["borderline"] = bool(-0.15 <= r["success_vs_control"] < -0.10 and r["spread_ratio_arm"] >= 1.5)
    q = [r for r in res[1:] if r["qualifies"]]
    # [2026-11-09 noise sweep] largest early arm spread among qualifying settings (was: largest scale parsed from
    # the name, which breaks on names like x2.0_all; same choice on the pilot, where spread rose with the scale)
    choice = max(q, key=lambda r: r["spread_ratio_arm"])["arm"] if q else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps({"runs": runs, "early_steps": EARLY, "results": res,
                                                "choice": choice}, indent=2))
    cols = ["arm", "episodes", "success", "success_vs_control", "steps_success", "spread_arm", "spread_gripper",
            "spread_ratio_arm", "spread_ratio_gripper", "qualifies", "borderline"]
    t = pd.DataFrame(res).reindex(columns=cols)
    t.to_csv(out / "pilot_table.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    for r in res:
        if "chunks" in r:
            print(r["arm"], json.dumps(r["chunks"]), f"CI {np.round(r['success_vs_control_ci95'], 3).tolist()}")
    print("choice:", choice)


if __name__ == "__main__":
    main()
