"""Failure types of moka-pot timeouts from recorded sim state (docs/libero_plan.md, failure-type tracking).

Same idea as the by-eye pass on the Stage 1 check (plan log 2026-10-28), computed from the raw files of
scripts/libero_rollouts.py --args.save-raw. "Stuck" is judged by what the robot is doing when time runs out.
Thresholds [ours], metres / policy steps (10 Hz); checked against frames of sampled episodes:
  stall          a stretch in which no pot moves more than MOVE from where it was at the stretch's start. Every
                 episode has some (driving to a pot), so the cut-off is the 95th percentile of the longest stall
                 over the train *successes* (passed with --args.cutoff for the val split)
  types, first match wins:
  knocked_over        a pot tilted > 45 deg at the end (orientation read by setting the final MuJoCo state)
  stuck_placing_pot2  one pot on the stove, the other lifted within JAM of it for >= JAM_STEPS steps at the end
                      (pressing it against the first pot)
  stuck_pot1          the final stall is above the cut-off and no pot is on the stove
  stuck_pot2          the final stall is above the cut-off and one pot is on the stove
  near_miss           one pot on the stove and the other lifted within APPROACH of it at the end
  stalled_earlier     the longest stall is above the cut-off but ended before the timeout (lost time, then
                      progressed again)
  slow                none of the above
Labels stay success / failure; the types are for analysis only (Karen, 2026-10-28).
Runs in the LIBERO client env. Writes failure_types.csv (failures) and stalls.csv (all episodes) next to
episodes.csv.

Usage:
    python scripts/libero_failure_types.py --args.run-dir <run>/train
    python scripts/libero_failure_types.py --args.run-dir <run>/val --args.cutoff <train cut-off>
"""

import csv
import dataclasses
import pathlib
from typing import Optional

from libero.libero import benchmark
import numpy as np
import tyro

import libero_rollouts as lr

MOVE, LIFT, JAM, APPROACH = 0.02, 0.02, 0.15, 0.20
JAM_STEPS = 60
POTS = ("moka_pot_1", "moka_pot_2")


@dataclasses.dataclass
class Args:
    run_dir: str
    cutoff: Optional[float] = None  # stall cut-off in steps; default: p95 over this run's successes


def tilt_deg(quat_xyzw):
    """Angle between the pot's up axis and world up."""
    x, y, z, w = quat_xyzw
    up_z = 1 - 2 * (x * x + y * y)  # z component of the rotated z axis
    return float(np.degrees(np.arccos(np.clip(up_z, -1, 1))))


def pot_positions(raw):
    keys = list(raw["object_pos_keys"])
    return {p: raw["object_pos"][:, 3 * keys.index(f"{p}_pos"):][:, :3] for p in POTS}


def longest_stall(pos):
    """(length, start) of the longest stretch where no pot moves more than MOVE from its position at its start."""
    T = len(next(iter(pos.values())))
    best, best_t = 0, 0
    for t in range(T):
        moved = np.zeros(T - t, dtype=bool)
        for p in pos.values():
            moved |= np.linalg.norm(p[t:] - p[t], axis=1) > MOVE
        n = int(np.argmax(moved)) if moved.any() else T - t
        if n > best:
            best, best_t = n, t
    return best, best_t


def final_stall(pos):
    """Steps at the end of the episode during which no pot moved more than MOVE (the stall the timeout hit)."""
    T = len(next(iter(pos.values())))
    for t in range(T):
        if all(np.linalg.norm(p[t:] - p[t], axis=1).max() <= MOVE for p in pos.values()):
            return T - t
    return 0


def tail(mask):
    """Length of the final run of True."""
    return int(len(mask) - np.flatnonzero(~mask)[-1] - 1) if (~mask).any() else len(mask)


def classify(raw, tilts, cutoff):
    pos = pot_positions(raw)
    placed = raw["goals"][-1, :2]
    stall, t0 = longest_stall(pos)
    fin = final_stall(pos)
    feats = {"placed_end": "".join(str(int(g)) for g in placed), "tilt_max": round(max(tilts.values()), 1),
             "stall": stall, "stall_start": t0, "final_stall": fin}
    if max(tilts.values()) > 45:
        return "knocked_over", feats
    if placed.sum() == 1:
        r, on = POTS[int(np.argmin(placed))], POTS[int(np.argmax(placed))]
        lifted = pos[r][:, 2] > pos[r][0, 2] + LIFT
        d = np.linalg.norm(pos[r][:, :2] - pos[on][:, :2], axis=1)
        feats.update(d_other_end=round(float(d[-1]), 3), jam_steps=tail(lifted & (d < JAM)))
        if feats["jam_steps"] >= JAM_STEPS:
            return "stuck_placing_pot2", feats
    if fin > cutoff:
        return ("stuck_pot2" if placed.any() else "stuck_pot1"), feats
    if placed.sum() == 1 and lifted[-1] and d[-1] < APPROACH:
        return "near_miss", feats
    if stall > cutoff:
        return "stalled_earlier", feats
    return "slow", feats


def main(args: Args) -> None:
    run = pathlib.Path(args.run_dir)
    rows = [r for r in csv.DictReader(open(run / "episodes.csv")) if r["raw"] and not r["error"]]
    raws = {}
    for r in rows:
        with np.load(run / "raw" / r["raw"]) as z:
            raws[r["episode"]] = {k: z[k] for k in z.files if k not in ("image", "wrist_image")}
    stalls = {e: longest_stall(pot_positions(raw)) for e, raw in raws.items()}
    succ = [stalls[r["episode"]][0] for r in rows if r["success"] == "1"]
    if succ:
        print(f"longest stall over {len(succ)} successes: median {np.median(succ):.0f}, "
              f"p95 {np.percentile(succ, 95):.0f}, max {max(succ)} steps", flush=True)
    cutoff = args.cutoff if args.cutoff is not None else float(np.percentile(succ, 95))
    with open(run / "stalls.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "success", "stall", "stall_start"])
        w.writerows([r["episode"], r["success"], *stalls[r["episode"]]] for r in rows)

    suite = benchmark.get_benchmark_dict()[rows[0]["suite"]]()
    env, _ = lr._get_libero_env(suite.get_task(int(rows[0]["task_id"])), lr.LIBERO_ENV_RESOLUTION,
                                int(rows[0]["env_seed"]))
    env.reset()
    out = []
    for r in rows:
        if r["success"] == "1":
            continue
        raw = raws[r["episode"]]
        obs = env.set_init_state(raw["sim_end"])
        tilts = {p: tilt_deg(obs[f"{p}_quat"]) for p in POTS}
        kind, feats = classify(raw, tilts, cutoff)
        out.append({"episode": r["episode"], "init_state": r["init_state"], "type": kind, **feats})
        print(out[-1], flush=True)
    env.close()
    fields = ["episode", "init_state", "type", "placed_end", "tilt_max", "stall", "stall_start", "final_stall",
              "d_other_end", "jam_steps"]
    with open(run / "failure_types.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    counts = {k: sum(o["type"] == k for o in out) for k in sorted({o["type"] for o in out})}
    print(f"{len(out)} failures (stall cut-off {cutoff:.0f} steps): {counts} -> {run / 'failure_types.csv'}")


if __name__ == "__main__":
    tyro.cli(main)
