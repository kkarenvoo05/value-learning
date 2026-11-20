"""Mid-task start points for a collection round [2026-11-14; Karen: "run mid-task starts as the next experiment"].

    .venv/bin/python scripts/libero_midtask_starts.py --table <table dir> --rounds 2 3 4 5 --n-fixed 60 --n-fresh 40 \
        --out <dir>/starts.csv

Why (reports/2026-11-14-failure-timing): the plateaued policies (AG round 4 0.794, DSRL 1300 0.824) fail at the second
pot (hover without grasping, knocking the first pot off) although 75-84% of one-pot failures had time left; a normal
round spends most of each episode on the easy first pot. A mid-task round collects from the moment the first pot is
on, so all its data is second-stage.
Start point [ours]: a train-split rollout of --table (AG collection rounds only by default; held-out states never
appear in a table's train split) at k = the first step after which exactly one pot is on the stove (goals[:, :2]) +
--offset steps (the gripper still holds or is releasing the pot: the hand-over is a state the policy itself reaches),
only if that first-pot step is <= --max-first (a start with little time left is wasted) and the pot stays on (exactly
one pot, every step from the first-pot step to k + --stable) [ours; test 2026-11-14: a source whose pot teeters on the
stove edge flips its "on" predicate at a 1e-14 state difference, so the replay's hand-over had no pot on; 19 of the
first 100 starts were unsteady]. One start per source
episode, sampled with --seed; the 60 / 40 fixed / fresh mix of a normal round. scripts/libero_rollouts.py
--args.start-from replays the source to k and checks the hand-over state.
Writes starts.csv (start_id, source_raw, start_step, init_state, env_seed, layout, source_round, source_success,
first_pot_step, source_episode_index) and prints the counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def first_pot_step(raw: str, offset: int, stable: int) -> tuple[int, bool]:
    """(first step with exactly one pot on or -1, whether it stays exactly one through first + offset + stable)."""
    with np.load(raw) as z:
        g = z["goals"][:, :2].sum(1)
    hit = np.flatnonzero(g == 1)
    if not len(hit):
        return -1, False
    f = int(hit[0])
    w = g[f:f + offset + stable + 1]
    return f, bool(len(w) == offset + stable + 1 and (w == 1).all())


def _runs(mask: np.ndarray, val: bool) -> list[tuple[int, int]]:
    out, s = [], None
    for i, m in enumerate(mask):
        if m == val and s is None:
            s = i
        if m != val and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def recovery_step(raw: str) -> tuple[int, str]:
    """[2026-11-19, reports/2026-11-19-residual-failures] (k, kind) for a recovery start, or (-1, ""): k = 20 steps
    after a placed pot (>= 10 steps on) has been off the stove for 20 steps ("lost", the failure-class rule of
    libero_week9_nov17.py), or 10 steps after a pot first sits 2 cm below its start height ("tipped"); the earlier."""
    with np.load(raw) as z:
        g, op = z["goals"][:, :2].astype(bool), z["object_pos"]
    best = (-1, "")
    for p in range(2):
        on = [r for r in _runs(g[:, p], True) if r[1] - r[0] >= 10]
        if on:
            off = [r for r in _runs(g[:, p], False) if r[0] > on[0][0] and r[1] - r[0] >= 20]
            if off and (best[0] < 0 or off[0][0] + 20 < best[0]):
                best = (off[0][0] + 20, "lost")
        tip = np.flatnonzero(op[:, 6 * p + 2] < op[0, 6 * p + 2] - 0.02)
        if len(tip) and (best[0] < 0 or int(tip[0]) + 10 < best[0]):
            best = (int(tip[0]) + 10, "tipped")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["second", "recovery"], default="second", help="[2026-11-19] second = hand-over "
                    "after the first pot is on (default, unchanged); recovery = hand-over after a pot is lost / tipped")
    ap.add_argument("--min-left", type=int, default=150, help="--kind recovery: at least this many of the 520 steps "
                    "left after the hand-over")
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--table", required=True)
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 3, 4, 5], help="table rounds (2 = AG round 1)")
    ap.add_argument("--n-fixed", type=int, default=60)
    ap.add_argument("--n-fresh", type=int, default=40)
    ap.add_argument("--offset", type=int, default=10)
    ap.add_argument("--max-first", type=int, default=400)
    ap.add_argument("--stable", type=int, default=20, help="pot must stay on through k + this many steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eps = pd.read_csv(Path(a.table) / "episodes.csv", dtype={"goals_end": str})
    eps = eps[(eps.split == "train") & eps["round"].isin(a.rounds)].copy()
    seeds = {}
    for run in eps.run.unique():  # env seed per source run (the client re-creates the env with it)
        er = pd.read_csv(Path(eps[eps.run == run].raw.iloc[0]).parent.parent / "episodes.csv")
        assert er.env_seed.nunique() == 1, run
        seeds[run] = int(er.env_seed.iloc[0])
    eps["env_seed"] = eps.run.map(seeds)
    if a.kind == "recovery":
        return write_recovery(eps, a)
    fs = [first_pot_step(r, a.offset, a.stable) for r in eps.raw]
    eps["first_pot_step"], eps["steady"] = [f for f, _ in fs], [st for _, st in fs]
    early = (eps.first_pot_step >= 0) & (eps.first_pot_step <= a.max_first) & (eps.first_pot_step + a.offset < eps.policy_steps)
    ok = eps[early & eps.steady]
    print(f"{len(eps)} source episodes, {int(early.sum())} with a first pot by step {a.max_first}, {len(ok)} of them steady "
          f"(fixed {int((ok.layout == 'fixed').sum())}, fresh {int((ok.layout == 'fresh').sum())})")
    rng = np.random.default_rng(a.seed)
    pick = []
    for lay, n in (("fixed", a.n_fixed), ("fresh", a.n_fresh)):
        c = ok[ok.layout == lay]
        assert len(c) >= n, (lay, len(c), n)
        pick.append(c.iloc[rng.choice(len(c), n, replace=False)])
    s = pd.concat(pick, ignore_index=True)
    out = pd.DataFrame({"start_id": np.arange(len(s)), "source_raw": s.raw, "start_step": s.first_pot_step + a.offset,
                        "init_state": s.init_state.astype(int), "env_seed": s.env_seed, "layout": s.layout,
                        "source_round": s["round"], "source_success": s.success, "first_pot_step": s.first_pot_step,
                        "source_episode_index": s.episode_index})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.groupby(["layout", "source_round"]).size().to_string())
    print(f"start step median {out.start_step.median():.0f} (IQR {out.start_step.quantile(.25):.0f}-"
          f"{out.start_step.quantile(.75):.0f}); source success {out.source_success.mean():.2f}; "
          f"fixed states {out[out.layout == 'fixed'].init_state.nunique()} -> {a.out}")


def write_recovery(eps: pd.DataFrame, a) -> None:
    """--kind recovery: every eligible source up to --n-fixed / --n-fresh (fewer if not enough), same CSV columns as
    the second-pot starts plus kind (lost / tipped). first_pot_step holds the source's own first-pot step (or -1)."""
    rk = [recovery_step(r) for r in eps.raw]
    eps["start_step"], eps["kind"] = [k for k, _ in rk], [s for _, s in rk]
    eps["first_pot_step"] = [first_pot_step(r, 0, 0)[0] for r in eps.raw]
    ok = eps[(eps.start_step > 0) & (eps.start_step < eps.policy_steps) & (a.max_steps - eps.start_step >= a.min_left)]
    print(f"{len(eps)} source episodes, {int((eps.start_step > 0).sum())} with a lost / tipped pot, {len(ok)} with "
          f">= {a.min_left} steps left (fixed {int((ok.layout == 'fixed').sum())}, fresh {int((ok.layout == 'fresh').sum())})")
    rng = np.random.default_rng(a.seed)
    pick = []
    for lay, n in (("fixed", a.n_fixed), ("fresh", a.n_fresh)):
        c = ok[ok.layout == lay]
        pick.append(c.iloc[rng.choice(len(c), min(n, len(c)), replace=False)])
    s = pd.concat(pick, ignore_index=True)
    out = pd.DataFrame({"start_id": np.arange(len(s)), "source_raw": s.raw, "start_step": s.start_step,
                        "init_state": s.init_state.astype(int), "env_seed": s.env_seed, "layout": s.layout,
                        "source_round": s["round"], "source_success": s.success, "first_pot_step": s.first_pot_step,
                        "source_episode_index": s.episode_index, "kind": s.kind})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.groupby(["layout", "kind"]).size().to_string())
    print(f"start step median {out.start_step.median():.0f}; source success {out.source_success.mean():.2f} -> {a.out}")


if __name__ == "__main__":
    main()
