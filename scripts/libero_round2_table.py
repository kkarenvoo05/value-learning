"""RECAP round 2 (item 4, 2026-11-07; paper: pi*0.6 Alg. 1 / Sec. V-D): append a policy's new rollouts to the table.

Usage: .venv/bin/python scripts/libero_round2_table.py --base <table dir> --runs <collection run> [<run> ...] \
           --round 2 --out <new table dir>

RECAP collects data with the current policy, then trains V and the policy "on all of the data collected for the task
so far", each time from the pre-trained checkpoint. Here: every row of --base (round 1 and earlier) unchanged, plus
every episode of each collection run (scripts/libero_rollouts.py with --args.save-raw: the 30 fixed train states and
/ or fresh layouts) as split "train", episode_index = id_base(round) + k * 10000 + episode (run k), `round` column;
id_base = round * 100000, one block higher from round 10 on (2026-11-11) so ids skip the demo range DEMO_OFFSET + i.
Refuses held-out (val / test) init states, client errors and frame-count mismatches. Then frame_targets.parquet
(vl.libero_targets, unchanged config), checked equal to --base's targets on --base's episodes. Trainers read raw
files directly; vl.libero_resnet --every 4 builds its frame cache on first use.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vl.config import load_config
from vl.libero_finetune_data import DEMO_OFFSET, held_out_states
from vl.libero_targets import manifest_from_episodes
from vl.targets import build_targets

KEY = ["episode_index", "frame_index"]


def id_base(rnd: int) -> int:
    """First episode_index of a round: round * 100000; rounds >= 10 one block up (1,100,000, ...) past the demo ids.
    Rounds 2-9 unchanged, so tables built before 2026-11-11 keep their ids."""
    return rnd * 100000 + (100000 if rnd >= 10 else 0)


def collected(run: Path, base_id: int) -> pd.DataFrame:
    fr = pd.read_csv(run / "episodes.csv", dtype={"goals_end": str})
    bad = fr[fr.error.notna() | fr.raw.isna()]
    assert bad.empty, f"{run}: {len(bad)} episodes with a client error or no raw file"
    held = held_out_states()
    assert not set(fr.init_state[fr.init_state >= 0]) & held, f"{run}: held-out init states"
    out = pd.DataFrame({
        "episode_index": base_id + fr.episode, "run": run.name, "split": "train", "init_state": fr.init_state,
        "repeat": fr.repeat, "source_episode": fr.episode, "policy_steps": fr.policy_steps, "success": fr.success,
        "timeout": fr.timeout, "goals_end": fr.goals_end, "raw": [str(run / "raw" / r) for r in fr.raw],
        "layout": np.where(fr.init_state < 0, "fresh", "fixed")})
    for r in out.itertuples():
        with np.load(r.raw) as z:
            assert len(z["state"]) == r.policy_steps, (r.raw, len(z["state"]), r.policy_steps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "libero_moka.yaml"))
    a = ap.parse_args()
    assert a.round >= 2
    base, out = Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(base / "episodes.csv", dtype={"goals_end": str})
    if "round" not in old.columns:
        old["round"] = 1
    new = pd.concat([collected(Path(r), id_base(a.round) + k * 10000) for k, r in enumerate(a.runs)],
                    ignore_index=True).assign(round=a.round)
    eps = pd.concat([old, new], ignore_index=True)
    assert eps.episode_index.is_unique
    assert not new.episode_index.between(DEMO_OFFSET, DEMO_OFFSET + 99999).any(), "round ids collide with demo ids"
    eps.to_csv(out / "episodes.csv", index=False)
    cfg = load_config(a.config)
    man = manifest_from_episodes(eps)
    ft = build_targets(man, cfg["targets"]).merge(man[["episode_uid", "init_state"]], on="episode_uid")
    ref = pd.read_parquet(base / "frame_targets.parquet").sort_values(KEY)
    mine = ft[ft.episode_index.isin(old.episode_index)].sort_values(KEY)
    for c in KEY + ["value", "bin"]:
        assert np.array_equal(mine[c].to_numpy(), ref[c].to_numpy()), f"base targets differ in {c}"
    ft.to_parquet(out / "frame_targets.parquet")
    print(eps.groupby(["round", "layout", "split"]).agg(episodes=("episode_index", "size"),
                                                        success=("success", "mean")).to_string())
    print(f"wrote {out}: {len(eps)} episodes ({len(new)} new), {len(ft)} target frames; base targets reproduced")


if __name__ == "__main__":
    main()
