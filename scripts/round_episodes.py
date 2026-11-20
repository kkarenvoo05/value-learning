"""[2026-11-18] Train episode indices of one table round, one per line (pi05_policy_finetune.py --episodes).
Stacked AG (scripts/ag_rounds.sh STACK=1) trains each round's head only on the episodes collected by the previous
round's policy, i.e. the data whose behaviour policy is the stacked reference.
    .venv/bin/python scripts/round_episodes.py --table <table dir> --round <table round> --out <txt>
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True)
ap.add_argument("--round", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
e = pd.read_csv(Path(a.table) / "episodes.csv")
sel = e[(e["round"] == a.round) & (e["split"] == "train")].episode_index.astype(int)
assert len(sel) > 0, f"no train episodes in table round {a.round} of {a.table}"
Path(a.out).write_text("".join(f"{i}\n" for i in sel))
print(f"{len(sel)} train episodes of table round {a.round} ({int(e.loc[sel.index, 'success'].sum())} successes) -> {a.out}")
