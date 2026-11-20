"""Distillation table (2026-11-14; Karen: "set up the distillation experiment with the chunk target fix").

    .venv/bin/python scripts/libero_distill_table.py --run <collection run> --out <table dir>

= scripts/libero_round2_table.py --base tables/moka_fixed_fresh --round 2 --runs <run> --out <table dir> (round-0
rows unchanged, the collection appended as round 2, frame targets rebuilt and checked), then checks that every new
raw file carries the sampled chunks the fine-tune trains on (chunk_actions / chunk_t, scripts/libero_rollouts.py
Changes 8; vl.libero_finetune_data.episode_chunks), and writes <out>/new_episodes.txt (the round-2 episode ids, for
pi05_policy_finetune.py --episodes; the imitation arm keeps their successes) and <out>/distill_summary.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vl.libero_finetune_data import episode_chunks

BASE = "/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "libero_round2_table.py"), "--base", BASE, "--round", "2",
                    "--runs", a.run, "--out", a.out], check=True)
    out = Path(a.out)
    eps = pd.read_csv(out / "episodes.csv", dtype={"goals_end": str})
    new = eps[eps["round"] == 2].sort_values("episode_index")
    assert len(new) and (new.split == "train").all()
    replan, switch, actor_share = [], [], []
    for r in new.itertuples():
        with np.load(r.raw) as z:
            ch = episode_chunks(z, int(r.policy_steps))
            assert ch is not None, f"{r.raw}: no saved chunks (collected before Changes 8?)"
            assert np.array_equal(np.flatnonzero(z["new_chunk"]), ch[0]), r.raw
            assert np.allclose(ch[1][:, 0], z["actions"][ch[0]]), r.raw
            replan.append(len(ch[0]))
            if "switch_step" in z.files:
                switch.append(int(z["switch_step"]))
                actor_share.append(float(np.mean(z["chunk_actor"][: len(ch[0])])))
    new = new.assign(replan_frames=replan)
    (out / "new_episodes.txt").write_text("".join(f"{e}\n" for e in new.episode_index))
    s = new.success == 1
    info = {"run": a.run, "episodes": len(new), "success": float(s.mean()),
            "success_by_layout": new.groupby("layout").success.mean().to_dict(),
            "successes": int(s.sum()), "replan_frames_successes": int(new.replan_frames[s].sum()),
            "executed_frames_successes": int(new.policy_steps[s].sum())}
    if switch:
        sw = np.array(switch)
        info.update({"switch_step_median": float(np.median(sw)), "switch_step_max": int(sw.max()),
                     "actor_chunk_share_mean": float(np.mean(actor_share))})
        if np.unique(sw).size >= 3:  # the dsrl collection switches at 0 throughout: no terciles
            info["success_by_switch_tercile"] = new.assign(sw=sw).groupby(
                pd.qcut(sw, 3, labels=["early", "mid", "late"]), observed=True).success.mean().to_dict()
    (out / "distill_summary.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(json.dumps(info, default=str), flush=True)


if __name__ == "__main__":
    main()
