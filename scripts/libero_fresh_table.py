"""Combined fixed + fresh-layout episode table for the fresh-layout arms (docs/libero_plan.md 2026-10-31) [ours].

Usage: .venv/bin/python scripts/libero_fresh_table.py --fresh-run <runs/fresh450_libero_10_<job>> --out <table dir>

Writes <out>/episodes.csv in scripts/libero_to_lerobot.py's format, plus a `layout` column:
  - every Stage 2 episode (fixed layouts; train and val) with its LeRobot episode_index unchanged and `raw` made
    absolute, so the inner / val episodes and their ids are the same as in every earlier run;
  - every fresh-layout episode of --fresh-run as split "train", init_state -1, episode_index 1000 + its episode
    number (no collision with Stage 2's 0..569).
Then <out>/frame_targets.parquet with vl.libero_targets (unchanged), checked equal to the Stage 2 targets on the
Stage 2 episodes, and three fit lists in <out>/subsets/:
  fit_fixed.txt          the 375 Stage 2 fit episodes (= every earlier full-data run)
  fit_fresh375_s0.txt    375 of the fresh episodes (rng seed 0): matched in size to fit_fixed
  fit_fixed_fresh.txt    fit_fixed + every fresh episode
Inner (early stopping on 5 fixed held-out train states) and val (10 fixed held-out states) are the same for all arms.
The trainers read raw files directly, so no LeRobot conversion is needed for training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vl.config import load_config
from vl.libero_targets import manifest_from_episodes
from vl.targets import build_targets

ROOT = Path("/iris/u/ankile/karenvo/libero")
STAGE2 = ROOT / "lerobot" / "libero_moka_stage2"
STAGE2_RUN = ROOT / "runs" / "stage2_moka_18155849"
INNER = {10, 16, 22, 26, 35}  # vl.libero_probe.inner_split's states (asserted by the trainers)
FRESH_ID0 = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="/move/u/karenvo/Projects/value-learning/configs/libero_moka.yaml")
    a = ap.parse_args()
    out, run = Path(a.out), Path(a.fresh_run)
    (out / "subsets").mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(STAGE2 / "episodes.csv", dtype={"goals_end": str})
    old["raw"] = [str(STAGE2_RUN / r) for r in old.raw]
    old["layout"] = "fixed"

    fr = pd.read_csv(run / "episodes.csv", dtype={"goals_end": str})
    bad = fr[fr.error.notna() | fr.raw.isna()]
    assert bad.empty, f"{len(bad)} fresh episodes with a client error"
    # split is whatever --args.split-name the collection got ("fresh" in the check run, empty in fresh450)
    assert (fr.init_state == -1).all() and fr.split.fillna("fresh").eq("fresh").all()
    new = pd.DataFrame({
        "episode_index": FRESH_ID0 + fr.episode, "run": run.name, "split": "train", "init_state": -1,
        "repeat": fr.repeat, "source_episode": fr.episode, "policy_steps": fr.policy_steps, "success": fr.success,
        "timeout": fr.timeout, "goals_end": fr.goals_end, "raw": [str(run / "raw" / r) for r in fr.raw],
        "layout": "fresh"})
    for r in new.itertuples():  # frames come from len(state) in the trainers; it must equal policy_steps
        with np.load(r.raw) as z:
            assert len(z["state"]) == r.policy_steps, (r.raw, len(z["state"]), r.policy_steps)
    eps = pd.concat([old, new], ignore_index=True)
    assert eps.episode_index.is_unique
    eps.to_csv(out / "episodes.csv", index=False)

    cfg = load_config(a.config)
    man = manifest_from_episodes(eps)
    ft = build_targets(man, cfg["targets"]).merge(man[["episode_uid", "init_state"]], on="episode_uid")
    ref = pd.read_parquet(STAGE2 / "frame_targets.parquet")
    mine = ft[ft.episode_index.isin(old.episode_index)].sort_values(["episode_index", "frame_index"])
    ref = ref.sort_values(["episode_index", "frame_index"])
    for c in ["episode_index", "frame_index", "value", "bin"]:
        assert np.array_equal(mine[c].to_numpy(), ref[c].to_numpy()), f"Stage 2 targets differ in {c}"
    ft.to_parquet(out / "frame_targets.parquet")

    tr = old[old.split == "train"]
    fit_fixed = tr[~tr.init_state.isin(INNER)].episode_index.tolist()
    assert len(fit_fixed) == 375
    fresh = new.episode_index.to_numpy()
    fresh375 = sorted(np.random.default_rng(0).choice(fresh, size=min(375, len(fresh)), replace=False).tolist())
    lists = {"fit_fixed.txt": fit_fixed, "fit_fresh375_s0.txt": fresh375,
             "fit_fixed_fresh.txt": fit_fixed + fresh.tolist()}
    for name, ids in lists.items():
        (out / "subsets" / name).write_text(f"# {name}: {len(ids)} episodes ({run.name})\n"
                                            + "\n".join(map(str, ids)) + "\n")
    s = eps.groupby(["layout", "split"]).agg(episodes=("episode_index", "size"), success=("success", "mean"),
                                             mean_steps=("policy_steps", "mean"))
    print(s.to_string())
    sub = eps.set_index("episode_index").loc[fresh375]
    print(f"fresh375: success {sub.success.mean():.3f}, mean steps {sub.policy_steps.mean():.0f}")
    print(f"wrote {out}: {len(eps)} episodes, {len(ft)} target frames; Stage 2 targets reproduced exactly")


if __name__ == "__main__":
    main()
