"""Stage-1 value pre-training on the LIBERO demos (docs/libero_demo_pretrain_draft.md): targets and split.

Pure table logic (project venv); the converter that writes the raw files is scripts/libero_demo_table.py
(openpi env). Every demo is a success, so the target is -(steps to success), with vl.targets unchanged.

Normalisation [RECAP §V-C: "we normalize the values per task based on the maximum episode length of the task";
the per-suite max_steps of openpi's examples/libero/main.py as that length is our choice]: each suite gets the
moka scheme with its own horizon, horizon_steps = c_fail_steps = max_steps, so a success always lands in
(-0.5, 0] and libero_10 (moka's suite) keeps moka's 1040-step scale exactly.

Split [ours]: role "fit" or "inner" per episode; `inner_frac` of each task's demos (rng `seed`, at least one per
task) are held out for stage-1 checkpoint selection. Stage 1 has no separate val set.
"""

import copy

import numpy as np
import pandas as pd

from vl.targets import build_targets

FPS = 10  # physical-intelligence/libero (meta/info.json)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def assign_roles(ep: pd.DataFrame, inner_frac: float = 0.1, seed: int = 0) -> pd.Series:
    """'fit' / 'inner' per row of ep (needs columns task, episode_index); deterministic in (ep, seed)."""
    rng = np.random.default_rng(seed)
    role = pd.Series("fit", index=ep.index)
    for _, g in ep.sort_values("episode_index").groupby("task", sort=True):
        k = max(1, int(round(inner_frac * len(g))))
        if k >= len(g):
            raise ValueError(f"task {g.task.iloc[0]!r} has {len(g)} demos, cannot hold out {k}")
        role.loc[rng.choice(g.index.values, size=k, replace=False)] = "inner"
    return role


def manifest_from_demos(ep: pd.DataFrame) -> pd.DataFrame:
    """vl.targets manifest from the demo episodes.csv (T = policy_steps - 1, outcome success)."""
    if not (ep.success.astype(int) == 1).all():
        raise ValueError("demo table: every episode must be a success")
    return pd.DataFrame({
        "episode_uid": "libero_demos/" + ep.episode_index.astype(str),
        "repo": "libero_demos",
        "episode_index": ep.episode_index.astype(int).values,
        "split": ep.split.values,
        "exclude_reason": "",
        "T": ep.policy_steps.astype(int).values - 1,
        "fps": float(FPS),
        "outcome": "success",
        "suite": ep.suite.values,
    })


def demo_targets(ep: pd.DataFrame, base_cfg: dict) -> pd.DataFrame:
    """frame_targets rows for every demo frame, each suite normalised by its own max_steps."""
    man = manifest_from_demos(ep)
    parts = []
    for suite, m in man.groupby("suite", sort=True):
        if suite not in MAX_STEPS:
            raise ValueError(f"unknown suite {suite!r}")
        too_long = m[m["T"] + 1 > MAX_STEPS[suite]]  # m.T would be the transpose
        if len(too_long):
            raise ValueError(f"{len(too_long)} {suite} demos longer than max_steps {MAX_STEPS[suite]}")
        cfg = copy.deepcopy(base_cfg)
        cfg["horizon_steps"] = cfg["c_fail_steps"] = MAX_STEPS[suite]
        parts.append(build_targets(m, cfg).assign(suite=suite, norm_steps=2 * MAX_STEPS[suite]))
    ft = pd.concat(parts, ignore_index=True)
    assert ft.in_support.all() and (ft.value > -0.5).all() and (ft.value <= 0).all()
    return ft


def main():
    """Write <dataset>/frame_targets.parquet (+ targets_summary.md) from <dataset>/episodes.csv."""
    import argparse
    from pathlib import Path

    from vl.config import load_config
    from vl.targets import summarize
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="demo table dir (scripts/libero_demo_table.py)")
    ap.add_argument("--config", default="configs/libero_moka.yaml", help="targets block; horizons set per suite")
    a = ap.parse_args()
    ds = Path(a.dataset)
    ep = pd.read_csv(ds / "episodes.csv")
    ft = demo_targets(ep, load_config(a.config)["targets"])
    ft = ft.merge(ep[["episode_index", "init_state", "role", "task"]], on="episode_index", validate="many_to_one")
    assert len(ft) == ep.policy_steps.sum()
    ft.to_parquet(ds / "frame_targets.parquet")
    s = ft.groupby(["suite", "role"]).agg(episodes=("episode_index", "nunique"), frames=("value", "size"),
                                           value_min=("value", "min"), value_max=("value", "max")).reset_index()
    (ds / "targets_summary.md").write_text(
        f"# LIBERO demo targets (stage 1)\n\n`python -m vl.libero_demos --dataset {ds}`; per-suite scale "
        f"2 x max_steps {MAX_STEPS}.\n\n" + s.to_markdown(index=False, floatfmt=".4f") + "\n\n"
        + summarize(ft).to_markdown(index=False, floatfmt=".4f") + "\n")
    print(f"wrote {ds / 'frame_targets.parquet'}: {len(ft)} frames\n{s.to_string(index=False)}")


if __name__ == "__main__":
    main()
