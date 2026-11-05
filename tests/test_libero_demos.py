"""Stage-1 demo targets and split (vl.libero_demos): hand-computed numbers, per-suite normalisation."""

import pandas as pd
import pytest

from vl.config import load_config
from vl.libero_demos import MAX_STEPS, assign_roles, demo_targets, manifest_from_demos

CFG = load_config("configs/libero_moka.yaml")["targets"]


def table(rows):
    return pd.DataFrame(rows, columns=["episode_index", "suite", "task", "policy_steps", "success", "split"])


def test_spatial_demo_hand_computed():
    # 101 frames in libero_spatial (max_steps 220): scale 440 steps; frame 0 is 100 steps from success.
    ft = demo_targets(table([(0, "libero_spatial", "a", 101, 1, "train")]), CFG)
    assert len(ft) == 101
    assert ft.iloc[0].value == pytest.approx(-100 / 440)
    assert ft.iloc[0].bin == round(200 * (1 - 100 / 440))             # 155 (154.55)
    assert ft.iloc[-1].value == 0.0 and ft.iloc[-1].bin == 200
    assert (ft.norm_steps == 440).all()


def test_libero_10_keeps_the_moka_scale():
    # The same 412-step success as tests/test_libero_targets.py: -411 / 1040, bin 121.
    ft = demo_targets(table([(5, "libero_10", "m", 412, 1, "train")]), CFG)
    assert ft.iloc[0].value == pytest.approx(-411 / 1040) and ft.iloc[0].bin == 121


def test_each_suite_uses_its_own_scale():
    rows = [(i, s, s, 51, 1, "train") for i, s in enumerate(["libero_spatial", "libero_object", "libero_goal",
                                                              "libero_10"])]
    ft = demo_targets(table(rows), CFG)
    first = ft[ft.frame_index == 0].set_index("suite").value
    for s in first.index:
        assert first[s] == pytest.approx(-50 / (2 * MAX_STEPS[s]))
    assert CFG["horizon_steps"] == 520  # the base config is not mutated


def test_rejects_failures_and_overlong_demos():
    with pytest.raises(ValueError, match="success"):
        manifest_from_demos(table([(0, "libero_10", "m", 10, 0, "train")]))
    with pytest.raises(ValueError, match="longer than max_steps"):
        demo_targets(table([(0, "libero_spatial", "a", 221, 1, "train")]), CFG)


def test_roles_per_task_deterministic():
    ep = table([(i, "libero_goal", f"t{i % 3}", 50, 1, "train") for i in range(60)])
    r = assign_roles(ep, 0.1, 0)
    assert r.value_counts().to_dict() == {"fit": 54, "inner": 6}
    assert (ep[r == "inner"].groupby("task").size() == 2).all()           # 2 of each task's 20
    assert r.equals(assign_roles(ep.sample(frac=1, random_state=3), 0.1, 0).sort_index())  # row order does not matter
    assert not r.equals(assign_roles(ep, 0.1, 1))


def test_resnet_table_meta_carries_roles(tmp_path):
    from vl.libero_resnet import table_meta
    eps = pd.DataFrame({"episode_index": [0, 1], "split": ["train", "train"], "init_state": [-1, -1],
                        "source_episode": [0, 1], "policy_steps": [9, 5], "success": [1, 1],
                        "role": ["fit", "inner"]})
    eps.to_csv(tmp_path / "episodes.csv", index=False)
    pd.DataFrame([(e, t, -0.1 * t, t) for e, n in [(0, 9), (1, 5)] for t in range(n)],
                 columns=["episode_index", "frame_index", "value", "bin"]).to_parquet(tmp_path / "frame_targets.parquet")
    m = table_meta(tmp_path, 4)
    assert m.role.tolist() == ["fit"] * 3 + ["inner"] * 2
    eps.drop(columns="role").to_csv(tmp_path / "episodes.csv", index=False)
    assert "role" not in table_meta(tmp_path, 4).columns  # moka tables unchanged
