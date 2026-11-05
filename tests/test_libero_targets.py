"""Check A for the LIBERO targets: hand-computed numbers with the configs/libero_moka.yaml targets block.

One timestep = one policy step; horizon_steps = c_fail_steps = 520 (libero_10 max_steps), so the value scale
is 1040 steps; 201 bins on [-1, 0] (width 0.005 = 5.2 steps).
"""

import numpy as np
import pandas as pd
import pytest

from vl.config import load_config
from vl.libero_targets import manifest_from_episodes
from vl.targets import build_targets, episode_targets

CFG = load_config("configs/libero_moka.yaml")["targets"]


def test_config_is_the_planned_scale():
    assert CFG["horizon_steps"] == 520 and CFG["c_fail_steps"] == 520
    assert CFG["n_bins"] == 201 and CFG["v_min"] == -1.0 and CFG["v_max"] == 0.0
    assert "timeout" in CFG["failure_outcomes"]


def test_success_hand_computed():
    # 412 policy steps (the Stage 1 median success) -> frames 0..411, T = 411.
    e = episode_targets(T=411, fps=10, outcome="success", cfg=CFG)
    assert len(e) == 412
    assert e.iloc[0].return_steps == -411
    assert e.iloc[0].value == pytest.approx(-411 / 1040)          # -0.395192
    assert e.iloc[0].bin == 121                                    # round(120.96)
    assert e.iloc[-1].value == 0.0 and e.iloc[-1].bin == 200


def test_timeout_hand_computed():
    # A timeout runs all 520 steps -> T = 519; first frame: 519 steps to go plus the 520-step penalty.
    e = episode_targets(T=519, fps=10, outcome="timeout", cfg=CFG)
    assert e.iloc[0].return_steps == -1039
    assert e.iloc[0].value == pytest.approx(-1039 / 1040)         # -0.999038, still inside the support
    assert e.iloc[0].bin == 0
    assert e.iloc[-1].value == -0.5 and e.iloc[-1].bin == 100
    assert e.in_support.all()


def test_every_possible_episode_is_in_support_and_separates_at_minus_half():
    for T in (0, 100, 519):
        s = episode_targets(T=T, fps=10, outcome="success", cfg=CFG)
        f = episode_targets(T=T, fps=10, outcome="timeout", cfg=CFG)
        assert s.in_support.all() and f.in_support.all()
        assert (s.value > -0.5).all() and (f.value <= -0.5).all()
    assert np.allclose(np.diff(episode_targets(T=50, fps=10, outcome="success", cfg=CFG).value), 1 / 1040)


def _episodes():
    return pd.DataFrame({"episode_index": [0, 1, 2], "split": ["train", "train", "val"],
                         "init_state": [0, 0, 9], "policy_steps": [412, 520, 300],
                         "success": [1, 0, 1], "timeout": [0, 1, 0]})


def test_manifest_and_targets_from_episode_table():
    man = manifest_from_episodes(_episodes())
    assert list(man["T"]) == [411, 519, 299]
    assert list(man.outcome) == ["success", "timeout", "success"]
    ft = build_targets(man, CFG)
    assert len(ft) == 412 + 520 + 300                              # one row per recorded frame, none after T
    assert ft.groupby("episode_uid").frame_index.max().tolist() == [411, 519, 299]
    assert set(ft[ft.split == "val"].episode_uid) == {"libero_moka/2"}


def test_rejects_episode_that_is_neither_success_nor_timeout():
    ep = _episodes()
    ep.loc[1, "timeout"] = 0                                       # a failure before the step limit: impossible here
    with pytest.raises(ValueError):
        manifest_from_episodes(ep)
