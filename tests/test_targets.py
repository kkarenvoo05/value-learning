"""Hand-computed checks on synthetic episodes for value-target construction.

Reference numbers use fps = 15, horizon_s = c_fail_s = 60 (scale 120 s), 201 bins on [-1, 0]
(width 0.005), as in configs/targets.yaml.
"""

import numpy as np
import pandas as pd
import pytest

from vl.targets import bin_centers, build_targets, episode_targets, summarize

CFG = {"horizon_s": 60.0, "c_fail_s": 60.0, "failure_outcomes": ["failure", "timeout"],
       "success_outcomes": ["success"], "n_bins": 201, "v_min": -1.0, "v_max": 0.0}


def test_bin_centers_span_support_with_expected_width():
    c = bin_centers(CFG)
    assert len(c) == 201
    assert c[0] == -1.0 and c[-1] == 0.0
    assert np.allclose(np.diff(c), 0.005)


def test_success_episode_hand_computed():
    # T = 30 at 15 fps: the first frame is 2 s from the end.
    e = episode_targets(T=30, fps=15, outcome="success", cfg=CFG)
    assert len(e) == 31 and list(e.frame_index) == list(range(31))   # nothing after T
    first, last = e.iloc[0], e.iloc[-1]
    assert first.return_s == pytest.approx(-2.0)
    assert first.value == pytest.approx(-2.0 / 120)                    # -0.016667
    assert first.bin == 197                                            # round(196.667)
    assert last.return_s == 0.0 and last.value == 0.0 and last.bin == 200
    assert e.in_support.all()


def test_failure_episode_hand_computed():
    # T = 15 at 15 fps: 1 s to the end plus the 60 s failure penalty.
    e = episode_targets(T=15, fps=15, outcome="failure", cfg=CFG)
    first, last = e.iloc[0], e.iloc[-1]
    assert first.return_s == pytest.approx(-61.0)
    assert first.value == pytest.approx(-61.0 / 120)                   # -0.508333
    assert first.bin == 98                                             # round(98.333)
    assert last.return_s == -60.0 and last.value == -0.5 and last.bin == 100


def test_timeout_gets_the_same_targets_as_failure():
    f = episode_targets(T=40, fps=15, outcome="failure", cfg=CFG)
    t = episode_targets(T=40, fps=15, outcome="timeout", cfg=CFG)
    assert np.array_equal(f.value, t.value) and np.array_equal(f.bin, t.bin)


def test_value_rises_linearly_by_one_frame_per_step():
    e = episode_targets(T=100, fps=15, outcome="success", cfg=CFG)
    assert np.allclose(np.diff(e.value), 1 / (15 * 120))
    assert np.allclose(np.diff(e.time_to_go_s), -1 / 15)


def test_successes_and_failures_separate_at_minus_half():
    s = episode_targets(T=840, fps=15, outcome="success", cfg=CFG)     # longest observed: 56 s
    f = episode_targets(T=0, fps=15, outcome="failure", cfg=CFG)
    assert (s.value > -0.5).all()
    assert (f.value <= -0.5).all()


def test_out_of_support_is_flagged_not_clipped():
    # A 70 s failure: first frame R = -130 s, v = -1.0833, outside [-1, 0].
    e = episode_targets(T=1050, fps=15, outcome="failure", cfg=CFG)
    first = e.iloc[0]
    assert first.value == pytest.approx(-130 / 120)
    assert not first.in_support and first.bin == -1
    # Frames within the last 60 s are back inside the support.
    assert e[e.time_to_go_s <= 60].in_support.all()


def test_value_just_past_the_edge_still_maps_to_the_edge_bin():
    # Anything within half a bin of the edge belongs to the edge bin.
    from vl.targets import value_to_bin
    bins, ok = value_to_bin(np.array([-1.0024, -1.0026, 0.0]), CFG)
    assert list(bins) == [0, -1, 200] and list(ok) == [True, False, True]


def test_unknown_outcome_raises():
    with pytest.raises(ValueError):
        episode_targets(T=10, fps=15, outcome="unknown", cfg=CFG)


def _manifest():
    return pd.DataFrame([
        {"episode_uid": "A:0", "repo": "A", "episode_index": 0, "split": "train",
         "outcome": "success", "T": 30, "fps": 15, "exclude_reason": ""},
        {"episode_uid": "A:1", "repo": "A", "episode_index": 1, "split": "val",
         "outcome": "timeout", "T": 15, "fps": 15, "exclude_reason": ""},
        {"episode_uid": "A:2", "repo": "A", "episode_index": 2, "split": "train",
         "outcome": "unknown", "T": np.nan, "fps": 15, "exclude_reason": "unknown_outcome"},
        {"episode_uid": "X:0", "repo": "X", "episode_index": 0, "split": "audit_only",
         "outcome": "success", "T": 10, "fps": 15, "exclude_reason": "not_in_split"},
        {"episode_uid": "A:3", "repo": "A", "episode_index": 3, "split": "train",
         "outcome": "failure", "T": 1050, "fps": 15, "exclude_reason": ""},
    ])


def test_build_targets_keeps_usable_episodes_only():
    ft = build_targets(_manifest(), CFG)
    assert set(ft.episode_uid) == {"A:0", "A:1", "A:3"}
    assert len(ft) == 31 + 16 + 1051
    assert set(ft.split) == {"train", "val"}
    assert (ft[ft.episode_uid == "A:1"].failed).all()


def test_summary_counts_out_of_support_frames():
    ft = build_targets(_manifest(), CFG)
    s = summarize(ft)
    row = s.set_index(["split", "outcome"]).loc[("train", "failure")]
    # Outside the support means v below -1 by more than half a bin (0.0025 = 0.3 s), i.e.
    # time-to-go > 60.3 s = 904.5 frames: t = 0..145 -> 146 frames.
    assert row.frames == 1051 and row.out_of_support == 146
    assert s.out_of_support.sum() == 146
