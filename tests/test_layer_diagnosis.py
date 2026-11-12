import numpy as np

from vl.layer_diagnosis import aligned_after_action, pair_metadata, per_seed_metrics


def test_episode_weights_equal_literal_resampling():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(2, 4, 15))
    y = rng.normal(size=(4, 15))
    v[:, 2, -2:], y[2, -2:] = np.nan, np.nan
    success = np.array([0, 1, 0, 1])
    frames = np.arange(15) * 4
    take = np.array([0, 0, 1, 3, 3])
    weighted = per_seed_metrics(v, y, success, frames, np.bincount(take, minlength=4))
    expanded = per_seed_metrics(v[:, take], y[take], success[take], frames)
    for key in weighted:
        np.testing.assert_allclose(weighted[key], expanded[key], rtol=1e-12, atol=1e-12)


def test_goals_are_aligned_to_observation_and_require_complete_hold():
    goals = np.zeros((40, 3), bool)
    goals[14:, 0] = True  # Visible beginning at observation 15.
    state = np.zeros((40, 8))
    positions = np.zeros((40, 2, 3))
    assert not aligned_after_action(goals)[14, 0]
    assert aligned_after_action(goals)[15, 0]
    assert pair_metadata(goals, positions, state, 12, 16)["placement"]
    assert not pair_metadata(goals, positions, state, 8, 12)["placement"]
    assert not pair_metadata(goals, positions, state, 12, 16)["no_goal_transition"]
    goals[:] = False
    goals[37:, 0] = True
    assert not pair_metadata(goals, positions, state, 35, 39)["placement"]


def test_two_simultaneous_goal_changes_are_not_hidden_by_equal_count():
    goals = np.zeros((50, 3), bool)
    goals[:25, 0] = True
    goals[25:, 1] = True
    result = pair_metadata(goals, np.zeros((50, 2, 3)), np.zeros((50, 8)), 24, 28)
    assert result["pot_count_a"] == result["pot_count_b"] == 1
    assert not result["no_goal_transition"]
