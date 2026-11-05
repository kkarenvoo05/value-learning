"""Synthetic checks for the progress probe's label and time-matched AUC (vl.libero_progress)."""

import numpy as np
import pytest

from vl.libero_progress import clock_scores, pots_on, time_matched_auc


def test_pots_on_shows_the_state_after_the_previous_action():
    goals = np.zeros((10, 3), bool)
    goals[:, 2] = True               # third predicate (stove) is ignored
    goals[4:, 0] = True              # pot 1 on after action 4
    goals[7:, 1] = True              # pot 2 on after action 7
    t = np.arange(10)
    assert pots_on(goals, t).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
    assert pots_on(goals, np.array([0]))[0] == 0  # image 0 shows no action yet


def test_a_clock_scores_one_half_on_the_time_matched_auc():
    rng = np.random.default_rng(0)
    t = np.repeat(np.arange(0, 400, 4), 30)
    y = (rng.random(len(t)) < t / 400).astype(int)   # pots are on more often late: plain AUC rewards time
    s = clock_scores(t, y, t)
    assert time_matched_auc(y, s, t) == pytest.approx(0.5)
    from vl.metrics import auc
    assert auc(y, s) > 0.7


def test_time_matched_auc_rewards_reading_the_label_at_fixed_time():
    rng = np.random.default_rng(1)
    t = np.repeat(np.arange(0, 400, 4), 30)
    y = (rng.random(len(t)) < 0.5).astype(int)
    assert time_matched_auc(y, y + 0.01 * rng.random(len(t)), t) == pytest.approx(1.0)
    assert time_matched_auc(y, rng.random(len(t)), t) == pytest.approx(0.5, abs=0.03)


def test_time_matched_auc_skips_bins_with_one_class():
    t = np.array([0, 4, 40, 44])
    y = np.array([0, 0, 0, 1])          # only the second 2 s bin has both classes
    s = np.array([0.9, 0.9, 0.1, 0.2])  # the first bin's high scores on negatives must not count
    assert time_matched_auc(y, s, t) == pytest.approx(1.0)
    assert np.isnan(time_matched_auc(np.zeros(4, int), s, t))
