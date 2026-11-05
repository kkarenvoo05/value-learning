import numpy as np

from vl.libero_subtask import scoped_targets


def test_scoped_targets_reached_milestone():
    f = np.array([0, 4, 8, 12, 16])
    v = -(16 - f + 520) / 1040.0  # a failed episode's whole-task target
    s, t = scoped_targets(f, v, 8.0, 1040.0)
    assert s.tolist() == [0, 0, 1, 1, 1]
    assert np.allclose(t[:2], [-8 / 1040, -4 / 1040]) and np.allclose(t[2:], v[2:])


def test_scoped_targets_no_milestone_is_unchanged():
    f = np.arange(0, 20, 4)
    v = -(16 - f + 520) / 1040.0
    s, t = scoped_targets(f, v, np.nan, 1040.0)
    assert (s == 0).all() and np.array_equal(t, v)
