"""Hand-computed checks for the n-step advantage in vl.advantages (normalization 1800 steps)."""

import numpy as np

from vl.advantages import advantage, centred_mean


def two_episodes():
    # episode 0: success, frames 0..4 (T = 4); episode 1: failure, frames 0..2 (T = 2)
    t = np.array([0, 1, 2, 3, 4, 0, 1, 2], float)
    T = np.array([4, 4, 4, 4, 4, 2, 2, 2], float)
    vT = np.array([0, 0, 0, 0, 0, -0.5, -0.5, -0.5])
    starts = np.array([0, 5, 8])
    return t, T, vT, starts


def test_target_as_value_gives_zero_advantage():
    t, T, vT, starts = two_episodes()
    target = -(T - t) / 1800 + vT
    for n in (1, 2, 3, 10):
        A, _ = advantage(target, t, T, vT, starts, n)
        assert np.allclose(A, 0)


def test_interior_and_terminal_hand_computed():
    t, T, vT, starts = two_episodes()
    V = np.array([-0.4, -0.3, -0.3, -0.1, -0.05, -0.2, -0.3, -0.45])
    A, term = advantage(V, t, T, vT, starts, 2)
    # success t=0: interior, -2/1800 + V(2) - V(0)
    assert np.isclose(A[0], -2 / 1800 + 0.1)
    # success t=2: t + n = T, terminal: -(2)/1800 + 0 - V(2)
    assert term[2] and np.isclose(A[2], -2 / 1800 + 0.3)
    # failure t=0: t + n = T, terminal with the failure value; never reads the next episode
    assert term[5] and np.isclose(A[5], -2 / 1800 - 0.5 + 0.2)
    assert term.tolist() == [False, False, True, True, True, True, True, True]


def test_centred_mean_stays_within_episode():
    x = np.array([0, 0, 0, 0, 10, 10, 10], float)
    out = centred_mean(x, np.array([0, 4, 7]), 2)
    assert np.allclose(out[:4], 0) and np.allclose(out[4:], 10)
