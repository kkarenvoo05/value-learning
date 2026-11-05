import numpy as np

from vl.libero_advantages import NORM, V_FAIL, advantage


def grid_episode(T, success):
    t = np.arange(0, T + 1, 4, dtype=float)
    vT = 0.0 if success else V_FAIL
    return t, np.full_like(t, T), np.full_like(t, vT), -(T - t) / NORM + vT


def test_target_as_v_gives_zero_advantage():
    parts = [grid_episode(519, False), grid_episode(408, True)]
    t, T, vT, V = (np.concatenate(x) for x in zip(*parts))
    ep = np.repeat([0, 1], [len(p[0]) for p in parts])
    for n in (12, 20, 48):
        A, term = advantage(V, t, T, vT, ep, n)
        assert np.abs(A).max() < 1e-12
        assert (term == (t + n >= T)).all()


def test_constant_v_interior_is_step_cost_and_terminal_uses_outcome():
    t, T, vT, _ = grid_episode(40, False)
    V = np.full_like(t, -0.3)
    A, term = advantage(V, t, T, vT, np.zeros(len(t), int), 20)
    assert np.allclose(A[~term], -20 / NORM)
    assert np.allclose(A[term], -(T[term] - t[term]) / NORM + V_FAIL + 0.3)
