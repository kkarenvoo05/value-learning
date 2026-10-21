"""Proprio helpers: lag sign convention, Euler order, wrap continuity of the 10-d input."""

import numpy as np

from vl.proprio import LAGS, euler_to_matrix, lagged_corr, peak, proprio_vector


def test_lag_sign_video_late():
    # video[k] = speed[k - 3]: the video shows 3 frames late, so the peak must be at L = +3.
    rng = np.random.default_rng(0)
    speed = rng.random(300)
    video = np.r_[np.zeros(3), speed[:-3]]
    curve = np.array([lagged_corr(video, speed, L) for L in LAGS])
    assert peak(curve)[0] == 3


def test_euler_extrinsic_xyz_order():
    # Extrinsic x-y-z = Rz @ Ry @ Rx. A pure 90° yaw maps the x axis to y.
    R = euler_to_matrix(np.array([[0.0, 0.0, np.pi / 2]]))[0]
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    # Roll then yaw: the x axis is unchanged by roll, then yawed onto y.
    R = euler_to_matrix(np.array([[0.3, 0.0, np.pi / 2]]))[0]
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    assert not np.allclose(euler_to_matrix(np.array([[0.3, 0.2, 0.1]]), "xyz"),
                           euler_to_matrix(np.array([[0.3, 0.2, 0.1]]), "XYZ"))


def test_rotation_is_orthonormal():
    e = np.random.default_rng(1).uniform(-np.pi, np.pi, (50, 3))
    R = euler_to_matrix(e)
    assert np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1)


def test_roll_wrap_is_continuous_in_the_input():
    # Roll just below +pi and just above -pi are the same physical pose: the Euler number jumps by
    # ~2 pi, the 10-d input must not.
    s = np.array([[0.5, 0.0, 0.2, np.pi - 1e-3, 0.1, 0.4, 0.8],
                  [0.5, 0.0, 0.2, -np.pi + 1e-3, 0.1, 0.4, 0.8]])
    X = proprio_vector(s)
    assert X.shape == (2, 10)
    assert np.abs(X[0] - X[1]).max() < 1e-2
    assert np.allclose(X[:, :3], s[:, :3]) and np.allclose(X[:, 9], s[:, 6])
