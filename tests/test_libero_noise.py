import numpy as np
import pytest

from vl.libero_noise import NOISE_KEY, MatchedNoisePolicy, matched_noise

SHAPE = (10, 32)


def test_same_key_same_noise_regardless_of_order():
    a = [matched_noise([8, k, 0, c], 0, SHAPE) for k in (9, 12) for c in range(3)]
    b = [matched_noise([8, k, 0, c], 0, SHAPE) for k in (12, 9) for c in reversed(range(3))]
    assert np.array_equal(a[0], b[5]) and np.array_equal(a[5], b[0])
    assert a[0].dtype == np.float32 and a[0].shape == SHAPE


@pytest.mark.parametrize("other", [[8, 9, 0, 1], [8, 12, 0, 0], [8, 9, 1, 0], [7, 9, 0, 0]])
def test_every_key_field_changes_the_noise(other):
    assert not np.allclose(matched_noise([8, 9, 0, 0], 0, SHAPE), matched_noise(other, 0, SHAPE))


def test_seed_changes_noise_and_fresh_layout_key_allowed():
    assert not np.allclose(matched_noise([8, 9, 0, 0], 0, SHAPE), matched_noise([8, 9, 0, 0], 1, SHAPE))
    matched_noise([8, -1, 3, 0], 0, SHAPE)  # fresh layouts use init_state -1
    with pytest.raises(AssertionError):
        matched_noise([8, 9, 0], 0, SHAPE)


def test_noise_is_standard_normal():
    x = np.concatenate([matched_noise([8, i, r, c], 0, SHAPE).ravel()
                        for i in range(10) for r in range(3) for c in range(5)])
    assert abs(x.mean()) < 0.02 and abs(x.std() - 1) < 0.02


class _Inner:
    metadata = {}

    def __init__(self):
        self.got = []

    def infer(self, obs, noise=None):
        self.got.append((dict(obs), noise))
        return {"actions": np.zeros(SHAPE)}

    def reset(self):
        pass


def test_wrapper_keys_only_keyed_requests():
    inner = _Inner()
    p = MatchedNoisePolicy(inner, SHAPE, seed=0)
    p.infer({"prompt": "x"})
    p.infer({"prompt": "x", NOISE_KEY: np.array([8, 9, 0, 2])})
    (o0, n0), (o1, n1) = inner.got
    assert n0 is None and NOISE_KEY not in o1
    assert np.array_equal(n1, matched_noise([8, 9, 0, 2], 0, SHAPE))
    with pytest.raises(AssertionError):
        p.infer({"prompt": "x", NOISE_KEY: np.array([8, 9, 0, 2]), "noise_scale": 1.5})
