import numpy as np

from vl import keyframes as K


def _episode(T=120):
    state = np.zeros((T, 8))
    state[:, 0] = np.linspace(0, 1.0, T)          # moving 1 m over the episode...
    state[60:90, 0] = state[60, 0]                 # ...except a stall from 60 to 89
    state[:, 6], state[:, 7] = 0.04, -0.04
    actions = np.zeros((T, 7))
    actions[:, 6] = -1
    actions[30:70, 6] = 1                          # close at 30, open at 70
    return state, actions


def test_gripper_switches():
    _, a = _episode()
    assert K.gripper_switches(a) == [(30, 1), (70, -1)]


def test_stall_mask_matches_definition():
    s, _ = _episode()
    m = K.stall_mask(s)
    assert m[60] and m[69] and not m[70 + 1 + 20] and not m[10]
    assert not m[-K.STALL_N:].any()                # no lookahead past the end


def test_key_frames_windows_and_categories():
    s, a = _episode()
    rng = np.random.default_rng(0)
    for _ in range(20):
        kf = K.key_frames(s, a, rng)
        grip = [(t, d) for t, c, d in kf if c == "grip"]
        assert len(grip) == 2
        for (t, d), (f, d2) in zip(grip, [(30, 1), (70, -1)]):
            assert t <= f <= t + K.EXEC - 1 and d == d2
        stalls = [t for t, c, _ in kf if c == "stall"]
        assert all(K.stall_mask(s)[t] for t in stalls) and len(stalls) == 3
        assert not any(66 <= t <= 70 for t in stalls)  # grip windows excluded from stalls
        assert len([1 for _, c, _ in kf if c == "moving"]) == 2


def test_earlier_switch_sign():
    late = np.array([-1, -1, -1, 1, 1.0])
    early = np.array([-1, 1, 1, 1, 1.0])
    assert K.earlier_switch(early, late, +1) > 0      # closing: positive tag closes earlier
    op_early = np.array([1, -1, -1, -1, -1.0])
    op_late = np.array([1, 1, 1, -1, -1.0])
    assert K.earlier_switch(op_early, op_late, -1) > 0  # opening: positive tag opens earlier
