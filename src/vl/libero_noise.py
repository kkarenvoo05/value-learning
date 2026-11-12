"""Matched (common-random-numbers) flow-matching noise for LIBERO evals [ours, 2026-11-06].

Without it, the policy server draws each chunk's start noise from one rng stream split on every infer call, so the
noise an episode sees depends on how many calls came before it (episode lengths, episode order): two arms evaluated
on the same init states and env seed never share noise, and their per-episode outcomes are only weakly paired.

With it, the client tags every request with a key (task_id, init_state, repeat, chunk index within the episode) and
the server draws the start noise from that key alone. Episode (task, init, repeat) then gets the same noise for its
k-th chunk in every eval with the same --noise-seed, whatever the order or the length of other episodes. The noise
is still N(0, I), so each arm is the same stochastic policy as before; only the comparison between arms is paired.
"""

import numpy as np

NOISE_KEY = "noise_key"  # observation field: int array [task_id, init_state, repeat, chunk]


def matched_noise(key, seed: int, shape) -> np.ndarray:
    """N(0, I) float32 of `shape`, a deterministic function of (seed, key). init_state -1 (fresh layouts) is allowed."""
    k = [int(x) for x in np.asarray(key).reshape(-1)]
    assert len(k) == 4, f"noise key must be [task_id, init_state, repeat, chunk], got {k}"
    entropy = [int(seed), k[0], k[1] + 1, k[2], k[3]]  # SeedSequence needs non-negative ints
    assert min(entropy) >= 0, f"negative noise key entry: {k}"
    return np.random.default_rng(np.random.SeedSequence(entropy)).standard_normal(tuple(shape)).astype(np.float32)


class MatchedNoisePolicy:
    """Wraps a served policy whose infer(obs, noise=...) accepts explicit start noise. Requests carrying NOISE_KEY get
    matched_noise(key); requests without it are passed through unchanged."""

    def __init__(self, policy, shape, seed: int = 0):
        self._policy, self._shape, self._seed, self.calls = policy, tuple(shape), int(seed), 0

    @property
    def metadata(self):
        return self._policy.metadata

    def reset(self):
        self._policy.reset()

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        key = obs.pop(NOISE_KEY, None)
        if key is None:
            return self._policy.infer(obs)
        assert "noise_scale" not in obs, "matched noise and noise_scale together are not supported"
        self.calls += 1
        return self._policy.infer(obs, noise=matched_noise(key, self._seed, self._shape))
