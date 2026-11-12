"""Fitted n-step TD value targets and value-binned label sets for the moka train rollouts [2026-11-07, item 3].

Why: our V is Monte Carlo (each frame's target is that episode's realised return), so the luck of everything after
frame t sits in the target. Dissecting Advantage-Guided Post-Training (arXiv 2609.28161) found n-step TD critics
order frames better than MC ones (MC "generally closer to chance"); SeeQ (2609.22085, Table 2) ranks task-level MC
the noisiest objective. RECAP itself uses MC [paper: Eq. 1].

Target [ours: one fitted iteration, no target network]: with V0 = the 5-fold out-of-fold ResNet V (every train frame
predicted by a model that never fit it),
    y_t = -n/1040 + V0(o_{t+n})                    if t + n < T   (interior: bootstrapped on V0 after n frames)
    y_t = -(T - t)/1040 + v_T                      if t + n >= T  (terminal: the outcome, as the MC target)
i.e. y = A + V0 with A = vl.libero_advantages.advantage (the same n-step estimate the labels use), so the luck
after t + n is replaced by V0's expectation. A new 5-fold OOF ResNet trained on y (vl.libero_resnet --targets)
gives V1; labels then come from V1 exactly as the current ones come from V0. With V0 = the MC target itself, y equals
the MC target (fixed point; tested).

Labels: A = advantage(V, n = 20), value-binned (Dissecting Eq. 4, scripts/libero_value_calibration.value_binned),
40% positive [paper: RECAP App. A-F fine-tuning share] = the rule of the current `resnet_valuebinned_p40` labels.
numpy / pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vl.libero_advantages import V_FAIL, advantage

BINS, MIN_BIN, SIG_FLOOR = 8, 32, 0.1  # = scripts/libero_value_calibration.py
KEY = ["episode_index", "frame_index"]


def td_targets(d: pd.DataFrame, V: np.ndarray, n: int = 20) -> np.ndarray:
    """d: one row per grid frame (episode_index, frame_index, T, success), sorted by episode then frame; V: V0 on
    those rows. Returns the n-step bootstrapped target y (same rows)."""
    _check_sorted(d)
    t, T, ep = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.episode_index.to_numpy()
    vT = np.where(d.success.to_numpy() == 1, 0.0, V_FAIL)
    A, _ = advantage(np.asarray(V, float), t, T, vT, ep, n)
    return A + np.asarray(V, float)


def value_binned(A: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Dissecting Eq. 4 as scripts/libero_value_calibration.value_binned (8 equal bins of V on [-1, 0])."""
    edges = np.linspace(-1, 0, BINS + 1)
    b = np.clip(np.digitize(V, edges[1:-1]), 0, BINS - 1)
    mu, sd = A.mean(), A.std()
    z = np.empty_like(A)
    for k in range(BINS):
        m = b == k
        if m.sum() >= MIN_BIN:
            z[m] = (A[m] - A[m].mean()) / max(A[m].std(), SIG_FLOOR * sd)
        else:
            z[m] = (A[m] - mu) / sd
    return z


def labels_from_v(d: pd.DataFrame, V: np.ndarray, n: int = 20, pos_share: float = 0.4, binned: bool = True,
                  groups: np.ndarray | None = None) -> pd.DataFrame:
    """Value-binned n-step labels from a V on the train grid: episode_index, frame_index, positive, A, z, V, term.

    binned=False [2026-11-08, RECAP's exact rule, paper Eq. 5 / App. A-F]: one threshold on the raw advantage, z = A.
    groups (one value per row, e.g. the collection round): the pos_share cut is taken within each group instead of
    over the whole pool [RECAP: ~40% of each round's data]; value bins (if binned) stay pooled."""
    _check_sorted(d)
    t, T, ep = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.episode_index.to_numpy()
    vT = np.where(d.success.to_numpy() == 1, 0.0, V_FAIL)
    V = np.asarray(V, float)
    A, term = advantage(V, t, T, vT, ep, n)
    z = value_binned(A, V) if binned else A.copy()
    if groups is None:
        groups = np.zeros(len(z), int)
    groups = np.asarray(groups)
    pos = np.zeros(len(z), np.int8)
    for g in np.unique(groups):
        m = groups == g
        pos[m] = z[m] > np.percentile(z[m], 100 * (1 - pos_share))
    return pd.DataFrame({"episode_index": d.episode_index.to_numpy(), "frame_index": d.frame_index.to_numpy(),
                         "positive": pos, "A": A, "z": z, "V": V, "term": term})


HINDSIGHT_N = 524  # > the longest moka episode (T <= 519), multiple of the grid step 4


def hindsight_labels(d: pd.DataFrame, V: np.ndarray, pos_share: float = 0.4) -> pd.DataFrame:
    """Coach test [2026-11-08, ours]: the current label rule with the lookahead stretched past the episode end, so
    every frame is terminal and A = G_t - V(t), G_t = -(T - t)/1040 + v_T the realised return (what actually
    happened next) instead of V's guess at t + n. V only sets the baseline and the value bins. Same value-binned
    40% rule as labels_from_v."""
    assert (d["T"] < HINDSIGHT_N).all()
    lab = labels_from_v(d, V, HINDSIGHT_N, pos_share)
    assert lab.term.all()
    return lab


def _check_sorted(d: pd.DataFrame):
    k = d[KEY].to_numpy()
    assert (np.lexsort((k[:, 1], k[:, 0])) == np.arange(len(k))).all(), "rows must be sorted by episode, frame"
    assert not d.duplicated(KEY).any()
