"""CPU statistics for the fixed-checkpoint layer diagnostic (our design, 2026-11-05).

No training targets or splits are modified. The historical 5-grid-frame change is 1 s at 20 Hz.
"""

import numpy as np


def aligned_after_action(x):
    """Recorded goals/object positions follow action t; image t precedes it."""
    x = np.asarray(x)
    return np.concatenate([x[:1], x[:-1]], axis=0)


def per_seed_metrics(values, targets, success, frames, weights=None):
    """Metrics on [seed, episode, frame] arrays; integer weights replicate entire episodes.

    Recompute outcome/time means on each draw. Noise is residual SD, including real differences
    in task progress; it is not an estimate of exclusively erroneous prediction changes.
    """
    v = np.asarray(values, np.float64)
    y = np.asarray(targets, np.float64)
    w = np.ones(v.shape[1]) if weights is None else np.asarray(weights, np.float64)
    ok = np.isfinite(y)
    assert v.ndim == 3 and v.shape[1:] == y.shape
    assert np.array_equal(np.isfinite(v), np.broadcast_to(ok, v.shape))
    count = (ok * w[:, None]).sum()
    result = {"mae": (np.where(ok[None], np.abs(v - y[None]), 0) * w[None, :, None]).sum((1, 2)) / count}
    for lag, name in ((1, "0p2s"), (5, "1s")):
        d = v[:, :, lag:] - v[:, :, :-lag]
        eligible = (frames[lag:] <= 340) & ((frames[lag:] - frames[:-lag]) == 4 * lag)
        good = np.isfinite(d) & eligible[None, None]
        weighted = good * w[None, :, None]
        d0 = np.where(good, d, 0)
        means, residual_sums = [], []
        for outcome in (0, 1):
            wo = weighted * (np.asarray(success) == outcome)[None, :, None]
            n = wo.sum(1)
            total = (d0 * wo).sum(1)
            mean = np.divide(total, n, out=np.full_like(total, np.nan), where=n > 0)
            means.append(mean)
            ss = (d0 * d0 * wo).sum(1) - np.divide(total * total, n, out=np.zeros_like(total), where=n > 0)
            residual_sums.append(ss.sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            signal = np.nanmean(means[1] - means[0], axis=1)
            noise = np.sqrt(np.maximum(sum(residual_sums), 0) / (weighted.sum((1, 2)) - 1))
            result[f"signal_{name}"] = signal
            result[f"noise_{name}"] = noise
            result[f"snr_{name}"] = signal / noise
    return result


def hierarchical_comparison(a, b, targets, success, frames, scenes, draws=2000, seed=20261105):
    """B minus A: paired scenes/episodes; independent training-seed resampling per arm."""
    rng = np.random.default_rng(seed)
    unique = np.unique(scenes)
    groups = {s: np.flatnonzero(scenes == s) for s in unique}
    ma, mb = [per_seed_metrics(v, targets, success, frames) for v in (a, b)]
    samples = {k: [] for k in ma}
    for _ in range(draws):
        episodes = np.concatenate([rng.choice(groups[s], len(groups[s])) for s in rng.choice(unique, len(unique))])
        weights = np.bincount(episodes, minlength=len(scenes))
        aa, bb = [per_seed_metrics(v, targets, success, frames, weights) for v in (a, b)]
        ia, ib = rng.integers(a.shape[0], size=a.shape[0]), rng.integers(b.shape[0], size=b.shape[0])
        for k in samples:
            samples[k].append(float(bb[k][ib].mean() - aa[k][ia].mean()))
    return {k: {"difference": float(mb[k].mean() - ma[k].mean()),
                "ci95": np.nanpercentile(samples[k], [2.5, 97.5]).tolist(),
                "finite_draws": int(np.isfinite(samples[k]).sum())} for k in samples}


def pair_metadata(goals, object_positions, state, t0, t1, margin=10, hold=10):
    """Observational filters, not a claim that true state value is unchanged."""
    goals = aligned_after_action(goals).astype(bool)
    pos = aligned_after_action(object_positions)
    lo, hi = max(0, t0 - margin), min(len(goals), t1 + margin + 1)
    transitions = np.any(goals[lo + 1:hi] != goals[lo:hi - 1])
    displacement = np.linalg.norm(pos[t0:t1 + 1] - pos[t0], axis=-1)
    motion = float(displacement.max())
    placement = False
    for pot in range(2):
        # A transition in (t0, t1], sustained for a full HOLD steps.
        for t in range(t0 + 1, t1 + 1):
            if not goals[t - 1, pot] and goals[t, pot] and t + hold <= len(goals):
                placement |= bool(goals[t:t + hold, pot].all())
    return {"no_goal_transition": bool(not transitions), "pot_motion_m": motion,
            "eef_motion_m": float(np.linalg.norm(state[t1, :3] - state[t0, :3])),
            "gripper_motion": float(np.linalg.norm(state[t1, 6:8] - state[t0, 6:8])),
            "placement": bool(placement), "pot_count_a": int(goals[t0, :2].sum()),
            "pot_count_b": int(goals[t1, :2].sum())}
