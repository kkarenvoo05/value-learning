"""Synthetic checks for the Stage 3a probe's clock baseline and metrics (vl.libero_probe)."""

import numpy as np
import pandas as pd
import pytest

from vl.libero_probe import CENTERS, N_BINS, clock_probs, inner_split, metrics


def _frames(eps):
    """eps: list of (episode_index, init_state, success, T, split); one row per 4th frame, bins from the targets."""
    rows = []
    for e, s, ok, T, split in eps:
        for t in range(0, T + 1, 4):
            v = -(T - t) / 1040 - (0 if ok else 0.5)
            rows.append({"episode_index": e, "init_state": s, "success": ok, "T": T, "split": split,
                         "frame_index": t, "value": v, "bin": int(np.rint((v + 1) / 0.005))})
    return pd.DataFrame(rows)


def test_clock_is_a_distribution_per_frame_and_knows_only_time():
    tr = _frames([(0, 0, 1, 400, "train"), (1, 1, 0, 519, "train")])
    p = clock_probs(tr, tr)
    assert p.shape == (len(tr), N_BINS) and np.allclose(p.sum(1), 1)
    # two frames with the same frame index get the same prediction, whatever their episode
    same_t = tr[tr.frame_index == 100].index
    assert np.allclose(p[same_t[0]], p[same_t[1]])


def test_perfect_prediction_scores_zero_mae_and_auc_one():
    d = _frames([(0, 0, 1, 400, "val"), (1, 1, 0, 519, "val"), (2, 2, 1, 380, "val"), (3, 3, 0, 519, "val")])
    p = np.zeros((len(d), N_BINS))
    p[np.arange(len(d)), d.bin.values] = 1
    m = metrics(p, d)
    assert m["mae"] < 0.0026                       # half a bin width: binning error only
    assert m["ce"] == pytest.approx(0, abs=1e-9)
    assert all(m[f"auc_{s}s"] == 1.0 for s in (5, 10, 20, 30))


def test_uniform_prediction_has_log_201_ce_and_chance_auc():
    d = _frames([(0, 0, 1, 400, "val"), (1, 1, 0, 519, "val")])
    m = metrics(np.full((len(d), N_BINS), 1 / N_BINS), d)
    assert m["ce"] == pytest.approx(np.log(N_BINS))
    assert m["auc_10s"] == 0.5
    assert CENTERS[0] == -1 and CENTERS[-1] == 0


def test_inner_split_holds_out_whole_train_states_only():
    d = _frames([(i, i, i % 2, 400, "train") for i in range(10)] + [(10, 20, 1, 400, "val")])
    m = inner_split(d, k=3)
    held = set(d[m].init_state)
    assert len(held) == 3 and not (d[m].split == "val").any()
    assert not set(d[~m & (d.split == "train")].init_state) & held


def test_outcome_clock_mixture_is_a_distribution_and_certain_outcome_picks_its_histogram():
    from vl.libero_probe import clock_by_outcome, outcome_clock_probs
    # 20 failures, as with real counts (~180 per frame index): the 0.1-per-bin smoothing does not swamp them
    tr = _frames([(0, 0, 1, 400, "train"), (2, 2, 1, 360, "train")] + [(e, e, 0, 519, "train") for e in range(3, 23)])
    hs, hf = clock_by_outcome(tr, 520)
    fr = tr.frame_index.values
    p = outcome_clock_probs(np.full(len(tr), 0.3), fr, hs, hf)
    assert np.allclose(p.sum(1), 1)
    assert np.allclose(outcome_clock_probs(np.ones(len(tr)), fr, hs, hf), hs[fr])
    # a failure frame's true bin gets most of the failure histogram's mass (failures all end at step 519)
    f = tr[tr.success == 0].iloc[10]
    assert hf[f.frame_index, int(f.bin)] > 0.5


def test_history_appends_the_same_episodes_past_frame_and_zeros_before_the_start():
    from vl.libero_probe import with_history
    d = _frames([(0, 0, 1, 40, "train"), (1, 1, 0, 40, "train")])            # frames 0, 4, ..., 40
    X = np.stack([d.episode_index.values * 1000 + d.frame_index.values, np.ones(len(d))], 1).astype(np.float32)
    H = with_history(X, d, [8])
    assert H.shape == (len(d), 2 + 2 + 1)
    r = d[(d.episode_index == 1) & (d.frame_index == 20)].index[0]
    assert H[r, 2] == 1000 + 12 and H[r, 4] == 1                              # same episode, 8 frames back, has = 1
    r0 = d[(d.episode_index == 1) & (d.frame_index == 4)].index[0]
    assert H[r0, 2] == 0 and H[r0, 3] == 0 and H[r0, 4] == 0                  # before the start: zeros, has = 0
    with pytest.raises(ValueError):
        with_history(X, d, [6])                                              # not on the 4-frame grid


def test_coarsen_grid_is_the_exact_block_mean_and_keeps_the_text_token():
    from vl.libero_probe import coarsen_grid, features_for
    rng = np.random.default_rng(0)
    img = rng.normal(size=(3, 2, 16, 16, 5))                    # (n, image, 16 x 16 tokens, w)
    def cells(g):                                               # the extractor's grid_pool layout
        k = 16 // g
        return img.reshape(3, 2, g, k, g, k, 5).mean((3, 5)).reshape(3, 2 * g * g, 5)
    text = rng.normal(size=(3, 1, 5))
    F4 = np.concatenate([cells(4), text], 1)
    for g in (4, 2, 1):
        assert np.allclose(coarsen_grid(F4, g), np.concatenate([cells(g), text], 1))
    assert features_for(F4, "grid2").shape == (3, 9, 5)
    with pytest.raises(AssertionError):
        coarsen_grid(F4, 3)


def test_history_on_grid_tokens_appends_past_tokens_and_a_flag_token():
    from vl.libero_probe import with_history
    meta = pd.DataFrame({"episode_index": [0, 0, 0, 1, 1], "frame_index": [0, 4, 8, 0, 4]})
    X = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2) + 1
    H = with_history(X, meta, [4])
    assert H.shape == (5, 3 + 3 + 1, 2)
    assert np.array_equal(H[:, :3], X)
    assert np.array_equal(H[1, 3:6], X[0]) and np.array_equal(H[4, 3:6], X[3])     # same episode, 4 frames back
    assert (H[[0, 3], 3:6] == 0).all()                                            # no past at the episode start
    assert np.array_equal(H[:, 6, 0], [0, 1, 1, 0, 1]) and (H[:, 6] == H[:, 6, :1]).all()
    # 2-D features unchanged: [X, past, flag]
    H2 = with_history(X[:, 0], meta, [4])
    assert H2.shape == (5, 5) and np.array_equal(H2[:, 4], [0, 1, 1, 0, 1])
