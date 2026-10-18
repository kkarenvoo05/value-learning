"""Repo-aware metrics: AUC, repo-only constant, offset-free shape error."""

import numpy as np
import pandas as pd
import pytest

from vl.metrics import auc, score


def frames(uid, repo, outcome, value, V):
    n = len(value)
    return pd.DataFrame({"episode_uid": uid, "repo": repo, "outcome": outcome, "frame_index": np.arange(n),
                         "value": value, "V": V, "ce": 1.0})


def test_auc_perfect_reversed_and_ties():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.ones(4)) == 0.5
    assert np.isnan(auc(np.ones(3, int), np.arange(3.0)))


def test_offset_does_not_change_shape_error():
    v = np.linspace(-0.3, 0.0, 30)
    a = frames("a", "real-marker-d2-r01-eval", "success", v, v)          # perfect
    b = frames("b", "real-marker-d2-r01-eval", "success", v, v - 0.2)    # perfect shape, offset
    m = score(pd.concat([a, b]), {"real-marker-d2-r01-eval": -0.1})
    assert m["offset_free_mae"] == pytest.approx(0.0, abs=1e-12)
    assert m["val_mae"] == pytest.approx(0.1)                             # offset counts here
    assert m["offset_free_mae_flat"] == pytest.approx(np.abs(v - v.mean()).mean())


def test_repo_constant_and_eval_repo_split():
    a = frames("a", "real-marker-d2-c01-dagger-mixed", "success", np.zeros(4), np.zeros(4))
    b = frames("b", "real-marker-d2-r02-eval", "failure", np.full(4, -0.6), np.full(4, -0.5))
    m = score(pd.concat([a, b]), {"real-marker-d2-c01-dagger-mixed": -0.1, "real-marker-d2-r02-eval": -0.7})
    assert m["repo_const_mae"] == pytest.approx(0.1)
    assert m["eval_repos_mae"] == pytest.approx(0.1)            # only episode b
    assert m["eval_repos_repo_const_mae"] == pytest.approx(0.1)
