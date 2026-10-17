"""Integrity checks on the built manifest (skipped until `python -m vl.splits` has run)."""

from pathlib import Path

import pandas as pd
import pytest

from vl.config import load_config

CFG = load_config(Path(__file__).parents[1] / "configs/dataset.yaml")
MANIFEST = Path(__file__).parents[1] / CFG["paths"]["manifest"]
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built")


@pytest.fixture(scope="module")
def m():
    df = pd.read_parquet(MANIFEST)
    return df[df.split != "audit_only"]


def test_episode_identity_unique(m):
    assert m.episode_uid.is_unique
    assert not m.duplicated(["repo", "episode_index"]).any()
    # Every row is pinned to the resolved commit.
    assert m.commit.str.fullmatch(r"[0-9a-f]{40}").all()


def test_every_configured_repo_present_with_expected_split(m):
    for r in CFG["train_pool"]:
        assert set(m[m.repo == r].split) <= {"train", "val"}
    for r, s in CFG["test"].items():
        assert set(m[m.repo == r].split) == {s}


def test_split_disjoint_by_group_and_source(m):
    tv = m[m.split.isin(["train", "val"])]
    assert (tv.groupby("group_id").split.nunique() == 1).all()
    # No initial state, source identity or content fingerprint crosses train/val.
    for col in ["init_state_key", "source_identity", "content_fingerprint"]:
        assert (tv.groupby(col).split.nunique() == 1).all(), col


def test_no_unflagged_train_test_overlap(m):
    for col in ["init_state_key", "source_identity", "content_fingerprint"]:
        test_keys = set(m[m.split.str.startswith("test")][col])
        leak = m[m.split.isin(["train", "val"]) & m[col].isin(test_keys)]
        assert leak.test_overlap_conflict.all(), f"unflagged overlap via {col}"


def test_endpoints_valid_for_usable_episodes(m):
    use = m[m.exclude_reason == ""]
    assert use.outcome.isin(["success", "failure", "timeout"]).all()
    assert (use["T"] >= 0).all() and (use["T"] <= use.raw_length - 1).all()
    assert (use.valid_length == use["T"] + 1).all()


def test_frame_flags_agree_with_reviewed_endpoint(m):
    sf = m[m.outcome.isin(["success", "failure"])]
    assert (sf.first_done == sf["T"]).all()
    st = m[m.soft_truncate]
    assert (st.n_valid == st["T"] + 1).all()


def test_synchronized_sampling(m):
    # Timestamps are frame_index / fps; every available camera's video span equals the frame count.
    assert m.frame_index_contiguous.all()
    assert (m.n_rows == m.raw_length).all()
    assert (m.ts_max_abs_err_s < 1e-3).all()
    use = m[m.exclude_reason == ""]
    for cam in ["wrist_left", "side_1"]:
        assert m[f"cam_{cam}"].all()
        # Every frame we may sample (0..T) must exist in the video. A short tail after T is
        # tolerated (known case: r00-eval ep 35 wrist_left lacks only the post-endpoint last frame).
        covered = use.raw_length + use[f"cam_{cam}_span_err_frames"].round()
        assert (covered >= use.valid_length).all(), cam
        assert (m[f"cam_{cam}_span_err_frames"].abs() <= 1).all(), cam


def test_initial_state_constant_within_episode(m):
    assert m.init_state_constant.all()
