"""Unit tests on synthetic inputs: label precedence and episode grouping."""

import pandas as pd

from vl.manifest import latest_events, resolve_label
from vl.splits import UnionFind, assign_splits, build_groups


def ev(ep, kind, payload, ts):
    return {"episode_index": ep, "label_kind": "outcome", "payload": payload,
            "source": {"kind": kind, "tool": kind}, "ts": ts}


def test_latest_human_label_wins_over_auto_and_earlier_human():
    events = [
        ev(0, "auto", {"new_outcome": "timeout", "outcome_frame": 799}, "2026-01-01T00:00:00"),
        ev(0, "human", {"new_outcome": "failure", "outcome_frame": 300}, "2026-01-02T00:00:00"),
        ev(0, "human", {"new_outcome": "success", "outcome_frame": 410}, "2026-01-03T00:00:00"),
    ]
    latest = latest_events(events)
    lab = resolve_label(latest.get((0, "human")), latest.get((0, "auto")))
    assert (lab["outcome"], lab["T"], lab["label_source"]) == ("success", 410, "human")


def test_human_skip_falls_back_to_auto_and_is_marked():
    events = [
        ev(1, "auto", {"new_outcome": "timeout", "outcome_frame": 799}, "2026-01-01T00:00:00"),
        ev(1, "human", {"action": "skip"}, "2026-01-02T00:00:00"),
    ]
    latest = latest_events(events)
    lab = resolve_label(latest.get((1, "human")), latest.get((1, "auto")))
    assert (lab["outcome"], lab["T"], lab["label_source"]) == ("timeout", 799, "auto_after_human_skip")


def test_missing_labels_are_unknown_not_failure():
    lab = resolve_label(None, None)
    assert lab["outcome"] == "unknown" and lab["T"] is None


def test_stage_events_never_become_outcomes():
    events = [{"episode_index": 2, "label_kind": "stage", "payload": {"summary": {"stage": 7}},
               "source": {"kind": "vlm"}, "ts": "2026-01-01"}]
    latest = latest_events(events)
    assert resolve_label(latest.get((2, "human")), latest.get((2, "auto")))["outcome"] == "unknown"


def test_union_find_transitive():
    uf = UnionFind("abcd")
    uf.union("a", "b"); uf.union("c", "d"); uf.union("b", "d")
    assert len({uf.find(x) for x in "abcd"}) == 1


def _synthetic_manifest():
    rows = []
    # Two pool repos sharing an initial state (must land in one group), plus one test episode
    # whose initial state matches a pool episode (must be flagged, not moved).
    def row(repo, ep, state, fp, outcome="success"):
        return {"repo": repo, "episode_index": ep, "episode_uid": f"{repo}:{ep}",
                "repo_id": repo, "release_block_repo": None,
                "source_repo": repo, "source_episode_index": ep, "content_fingerprint": fp,
                "outcome": outcome, "init_pen_x": state, "init_pen_y": 0.0, "init_pen_yaw": 0.0,
                "init_holder_x": 0.15, "init_holder_y": 0.0}
    for i in range(40):
        rows.append(row("A", i, 0.001 * i, f"a{i}", "success" if i % 3 else "failure"))
    rows.append(row("B", 0, 0.0, "b0"))           # same state as A:0
    rows.append(row("A", 99, 0.5, "dup"))
    rows.append(row("B", 99, 0.6, "dup"))         # same content as A:99
    rows.append(row("T", 0, 0.001 * 5, "t0"))      # same state as A:5 -> conflict
    return pd.DataFrame(rows)


def test_groups_and_split_are_disjoint_and_deterministic():
    cfg = {"train_pool": ["A", "B"], "test": {"T": "test_t"},
           "validation": {"fraction": 0.2, "seed": 0}}
    m = build_groups(_synthetic_manifest())
    gid = m.set_index("episode_uid").group_id
    assert gid["A:0"] == gid["B:0"]
    assert gid["A:99"] == gid["B:99"]
    assert gid["A:5"] == gid["T:0"]
    s1, s2 = assign_splits(m, cfg), assign_splits(m, cfg)
    assert (s1.split == s2.split).all()
    # No group spans train and val.
    assert (s1[s1.split.isin(["train", "val"])].groupby("group_id").split.nunique() == 1).all()
    # The test episode keeps its split; the conflicting pool episode is flagged.
    s = s1.set_index("episode_uid")
    assert s.loc["T:0", "split"] == "test_t"
    assert bool(s.loc["A:5", "test_overlap_conflict"])
    assert (s1.split == "val").sum() > 0


def test_center_crop_box_matches_torchvision_resize_256_crop_224():
    from vl.preview import center_crop_box
    # 640x480: short side 480 -> 256 (scale 8/15); a 224 crop keeps a 420 px square, centered.
    assert center_crop_box(640, 480) == (110, 30, 530, 450)
