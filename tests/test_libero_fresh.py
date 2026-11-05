"""Fresh-layout episodes (init_state -1) in the LIBERO trainers (docs/libero_plan.md 2026-10-31)."""

import pandas as pd

from vl.libero_probe import inner_split
from vl.libero_resnet import table_meta


def _meta(states):
    return pd.DataFrame({"split": "train", "init_state": states, "episode_index": range(len(states))})


def test_inner_split_never_holds_out_fresh_and_keeps_fixed_states():
    fixed = [s for s in range(36) for _ in range(2)]
    a = inner_split(_meta(fixed))
    b = inner_split(_meta(fixed + [-1] * 50))
    assert (a == b[:len(fixed)]).all()           # same held-out fixed states with fresh rows added
    assert not b[len(fixed):].any()              # fresh rows are always fit


def test_table_meta_grid_and_targets(tmp_path):
    eps = pd.DataFrame({"episode_index": [0, 1000], "split": ["train", "train"], "init_state": [3, -1],
                        "source_episode": [0, 0], "policy_steps": [9, 5], "success": [1, 0]})
    eps.to_csv(tmp_path / "episodes.csv", index=False)
    ft = pd.DataFrame([(e, t, -0.1 * t, t) for e, n in [(0, 9), (1000, 5)] for t in range(n)],
                      columns=["episode_index", "frame_index", "value", "bin"])
    ft.to_parquet(tmp_path / "frame_targets.parquet")
    m = table_meta(tmp_path, 4)
    assert m[m.episode_index == 0].frame_index.tolist() == [0, 4, 8]
    assert m[m.episode_index == 1000].frame_index.tolist() == [0, 4]
    assert (m["T"] == m.episode_index.map({0: 8, 1000: 4})).all()
    assert (m.bin == m.frame_index).all()


def test_oof_split_labels_every_train_episode_once():
    from vl.libero_oof import oof_split
    rows = []
    for e in range(200):
        s = e % 30 if e < 120 else -1          # 30 fixed states x 4 + 80 fresh episodes
        sp = "val" if e in (5, 6) else "train"
        rows += [(e, s, sp, t) for t in (0, 4)]
    meta = pd.DataFrame(rows, columns=["episode_index", "init_state", "split", "frame_index"])
    seen = []
    for k in range(5):
        held, inner = oof_split(meta, k, 5)
        assert not (held & inner).any()
        assert (meta.split[held | inner] == "train").all()
        h, i = meta[held], meta[inner]
        assert set(h.init_state[h.init_state >= 0]).isdisjoint(set(meta[~held & (meta.init_state >= 0)
                                                                         & (meta.split == "train")].init_state))
        assert i.init_state.nunique() == 5 and (i.init_state >= 0).all()
        seen += list(h.episode_index.unique())
    train_eps = set(meta[meta.split == "train"].episode_index)
    assert sorted(seen) == sorted(train_eps)   # every train episode held out exactly once
