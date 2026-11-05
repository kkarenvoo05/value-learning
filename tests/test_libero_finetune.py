"""Stage 6 fine-tuning data (src/vl/libero_finetune_data.py) on synthetic episodes."""

import numpy as np
import pandas as pd
import pytest

from vl import libero_finetune_data as fd


def eps_table():
    return pd.DataFrame({"episode_index": [0, 1, 2, 3, 4], "split": ["train", "train", "train", "val", "train"],
                         "init_state": [0, 1, -1, 9, 2], "success": [1, 0, 1, 1, 0],
                         "raw": ["a", "b", "c", "d", "e"]})


def test_action_chunk_repeats_last():
    a = np.arange(12, dtype=float).reshape(6, 2)
    assert (fd.action_chunk(a, 0, 3) == a[:3]).all()
    c = fd.action_chunk(a, 4, 4)
    assert (c == a[[4, 5, 5, 5]]).all() and c.shape == (4, 2)


def test_tag_prompt():
    assert fd.tag_prompt("put x", None) == "put x"
    assert fd.tag_prompt("put x", True) == "put x. Advantage: positive"
    assert fd.tag_prompt("put x", False) == "put x. Advantage: negative"


def test_select_episodes_arms_and_holdout():
    e = eps_table()
    held = {9, 5}
    assert fd.select_episodes(e, "imitation", held_out=held).episode_index.tolist() == [0, 2]
    assert fd.select_episodes(e, "advantage", held_out=held).episode_index.tolist() == [0, 1, 2, 4]
    assert fd.select_episodes(e, "advantage", episodes=[1, 4], held_out=held).episode_index.tolist() == [1, 4]
    with pytest.raises(AssertionError):
        fd.select_episodes(e, "advantage", episodes=[3], held_out=held)   # a val episode
    e.loc[1, "init_state"] = 5
    with pytest.raises(AssertionError):
        fd.select_episodes(e, "advantage", held_out=held)                 # train row on a held-out state


def test_real_split_holds_out_val_and_test():
    held = fd.held_out_states()
    assert len(held) == 20 and 9 in held and 5 in held and 0 not in held


def test_frame_labels_floor_on_grid():
    lab = pd.DataFrame({"episode_index": [7, 7, 7, 8], "frame_index": [0, 4, 8, 0], "positive": [1, 0, 1, 0]})
    assert fd.frame_labels(lab, 7, 10).tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    with pytest.raises(AssertionError):
        fd.frame_labels(lab, 9, 5)                                         # no labels
    with pytest.raises(AssertionError):
        fd.frame_labels(lab[lab.frame_index > 0], 7, 10)                   # none at frame 0


def fake_loader(n_by_ep, labels=None):
    def load(raw, lab, e):
        n = n_by_ep[e]
        ep = {"image": np.full((n, 2, 2, 3), e, np.uint8), "wrist_image": np.zeros((n, 2, 2, 3), np.uint8),
              "state": np.zeros((n, 8)), "actions": np.arange(n, dtype=float)[:, None].repeat(7, 1), "prompt": "p"}
        ep["label"] = None if lab is None else fd.frame_labels(lab, e, n)
        return ep
    return load


def test_stream_covers_every_frame_once_per_epoch():
    sel = pd.DataFrame({"episode_index": [0, 1, 2], "raw": ["a", "b", "c"]})
    n = {0: 5, 1: 7, 2: 4}   # 16 frames; group = all 3 episodes, batch 4 -> exactly 4 batches per epoch
    st = fd.EpisodeStream(sel, batch=4, seed=0, group=3, loader=fake_loader(n))
    it = iter(st)
    seen = [tuple(m[:2]) for _ in range(4) for m in next(it)[1]]
    assert sorted(seen) == sorted((e, t) for e in n for t in range(n[e]))
    out, meta = next(it)
    for s, (e, t, tag) in zip(out, meta):
        assert s["observation/image"][0, 0, 0] == e and s["prompt"] == "p" and tag == -1
        assert (s["actions"][:, 0] == np.minimum(np.arange(t, t + fd.HORIZON), n[e] - 1)).all()


def test_stream_tags_follow_labels_and_dropout():
    sel = pd.DataFrame({"episode_index": [0], "raw": ["a"]})
    lab = pd.DataFrame({"episode_index": [0, 0], "frame_index": [0, 4], "positive": [1, 0]})
    st = fd.EpisodeStream(sel, batch=8, seed=1, group=1, labels=lab, loader=fake_loader({0: 8}))
    out, meta = next(iter(st))
    for s, (e, t, tag) in zip(out, meta):
        assert tag == (1 if t < 4 else 0)
        assert s["prompt"] == fd.tag_prompt("p", bool(tag))
    st = fd.EpisodeStream(sel, batch=8, seed=1, group=1, labels=lab, tag_dropout=1.0, loader=fake_loader({0: 8}))
    out, meta = next(iter(st))
    assert all(s["prompt"] == "p" for s in out) and (meta[:, 2] == -1).all()
