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


def test_demo_rows_and_positive_labels(tmp_path):
    pd.DataFrame({"episode_index": [0, 1, 2], "task": ["a", "put both moka pots on the stove", "b"],
                  "success": 1, "raw": ["x0", "x1", "x2"]}).to_csv(tmp_path / "episodes.csv", index=False)
    d = fd.demo_episodes(tmp_path, "put both moka pots on the stove")
    assert d.episode_index.tolist() == [fd.DEMO_OFFSET + 1] and d.raw.tolist() == ["x1"]
    assert (d.split == "train").all() and (d.init_state == -1).all() and d.demo.all()
    lab = fd.demo_labels(d)
    assert fd.frame_labels(lab, fd.DEMO_OFFSET + 1, 7).tolist() == [1] * 7
    with pytest.raises(AssertionError):
        fd.demo_episodes(tmp_path, "missing task")


def test_weights_floor_on_grid_and_reach_samples():
    w = pd.DataFrame({"episode_index": [0, 0], "frame_index": [0, 4], "weight": [0.5, 2.0]})
    assert fd.frame_weights(w, 0, 6).tolist() == [0.5] * 4 + [2.0] * 2

    def load(raw, lab, e, weights=None):
        ep = {"image": np.zeros((6, 2, 2, 3)), "wrist_image": np.zeros((6, 2, 2, 3)), "state": np.zeros((6, 8)),
              "actions": np.zeros((6, 7)), "prompt": "p", "label": None}
        ep["weight"] = None if weights is None else fd.frame_weights(weights, e, 6)
        return ep
    sel = pd.DataFrame({"episode_index": [0], "raw": ["a"]})
    out, meta = next(iter(fd.EpisodeStream(sel, batch=6, seed=0, group=1, loader=load, weights=w)))
    assert all(s["weight"] == (0.5 if t < 4 else 2.0) for s, (_, t, _) in zip(out, meta))
    out, _ = next(iter(fd.EpisodeStream(sel, batch=6, seed=0, group=1, loader=load)))
    assert all("weight" not in s for s in out)


def test_weighted_arm_keeps_failures():
    eps = pd.DataFrame({"episode_index": [0, 1], "split": "train", "init_state": [0, 1], "success": [1, 0]})
    assert len(fd.select_episodes(eps, "weighted", held_out=set())) == 2
    assert len(fd.select_episodes(eps, "imitation", held_out=set())) == 1


def test_tag_action_shift_moves_only_tagged_dim():
    """Tag positive control [2026-11-06]: +shift on dim SHIFT_DIM for the positive tag, -shift for the negative,
    nothing for untagged samples, and the episode's stored actions are not modified."""
    import numpy as np
    from vl import libero_finetune_data as fd
    T = 30
    ep = {"image": np.zeros((T, 2, 2, 3), np.uint8), "wrist_image": np.zeros((T, 2, 2, 3), np.uint8),
          "state": np.zeros((T, 8)), "actions": np.random.default_rng(0).normal(size=(T, 7)), "prompt": "p"}
    before = ep["actions"].copy()
    base = fd.sample(ep, 5, None)["actions"]
    pos, neg, none = (fd.sample(ep, 5, tag, 0.1)["actions"] for tag in (True, False, None))
    d = fd.SHIFT_DIM
    assert np.allclose(pos[:, d] - base[:, d], 0.1) and np.allclose(neg[:, d] - base[:, d], -0.1)
    other = [k for k in range(7) if k != d]
    assert np.array_equal(pos[:, other], base[:, other]) and np.array_equal(none, base)
    assert np.array_equal(ep["actions"], before)


def demo_csv(tmp_path):
    """Two tasks: 'a' with 10 demos (lengths 100..190, reversed by index), 'b' with 4 (a tie at 50)."""
    d = pd.DataFrame({"episode_index": range(14), "task": ["a"] * 10 + ["b"] * 4,
                      "policy_steps": [190 - 10 * i for i in range(10)] + [50, 50, 80, 70],
                      "success": 1, "role": ["fit"] * 9 + ["inner"] + ["fit"] * 3 + ["inner"],
                      "raw": [f"x{i}" for i in range(14)]})
    d.to_csv(tmp_path / "episodes.csv", index=False)
    return d


def test_speed_labels_fastest_share_per_task(tmp_path):
    d = demo_csv(tmp_path)
    lab = fd.speed_labels(d, 0.3)
    assert len(lab) == 14 and (lab.frame_index == 0).all()
    pos = set(lab[lab.positive == 1].episode_index - fd.DEMO_OFFSET)
    assert pos == {9, 8, 7, 10}  # task a: the 3 shortest (100, 110, 120); task b: round(1.2) = 1, tie -> lower index
    assert fd.frame_labels(lab, fd.DEMO_OFFSET + 9, 5).tolist() == [1] * 5
    assert fd.frame_labels(lab, fd.DEMO_OFFSET + 0, 5).tolist() == [0] * 5
    assert lab.equals(fd.speed_labels(d.sample(frac=1, random_state=3), 0.3))  # order-independent


def test_speed_labels_refuse_degenerate_share(tmp_path):
    d = demo_csv(tmp_path)
    with pytest.raises(AssertionError):
        fd.speed_labels(d[d.task == "b"], 0.1)  # round(0.4) = 0 positives


def test_pretrain_demo_episodes_roles_and_offset(tmp_path):
    demo_csv(tmp_path)
    fit = fd.pretrain_demo_episodes(tmp_path)
    assert len(fit) == 12 and set(fit.task) == {"a", "b"}
    assert (fit.episode_index >= fd.DEMO_OFFSET).all() and fit.demo.all() and (fit.init_state == -1).all()
    inner = fd.pretrain_demo_episodes(tmp_path, ("inner",))
    assert sorted(inner.episode_index - fd.DEMO_OFFSET) == [9, 13]


def test_oversample_demos_hits_the_frame_share():  # step-1 recipe grid, 2026-11-08
    roll = pd.DataFrame({"episode_index": range(900), "demo": False, "policy_steps": 460})
    dem = pd.DataFrame({"episode_index": range(10**6, 10**6 + 29), "demo": True, "policy_steps": 324})
    sel = pd.concat([roll, dem], ignore_index=True)
    for p in (0.5, 0.75):
        out, k, share = fd.oversample_demos(sel, p)
        assert abs(share - p) < 0.01 and len(out) == 900 + 29 * k
        assert (out[~out.demo].episode_index.value_counts() == 1).all()
        assert (out[out.demo].episode_index.value_counts() == k).all()
        f = out.groupby("demo").policy_steps.sum()
        assert abs(f[True] / f.sum() - share) < 1e-9
    with pytest.raises(AssertionError):
        fd.oversample_demos(roll, 0.5)  # no demos to repeat
