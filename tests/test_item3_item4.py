"""2026-11-07 builds: fitted n-step TD targets / value-binned labels (vl.libero_td, vl.libero_resnet --targets), the
FAST encoder's padding (vl.fast_tokens) and subtask prompts (vl.libero_finetune_data)."""

import numpy as np
import pandas as pd
import pytest

from vl import fast_tokens as ft
from vl import libero_finetune_data as fd
from vl import libero_td as td
from vl.libero_advantages import NORM, V_FAIL


def grid(lengths=(41, 61, 29), success=(1, 0, 1)):
    rows = []
    for e, (n, s) in enumerate(zip(lengths, success)):
        T = n - 1
        for t in range(0, n, 4):
            rows.append((e, t, T, s))
    return pd.DataFrame(rows, columns=["episode_index", "frame_index", "T", "success"])


def mc_target(d):
    return (-(d["T"] - d.frame_index) - NORM / 2 * (1 - d.success)) / NORM


def test_td_target_of_the_mc_target_is_itself():
    d = grid()
    v = mc_target(d).to_numpy()
    assert np.allclose(v[d.success.to_numpy() == 0].min(), V_FAIL - (60 / NORM), atol=1e-9)
    assert np.allclose(td.td_targets(d, v, 20), v, atol=1e-12)


def test_td_target_bootstraps_after_n_frames():
    d = grid()
    v = np.random.default_rng(0).uniform(-1, 0, len(d))
    y = td.td_targets(d, v, 20)
    i = 0  # episode 0, frame 0: interior, bootstraps on frame 20 (5 grid rows later)
    assert np.isclose(y[i], -20 / NORM + v[i + 5])
    last = d.index[(d.episode_index == 0)][-1]  # frame 40 = T: terminal, the outcome
    assert np.isclose(y[last], 0.0)


def test_labels_share_and_columns():
    d = grid((401, 333, 297), (1, 0, 1))
    v = np.random.default_rng(1).uniform(-1, 0, len(d))
    lab = td.labels_from_v(d, v, 20, 0.4)
    assert list(lab.columns[:3]) == ["episode_index", "frame_index", "positive"]
    assert abs(lab.positive.mean() - 0.4) < 0.01


def test_labels_recap_threshold_and_per_group_share():  # RECAP exact rule, 2026-11-08
    d = grid((401, 333, 297, 361), (1, 0, 1, 0))
    v = np.random.default_rng(2).uniform(-1, 0, len(d))
    thr = td.labels_from_v(d, v, 20, 0.4, binned=False)
    assert np.allclose(thr.z, thr.A)  # no value binning: one threshold on the raw advantage
    assert (thr.positive == (thr.A > np.percentile(thr.A, 60))).all()
    g = (d.episode_index >= 2).to_numpy().astype(int)  # two "rounds"
    by = td.labels_from_v(d, v, 20, 0.4, binned=False, groups=g)
    for k in (0, 1):
        assert abs(by.positive[g == k].mean() - 0.4) < 0.01
    assert np.allclose(td.labels_from_v(d, v, 20, 0.4).z, by.z * 0 + td.value_binned(by.A.to_numpy(), v))


def test_hindsight_labels_use_the_realised_return():  # coach test, 2026-11-08
    d = grid((401, 333, 297), (1, 0, 1))
    v = np.random.default_rng(2).uniform(-1, 0, len(d))
    lab = td.hindsight_labels(d, v)
    assert lab.term.all()
    assert np.allclose(lab.A, mc_target(d).to_numpy() - v)
    assert abs(lab.positive.mean() - 0.4) < 0.01
    with pytest.raises(AssertionError):
        td.hindsight_labels(grid((601,), (1,)), np.zeros(len(grid((601,), (1,)))))


def test_unsorted_rows_refused():
    d = grid().iloc[::-1].reset_index(drop=True)
    with pytest.raises(AssertionError):
        td.td_targets(d, np.zeros(len(d)), 20)


def test_replace_targets_train_only_and_bins():
    torch = pytest.importorskip("torch")  # noqa: F841  (vl.libero_resnet imports torch)
    from vl.libero_probe import CENTERS
    from vl.libero_resnet import replace_targets
    meta = pd.DataFrame({"episode_index": [0, 0, 1, 1], "frame_index": [0, 4, 0, 4],
                         "split": ["train", "train", "val", "val"], "value": [-0.5, -0.4, -0.3, -0.2],
                         "bin": [100, 120, 140, 160]})
    tg = pd.DataFrame({"episode_index": [0, 0], "frame_index": [0, 4], "value": [-0.25, -0.1]})
    m = replace_targets(meta, tg)
    assert m.value.tolist() == [-0.25, -0.1, -0.3, -0.2]
    assert m.bin.tolist()[:2] == [int(np.abs(CENTERS + 0.25).argmin()), int(np.abs(CENTERS + 0.1).argmin())]
    assert m.bin.tolist()[2:] == [140, 160]
    with pytest.raises(AssertionError):
        replace_targets(meta, tg.iloc[:1])  # a train frame without a target


def test_fast_dct_matches_matrix_form():
    from vl.discrete_actions import dct_matrix
    x = np.random.default_rng(2).normal(size=(3, 10, 7))
    assert np.allclose(ft.dct_ortho(x, 1), np.einsum("kt,btd->bkd", dct_matrix(10), x), atol=1e-5)


def test_fast_strings_time_major_and_offset():
    a = np.zeros((1, 10, 7))
    s = ft.chunk_strings(a)[0]
    assert len(s) == 70 and set(map(ord, s)) == {354}  # coefficient 0 -> chr(0 - min_token)


def test_fast_encode_pads_and_truncates():
    a = np.random.default_rng(3).normal(scale=0.3, size=(4, 10, 7))
    lens = [5, 0, 96, 120]
    enc_calls = iter(lens)
    tok, mask, cut = ft.encode(a, encoder=lambda s: list(range(next(enc_calls))), f_max=96)
    assert tok.shape == mask.shape == (4, 96)
    assert mask.sum(1).tolist() == [5, 0, 96, 96] and cut == 1
    assert (tok[~mask] == 0).all()


def test_subtask_prompt_formats():
    p = "put both moka pots on the stove"
    assert fd.tag_prompt(p, None) == p
    assert fd.tag_prompt(p, True) == p + ". Advantage: positive"
    assert fd.tag_prompt(p, False, 1) == p + ". Subtask: put the second moka pot on the stove. Advantage: negative"
    assert fd.tag_prompt(p, None, 0) == p + ". Subtask: put the first moka pot on the stove"


def test_frame_subtasks_follow_the_image():
    goals = np.zeros((6, 3), bool)
    goals[2:, 0] = True       # first pot on the stove after action 2
    goals[4:, 1] = True
    assert fd.frame_subtasks(goals).tolist() == [0, 0, 0, 1, 1, 1]  # image t shows goals[t - 1]


def test_load_episode_subtask_only_with_goals(tmp_path, monkeypatch):
    n = 5
    base = dict(image=np.zeros((n, 2, 2, 3), np.uint8), wrist_image=np.zeros((n, 2, 2, 3), np.uint8),
                state=np.zeros((n, 8)), actions=np.zeros((n, 7)), prompt="put both moka pots on the stove")
    goals = np.zeros((n, 3), bool); goals[1:, 0] = True
    np.savez(tmp_path / "roll.npz", goals=goals, **base)
    np.savez(tmp_path / "demo.npz", **base)
    ep = fd.load_episode(str(tmp_path / "roll.npz"), None, 0, subtask=True)
    assert ep["subtask"].tolist() == [0, 0, 1, 1, 1]
    assert "Subtask: put the second" in fd.sample(ep, 3, True)["prompt"]
    assert fd.load_episode(str(tmp_path / "demo.npz"), None, 0, subtask=True)["subtask"] is None
    assert fd.load_episode(str(tmp_path / "roll.npz"), None, 0)["subtask"] is None  # default off
    monkeypatch.setattr(fd, "SUBTASK_PROMPT", True)
    assert fd.load_episode(str(tmp_path / "roll.npz"), None, 0)["subtask"] is not None


def test_round2_collected_rows(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "r2", Path(__file__).resolve().parents[1] / "scripts" / "libero_round2_table.py")
    r2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r2)
    (tmp_path / "raw").mkdir()
    for k, n in enumerate([7, 5]):
        np.savez(tmp_path / "raw" / f"ep{k:04d}.npz", state=np.zeros((n, 8)))
    csv = pd.DataFrame({"episode": [0, 1], "task_id": 8, "init_state": [0, -1], "repeat": [0, 0],
                        "success": [1, 0], "timeout": [0, 1], "policy_steps": [7, 5], "max_steps": 520,
                        "goals_end": ["111", "010"], "error": [None, None], "video": "", "raw": ["ep0000.npz", "ep0001.npz"]})
    csv.to_csv(tmp_path / "episodes.csv", index=False)
    out = r2.collected(tmp_path, 200000)
    assert out.episode_index.tolist() == [200000, 200001] and out.layout.tolist() == ["fixed", "fresh"]
    assert (out.split == "train").all()
    csv.assign(init_state=[9, -1]).to_csv(tmp_path / "episodes.csv", index=False)  # 9 = a value-val state
    with pytest.raises(AssertionError):
        r2.collected(tmp_path, 200000)


def test_curve_subsample_spreads_over_states():  # coach-test data curve, 2026-11-08
    import importlib, sys as _s
    _s.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    ish = importlib.import_module("libero_tag_ideal_shift")
    ueps = np.arange(450)
    states = np.repeat(np.arange(30), 15)
    for n in (30, 56, 225):
        pick = ish.stratified_episodes(ueps, states, n, 0)
        assert len(pick) == n == len(set(pick.tolist()))
        c = np.bincount(states[pick], minlength=30)
        assert c.max() - c.min() <= 1  # as even as possible
    assert not np.array_equal(ish.stratified_episodes(ueps, states, 56, 0), ish.stratified_episodes(ueps, states, 56, 1))
    assert set(ish.stratified_episodes(ueps, states, 450, 3).tolist()) == set(ueps.tolist())
    t = pd.DataFrame({"group": "1", "n_sub": [56] * 4, "sub_seed": [0, 0, 1, 1],
                      "labels": ["vb", "random 40%"] * 2, "ideal_shift_raw": [0.02, 0.01, 0.03, 0.01],
                      "ideal_shift_arm": 0.0, "ideal_shift_gripper": 0.0, "tag_mse_gain": 0.0,
                      "gain_action_over_state": [0.1, np.nan, 0.1, np.nan]})
    c = ish.curve_table(t).set_index("kind")
    assert np.isclose(c.loc["labels", "shift_over_random"], 0.015) and c.loc["labels", "subsamples"] == 2
