"""FrameReader returns exactly frame t, including for episodes packed back to back in one file."""

import av
import numpy as np
import pandas as pd
import pytest

from vl.frames import FrameReader, build_frame_index

FPS = 15


def write_numbered_video(path, n_frames, gop=2):
    """Each frame is a flat gray level 4*i, so the decoded frame number is readable back."""
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=FPS)
    s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
    s.gop_size = gop
    s.codec_context.qmin = s.codec_context.qmax = 1   # near-lossless, keeps flat levels exact
    for i in range(n_frames):
        img = np.full((48, 64, 3), 4 * i, np.uint8)
        c.mux(s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
    c.mux(s.encode())
    c.close()


def frame_number(img):
    return int(round(img.mean() / 4))


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "file-000.mp4"
    write_numbered_video(p, 60)
    return p


@pytest.mark.parametrize("from_frame,length", [(0, 20), (20, 25), (45, 15)])
def test_reads_exact_frame_of_packed_episode(video, from_frame, length):
    r = FrameReader()
    for t in list(range(length)) + [length - 1, 0, length // 2]:   # forward, then backward seeks
        img = r.read(str(video), from_frame / FPS, t, FPS)
        assert frame_number(img) == from_frame + t


def test_past_end_raises(video):
    with pytest.raises(RuntimeError):
        FrameReader().read(str(video), 45 / FPS, 15, FPS)   # one frame past the last


def test_frame_index_keeps_only_valid_train_val_frames():
    man = pd.DataFrame({"episode_uid": ["a", "b"], "repo": ["r", "r"], "episode_index": [0, 1],
                        "T": [2, 1], "fps": [15, 15], "source_type": "x", "policy_key": "p",
                        "has_human_frames": False, "soft_truncate": False})
    tg = pd.DataFrame({"episode_uid": ["a"] * 3 + ["b"] * 2, "repo": "r",
                       "episode_index": [0, 0, 0, 1, 1], "split": ["train"] * 3 + ["test_x"] * 2,
                       "frame_index": [0, 1, 2, 0, 1], "in_support": True, "outcome": "success"})
    locs = pd.DataFrame({"repo": "r", "episode_index": [0, 1], "cam": "wrist_left",
                         "path": ["f0", "f1"], "from_ts": [0.0, 1.0],
                         "to_ts": [3 / 15, 1.0 + 2 / 15]})
    df = build_frame_index(man, tg, locs, ["wrist_left"])
    assert df.episode_uid.tolist() == ["a"] * 3          # test-split episode dropped
    assert (df.frame_index <= df["T"]).all()
    assert df.path_wrist_left.eq("f0").all()


def test_frame_index_rejects_frame_past_episode_span():
    man = pd.DataFrame({"episode_uid": ["a"], "repo": ["r"], "episode_index": [0], "T": [2], "fps": [15],
                        "source_type": "x", "policy_key": "p", "has_human_frames": False,
                        "soft_truncate": False})
    tg = pd.DataFrame({"episode_uid": "a", "repo": "r", "episode_index": 0, "split": "train",
                       "frame_index": [0, 1, 2], "in_support": True, "outcome": "success"})
    # Video holds only 2 frames for this episode; frame 2 would be the next episode's frame 0.
    locs = pd.DataFrame({"repo": ["r"], "episode_index": [0], "cam": "wrist_left", "path": ["f0"],
                         "from_ts": [0.0], "to_ts": [2 / 15]})
    with pytest.raises(AssertionError):
        build_frame_index(man, tg, locs, ["wrist_left"])
