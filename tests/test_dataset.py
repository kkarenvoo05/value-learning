"""Dataset items carry only images as model input; preprocessing is fixed and invertible."""

import numpy as np
import pandas as pd
import torch

from test_frames import FPS, frame_number, write_numbered_video
from vl.dataset import FrameDataset, crop_resize, normalize, unnormalize

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def cfg():
    return {"cameras": ["wrist_left", "side_1"], "image_size": 16,
            "crops": {"wrist_left": [8, 0, 56, 48], "side_1": [0, 0, 48, 48]}}


def index(tmp_path):
    p = tmp_path / "file-000.mp4"
    write_numbered_video(p, 30)
    n = 10
    return pd.DataFrame({
        "path_wrist_left": str(p), "from_ts_wrist_left": 5 / FPS,
        "path_side_1": str(p), "from_ts_side_1": 5 / FPS,
        "frame_index": np.arange(n), "fps": FPS, "bin": np.arange(n) + 100,
        "value": -np.arange(n) / 100, "outcome": "success", "T": n - 1, "time_to_go_s": 1.0})


def test_item_has_only_images_as_input(tmp_path):
    ds = FrameDataset(index(tmp_path), cfg())
    item = ds[3]
    assert set(item) == {"images", "bin", "value", "row"}
    assert set(item["images"]) == {"wrist_left", "side_1"}
    for img in item["images"].values():
        assert img.shape == (3, 16, 16) and img.dtype == torch.uint8
    assert item["bin"] == 103 and item["row"] == 3


def test_item_is_frame_t_of_its_episode(tmp_path):
    ds = FrameDataset(index(tmp_path), cfg())
    for i in [0, 4, 9]:
        for img in ds[i]["images"].values():
            assert frame_number(img.permute(1, 2, 0).numpy()) == 5 + i   # episode starts at frame 5


def test_crop_resize_uses_the_fixed_box():
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, 32:] = 200                       # right half bright
    out = crop_resize(img, (32, 0, 64, 32), 8)
    assert out.shape == (8, 8, 3) and (out == 200).all()


def test_normalize_roundtrip():
    x = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    y = normalize(x, MEAN, STD)
    assert y.dtype == torch.float32 and abs(float(y.mean())) < 3
    assert torch.equal(unnormalize(y, MEAN, STD), x)
