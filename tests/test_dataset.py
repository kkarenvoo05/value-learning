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


AUG = {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.03, "crop_shift_px": 6, "crop_scale": [0.9, 1.0]}


def test_no_augment_is_byte_identical(tmp_path):
    a, b = FrameDataset(index(tmp_path), cfg()), FrameDataset(index(tmp_path), cfg(), augment={})
    for i in range(3):
        for c in cfg()["cameras"]:
            assert torch.equal(a[i]["images"][c], b[i]["images"][c])


def test_augment_keeps_shape_dtype_and_labels_and_changes_pixels(tmp_path):
    plain, aug = FrameDataset(index(tmp_path), cfg()), FrameDataset(index(tmp_path), cfg(), augment=AUG)
    torch.manual_seed(0)
    changed = 0
    for i in range(5):
        p, q = plain[i], aug[i]
        assert (q["bin"], q["value"], q["row"]) == (p["bin"], p["value"], p["row"])
        for c in cfg()["cameras"]:
            assert q["images"][c].shape == p["images"][c].shape and q["images"][c].dtype == torch.uint8
            changed += not torch.equal(q["images"][c], p["images"][c])
    assert changed >= 8


def test_jittered_box_stays_square_and_inside_the_frame():
    from vl.dataset import jitter_box
    torch.manual_seed(0)
    for box in ([80, 0, 560, 480], [140, 0, 560, 420]):     # the real wrist / side crops
        for _ in range(500):
            x0, y0, x1, y1 = jitter_box(box, {"crop_shift_px": 24, "crop_scale": [0.9, 1.0]}, (480, 640))
            side = x1 - x0
            assert y1 - y0 == side and 0.9 * 420 - 1 <= side <= box[2] - box[0]
            assert 0 <= x0 and x1 <= 640 and 0 <= y0 and y1 <= 480
            assert abs((x0 + x1) / 2 - (box[0] + box[2]) / 2) <= 24 + 1
