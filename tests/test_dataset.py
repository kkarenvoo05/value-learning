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


def test_crop_resize_pads_outside_the_frame():
    img = np.full((48, 64, 3), 200, np.uint8)
    out = crop_resize(img, (-32, 0, 32, 64), 8, fill=(1, 2, 3))   # left half and bottom quarter off-frame
    assert (out[:5, 5:] == 200).all() and (out[:, :3] == (1, 2, 3)).all() and (out[7:] == (1, 2, 3)).all()   # away from the seam
    assert (crop_resize(img, (100, 100, 132, 132), 8, fill=(1, 2, 3)) == (1, 2, 3)).all()   # fully outside


def test_padded_box_is_not_clamped_and_can_zoom_out():
    from vl.dataset import jitter_box
    torch.manual_seed(0)
    box, aug = [80, 0, 560, 480], {"crop_shift_px": 72, "crop_scale": [0.9, 1.15], "crop_pad": True}
    sides, dys = [], []
    for _ in range(2000):
        x0, y0, x1, y1 = jitter_box(box, aug, (480, 640))
        assert y1 - y0 == x1 - x0
        assert abs((x0 + x1) / 2 - 320) <= 72 + 1 and abs((y0 + y1) / 2 - 240) <= 72 + 1
        sides.append(x1 - x0), dys.append((y0 + y1) / 2 - 240)
    assert max(sides) > 480 and min(sides) < 0.92 * 480        # zooms out and in
    assert min(dys) < -60 and max(dys) > 60                     # full-height box now moves vertically


def test_padded_augment_fills_with_the_imagenet_mean(tmp_path):
    c = cfg() | {"imagenet_mean": MEAN}
    ds = FrameDataset(index(tmp_path), c, augment={"crop_scale": [3.0, 3.0], "crop_pad": True})
    img = ds[0]["images"]["side_1"]                             # 3x zoom-out: border is all padding
    assert img[:, 0, 0].tolist() == [round(255 * m) for m in MEAN]


FULL = {"n": 3, "brightness": 0.2, "contrast": 0.2, "saturation": 0.5, "hue": 0.05, "sharpness": 0.5,
        "affine": {"degrees": 5.0, "translate": 0.05}}


def test_full_augment_picks_exactly_n_transforms_uniformly_in_order():
    from vl.dataset import pick_subset
    torch.manual_seed(0)
    counts = np.zeros(6)
    for _ in range(3000):
        idx = pick_subset(6, 3)
        assert len(idx) == 3 and len(set(idx)) == 3 and idx == sorted(idx)
        counts[idx] += 1
    assert np.allclose(counts / 3000, 0.5, atol=0.04)      # each transform in half the images


def test_full_augment_keeps_shape_dtype_and_zero_ranges_are_identity():
    from vl.dataset import full_augment
    torch.manual_seed(0)
    x = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    y = full_augment(x, FULL)
    assert y.shape == x.shape and y.dtype == torch.uint8 and not torch.equal(x, y)
    still = {"n": 1, "affine": {"degrees": 0.0, "translate": 0.0}}
    assert torch.equal(full_augment(x, still), x)


def test_full_augment_affine_translates_with_black_fill():
    from vl.dataset import full_augment
    torch.manual_seed(0)
    x = torch.full((3, 40, 40), 200, dtype=torch.uint8)
    shifted = 0
    for _ in range(20):
        y = full_augment(x, {"n": 1, "affine": {"translate": 0.1}})
        assert set(y.unique().tolist()) <= {0, 200}
        shifted += bool((y == 0).any())
    assert shifted >= 15


def test_full_augment_in_dataset_changes_train_items_only(tmp_path):
    plain, aug = FrameDataset(index(tmp_path), cfg()), FrameDataset(index(tmp_path), cfg(), augment={"full": FULL})
    torch.manual_seed(0)
    changed = sum(not torch.equal(plain[i]["images"][c], aug[i]["images"][c])
                  for i in range(5) for c in cfg()["cameras"])
    assert changed >= 8


def test_per_camera_augment_touches_only_its_camera(tmp_path):
    cams = cfg()["cameras"]
    plain = FrameDataset(index(tmp_path), cfg())
    aug = FrameDataset(index(tmp_path), cfg(), augment={"per_camera": {cams[0]: {"full": FULL}}})
    torch.manual_seed(0)
    assert sum(not torch.equal(plain[i]["images"][cams[0]], aug[i]["images"][cams[0]]) for i in range(5)) >= 4
    assert all(torch.equal(plain[i]["images"][c], aug[i]["images"][c]) for i in range(5) for c in cams[1:])


def test_proprio_item_is_row_t_of_its_episode(tmp_path):
    from vl.proprio import INPUT_NAMES
    idx = index(tmp_path).assign(repo="r", episode_index=7)
    tab = pd.DataFrame({"repo": "r", "episode_index": 7, "frame_index": np.arange(10)[::-1]}   # shuffled order
                       ).assign(**{n: np.arange(10)[::-1] * 10.0 + j for j, n in enumerate(INPUT_NAMES)})
    tab.to_parquet(tmp_path / "proprio.parquet")
    c = cfg() | {"proprio": {"enabled": True, "table": str(tmp_path / "proprio.parquet")}}
    item = FrameDataset(idx, c)[4]
    assert set(item) == {"images", "bin", "value", "row", "proprio"}
    assert item["proprio"].dtype == torch.float32
    assert torch.equal(item["proprio"], torch.arange(10, dtype=torch.float32) + 40)
    # Disabled (the baseline): no proprio key, even if a table is configured.
    c["proprio"]["enabled"] = False
    assert "proprio" not in FrameDataset(idx, c)[4]


def test_proprio_missing_frame_raises(tmp_path):
    import pytest
    from vl.proprio import INPUT_NAMES
    idx = index(tmp_path).assign(repo="r", episode_index=7)
    pd.DataFrame({"repo": "r", "episode_index": 7, "frame_index": np.arange(9)}).assign(
        **{n: 0.0 for n in INPUT_NAMES}).to_parquet(tmp_path / "p.parquet")
    with pytest.raises(ValueError):
        FrameDataset(idx, cfg() | {"proprio": {"enabled": True, "table": str(tmp_path / "p.parquet")}})


def test_squash_keeps_the_whole_frame_and_distorts_it():
    from vl.dataset import squash_resize
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, 48:] = 200                       # right quarter bright
    out = squash_resize(img, 16)
    assert out.shape == (16, 16, 3)
    assert (out[:, 13:] == 200).all() and (out[:, :11] == 0).all()   # the quarter lands in the last 4 columns


def test_letterbox_keeps_aspect_with_black_bars():
    from vl.dataset import letterbox_resize
    img = np.full((480, 640, 3), 120, np.uint8)
    out = letterbox_resize(img, 224)
    assert out.shape == (224, 224, 3)
    assert (out[:28] == 0).all() and (out[196:] == 0).all()          # 224x168 image, 28-px bars
    assert (out[28:196] == 120).all()


def test_input_overrides_validate_and_do_not_mutate():
    import pytest
    from vl.dataset import input_overrides
    base = cfg()
    out = input_overrides(base, ["wrist_left=16,0,64,48"], ["side_1=letterbox"])
    assert out["crops"]["wrist_left"] == [16, 0, 64, 48] and out["resize"] == {"side_1": "letterbox"}
    assert base["crops"]["wrist_left"] == [8, 0, 56, 48] and "resize" not in base
    for bad in (dict(crops=["wrist_left=0,0,10,20"]), dict(crops=["nope=0,0,8,8"]), dict(resize=["side_1=stretch"])):
        with pytest.raises(ValueError):
            input_overrides(base, **bad)


def test_resize_modes_in_dataset(tmp_path):
    import pytest
    from vl.dataset import input_overrides, letterbox_resize, squash_resize
    from vl.frames import FrameReader
    idx = index(tmp_path)
    plain = FrameDataset(idx, cfg())
    same = FrameDataset(idx, input_overrides(cfg(), [], []))                 # defaults: byte-identical
    ds = FrameDataset(idx, input_overrides(cfg(), [], ["wrist_left=squash", "side_1=letterbox"]))
    raw = FrameReader().read(idx.path_wrist_left[2], idx.from_ts_wrist_left[2], 2, FPS)
    for c in cfg()["cameras"]:
        assert torch.equal(plain[2]["images"][c], same[2]["images"][c])
    assert np.array_equal(ds[2]["images"]["wrist_left"].permute(1, 2, 0).numpy(), squash_resize(raw, 16))
    assert np.array_equal(ds[2]["images"]["side_1"].permute(1, 2, 0).numpy(), letterbox_resize(raw, 16))
    with pytest.raises(ValueError):                                          # crop jitter undefined there
        FrameDataset(idx, input_overrides(cfg(), [], ["side_1=squash"]), augment={"crop_shift_px": 4})
    FrameDataset(idx, input_overrides(cfg(), [], ["side_1=squash"]),
                 augment={"per_camera": {"wrist_left": {"crop_shift_px": 4}}})  # jitter on a crop camera is fine


def hist_cfg(gap_s):
    return cfg() | {"history": {"gap_s": gap_s}}


def test_history_item_is_frame_t_minus_gap_and_masked_before_the_start(tmp_path):
    gap = 4 / FPS                                       # 4 frames
    ds = FrameDataset(index(tmp_path), hist_cfg(gap))
    for i in [0, 3, 4, 9]:
        item = ds[i]
        assert set(item) == {"images", "bin", "value", "row", "past"}
        assert item["past"]["has_past"] == float(i >= 4)
        for c in cfg()["cameras"]:
            now = frame_number(item["images"][c].permute(1, 2, 0).numpy())
            was = frame_number(item["past"]["images"][c].permute(1, 2, 0).numpy())
            assert now == 5 + i
            assert was == (5 + i - 4 if i >= 4 else now)   # before the start: placeholder = frame t


def test_history_off_has_no_past_and_is_byte_identical(tmp_path):
    a, b = FrameDataset(index(tmp_path), cfg()), FrameDataset(index(tmp_path), cfg() | {"history": None})
    assert "past" not in a[5] and "past" not in b[5]
    assert all(torch.equal(a[5]["images"][c], b[5]["images"][c]) for c in cfg()["cameras"])


def test_history_past_gets_the_same_augmentation_draw(tmp_path):
    """Frames t and t - gap differ only in content: an identical source frame gives identical inputs."""
    idx = index(tmp_path)
    ds = FrameDataset(idx, hist_cfg(2 / FPS), augment={"full": FULL} | AUG)
    ds.t = np.full(len(idx), 6)                         # point every row at frame 6 ...
    ds.gap_frames = np.zeros(len(idx), int)             # ... and its "past" at the same frame
    torch.manual_seed(0)
    plain = FrameDataset(idx, cfg())
    for i in range(4):
        item = ds[i]
        for c in cfg()["cameras"]:
            assert torch.equal(item["images"][c], item["past"]["images"][c])
            assert not torch.equal(item["images"][c], plain[6]["images"][c])   # augmentation did run
    # and the draws still differ between items and cameras
    a, b = ds[0], ds[1]
    assert not torch.equal(a["images"]["wrist_left"], b["images"]["wrist_left"])


def test_history_gap_below_one_frame_is_refused(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        FrameDataset(index(tmp_path), hist_cfg(0.01))


def proprio_table(n=30):
    """One episode at 15 Hz: moving 0.5 m/s up to frame 11, then still. Speed compares frames t and t - 7,
    so frame 18 (vs 11) is the first still frame."""
    x = np.minimum(np.arange(n), 11) * 0.5 / 15
    return pd.DataFrame({"repo": "r", "episode_index": 0, "frame_index": np.arange(n), "x": x, "y": 0.0, "z": 0.0})


def test_still_seconds_is_causal_and_counts_up_from_the_last_motion():
    from vl.dataset import still_seconds
    t = proprio_table()
    s = still_seconds(t.sample(frac=1, random_state=0), 0.01).sort_index().to_numpy()   # row order doesn't matter
    assert (s[:18] == 0).all()
    assert np.allclose(s[18:], (np.arange(18, 30) - 17) / 15)
    t2 = t.copy()
    t2.loc[25:, "x"] += 1.0                                   # move again at frame 25
    s2 = still_seconds(t2, 0.01).to_numpy()
    assert np.array_equal(s[:25], s2[:25]) and s2[25] == 0     # causal: the past is unchanged


def test_still_item_is_seconds_still_at_frame_t(tmp_path):
    proprio_table().to_parquet(tmp_path / "p.parquet")
    idx = index(tmp_path).assign(repo="r", episode_index=0, frame_index=np.arange(10) + 15)
    c = cfg() | {"still_time": {"enabled": True, "table": str(tmp_path / "p.parquet"), "still_mps": 0.01}}
    ds = FrameDataset(idx, c)
    assert set(ds[0]) == {"images", "bin", "value", "row", "still"}
    assert ds[0]["still"].shape == (1,) and float(ds[0]["still"]) == 0.0          # frame 15: moving
    assert np.isclose(float(ds[9]["still"]), (24 - 17) / 15)                      # frame 24
    assert "still" not in FrameDataset(idx, cfg())[0]
