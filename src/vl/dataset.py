"""PyTorch Dataset of (wrist_left, side_1) image pairs and their value-bin targets.

One item = one row of data/frames/frame_index.parquet (a valid frame t <= T):
    {"images": {cam: uint8 tensor (3, 224, 224)}, "bin": int, "value": float, "row": int}
Only `images` may go into the model. `bin` / `value` are the loss target, and `row` points back
into the frame index for breakdowns (outcome, repo, ...). Everything else that defines the label
(T, outcome, elapsed time, time-to-go, policy, proprioception) never leaves the index.

Images stay uint8 here (cheap to move between processes); `normalize` turns them into
ImageNet-normalized floats on the GPU.

Optional training augmentation (Stage 5, `augment` in a train.yaml section; never for validation),
drawn independently per camera and per item with torch's RNG (seeded per DataLoader worker) [ours]:
    crop_shift_px: s   shift the fixed crop box by up to +-s px in x and y (before the resize)
    crop_scale: [a, b] shrink the box side by a factor in [a, b] around its centre
                       (the box is then clamped inside the frame, so a full-height box only moves sideways)
    brightness / contrast / saturation: j  multiply by a factor in [1 - j, 1 + j]
    hue: h             shift hue by a value in [-h, h] (fraction of the colour wheel)

Benchmark / sampling log:
    .venv/bin/python -m vl.dataset --workers 3 --batches 100
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vl.config import load_config
from vl.frames import FrameReader


def crop_resize(img: np.ndarray, box, size: int) -> np.ndarray:
    """Fixed crop (x0, y0, x1, y1), then bilinear resize to size x size. HWC uint8 in and out."""
    x0, y0, x1, y1 = box
    return np.array(Image.fromarray(img[y0:y1, x0:x1]).resize((size, size), Image.BILINEAR))


def jitter_box(box, aug: dict, frame_hw: tuple[int, int]):
    """Random square box near `box`: scaled around its centre, shifted, then clamped into the frame."""
    x0, y0, x1, y1 = box
    side = x1 - x0
    lo, hi = aug.get("crop_scale", [1.0, 1.0])
    side = int(round(side * (lo + (hi - lo) * torch.rand(()).item())))
    shift = aug.get("crop_shift_px", 0)
    dx, dy = ((torch.rand(2) * 2 - 1) * shift).round().int().tolist()
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    H, W = frame_hw
    nx0 = int(min(max(round(cx - side / 2), 0), W - side))
    ny0 = int(min(max(round(cy - side / 2), 0), H - side))
    return nx0, ny0, nx0 + side, ny0 + side


def color_jitter(img: torch.Tensor, aug: dict) -> torch.Tensor:
    """Photometric jitter of one uint8 (3, H, W) image; factors drawn per call."""
    import torchvision.transforms.v2.functional as TF

    def factor(j):
        return 1 + (torch.rand(()).item() * 2 - 1) * j
    ops = []
    if aug.get("brightness"):
        ops.append(lambda x: TF.adjust_brightness(x, factor(aug["brightness"])))
    if aug.get("contrast"):
        ops.append(lambda x: TF.adjust_contrast(x, factor(aug["contrast"])))
    if aug.get("saturation"):
        ops.append(lambda x: TF.adjust_saturation(x, factor(aug["saturation"])))
    if aug.get("hue"):
        ops.append(lambda x: TF.adjust_hue(x, (torch.rand(()).item() * 2 - 1) * aug["hue"]))
    for k in torch.randperm(len(ops)).tolist():   # random order, as torchvision's ColorJitter
        img = ops[k](img)
    return img


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    """uint8 (..., 3, H, W) -> float, ImageNet-normalized."""
    m = torch.tensor(mean, device=x.device).view(3, 1, 1)
    s = torch.tensor(std, device=x.device).view(3, 1, 1)
    return (x.float() / 255.0 - m) / s


def unnormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Inverse of `normalize`, back to uint8 (for looking at what the model sees)."""
    m = torch.tensor(mean, device=x.device).view(3, 1, 1)
    s = torch.tensor(std, device=x.device).view(3, 1, 1)
    return ((x * s + m) * 255.0).round().clamp(0, 255).to(torch.uint8)


class FrameDataset(Dataset):
    def __init__(self, index: pd.DataFrame, cfg: dict, augment: dict | None = None):
        self.cams, self.size, self.crops = cfg["cameras"], cfg["image_size"], cfg["crops"]
        self.augment = augment or {}
        self.index = index.reset_index(drop=True)
        # Plain arrays: fast to index and safe to share with forked workers.
        self.paths = {c: self.index[f"path_{c}"].to_numpy() for c in self.cams}
        self.from_ts = {c: self.index[f"from_ts_{c}"].to_numpy() for c in self.cams}
        self.t = self.index.frame_index.to_numpy()
        self.fps = self.index.fps.to_numpy()
        self.bin = self.index.bin.to_numpy()
        self.value = self.index.value.to_numpy()
        self._reader, self._pid = None, None

    def __len__(self):
        return len(self.index)

    def reader(self) -> FrameReader:
        if self._pid != os.getpid():   # one decoder per worker process, never shared across a fork
            self._reader, self._pid = FrameReader(), os.getpid()
        return self._reader

    def __getitem__(self, i):
        images = {}
        for c in self.cams:
            img = self.reader().read(self.paths[c][i], self.from_ts[c][i], int(self.t[i]), self.fps[i])
            box = self.crops[c]
            if self.augment.get("crop_shift_px") or self.augment.get("crop_scale"):
                box = jitter_box(box, self.augment, img.shape[:2])
            x = torch.from_numpy(crop_resize(img, box, self.size)).permute(2, 0, 1).contiguous()
            if any(self.augment.get(k) for k in ("brightness", "contrast", "saturation", "hue")):
                x = color_jitter(x, self.augment)
            images[c] = x
        return {"images": images, "bin": int(self.bin[i]), "value": float(self.value[i]), "row": i}


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    df = pd.read_parquet(cfg["paths"]["frame_index"])
    return df[df.split == split].reset_index(drop=True)


def make_loader(ds: FrameDataset, batch_size: int, workers: int, shuffle: bool, seed: int = 0):
    """Uniform sampling over valid frames [ours]: a shuffled pass over every frame."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, generator=g,
                      persistent_workers=workers > 0, pin_memory=True, drop_last=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=100)
    args = ap.parse_args()
    cfg = load_config(args.config)
    idx = load_split(cfg, "train")
    ds = FrameDataset(idx, cfg)
    dl = make_loader(ds, args.batch_size, args.workers, shuffle=True)
    rows, t0 = [], None
    for k, b in enumerate(dl):
        if k == 2:
            t0 = time.time()   # skip worker start-up
        rows.append(b["row"].numpy())
        if k + 1 == args.batches:
            break
    dt = time.time() - t0
    n = args.batches - 2
    print(f"workers={args.workers} batch={args.batch_size}: {n / dt:.2f} batches/s, "
          f"{n * args.batch_size / dt:.0f} samples/s ({2 * n * args.batch_size / dt:.0f} frames/s)")
    seen = idx.iloc[np.concatenate(rows)]
    for col in ["outcome", "repo"]:
        mix = pd.DataFrame({"sampled": seen[col].value_counts(normalize=True),
                            "index": idx[col].value_counts(normalize=True)}).round(3)
        print(mix.to_string())


if __name__ == "__main__":
    main()
