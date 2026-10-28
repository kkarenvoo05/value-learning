"""PyTorch Dataset of (wrist_left, side_1) image pairs and their value-bin targets.

One item = one row of data/frames/frame_index.parquet (a valid frame t <= T):
    {"images": {cam: uint8 tensor (3, 224, 224)}, "bin": int, "value": float, "row": int}
With `proprio.enabled` in the data config (Stage 5 ablation 2), items also carry
    "proprio": float32 tensor (10,), the raw state of row t (`vl.proprio`; z-scored inside the model).
Only `images` (and `proprio` when enabled) may go into the model. `bin` / `value` are the loss target, and `row` points back
into the frame index for breakdowns (outcome, repo, ...). Everything else that defines the label
(T, outcome, elapsed time, time-to-go, policy, proprioception) never leaves the index.

Images stay uint8 here (cheap to move between processes); `normalize` turns them into
ImageNet-normalized floats on the GPU.

Optional training augmentation (Stage 5, `augment` in a train.yaml section; never for validation),
drawn independently per camera and per item with torch's RNG (seeded per DataLoader worker) [ours]:
    crop_shift_px: s   shift the fixed crop box by up to +-s px in x and y (before the resize)
    crop_scale: [a, b] scale the box side by a factor in [a, b] around its centre (b > 1 zooms out)
                       (the box is then clamped inside the frame, so a full-height box only moves sideways)
    crop_pad: true     don't clamp: the box may leave the frame, and the part outside is filled with
                       the ImageNet mean colour (0 after normalization) [ours]. Needed for shifts past
                       the frame edge and for zooming out (crop_scale b > 1)
    brightness / contrast / saturation: j  multiply by a factor in [1 - j, 1 + j]
    hue: h             shift hue by a value in [-h, h] (fraction of the colour wheel)
    full: {n, brightness, contrast, saturation, hue, sharpness, affine: {degrees, translate}}
                       "full image augmentation": per image, exactly n of the listed transforms,
                       chosen uniformly without replacement and applied in the listed order (the
                       LeRobot ImageTransforms scheme, reimplemented in torchvision). Same ranges as
                       above; sharpness: s -> factor in [1 - s, 1 + s]; affine: rotate by up to
                       +-degrees and translate by up to +-translate x image size, nearest-neighbour,
                       black fill (torchvision RandomAffine defaults). Applied after the crop/resize.
    per_camera: {cam: {...}}  a separate augment dict per camera (any of the keys above); a camera
                       not listed is not augmented. Replaces the shared keys [ours, Stage 5 split]

Frame -> 224 input per camera (Stage 5.5 Gate 0), data config `resize: {cam: mode}` (default crop):
    crop       the fixed square box `crops[cam]`, resized (the Stage 3 default; crop jitter applies here)
    squash     the whole 640x480 frame resized to 224x224, aspect not kept
    letterbox  the whole frame scaled to 224x168 with black bars top and bottom (openpi resize_with_pad)
`full` augmentation runs after any of them; crop jitter is refused for squash / letterbox.

History (Stage 5.5 B2 [ours]), data config `history: {gap_s: g}`: items also carry
    "past": {"images": {cam: uint8 (3, 224, 224)}, "has_past": float 0/1}
the frame g seconds before t (round(g * fps) frames) of the same episode, through the same crop and
the *same* augmentation draw as frame t (the torch RNG state is replayed per camera), so a robot
that has not moved gives two identical inputs. When t - gap < 0 the past does not exist: has_past = 0
and the past image is a copy of frame t, which the model masks out (no first-frame fallback).
Several past frames (Stage 5.5 C2b [ours]), data config `history: {gaps_s: [g1, g2, ...]}` (ascending):
    "past": {"images": {cam: uint8 (k, 3, 224, 224)}, "has_past": float (k,)}
one frame per gap, each through the same augmentation draw as frame t and masked on its own.

Still time (Stage 5.5 B3 follow-up, upper bound for history [ours]), data config `still_time.enabled`:
items also carry "still": float32 tensor (1,), seconds since the gripper last moved faster than
`still_time.still_mps` at frame t (`still_seconds`: the `vl.history_plots.stall_seconds` rule, causal,
from the proprio table). A perfect "how long has nothing changed" signal that image history could at
best recover; raw seconds here, log1p inside the model.

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


def crop_resize(img: np.ndarray, box, size: int, fill=(0, 0, 0)) -> np.ndarray:
    """Crop (x0, y0, x1, y1), then bilinear resize to size x size. HWC uint8 in and out.
    A box that leaves the frame is padded with `fill`; a box inside it is a plain slice."""
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        canvas = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
        canvas[:] = fill
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        crop = canvas
    else:
        crop = img[y0:y1, x0:x1]
    return np.array(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


RESIZE_MODES = ("crop", "squash", "letterbox")


def squash_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Whole frame -> size x size, bilinear; the aspect ratio is not kept (640x480 is squeezed
    horizontally by 4/3) [Stage 5.5 Gate 0 arm R2]."""
    return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def letterbox_resize(img: np.ndarray, size: int, fill=(0, 0, 0)) -> np.ndarray:
    """Whole frame scaled so its long side is `size` (aspect kept, bilinear), centred on a size x size
    canvas of `fill` (640x480 -> 224x168 + 28-px bars top and bottom). Black fill, as openpi's
    `resize_with_pad` does for pi0 / pi0.5 [Stage 5.5 Gate 0 arm R3]."""
    H, W = img.shape[:2]
    s = size / max(H, W)
    w, h = round(W * s), round(H * s)
    out = np.empty((size, size, 3), np.uint8)
    out[:] = fill
    y0, x0 = (size - h) // 2, (size - w) // 2
    out[y0:y0 + h, x0:x0 + w] = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    return out


def input_overrides(dcfg: dict, crops: list[str] | None = None, resize: list[str] | None = None) -> dict:
    """Apply per-run `--crop CAM=x0,y0,x1,y1` / `--resize CAM=MODE` overrides to a data config (copy).
    The result's `crops` and `resize` are what a run records in its config.json."""
    out = dict(dcfg, crops=dict(dcfg["crops"]), resize=dict(dcfg.get("resize", {})))
    for a in crops or []:
        cam, box = a.split("=")
        b = [int(v) for v in box.split(",")]
        if cam not in out["crops"] or len(b) != 4 or b[2] - b[0] != b[3] - b[1] or b[2] <= b[0]:
            raise ValueError(f"--crop {a}: need a known camera and a square box x0,y0,x1,y1")
        out["crops"][cam] = b
    for a in resize or []:
        cam, mode = a.split("=")
        if cam not in out["crops"] or mode not in RESIZE_MODES:
            raise ValueError(f"--resize {a}: need a known camera and one of {RESIZE_MODES}")
        out["resize"][cam] = mode
    return out


def jitter_box(box, aug: dict, frame_hw: tuple[int, int]):
    """Random square box near `box`: scaled around its centre, shifted, then clamped into the frame
    (unless `crop_pad`, which lets it leave the frame)."""
    x0, y0, x1, y1 = box
    side = x1 - x0
    lo, hi = aug.get("crop_scale", [1.0, 1.0])
    side = int(round(side * (lo + (hi - lo) * torch.rand(()).item())))
    shift = aug.get("crop_shift_px", 0)
    dx, dy = ((torch.rand(2) * 2 - 1) * shift).round().int().tolist()
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    nx0, ny0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    if not aug.get("crop_pad"):
        H, W = frame_hw
        nx0, ny0 = min(max(nx0, 0), W - side), min(max(ny0, 0), H - side)
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


FULL_ORDER = ("brightness", "contrast", "saturation", "hue", "sharpness", "affine")


def pick_subset(k: int, n: int) -> list[int]:
    """n of k indices, uniform without replacement, in increasing order."""
    return sorted(torch.multinomial(torch.ones(k), min(n, k)).tolist())


def full_augment(img: torch.Tensor, full: dict) -> torch.Tensor:
    """'Full image augmentation' of one uint8 (3, H, W) image: `full["n"]` of the configured
    transforms, picked per call, applied in FULL_ORDER."""
    import torchvision.transforms.v2.functional as TF

    def u(j):
        return (torch.rand(()).item() * 2 - 1) * j

    def affine(x):
        a = full["affine"]
        H, W = x.shape[-2:]
        t = a.get("translate", 0)
        return TF.affine(x, angle=u(a.get("degrees", 0)), translate=[round(u(t * W)), round(u(t * H))],
                         scale=1.0, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.NEAREST, fill=0)
    make = {"brightness": lambda x: TF.adjust_brightness(x, 1 + u(full["brightness"])),
            "contrast": lambda x: TF.adjust_contrast(x, 1 + u(full["contrast"])),
            "saturation": lambda x: TF.adjust_saturation(x, 1 + u(full["saturation"])),
            "hue": lambda x: TF.adjust_hue(x, u(full["hue"])),
            "sharpness": lambda x: TF.adjust_sharpness(x, 1 + u(full["sharpness"])),
            "affine": affine}
    ops = [make[k] for k in FULL_ORDER if full.get(k)]
    for k in pick_subset(len(ops), full["n"]):
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
        self.resize = cfg.get("resize") or {}
        for c in self.cams:
            mode = self.resize.get(c, "crop")
            if mode not in RESIZE_MODES:
                raise ValueError(f"resize mode {mode!r} for {c} not in {RESIZE_MODES}")
            aug = self.augment["per_camera"].get(c, {}) if "per_camera" in self.augment else self.augment
            if mode != "crop" and (aug.get("crop_shift_px") or aug.get("crop_scale")):
                raise ValueError(f"crop jitter is only defined for resize mode 'crop' ({c} is {mode!r})")
        self.pad_fill = tuple(round(255 * m) for m in cfg.get("imagenet_mean", [0.485, 0.456, 0.406]))
        self.index = index.reset_index(drop=True)
        # Plain arrays: fast to index and safe to share with forked workers.
        self.paths = {c: self.index[f"path_{c}"].to_numpy() for c in self.cams}
        self.from_ts = {c: self.index[f"from_ts_{c}"].to_numpy() for c in self.cams}
        self.t = self.index.frame_index.to_numpy()
        self.fps = self.index.fps.to_numpy()
        self.bin = self.index.bin.to_numpy()
        self.value = self.index.value.to_numpy()
        self.proprio = load_proprio(self.index, cfg) if cfg.get("proprio", {}).get("enabled") else None
        self.still = load_still(self.index, cfg) if (cfg.get("still_time") or {}).get("enabled") else None
        hist = cfg.get("history") or {}
        gap, gaps = hist.get("gap_s"), hist.get("gaps_s")
        if gap is not None and gaps is not None:
            raise ValueError("history takes gap_s (one past frame) or gaps_s (several), not both")
        self.gap_frames = None if gap is None else np.round(gap * self.fps).astype(int)
        if self.gap_frames is not None and (self.gap_frames < 1).any():
            raise ValueError(f"history gap {gap} s is shorter than one frame")
        self.gaps_frames = None
        if gaps is not None:
            if list(gaps) != sorted(set(gaps)):
                raise ValueError(f"history gaps_s {gaps} must be distinct and ascending")
            self.gaps_frames = [np.round(g * self.fps).astype(int) for g in gaps]
            if any((g < 1).any() for g in self.gaps_frames):
                raise ValueError(f"history gaps_s {gaps}: a gap is shorter than one frame")
        self._reader, self._pid = None, None

    def __len__(self):
        return len(self.index)

    def reader(self) -> FrameReader:
        if self._pid != os.getpid():   # one decoder per worker process, never shared across a fork
            self._reader, self._pid = FrameReader(), os.getpid()
        return self._reader

    def _input(self, c: str, i: int, t: int) -> torch.Tensor:
        """Frame t of row i's episode, camera c -> the model's uint8 (3, size, size) input, augmented."""
        aug = self.augment["per_camera"].get(c, {}) if "per_camera" in self.augment else self.augment
        img = self.reader().read(self.paths[c][i], self.from_ts[c][i], t, self.fps[i])
        mode = self.resize.get(c, "crop")
        if mode == "crop":
            box = self.crops[c]
            if aug.get("crop_shift_px") or aug.get("crop_scale"):
                box = jitter_box(box, aug, img.shape[:2])
            x = crop_resize(img, box, self.size, self.pad_fill)
        elif mode == "squash":
            x = squash_resize(img, self.size)
        else:
            x = letterbox_resize(img, self.size)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        if any(aug.get(k) for k in ("brightness", "contrast", "saturation", "hue")):
            x = color_jitter(x, aug)
        if aug.get("full"):
            x = full_augment(x, aug["full"])
        return x

    def __getitem__(self, i):
        if self.gaps_frames is not None:
            return self._multi_past_item(i)
        t = int(self.t[i])
        tp = None if self.gap_frames is None else t - int(self.gap_frames[i])
        images, past = {}, {}
        for c in self.cams:
            rng = torch.get_rng_state()
            images[c] = self._input(c, i, t)
            if tp is not None and tp >= 0:   # replay the same augmentation draw on the past frame
                after = torch.get_rng_state()
                torch.set_rng_state(rng)
                past[c] = self._input(c, i, tp)
                torch.set_rng_state(after)
            elif tp is not None:
                past[c] = images[c].clone()   # no past yet: placeholder, masked by has_past = 0
        item = {"images": images, "bin": int(self.bin[i]), "value": float(self.value[i]), "row": i}
        if tp is not None:
            item["past"] = {"images": past, "has_past": float(tp >= 0)}
        if self.proprio is not None:
            item["proprio"] = torch.from_numpy(self.proprio[i].copy())
        if self.still is not None:
            item["still"] = torch.tensor([self.still[i]], dtype=torch.float32)
        return item


    def _multi_past_item(self, i):
        """C2b: frame t plus one past frame per gap, all through the same augmentation draw."""
        t = int(self.t[i])
        tps = [t - int(g[i]) for g in self.gaps_frames]
        images, past = {}, {}
        for c in self.cams:
            rng = torch.get_rng_state()
            images[c] = self._input(c, i, t)
            after = torch.get_rng_state()
            frames = []
            for tp in tps:
                if tp >= 0:
                    torch.set_rng_state(rng)
                    frames.append(self._input(c, i, tp))
                else:
                    frames.append(images[c].clone())    # no past yet: placeholder, masked by has_past = 0
            torch.set_rng_state(after)
            past[c] = torch.stack(frames)
        item = {"images": images, "bin": int(self.bin[i]), "value": float(self.value[i]), "row": i,
                "past": {"images": past, "has_past": torch.tensor([float(tp >= 0) for tp in tps])}}
        if self.proprio is not None:
            item["proprio"] = torch.from_numpy(self.proprio[i].copy())
        if self.still is not None:
            item["still"] = torch.tensor([self.still[i]], dtype=torch.float32)
        return item


def still_seconds(tab: pd.DataFrame, still_mps: float) -> pd.Series:
    """Per proprio row: seconds the gripper has been still at that frame (speed over the last 7 frames
    below still_mps, 15 Hz), counting back from the frame; 0 while moving or in the first 7 frames.
    Same rule as `vl.history_plots.stall_seconds`. Index-aligned with `tab`."""
    p = tab.sort_values(["repo", "episode_index", "frame_index"])
    g = p.groupby(["repo", "episode_index"])
    speed = np.sqrt(sum((p[a] - g[a].shift(7)) ** 2 for a in "xyz")) / (7 / 15)
    still = speed.fillna(np.inf) < still_mps
    key = [p.repo, p.episode_index]
    run = (~still).groupby(key).cumsum()
    return (still.groupby(key + [run]).cumsum() / 15).reindex(tab.index)


def load_still(index: pd.DataFrame, cfg: dict) -> np.ndarray:
    """(len(index),) float32 still seconds, row-aligned with the index; every frame must be found."""
    keys = ["repo", "episode_index", "frame_index"]
    st = cfg["still_time"]
    tab = pd.read_parquet(st["table"], columns=keys + list("xyz"))
    if (index.fps != 15).any():
        raise ValueError("still_seconds assumes 15 Hz")
    tab["still_s"] = still_seconds(tab, st["still_mps"])
    s = index[keys].merge(tab[keys + ["still_s"]], on=keys, how="left", validate="one_to_one").still_s.to_numpy(np.float32)
    if np.isnan(s).any():
        raise ValueError(f"{np.isnan(s).sum()} frames have no proprio row in {st['table']}")
    return s


def load_proprio(index: pd.DataFrame, cfg: dict) -> np.ndarray:
    """(len(index), 10) float32 raw proprio, row-aligned with the index; every frame must be found."""
    from vl.proprio import INPUT_NAMES
    keys = ["repo", "episode_index", "frame_index"]
    tab = pd.read_parquet(cfg["proprio"]["table"])
    m = index[keys].merge(tab, on=keys, how="left", validate="one_to_one")
    X = m[INPUT_NAMES].to_numpy(np.float32)
    if np.isnan(X).any():
        raise ValueError(f"{np.isnan(X).any(1).sum()} frames have no proprio row in {cfg['proprio']['table']}")
    return X


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
