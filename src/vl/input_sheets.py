"""Stage 5.5 review sheets of the model's image input: Gate 0 (crop alignment) and Track A
(images after augmentation).

Usage:
    .venv/bin/python -m vl.input_sheets gate0   # -> reports/figures/stage5_5/gate0/
    .venv/bin/python -m vl.input_sheets aug     # -> reports/figures/stage5_5/aug/

Every 224 px image goes through the training code (FrameDataset -> default_collate -> normalize)
and is un-normalized back to pixels (`vl.data_figures.model_view`), so the sheets show the model's
actual input, augmentation included. Raw 640x480 frames are drawn at half size with the crop box.

gate0 also measures, per repo and camera, where the scene sits relative to the box [ours]:
  - motion map: mean |gray(t + 2) - gray(t)| over sampled frames of sampled episodes. For `side_1`
    (fixed camera) this is where the arm, marker and holder move. Reported: share of the motion
    inside the box, motion centroid minus box centre, the 2.5-97.5% motion extent.
  - static map: per-pixel temporal std of gray. For `wrist_left` (camera on the gripper) the
    low-std region is the gripper itself, so its position tells whether the mount moved.
  - camera shift: phase correlation of each repo's mean frame against the pooled mean frame
    (integer px; a fixed camera that moved between rounds shows up as a non-zero shift).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from vl.config import load_config
from vl.data_figures import model_view
from vl.dataset import FULL_ORDER, FrameDataset, crop_resize, input_overrides
from vl.frames import FrameReader

RAW = (320, 240)          # raw frame drawn at half size
GAP = 6
SHORT = lambda repo: repo.replace("real-marker-d2-", "")


def episode_picks(idx: pd.DataFrame, repo: str, seed: int = 0) -> list[tuple[str, int]]:
    """(label, row) for one repo: a success at start / approach / insertion / seated, plus a stall
    (timeout, 3/4 in) or, if the repo has no timeout, a failure at T."""
    sub = idx[idx.repo == repo]
    eps = sub.groupby("episode_uid").head(1)
    row_of = lambda uid, t: int(sub.index[(sub.episode_uid == uid) & (sub.frame_index == t)][0])
    s = eps[eps.outcome == "success"].sample(1, random_state=seed).iloc[0]
    picks = [(f"success {name}", row_of(s.episode_uid, int(round(q * s["T"]))))
             for q, name in [(0.0, "start"), (0.4, "approach"), (0.8, "insertion"), (1.0, "seated T")]]
    st = eps[eps.outcome == "timeout"]
    if len(st):
        e = st.sample(1, random_state=seed).iloc[0]
        picks.append(("stall (timeout, 3/4)", row_of(e.episode_uid, (3 * int(e["T"])) // 4)))
    elif (eps.outcome == "failure").any():
        e = eps[eps.outcome == "failure"].sample(1, random_state=seed).iloc[0]
        picks.append(("failure at T", row_of(e.episode_uid, int(e["T"]))))
    else:   # all-success repo (no timeout, no failure in train)
        picks.append(("success 0.6 (no fail/timeout)", row_of(s.episode_uid, int(round(0.6 * s["T"])))))
    return picks


def extreme_boxes(box, aug: dict, frame_hw) -> list[tuple]:
    """The corner cases of `jitter_box` for one augment dict: min/max scale x the four shift corners
    (clamped into the frame as training does unless crop_pad)."""
    x0, y0, x1, y1 = box
    out = []
    for sc in aug.get("crop_scale", [1.0, 1.0]):
        side = int(round((x1 - x0) * sc))
        s = aug.get("crop_shift_px", 0)
        for dx in (-s, s):
            for dy in (-s, s):
                cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
                nx0, ny0 = int(round(cx - side / 2)), int(round(cy - side / 2))
                if not aug.get("crop_pad"):
                    H, W = frame_hw
                    nx0, ny0 = min(max(nx0, 0), W - side), min(max(ny0, 0), H - side)
                out.append((nx0, ny0, nx0 + side, ny0 + side))
    return out


def draw_raw(reader, e, cam, box, extra_boxes=()) -> Image.Image:
    raw = Image.fromarray(reader.read(e[f"path_{cam}"], e[f"from_ts_{cam}"], int(e.frame_index), e.fps))
    d = ImageDraw.Draw(raw)
    for b in extra_boxes:
        d.rectangle(b, outline=(255, 200, 0), width=2)
    d.rectangle(box, outline="cyan", width=4)
    return raw.resize(RAW, Image.BILINEAR)


def gate0_sheet(ds, idx, cfg, train_cfg, out_dir: Path):
    """Per camera: one row per repo, five frames; each cell = raw frame with the box (cyan) and the
    A2 crop-jitter extremes (yellow), next to the 224 px model input."""
    reader, S = FrameReader(), cfg["image_size"]
    a2 = train_cfg["baseline_lowlr_aug_crop"]["augment"]
    repos = sorted(idx.repo.unique())
    picks = {r: episode_picks(idx, r) for r in repos}
    views = {r: model_view(ds, [ds[row] for _, row in picks[r]], cfg) for r in repos}
    cw, ch = RAW[0] + S + GAP, max(RAW[1], S) + 18
    for cam in ds.cams:
        sheet = Image.new("RGB", (5 * cw, len(repos) * ch + 20), "white")
        d = ImageDraw.Draw(sheet)
        d.text((4, 4), f"{cam}: crop {cfg['crops'][cam]} (cyan) -> {S}x{S}; yellow = A2 jitter extremes "
               f"(shift +-{a2['crop_shift_px']} px, scale {a2['crop_scale']}, clamped)", fill="black")
        for i, r in enumerate(repos):
            for k, ((label, row), v) in enumerate(zip(picks[r], views[r])):
                e = idx.loc[row]
                raw = draw_raw(reader, e, cam, cfg["crops"][cam],
                               extreme_boxes(cfg["crops"][cam], a2, (480, 640)))
                x, y = k * cw, 20 + i * ch
                sheet.paste(raw, (x, y + 16))
                sheet.paste(Image.fromarray(v[cam]), (x + RAW[0] + 2, y + 16))
                d.text((x + 2, y + 3), f"{SHORT(r)} ep{e.episode_index} t={e.frame_index}/{e['T']}  {label}",
                       fill="black")
        p = out_dir / f"backprojection_{cam}.jpg"
        sheet.save(p, quality=92)
        print("->", p)


def sample_frames(idx: pd.DataFrame, repo: str, n_eps: int, per_ep: int, seed: int = 0):
    """(row, row of the frame 2 later) pairs, evenly spaced through n_eps random episodes."""
    sub = idx[idx.repo == repo]
    uids = sub.episode_uid.drop_duplicates().sample(min(n_eps, sub.episode_uid.nunique()), random_state=seed)
    pairs = []
    for u in uids:
        ep = sub[sub.episode_uid == u].sort_values("frame_index")
        T = int(ep.frame_index.max())
        for t in np.linspace(0, max(T - 2, 0), per_ep).round().astype(int):
            a, b = ep.index[ep.frame_index == t], ep.index[ep.frame_index == t + 2]
            if len(a) and len(b):
                pairs.append((int(a[0]), int(b[0])))
    return pairs


def gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Integer (dx, dy) that best moves b onto a (phase correlation)."""
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    R = A * np.conj(B)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    return int(dx if dx <= W // 2 else dx - W), int(dy if dy <= H // 2 else dy - H)


def motion_stats(m: np.ndarray, box) -> dict:
    x0, y0, x1, y1 = box
    tot = m.sum()
    ys, xs = np.mgrid[:m.shape[0], :m.shape[1]]
    cx, cy = (m * xs).sum() / tot, (m * ys).sum() / tot
    q = lambda prof: (int(np.searchsorted(np.cumsum(prof) / prof.sum(), 0.025)),
                      int(np.searchsorted(np.cumsum(prof) / prof.sum(), 0.975)))
    (qx0, qx1), (qy0, qy1) = q(m.sum(0)), q(m.sum(1))
    return {"motion_in_box": m[y0:y1, x0:x1].sum() / tot,
            "centroid_dx": cx - (x0 + x1) / 2, "centroid_dy": cy - (y0 + y1) / 2,
            "motion_x": f"{qx0}-{qx1}", "motion_y": f"{qy0}-{qy1}"}


def heat(m: np.ndarray) -> Image.Image:
    v = np.clip(m / np.percentile(m, 99.5), 0, 1)
    rgb = np.stack([v, v ** 2, 1 - v], -1)     # blue (still) -> red/yellow (moving)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def alignment(idx, cfg, out_dir: Path, n_eps: int, per_ep: int):
    """Motion / static maps and camera-shift numbers per repo and camera -> alignment.md + maps."""
    reader = FrameReader()
    repos = sorted(idx.repo.unique())
    rows, maps = [], {}
    for r in repos:
        pairs = sample_frames(idx, r, n_eps, per_ep)
        for cam in cfg["cameras"]:
            g = []
            mot = np.zeros((480, 640), np.float64)
            for a, b in pairs:
                ea, eb = idx.loc[a], idx.loc[b]
                fa = gray(reader.read(ea[f"path_{cam}"], ea[f"from_ts_{cam}"], int(ea.frame_index), ea.fps))
                fb = gray(reader.read(eb[f"path_{cam}"], eb[f"from_ts_{cam}"], int(eb.frame_index), eb.fps))
                mot += np.abs(fb - fa)
                g.append(fa)
            g = np.stack(g)
            maps[r, cam] = (mot / len(pairs), g.mean(0), g.std(0))
            rows.append({"repo": SHORT(r), "camera": cam, "frames": len(pairs),
                         **motion_stats(maps[r, cam][0], cfg["crops"][cam])})
            print(f"{SHORT(r)} {cam}: {len(pairs)} frame pairs")
    df = pd.DataFrame(rows)
    for cam in cfg["cameras"]:
        pooled = np.mean([maps[r, cam][1] for r in repos], 0)
        for r in repos:
            dx, dy = phase_shift(pooled, maps[r, cam][1])
            sel = (df.repo == SHORT(r)) & (df.camera == cam)
            df.loc[sel, "cam_shift_dx"], df.loc[sel, "cam_shift_dy"] = dx, dy
    # Map sheet: per camera, one column per repo: mean frame, motion map, static (std) map, all with the box.
    for cam in cfg["cameras"]:
        box = cfg["crops"][cam]
        sheet = Image.new("RGB", (len(repos) * (RAW[0] + GAP), 3 * (RAW[1] + 16) + 20), "white")
        d = ImageDraw.Draw(sheet)
        d.text((4, 4), f"{cam}: rows = mean frame / motion |g(t+2)-g(t)| / temporal std; cyan = crop {box}",
               fill="black")
        for i, r in enumerate(repos):
            mot, mean, std = maps[r, cam]
            for j, im in enumerate([Image.fromarray(mean.clip(0, 255).astype(np.uint8)).convert("RGB"),
                                    heat(mot), heat(std)]):
                ImageDraw.Draw(im).rectangle(box, outline="cyan", width=4)
                x, y = i * (RAW[0] + GAP), 20 + j * (RAW[1] + 16)
                sheet.paste(im.resize(RAW, Image.BILINEAR), (x, y + 14))
                if j == 0:
                    d.text((x + 2, y + 1), SHORT(r), fill="black")
        p = out_dir / f"maps_{cam}.jpg"
        sheet.save(p, quality=90)
        print("->", p)
    for c in ("motion_in_box", "centroid_dx", "centroid_dy"):
        df[c] = df[c].round(3 if c == "motion_in_box" else 1)
    (out_dir / "alignment.md").write_text(
        "# Gate 0 alignment numbers (`vl.input_sheets gate0`)\n\n"
        f"{n_eps} random train episodes per repo, {per_ep} evenly spaced frame pairs (t, t+2) each. "
        "Pixel units of the 640x480 source frame. motion_in_box = share of mean |gray(t+2) - gray(t)| "
        "inside the crop box; centroid = motion centroid minus box centre (+x right, +y down); "
        "motion_x/y = 2.5-97.5% extent of the motion; cam_shift = phase-correlation shift of the repo's "
        "mean frame onto the pooled mean frame of all repos.\n\n" + df.to_markdown(index=False) + "\n")
    print(df.to_string(index=False))


AUG_ARMS = [("A1 photometric", "baseline_lowlr_aug_photo"), ("A2 crop jitter", "baseline_lowlr_aug_crop"),
            ("full aug", "baseline_lowlr_aug_full"), ("crop + full", "baseline_lowlr_aug_crop_full"),
            ("split: crop wrist / full side", "baseline_lowlr_aug_split_cropwrist")]


def aug_sheets(idx, cfg, train_cfg, out_dir: Path, n_draws: int):
    """Per frame: rows = augmentation arm x camera, columns = un-augmented input + n_draws draws, each
    drawn through FrameDataset with that arm's `augment` block (seeded per draw)."""
    S = cfg["image_size"]
    picks = episode_picks(idx, "real-marker-d2-r04-eval")
    frames = [p for p in picks if p[0] in ("success approach", "success insertion", "stall (timeout, 3/4)")]
    plain = FrameDataset(idx, cfg)
    for label, row in frames:
        e = idx.loc[row]
        nrow = len(AUG_ARMS) * len(cfg["cameras"])
        sheet = Image.new("RGB", (190 + (n_draws + 1) * (S + 4), nrow * (S + 4) + 24), "white")
        d = ImageDraw.Draw(sheet)
        d.text((4, 4), f"{SHORT(e.repo)} ep{e.episode_index} t={e.frame_index}/{e['T']} ({label}): column 1 = no "
               f"augmentation, then {n_draws} training draws (validation is never augmented)", fill="black")
        base = model_view(plain, [plain[row]], cfg)[0]
        for a, (name, section) in enumerate(AUG_ARMS):
            ds = FrameDataset(idx, cfg, augment=train_cfg[section]["augment"])
            draws = []
            for k in range(n_draws):
                torch.manual_seed(1000 * a + k)
                draws.append(ds[row])
            views = model_view(ds, draws, cfg)
            for c, cam in enumerate(cfg["cameras"]):
                y = 24 + (a * len(cfg["cameras"]) + c) * (S + 4)
                d.text((4, y + S // 2 - 12), f"{name}\n{cam}\n({section.removeprefix('baseline_lowlr_')})", fill="black")
                sheet.paste(Image.fromarray(base[cam]), (190, y))
                for k, v in enumerate(views):
                    sheet.paste(Image.fromarray(v[cam]), (190 + (k + 1) * (S + 4), y))
        tag = label.split(" (")[0].replace(" ", "_")
        p = out_dir / f"aug_{tag}.jpg"
        sheet.save(p, quality=92)
        print("->", p)


def box_compare(idx, cfg, cam: str, alt: tuple, out_dir: Path, repos=("c01-dagger-mixed", "c02-dagger-mixed",
                "r01-eval", "r04-eval")):
    """Current box (cyan) vs an alternative (magenta) for one camera: raw frame with both, then both
    224 inputs. Rows = repos, columns = start / approach / insertion / stall-or-end."""
    reader, S = FrameReader(), cfg["image_size"]
    cur = tuple(cfg["crops"][cam])
    cw, ch = RAW[0] + 2 * S + 2 * GAP, max(RAW[1], S) + 18
    sheet = Image.new("RGB", (4 * cw, len(repos) * ch + 20), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"{cam}: cyan = current {cur} (1st input), magenta = alternative {alt} (2nd input)", fill="black")
    for i, r in enumerate(repos):
        picks = episode_picks(idx, "real-marker-d2-" + r)
        picks = [picks[0], picks[1], picks[2], picks[4]]
        for k, (label, row) in enumerate(picks):
            e = idx.loc[row]
            img = reader.read(e[f"path_{cam}"], e[f"from_ts_{cam}"], int(e.frame_index), e.fps)
            raw = Image.fromarray(img)
            dr = ImageDraw.Draw(raw)
            dr.rectangle(cur, outline="cyan", width=5)
            dr.rectangle(alt, outline="magenta", width=5)
            x, y = k * cw, 20 + i * ch
            sheet.paste(raw.resize(RAW, Image.BILINEAR), (x, y + 16))
            for j, b in enumerate([cur, alt]):
                sheet.paste(Image.fromarray(crop_resize(img, b, S)), (x + RAW[0] + 2 + j * (S + GAP), y + 16))
            d.text((x + 2, y + 3), f"{r} ep{e.episode_index} t={e.frame_index}/{e['T']}  {label}", fill="black")
    p = out_dir / f"box_compare_{cam}.jpg"
    sheet.save(p, quality=92)
    print("->", p)


# Lars's LeRobot ImageTransformsConfig (2026-10-21), as `baseline_lowlr_aug_full` encodes it: per image
# 3 of these 6, uniform weights, applied in this order (random_order=False). (low, high) per transform.
LEROBOT = {"brightness": (0.8, 1.2), "contrast": (0.8, 1.2), "saturation": (0.5, 1.5), "hue": (-0.05, 0.05),
           "sharpness": (0.5, 1.5), "affine": ((-5.0, -0.05), (5.0, 0.05))}   # affine: (degrees, translate)


def fixed_op(x: torch.Tensor, name: str, v) -> torch.Tensor:
    """One transform of `vl.dataset.full_augment` at a fixed value instead of a random draw (same
    torchvision calls). affine: v = (degrees, tx, ty), tx/ty as a fraction of the image size."""
    import torchvision.transforms.v2.functional as TF
    if name == "affine":
        H, W = x.shape[-2:]
        return TF.affine(x, angle=v[0], translate=[round(v[1] * W), round(v[2] * H)], scale=1.0,
                         shear=[0.0, 0.0], interpolation=TF.InterpolationMode.NEAREST, fill=0)
    return {"brightness": TF.adjust_brightness, "contrast": TF.adjust_contrast, "saturation": TF.adjust_saturation,
            "hue": TF.adjust_hue, "sharpness": TF.adjust_sharpness}[name](x, v)


def check_fixed_ops(x: torch.Tensor, full: dict):
    """fixed_op must reproduce full_augment exactly: replay full_augment's RNG draws for each transform
    on its own (n = 1) and compare pixels."""
    from vl.dataset import full_augment, pick_subset
    for name in FULL_ORDER:
        cfg1 = {"n": 1, name: full[name]}
        torch.manual_seed(7)
        ref = full_augment(x, cfg1)
        torch.manual_seed(7)
        pick_subset(1, 1)
        u = lambda j: (torch.rand(()).item() * 2 - 1) * j
        if name == "affine":
            a = full["affine"]
            v = (u(a["degrees"]), u(a["translate"] * x.shape[-1]) / x.shape[-1], u(a["translate"] * x.shape[-2]) / x.shape[-2])
        elif name in ("brightness", "contrast", "saturation", "sharpness"):
            v = 1 + u(full[name])
        else:
            v = u(full[name])
        assert torch.equal(fixed_op(x, name, v), ref), f"fixed_op != full_augment for {name}"


def lerobot_sheets(idx, cfg, train_cfg, out_dir: Path):
    """Lars's augmentation at its extremes, not random draws.
    1. singles.jpg: each transform alone at low / none / high, both cameras.
    2. combos_{cam}.jpg: all 20 three-transform subsets (what training applies), each at all-low and
       all-high, so every combination's worst case is on the sheet once.
    Also hue numbers: median hue of the holder (orange) and the table edge (yellow) before/after the
    hue extremes -> hue.md."""
    from itertools import combinations
    full = train_cfg["baseline_lowlr_aug_full"]["augment"]["full"]
    want = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.5, "hue": 0.05, "sharpness": 0.5,
            "affine": {"degrees": 5.0, "translate": 0.05}}
    assert full == {"n": 3, **want}, f"baseline_lowlr_aug_full is not Lars's config: {full}"
    S = cfg["image_size"]
    plain = FrameDataset(idx, cfg)
    label, row = episode_picks(idx, "real-marker-d2-r04-eval")[2]          # r04 success, insertion
    e = idx.loc[row]
    base = {c: torch.from_numpy(v).permute(2, 0, 1).contiguous() for c, v in model_view(plain, [plain[row]], cfg)[0].items()
            if c in cfg["cameras"]}
    for c in cfg["cameras"]:
        check_fixed_ops(base[c], full)
    print("fixed_op reproduces full_augment for all 6 transforms (both cameras)")

    def val(name, side):
        lo, hi = LEROBOT[name]
        if name == "affine":
            (d, t) = lo if side == 0 else hi
            return (d, t, t)
        return lo if side == 0 else hi
    show = lambda x: Image.fromarray(x.permute(1, 2, 0).numpy())
    title = f"{SHORT(e.repo)} ep{e.episode_index} t={e.frame_index}/{e['T']} ({label}); 224 px loader input"

    # 1. singles
    rows = [("brightness", "x0.8", "x1.2"), ("contrast", "x0.8", "x1.2"), ("saturation", "x0.5", "x1.5"),
            ("hue", "-0.05 (-18 deg)", "+0.05 (+18 deg)"), ("sharpness", "x0.5", "x1.5"),
            ("rotate", "-5 deg", "+5 deg"), ("translate", "-5% x,y (-11 px)", "+5% x,y (+11 px)")]
    sheet = Image.new("RGB", (120 + 6 * (S + 4) + 10, 40 + len(rows) * (S + 18)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), title + ". Each transform alone at its two extremes; middle column = no transform", fill="black")
    for c, cam in enumerate(cfg["cameras"]):
        d.text((120 + c * 3 * (S + 4) + 10 * c, 22), f"{cam}:   low   |   none   |   high", fill="black")
    for i, (name, lo, hi) in enumerate(rows):
        y = 40 + i * (S + 18)
        d.text((4, y + S // 2), name, fill="black")
        for c, cam in enumerate(cfg["cameras"]):
            x0 = 120 + c * (3 * (S + 4) + 10)
            for j, side in enumerate([0, None, 1]):
                if side is None:
                    im = base[cam]
                elif name == "rotate":
                    im = fixed_op(base[cam], "affine", ((-5.0, 5.0)[side], 0.0, 0.0))
                elif name == "translate":
                    t = (-0.05, 0.05)[side]
                    im = fixed_op(base[cam], "affine", (0.0, t, t))
                else:
                    im = fixed_op(base[cam], name, val(name, side))
                sheet.paste(show(im), (x0 + j * (S + 4), y + 14))
                d.text((x0 + j * (S + 4), y + 1), "none" if side is None else (lo, hi)[side], fill="black")
    p = out_dir / "singles.jpg"
    sheet.save(p, quality=92)
    print("->", p)

    # 2. combos: 20 subsets x (all low, all high); affine at both extremes = rotate +-5 and shift +-5%
    combos = list(combinations(FULL_ORDER, 3))
    for cam in cfg["cameras"]:
        cols = 8
        sheet = Image.new("RGB", (cols * (S + 4) + 4, 24 + (len(combos) * 2 // cols) * (S + 30)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((4, 4), f"{cam}: {title}. All 20 combinations of 3 transforms (applied in order "
               f"{', '.join(FULL_ORDER)}), each at all-low then all-high", fill="black")
        for k, combo in enumerate(combos):
            for side in (0, 1):
                im = base[cam]
                for name in combo:
                    im = fixed_op(im, name, val(name, side))
                t = 2 * k + side
                x, y = (t % cols) * (S + 4) + 2, 24 + (t // cols) * (S + 30)
                d.text((x, y + 1), f"{'+'.join(n[:5] for n in combo)}\n{'low' if side == 0 else 'high'}", fill="black")
                sheet.paste(show(im), (x, y + 28))
        p = out_dir / f"combos_{cam}.jpg"
        sheet.save(p, quality=92)
        print("->", p)

    # hue numbers: holder = saturated orange/red pixels, table edge = saturated yellow pixels
    lines = ["# Hue check (`vl.input_sheets lerobot`)", "",
             f"{title}. Hue on a 0-360 deg wheel (PIL HSV), circular mean over saturated pixels (S > 0.5, V > 0.3) of "
             "each colour class, classified on the un-augmented image; the same pixels are re-measured after "
             "the shift (`adjust_hue` +-0.05 = +-18 deg).", "",
             "| camera | class | pixels | hue none | hue -0.05 | hue +0.05 |", "|---|---|---|---|---|---|"]
    for cam in cfg["cameras"]:
        hsv = lambda x: np.asarray(show(x).convert("HSV")).astype(np.float32) / 255
        h0 = hsv(base[cam])
        sat = (h0[..., 1] > 0.5) & (h0[..., 2] > 0.3)
        hue0 = h0[..., 0] * 360
        classes = {"holder (orange/red)": sat & ((hue0 < 35) | (hue0 > 340)),
                   "table edge (yellow)": sat & (hue0 >= 40) & (hue0 < 70)}
        shifted = {s: hsv(fixed_op(base[cam], "hue", s))[..., 0] * 360 for s in (-0.05, 0.05)}
        circ = lambda h: float(np.degrees(np.angle(np.exp(1j * np.radians(h)).mean())) % 360) if h.size else float("nan")
        for cname, m in classes.items():
            lines.append(f"| {cam} | {cname} | {int(m.sum())} | {circ(hue0[m]):.0f} | {circ(shifted[-0.05][m]):.0f} | "
                         f"{circ(shifted[0.05][m]):.0f} |")
    lines += ["", "Hue values here are circular means (red wraps at 0/360)."]
    (out_dir / "hue.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# Stage 5.5 Gate 0 arms: (name, --crop args, --resize args) for vl.train_baseline [ours, Karen 2026-10-23].
GATE0_ARMS = [("R0 current", [], []),
              ("R1 wrist flush right", ["wrist_left=160,0,640,480"], []),
              ("R1b wrist half shift", ["wrist_left=120,0,600,480"], []),
              ("R2 squash", [], ["wrist_left=squash", "side_1=squash"]),
              ("R3 letterbox", [], ["wrist_left=letterbox", "side_1=letterbox"]),
              ("R4 side full height", ["side_1=80,0,560,480"], [])]


def arm_sheet(idx, cfg, out_dir: Path):
    """Each Gate 0 arm's un-augmented 224 input (loader path), both cameras, on the same frames:
    r04 approach / insertion, c01 approach (holder upper left in the wrist view), r01 stall."""
    S = cfg["image_size"]
    frames = [episode_picks(idx, "real-marker-d2-r04-eval")[1], episode_picks(idx, "real-marker-d2-r04-eval")[2],
              episode_picks(idx, "real-marker-d2-c01-dagger-mixed")[1], episode_picks(idx, "real-marker-d2-r01-eval")[4]]
    cols = len(frames) * len(cfg["cameras"])
    sheet = Image.new("RGB", (170 + cols * (S + 4), 40 + len(GATE0_ARMS) * (S + 4)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), "Gate 0 arms: each arm's model input (no augmentation; training adds the full aug on top)", fill="black")
    for k, (label, row) in enumerate(frames):
        e = idx.loc[row]
        for c, cam in enumerate(cfg["cameras"]):
            d.text((170 + (k * len(cfg["cameras"]) + c) * (S + 4), 22),
                   f"{SHORT(e.repo)} {label.replace('success ', '')} / {cam}", fill="black")
    for a, (name, crops, resize) in enumerate(GATE0_ARMS):
        ds = FrameDataset(idx, input_overrides(cfg, crops, resize))
        views = model_view(ds, [ds[r] for _, r in frames], cfg)
        y = 40 + a * (S + 4)
        d.text((4, y + S // 2 - 20), name + "\n" + "\n".join(crops + resize), fill="black")
        for k, v in enumerate(views):
            for c, cam in enumerate(cfg["cameras"]):
                sheet.paste(Image.fromarray(v[cam]), (170 + (k * len(cfg["cameras"]) + c) * (S + 4), y))
    p = out_dir / "arms.jpg"
    sheet.save(p, quality=92)
    print("->", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate0", "aug", "lerobot", "box", "arms"])
    ap.add_argument("--box-cam", default="wrist_left")
    ap.add_argument("--alt-box", default="160,0,640,480", help="box: x0,y0,x1,y1 of the alternative box")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--out", default="reports/figures/stage5_5")
    ap.add_argument("--eps", type=int, default=6, help="gate0: episodes per repo for the maps")
    ap.add_argument("--per-ep", type=int, default=20, help="gate0: frame pairs per episode")
    ap.add_argument("--draws", type=int, default=8, help="aug: draws per arm")
    ap.add_argument("--skip-maps", action="store_true")
    args = ap.parse_args()
    cfg, tcfg = load_config(args.config), load_config(args.train_config)
    idx = pd.read_parquet(cfg["paths"]["frame_index"])
    idx = idx[idx.split == "train"].reset_index(drop=True)
    out = Path(args.out) / args.what
    out.mkdir(parents=True, exist_ok=True)
    if args.what == "gate0":
        gate0_sheet(FrameDataset(idx, cfg), idx, cfg, tcfg, out)
        if not args.skip_maps:
            alignment(idx, cfg, out, args.eps, args.per_ep)
    elif args.what == "aug":
        aug_sheets(idx, cfg, tcfg, out, args.draws)
    elif args.what == "arms":
        arm_sheet(idx, cfg, Path(args.out) / "gate0")
    elif args.what == "lerobot":
        lerobot_sheets(idx, cfg, tcfg, out)
    else:
        box_compare(idx, cfg, args.box_cam, tuple(int(v) for v in args.alt_box.split(",")), out)


if __name__ == "__main__":
    main()
