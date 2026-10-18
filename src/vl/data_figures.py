"""Review material for the Stage 3 data pipeline: crops (3.4), Check B alignment (3.5), sync lag,
and short "what the model sees" videos.

Usage:
    .venv/bin/python -m vl.data_figures        # -> reports/figures/stage3/, data/review_videos/

Everything goes through the same code the training loop uses (FrameDataset -> DataLoader ->
normalize), then `unnormalize` back to pixels, so the pictures show the model's actual input.
Reference values (timestamp, target) are recomputed independently from the trajectory parquet
and the Section 8 formula, not copied from the frame index.
"""

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torch.utils.data import default_collate

from vl.config import load_config
from vl.dataset import FrameDataset, normalize, unnormalize
from vl.frames import FrameReader

RAW = (320, 240)


def model_view(ds: FrameDataset, items: list[dict], cfg: dict) -> list[dict]:
    """Batch items the way training does and return each camera as the model sees it (uint8 HWC)."""
    b = default_collate(items)                      # the DataLoader's own batching
    for c in ds.cams:
        x = normalize(b["images"][c], cfg["imagenet_mean"], cfg["imagenet_std"])   # model input
        b["images"][c] = unnormalize(x, cfg["imagenet_mean"], cfg["imagenet_std"])
    return [{c: b["images"][c][k].permute(1, 2, 0).numpy() for c in ds.cams}
            | {"row": int(b["row"][k]), "bin": int(b["bin"][k]), "value": float(b["value"][k])}
            for k in range(len(items))]


def pick_review_rows(idx: pd.DataFrame, man: pd.DataFrame) -> list[tuple[str, int]]:
    """(label, row) pairs for the crop sheet: task phases, a stall, a failure end, init extremes."""
    first = idx.groupby("episode_uid").head(1).merge(
        man[["episode_uid", "init_pen_x", "init_pen_y", "init_holder_x", "init_holder_y"]], on="episode_uid")
    row_of = lambda uid, t: int(idx.index[(idx.episode_uid == uid) & (idx.frame_index == t)][0])
    picks = []
    s = first[(first.outcome == "success") & (first.repo == "real-marker-d2-r04-eval")].sample(1, random_state=0).iloc[0]
    for q, name in [(0.0, "start"), (0.4, "approach"), (0.8, "insertion"), (1.0, "seated (T)")]:
        picks.append((f"success {name}", row_of(s.episode_uid, int(round(q * s["T"])))))
    f = first[first.outcome == "failure"].sample(1, random_state=0).iloc[0]
    picks.append(("failure at T", row_of(f.episode_uid, int(f["T"]))))
    st = first[(first.outcome == "timeout") & first.soft_truncate].sample(1, random_state=0).iloc[0]
    picks.append(("stall (timeout, mid)", row_of(st.episode_uid, int(st["T"]) // 2 + int(st["T"]) // 4)))
    for col in ["init_pen_x", "init_holder_y"]:
        for e, tag in [(first.loc[first[col].idxmin()], "min"), (first.loc[first[col].idxmax()], "max")]:
            picks.append((f"{col} {tag}, f=0", row_of(e.episode_uid, 0)))
    return picks


def crop_sheets(ds, idx, man, cfg, out_dir: Path):
    picks = pick_review_rows(idx, man)
    views = model_view(ds, [ds[r] for _, r in picks], cfg)
    reader, S = FrameReader(), cfg["image_size"]
    for cam in ds.cams:
        W = RAW[0] + S + 12
        sheet = Image.new("RGB", (2 * W, ((len(picks) + 1) // 2) * (RAW[1] + 20)), "white")
        d = ImageDraw.Draw(sheet)
        for k, ((label, r), v) in enumerate(zip(picks, views)):
            e = idx.loc[r]
            raw = Image.fromarray(reader.read(e[f"path_{cam}"], e[f"from_ts_{cam}"], int(e.frame_index), e.fps))
            sx = RAW[0] / raw.width
            ImageDraw.Draw(raw).rectangle(cfg["crops"][cam], outline="cyan", width=4)
            x, y = (k % 2) * W, (k // 2) * (RAW[1] + 20)
            sheet.paste(raw.resize(RAW), (x, y + 16))
            sheet.paste(Image.fromarray(v[cam]), (x + RAW[0] + 4, y + 16))
            d.text((x + 2, y + 2), f"{label}: {e.repo[15:]} ep{e.episode_index} t={e.frame_index}/{e['T']}",
                   fill="black")
        p = out_dir / f"crops_{cam}.jpg"
        sheet.save(p, quality=90)
        print("->", p)


def parquet_row(video_root: Path, repo: str, ep: int, t: int) -> pd.Series:
    cols = ["episode_index", "frame_index", "timestamp", "is_valid"]
    for f in sorted((video_root / repo / "data").rglob("*.parquet")):
        d = pd.read_parquet(f, columns=cols, filters=[("episode_index", "==", ep), ("frame_index", "==", t)])
        if len(d):
            return d.iloc[0]
    raise KeyError((repo, ep, t))


def check_b(ds, idx, cfg, tcfg, out_dir: Path, n=8):
    """Samples through the real loader, annotated with independently recomputed references."""
    rows = idx.groupby("outcome").sample(n // 3 + 1, random_state=1).sample(n, random_state=1).index.tolist()
    views = model_view(ds, [ds[r] for r in rows], cfg)
    S = cfg["image_size"]
    sheet = Image.new("RGB", (2 * S + 300, n * (S + 6)), "white")
    d = ImageDraw.Draw(sheet)
    lines = []
    for k, (r, v) in enumerate(zip(rows, views)):
        e = idx.loc[r]
        pq = parquet_row(Path(cfg["paths"]["video_root"]), e.repo, int(e.episode_index), int(e.frame_index))
        failed = e.outcome in tcfg["failure_outcomes"]
        v_ref = (-(e["T"] - e.frame_index) / e.fps - tcfg["c_fail_s"] * failed) / (tcfg["horizon_s"] + tcfg["c_fail_s"])
        centers = np.linspace(tcfg["v_min"], tcfg["v_max"], tcfg["n_bins"])
        ok = (abs(v["value"] - v_ref) < 1e-9 and int(pq.is_valid) == 1 and v["row"] == r
              and abs(pq.timestamp - e.frame_index / e.fps) < 1e-3 and abs(centers[v["bin"]] - v_ref) <= 0.0025 + 1e-9)
        y = k * (S + 6)
        sheet.paste(Image.fromarray(v["wrist_left"]), (0, y))
        sheet.paste(Image.fromarray(v["side_1"]), (S + 2, y))
        txt = (f"{e.repo[15:]} ep{e.episode_index}\nt = {e.frame_index} (parquet ts {pq.timestamp:.3f} s)\n"
               f"outcome = {e.outcome}, T = {e['T']}\ntarget v = {v['value']:.4f} (recomputed {v_ref:.4f})\n"
               f"bin = {v['bin']} (center {centers[v['bin']]:.3f})\nis_valid = {int(pq.is_valid)}  "
               f"{'OK' if ok else 'MISMATCH'}")
        d.multiline_text((2 * S + 10, y + 4), txt, fill="black" if ok else "red")
        lines.append({"repo": e.repo, "episode": int(e.episode_index), "t": int(e.frame_index), "T": int(e["T"]),
                      "outcome": e.outcome, "value": round(v["value"], 4), "recomputed": round(v_ref, 4),
                      "bin": v["bin"], "parquet_ts": round(float(pq.timestamp), 3), "checks_ok": ok})
    p = out_dir / "check_b_alignment.jpg"
    sheet.save(p, quality=92)
    print("->", p)
    return pd.DataFrame(lines)


def motion_signal(path, from_ts, n, fps):
    """Mean absolute frame difference (80x60 gray) for frames 0..n-1 of an episode."""
    with av.open(path) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        c.seek(round(from_ts / tb), stream=s, backward=True)
        frames = []
        for fr in c.decode(s):
            if fr.pts * tb < from_ts - 0.5 / fps:
                continue
            frames.append(fr.reformat(width=80, height=60, format="gray").to_ndarray().astype(np.float32))
            if len(frames) == n:
                break
    return np.abs(np.diff(np.stack(frames), axis=0)).mean(axis=(1, 2))


def trajectory(video_root: Path, repo: str, ep: int, col: str) -> np.ndarray:
    for f in sorted((video_root / repo / "data").rglob("*.parquet")):
        d = pd.read_parquet(f, columns=["episode_index", "frame_index", col], filters=[("episode_index", "==", ep)])
        if len(d):
            return np.stack(d.sort_values("frame_index")[col].to_numpy()).astype(np.float64)
    raise KeyError((repo, ep))


def lagged_corr(a, b, L):
    """Pearson correlation of a[k + L] with b[k]."""
    x, y = a[max(0, L):len(a) + min(0, L)], b[max(0, -L):len(b) - max(0, L)]
    return float(np.corrcoef(x, y)[0, 1])


def sync_lags(idx, video_root: Path, n_per_repo=3, max_lag=10):
    """Per camera: lag (frames) at which image motion best matches end-effector speed from the
    parquet. 0 = video frame k shows the robot as recorded in row k. Only meaningful when the
    correlation is clearly positive; flat curves make the argmax arbitrary (see r_at_0)."""
    eps = idx.groupby("episode_uid").head(1)
    rows = []
    for _, e in eps.groupby("repo").sample(n_per_repo, random_state=7).iterrows():
        n = int(e["T"]) + 1
        pos = trajectory(video_root, e.repo, int(e.episode_index), "observation.state.cartesian_position")[:n, :3]
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        out = {"repo": e.repo, "episode": int(e.episode_index)}
        for cam in ["wrist_left", "side_1"]:
            m = motion_signal(e[f"path_{cam}"], e[f"from_ts_{cam}"], n, e.fps)
            c = {L: lagged_corr(m, speed, L) for L in range(-max_lag, max_lag + 1)}
            best = max(c, key=c.get)
            out |= {f"{cam}_lag": best, f"{cam}_r": round(c[best], 2), f"{cam}_r_at_0": round(c[0], 2)}
        rows.append(out)
    return pd.DataFrame(rows)


def gripper_event_lags(idx, video_root: Path, n_per_repo=5, win=12):
    """Frame of fastest gripper closing (parquet) vs frame of largest change in the finger region
    of the wrist view. A constant video/parquet offset would show the same lag everywhere."""
    eps = idx.groupby("episode_uid").head(1)
    rows = []
    for _, e in eps.groupby("repo").sample(n_per_repo, random_state=3).iterrows():
        g = trajectory(video_root, e.repo, int(e.episode_index), "observation.state.gripper_position")[: int(e["T"]) + 1]
        dg = np.diff(g.ravel())
        k = int(np.argmax(dg))
        if dg[k] < 0.05:
            continue                     # gripper never closes clearly
        lo, hi = max(1, k - win), min(len(g) - 1, k + win)
        x0, y0, x1, y1 = 160, 300, 480, 480   # fingers: bottom centre of the 640x480 wrist frame
        with av.open(e.path_wrist_left) as c:
            s = c.streams.video[0]
            tb = float(s.time_base)
            c.seek(round(e.from_ts_wrist_left / tb), stream=s, backward=True)
            F = []
            for fr in c.decode(s):
                if fr.pts * tb < e.from_ts_wrist_left - 0.5 / e.fps:
                    continue
                F.append(fr.to_ndarray(format="gray")[y0:y1, x0:x1].astype(np.float32))
                if len(F) == hi + 1:
                    break
        m = np.abs(np.diff(np.stack(F), axis=0)).mean(axis=(1, 2))
        j = lo + int(np.argmax(m[lo:hi]))
        rows.append({"repo": e.repo, "episode": int(e.episode_index), "grip_close_step": k,
                     "finger_motion_peak": j, "lag": j - k})
    return pd.DataFrame(rows)


def review_videos(ds, idx, cfg, out_dir: Path):
    """Both cropped views side by side, with t and the target value, for a few episodes."""
    first = idx.groupby("episode_uid").head(1)
    picks = {
        "success_autonomous": first[(first.outcome == "success") & (first.repo == "real-marker-d2-r04-eval")],
        "success_teleop": first[(first.outcome == "success") & (first.repo == "real-marker-d2-c00-teleop-mixed")],
        "failure": first[first.outcome == "failure"],
        "timeout_stall": first[(first.outcome == "timeout") & first.soft_truncate],
        "intervention": first[first.has_human_frames & (first.repo != "real-marker-d2-c00-teleop-mixed")],
    }
    S, made = cfg["image_size"], []
    for name, cand in picks.items():
        e = cand.sample(1, random_state=0).iloc[0]
        rows = idx.index[idx.episode_uid == e.episode_uid].tolist()
        p = out_dir / f"{name}_{e.repo[15:]}_ep{e.episode_index:04d}.mp4"
        with av.open(str(p), "w") as c:
            st = c.add_stream("libx264", rate=int(e.fps))
            st.width, st.height, st.pix_fmt = 2 * S, S + 28, "yuv420p"
            for r in rows:
                it = ds[r]
                fr = Image.new("RGB", (2 * S, S + 28), "black")
                fr.paste(Image.fromarray(it["images"]["wrist_left"].permute(1, 2, 0).numpy()), (0, 0))
                fr.paste(Image.fromarray(it["images"]["side_1"].permute(1, 2, 0).numpy()), (S, 0))
                t = int(idx.at[r, "frame_index"])
                ImageDraw.Draw(fr).text((4, S + 4), f"{e.outcome} t={t}/{e['T']} ({t / e.fps:5.2f}s)  "
                                                     f"target v={it['value']:+.3f}", fill="white")
                c.mux(st.encode(av.VideoFrame.from_ndarray(np.asarray(fr), format="rgb24")))
            c.mux(st.encode())
        made.append(p)
        print("->", p)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config("configs/targets.yaml")
    ds_cfg = load_config(cfg["dataset_config"])
    idx = pd.read_parquet(cfg["paths"]["frame_index"])
    idx = idx[idx.split == "train"].reset_index(drop=True)
    man = pd.read_parquet(ds_cfg["paths"]["manifest"])
    ds = FrameDataset(idx, cfg)
    out = Path(cfg["paths"]["figure_dir"])
    out.mkdir(parents=True, exist_ok=True)
    crop_sheets(ds, idx, man, cfg, out)
    b = check_b(ds, idx, cfg, tcfg, out)
    print(b.to_string(index=False))
    vroot = Path(cfg["paths"]["video_root"])
    lags = sync_lags(idx, vroot)
    print(lags.to_string(index=False))
    grip = gripper_event_lags(idx, vroot)
    print(grip.lag.value_counts().sort_index().to_string())
    vids = Path("data/review_videos")
    vids.mkdir(parents=True, exist_ok=True)
    review_videos(ds, idx, cfg, vids)
    (out / "check_b_alignment.md").write_text(
        "# Check B samples\n\n" + b.to_markdown(index=False)
        + "\n\n# Camera motion vs end-effector speed\n\nPer camera, the lag (frames) maximizing the "
        "correlation of image motion with parquet end-effector speed; r_at_0 is the correlation at lag 0.\n\n"
        + lags.to_markdown(index=False)
        + "\n\n# Gripper-close event vs wrist finger motion\n\n" + grip.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
