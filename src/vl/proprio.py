"""Stage 5 proprioception: video <-> proprio timing recheck, input statistics, review videos.

Usage:
    .venv/bin/python -m vl.proprio --sync --per-repo 10 --workers 4   # -> reports/proprio_sync.md
    .venv/bin/python -m vl.proprio --stats                            # -> data/proprio/stats.json
    .venv/bin/python -m vl.proprio --table                            # -> data/proprio/frame_proprio.parquet
    .venv/bin/python -m vl.proprio --videos                           # -> data/review_videos/stage5_proprio/

Timing. Stage 3 found camera motion vs end-effector speed peaking at lag 0 or +6 frames. The eval
runner executes 6-step action chunks (`n_action_steps: 6` in every eval repo's results.json), so
autonomous speed is roughly 6-periodic and its lag-correlation has side peaks at +-6. This check
(1) pools the lag-correlation curve over many episodes instead of taking per-episode argmaxes, and
(2) splits teleop (c00, no chunks) from DAgger (c01-c05) and autonomous eval (r00-r04). Teleop is
the clean test of a file offset. Lag convention: corr(video[k + L], parquet[k]); L > 0 means the
video shows at frame k + L what the parquet records at row k.

Candidate proprio input [ours]: `observation.state` = EE position xyz (m), EE orientation as 3
Euler angles (rad; roll sits at +-pi and wraps), gripper position. The model would get
xyz (3) + the first two columns of the rotation matrix (6, continuous, no wrap) + gripper (1) = 10
numbers, each z-scored with training-split statistics only (`--stats`).
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vl.config import load_config
from vl.dataset import crop_resize
from vl.frames import FrameReader

LAGS = range(-12, 13)
GROUPS = {"teleop (c00)": "c00", "DAgger (c01-c05)": "c0[1-5]", "autonomous eval (r00-r04)": "r0[0-4]"}
FINGERS = (160, 300, 480, 480)   # x0, y0, x1, y1: finger region at the bottom of the 640x480 wrist frame
INPUT_NAMES = ["x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21", "gripper"]


# ---------------------------------------------------------------- trajectory -> proprio vector

def euler_to_matrix(e: np.ndarray, order: str = "xyz") -> np.ndarray:
    """(n, 3) angles -> (n, 3, 3). 'xyz' = extrinsic x, then y, then z: R = Rz @ Ry @ Rx.
    'XYZ' = intrinsic: R = Rx @ Ry @ Rz."""
    c, s = np.cos(e), np.sin(e)
    one, zero = np.ones(len(e)), np.zeros(len(e))
    Rx = np.stack([one, zero, zero, zero, c[:, 0], -s[:, 0], zero, s[:, 0], c[:, 0]], 1).reshape(-1, 3, 3)
    Ry = np.stack([c[:, 1], zero, s[:, 1], zero, one, zero, -s[:, 1], zero, c[:, 1]], 1).reshape(-1, 3, 3)
    Rz = np.stack([c[:, 2], -s[:, 2], zero, s[:, 2], c[:, 2], zero, zero, zero, one], 1).reshape(-1, 3, 3)
    return Rz @ Ry @ Rx if order == "xyz" else Rx @ Ry @ Rz


def proprio_vector(state: np.ndarray) -> np.ndarray:
    """(n, 7) observation.state -> (n, 10) [xyz, R[:, 0], R[:, 1], gripper] (unnormalized)."""
    R = euler_to_matrix(state[:, 3:6])
    return np.concatenate([state[:, :3], R[:, :, 0], R[:, :, 1], state[:, 6:7]], axis=1)


def episode_table(video_root: Path, repo: str, ep: int, cols: list[str]) -> pd.DataFrame:
    for f in sorted((video_root / repo / "data").rglob("*.parquet")):
        d = pd.read_parquet(f, columns=["episode_index", "frame_index"] + cols, filters=[("episode_index", "==", ep)])
        if len(d):
            return d.sort_values("frame_index").reset_index(drop=True)
    raise KeyError((repo, ep))


def arr(d: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(d[col].to_numpy()).astype(np.float64).reshape(len(d), -1)


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Angle (rad) of each relative rotation R[k+1] R[k]^T."""
    rel = R[1:] @ np.transpose(R[:-1], (0, 2, 1))
    return np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))


# ---------------------------------------------------------------- timing

def gray_frames(path: str, from_ts: float, n: int, fps: float) -> np.ndarray:
    """Frames 0..n-1 of an episode, full-resolution gray uint8 (n, 480, 640)."""
    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        c.seek(round(from_ts / tb), stream=s, backward=True)
        for fr in c.decode(s):
            if fr.pts * tb < from_ts - 0.5 / fps:
                continue
            out.append(fr.to_ndarray(format="gray"))
            if len(out) == n:
                break
    return np.stack(out)


def motion(F: np.ndarray, box=None) -> np.ndarray:
    """Mean absolute frame difference, after 4x4 average pooling (160x120)."""
    if box is not None:
        x0, y0, x1, y1 = box
        F = F[:, y0:y1, x0:x1]
    n, h, w = F.shape
    F = F[:, : h // 4 * 4, : w // 4 * 4].reshape(n, h // 4, 4, w // 4, 4).astype(np.float32).mean((2, 4))
    return np.abs(np.diff(F, axis=0)).mean((1, 2))


def lagged_corr(a, b, L):
    """Pearson correlation of a[k + L] with b[k]."""
    x, y = a[max(0, L):len(a) + min(0, L)], b[max(0, -L):len(b) - max(0, L)]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def autocorr(a, L):
    return lagged_corr(a, a, L) if L else 1.0


def episode_sync(job: dict) -> dict:
    """All lag curves for one episode."""
    d = episode_table(Path(job["video_root"]), job["repo"], int(job["episode_index"]),
                      ["observation.state", "observation.state.cartesian_velocity"])
    n = int(job["T"]) + 1
    st = arr(d, "observation.state")[:n]
    R = euler_to_matrix(st[:, 3:6])
    sig = {"pos": np.linalg.norm(np.diff(st[:, :3], axis=0), axis=1), "rot": rotation_angle(R),
           "grip": np.abs(np.diff(st[:, 6]))}
    vid = {}
    for cam in ["wrist_left", "side_1"]:
        F = gray_frames(job[f"path_{cam}"], job[f"from_ts_{cam}"], n, job["fps"])
        vid[cam] = motion(F)
        if cam == "wrist_left":
            vid["fingers"] = motion(F, FINGERS)
    m = min(len(v) for v in vid.values())   # r00/35 wrist_left is one frame short after T
    out = {k: job[k] for k in ["repo", "episode_index", "T"]} | {"n_motion": m}
    pairs = {"side_1~pos": ("side_1", "pos"), "wrist_left~pos": ("wrist_left", "pos"),
             "wrist_left~rot": ("wrist_left", "rot"), "fingers~grip": ("fingers", "grip")}
    for name, (v, s) in pairs.items():
        out[name] = [lagged_corr(vid[v][:m], sig[s][:m], L) for L in LAGS]
    out["autocorr_pos"] = [autocorr(sig["pos"], L) for L in range(0, 13)]
    # Internal parquet check: reported translational velocity vs position differences.
    vel = arr(d, "observation.state.cartesian_velocity")[:n, :3]
    dp = np.diff(st[:, :3], axis=0)
    out["vel~dpos"] = [lagged_corr(vel[1:, 0], dp[:, 0], L) for L in range(-3, 4)]
    # Euler convention: angular velocity from finite differences vs the reported rotational velocity.
    wrep = arr(d, "observation.state.cartesian_velocity")[:n, 3:6]
    for order in ["xyz", "XYZ"]:
        Ro = euler_to_matrix(st[:, 3:6], order)
        rel = Ro[1:] @ np.transpose(Ro[:-1], (0, 2, 1))   # world-frame increment
        w = np.stack([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], 1) / 2
        best = max(np.nanmean([lagged_corr(wrep[1:, j], w[:, j], L) for j in range(3)]) for L in range(-2, 3))
        out[f"euler_{order}"] = best
    return out


def peak(curve: np.ndarray) -> tuple[int, float]:
    """Integer argmax lag and a parabolic sub-frame refinement."""
    k = int(np.nanargmax(curve))
    L = list(LAGS)
    if 0 < k < len(curve) - 1:
        a, b, c = curve[k - 1], curve[k], curve[k + 1]
        den = a - 2 * b + c
        return L[k], L[k] + (0.5 * (a - c) / den if den else 0.0)
    return L[k], float(L[k])


def sync(idx: pd.DataFrame, video_root: Path, per_repo: int, workers: int, seed: int, out_md: Path):
    first = idx[idx.split == "train"].groupby("episode_uid").head(1)
    eps = first.groupby("repo").sample(per_repo, random_state=seed)
    cols = ["repo", "episode_index", "T", "fps", "path_wrist_left", "from_ts_wrist_left", "path_side_1", "from_ts_side_1"]
    jobs = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} | {"video_root": str(video_root)}
            for r in eps[cols].to_dict("records")]
    with Pool(workers) as p:
        res = pd.DataFrame(p.map(episode_sync, jobs, chunksize=1))
    Path("data/proprio").mkdir(parents=True, exist_ok=True)
    res.to_pickle("data/proprio/sync_curves.pkl")   # per-episode curves
    rng = np.random.default_rng(0)
    lines = [f"# Video <-> proprio timing recheck (Stage 5)\n",
             f"{len(res)} training episodes ({per_repo} per repo, seed {seed}), frames 0..T. "
             "Lag L: corr(video motion[k + L], parquet signal[k]); L > 0 = video late. Video motion = mean "
             "|frame difference| at 160x120 (gray); `fingers` = the finger region of the wrist view. Parquet "
             "signals: `pos` = |Δ EE position|, `rot` = EE rotation angle per step, `grip` = |Δ gripper|. Pooled "
             "curve = mean of the per-episode curves; 95% CI of its sub-frame peak from 1,000 episode bootstraps.\n"]
    rows, curves = [], {}
    for gname, pat in GROUPS.items():
        g = res[res.repo.str.contains(pat)]
        for pair in ["side_1~pos", "wrist_left~pos", "wrist_left~rot", "fingers~grip"]:
            C = np.array(g[pair].tolist())
            pooled = np.nanmean(C, 0)
            Li, Lf = peak(pooled)
            boot = [peak(np.nanmean(C[rng.integers(0, len(C), len(C))], 0))[1] for _ in range(1000)]
            per_ep = [list(LAGS)[int(np.nanargmax(c))] for c in C if np.isfinite(c).any() and np.nanmax(c) >= 0.5]
            r0, r6 = pooled[list(LAGS).index(0)], pooled[list(LAGS).index(6)]
            rows.append({"group": gname, "pair": pair, "episodes": len(C), "pooled peak": Li,
                         "sub-frame": round(Lf, 2), "95% CI": f"[{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]",
                         "r(0)": round(r0, 3), "r(+6)": round(r6, 3), "r(-6)": round(pooled[list(LAGS).index(-6)], 3),
                         "per-episode argmax (r>=0.5)": ", ".join(f"{k:+d}:{v}" for k, v in pd.Series(per_ep).value_counts().sort_index().items())})
            curves[(gname, pair)] = pooled
    lines += ["## Pooled lag peaks\n", pd.DataFrame(rows).to_markdown(index=False), ""]
    ac = pd.DataFrame({g: np.nanmean(np.array(res[res.repo.str.contains(p)]["autocorr_pos"].tolist()), 0)
                       for g, p in GROUPS.items()}, index=range(0, 13)).T.round(3)
    lines += ["## Autocorrelation of EE speed (6-step action chunks)\n",
              "Mean over episodes of corr(speed[k + L], speed[k]). A bump at L = 6 (and 12) is the chunk period.\n",
              ac.to_markdown(), ""]
    v = np.nanmean(np.array(res["vel~dpos"].tolist()), 0)
    lines += ["## Parquet-internal checks\n",
              "Reported `cartesian_velocity` x vs Δ position x, corr(vel[k + 1 + L], pos[k + 1] − pos[k]): "
              + ", ".join(f"L={L:+d}: {c:.3f}" for L, c in zip(range(-3, 4), v)) + "\n",
              f"Euler convention (mean corr of finite-difference world angular velocity with the reported "
              f"rotational velocity, best lag in −2..2): extrinsic `xyz` {res.euler_xyz.mean():.3f}, intrinsic `XYZ` "
              f"{res.euler_XYZ.mean():.3f}\n"]
    out_md.write_text("\n".join(lines))
    plot_curves(curves, ac, out_md.with_suffix(".png"))
    print("\n".join(lines))
    print("->", out_md)


def plot_curves(curves: dict, ac: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = ["side_1~pos", "wrist_left~pos", "wrist_left~rot", "fingers~grip"]
    fig, axs = plt.subplots(1, len(pairs) + 1, figsize=(19, 3.4))
    for ax, pair in zip(axs, pairs):
        for g in GROUPS:
            ax.plot(list(LAGS), curves[(g, pair)], marker=".", label=g)
        ax.axvline(0, color="gray", lw=0.6); ax.axvline(6, color="gray", lw=0.6, ls=":"); ax.axvline(-6, color="gray", lw=0.6, ls=":")
        ax.set_title(f"video {pair.split('~')[0]} vs parquet {pair.split('~')[1]}", fontsize=9)
        ax.set_xlabel("lag L (frames; > 0 = video late)"); ax.set_ylabel("pooled corr")
    axs[0].legend(fontsize=7)
    for g in GROUPS:
        axs[-1].plot(ac.columns, ac.loc[g], marker=".", label=g)
    axs[-1].set_title("EE speed autocorrelation", fontsize=9); axs[-1].set_xlabel("lag (frames)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print("->", path)


# ---------------------------------------------------------------- statistics

def frame_table(idx: pd.DataFrame, video_root: Path, out: Path):
    """Raw (unnormalized) 10-d input for every frame in the frame index (train + val; never test),
    keyed by (repo, episode_index, frame_index). Read by FrameDataset when proprio is enabled."""
    keys = idx[["repo", "episode_index", "frame_index"]]
    parts = []
    for repo, g in keys.groupby("repo"):
        for f in sorted((video_root / repo / "data").rglob("*.parquet")):
            d = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]).merge(g, on=["episode_index", "frame_index"])
            if len(d):
                X = proprio_vector(arr(d, "observation.state")).astype(np.float32)
                parts.append(d[["repo", "episode_index", "frame_index"]].assign(**{n: X[:, i] for i, n in enumerate(INPUT_NAMES)}))
    tab = pd.concat(parts, ignore_index=True)
    assert len(tab) == len(keys) and not tab.duplicated(["repo", "episode_index", "frame_index"]).any(), (len(tab), len(keys))
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out, index=False)
    print(f"{len(tab):,} frames -> {out}")


def stats(idx: pd.DataFrame, video_root: Path, out: Path):
    """Mean/std of the 10-d input over valid training frames (t <= T) only."""
    tr = idx[idx.split == "train"][["repo", "episode_index", "frame_index"]]
    parts = []
    for repo, g in tr.groupby("repo"):
        for f in sorted((video_root / repo / "data").rglob("*.parquet")):
            d = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
            d = d.merge(g, on=["episode_index", "frame_index"])
            if len(d):
                parts.append(proprio_vector(arr(d, "observation.state")))
    X = np.concatenate(parts)
    assert len(X) == len(tr), (len(X), len(tr))
    s = {"names": INPUT_NAMES, "mean": X.mean(0).tolist(), "std": X.std(0).tolist(), "n_frames": len(X),
         "min": X.min(0).tolist(), "max": X.max(0).tolist(), "split": "train", "frames": "t <= T"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1))
    print(pd.DataFrame({k: s[k] for k in ["mean", "std", "min", "max"]}, index=INPUT_NAMES).round(4).to_string())
    print("->", out)


# ---------------------------------------------------------------- review videos

VIDEO_PICKS = [   # (label, repo suffix, episode or None = sampled, outcome filter)
    ("success_autonomous", "r04-eval", None, "success"),
    ("timeout_holding_worst", "r04-eval", 101, None),      # Stage 5 worst case: p_succ ~0.9 while stuck
    ("timeout_holding_mirror", "r02-eval", 45, None),       # looks like r03/68 at the holder
    ("success_late_seat_mirror", "r03-eval", 68, None),     # seats only in the last 1-2 s
    ("failure", "r00-eval", None, "failure"),
    ("success_teleop", "c00-teleop-mixed", None, "success"),
    ("dagger_intervention", "c02-dagger-mixed", None, "human"),
]


def panel_figure(t, X, Z, v, human, fps, T, title):
    """Static plot of the 10 inputs over the episode; returns (PIL image, time -> pixel x)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(6.72, 4.6), dpi=100, sharex=True,
                            gridspec_kw={"height_ratios": [1.2, 1.2, 0.8, 0.8]})
    groups = [("position (z-scored)", [0, 1, 2], ["x", "y", "z"]),
              ("rotation 6D (z-scored)", list(range(3, 9)), INPUT_NAMES[3:9]),
              ("gripper (z-scored)", [9], ["gripper"])]
    for ax, (name, cols, labels) in zip(axs, groups):
        for c, lab in zip(cols, labels):
            ax.plot(t, Z[:, c], lw=1, label=lab)
        ax.set_ylabel(name, fontsize=6); ax.legend(fontsize=5, ncol=6, loc="upper left")
    axs[3].plot(t, v, color="black", lw=1.3); axs[3].set_ylim(-1.02, 0.02); axs[3].set_ylabel("target v", fontsize=6)
    for ax in axs:
        ax.tick_params(labelsize=6)
        ax.axvline(T / fps, color="gray", ls=":", lw=1)
        for a, b in human:
            ax.axvspan(a / fps, (b + 1) / fps, color="tab:blue", alpha=0.15, lw=0)
    axs[-1].set_xlabel("time (s)", fontsize=7)
    axs[0].set_title(title, fontsize=7)
    fig.tight_layout(h_pad=0.2)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    x0, x1 = axs[0].get_xlim()
    bb = axs[0].get_window_extent()
    tx = lambda s: int(bb.x0 + (s - x0) / (x1 - x0) * bb.width)
    plt.close(fig)
    return img, tx


def bars(Z_row: np.ndarray, w: int, h: int) -> Image.Image:
    """The model's 10 numbers at this frame as horizontal bars, clipped to +-3."""
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((4, 2), "model input (z-scored)", fill="black")
    mid, rowh = w // 2 + 20, (h - 18) // len(Z_row)
    d.line([(mid, 16), (mid, h)], fill="gray")
    for i, (name, z) in enumerate(zip(INPUT_NAMES, Z_row)):
        y = 18 + i * rowh
        L = int(np.clip(z, -3, 3) / 3 * (w - mid - 6))
        d.rectangle([min(mid, mid + L), y + 2, max(mid, mid + L), y + rowh - 3], fill=(70, 120, 200))
        d.text((4, y + 2), f"{name:>7} {z:+5.2f}", fill="black")
    return im


def videos(idx: pd.DataFrame, man: pd.DataFrame, video_root: Path, cfg: dict, st: dict, lag: int,
           vid_dir: Path, fig_dir: Path, seed: int):
    vid_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    mean, std = np.array(st["mean"]), np.array(st["std"])
    first = idx.groupby("episode_uid").head(1)
    reader, S = FrameReader(), cfg["image_size"]
    for k, (label, suffix, ep, filt) in enumerate(VIDEO_PICKS):
        cand = first[first.repo == f"real-marker-d2-{suffix}"]
        if ep is not None:
            e = cand[cand.episode_index == ep].iloc[0]
        else:
            cand = cand[cand.split == "train"]
            cand = cand[cand.has_human_frames] if filt == "human" else cand[cand.outcome == filt]
            e = cand.sample(1, random_state=seed).iloc[0]
        rows = idx[idx.episode_uid == e.episode_uid].sort_values("frame_index")
        d = episode_table(video_root, e.repo, int(e.episode_index), ["observation.state", "source"])
        n, fps = int(e["T"]) + 1, float(e.fps)
        X = proprio_vector(arr(d, "observation.state")[:n])
        Z = (X - mean) / std
        human = [(a, b) for a, b in _intervals(arr(d, "source")[:n, 0] == 1)]
        mrow = man[man.episode_uid == e.episode_uid].iloc[0] if "episode_uid" in man else None
        vlm = getattr(mrow, "machine_failure_mode", None) if mrow is not None else None
        name = f"{k:02d}_{label}_{suffix}_ep{int(e.episode_index):04d}"
        title = (f"{suffix} ep{int(e.episode_index)} ({e.split}) | {e.outcome}, T = {e['T'] / fps:.1f} s"
                 + (f" | VLM: {vlm}" if isinstance(vlm, str) else "") + f" | video frame t <-> state row t{lag:+d}")
        panel, tx = panel_figure(np.arange(n) / fps, X, Z, rows.value.to_numpy(), human, fps, int(e["T"]), title)
        panel.save(fig_dir / f"{name}.png")
        W = panel.width                          # 672 = 3 x 224
        H = S + 18 + panel.height
        H += H % 2
        st_raw = arr(d, "observation.state")[:n]
        with av.open(str(vid_dir / f"{name}.mp4"), "w") as c:
            vs = c.add_stream("libx264", rate=int(fps))
            vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
            for t in range(n):
                j = min(max(t + lag, 0), n - 1)
                fr = Image.new("RGB", (W, H), "white")
                for i, cam in enumerate(cfg["cameras"]):
                    img = reader.read(e[f"path_{cam}"], e[f"from_ts_{cam}"], t, fps)
                    fr.paste(Image.fromarray(crop_resize(img, cfg["crops"][cam], S)), (i * S, 0))
                fr.paste(bars(Z[j], W - 2 * S, S), (2 * S, 0))
                fr.paste(panel, (0, S + 18))
                dd = ImageDraw.Draw(fr)
                x = tx(t / fps)
                dd.line([(x, S + 18), (x, H)], fill="red", width=2)
                p, r = st_raw[j, :3] * 1000, np.degrees(st_raw[j, 3:6])
                hum = any(a <= t <= b for a, b in human)
                dd.text((4, S + 3), f"t={t / fps:5.2f}s  xyz=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) mm  "
                                    f"rpy=({r[0]:.0f},{r[1]:.0f},{r[2]:.0f})°  grip={st_raw[j, 6]:.3f}  "
                                    f"v={rows.value.iloc[t]:+.3f}{'  HUMAN' if hum else ''}", fill="black")
                c.mux(vs.encode(av.VideoFrame.from_ndarray(np.asarray(fr), format="rgb24")))
            c.mux(vs.encode())
        print("->", vid_dir / f"{name}.mp4")
    reader.close()


HOLDER_PAIRS = [("r04-eval", 101), ("r02-eval", 45), ("r02-eval", 44), ("r01-eval", 48),   # timeouts held at the holder
                ("r03-eval", 68), ("r02-eval", 42), ("r01-eval", 103), ("r03-eval", 132)]  # successes that seat late


def holder_compare(idx: pd.DataFrame, video_root: Path, path: Path):
    """Raw EE height and gripper over the last 20 s of the Stage 5 worst-case mirror pairs (val):
    does proprio separate 'stuck at the holder' from 'about to seat' where one image does not?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    first = idx.groupby("episode_uid").head(1)
    for suffix, ep in HOLDER_PAIRS:
        e = first[(first.repo == f"real-marker-d2-{suffix}") & (first.episode_index == ep)].iloc[0]
        n, fps = int(e["T"]) + 1, float(e.fps)
        st = arr(episode_table(video_root, e.repo, ep, ["observation.state"]), "observation.state")[:n]
        tt = (np.arange(n) - (n - 1)) / fps
        m = tt >= -20
        ls = "-" if e.outcome == "success" else "--"
        axs[0].plot(tt[m], st[m, 2] * 1000, ls, lw=1.2, label=f"{suffix} {ep} ({e.outcome})")
        axs[1].plot(tt[m], st[m, 6], ls, lw=1.2)
    axs[0].set_ylabel("EE z (mm)"); axs[1].set_ylabel("gripper position")
    for ax in axs:
        ax.set_xlabel("time to T (s)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Val worst-case mirror pairs: solid = success (seats late), dashed = timeout held at the holder", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print("->", path)


def _intervals(flag: np.ndarray):
    d = np.diff(np.r_[0, flag.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--per-repo", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lag", type=int, default=0, help="pair video frame t with state row t + lag")
    args = ap.parse_args()
    cfg = load_config(args.config)
    idx = pd.read_parquet(cfg["paths"]["frame_index"])
    vroot = Path(cfg["paths"]["video_root"])
    stats_path = Path("data/proprio/stats.json")
    if args.sync:
        sync(idx, vroot, args.per_repo, args.workers, args.seed, Path("reports/proprio_sync.md"))
    if args.stats:
        stats(idx, vroot, stats_path)
    if args.table:
        frame_table(idx, vroot, Path(cfg["proprio"]["table"]))
    if args.videos:
        man = pd.read_parquet(load_config(cfg["dataset_config"])["paths"]["manifest"])
        videos(idx, man, vroot, cfg, json.loads(stats_path.read_text()), args.lag,
               Path("data/review_videos/stage5_proprio"), Path("reports/figures/stage5/proprio"), args.seed)


if __name__ == "__main__":
    main()
