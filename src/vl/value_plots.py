"""Synchronized camera / value plots for systematically chosen validation episodes (Stage 4).

Usage:
    .venv/bin/python -m vl.value_plots --run baseline
        -> reports/figures/stage4/<run>/<k>_<kind>_<repo>_ep<idx>.png   (filmstrip + curve)
           data/review_videos/stage4/<run>/<same name>.mp4             (video + moving cursor)

Reads the run's val_predictions.parquet (the best checkpoint's per-frame V); no GPU needed.
With --dist, the background shows the model's full predicted distribution over the 201 value bins
at every frame (darker = more probability), from data/runs/<run>/val_probs_plotted.npz or
val_probs.npz (`vl.value_dist`); V is the mean of that column. Outputs get a `_dist` suffix.
Curves: target v (the Monte-Carlo countdown, black), predicted V = Σ p_b v_b (raw, unsmoothed,
orange). Shaded: frames where the trajectory parquet's `source` is 1 (human in control; the
`intervention` column only marks takeover onsets). Dotted line: the reviewed endpoint T. The *targets* are straight lines by
construction; any drop or rise that is not on the target line comes from the model.

Episodes, one per kind, chosen with a fixed rule:
  autonomous success / failure / soft-truncated timeout / timeout at the cap: the episode whose
  MAE is the median of its kind; assisted success / assisted failure / teleop demo: same;
  typical: median MAE over all val episodes; worst: highest MAE; worst success: lowest mean V
  among successes; drop failure: median MAE among episodes whose VLM failure mode (a machine
  label, diagnostic only) is a dropped, slipped or released marker.
"""

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vl.config import load_config
from vl.data_figures import trajectory
from vl.dataset import crop_resize
from vl.frames import FrameReader


def median_pick(g: pd.DataFrame) -> str | None:
    if g.empty:
        return None
    return (g.mae - g.mae.median()).abs().idxmin()


def choose_episodes(pred: pd.DataFrame, man: pd.DataFrame) -> list[tuple[str, str]]:
    e = (pred.assign(err=(pred.V - pred.value).abs())
         .groupby("episode_uid").agg(repo=("repo", "first"), outcome=("outcome", "first"), T=("T", "first"),
                                     mae=("err", "mean"), V_mean=("V", "mean"))
         .join(man.set_index("episode_uid")[["source_type", "has_human_frames", "soft_truncate", "machine_failure_mode"]]))
    auto = (e.source_type == "eval") & ~e.has_human_frames
    kinds = [
        ("autonomous_success", median_pick(e[auto & (e.outcome == "success")])),
        ("autonomous_failure", median_pick(e[auto & (e.outcome == "failure")])),
        ("timeout_stall", median_pick(e[auto & (e.outcome == "timeout") & e.soft_truncate])),
        ("timeout_at_cap", median_pick(e[auto & (e.outcome == "timeout") & (e["T"] >= 799)])),
        ("assisted_success", median_pick(e[(e.source_type == "dagger") & e.has_human_frames & (e.outcome == "success")])),
        ("assisted_failure", median_pick(e[(e.source_type == "dagger") & e.has_human_frames & (e.outcome == "failure")])),
        ("teleop_demo", median_pick(e[e.source_type == "teleop"])),
        ("typical_error", median_pick(e)),
        ("worst_error", e.mae.idxmax()),
        ("worst_success", e[e.outcome == "success"].V_mean.idxmin()),
        # "fall at a drop": the VLM failure-mode label (diagnostic only) names a lost marker
        ("drop_failure", median_pick(e[e.machine_failure_mode.isin(
            ["dropped_marker_during_transport", "marker_slipped_from_gripper", "marker_released_at_holder"])])),
    ]
    out, seen = [], set()
    for k, uid in kinds:
        if uid is not None and uid not in seen:
            out.append((k, uid))
            seen.add(uid)
    return out


def intervals(flag: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] frame ranges where flag is 1."""
    d = np.diff(np.r_[0, flag.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def curve_figure(e: pd.DataFrame, human: list, title: str, size=(9.6, 3.2), dpi=100, probs: np.ndarray | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    fps = float(e.fps.iloc[0])
    t = e.frame_index.to_numpy() / fps
    if probs is not None:
        # Column = one frame's distribution over the 201 bins; each bin spans its center +- half a width.
        # Each column is scaled to its own maximum, so a confident and a spread-out prediction are
        # both visible; the colour bar is "relative probability within the frame".
        n = probs.shape[1]
        hw = 0.5 / (n - 1)
        rel = probs / probs.max(axis=1, keepdims=True)
        im = ax.imshow(rel.T, origin="lower", aspect="auto", cmap="Purples", vmin=0, vmax=1, interpolation="nearest",
                       extent=[t[0] - 0.5 / fps, t[-1] + 0.5 / fps, -1 - hw, 0 + hw], zorder=0)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label("p(bin) / max", fontsize=7); cb.ax.tick_params(labelsize=6)
    for a, b in human:
        ax.axvspan(a / fps, (b + 1) / fps, color="tab:blue", alpha=0.15, lw=0)
    ax.plot(t, e.value, color="black", lw=1.6, label="target v (MC countdown)")
    ax.plot(t, e.V, color="tab:orange", lw=1.3, label="predicted V (raw)" + (" = mean of the shaded distribution" if probs is not None else ""))
    ax.axvline(e["T"].iloc[0] / fps, color="gray", ls=":", lw=1)
    ax.set_ylim(-1.02, 0.02); ax.set_xlim(0, t[-1] + 0.2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("normalized value")
    ax.axhline(-0.5, color="gray", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if human:
        import matplotlib.patches as mp
        handles.append(mp.Patch(color="tab:blue", alpha=0.15)); labels.append("human in control")
    ax.legend(handles, labels, fontsize=7, loc="lower right")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig, ax


def render(fig) -> Image.Image:
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def episode_outputs(kind, k, e, idx_row, man_row, human, reader, cfg, fig_dir, vid_dir, n_strip=8, S=224,
                    probs=None, video=True):
    mae = float((e.V - e.value).abs().mean())
    name = f"{k:02d}_{kind}_{e.repo.iloc[0][15:]}_ep{int(e.episode_index.iloc[0]):04d}" + ("_dist" if probs is not None else "")
    title = (f"{kind}: {e.repo.iloc[0][15:]} ep{int(e.episode_index.iloc[0])} | {e.outcome.iloc[0]}, "
             f"T = {e['T'].iloc[0] / e.fps.iloc[0]:.1f} s | episode MAE {mae:.3f}"
             + (f" | VLM: {man_row.machine_failure_mode}" if isinstance(man_row.machine_failure_mode, str) else ""))
    fig, ax = curve_figure(e, human, title, probs=probs)
    curve = render(fig)
    # Map time -> curve pixel column, for the strip ticks and the video cursor.
    x0, x1 = ax.get_xlim()
    bbox = ax.get_window_extent()
    tx = lambda t: int(bbox.x0 + (t - x0) / (x1 - x0) * bbox.width)
    import matplotlib.pyplot as plt
    plt.close(fig)

    frames = e.frame_index.to_numpy()

    def views(t):
        out = []
        for cam in cfg["cameras"]:
            img = reader.read(idx_row[f"path_{cam}"], idx_row[f"from_ts_{cam}"], int(t), float(idx_row.fps))
            out.append(Image.fromarray(crop_resize(img, cfg["crops"][cam], S)))
        return out

    # Filmstrip: n_strip evenly spaced frames, wrist above side, value under each.
    ts = np.linspace(0, frames[-1], n_strip).round().astype(int)
    tw = curve.width // n_strip
    sheet = Image.new("RGB", (curve.width, 2 * tw + 14 + curve.height), "white")
    d = ImageDraw.Draw(sheet)
    for j, t in enumerate(ts):
        w, s = views(t)
        sheet.paste(w.resize((tw, tw)), (j * tw, 0))
        sheet.paste(s.resize((tw, tw)), (j * tw, tw))
        row = e[e.frame_index == t].iloc[0]
        d.text((j * tw + 3, 2 * tw + 1), f"{t / row.fps:.1f}s V={row.V:+.2f} v={row.value:+.2f}", fill="black")
    sheet.paste(curve, (0, 2 * tw + 14))
    for t in ts:   # tick marks on the curve where the strip frames are
        x = tx(t / e.fps.iloc[0])
        d.line([(x, 2 * tw + 14), (x, 2 * tw + 22)], fill="red", width=2)
    sheet.save(fig_dir / f"{name}.png")
    if not video:
        return name, mae

    # Video: both views + the curve with a moving cursor.
    W = 2 * S
    cw = W * 2
    curve_v = curve.resize((cw, int(curve.height * cw / curve.width)))
    scale = cw / curve.width
    H = S + curve_v.height + 20
    with av.open(str(vid_dir / f"{name}.mp4"), "w") as c:
        st = c.add_stream("libx264", rate=int(e.fps.iloc[0]))
        st.width, st.height, st.pix_fmt = cw, H + (H % 2), "yuv420p"
        for _, row in e.iterrows():
            w, s = views(row.frame_index)
            fr = Image.new("RGB", (st.width, st.height), "white")
            fr.paste(w, ((cw - W) // 2, 0)); fr.paste(s, ((cw - W) // 2 + S, 0))
            fr.paste(curve_v, (0, S + 20))
            dd = ImageDraw.Draw(fr)
            x = int(tx(row.frame_index / row.fps) * scale)
            dd.line([(x, S + 20), (x, S + 20 + curve_v.height)], fill="red", width=2)
            hum = any(a <= row.frame_index <= b for a, b in human)
            dd.text((4, S + 4), f"t={row.frame_index / row.fps:5.2f}s  V={row.V:+.3f}  target v={row.value:+.3f}"
                                f"{'  HUMAN' if hum else ''}", fill="black")
            c.mux(st.encode(av.VideoFrame.from_ndarray(np.asarray(fr), format="rgb24")))
        c.mux(st.encode())
    return name, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--dist", action="store_true", help="shade the full predicted distribution behind the curves")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    tc = load_config(args.train_config)
    cfg = load_config(tc["data_config"])
    man = pd.read_parquet(load_config(cfg["dataset_config"])["paths"]["manifest"])
    run = Path(tc["paths"]["runs"]) / args.run
    pred = pd.read_parquet(run / "val_predictions.parquet")
    idx = pd.read_parquet(cfg["paths"]["frame_index"])
    pred = pred.merge(idx[["episode_uid", "frame_index", "fps"]], on=["episode_uid", "frame_index"])
    fig_dir = Path("reports/figures/stage4") / args.run
    vid_dir = Path("data/review_videos/stage4") / args.run
    fig_dir.mkdir(parents=True, exist_ok=True); vid_dir.mkdir(parents=True, exist_ok=True)
    reader, vroot = FrameReader(), Path(cfg["paths"]["video_root"])
    dist = None
    if args.dist:
        f = next(p for p in [run / "val_probs_plotted.npz", run / "val_probs.npz"] if p.exists())
        z = np.load(f)
        dist = {(u, int(t)): i for i, (u, t) in enumerate(zip(z["episode_uid"], z["frame_index"]))}
        probs_all, V_dist = z["probs"].astype(np.float32), z["V"]
    rows = []
    for k, (kind, uid) in enumerate(choose_episodes(pred, man), 1):
        e = pred[pred.episode_uid == uid].sort_values("frame_index")
        ir = idx[idx.episode_uid == uid].iloc[0]
        mr = man.set_index("episode_uid").loc[uid]
        # `source` == 1 marks human-controlled frames; `intervention` only marks takeover onsets.
        flag = trajectory(vroot, ir.repo, int(ir.episode_index), "source").ravel()[: int(ir["T"]) + 1] == 1
        human = intervals(flag)
        probs = None
        if dist is not None:
            rows_ = [dist[(uid, int(t))] for t in e.frame_index]
            probs = probs_all[rows_]
            # The distributions were recomputed (possibly another GPU / AMP dtype): check they match V.
            assert np.abs(V_dist[rows_] - e.V.to_numpy()).max() < 0.03, "recomputed distributions disagree with V"
        name, mae = episode_outputs(kind, k, e, ir, mr, human, reader, cfg, fig_dir, vid_dir, probs=probs,
                                    video=not args.no_video)
        rows.append({"kind": kind, "file": name, "outcome": e.outcome.iloc[0], "T_s": round(ir["T"] / ir.fps, 1),
                     "MAE": round(mae, 3), "human_frames": int(flag.sum())})
        print("->", name, flush=True)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
