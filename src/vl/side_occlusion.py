"""Stage 4 occlusion question: does V react when the robot arm covers the `side_1` view?

Usage:
    .venv/bin/python -m vl.side_occlusion --run lowlr_8k --seeds lowlr_8k lowlr_8k_s1 lowlr_8k_s2
        -> data/analysis/side_1_val_1fps.npz            (cached crops, val frames at 1 fps)
           reports/side_occlusion_<run>.md               (tables)
           reports/figures/stage4/side_occlusion_sheet.jpg (proxy check: frames + masks)

No reviewed occlusion labels exist, so occlusion is a proxy [ours]:
- background: per repo, the pixel-wise median of that repo's val `side_1` crops at 1 fps (the
  arm moves, so each pixel is empty table/holder most of the time). Image statistics only, no
  labels.
- occluded pixel: max over RGB of |frame - background| > DIFF_THRESHOLD, on the training crop
  resized to 112 x 112, then a 3 x 3 majority filter against noise.
- occlusion = fraction of occluded pixels in the crop. Includes the gripper, the arm and anything
  the arm carries or moves (the marker), so it is "how much of the side view changed", which is
  what the model sees anyway.
The check is whether V shifts or jumps with occlusion beyond what task phase explains: matched
(repo, outcome, 2 s elapsed-time) strata, and changes between consecutive 1 s samples.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vl.analysis import annotate, load_trajectories, md
from vl.config import load_config
from vl.dataset import crop_resize
from vl.frames import FrameReader

FPS = 15
SIZE = 112
DIFF_THRESHOLD = 40


def extract(idx: pd.DataFrame, box, cache: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """side_1 crops (SIZE x SIZE) of every 15th valid val frame, decoded exactly by timestamp."""
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        rows = pd.DataFrame({k: z[k] for k in ["repo", "episode_index", "frame_index"]})
        return rows, z["img"]
    rows = idx[(idx.split == "val") & (idx.frame_index % FPS == 0)].reset_index(drop=True)
    reader, imgs = FrameReader(), []
    for i, r in enumerate(rows.itertuples()):
        imgs.append(crop_resize(reader.read(r.path_side_1, r.from_ts_side_1, int(r.frame_index), r.fps), box, SIZE))
        if i % 500 == 0:
            print(f"  decoded {i}/{len(rows)}", flush=True)
    img = np.stack(imgs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, img=img, repo=rows.repo.to_numpy().astype(str), episode_index=rows.episode_index.to_numpy(),
             frame_index=rows.frame_index.to_numpy())
    return rows[["repo", "episode_index", "frame_index"]], img


def masks(rows: pd.DataFrame, img: np.ndarray) -> tuple[np.ndarray, dict]:
    bg = {repo: np.median(img[ix], axis=0) for repo, ix in rows.groupby("repo").indices.items()}
    m = np.zeros(img.shape[:3], dtype=bool)
    for repo, ix in rows.groupby("repo").indices.items():
        d = np.abs(img[ix].astype(np.int16) - bg[repo].astype(np.int16)).max(axis=-1) > DIFF_THRESHOLD
        # 3 x 3 majority filter: a pixel counts if at least 5 of its 9 neighbours (incl. itself) do.
        p = np.pad(d, ((0, 0), (1, 1), (1, 1)))
        s = sum(p[:, i:i + SIZE, j:j + SIZE].astype(np.int8) for i in range(3) for j in range(3))
        m[ix] = s >= 5
    return m, bg


def sheet(rows: pd.DataFrame, img: np.ndarray, m: np.ndarray, occ: np.ndarray, out: Path, n: int = 8):
    """Proxy check: random frames from each occlusion quartile (seed 0), image above its mask."""
    q = pd.qcut(occ, 4, labels=False)
    rng = np.random.default_rng(0)
    tile, label_w = SIZE * 2, 80
    canvas = Image.new("RGB", (label_w + n * tile, 4 * 2 * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for k in range(4):
        pick = rng.choice(np.flatnonzero(q == k), size=n, replace=False)
        for j, i in enumerate(pick):
            over = img[i].copy()
            over[m[i]] = (0.5 * over[m[i]] + [127, 0, 0]).astype(np.uint8)
            canvas.paste(Image.fromarray(img[i]).resize((tile, tile)), (label_w + j * tile, 2 * k * tile))
            canvas.paste(Image.fromarray(over).resize((tile, tile)), (label_w + j * tile, (2 * k + 1) * tile))
            draw.text((label_w + j * tile + 3, 2 * k * tile + 3), f"{occ[i]:.2f}", fill="yellow")
        draw.text((4, 2 * k * tile + tile - 6), f"quartile {k + 1}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)
    return out


def section(p: pd.DataFrame) -> str:
    """p: annotated val predictions at the 1 fps samples, with column `occ`."""
    out = []
    p = p.copy()
    p["occ_q"] = pd.qcut(p.occ, 4, labels=["q1 (least)", "q2", "q3", "q4 (most)"])
    t = p.groupby("occ_q", observed=True).agg(frames=("err", "size"), occlusion=("occ", "mean"), t_s=("t_s", "mean"),
                                               V=("V", "mean"), target=("value", "mean"), MAE=("err", "mean"),
                                               bias=("bias", "mean"))
    out.append("Raw, by occlusion quartile (confounded with task phase):\n\n" + md(t))
    # Matched: within (repo, outcome, 2 s elapsed-time) strata, top occlusion quartile minus the rest.
    p["tbin"] = (p.t_s // 2).astype(int)
    p["high"] = p.occ_q == "q4 (most)"
    rows = []
    for kind, sub in [("all", p), ("autonomous", p[p.kind == "autonomous"]), ("robot-control frames", p[p.source == 0])]:
        g = sub.groupby(["repo", "outcome", "tbin", "high"])[["V", "err", "bias", "value", "occ"]].mean().unstack("high")
        n = sub.groupby(["repo", "outcome", "tbin", "high"]).size().unstack("high")
        ok = n.notna().all(axis=1)
        w = n.loc[ok, True]
        d = {k: float(((g.loc[ok, (k, True)] - g.loc[ok, (k, False)]) * w).sum() / w.sum()) for k in ["V", "err", "bias", "value", "occ"]}
        rows.append({"frames": kind, "matched strata": int(ok.sum()), "high-occlusion frames": int(w.sum()),
                     "Δocclusion": d["occ"], "ΔV": d["V"], "Δtarget": d["value"], "ΔMAE": d["err"], "Δbias": d["bias"]})
    out.append("Matched (same repo, outcome and 2 s elapsed-time bin), top occlusion quartile minus the other "
               "three:\n\n" + md(pd.DataFrame(rows).set_index("frames")))
    # Consecutive 1 s samples: does V jump when occlusion jumps? Robot control, skipping the last
    # 2 s before T (the seating jump) and pairs straddling a gap.
    p = p.sort_values(["episode_uid", "frame_index"])
    prev = p.groupby("episode_uid")[["V", "occ", "value", "frame_index"]].shift()
    st = p.assign(dV=p.V - prev.V, docc=p.occ - prev.occ, gap=p.frame_index - prev.frame_index).dropna(subset=["dV"])
    st = st[(st.gap == FPS) & (st.source == 0) & (st.to_T_s >= 2)]
    r = float(np.corrcoef(st.dV, st.docc)[0, 1])
    jump = st.docc.abs() > st.docc.abs().quantile(0.9)
    rows = []
    for thr in [0.1, 0.2]:
        drop = st.dV < -thr
        rows.append({"|ΔV| >": thr, "drops": int(drop.sum()), "P(drop | occlusion jump)": float(drop[jump].mean()),
                     "P(drop | no jump)": float(drop[~jump].mean()),
                     "P(drop | occlusion rises)": float(drop[jump & (st.docc > 0)].mean()),
                     "P(drop | occlusion falls)": float(drop[jump & (st.docc < 0)].mean())})
    out.append(f"Consecutive 1 s samples ({len(st):,} pairs; robot control; > 2 s before T). corr(ΔV, Δocclusion) = "
               f"{r:+.3f}. Occlusion jump = top 10% of |Δocclusion| (> {st.docc.abs().quantile(0.9):.3f}):\n\n"
               + md(pd.DataFrame(rows).set_index("|ΔV| >")))
    return "\n\n".join(out), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--train-config", default="configs/train.yaml")
    args = ap.parse_args()
    tc = load_config(args.train_config)
    dcfg = load_config(tc["data_config"])
    dscfg = load_config(dcfg["dataset_config"])
    runs_dir = Path(tc["paths"]["runs"])
    idx = pd.read_parquet(dcfg["paths"]["frame_index"])
    rows, img = extract(idx, dcfg["crops"]["side_1"], Path("data/analysis/side_1_val_1fps.npz"))
    m, _ = masks(rows, img)
    occ = m.mean(axis=(1, 2))
    print("->", sheet(rows, img, m, occ, Path("reports/figures/stage4/side_occlusion_sheet.jpg")))
    o = rows.assign(occ=occ)
    rmed = idx[idx.split == "train"].groupby("repo").value.median().to_dict()
    man = pd.read_parquet(dscfg["paths"]["manifest"])
    sec = [f"# Side-view occlusion vs V: `{args.run}` (generated by `vl.side_occlusion`)\n",
           f"Validation `side_1` at 1 fps ({len(o):,} frames). Occlusion proxy: fraction of the {SIZE}x{SIZE} "
           f"training crop differing from the repo's median image by > {DIFF_THRESHOLD} (see module docstring). "
           f"Mean {occ.mean():.3f}, median {np.median(occ):.3f}, 90th pct {np.quantile(occ, 0.9):.3f}.\n"]
    corr = {}
    for run in [args.run] + [s for s in args.seeds if s != args.run]:
        pred = pd.read_parquet(runs_dir / run / "val_predictions.parquet")
        traj = load_trajectories(Path(dcfg["paths"]["video_root"]), pred, Path("data/analysis/val_trajectory.parquet"))
        p = annotate(pred, man, traj, rmed).merge(o, on=["repo", "episode_index", "frame_index"], how="inner")
        text, corr[run] = section(p)
        sec += [f"## `{run}`\n", text + "\n"]
    if len(corr) > 1:
        sec.append("corr(ΔV, Δocclusion) per seed: " + ", ".join(f"{k} {v:+.3f}" for k, v in corr.items()) + "\n")
    out = Path(tc["paths"].get("reports", "reports")) / f"side_occlusion_{args.run}.md"
    out.write_text("\n".join(sec))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
