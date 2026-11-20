"""Value / advantage overlay videos for the task-64 chain labels vs moka [2026-11-19; Karen: "can i observe the values
throughout the videos ... make sure it is the same quality of values for the task 64. generate a few videos that are
representative of the dataset"].

Per episode, one MP4 (20 fps = real time): agentview | wrist on top; under it, over the whole episode with a
playhead, (1) the out-of-fold ResNet V the labels used and the true MC target (value = -steps to go / 1040, -1 for a
failure: frame_targets.parquet), (2) the advantage A and each grid frame's label (green = positive, red = negative).
Values / labels live on the every-4th-step grid; they are held between grid frames.

Tables (both: 50 base rollouts, chain V fit on that table only, vb40 labels; the matched-budget pair):
  t64   tables/t64_agr_t64b50cv_r0, labels agr_t64b50cv_cv40_r0 (task-64 runs t64b50cv / t64d50 trained on these)
  moka  tables/moka_agr_b50_r0,     labels agr_b50cv_cv40_r0    (the moka 50-episode chain)
Episodes [ours]: per table, the successes at the 25th / 75th percentile of length, and failures: task 64 one per
base-failure mode present (reports/2026-11-17-ag-rounds/t64/t64.json, scripts/libero_week9_t64.py), moka 2 at random
(seed 0). Also writes quality.json: per table, corr(V, MC), MAE, AUC of V for the episode's outcome (all grid frames,
and at the first frame), P(positive) on success / failure episodes.

Usage: .venv/bin/python scripts/libero_value_videos.py --out reports/2026-11-19-t64-value-videos
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

TB = Path("/iris/u/ankile/karenvo/libero/tables")
REPO = Path(__file__).resolve().parents[1]
TABLES = {"t64": (TB / "t64_agr_t64b50cv_r0", "agr_t64b50cv_cv40_r0"),
          "moka": (TB / "moka_agr_b50_r0", "agr_b50cv_cv40_r0")}
HZ, W, PLOT_H = 20, 896, 360
BLUE, INK, GREEN, RED, GREY = "#3b6fd8", "#222222", "#3a9a5b", "#d0473a", "#8a8f98"


def auc(score: np.ndarray, y: np.ndarray) -> float:
    r = pd.Series(score).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(name: str):
    t, lab = TABLES[name]
    eps = pd.read_csv(t / "episodes.csv")
    lb = pd.read_parquet(t / "labels" / lab / "labels.parquet")
    ft = pd.read_parquet(t / "frame_targets.parquet", columns=["episode_index", "frame_index", "value"])
    d = lb.merge(ft, on=["episode_index", "frame_index"], how="left")
    assert d.value.notna().all(), name
    eps = eps[eps.episode_index.isin(lb.episode_index.unique())].copy()
    d = d.merge(eps[["episode_index", "success"]], on="episode_index")
    return eps, d, ft


def quality(d: pd.DataFrame) -> dict:
    y = d.success.values.astype(bool)
    first = d.sort_values("frame_index").groupby("episode_index").head(1)
    return {"episodes": int(d.episode_index.nunique()), "success_rate": round(float(first.success.mean()), 3),
            "corr_V_mc": round(float(np.corrcoef(d.V, d.value)[0, 1]), 3),
            "mae_V_mc": round(float((d.V - d.value).abs().mean()), 3),
            "auc_V_outcome_all_frames": round(auc(d.V.values, y), 3),
            "auc_V_outcome_first_frame": round(auc(first.V.values, first.success.values.astype(bool)), 3),
            "p_pos_success_eps": round(float(d[y].positive.mean()), 3),
            "p_pos_failure_eps": round(float(d[~y].positive.mean()), 3),
            "auc_A_outcome": round(auc(d.A.values, y), 3)}


def pick(name: str, eps: pd.DataFrame) -> list[tuple[int, str]]:
    s = eps[eps.success == 1].sort_values("policy_steps")
    out = [(int(s.iloc[int(q * (len(s) - 1))].episode_index), f"success, {q:.0%} length pct") for q in (0.25, 0.75)]
    f = eps[eps.success == 0]
    if name == "t64":
        modes = {(e["set"], e["ep"]): e["mode"] for e in
                 json.load(open(REPO / "reports/2026-11-17-ag-rounds/t64/t64.json"))["episodes"]}
        f = f.assign(mode=[modes.get((r.run.split("_")[2], int(r.source_episode)), "?") for r in f.itertuples()])
        for m, g in f.groupby("mode"):
            out.append((int(g.episode_index.iloc[0]), f"failure: {m}"))
    else:
        for e in f.sample(2, random_state=0).episode_index:
            out.append((int(e), "failure"))
    return out


def plot_panel(de: pd.DataFrame, T: int, title: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W / 100, PLOT_H / 100), dpi=100, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    t = de.frame_index.values / HZ
    a1.plot(t, de.value.values, color=INK, lw=1.4, ls=(0, (4, 2)), label="true MC target")
    a1.plot(t, de.V.values, color=BLUE, lw=2, label="ResNet V (out-of-fold, used for labels)")
    a1.set_ylim(-1.05, 0.05); a1.set_ylabel("value (0 = done)", fontsize=8)
    a1.legend(fontsize=7, loc="lower right", frameon=False); a1.set_title(title, fontsize=9, loc="left")
    a2.plot(t, de.A.values, color=GREY, lw=1.2)
    pos = de.positive.values.astype(bool)
    a2.scatter(t[pos], de.A.values[pos], s=7, color=GREEN, label="positive", zorder=3)
    a2.scatter(t[~pos], de.A.values[~pos], s=7, color=RED, label="negative", zorder=3)
    a2.axhline(0, color=GREY, lw=0.6); a2.set_ylabel("advantage A", fontsize=8); a2.set_xlabel("time (s)", fontsize=8)
    a2.legend(fontsize=7, loc="lower right", frameon=False, ncol=2)
    for a in (a1, a2):
        a.set_xlim(0, T / HZ); a.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0 = a1.transData.transform((0, 0))[0]; x1 = a1.transData.transform((T / HZ, 0))[0]
    y_top = img.shape[0] - a1.get_window_extent().y1; y_bot = img.shape[0] - a2.get_window_extent().y0
    plt.close(fig)
    return img, x0, x1, int(y_top), int(y_bot)


def video(name: str, e: int, why: str, eps: pd.DataFrame, d: pd.DataFrame, out: Path) -> dict:
    r = eps[eps.episode_index == e].iloc[0]
    z = np.load(r.raw)
    im, wr = z["image"], z["wrist_image"]
    T = len(im)
    de = d[d.episode_index == e].sort_values("frame_index")
    title = f"{name}  ep {e}  {why}  ({'success' if r.success else 'failure'}, {T} steps)"
    panel, x0, x1, yt, yb = plot_panel(de, T, title)
    font = ImageFont.load_default()
    path = out / f"{name}_ep{e}_{'success' if r.success else 'failure'}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{W}x{448 + panel.shape[0]}", "-r", str(HZ), "-i", "-", "-pix_fmt", "yuv420p",
                           "-vcodec", "libx264", "-crf", "23", str(path)], stdin=subprocess.PIPE)
    gi = np.searchsorted(de.frame_index.values, np.arange(T), side="right") - 1
    for k in range(T):
        top = np.concatenate([im[k], wr[k]], axis=1)  # 224 x 448
        top = np.asarray(Image.fromarray(top).resize((W, 448), Image.NEAREST))
        p = panel.copy()
        x = int(round(x0 + (x1 - x0) * k / T))
        p[yt:yb, max(x - 1, 0):x + 1] = (230, 120, 40)
        fr = Image.fromarray(np.concatenate([top, p], axis=0))
        g = de.iloc[max(gi[k], 0)]
        ImageDraw.Draw(fr).text((8, 8), f"t {k / HZ:5.1f}s  V {g.V:+.2f}  MC {g.value:+.2f}  A {g.A:+.3f}  "
                                f"{'POS' if g.positive else 'neg'}", fill=(255, 255, 255), font=font,
                                stroke_width=2, stroke_fill=(0, 0, 0))
        ff.stdin.write(np.asarray(fr).tobytes())
    ff.stdin.close(); assert ff.wait() == 0, path
    return {"table": name, "episode": e, "why": why, "success": int(r.success), "steps": T, "video": path.name,
            "V_first": round(float(de.V.iloc[0]), 3), "mc_first": round(float(de.value.iloc[0]), 3),
            "p_pos": round(float(de.positive.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"quality": {}, "videos": []}
    for name in TABLES:
        eps, d, _ = load(name)
        res["quality"][name] = quality(d)
        for e, why in pick(name, eps):
            res["videos"].append(video(name, e, why, eps, d, out))
            print(res["videos"][-1])
    json.dump(res, open(out / "videos.json", "w"), indent=1)
    print(json.dumps(res["quality"], indent=1))


if __name__ == "__main__":
    main()
