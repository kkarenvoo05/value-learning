"""Stage 5.5 B1: side-by-side history series [Lars]. At which gap does a stall look like "no change"
while a success still visibly progresses?

Usage:
    .venv/bin/python -m vl.history_series            # -> reports/figures/stage5_5/history/

Sheets (`series_<kind>_<repo>_ep<k>.jpg`): one per example episode. Rows = step size Δ in GAPS_S,
columns = t − 4Δ, …, t − Δ, t (current frame, red outline), both cameras side by side. Each cell is
the model's 224 px input (FrameDataset, no augmentation). Under each past frame: |Δpix| = mean
absolute uint8 difference to the current frame on that camera. A past frame before the episode
start is drawn as a grey "before start" cell (B2 will mask it, not fall back to frame 0).

Kinds [ours, the hold-phase rule of `vl.history_probe`: eval repos r00–r04, > 5 s in, > 2 s before T]:
  stall         timeout with machine label `timeout_holding_marker`, current frame = the first frame
                where the gripper has been still >= STALL_S seconds (`vl.history_plots.stall_seconds`)
  success_hold  success, current frame = T − 2.5 s (holding, just before seating)
  failure       failure (not timeout), current frame = T
  early         success, current frame = 2 s (most of the past does not exist yet)
Machine (VLM) failure modes are diagnostics; they only pick examples here. Train split only.

Curve (`pixel_change_vs_gap.png` + `.csv`) [ours]: mean |Δpix| between frame t and t − Δ for Δ in
CURVE_GAPS_S, per camera, over up to CURVE_EPS episodes × CURVE_FRAMES frames per kind (stall:
frames with the gripper still >= STALL_S, so gaps <= STALL_S are still *by selection*; stall_any:
any hold-phase frame of the same stall episodes, the non-circular comparison; success_hold /
failure: hold-phase frames), 95% bootstrap
over episodes. A pixel proxy for "visible change", not what the model sees as change.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from vl.config import load_config
from vl.dataset import FrameDataset
from vl.history_plots import GRID, INK, MUTED, stall_seconds, style

GAPS_S = [0.5, 1, 2, 4, 8]
N_BACK = 4
CURVE_GAPS_S = [0.5, 1, 2, 4, 8, 16]
STALL_S = 10.0        # [ours] the gripper still for 10 s: 4 × 2 s back is all inside the stall
N_SHEETS = {"stall": 4, "success_hold": 4, "failure": 3, "early": 2}
CURVE_EPS, CURVE_FRAMES = 40, 3
KIND = {"stall": ("stall, gripper still >= 10 s", "#6250d6", "x"),
        "stall_any": ("stall episode, any hold frame", "#d6409f", "^"),
        "success_hold": ("success (hold phase)", "#1baf7a", "o"),
        "failure": ("failure (hold phase)", "#eb6834", "s")}
GAP = 4
FONT = ImageFont.truetype(matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf", 13)   # has Δ
SHORT = lambda repo: repo.replace("real-marker-d2-", "").split("-")[0]


def load_index(cfg: dict) -> pd.DataFrame:
    idx = pd.read_parquet(cfg["paths"]["frame_index"])
    idx = idx[idx.split == "train"].reset_index(drop=True)
    man = pd.read_parquet("data/manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    idx = idx.merge(man, on="episode_uid", how="left", validate="m:1")
    idx = idx.merge(stall_seconds(), on=["repo", "episode_index", "frame_index"], how="left", validate="1:1")
    assert idx.stall_s.notna().all()
    idx["t_s"] = idx.frame_index / idx.fps
    idx["eval"] = idx.repo.str.contains("-eval")
    idx["hold"] = idx["eval"] & (idx.t_s > 5) & (idx["T"] / idx.fps - idx.t_s > 2)
    return idx


def candidates(idx: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Every eligible current frame of this kind (rows of idx)."""
    if kind == "stall":
        return idx[idx.hold & (idx.outcome == "timeout") & (idx.machine_failure_mode == "timeout_holding_marker")
                   & (idx.stall_s >= STALL_S)]
    if kind == "stall_any":       # curve only: same episodes, no stillness condition (not circular)
        return idx[idx.hold & (idx.outcome == "timeout") & (idx.machine_failure_mode == "timeout_holding_marker")]
    if kind == "success_hold":
        return idx[idx.hold & (idx.outcome == "success")]
    if kind == "failure":
        return idx[idx["eval"] & (idx.outcome == "failure") & (idx.t_s > 5)]
    return idx[idx["eval"] & (idx.outcome == "success")]


def sheet_frame(cands: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One current frame per episode: the sheet rule of the module docstring."""
    g = cands.sort_values("frame_index").groupby("episode_uid")
    if kind == "stall":
        return g.head(1)
    target = {"success_hold": lambda e: e["T"] - round(2.5 * e.fps), "failure": lambda e: e["T"],
              "early": lambda e: round(2 * e.fps)}[kind]
    rows = [ep.iloc[(ep.frame_index - target(ep.iloc[0])).abs().argmin()] for _, ep in g]
    return pd.DataFrame(rows)


def spread_pick(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """n rows, cycling through repos so the examples come from different rounds."""
    df = df.sample(frac=1, random_state=seed)
    df = df.assign(_k=df.groupby("repo").cumcount()).sort_values(["_k", "repo"])
    return df.head(n).drop(columns="_k")


def past_row(idx_ep: pd.Series, t: int, gap_s: float, k: int, fps: int):
    """Index row of the frame k·gap back, or None if it is before the episode start."""
    f = t - int(round(k * gap_s * fps))
    return None if f < 0 else idx_ep.get(f)


_DS = None


def _init(idx, cfg):
    global _DS
    _DS = FrameDataset(idx, cfg)


def _images(row: int) -> dict[str, np.ndarray]:
    return {c: x.permute(1, 2, 0).numpy() for c, x in _DS[row]["images"].items()}


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def render_sheet(idx, ep_rows: pd.Series, e: pd.Series, kind: str, pool, cams, S, out: Path):
    t, fps = int(e.frame_index), int(e.fps)
    need = {(gi, k): past_row(ep_rows, t, g, k, fps) for gi, g in enumerate(GAPS_S) for k in range(1, N_BACK + 1)}
    rows = sorted({int(e.name)} | {r for r in need.values() if r is not None})
    imgs = dict(zip(rows, pool.map(_images, rows)))
    cur = imgs[int(e.name)]
    cw, ch, lab = S + GAP, S + 18, 70
    block = (N_BACK + 1) * cw
    W, H = lab + len(cams) * (block + 3 * GAP), 44 + len(GAPS_S) * ch
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.font = FONT
    d.text((4, 4), f"{SHORT(e.repo)} ep{e.episode_index}  {kind}  outcome {e.outcome}  "
           f"({e.machine_failure_mode})  current t = {t / fps:.1f} s of {e['T'] / fps:.1f} s, "
           f"gripper still {e.stall_s:.1f} s   |  columns t-4Δ ... t-Δ, t (red)", fill="black")
    for ci, cam in enumerate(cams):
        x0 = lab + ci * (block + 3 * GAP)
        d.text((x0, 22), cam, fill="black")
        for gi, g in enumerate(GAPS_S):
            y = 40 + gi * ch
            d.text((4, y + S // 2), f"Δ = {g:g} s", fill="black")
            for col in range(N_BACK + 1):
                k = N_BACK - col
                x = x0 + col * cw
                r = int(e.name) if k == 0 else need[(gi, k)]
                if r is None:
                    d.rectangle((x, y, x + S - 1, y + S - 1), fill=(200, 200, 200))
                    d.text((x + 8, y + S // 2), f"-{k * g:g} s: before start", fill="black")
                    continue
                sheet.paste(Image.fromarray(imgs[r][cam]), (x, y))
                if k == 0:
                    d.rectangle((x, y, x + S - 1, y + S - 1), outline="red", width=3)
                    d.text((x + 2, y + S + 2), "t (now)", fill="black")
                else:
                    d.text((x + 2, y + S + 2), f"-{k * g:g} s  |Δpix| {diff(imgs[r][cam], cur[cam]):.1f}",
                           fill="black")
    p = out / f"series_{kind}_{SHORT(e.repo)}_ep{e.episode_index}.jpg"
    sheet.save(p, quality=90)
    print("->", p, flush=True)


def _pair(args):
    r0, r1 = args
    a, b = _images(r0), _images(r1)
    return {c: diff(a[c], b[c]) for c in a}


def curve(idx, pool, cams, out: Path, seed: int = 0):
    by_ep = {u: g.set_index("frame_index")["row"] for u, g in idx.assign(row=idx.index).groupby("episode_uid")}
    jobs, meta = [], []
    for kind in KIND:
        c = candidates(idx, kind)
        eps = c.episode_uid.drop_duplicates().sample(min(CURVE_EPS, c.episode_uid.nunique()), random_state=seed)
        for u in eps:
            fr = c[c.episode_uid == u]
            for _, e in fr.sample(min(CURVE_FRAMES, len(fr)), random_state=seed).iterrows():
                for g in CURVE_GAPS_S:
                    r = past_row(by_ep[u], int(e.frame_index), g, 1, int(e.fps))
                    if r is not None:
                        jobs.append((int(e.name), int(r)))
                        meta.append((kind, u, g))
    print(f"curve: {len(jobs)} frame pairs", flush=True)
    res = pool.map(_pair, jobs, chunksize=8)
    df = pd.DataFrame([dict(kind=k, episode_uid=u, gap_s=g, **r) for (k, u, g), r in zip(meta, res)])
    ep = df.groupby(["kind", "episode_uid", "gap_s"])[cams].mean().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for (kind, g), h in ep.groupby(["kind", "gap_s"]):
        for cam in cams:
            v = h[cam].to_numpy()
            bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
            rows.append(dict(kind=kind, gap_s=g, camera=cam, n_eps=len(v), mean=v.mean(),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "pixel_change_vs_gap.csv", index=False, float_format="%.3f")
    fig, axes = plt.subplots(1, len(cams), figsize=(6.2 * len(cams), 4.8), sharey=True)
    for ax, cam in zip(np.atleast_1d(axes), cams):
        for kind, (name, col, mk) in KIND.items():
            h = tab[(tab.kind == kind) & (tab.camera == cam)].sort_values("gap_s")
            ax.plot(h.gap_s, h["mean"], color=col, lw=2, marker=mk, ms=8, label=f"{name} (n={h.n_eps.max()})")
            ax.fill_between(h.gap_s, h.lo, h.hi, color=col, alpha=0.12, lw=0)
            last = h.iloc[-1]
            if cam == cams[0]:          # side_1 lines end too close together for labels; legend below
                ax.annotate(name.split(" (")[0], (last.gap_s, last["mean"]), xytext=(6, 0), textcoords="offset points",
                            color=INK, fontsize=9, va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(CURVE_GAPS_S, [f"{g:g}" for g in CURVE_GAPS_S])
        ax.set_xlabel("gap Δ between current and past frame (s)", color=INK)
        ax.set_title(cam, color=INK, loc="left")
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("mean |Δpixel| on the 224 px input (0–255)", color=INK)
    hs, ls = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(hs, ls, frameon=False, loc="lower center", ncol=len(KIND), fontsize=9)
    fig.suptitle("How much the model's input changes over a gap: stalls vs hold-phase successes vs failures",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = out / "pixel_change_vs_gap.png"
    fig.savefig(p, dpi=150)
    print("->", p, "and .csv", flush=True)
    print(tab.pivot_table(index=["camera", "kind"], columns="gap_s", values="mean").round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="reports/figures/stage5_5/history")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-curve", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    idx = load_index(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cams, S = cfg["cameras"], cfg["image_size"]
    with Pool(args.workers, initializer=_init, initargs=(idx, cfg)) as pool:
        for kind, n in N_SHEETS.items():
            picks = spread_pick(sheet_frame(candidates(idx, kind), kind), n)
            print(f"{kind}: {candidates(idx, kind).episode_uid.nunique()} eligible episodes, sheets for "
                  f"{[f'{SHORT(r)}/{e}' for r, e in zip(picks.repo, picks.episode_index)]}", flush=True)
            for _, e in picks.iterrows():
                ep_rows = pd.Series(idx.index[idx.episode_uid == e.episode_uid],
                                    index=idx.frame_index[idx.episode_uid == e.episode_uid].to_numpy())
                render_sheet(idx, ep_rows, e, kind, pool, cams, S, out)
        if not args.skip_curve:
            curve(idx, pool, cams, out)


if __name__ == "__main__":
    main()
