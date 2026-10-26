"""Stage 5.5 B1 follow-up: how long do hold-phase successes sit still, compared with stalls? A history
gap only flags a stall if successes rarely stay still that long.

Usage:
    .venv/bin/python -m vl.success_pauses       # -> reports/success_pauses.md,
                                                #    reports/figures/stage5_5/history/long_pause_successes.jpg

Episodes [ours, the hold-phase rule of `vl.history_series`: eval repos, > 5 s in, > 2 s before T]:
success vs stall (timeout with machine label `timeout_holding_marker`; the machine label only picks
episodes). "Still" = gripper speed over the last 0.5 s below a threshold (`vl.history_plots.stall_seconds`
rule, 1 cm/s there); STILL_MPS_SWEEP repeats it at 0.5 / 1 / 2 cm/s. Sections:
  1. pauses    train split: longest still stretch per episode, and the share of episodes / hold
               frames still >= each gap in GAPS_S
  2. pixels    train split: |Δpix| between the end of each episode's longest still stretch (>= 4 s)
               and 4 s earlier, up to N_PIX episodes per kind (checks the proprio stillness is visible)
  3. values    val split: targets and Gate 0 baseline predictions (`g0_r0_s{0,1,2}`, V averaged over
               seeds) for successes grouped by their longest pause, against stalls
Figure: timeline strips of LONG_PAUSE_EXAMPLES (wrist over side, model inputs, no augmentation).
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vl.config import load_config
from vl.history_plots import stall_seconds
from vl import history_series as hs

GAPS_S = [0.5, 1, 2, 3, 4, 6, 8]
STILL_MPS_SWEEP = [0.005, 0.01, 0.02]   # [ours] the 1 cm/s rule of history_plots, halved and doubled
PIX_GAP_S = 4
N_PIX = 30
VALUE_RUNS = [f"g0_r0_s{s}" for s in range(3)]   # Gate 0 baseline
LONG_PAUSE_EXAMPLES = ["r03/25", "r02/35", "r04/108"]
KEY = ["repo", "episode_index", "frame_index"]


def still_seconds(still_mps: float) -> pd.DataFrame:
    """`vl.history_plots.stall_seconds` with the speed threshold as a parameter."""
    p = pd.read_parquet("data/proprio/frame_proprio.parquet").sort_values(KEY)
    g = p.groupby(["repo", "episode_index"])
    speed = np.sqrt(sum((p[a] - g[a].shift(7)) ** 2 for a in "xyz")) / (7 / 15)
    still = speed.fillna(np.inf) < still_mps
    key = [p.repo, p.episode_index]
    run = (~still).groupby(key).cumsum()
    p["still_s"] = still.groupby(key + [run]).cumsum() / 15
    return p[KEY + ["still_s"]]


def kind(outcome: pd.Series, mode: pd.Series) -> np.ndarray:
    return np.where(outcome == "success", "success", np.where(mode == "timeout_holding_marker", "stall", "other"))


def pauses(idx: pd.DataFrame) -> list[str]:
    h = idx[idx.hold].assign(kind=lambda d: kind(d.outcome, d.machine_failure_mode))
    h = h[h.kind != "other"][KEY + ["kind"]]
    lines = []
    for thr in STILL_MPS_SWEEP:
        d = h.merge(still_seconds(thr), on=KEY, validate="1:1")
        ep = d.groupby(["kind", "repo", "episode_index"]).still_s.max().rename("longest").reset_index()
        desc = ep.groupby("kind").longest.describe(percentiles=[.5, .75, .9, .95])
        desc = desc.rename(columns={"count": "episodes"}).drop(columns=["std", "min"]).round(1)
        desc["episodes"] = desc.episodes.astype(int)
        share = pd.DataFrame([{"gap_s": gap,
                               **{f"{k} episodes": f"{(ep[ep.kind == k].longest >= gap).mean():.0%}" for k in ["success", "stall"]},
                               **{f"{k} hold frames": f"{(d[d.kind == k].still_s >= gap).mean():.0%}" for k in ["success", "stall"]}}
                              for gap in GAPS_S])
        lines += [f"### Still = gripper < {thr * 100:.1f} cm/s", "",
                  "Longest still stretch per episode in the hold phase (s):", "",
                  desc.to_markdown(), "", "Share still >= gap (episodes: any stretch that long; "
                  "frames: still for >= gap at that frame):", "", share.to_markdown(index=False), ""]
    return lines


def _pix(pair):
    a, b = hs._images(pair[0]), hs._images(pair[1])
    return {c: hs.diff(a[c], b[c]) for c in a}


def pixels(idx: pd.DataFrame, pool) -> list[str]:
    h = idx[idx.hold].assign(kind=lambda d: kind(d.outcome, d.machine_failure_mode))
    ends = h[(h.kind != "other") & (h.stall_s >= PIX_GAP_S)].sort_values("stall_s").groupby(["repo", "episode_index"]).tail(1)
    pick = pd.concat([g.sample(min(N_PIX, len(g)), random_state=0) for _, g in ends.groupby("kind")])
    row = pd.Series(idx.index, index=pd.MultiIndex.from_frame(idx[KEY]))
    past = [row[(r.repo, r.episode_index, r.frame_index - PIX_GAP_S * r.fps)] for r in pick.itertuples()]
    res = pool.map(_pix, list(zip(pick.index, past)))
    pick = pick.assign(**{f"|Δpix| {c}": [r[c] for r in res] for c in res[0]},
                       s_before_T=pick["T"] / pick.fps - pick.t_s)
    cols = [c for c in pick if c.startswith("|Δpix|")] + ["stall_s", "s_before_T"]
    tab = pick.groupby("kind")[cols].describe(percentiles=[.5]).round(1)
    tab = tab.loc[:, (slice(None), ["count", "50%", "max"])]
    tab.columns = [f"{a} {b}" for a, b in tab.columns]
    return ["Per kind: median / max over episodes; stall_s = length of the still stretch at its end, "
            "s_before_T = where it ends:", "", tab.T.to_markdown(), ""]


def values() -> list[str]:
    p = [pd.read_parquet(f"data/runs/{r}/val_predictions.parquet") for r in VALUE_RUNS]
    assert all((x[KEY].values == p[0][KEY].values).all() for x in p)
    d = p[0].drop(columns=["V", "ce"]).assign(V=np.mean([x.V.values for x in p], 0))
    man = pd.read_parquet("data/manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    d = d.merge(man, on="episode_uid", validate="m:1").merge(stall_seconds(), on=KEY, how="left", validate="1:1")
    assert d.stall_s.notna().all()
    d["to_go"] = (d["T"] - d.frame_index) / 15
    d = d[d.repo.str.contains("-eval") & (d.frame_index / 15 > 5) & (d.to_go > 2)]
    d = d[kind(d.outcome, d.machine_failure_mode) != "other"]
    longest = d.groupby("episode_uid").stall_s.transform("max")
    d["group"] = np.select([d.outcome != "success", longest >= 8, longest < 2],
                           ["stall", "success, longest pause >= 8 s", "success, longest pause < 2 s"],
                           "success, longest pause 2-8 s")
    d["still >= 4 s now"] = d.stall_s >= 4
    d["err"] = d.V - d.value
    t = d.groupby(["group", "still >= 4 s now"]).agg(
        episodes=("episode_uid", "nunique"), frames=("V", "size"), to_go_s=("to_go", "mean"),
        target=("value", "mean"), V=("V", "mean"), bias=("err", "mean"), MAE=("err", lambda e: e.abs().mean()),
        V_below_half=("V", lambda v: (v < -0.5).mean()))
    return [f"Val split, hold phase, `{'`, `'.join(VALUE_RUNS)}` (V = mean over seeds); still = gripper < 1 cm/s. "
            "V_below_half = share of frames predicted in the failure range:", "", t.round(3).to_markdown(), ""]


def strips(idx: pd.DataFrame, out: Path) -> None:
    rows = []
    for spec in LONG_PAUSE_EXAMPLES:
        repo, ep = spec.split("/")
        e = idx[idx.repo.str.contains(f"{repo}-eval") & (idx.episode_index == int(ep))]
        fps, T, last = e.fps.iloc[0], e["T"].iloc[0], e.frame_index.max()
        ts = sorted({t for t in [5, 10, 15, 20, 25, 30, 35, T / fps - 2, T / fps] if t * fps <= last})
        cells = []
        for t in ts:
            r = e.index[np.argmin(np.abs(e.frame_index - t * fps))]
            im = hs._images(r)
            cell = Image.fromarray(np.concatenate([im["wrist_left"], im["side_1"]], 0)).resize((150, 300))
            ImageDraw.Draw(cell).rectangle([0, 0, 150, 16], fill="white")
            ImageDraw.Draw(cell).text((3, 2), f"{t:.1f} s{' = T' if abs(t - T / fps) < .05 else ''}  still "
                                      f"{e.loc[r, 'stall_s']:.0f} s", fill="black")
            cells.append(np.array(cell))
        row = np.concatenate(cells, 1)
        hdr = Image.new("RGB", (row.shape[1], 18), "white")
        ImageDraw.Draw(hdr).text((3, 3), f"{spec}  {e.outcome.iloc[0]}  T = {T / fps:.1f} s", fill="black")
        rows.append(np.concatenate([np.array(hdr), row], 0))
    w = max(r.shape[1] for r in rows)
    Image.fromarray(np.concatenate([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255)
                                    for r in rows], 0)).save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--report", default="reports/success_pauses.md")
    ap.add_argument("--fig", default="reports/figures/stage5_5/history/long_pause_successes.jpg")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    idx = hs.load_index(cfg)
    hs._init(idx, cfg)
    lines = ["# Success pauses vs stalls (Stage 5.5 B1 follow-up)", "",
             "Generated by `.venv/bin/python -m vl.success_pauses`; see the module docstring for definitions.", "",
             "## 1. Pause lengths (train split)", ""] + pauses(idx)
    with Pool(args.workers, initializer=hs._init, initargs=(idx, cfg)) as pool:
        lines += [f"## 2. Is the proprio stillness visible? |Δpix| over {PIX_GAP_S} s (train split)", ""] + pixels(idx, pool)
    lines += ["## 3. Targets and baseline predictions", ""] + values()
    Path(args.report).write_text("\n".join(lines))
    strips(idx, Path(args.fig))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
