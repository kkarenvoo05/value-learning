"""Stage 5.5 C4 follow-up: where does the wrist camera make not-success frames look better than they are?

C4 (`reports/camera_contrib.md`) found that on failures / timeouts, removing the wrist camera lowers val CE
and lowers V. This script localises that on the saved masking outputs (CPU only, no new model runs):

    .venv/bin/python scripts/wrist_optimism.py            -> reports/wrist_optimism.md
                                                          -> reports/figures/stage5_5/wrist_optimism/*.jpg

Per val frame and arm (mean over 3 seeds) [ours]:
  wrist_dV  = V_full - V_without_wrist   (> 0: the wrist raises V)
  wrist_dCE = CE_full - CE_without_wrist (> 0: the wrist makes the frame's loss worse = "wrist hurts")
"without wrist" = `mean` intervention (features -> val mean) for R0 / C1, `drop` (token masked out of the
readout) for C3, which uses no stand-in input and so checks that the effect is not an artefact of the mean.
Groupings reuse `vl.analysis.annotate` (stall = robot in control, >= 3 s in, 3 s mean speed < 15 mm/s) and
the manifest's machine_failure_mode (a VLM label: diagnostic, not truth).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vl.analysis import annotate, load_trajectories, md
from vl.config import load_config
from vl.dataset import FrameDataset

ARMS = {"R0": ("g0_r0", "mean"), "C1": ("c1_percam", "mean"), "C3": ("c3_attn", "drop")}
KEYS = ["episode_uid", "frame_index"]


def load_arm(runs: Path, arm: str, kind: str) -> pd.DataFrame:
    per = []
    for s in range(3):
        m = pd.read_parquet(runs / f"{arm}_s{s}" / "val_cam_mask.parquet")
        per.append(pd.DataFrame({"episode_uid": m.episode_uid, "frame_index": m.frame_index,
                                 "V": m.V_full, "V_nw": m[f"V_{kind}_wrist_left"], "V_ns": m[f"V_{kind}_side_1"],
                                 "ce": m.ce_full, "ce_nw": m[f"ce_{kind}_wrist_left"], "ce_ns": m[f"ce_{kind}_side_1"]}))
    d = pd.concat(per).groupby(KEYS, sort=False).mean().reset_index()
    d["wrist_dV"], d["wrist_dCE"] = d.V - d.V_nw, d.ce - d.ce_nw
    d["side_dV"], d["side_dCE"] = d.V - d.V_ns, d.ce - d.ce_ns
    return d


def summary(g) -> pd.DataFrame:
    t = pd.DataFrame({"episodes": g.episode_uid.nunique(), "frames": g.size(), "target": g.value.mean(),
                      "V full": g.V.mean(), "V no wrist": g.V_nw.mean(), "wrist ΔV": g.wrist_dV.mean(),
                      "wrist ΔCE (>0 hurts)": g.wrist_dCE.mean(), "side ΔV": g.side_dV.mean(),
                      "bias full": g.bias.mean(), "bias no wrist": g.bias_nw.mean()})
    return t


def tile(img: np.ndarray, lines: list[str], w: int = 224) -> Image.Image:
    im = Image.fromarray(img).resize((w, w))
    ImageDraw.Draw(im).multiline_text((3, 3), "\n".join(lines), fill=(255, 255, 0))
    return im


def sheet(ds: FrameDataset, rows: pd.DataFrame, idx_pos: dict, out: Path, cols: int = 4):
    """Each cell: wrist | side model inputs (224 px, no augmentation) with V / target annotations."""
    cells = []
    for _, r in rows.iterrows():
        ims = ds[idx_pos[(r.episode_uid, r.frame_index)]]["images"]
        w, s = (ims[c].permute(1, 2, 0).numpy() for c in ["wrist_left", "side_1"])
        head = [f"{r.repo_short} ep{r.episode_index} t={r.t_s:.1f}s/{r['T'] / 15:.0f}s", f"{r.outcome} {r.fmode}"]
        cell = Image.new("RGB", (448, 224))
        cell.paste(tile(w, head + [f"V {r.V:+.2f}  no-wrist {r.V_nw:+.2f}", f"target {r.value:+.2f}"]), (0, 0))
        cell.paste(tile(s, [f"stall={bool(r.stall)}", f"wrist dCE {r.wrist_dCE:+.2f}"]), (224, 0))
        cells.append(cell)
    n = len(cells)
    grid = Image.new("RGB", (cols * 452, ((n + cols - 1) // cols) * 228), (40, 40, 40))
    for i, c in enumerate(cells):
        grid.paste(c, ((i % cols) * 452, (i // cols) * 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/wrist_optimism.md")
    ap.add_argument("--figs", default="reports/figures/stage5_5/wrist_optimism")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    tc = load_config("configs/train.yaml")
    dcfg = load_config(tc["data_config"])
    man = pd.read_parquet(load_config(dcfg["dataset_config"])["paths"]["manifest"])
    runs = Path(tc["paths"]["runs"])
    idx = pd.read_parquet(dcfg["paths"]["frame_index"])
    rmed = idx[idx.split == "train"].groupby("repo").value.median().to_dict()
    pred = pd.read_parquet(runs / "g0_r0_s0" / "val_predictions.parquet").drop(columns=["V", "ce"])
    traj = load_trajectories(Path(dcfg["paths"]["video_root"]), pred, Path("data/analysis/val_trajectory.parquet"))
    base = annotate(pred.assign(V=0.0, ce=0.0), man, traj, rmed).drop(columns=["V", "ce", "err", "bias", "cerr"])
    base["fmode"] = base.machine_failure_mode.fillna("(none)")
    base["success"] = base.outcome == "success"
    hold = (base.t_s > 5) & (base.to_T_s > 2)
    base["phase"] = np.select([base.to_T_s <= 5, base.t_s <= 5], ["last 5 s", "first 5 s"], "middle")

    arms = {}
    for name, (arm, kind) in ARMS.items():
        d = base.merge(load_arm(runs, arm, kind), on=KEYS, validate="1:1")
        d["bias"], d["bias_nw"] = d.V - d.value, d.V_nw - d.value
        arms[name] = d
    r0 = arms["R0"]
    ns = ~r0.success

    L = ["# Where the wrist camera makes not-success frames look better (C4 follow-up; `scripts/wrist_optimism.py`)\n",
         "Saved masking outputs only (no new runs). Per frame, mean over 3 seeds: wrist ΔV = V_full − V_without_wrist "
         "(> 0: the wrist raises V); wrist ΔCE = CE_full − CE_without_wrist (> 0: the wrist makes the loss worse). "
         "R0 / C1: `mean` intervention; C3: `drop` (token masked, no stand-in). bias = V − target. Groupings [ours]; "
         "machine_failure_mode is a VLM label (diagnostic).\n"]

    # 1. Is it the same in all three arms, and is it optimism (bias) that the wrist adds?
    rows = []
    for name, d in arms.items():
        for lab, m in [("success", d.success), ("not success", ~d.success)]:
            g = d[m]
            rows.append({"arm": name, "frames": lab, "bias full": g.bias.mean(), "bias no wrist": g.bias_nw.mean(),
                         "wrist ΔV": g.wrist_dV.mean(), "wrist ΔCE": g.wrist_dCE.mean(),
                         "share of frames wrist hurts": (g.wrist_dCE > 0).mean()})
    L += ["## 1. All arms: does the wrist add optimism on not-success frames?\n", md(pd.DataFrame(rows).set_index("arm")) + "\n"]

    # 2. R0 not-success breakdowns
    L.append("## 2. R0, not-success frames: where the wrist harm sits\n")
    for title, by in [("outcome × phase", ["outcome", "phase"]), ("stall (robot holding still) vs moving", ["outcome", "stall"]),
                      ("hold phase (> 5 s in, > 2 s before T)", [hold[ns].rename("hold")]),
                      ("machine_failure_mode", ["fmode"]), ("episode kind", ["kind"]), ("repo", ["repo_short"])]:
        g = r0[ns].groupby(by)
        L += [f"### {title}\n", md(summary(g)) + "\n"]
    # Control: the same tables on success frames, so the not-success numbers have a reference.
    L += ["### control: success frames, by phase\n", md(summary(r0[r0.success].groupby("phase"))) + "\n"]

    # 3. Concentration: is it a few episodes?
    ep = r0[ns].groupby("episode_uid").agg(repo=("repo_short", "first"), outcome=("outcome", "first"), mode=("fmode", "first"),
                                           kind=("kind", "first"), frames=("V", "size"), dCE=("wrist_dCE", "mean"),
                                           dV=("wrist_dV", "mean"), bias=("bias", "mean"), bias_nw=("bias_nw", "mean"))
    ep["harm"] = ep.dCE * ep.frames
    pos = ep.harm.clip(lower=0).sort_values(ascending=False)
    k = int(np.ceil(0.1 * len(ep)))
    # Same per-episode number in C1 / C3: does an episode where the wrist hurts in R0 also hurt in the others?
    for name in ["C1", "C3"]:
        d = arms[name]
        ep[f"dCE {name}"] = d[~d.success].groupby("episode_uid").wrist_dCE.mean()
    corr = ep[["dCE", "dCE C1", "dCE C3"]].corr(method="spearman")
    L += ["## 3. Per episode (R0, not success)\n",
          f"{(ep.dCE > 0).sum()} of {len(ep)} not-success episodes have mean wrist ΔCE > 0. Top 10% of episodes "
          f"({k}) hold {pos.iloc[:k].sum() / pos.sum():.0%} of the positive frame-weighted harm. Rank correlation of "
          f"per-episode wrist ΔCE, R0 vs C1 {corr.loc['dCE', 'dCE C1']:.2f}, R0 vs C3 {corr.loc['dCE', 'dCE C3']:.2f}.\n",
          md(ep.sort_values("dCE", ascending=False).drop(columns="harm").head(20)) + "\n",
          "### least wrist harm (most wrist help)\n", md(ep.sort_values("dCE").drop(columns="harm").head(8)) + "\n"]

    # 4. Matched comparison: hold-phase stalls of failures vs successes at the same point. Does the wrist
    # raise V *more* on stalled failures than on stalled successes (= confusing them), or equally?
    st = r0[hold & r0.stall]
    t = summary(st.groupby("success"))
    t.index = ["not success" if not i else "success" for i in t.index]
    L += ["## 4. Hold-phase stall frames: successes vs not\n",
          "If the wrist raises V on stalled not-success frames as much as on stalled successes, it cannot tell "
          "them apart; if it raises them *more*, the wrist view of a stall looks like a success.\n", md(t) + "\n"]
    from vl.metrics import auc
    for name, d in arms.items():
        s = d[hold & d.stall]
        L.append(f"- {name}: AUC(success vs not, hold-phase stall frames) with wrist {auc(s.success, s.V):.3f}, "
                 f"without wrist {auc(s.success, s.V_nw):.3f}")
    L.append("")

    Path(args.out).write_text("\n".join(L))
    print(f"-> {args.out}")
    ep.to_csv(Path(args.out).with_suffix(".episodes.csv"))

    if args.no_figs:
        return
    val_idx = idx[idx.split == "val"].reset_index(drop=True)
    pos_of = {k: i for i, k in enumerate(zip(val_idx.episode_uid, val_idx.frame_index))}
    ds = FrameDataset(val_idx, dcfg)
    figs = Path(args.figs)
    # Worst frames, one per episode (the frame of max wrist ΔCE), top 16 episodes by per-episode harm.
    top_eps = ep.sort_values("dCE", ascending=False).index[:16]
    worst = r0[r0.episode_uid.isin(top_eps)].sort_values("wrist_dCE", ascending=False).groupby("episode_uid").head(1)
    sheet(ds, worst.set_index("episode_uid").loc[top_eps].reset_index(), pos_of, figs / "worst_not_success.jpg")
    # Same episodes over time: 6 evenly spaced frames each for the top 4.
    for e in top_eps[:4]:
        d = r0[r0.episode_uid == e].sort_values("frame_index")
        sheet(ds, d.iloc[np.linspace(0, len(d) - 1, 8).astype(int)], pos_of,
              figs / f"episode_{d.repo_short.iloc[0]}_ep{d.episode_index.iloc[0]}.jpg")
    # Reference: success hold-phase stall frames the wrist raises most (what a "good-looking" wrist view is).
    ref = r0[r0.success & hold & r0.stall].sort_values("wrist_dV", ascending=False).groupby("episode_uid").head(1).head(8)
    sheet(ds, ref, pos_of, figs / "reference_success_stalls.jpg")
    print(f"-> {figs}/")


if __name__ == "__main__":
    main()
