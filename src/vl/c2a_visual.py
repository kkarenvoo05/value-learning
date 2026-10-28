"""Stage 5.5 C2a visual check: what does the stacked-history model react to on stall frames?

Usage:
    .venv/bin/python -m vl.c2a_visual                  # -> reports/c2a_visual.md, reports/figures/stage5_5/c2a_visual/

Val split, best checkpoints of `--arm` (default c2a_stack2, seeds 0-2), no augmentation. Groups are the
hold-phase groups of `vl.history_gain.hold_groups` (eval repos, > 5 s in, > 2 s before T; "still" = gripper
still >= 4 s at that frame). Two parts [ours]:

1. Counterfactual past frames, `--per-group` frames sampled per group (seeded, spread over episodes), every seed:
     real   the actual frame t - gap (what the model was trained on)
     still  past := frame t - 1 (1/15 s back), has_past = 1: "nothing moved in the last gap seconds", with the
            video's real frame-to-frame noise (an exact copy of frame t never occurs in the data)
     copy   past := frame t exactly (out of distribution: reported, not interpreted on its own)
     none   has_past = 0 (grey past channels, as in the first gap seconds of an episode)
   V real vs still: does the model's judgement come from "nothing changed"? If V(real) ~ V(still) on
   still stalls, yes; if V(real) is lower than V(still), the past frame carries more than stillness.
   V real vs none: how much the past moves V at all.
2. Sheets: frames where the arm lowers p_succ most vs `--ref` (mean over seeds, saved predictions), one per
   episode: still stalls (the gain) and still frozen successes (the cost), plus moving successes as a
   control. Per camera: current frame, past frame, |current - past| (x4), and |input gradient x input| of V
   (seed 0) over the current and the past channels, so one sees which pixels of which frame V depends on.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset

from vl.config import load_config
from vl.dataset import FrameDataset, load_split
from vl.history_gain import RUNS, hold_groups, load, seeds
from vl.model import build_model, model_inputs, run_data_config

GROUPS = {  # name -> (hold group, still >= 4 s now)
    "stall, still": ("stall", True),
    "stall, moving": ("stall", False),
    "success frozen, still": ("success, frozen (pause >= 8 s)", True),
    "success frozen, moving": ("success, frozen (pause >= 8 s)", False),
    "success pause < 2 s": ("success, pause < 2 s", False),
}


def sample(h: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Up to n frames per group, spread over episodes: shuffle, then round-robin over episodes."""
    out = []
    for name, (g, still) in GROUPS.items():
        d = h[(h.group == g) & (h["still >= 4 s now"] == still)]
        d = d.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        d = d.assign(_k=d.groupby("episode_uid").cumcount()).sort_values("_k").head(n).drop(columns="_k")
        out.append(d.assign(cf_group=name))
    return pd.concat(out, ignore_index=True)


class WithNear(Dataset):
    """The arm's items plus "near": each camera's frame t - 1 (a one-frame-gap past), for the "still" counterfactual."""

    def __init__(self, val, dcfg):
        self.ds = FrameDataset(val, dcfg)
        self.near = FrameDataset(val, dict(dcfg, history={"gap_s": 1 / 15}))
        assert (self.near.gap_frames == 1).all()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return dict(self.ds[i], near=self.near[i]["past"]["images"])


def loader(dcfg, val, keys: pd.DataFrame, workers: int):
    pos = val.reset_index().merge(keys[["episode_uid", "frame_index"]], on=["episode_uid", "frame_index"])["index"]
    assert len(pos) == len(keys), "sampled frames missing from the val index"
    ds = WithNear(val, dcfg)
    return ds.ds, DataLoader(Subset(ds, pos.tolist()), batch_size=32, num_workers=workers)


@torch.no_grad()
def counterfactuals(model, dl, device) -> pd.DataFrame:
    rows = []
    for b in dl:
        inp = model_inputs(b, device)
        V = {}
        V["real"] = model.expected_value(model(**inp))
        ones = torch.ones_like(inp["past"]["has_past"])
        still = {"images": {c: b["near"][c].to(device) for c in model.cams}, "has_past": ones}
        V["still"] = model.expected_value(model(**dict(inp, past=still)))
        copy = {"images": {c: inp["images"][c] for c in model.cams}, "has_past": ones}
        V["copy"] = model.expected_value(model(**dict(inp, past=copy)))
        none = dict(inp["past"], has_past=torch.zeros_like(inp["past"]["has_past"]))
        V["none"] = model.expected_value(model(**dict(inp, past=none)))
        rows.append(pd.DataFrame({"row": b["row"].numpy(), "has_past": inp["past"]["has_past"].cpu().numpy(),
                                  **{f"V_{k}": v.cpu().numpy() for k, v in V.items()}}))
    return pd.concat(rows, ignore_index=True)


def saliency(model, item, device) -> dict:
    """|d V / d input x input| summed over colour channels, per camera, for the current and the past frame."""
    model.eval()
    now = {c: model.normalize(item["images"][c][None].to(device)).requires_grad_() for c in model.cams}
    past = {c: model.normalize(item["past"]["images"][c][None].to(device)).requires_grad_() for c in model.cams}
    m = torch.tensor([[item["past"]["has_past"]]], device=device, dtype=torch.float32)
    x = torch.cat([torch.cat([now[c], past[c] * m.view(-1, 1, 1, 1)], dim=1) for c in model.cams])
    f = model.backbone(x)
    v = model.expected_value(model.head(torch.cat([*f.chunk(len(model.cams)), m], dim=1)))
    with torch.no_grad():   # the hand-built forward above must be the model's own
        ref = model.expected_value(model(**model_inputs({"images": {c: item["images"][c][None] for c in model.cams},
                                                         "past": {"images": {c: item["past"]["images"][c][None] for c in model.cams},
                                                                  "has_past": m[:, 0].cpu()}}, device)))
    assert torch.allclose(v.detach(), ref, atol=1e-5), (float(v), float(ref))
    v.sum().backward()
    return {c: {k: (t[c].grad * t[c]).detach().abs().sum(1)[0].cpu().numpy() for k, t in [("now", now), ("past", past)]}
            for c in model.cams}


def heat(s: np.ndarray, vmax: float) -> np.ndarray:
    """Gradient x input is speckled pixel to pixel; a small blur shows where it concentrates."""
    a = np.clip(s / vmax, 0, 1)
    im = Image.fromarray((np.stack([a, a ** 2, np.zeros_like(a)], -1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(2)))


def sheet(rows: pd.DataFrame, ds, model, device, title: str, out: Path):
    W = 160
    cols = ["now", "past", "|diff| x4", "saliency now", "saliency past"]
    cells = []
    for _, r in rows.iterrows():
        item = ds[int(r.val_row)]
        sal = saliency(model, item, device)
        tiles = []
        for c in model.cams:
            now = item["images"][c].permute(1, 2, 0).numpy()
            past = item["past"]["images"][c].permute(1, 2, 0).numpy()
            diff = np.clip(np.abs(now.astype(int) - past.astype(int)) * 4, 0, 255).astype(np.uint8)
            vmax = np.percentile(np.concatenate([sal[c]["now"].ravel(), sal[c]["past"].ravel()]), 99.5) + 1e-12
            tiles += [now, past, diff, heat(sal[c]["now"], vmax), heat(sal[c]["past"], vmax)]
        row = Image.new("RGB", (W * len(tiles), W + 30), (30, 30, 30))
        for k, t in enumerate(tiles):
            row.paste(Image.fromarray(t).resize((W, W)), (k * W, 30))
        d = ImageDraw.Draw(row)
        d.text((4, 2), f"{r.repo_short} ep{r.episode_index} t={r.t_s:.1f}s  still {r.stall_s:.0f}s  {r.outcome}  "
                       f"target {r.value:+.2f} | V {r.V_arm:+.2f} (ref {r.V_ref:+.2f})  p_succ {r.p_arm:.2f} (ref {r.p_ref:.2f})",
               fill=(255, 255, 0))
        d.text((4, 15), f"seed-0 counterfactual V: real {r.V_real:+.2f}  past=t-1 {r.V_still:+.2f}  "
                        f"past=exact copy {r.V_copy:+.2f}  no past {r.V_none:+.2f}",
               fill=(180, 220, 255))
        cells.append(row)
    head = Image.new("RGB", (cells[0].width, 34), (0, 0, 0))
    d = ImageDraw.Draw(head)
    d.text((4, 2), title, fill=(255, 255, 255))
    for k, c in enumerate([f"{cam}: {n}" for cam in model.cams for n in cols]):
        d.text((k * W + 4, 18), c, fill=(200, 200, 200))
    grid = Image.new("RGB", (head.width, head.height + sum(c.height for c in cells)))
    grid.paste(head, (0, 0))
    y = head.height
    for c in cells:
        grid.paste(c, (0, y))
        y += c.height
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="c2a_stack2")
    ap.add_argument("--ref", default="g0_r0")
    ap.add_argument("--per-group", type=int, default=100)
    ap.add_argument("--sheet", type=int, default=10, help="frames per sheet")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="reports/c2a_visual.md")
    ap.add_argument("--figs", default="reports/figures/stage5_5/c2a_visual")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    tc = load_config("configs/train.yaml")
    d0, t0 = load_config(tc["data_config"]), load_config(tc["target_config"])
    runs = [f"{args.arm}_s{s}" for s in seeds(args.arm)]
    dcfg = run_data_config(d0, RUNS / runs[0])
    assert (dcfg.get("history") or {}).get("stack") and (dcfg["history"].get("gap_s") is not None), \
        f"{args.arm}: built for single-gap stacked history runs (C2a)"
    val = load_split(dcfg, "val")

    d = load([args.ref, args.arm])
    for name, a in [("ref", args.ref), ("arm", args.arm)]:
        rs = [f"{a}_s{s}" for s in seeds(a)]
        d[f"V_{name}"] = d[[f"V@{r}" for r in rs]].mean(1)
        d[f"p_{name}"] = d[[f"p_succ@{r}" for r in rs]].mean(1)
    d["repo_short"] = d.repo.str.extract(r"(r\d\d)")[0]
    vpos = val.reset_index()[["index", "episode_uid", "frame_index"]].rename(columns={"index": "val_row"})
    d = d.merge(vpos, on=["episode_uid", "frame_index"], validate="1:1")
    h = hold_groups(d)

    # 1. Counterfactuals, every seed
    cf_rows = sample(h, args.per_group, rng)
    ds, dl = loader(dcfg, val, cf_rows, args.workers)
    per_seed = []
    for r in runs:
        model = build_model(run_data_config(d0, RUNS / r), t0, pretrained=False).to(device)
        model.load_state_dict(torch.load(RUNS / r / "best.pt", map_location=device)["model"])
        cf = counterfactuals(model.eval(), dl, device).rename(columns={"row": "val_row"})
        per_seed.append(cf.assign(run=r))
        print(f"{r}: {len(cf)} frames", flush=True)
    cf = pd.concat(per_seed).merge(cf_rows, on="val_row", validate="m:1")
    assert (cf.has_past == 1).all(), "hold-phase frames are > 5 s in, so every sampled frame has a past"
    cf["real - still"], cf["real - none"] = cf.V_real - cf.V_still, cf.V_real - cf.V_none
    by_seed = cf.groupby(["cf_group", "run"])[["V_real", "V_still", "V_copy", "V_none", "real - still", "real - none"]].mean()
    tab = by_seed.groupby("cf_group").agg(["mean", "std"])
    tab.columns = [f"{a} ({b})" if b == "std" else a for a, b in tab.columns]
    n = cf.groupby("cf_group").agg(episodes=("episode_uid", "nunique"), frames=("val_row", "nunique"),
                                   target=("value", "mean"), still_s=("stall_s", "mean"))
    tab = n.join(tab).loc[list(GROUPS)]

    # 2. Sheets (seed 0 model, frames chosen on the mean-over-seeds predictions)
    model = build_model(run_data_config(d0, RUNS / runs[0]), t0, pretrained=False).to(device)
    model.load_state_dict(torch.load(RUNS / runs[0] / "best.pt", map_location=device)["model"])
    model.eval()
    h = h.assign(dp=h.p_arm - h.p_ref)
    picks = {
        "stall_still_most_lowered": ("still stalls where C2a lowers p_succ most vs R0 (the gain)",
                                     h[(h.group == "stall") & h["still >= 4 s now"]].sort_values("dp")),
        "frozen_success_still_most_lowered": ("still frozen successes where C2a lowers p_succ most vs R0 (the cost)",
                                              h[(h.group == "success, frozen (pause >= 8 s)") & h["still >= 4 s now"]].sort_values("dp")),
        "moving_success_control": ("moving successes, random (control)",
                                   h[h.group == "success, pause < 2 s"].sample(frac=1, random_state=0)),
    }
    figs = []
    for key, (title, rows) in picks.items():
        rows = rows.drop_duplicates("episode_uid").head(args.sheet).copy()
        _, dl1 = loader(dcfg, val, rows, 0)
        cf1 = counterfactuals(model, dl1, device).rename(columns={"row": "val_row"})
        rows = rows.merge(cf1, on="val_row", validate="1:1")
        out = Path(args.figs) / f"{key}.jpg"
        sheet(rows, ds, model, device, f"{args.arm} seed 0 | {title}", out)
        figs.append((title, out, rows))
        print(f"-> {out}", flush=True)

    L = [f"# C2a visual check (`.venv/bin/python -m vl.c2a_visual --arm {args.arm}`)\n",
         "See the module docstring for definitions. V counterfactuals: mean over frames per seed, then mean (sd) "
         f"over seeds {runs}. 'real − still' < 0: the real past makes V lower than a frozen copy of the current "
         "frame would; 'real − none': how much the past frame moves V at all.\n",
         "## 1. Counterfactual past frames\n", tab.round(3).to_markdown(), "",
         "## 2. Sheets\n",
         "Per row and camera: current frame, past frame, |current − past| × 4, then |gradient × input| of V over the "
         "current and the past channels (seed 0, shared colour scale per camera). Header: target, V and p_succ of the "
         "arm vs the reference (mean over seeds), and seed 0's V with the real past / past = frame t − 1 / past = exact copy / no past.\n"]
    for title, out, rows in figs:
        L += [f"### {title}\n", f"![]({os.path.relpath(out, Path(args.out).parent)})\n",
              rows[["repo_short", "episode_index", "t_s", "stall_s", "value", "V_ref", "V_arm", "p_ref", "p_arm",
                    "V_real", "V_still", "V_copy", "V_none"]].round(3).to_markdown(index=False), ""]
    Path(args.out).write_text("\n".join(L))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
