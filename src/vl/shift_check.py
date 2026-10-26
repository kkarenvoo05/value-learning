"""Test-time crop shift on finished runs (Stage 5, no training): does losing the crop's absolute
position make a model less sure of success?

Hypothesis (b) in PROGRESS.md: A2 crop costs the success frames because shifting the crop box removes
absolute pixel position, which on fixed cameras encodes the arm's pose near the holder. If so, feeding
a model that never saw shifts (baseline, full aug) A2-shifted inputs at test time should lower p_succ
on success frames, more when the fixed side camera is shifted than the wrist camera.

Usage (GPU; see scripts/shift_check.sbatch):
    .venv/bin/python -u -m vl.shift_check --runs lowlr_8k lowlr_8k_s1 ...
        -> data/runs/shift_check/frames.npz  (per frame, run, condition: p_succ, V, ce)
    .venv/bin/python -m vl.shift_check --report    (CPU) -> reports/shift_check.md

Each val frame is decoded once and cropped twice per camera: the fixed box and one A2 box
(`jitter_box` with crop_shift_px 24, crop_scale 0.9-1.0, clamped), drawn once per frame and camera
from a fixed seed, so every run and condition sees the same shifted pixels. Conditions:
none (fixed boxes), both (both shifted), side (side_1 only), wrist (wrist_left only). [ours]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vl.config import load_config
from vl.dataset import FrameDataset, crop_resize, jitter_box, load_split
from vl.model import build_model, run_data_config
from vl.train import amp_dtype

A2 = {"crop_shift_px": 24, "crop_scale": [0.9, 1.0]}
CONDITIONS = {"none": (), "both": ("wrist_left", "side_1"), "side": ("side_1",), "wrist": ("wrist_left",)}
OUT = Path("data/runs/shift_check")


def draw_boxes(n: int, crops: dict, frame_hw=(480, 640), seed: int = 0) -> dict:
    """One A2 box per frame and camera, drawn in row order from a fixed seed."""
    torch.manual_seed(seed)
    return {c: np.array([jitter_box(box, A2, frame_hw) for _ in range(n)]) for c, box in crops.items()}


class ShiftPairs(FrameDataset):
    """Items carry each camera twice: fixed box (images) and the pre-drawn A2 box (shifted)."""

    def __init__(self, index, cfg, boxes):
        super().__init__(index, cfg)
        self.boxes = boxes

    def __getitem__(self, i):
        fixed, shifted = {}, {}
        for c in self.cams:
            img = self.reader().read(self.paths[c][i], self.from_ts[c][i], int(self.t[i]), self.fps[i])
            assert img.shape[:2] == (480, 640), img.shape
            for out, box in ((fixed, self.crops[c]), (shifted, tuple(self.boxes[c][i]))):
                out[c] = torch.from_numpy(crop_resize(img, box, self.size, self.pad_fill)).permute(2, 0, 1).contiguous()
        return {"images": fixed, "shifted": shifted, "bin": int(self.bin[i]), "row": i}


def run_models(args):
    cfg = load_config(args.config)
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    device, dtype = torch.device("cuda"), amp_dtype(cfg["amp"])
    val = load_split(dcfg, "val")
    if args.val_limit:
        val = val[val.episode_uid.isin(val.episode_uid.unique()[:args.val_limit])].reset_index(drop=True)
    runs = Path(cfg["paths"]["runs"])
    models = {}
    for r in args.runs:
        rc = run_data_config(dcfg, runs / r)
        assert rc["cameras"] == dcfg["cameras"] and rc["backbone"] == "resnet18" and not rc.get("proprio", {}).get("enabled")
        m = build_model(rc, tcfg).to(device).eval()
        ck = torch.load(runs / r / "best.pt", map_location=device)
        m.load_state_dict(ck["model"])
        models[r] = m
        print(f"{r}: best step {ck['step']}", flush=True)
    boxes = draw_boxes(len(val), dcfg["crops"], seed=args.seed)
    dl = DataLoader(ShiftPairs(val, dcfg, boxes), batch_size=128, shuffle=False, num_workers=args.workers,
                    pin_memory=True, prefetch_factor=4)
    v = torch.linspace(tcfg["v_min"], tcfg["v_max"], tcfg["n_bins"], device=device)
    succ = v > -0.5
    shape = (len(args.runs), len(CONDITIONS), len(val))
    p_succ, V, ce = (np.empty(shape, np.float32) for _ in range(3))
    with torch.no_grad():
        for i, b in enumerate(dl):
            rows, y = b["row"].numpy(), b["bin"].to(device)
            for k, cams in enumerate(CONDITIONS.values()):
                x = {c: (b["shifted"] if c in cams else b["images"])[c].to(device, non_blocking=True) for c in dcfg["cameras"]}
                for j, m in enumerate(models.values()):
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                        logits = m(x, None)
                    lp = torch.log_softmax(logits.float(), 1)
                    P = lp.exp()
                    p_succ[j, k, rows] = P[:, succ].sum(1).cpu().numpy()
                    V[j, k, rows] = (P * v).sum(1).cpu().numpy()
                    ce[j, k, rows] = (-lp.gather(1, y[:, None])[:, 0]).cpu().numpy()
            if i % 50 == 0:
                print(f"  batch {i}/{len(dl)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "frames.npz", p_succ=p_succ, V=V, ce=ce, runs=np.array(args.runs),
             conditions=np.array(list(CONDITIONS)), frame_index=val.frame_index.to_numpy(),
             episode_uid=val.episode_uid.to_numpy().astype(str),
             **{f"box_{c}": b for c, b in boxes.items()})
    print(f"-> {OUT / 'frames.npz'}", flush=True)


def report(args):
    from vl.analysis import md
    from vl.metrics import repo_medians, score

    cfg = load_config(args.config)
    dcfg = load_config(cfg["data_config"])
    z = np.load(OUT / "frames.npz")
    runs, conds = list(z["runs"]), list(z["conditions"])
    val = load_split(dcfg, "val")
    assert (val.frame_index.to_numpy() == z["frame_index"]).all() and (val.episode_uid.astype(str).to_numpy() == z["episode_uid"]).all()
    med = repo_medians(load_split(dcfg, "train"))
    arm = {r: "baseline" if r.startswith("lowlr") else r.rsplit("_s", 1)[0] for r in runs}
    base = val[["episode_uid", "repo", "outcome", "frame_index", "T", "value"]]
    ttg = pd.cut((val["T"] - val.frame_index) / 15, [-0.01, 1, 5, 15, 1e9], labels=["0-1 s", "1-5 s", "5-15 s", "> 15 s"])
    lines = ["# Test-time crop shift on finished runs (generated by `vl.shift_check --report`)\n",
             "Validation, best checkpoints. A2 box per frame and camera (shift +-24 px, scale 0.9-1.0, clamped), same "
             "boxes for every run. none = fixed crops; both / side / wrist = which cameras get the A2 box. "
             "p_succ = mass above -0.5. Mean over 3 seeds per arm; ± = sd over seeds.\n"]

    # 1. Consistency: the 'none' condition must reproduce each run's val_predictions.parquet.
    chk = []
    for j, r in enumerate(runs):
        vp = pd.read_parquet(Path(cfg["paths"]["runs"]) / r / "val_predictions.parquet")
        chk.append(f"{r} {np.abs(vp.V.to_numpy() - z['V'][j, 0]).max():.4f}")
    lines += ["Max |V(none) - val_predictions V| per run (AMP noise only): " + ", ".join(chk) + "\n"]

    # 2. Metrics per arm and condition.
    rows = []
    for j, r in enumerate(runs):
        for k, c in enumerate(conds):
            s = score(base.assign(V=z["V"][j, k], ce=z["ce"][j, k]), med)
            o = {f"MAE {oc}": float(np.abs(z["V"][j, k] - val.value)[val.outcome == oc].mean()) for oc in ("success", "failure", "timeout")}
            rows.append({"arm": arm[r], "condition": c, "val CE": s["val_ce"], "val MAE": s["val_mae"],
                         "shape MAE": s["offset_free_mae"], "eval-repo MAE": s["eval_repos_mae"],
                         "AUC 1 s": s["eval_repos_auc_1s"], **o})
    m = pd.DataFrame(rows).groupby(["arm", "condition"], sort=False)
    t = m.mean().round(3).astype(str) + " ± " + m.std().round(3).astype(str)
    lines += ["## Metrics\n", md(t) + "\n"]

    # 3. Mean p_succ change vs 'none', by outcome and time to the end (the crop signature).
    rows = []
    for j, r in enumerate(runs):
        for k, c in enumerate(conds[1:], 1):
            d = pd.DataFrame({"d": z["p_succ"][j, k] - z["p_succ"][j, 0], "outcome": val.outcome, "ttg": ttg})
            for (oc, w), g in d.groupby(["outcome", "ttg"], observed=True):
                rows.append({"arm": arm[r], "condition": c, "outcome": oc, "time to end": w, "d": g.d.mean()})
    d = pd.DataFrame(rows).groupby(["arm", "condition", "outcome", "time to end"], sort=False, observed=True).d
    t = (d.mean().round(3).astype(str) + " ± " + d.std().round(3).astype(str)).unstack("time to end")
    lines += ["## Change in mean p_succ vs none (shifted - fixed), by outcome and time to the end\n", md(t) + "\n"]

    # 4. Reference: crop arm vs baseline at fixed crops (what crop training did to p_succ).
    ps = pd.DataFrame({a: z["p_succ"][[j for j, r in enumerate(runs) if arm[r] == a], 0].mean(0) for a in sorted(set(arm.values()))})
    ps = ps.assign(outcome=val.outcome, ttg=ttg)
    ref = ps.groupby(["outcome", "ttg"], observed=True).mean()
    ref = ref.drop(columns="baseline").sub(ref.baseline, axis=0).round(3)
    lines += ["## Reference: mean p_succ of each arm minus baseline, fixed crops (3-seed means)\n", md(ref.unstack("ttg")) + "\n"]
    out = Path("reports/shift_check.md")
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-limit", type=int, help="first N validation episodes (smoke test)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else run_models(args)


if __name__ == "__main__":
    main()
