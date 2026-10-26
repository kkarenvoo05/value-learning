"""Stage 5.5 C4: how much each camera contributes to V, by masking it (the check the C4 caveat asks for:
readout attention weights are not contributions).

Usage:
    .venv/bin/python -u -m vl.camera_contrib --run c3_attn_s0        # GPU node -> data/runs/<run>/val_cam_mask.parquet
    .venv/bin/python -m vl.camera_contrib --report c3_attn g0_r0 c1_percam   # CPU -> reports/camera_contrib.md

Best checkpoint, eval mode, no augmentation, same AMP as the training-time validation pass. Backbone
features of every val frame are computed once; the head is then re-run with one camera removed [ours]:
  drop  (attention-fusion runs only) the camera's token is masked out of the readout attention
        (key_padding_mask): the model's own way of "not seeing" a token, no stand-in input
  mean  the camera's feature vector is replaced by its mean over all val frames: the same intervention
        for concat heads (R0, C1) and attention heads, so the arms can be compared
Per frame and intervention: V, p_succ (mass on bins > -0.5) and CE against the target bin. A camera's
contribution at a frame = |V_full - V_masked|; its share = wrist / (wrist + side) contribution. Masking
a camera the model leans on also moves V a lot; masking moves features off the training distribution
(`mean` more than `drop`), so contributions are relative, not a decomposition of V.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vl.config import load_config
from vl.dataset import FrameDataset, load_split
from vl.model import build_model, run_data_config
from vl.train import amp_dtype

FPS = 15.0


def masked_feats(f: torch.Tensor, cam: int, n_cams: int, mean: torch.Tensor) -> torch.Tensor:
    """(B, n_cams * feat) features with camera `cam`'s block replaced by `mean` (feat,)."""
    feat = f.shape[1] // n_cams
    g = f.clone()
    g[:, cam * feat:(cam + 1) * feat] = mean.to(g.dtype)
    return g


def drop_mask(B: int, cam: int, n_cams: int, device) -> torch.Tensor:
    """key_padding_mask (B, n_cams) that hides camera `cam`'s token only (True = ignore)."""
    m = torch.zeros(B, n_cams, dtype=torch.bool, device=device)
    m[:, cam] = True
    return m


def head_outputs(logits: torch.Tensor, bins: torch.Tensor, v: torch.Tensor, succ: torch.Tensor) -> dict:
    p = F.softmax(logits.float(), -1)
    return {"V": (p @ v), "p_succ": p[:, succ].sum(1), "ce": F.cross_entropy(logits.float(), bins, reduction="none")}


def run_masks(args, cfg):
    dcfg, tcfg = load_config(cfg["data_config"]), load_config(cfg["target_config"])
    run = Path(cfg["paths"]["runs"]) / args.run
    dcfg = run_data_config(dcfg, run)
    if dcfg.get("history") or dcfg.get("proprio", {}).get("enabled") or dcfg["still_time"]["enabled"]:
        raise SystemExit(f"{args.run}: only image-only runs (no history / proprio / still time) are supported")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg["amp"])
    val = load_split(dcfg, "val")
    if args.val_limit:
        val = val[val.episode_uid.isin(val.episode_uid.unique()[:args.val_limit])].reset_index(drop=True)
    model = build_model(dcfg, tcfg, pretrained=False).to(device).eval()
    ck = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    cams, attention = model.cams, dcfg["fusion"] == "attention"
    dl = DataLoader(FrameDataset(val, dcfg), batch_size=128, shuffle=False, num_workers=args.workers,
                    pin_memory=True, prefetch_factor=4)
    feats, rows, bins = None, [], []
    with torch.no_grad():
        for i, b in enumerate(dl):
            images = {c: x.to(device, non_blocking=True) for c, x in b["images"].items()}
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                f = model.features(images)
            if feats is None:
                feats = torch.empty(len(val), f.shape[1], dtype=f.dtype, device=device)
            feats[b["row"].to(device)] = f
            if i % 100 == 0:
                print(f"  features {i}/{len(dl)}", flush=True)
    bins = torch.as_tensor(val.bin.to_numpy().copy(), device=device)
    v, succ = model.bin_centers, model.bin_centers > -0.5
    mean = feats.float().mean(0)
    feat = feats.shape[1] // len(cams)
    out = {}
    with torch.no_grad():
        for s in range(0, len(val), 1024):
            f, yb = feats[s:s + 1024], bins[s:s + 1024]
            variants = {"full": lambda: model.head(f)}
            for k, c in enumerate(cams):
                variants[f"mean_{c}"] = lambda k=k: model.head(masked_feats(f, k, len(cams), mean[k * feat:(k + 1) * feat]))
                if attention:
                    variants[f"drop_{c}"] = lambda k=k: model.head(f, key_padding_mask=drop_mask(len(f), k, len(cams), device))
            for name, fn in variants.items():
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    logits = fn()
                for q, x in head_outputs(logits, yb, v, succ).items():
                    out.setdefault(f"{q}_{name}", []).append(x.cpu().numpy())
    res = pd.concat([val[["episode_uid", "frame_index"]].reset_index(drop=True),
                     pd.DataFrame({k: np.concatenate(x) for k, x in out.items()})], axis=1)
    pred = pd.read_parquet(run / "val_predictions.parquet")
    m = pred.merge(res, on=["episode_uid", "frame_index"], validate="1:1")
    gap = (m.V - m.V_full).abs()
    print(f"check vs val_predictions.parquet (training-time val pass): max |ΔV| {gap.max():.4f}, "
          f"mean {gap.mean():.5f}; CE {m.ce_full.mean():.4f} vs {m.ce.mean():.4f}", flush=True)
    res.to_parquet(run / "val_cam_mask.parquet")
    print(f"-> {run / 'val_cam_mask.parquet'} (best step {ck['step']}, {len(res):,} frames, "
          f"variants {list(variants)})", flush=True)


def phase(d: pd.DataFrame) -> pd.Series:
    """Coarse task phase x outcome [ours]: the last 5 s before T (insertion / seating for successes,
    the end of a failed attempt otherwise), the first 5 s (approach), and the middle."""
    to_go = (d["T"] - d.frame_index) / FPS
    out = np.where(d.outcome == "success", "success", "not success")
    ph = np.select([to_go <= 5, d.frame_index / FPS <= 5], ["last 5 s", "first 5 s"], "middle")
    return pd.Series(out, index=d.index) + ", " + ph


def report(arms: list[str], runs_dir: Path, out: Path):
    from vl.analysis import md
    lines = ["# Camera contributions by masking (Stage 5.5 C4; generated by `vl.camera_contrib --report`)\n",
             "Per val frame: |V_full − V_masked| when one camera is removed (`drop` = its token masked out of "
             "the readout attention, attention runs only; `mean` = its features replaced by their val mean). "
             "wrist share = wrist / (wrist + side) contribution, averaged over frames. ΔCE = CE with the camera "
             "removed − full CE (how much the val loss depends on it). Mean over the arm's seeds; ± = seed sd. "
             "Readout attention (C3, last block, mean over heads) shown next to it where it exists.\n"]
    cams = ["wrist_left", "side_1"]
    for arm in arms:
        per_seed = []
        for s in range(3):
            r = runs_dir / f"{arm}_s{s}"
            if not (r / "val_cam_mask.parquet").exists():
                continue
            m = pd.read_parquet(r / "val_cam_mask.parquet")
            pred = pd.read_parquet(r / "val_predictions.parquet")[["episode_uid", "frame_index", "T", "outcome", "repo"]]
            m = m.merge(pred, on=["episode_uid", "frame_index"], validate="1:1")
            if (r / "val_attn.parquet").exists():
                a = pd.read_parquet(r / "val_attn.parquet")
                last = max(int(c.rsplit("_l", 1)[1]) for c in a.columns if c.startswith("attn_"))
                m = m.merge(a[["episode_uid", "frame_index", f"attn_wrist_left_l{last}"]].rename(
                    columns={f"attn_wrist_left_l{last}": "attn_wrist"}), on=["episode_uid", "frame_index"], validate="1:1")
            m["phase"] = phase(m)
            rows = {}
            for kind in ["drop", "mean"]:
                if f"V_{kind}_{cams[0]}" not in m:
                    continue
                c = {cam: (m.V_full - m[f"V_{kind}_{cam}"]).abs() for cam in cams}
                m[f"{kind} wrist |ΔV|"], m[f"{kind} side |ΔV|"] = c[cams[0]], c[cams[1]]
                m[f"{kind} wrist share"] = c[cams[0]] / (c[cams[0]] + c[cams[1]]).clip(lower=1e-9)
                for cam, lab in zip(cams, ["wrist", "side"]):
                    m[f"{kind} ΔCE {lab}"] = m[f"ce_{kind}_{cam}"] - m.ce_full
            cols = [c for c in m.columns if c.startswith(("drop ", "mean ")) or c == "attn_wrist"]
            g = m.groupby("phase")[cols].mean()
            g.loc["all frames"] = m[cols].mean()
            g["frames"] = m.groupby("phase").size().reindex(g.index).fillna(len(m)).astype(int)
            if "attn_wrist" in m and "drop wrist share" in m:
                rho = m[["attn_wrist", "drop wrist share"]].corr(method="spearman").iloc[0, 1]
                g["rank corr attn vs drop share (all frames)"] = rho
            per_seed.append(g)
        if not per_seed:
            lines.append(f"## {arm}\n\nno `val_cam_mask.parquet` yet\n")
            continue
        cat = pd.concat(per_seed, keys=range(len(per_seed)))
        mean, sd = cat.groupby(level=1, sort=False).mean(), cat.groupby(level=1, sort=False).std()
        t = mean.map(lambda x: f"{x:.3f}")
        if len(per_seed) > 1:
            for c in mean.columns:
                if c != "frames":
                    t[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean[c], sd[c])]
        t["frames"] = mean["frames"].astype(int).astype(str)
        order = [p for p in ["success, first 5 s", "success, middle", "success, last 5 s", "not success, first 5 s",
                             "not success, middle", "not success, last 5 s", "all frames"] if p in t.index]
        lines += [f"## {arm} ({len(per_seed)} seeds)\n", md(t.loc[order]) + "\n"]
    out.write_text("\n".join(lines))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run")
    ap.add_argument("--report", nargs="+", metavar="ARM")
    ap.add_argument("--out", default="reports/camera_contrib.md")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-limit", type=int, help="only the first N validation episodes (smoke tests)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.report:
        report(args.report, Path(cfg["paths"]["runs"]), Path(args.out))
    elif args.run:
        run_masks(args, cfg)
    else:
        ap.error("--run or --report")


if __name__ == "__main__":
    main()
