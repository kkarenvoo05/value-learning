"""P1 (docs/libero_plan.md 2026-11-02): are pi05's image features unsteady in themselves? Frozen encoders on the same
frames, measured without fine-tuning.

extract (one GPU): one pass over the fixed + fresh frame cache (<dataset>/frames_every4.npy, vl.libero_resnet.table_meta
order), every encoder on both cameras, pooled feature per image -> <out>/<encoder>.npy float16 (n, 2, d), plus
<out>/meta.parquet. Encoders:
  resnet_in       ImageNet ResNet-18 (vl.model ValueModel backbone, pretrained, untrained here), 512-d avg pool
  siglip_pi05     pi05_libero's SigLIP (vl.model.SiglipPi05Features), mean of 256 tokens, 1152-d
  siglip_web      web SigLIP v1 So400m/14 224 (timm vit_so400m_patch14_siglip_224.webli: the PaliGemma-family encoder
                  without robot training), same mean of 256 final-norm tokens [ours: MAP head not used, for parity]
  resnet_ft       fine-tuned reference: resnet18_fixedfresh_s0_18237895/best.pt backbone (SNR 0.247 run)
  siglip_pi05_ft  fine-tuned reference: siglip_pi05_fixedfresh_s0_18269859/best.pt backbone (SNR 0.129 run)
For the two *_ft encoders the full model's val value probabilities are saved too (<out>/probs_<encoder>_val.npy, val
rows in meta order) for the P2 lump decomposition. fp16 autocast [ours]; checked against fp32 on 8 images at start.

analyze (CPU): per encoder, splits as vl.libero_resnet on the fixed + fresh table (fit = fit_fixed_fresh.txt minus
inner, inner = vl.libero_probe.inner_split's 5 states, val = the 10 val states); features = [agentview, wrist]
concatenated, centred on fit [ours throughout]:
  (a) robot-state share: ridge (alpha 1e-3 x n) from degree-2 polynomial features of the standardised 8-d robot state
      (raw `state` at each frame) to the top-64 fit PCs of the features; R^2 on val = share of that feature variance
      explained by the arm. Same with "progress" predictors (pots on stove one-hot, value target, frame / T_MAX) and
      with both, so the unique shares can be read off.
  (b) pot-on-stove linear probe: logistic regression on standardised features (fit), label = vl.libero_subtask >= 1.
      Val accuracy; over 2 s (5 grid frames) within val episodes, event signal = mean change of P(pot) over windows
      where the label goes 0 -> 1, noise = sd of the change over windows with a constant label; event SNR = ratio;
      flip rate = share of constant-label windows whose predicted class differs at the two ends.
  (c) value head: scripts/libero_head_swap.py's Head "mlp" (d -> 256 -> 201, value CE, AdamW 1e-3, 40 epochs,
      selection on inner CE), 3 seeds, scored with its score() (change SNR 2 s, milestone, MAE, AUC).
  (d) P2 lump decomposition (scripts/libero_flicker_modes.py) on the two fine-tuned models' val probabilities.
Writes <out>/analysis.json and prints the tables.

Usage: .venv/bin/python scripts/libero_frozen_encoders.py extract --out <dir>
       PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze --out <dir>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.libero_resnet import CAMS, table_meta  # noqa: E402
from vl.model import ValueModel  # noqa: E402

DATASET = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
FT = {"resnet_ft": ("resnet18", RUNS / "resnet18_fixedfresh_s0_18237895/best.pt"),
      "siglip_pi05_ft": ("siglip_pi05", RUNS / "siglip_pi05_fixedfresh_s0_18269859/best.pt")}
WEB = "timm/vit_so400m_patch14_siglip_224.webli"


class WebSiglip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(WEB, "model.safetensors")  # timm's own hub check fails here (memory: dinov2 setup)
        self.vit = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=True,
                                     pretrained_cfg_overlay={"file": path}, num_classes=0)

    def forward(self, x):  # x already in [-1, 1]
        return self.vit.forward_features(x).mean(1)


def encoders(device):
    """name -> (feature fn on uint8 (N, 3, H, W), full model or None)."""
    out = {}
    m = ValueModel(CAMS, backbone="resnet18").to(device).eval()
    out["resnet_in"] = (lambda x, m=m: m.backbone(m.normalize(x)), None)
    m = ValueModel(CAMS, backbone="siglip_pi05").to(device).eval()
    out["siglip_pi05"] = (lambda x, m=m: m.backbone(m.normalize(x)), None)
    w = WebSiglip().to(device).eval()
    out["siglip_web"] = (lambda x, w=w: w(x.float() / 255.0 * 2.0 - 1.0), None)
    for name, (bb, ckpt) in FT.items():
        m = ValueModel(CAMS, backbone=bb).to(device).eval()
        m.load_state_dict(torch.load(ckpt, map_location=device))
        out[name] = (lambda x, m=m: m.backbone(m.normalize(x)), m)
    return out


@torch.no_grad()
def extract(a):
    device = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = table_meta(DATASET, 4)
    arr = np.load(DATASET / "frames_every4.npy", mmap_mode="r")
    assert len(arr) == len(meta)
    meta.to_parquet(out / "meta.parquet")
    enc = encoders(device)
    x8 = torch.from_numpy(np.ascontiguousarray(arr[:4])).to(device).flatten(0, 1)
    for name, (fn, _) in enc.items():  # fp16 vs fp32 check
        f32 = fn(x8).float()
        with torch.autocast("cuda", dtype=torch.float16):
            f16 = fn(x8).float()
        rel = float((f16 - f32).norm() / f32.norm())
        print(f"{name}: dim {f32.shape[1]}, fp16 vs fp32 relative error {rel:.2e}", flush=True)
        assert rel < 2e-2, name
    n = len(meta)
    feats = {k: np.lib.format.open_memmap(out / f"{k}.tmp.npy", mode="w+", dtype=np.float16,
                                          shape=(n, 2, fn(x8[:1]).shape[1])) for k, (fn, _) in enc.items()}
    va = np.flatnonzero((meta.split == "val").values)
    probs = {k: np.zeros((len(va), 201), np.float32) for k, (_, m) in enc.items() if m is not None}
    vpos = {i: j for j, i in enumerate(va)}
    t0, B = time.time(), a.batch
    for s in range(0, n, B):
        x = torch.from_numpy(np.ascontiguousarray(arr[s:s + B])).to(device, non_blocking=True)
        b = len(x)
        flat = x.flatten(0, 1)  # (b * 2, 3, H, W): agentview, wrist per row
        with torch.autocast("cuda", dtype=torch.float16):
            for k, (fn, m) in enc.items():
                f = fn(flat).float().view(b, 2, -1)
                feats[k][s:s + b] = f.cpu().numpy().astype(np.float16)
                rows = [i for i in range(s, s + b) if i in vpos]
                if m is not None and rows:
                    lg = m.head(f[[i - s for i in rows]].flatten(1)).float()
                    probs[k][[vpos[i] for i in rows]] = torch.softmax(lg, -1).cpu().numpy()
        if (s // B) % 50 == 0:
            el = time.time() - t0
            print(f"{s + b}/{n} frames, {el:.0f} s, eta {el / (s + b) * (n - s - b) / 60:.0f} min", flush=True)
    for k, f in feats.items():
        f.flush()
        del f
        (out / f"{k}.tmp.npy").rename(out / f"{k}.npy")
    for k, p in probs.items():
        np.save(out / f"probs_{k}_val.npy", p)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


ENCODERS = ["resnet_in", "siglip_pi05", "siglip_web", "resnet_ft", "siglip_pi05_ft"]
T_MAX = 520


def robot_state(meta, out: Path) -> np.ndarray:
    """(n, 8) robot state at each meta frame, cached in <out>/state.npy."""
    import pandas as pd
    path = out / "state.npy"
    if path.exists():
        return np.load(path)
    raw_of = pd.read_csv(DATASET / "episodes.csv").set_index("episode_index").raw
    st = np.zeros((len(meta), 8), np.float32)
    for e, rows in meta.groupby("episode_index"):
        with np.load(raw_of[e]) as z:
            st[rows.index.values] = z["state"][rows.frame_index.values]
    np.save(path, st)
    return st


def poly2(z: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(z.shape[1])
    return np.concatenate([z, (z[:, :, None] * z[:, None, :])[:, iu[0], iu[1]]], 1)


def ridge_r2(P_fit, Y_fit, P_val, Y_val, alpha=1e-3) -> float:
    """Variance-weighted R^2 of a ridge fit (intercept via centring) of Y on P, on val."""
    pm, ym = P_fit.mean(0), Y_fit.mean(0)
    A, Yc = P_fit - pm, Y_fit - ym
    W = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ Yc)
    res = Y_val - (ym + (P_val - pm) @ W)
    return float(1 - (res ** 2).sum() / ((Y_val - Y_val.mean(0)) ** 2).sum())


def window_changes(meta_v, x: np.ndarray, lab: np.ndarray, lag: int = 5):
    """Within-episode changes of x over `lag` grid frames and the label at both ends (val rows in meta order)."""
    import pandas as pd
    q = pd.DataFrame({"e": meta_v.episode_index.values, "t": meta_v.frame_index.values, "x": x, "l": lab})
    q = q.sort_values(["e", "t"])
    g = q.groupby("e")
    q["dx"], q["l0"] = g.x.diff(lag), g.l.shift(lag)
    q = q[q.dx.notna()]
    return q.dx.values, q.l0.values.astype(int), q.l.values.astype(int), q


def analyze(a):
    import json

    import pandas as pd
    from libero_flicker_modes import split as lump_split
    from libero_head_swap import Head, fit, score  # noqa: F401  (Head used inside fit)

    from vl.libero_probe import CENTERS, inner_split
    from vl.libero_resnet import add_subtask
    out = Path(a.out)
    meta = pd.read_parquet(out / "meta.parquet").reset_index(drop=True)
    meta = add_subtask(meta, DATASET)
    st = robot_state(meta, out)
    inner = inner_split(meta)
    keep = {int(x) for x in (DATASET / "subsets" / a.fit_subset).read_text().splitlines() if x.strip() and not x.startswith("#")}
    fit_m = (meta.split == "train").values & ~inner & meta.episode_index.isin(keep).values
    tr, iv, va = np.flatnonzero(fit_m), np.flatnonzero(inner), np.flatnonzero((meta.split == "val").values)
    print(f"fit {len(tr)} frames, inner {len(iv)}, val {len(va)}", flush=True)
    zs = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-6)
    P_state = poly2(zs)
    sub = meta.subtask.values
    P_prog = np.concatenate([np.eye(3)[sub], meta.value.values[:, None], meta.frame_index.values[:, None] / T_MAX], 1)
    P_both = np.concatenate([P_state, P_prog], 1)
    res = {}
    for enc in a.encoders:
        X = np.load(out / f"{enc}.npy").astype(np.float32).reshape(len(meta), -1)
        X -= X[tr].mean(0)
        # (a) PCs of the fit features
        U, S, Vt = np.linalg.svd(X[tr[::4]], full_matrices=False)
        V = Vt[:64].T
        Y = X @ V
        var64 = float((S[:64] ** 2).sum() / (S ** 2).sum())
        r = {"dim": X.shape[1], "pc64_var_share": var64}
        for name, P in (("state", P_state), ("progress", P_prog), ("both", P_both)):
            r[f"r2_{name}"] = ridge_r2(P[tr], Y[tr], P[va], Y[va])
        # (b) linear pot probe
        sd = X[tr].std(0) + 1e-6
        Xs = torch.from_numpy(X / sd)
        y = torch.from_numpy((sub >= 1).astype(np.int64))
        torch.manual_seed(0)
        lin = torch.nn.Linear(Xs.shape[1], 2)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
        best, best_state = np.inf, None
        for ep in range(15):
            for ix in torch.randperm(len(tr)).split(512):
                j = torch.from_numpy(tr)[ix]
                loss = torch.nn.functional.cross_entropy(lin(Xs[j]), y[j])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                c = float(torch.nn.functional.cross_entropy(lin(Xs[iv]), y[iv]))
            if c < best:
                best, best_state = c, {k: v.clone() for k, v in lin.state_dict().items()}
        lin.load_state_dict(best_state)
        with torch.no_grad():
            pv = torch.softmax(lin(Xs[va]), -1)[:, 1].numpy()
        lv = sub[va] >= 1
        dx, l0, l1, _ = window_changes(meta.iloc[va], pv, lv.astype(int))
        ev, steady = (l0 == 0) & (l1 == 1), l0 == l1
        _, _, _, qq = window_changes(meta.iloc[va], (pv > 0.5).astype(float), lv.astype(int))
        r.update(probe_val_acc=float(((pv > 0.5) == lv).mean()), probe_event_signal=float(dx[ev].mean()),
                 probe_steady_noise=float(dx[steady].std()), probe_event_snr=float(dx[ev].mean() / dx[steady].std()),
                 probe_flip_rate=float((qq.dx.values[steady] != 0).mean()), probe_event_windows=int(ev.sum()))
        # (c) value head, 3 seeds
        Xt, bins = torch.from_numpy(X / sd), torch.from_numpy(meta.bin.values.astype(np.int64))
        succ = torch.from_numpy(meta.success.values.astype(np.int64))
        sc = []
        for seed in range(3):
            net, ce_in, ep = fit(Xt, bins, succ, torch.from_numpy(tr), torch.from_numpy(iv), "mlp", seed)
            with torch.no_grad():
                p = torch.softmax(net(Xt[va])[0], -1).numpy()
            pred = meta.iloc[va][["episode_index", "init_state", "frame_index", "T", "success", "value", "bin"]].copy()
            pred["ev"] = p @ CENTERS
            s = score(pred, "ev")
            s.update(inner_ce=ce_in, best_epoch=ep)
            sc.append(s)
        for k in ("change_snr_2s", "milestone", "mae_succ>20s", "mae", "auc10", "inner_ce"):
            v = np.array([s[k] for s in sc])
            r[f"head_{k}"], r[f"head_{k}_sd"] = float(v.mean()), float(v.std(ddof=1))
        res[enc] = r
        print(enc, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    # (d) P2 on the fine-tuned models' probabilities
    for enc in ("resnet_ft", "siglip_pi05_ft"):
        f = out / f"probs_{enc}_val.npy"
        if f.exists():
            pred = meta.iloc[va][["episode_index", "frame_index", "success"]].copy()
            P = np.load(f)
            pred["ev_expert"] = P @ CENTERS
            res.setdefault(enc, {}).update({f"p2_{k}": v for k, v in lump_split(pred, P).items()})
    (out / (f"analysis_{a.tag}.json" if a.tag else "analysis.json")).write_text(json.dumps(res, indent=2))
    t = pd.DataFrame(res).T
    print(t.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--out", default="/iris/u/ankile/karenvo/libero/features/frozen_encoders")
    e.add_argument("--batch", type=int, default=64, help="frames per batch (x 2 cameras)")
    n = sub.add_parser("analyze")
    n.add_argument("--out", default="/iris/u/ankile/karenvo/libero/features/frozen_encoders")
    n.add_argument("--encoders", nargs="+", default=ENCODERS)
    n.add_argument("--fit-subset", default="fit_fixed_fresh.txt",
                   help="episode list in <dataset>/subsets/ (P1b matched run: fit_fixed.txt)")
    n.add_argument("--tag", default="", help="write analysis_<tag>.json instead of analysis.json")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
