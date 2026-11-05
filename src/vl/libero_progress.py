"""Progress probe (docs/libero_plan.md, 2026-10-30): does a representation encode "a moka pot is on the stove"?

Usage (one GPU, this repo's .venv; see scripts/libero_progress.sbatch):
    .venv/bin/python -m vl.libero_progress --prog <scripts/pi05_progress_features.py out dir> \
        [--resnet-run /iris/u/ankile/karenvo/libero/runs/resnet18_18166232]
        -> reports/libero_progress.md, reports/libero_progress.json

Why [ours]: in 3a and 3b every pi05 model reads elapsed time but barely reacts when the first pot lands (milestone
excess +0.02-0.03 vs the ResNet's +0.06 and the oracle's +0.34). The losing models share the frozen backbone; the
ResNet is fine-tuned. If the frozen features do not encode the placed pot, no head on top can fix it (next step:
adapt the backbone); if they do, the value heads are not using it (next step: remove shortcuts, e.g. the clock).

Label: pots_on[t] = number of moka pots satisfying On(pot, cook_region) in image t, i.e. goals[t - 1, :2] (goals[t]
is evaluated after action t; image 0 has none; the same convention as vl.libero_milestone). y = pots_on >= 1.
Frames, splits and early stopping as in 3a (every 4th frame, the 5 inner train states, val = 10 unseen init states).

Sources (each z-scored with train statistics, per channel for token sets):
  pi05 final layer   3a mean pools (all / agentview / wrist), linear + MLP heads
  pi05 V layer l     per-layer prefix values, grid 4 (33 tokens x 256), 3a attention head (what 3b reads)
  pi05 V all layers  the same layers concatenated along the token axis, attention head
  expert h <run>     a trained 3b expert's final query output (what its value head reads), linear + MLP
  ResNet ImageNet    vl.model.ValueModel's frozen ImageNet ResNet-18, pooled 512-d per camera, linear + MLP
  ResNet value       the 3a ResNet-18 V (fine-tuned on value) backbone features, linear + MLP
  object positions   positive control: the sim's pot / end-effector positions (12-d), MLP. Should be ~perfect
  clock              P(y | 2 s time bin) from train: what time alone gives
Heads: vl.libero_probe.fit / AttnProbe unchanged, 2-way CE.

Metrics on val (the first is the one that answers the question; time alone scores 0.5 on it by construction):
  auc_time     time-matched AUC: AUC of P(y) within 2 s time bins, pair-weighted over bins (only frames at the same
               elapsed time are compared, so reading the clock earns nothing)
  auc          plain AUC over all val frames (time helps here: pots are on more often late)
No within-episode "before vs after the milestone" AUC: inside one episode "after" also means "later", so a clock
scores ~1 on it.
95% interval for auc_time: bootstrap over the 10 val init states (1000 resamples).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vl.libero_milestone import milestone
from vl.libero_probe import fit, inner_split, load, logits_of
from vl.metrics import auc

FEATS = Path("/iris/u/ankile/karenvo/libero/features/feats_every4_18163142.npz")
DATASET = Path("/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2")
RUN = Path("/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849")
REPORT = Path(__file__).resolve().parents[2] / "reports"
TBIN = 20  # 2 s
POOLS = {"all": 0, "agentview": 1, "wrist": 2}


def pots_on(goals: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Pots on the stove in image t (= after action t - 1); 0 at t = 0."""
    n = goals[:, :2].sum(1)
    return np.where(frames >= 1, n[np.maximum(frames - 1, 0)], 0)


def time_matched_auc(y: np.ndarray, s: np.ndarray, t: np.ndarray, tbin: int = TBIN) -> float:
    """AUC within time bins, weighted by each bin's positive x negative pairs; NaN if no bin has both classes."""
    num = den = 0.0
    for b in np.unique(t // tbin):
        m = (t // tbin) == b
        n1 = int(y[m].sum())
        n0 = int(m.sum()) - n1
        if n1 and n0:
            num += auc(y[m], s[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def labels(meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """(meta + pots_on, y, milestone; object positions (n, 12)) from the raw episode files."""
    assert (meta.index == np.arange(len(meta))).all()
    raw = pd.read_csv(DATASET / "episodes.csv").set_index("episode_index").raw
    on, obj, ms = np.zeros(len(meta), int), np.zeros((len(meta), 12), np.float32), np.full(len(meta), np.nan)
    for e, g in meta.groupby("episode_index"):
        with np.load(RUN / raw[e]) as z:
            goals, pos = z["goals"], z["object_pos"]
        t = g.frame_index.values
        on[g.index] = pots_on(goals, t)
        obj[g.index] = pos[np.maximum(t - 1, 0)]  # state shown in image t, as for the goals
        ms[g.index] = milestone(goals)
    return meta.assign(pots_on=on, y=(on >= 1).astype(int), milestone=ms), obj


def clock_scores(t_tr: np.ndarray, y_tr: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"b": t_tr // TBIN, "y": y_tr}).groupby("b").y.agg(["sum", "count"])
    p = (df["sum"] + 0.5) / (df["count"] + 1)
    return pd.Series(t_q // TBIN).map(p).fillna(float(y_tr.mean())).values


@torch.no_grad()
def resnet_features(arr: np.ndarray, state_dict: Path | None, device: str) -> np.ndarray:
    """(n, 1024): vl.model.ValueModel backbone features of [agentview, wrist] (ImageNet weights, or a trained V)."""
    from vl.model import ValueModel
    model = ValueModel(["agentview", "wrist"]).to(device).eval()
    if state_dict is not None:
        model.load_state_dict(torch.load(state_dict, map_location=device))
    out = np.zeros((len(arr), 1024), np.float32)
    for i in range(0, len(arr), 256):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + 256])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[i:i + 256] = model.features({"agentview": x[:, 0], "wrist": x[:, 1]}).float().cpu().numpy()
    return out


def zscore(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    if X.ndim == 3:  # token sets: per-channel statistics over train rows and tokens, as 3a
        Xt = X[tr].astype(np.float32).reshape(-1, X.shape[2])
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        return np.concatenate([((X[i:i + 4096].astype(np.float32) - mu) / sd).astype(np.float16)
                               for i in range(0, len(X), 4096)])
    X = X.astype(np.float32)
    return (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)


def score(meta, va, s, rng, n_boot=1000):
    m = meta[va].reset_index(drop=True)
    y, t = m.y.values, m.frame_index.values
    r = {"auc_time": time_matched_auc(y, s, t), "auc": auc(y, s)}
    states = m.init_state.unique()
    idx = {k: np.flatnonzero(m.init_state.values == k) for k in states}
    boots = []
    for _ in range(n_boot):
        ix = np.concatenate([idx[k] for k in rng.choice(states, len(states))])
        boots.append(time_matched_auc(y[ix], s[ix], t[ix]))
    r["auc_time_ci"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prog", required=True, help="scripts/pi05_progress_features.py output dir")
    ap.add_argument("--resnet-run", default="/iris/u/ankile/karenvo/libero/runs/resnet18_18166232")
    ap.add_argument("--heads", nargs="+", default=["linear", "mlp"], help="for vector sources (token sets: attn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPORT / "libero_progress"))
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    F, meta = load(FEATS, DATASET)
    meta, obj = labels(meta)
    inner = inner_split(meta)
    tr = (meta.split == "train").values & ~inner
    va = (meta.split == "val").values
    log(f"{len(meta)} frames; y = 1 on {meta.y[tr].mean():.3f} of fit, {meta.y[va].mean():.3f} of val; "
        f"val episodes with a milestone {meta[va].groupby('episode_index').milestone.first().notna().sum()}")

    # join the pi05_progress_features rows to the 3a rows by (split, episode, frame)
    prog = Path(a.prog)
    pm = np.load(prog / "meta.npz")
    key = lambda s, e, f: pd.MultiIndex.from_arrays([np.asarray(s).astype(str), np.asarray(e), np.asarray(f)])
    pos = key(pm["split"], pm["episode"], pm["frame_index"]).get_indexer(
        key(meta.split, meta.episode, meta.frame_index))
    assert (pos >= 0).all(), "progress features miss some 3a frames"
    layers = [int(x) for x in pm["layers"]]

    sources = {}  # name -> (loader, heads)
    for p, k in POOLS.items():
        sources[f"pi05 final / {p}"] = (lambda k=k: F[:, k], a.heads)
    V = np.load(prog / "v_layers.npy", mmap_mode="r")
    for j, l in enumerate(layers):
        sources[f"pi05 V layer {l}"] = (lambda j=j: np.asarray(V[:, j])[pos], ["attn"])
    sources["pi05 V all layers"] = (lambda: np.asarray(V).reshape(len(V), -1, V.shape[-1])[pos], ["attn"])
    for h in sorted(prog.glob("h_*.npy")):
        sources[f"expert h {h.stem[2:]}"] = (lambda h=h: np.load(h)[pos], a.heads)
    frames = np.load(FEATS.with_suffix(".frames.npy"), mmap_mode="r")
    assert len(frames) == len(meta)
    for name, sd in (("ImageNet", None), ("value", Path(a.resnet_run) / "best.pt")):
        cache = prog / f"resnet_{name.lower()}.npy"
        if not cache.exists():
            np.save(cache, resnet_features(frames, sd, device))
            log(f"ResNet {name} features -> {cache}")
        sources[f"ResNet {name}"] = (lambda cache=cache: np.load(cache), a.heads)
    sources["object positions (control)"] = (lambda: obj, ["mlp"])

    rng = np.random.default_rng(0)
    y = meta.y.values.astype(np.int64)
    res = {"clock": score(meta, va, clock_scores(meta.frame_index.values[tr], y[tr], meta.frame_index.values[va]),
                          rng)}
    log(f"clock: {res['clock']}")
    for name, (load_x, heads) in sources.items():
        X = zscore(load_x(), tr)
        for kind in heads:
            net, inner_ce, ep = fit(kind, X[tr], y[tr], X[inner], y[inner], seed=a.seed, device=device, n_out=2)
            s = torch.softmax(logits_of(net, X[va], device).float(), -1)[:, 1].cpu().numpy()
            r = score(meta, va, s, rng)
            r.update(inner_ce=inner_ce, best_epoch=ep)
            res[f"{name} / {kind}"] = r
            log(f"{name} / {kind}: {r}")
        del X

    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    lines = ["# Progress probe: is a moka pot on the stove?", "",
             "Generated by `python -m vl.libero_progress` (docs/libero_plan.md 2026-10-30). y = at least one pot on "
             "the stove in the image; val = 10 unseen init states, every 4th frame. **auc_time** compares only frames "
             "at the same elapsed time (clock = 0.5 by construction); auc = all frames (time helps).", "",
             "| source / head | auc_time [95% CI] | auc | inner CE |", "|---|---|---|---|"]
    for k, r in res.items():
        ci = r["auc_time_ci"]
        lines.append(f"| {k} | {r['auc_time']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {r['auc']:.3f} | "
                     f"{r.get('inner_ce') or float('nan'):.3f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    log(f"-> {out}.md")


if __name__ == "__main__":
    main()
