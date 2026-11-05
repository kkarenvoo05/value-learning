"""Progress-routed head + a layout-only scene code (docs/libero_plan.md log 2026-10-30, option 2 follow-up) [ours].

Option 2 (scripts/libero_progress_routed.py) routed V through a detached predicted "pot on the stove" bit plus a 2-d
scene code: V reacted at placement (+0.077, 3/3) but lost the layout edge (AUC 10 s 0.62). Here the scene code may
carry the layout but not progress: during training it is penalised for changing within an episode,
    penalty = (mean within-episode variance of the code) / (variance of the code over the batch),
scale-free so shrinking the code does not help. Batches are 32 fit episodes x 8 random frames each so every batch has
within-episode spread. Inference is single-frame and unchanged (the penalty is training-only). Same input (pi05 layer-5
values, 33 grid tokens), BCE on the bit, value CE, 201 bins, 3 seeds, scoring as scripts/libero_head_swap.py.
Arms, no clock (the fair comparison with the single-frame ResNet V):
  P + scene2, lambda 0        option 2's arm under the episode-grouped batches (paired reference)
  P + scene2, lambda 1 / 10   the layout-only penalty
  P + scene8, lambda 10       more room for the layout
Reading rule, set before the result: the penalty works if, at lambda > 0, AUC 10 s rises clearly above lambda 0 (towards
B's 0.77) while the milestone excess stays positive on 3/3 seeds and the change SNR does not fall.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/libero_layout_code.py [--seeds 0 1 2]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl.libero_probe import CENTERS, inner_split  # noqa: E402
from vl.libero_progress import labels  # noqa: E402
from libero_head_swap import FEAT, load_meta, score  # noqa: E402
from libero_oracle_bit import zscore  # noqa: E402
from libero_progress_routed import Routed, predict  # noqa: E402


class LayoutRouted(Routed):
    def forward(self, x, e, return_code=False):
        t = self.proj(x) + self.pos
        q, _ = self.attn(self.query.expand(len(x), -1, -1), t, t)
        r = q[:, 0]
        a = self.prog(r)
        c = self.code(r)
        v = self.out(torch.cat([torch.sigmoid(a).detach(), e, c], -1))
        return (v, a, c) if return_code else (v, a)


def grouped_batches(ep_of_tr: np.ndarray, rng: np.random.Generator, n_ep: int = 32, per: int = 8):
    """Yield (positions into tr, episode id per row): n_ep episodes x `per` frames, one pass over the episodes."""
    by_ep = {e: np.flatnonzero(ep_of_tr == e) for e in np.unique(ep_of_tr)}
    eps = rng.permutation(list(by_ep))
    for i in range(0, len(eps) - n_ep + 1, n_ep):
        idx, gid = [], []
        for k, e in enumerate(eps[i:i + n_ep]):
            f = by_ep[e]
            idx.append(rng.choice(f, size=per, replace=len(f) < per))
            gid.append(np.full(per, k))
        yield np.concatenate(idx), np.concatenate(gid)


def within_ratio(c: torch.Tensor, gid: torch.Tensor, n_ep: int) -> torch.Tensor:
    """Mean within-episode variance of the code / its variance over the batch (summed over code dims)."""
    mean = torch.zeros(n_ep, c.shape[1], device=c.device).index_add_(0, gid, c) / torch.bincount(gid, minlength=n_ep)[:, None]
    within = ((c - mean[gid]) ** 2).mean(0).sum()
    total = c.var(0, unbiased=False).sum() + 1e-6
    return within / total


def fit(net, X, bins, y, ep, tr, iv, seed, epochs, device, lam, passes_per_epoch=16):
    """One pass over the 375 fit episodes = 11 batches x 256 frames; 16 passes ~= 45k frames ~= one epoch of the
    ungrouped scripts (43,087 fit frames), so `epochs` means the same amount of data."""
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    ce, bce = nn.functional.cross_entropy, nn.functional.binary_cross_entropy_with_logits
    E0 = torch.zeros(len(X), 0)
    best, state, best_ep, pen_log = np.inf, None, -1, []
    for epoch in range(epochs):
        net.train()
        pens = []
        for _ in range(passes_per_epoch):
            for pos, gid in grouped_batches(ep[tr], rng):
                j = tr[pos]
                lv, la, c = net(X[j].to(device).float(), E0[j].to(device), return_code=True)
                g = torch.as_tensor(gid, device=device)
                pen = within_ratio(c, g, int(gid.max()) + 1)
                loss = ce(lv, bins[j].to(device)) + bce(la[:, 0], y[j].to(device)) + lam * pen
                opt.zero_grad()
                loss.backward()
                opt.step()
                pens.append(float(pen))
        pen_log.append(float(np.mean(pens)))
        cval = float(ce(predict(net, X, E0, iv, device)[0], bins[iv].to(device)))
        if cval < best:
            best, best_ep, state = cval, epoch, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, best, best_ep, pen_log


@torch.no_grad()
def code_ratio_val(net, X, ep, va, device):
    """The within-episode / total variance ratio of the code on val (whole val set, not batches)."""
    net.eval()
    cs = torch.cat([net(X[va[i:i + 2048]].to(device).float(), torch.zeros(len(va[i:i + 2048]), 0, device=device),
                        return_code=True)[2].cpu() for i in range(0, len(va), 2048)])
    _, gid = np.unique(ep[va], return_inverse=True)
    return float(within_ratio(cs, torch.as_tensor(gid), int(gid.max()) + 1))


ARMS = {"P + scene2, lambda 0": (2, 0.0), "P + scene2, lambda 1": (2, 1.0), "P + scene2, lambda 10": (2, 10.0),
        "P + scene8, lambda 10": (8, 10.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="reports/libero_layout_code.json")
    a = ap.parse_args()
    t0 = time.time()
    meta, _ = labels(load_meta().reset_index(drop=True))
    inner = inner_split(meta)
    tr = np.flatnonzero((meta.split == "train").values & ~inner)
    iv = np.flatnonzero(inner)
    va = np.flatnonzero((meta.split == "val").values)
    ep = meta.episode_index.values.astype(np.int64)
    bins = torch.as_tensor(meta.bin.values.astype(np.int64))
    y = torch.as_tensor(meta.y.values.astype(np.float32))
    centers = torch.as_tensor(CENTERS, dtype=torch.float32)
    layer_ids = list(np.load(FEAT / "meta.npz")["layers"])
    V = np.load(FEAT / "v_layers.npy", mmap_mode="r")
    X = torch.from_numpy(zscore(np.asarray(V[:, layer_ids.index(5)], np.float32), tr).astype(np.float16))
    print(f"[{time.time() - t0:.0f} s] ready on {a.device}", flush=True)
    rows, preds = [], []
    for seed in a.seeds:
        for name in a.arms:
            k, lam = ARMS[name]
            torch.manual_seed(seed)
            net, ice, best_ep, pen_log = fit(LayoutRouted(X.shape[1], X.shape[2], 0, k=k), X, bins, y, ep, tr, iv,
                                             seed, a.epochs, a.device, lam)
            E0 = torch.zeros(len(X), 0)
            lv, la = predict(net, X, E0, va, a.device)
            p = meta.iloc[va][["episode_index", "init_state", "frame_index", "success", "value", "y"]].copy()
            p["ev"] = (torch.softmax(lv, -1).cpu() @ centers).numpy()
            p["p_pot"] = torch.sigmoid(la[:, 0]).cpu().numpy()
            r = {"arm": name, "seed": seed, "inner_ce": ice, "best_epoch": best_ep, **score(p, "ev"),
                 "pot_acc": float(((p.p_pot > 0.5) == (p.y == 1)).mean()),
                 "code_within_ratio_val": code_ratio_val(net, X, ep, va, a.device), "train_penalty": pen_log,
                 "minutes": round((time.time() - t0) / 60, 1)}
            rows.append(r)
            preds.append(p.assign(arm=name, seed=seed))
            print(json.dumps(r), flush=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
    pd.concat(preds).to_parquet(Path(a.out).with_suffix(".val_predictions.parquet"))
    df = pd.DataFrame(rows)
    cols = ["inner_ce", "mae", "mae_succ>20s", "milestone", "change_snr_2s", "auc5", "auc10", "code_within_ratio_val"]
    pd.set_option("display.width", 250)
    print(df.groupby("arm", sort=False)[cols].agg(["mean", "std"]).round(3).to_string())
    print(df.groupby("arm", sort=False).milestone.apply(lambda m: f"{(m > 0).sum()}/{len(m)}").to_string())


if __name__ == "__main__":
    main()
