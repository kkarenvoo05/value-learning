"""IQL advantage labels for one round of an AG chain, from that chain's data only [2026-11-16; mentor: "get the
IQL-based version of AG tested too, especially as we start running many rounds"; Karen: "set up the 100 iql ag run"].

The policy side is unchanged (frozen pi0.5 + residual head trained on positive / negative tags). Only the tags change:
  - critic: IQL (Kostrikov et al. 2021) exactly as scripts/libero_iql.py (twin Q + expectile V, 2 x 256 MLPs,
    s = frozen ImageNet ResNet-18 features of agentview | wrist + 8-d proprio + t / 520, a = the next n = 20 executed
    actions + validity mask, n-step transition on the every-4th-frame grid, reward -n/1040, terminal return
    -(T - t)/1040 + {0 success, -0.5 failure}, gamma 1, Polyak 0.005), tau 0.9 [AG paper, OGBench setting],
    20k steps.
  - data: every train episode of --table, i.e. the chain's own rounds (its round-0 subsample + its later rounds).
    The critic is trained FROM SCRATCH each round on all of them; no model ever saw the other base rollouts.
    The image features come from ImageNet weights only (no task data).
  - out-of-fold: 5 folds by episode (rng 0); each frame's Q, V come from the fold model that never saw its episode.
  - tag: A = Q(s, a_data) - V(s); positive = top 40% of A within each round [RECAP share, as vb40; ours: no value
    binning, since Q - V is already relative to the state's V]. Every round is relabelled each time (the critic
    changes), unlike the vb40 chains, which keep earlier rounds' V.
Writes <out>/labels.parquet (episode_index, frame_index, positive, A, Q, V, term, round) and summary.json.
Image features are cached per raw file under --feat-cache (keyed by the raw path, so ids that repeat across chains
never collide).

    .venv/bin/python scripts/libero_iql_round_labels.py --table <tables>/moka_agr_b100iql_r1 --out <label dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libero_iql import train_iql  # noqa: E402
from vl.libero_advantages import NORM, V_FAIL  # noqa: E402
from vl.libero_resnet import table_meta  # noqa: E402

KEY = ["episode_index", "frame_index"]
FEAT_CACHE = Path("/iris/u/ankile/karenvo/libero/feat_cache/r18imagenet_every4")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f} s] {m}", flush=True)


class Feat:
    """Pooled ImageNet ResNet-18 features (1024 = agentview | wrist), fp16, as libero_tag_predictability.image_features."""

    def __init__(self, device):
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.net, self.device = net.to(device).eval().half(), device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1).half()
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1).half()

    @torch.no_grad()
    def __call__(self, img: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(img), 256):
            x = np.stack([img[s:s + 256], wrist[s:s + 256]], 1).transpose(0, 1, 4, 2, 3)  # (b, 2, 3, 224, 224)
            x = torch.from_numpy(np.ascontiguousarray(x)).to(self.device).view(-1, 3, 224, 224)
            x = x.half().div_(255).sub_(self.mean).div_(self.std)
            out.append(self.net(x).view(-1, 1024).float().cpu().numpy().astype(np.float16))
        return np.concatenate(out)


def episode_features(raw: str, frames: np.ndarray, feat: Feat, cache: Path) -> np.ndarray:
    f = cache / (hashlib.sha1(raw.encode()).hexdigest()[:20] + ".npz")
    if f.exists():
        with np.load(f) as z:
            if np.array_equal(z["frames"], frames):
                return z["feat"]
    with np.load(raw) as z:
        x = feat(z["image"][frames], z["wrist_image"][frames])
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, feat=x, frames=frames, raw=raw)
    tmp.rename(f)
    return x


def build(table: Path, n: int, device, cache: Path) -> dict:
    """IQL arrays for every train episode of the table on the every-4th-frame grid (libero_iql._build, any table)."""
    eps = pd.read_csv(table / "episodes.csv", dtype={"goals_end": str})
    if "round" not in eps.columns:
        eps["round"] = 1
    eps = eps[eps.split == "train"].set_index("episode_index")
    d = table_meta(table, 4)
    d = d[d.split == "train"].sort_values(KEY).reset_index(drop=True)
    d["round"] = d.episode_index.map(eps["round"]).to_numpy()
    feat = Feat(device)
    img = np.zeros((len(d), 1024), np.float16)
    st = np.zeros((len(d), 8), np.float32)
    act = np.zeros((len(d), n, 7), np.float32)
    msk = np.zeros((len(d), n), np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    groups = d.groupby("episode_index").indices
    for i, (e, idx) in enumerate(groups.items()):
        f = d.frame_index.to_numpy()[idx]
        img[idx] = episode_features(eps.raw[e], f, feat, cache)
        with np.load(eps.raw[e]) as z:
            s, a = z["state"], z["actions"]
        st[idx] = s[f]
        j = f[:, None] + np.arange(n)[None]
        msk[idx] = j < len(a)
        act[idx] = a[np.minimum(j, len(a) - 1)]
        if i % 100 == 0:
            log(f"  features / arrays {i + 1}/{len(groups)} episodes")
    S = np.concatenate([img.astype(np.float32), st, (d.frame_index.to_numpy() / 520.0)[:, None]], 1).astype(np.float32)
    A = np.concatenate([act.reshape(len(d), -1), msk], 1)
    t, T, ep = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.episode_index.to_numpy()
    k = n // 4
    term = t + n >= T
    nxt = np.arange(len(d)) + k
    ok = np.r_[ep[k:] == ep[:-k], np.zeros(k, bool)]
    assert (ok | term).all(), "interior lookahead must stay on the episode's grid"
    nxt = np.where(term, 0, nxt)
    vT = np.where(d.success.to_numpy() == 1, 0.0, V_FAIL)
    R = np.where(term, -(T - t) / NORM + vT, -n / NORM).astype(np.float32)
    return dict(d=d, S=S, A=A, nxt=nxt, term=term, R=R, vT=vT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat-cache", default=str(FEAT_CACHE))
    a = ap.parse_args()
    table, out = Path(a.table), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {device}; table {table}")
    D = build(table, a.n, device, Path(a.feat_cache))
    d = D["d"]
    ep_ids = np.sort(d.episode_index.unique())
    fold_of = dict(zip(ep_ids, np.random.default_rng(a.seed).permutation(len(ep_ids)) % a.folds))
    fold = d.episode_index.map(fold_of).to_numpy()
    log(f"{len(d)} frames, {len(ep_ids)} episodes, rounds {sorted(d['round'].unique().tolist())}")
    Q, V = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for k in range(a.folds):
        ho, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        f = train_iql(D, tr, a.tau, a.steps, device, a.seed + k)
        Q[ho], V[ho] = f(D["S"][ho], D["A"][ho])
        log(f"fold {k}: fit {len(set(d.episode_index[tr]))} episodes, predict {len(set(d.episode_index[ho]))}")
    assert np.isfinite(Q).all() and np.isfinite(V).all()
    A = Q - V
    pos = np.zeros(len(d), np.int8)
    for r in np.unique(d["round"]):
        m = (d["round"] == r).to_numpy()
        pos[m] = A[m] >= np.quantile(A[m], 1 - a.share)
    lab = pd.DataFrame({"episode_index": d.episode_index.to_numpy(), "frame_index": d.frame_index.to_numpy(),
                        "positive": pos, "A": A, "Q": Q, "V": V, "term": D["term"], "round": d["round"].to_numpy()})
    lab.to_parquet(out / "labels.parquet")
    succ = d.success.to_numpy()
    by_round = {str(int(r)): {"episodes": int(d.episode_index[d["round"] == r].nunique()),
                              "frames": int((d["round"] == r).sum()),
                              "positive": float(pos[(d["round"] == r).to_numpy()].mean()),
                              "positive_success_eps": float(pos[((d["round"] == r) & (succ == 1)).to_numpy()].mean()),
                              "positive_failure_eps": float(pos[((d["round"] == r) & (succ == 0)).to_numpy()].mean())}
                for r in np.unique(d["round"])}
    s = {"table": str(table), "critic": "IQL (libero_iql.train_iql), trained from scratch on this table's train episodes",
         "tau": a.tau, "n": a.n, "steps": a.steps, "folds": a.folds, "share": a.share, "seed": a.seed,
         "rule": f"positive = top {a.share:.0%} of A = Q - V within each round (out-of-fold)",
         "frames": int(len(lab)), "positive": float(pos.mean()),
         "corr_V_mc": float(np.corrcoef(V, d.value.to_numpy())[0, 1]),
         "mae_V_mc": float(np.abs(V - d.value.to_numpy()).mean()), "by_round": by_round}
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
