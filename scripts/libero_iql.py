"""IQL value functions on frozen features, scored against the simulator ground truth [2026-11-11; mentor 11-11: "explore
IQL value function learning in parallel"]. Small: MLPs on cached features, one small GPU (iris-ws-15's RTX 3080) or CPU.

    .venv/bin/python scripts/libero_iql.py --out reports/2026-11-11-iql-frozen [--taus 0.5 0.7 0.9] [--n 20] [--steps 20000]

Question: does IQL's action-conditioned advantage A = Q(s, a) - V(s) track the branching ground truth A* better than
the n-step advantage of a Monte Carlo V (every current label source: corr(A, A*) 0.04-0.14, ceiling 0.69;
reports/2026-11-10-value-vs-gt)?

Data [ours]: the 900 round-0 train rollouts (moka_fixed_fresh), every 4th frame (104,049 frames, the label grid).
  s = frozen ImageNet ResNet-18 features of agentview | wrist (feat_r18imagenet_trainframes.npy, row i = sorted
      train labels row i, as libero_tag_predictability.py) + proprio state (8) + t / 520
  a = the n executed actions a_t .. a_{t+n-1} (7-d each; last action repeated past the end, plus a validity mask)
  transition s_t -> s_{t+n}, reward -n/1040 (the labels' per-step cost); terminal when t + n >= T: return
      -(T - t)/1040 + v_T, v_T = 0 success / -0.5 failure (vl.libero_advantages; same units as the MC targets)
IQL (Kostrikov et al. 2021, Eq. 5-6): L_V = E |tau - 1(Q'(s,a) - V(s) < 0)| (Q'(s,a) - V(s))^2 with Q' = min of two
  target Qs (Polyak 0.005); L_Q = E (r + V(s') - Q(s,a))^2 (gamma = 1, terminals as above). tau = 0.5 is SARSA-like
  (the behaviour policy's V); larger tau leans toward the better actions in the data.
Control: a Monte Carlo V on the same s (MSE to the MC target), so features and IQL are not confounded.
OOF: vl.libero_oof.oof_split, 5 folds (same folds as the ResNet labels); each fold trains on the other folds' episodes
  (inner states unused, fixed steps) and predicts its own.
Reported per arm on the 424 ground-truth points (episode-bootstrap CIs, scripts/libero_value_vs_gt.stat):
  corr(V, V*), corr(A_q, A*) with A_q = Q - V, corr(A_v, A*) with A_v the n-step advantage of V, and on all
  held-out frames: MAE of V vs the MC target, and action sensitivity: corr(A_q, A_q with actions shuffled across
  frames) (near 1 = Q ignores the action, the risk with one action per state).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libero_value_vs_gt import corr, stat  # noqa: E402
from vl.libero_advantages import NORM, V_FAIL, advantage  # noqa: E402
from vl.libero_oof import oof_split  # noqa: E402

TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
FEAT = TABLE / "feat_r18imagenet_trainframes.npy"
ORDER = TABLE / "labels" / "oof_resnet_n20_p30" / "labels.parquet"  # row order of FEAT
POINTS = Path(__file__).resolve().parents[1] / "reports" / "2026-11-10-value-vs-gt" / "points.csv"
KEY = ["episode_index", "frame_index"]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f} s] {m}", flush=True)


def build(n: int, cache: Path | None = None) -> dict:
    """Arrays for transition length n; cached to <cache> (reading 900 raw files over NFS takes ~11 min)."""
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        d = pd.read_parquet(cache.with_suffix(".meta.parquet"))
        return dict(d=d, **{k: z[k] for k in ("S", "A", "nxt", "term", "R", "vT")})
    D = _build(n)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache.with_suffix(".tmp.npz"), **{k: D[k] for k in ("S", "A", "nxt", "term", "R", "vT")})
        D["d"].to_parquet(cache.with_suffix(".meta.parquet"))
        cache.with_suffix(".tmp.npz").rename(cache)
    return D


def _build(n: int) -> dict:
    d = pd.read_parquet(ORDER).sort_values(KEY).reset_index(drop=True)[KEY + ["V"]].rename(columns={"V": "V_resnet"})
    eps = pd.read_csv(TABLE / "episodes.csv").set_index("episode_index")
    ft = pd.read_parquet(TABLE / "frame_targets.parquet", columns=KEY + ["value"])
    d = d.merge(ft, on=KEY, how="left", validate="one_to_one")
    assert d.value.notna().all()
    d["T"] = d.episode_index.map(eps.policy_steps - 1).to_numpy()
    d["success"] = d.episode_index.map(eps.success).to_numpy()
    d["init_state"] = d.episode_index.map(eps.init_state).to_numpy()
    d["split"] = "train"
    img = np.load(FEAT, mmap_mode="r")
    assert len(img) == len(d), (len(img), len(d))
    st = np.zeros((len(d), 8), np.float32)
    act = np.zeros((len(d), n, 7), np.float32)
    msk = np.zeros((len(d), n), np.float32)
    for e, idx in d.groupby("episode_index").indices.items():
        with np.load(eps.raw[e]) as z:
            s, a = z["state"], z["actions"]
        f = d.frame_index.to_numpy()[idx]
        st[idx] = s[f]
        j = f[:, None] + np.arange(n)[None]
        msk[idx] = j < len(a)
        act[idx] = a[np.minimum(j, len(a) - 1)]
    S = np.concatenate([np.asarray(img, np.float32), st, (d.frame_index.to_numpy() / 520.0)[:, None]], 1).astype(np.float32)
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


def mlp(k: int, width: int = 256) -> nn.Module:
    return nn.Sequential(nn.Linear(k, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))


class Std:
    def __init__(self, x: np.ndarray, device):
        self.mu = torch.as_tensor(x.mean(0), device=device)
        self.sd = torch.as_tensor(x.std(0) + 1e-6, device=device)

    def __call__(self, x):
        return (x - self.mu) / self.sd


def train_iql(D, tr, tau, steps, device, seed, bs=1024, lr=3e-4, polyak=0.005):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    S = torch.as_tensor(D["S"], device=device)
    A = torch.as_tensor(D["A"], device=device)
    R = torch.as_tensor(D["R"], device=device)
    nxt = torch.as_tensor(D["nxt"], device=device)
    term = torch.as_tensor(D["term"], device=device)
    ns, na = Std(D["S"][tr], device), Std(D["A"][tr], device)
    q1, q2, v = mlp(S.shape[1] + A.shape[1]).to(device), mlp(S.shape[1] + A.shape[1]).to(device), mlp(S.shape[1]).to(device)
    t1, t2 = mlp(S.shape[1] + A.shape[1]).to(device), mlp(S.shape[1] + A.shape[1]).to(device)  # target Qs
    t1.load_state_dict(q1.state_dict()), t2.load_state_dict(q2.state_dict())
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    ov = torch.optim.Adam(v.parameters(), lr=lr)
    trt = torch.as_tensor(tr, device=device)
    for step in range(steps):
        ix = trt[torch.randint(len(trt), (bs,), generator=g).to(device)]
        s, a = ns(S[ix]), na(A[ix])
        sa = torch.cat([s, a], 1)
        with torch.no_grad():
            qt = torch.minimum(t1(sa), t2(sa)).squeeze(-1)
        u = qt - v(s).squeeze(-1)
        lv = (torch.abs(tau - (u < 0).float()) * u ** 2).mean()
        ov.zero_grad(set_to_none=True), lv.backward(), ov.step()
        with torch.no_grad():
            y = R[ix] + torch.where(term[ix], torch.zeros_like(R[ix]), v(ns(S[nxt[ix]])).squeeze(-1))
        lq = ((q1(sa).squeeze(-1) - y) ** 2).mean() + ((q2(sa).squeeze(-1) - y) ** 2).mean()
        oq.zero_grad(set_to_none=True), lq.backward(), oq.step()
        with torch.no_grad():
            for t_, q_ in ((t1, q1), (t2, q2)):
                for pt, pq in zip(t_.parameters(), q_.parameters()):
                    pt.mul_(1 - polyak).add_(polyak * pq)
        if (step + 1) % 5000 == 0:
            log(f"    tau {tau} step {step + 1}: L_V {lv.item():.5f} L_Q {lq.item():.5f}")
    return lambda Sx, Ax: _eval_iql(q1, q2, v, ns, na, Sx, Ax, device)


@torch.no_grad()
def _eval_iql(q1, q2, v, ns, na, Sx, Ax, device, bs=8192):
    Q, V = [], []
    for i in range(0, len(Sx), bs):
        s = ns(torch.as_tensor(Sx[i:i + bs], device=device))
        a = na(torch.as_tensor(Ax[i:i + bs], device=device))
        sa = torch.cat([s, a], 1)
        Q.append(torch.minimum(q1(sa), q2(sa)).squeeze(-1).cpu())
        V.append(v(s).squeeze(-1).cpu())
    return torch.cat(Q).numpy(), torch.cat(V).numpy()


def train_mc(D, tr, steps, device, seed, bs=1024, lr=3e-4):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    S = torch.as_tensor(D["S"], device=device)
    y = torch.as_tensor(D["d"].value.to_numpy(np.float32), device=device)
    ns = Std(D["S"][tr], device)
    v = mlp(S.shape[1]).to(device)
    o = torch.optim.Adam(v.parameters(), lr=lr)
    trt = torch.as_tensor(tr, device=device)
    for step in range(steps):
        ix = trt[torch.randint(len(trt), (bs,), generator=g).to(device)]
        loss = ((v(ns(S[ix])).squeeze(-1) - y[ix]) ** 2).mean()
        o.zero_grad(set_to_none=True), loss.backward(), o.step()

    @torch.no_grad()
    def pred(Sx, bs=8192):
        return torch.cat([v(ns(torch.as_tensor(Sx[i:i + bs], device=device))).squeeze(-1).cpu()
                          for i in range(0, len(Sx), bs)]).numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--n", type=int, default=20, help="transition length in frames (multiple of 4; labels use 20)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)), help="subset for a smoke test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="/iris/u/karenvo/iql_cache", help="built arrays, one file per n")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {device}{' ' + torch.cuda.get_device_name() if device == 'cuda' else ''}; building data (n = {a.n})")
    D = build(a.n, Path(a.cache_dir) / f"iql_round0_n{a.n}.npz")
    d = D["d"]
    log(f"{len(d)} frames, {d.episode_index.nunique()} episodes, s {D['S'].shape[1]}-d, a {D['A'].shape[1]}-d")
    arms = [f"iql_tau{t:g}" for t in a.taus] + ["mc"]
    P = {f"{c}:{arm}": np.full(len(d), np.nan) for arm in arms for c in (("V", "Q", "Qshuf") if arm != "mc" else ("V",))}
    rng = np.random.default_rng(a.seed)
    for k in a.folds:
        held, _ = oof_split(d, k)
        ho, tr = np.flatnonzero(held), np.flatnonzero(~held)
        log(f"fold {k}: fit {d.episode_index[tr].nunique()} episodes, predict {d.episode_index[ho].nunique()}")
        Ash = D["A"][ho][rng.permutation(len(ho))]
        for tau in a.taus:
            f = train_iql(D, tr, tau, a.steps, device, a.seed + k)
            q, v = f(D["S"][ho], D["A"][ho])
            qs, _ = f(D["S"][ho], Ash)
            arm = f"iql_tau{tau:g}"
            P[f"V:{arm}"][ho], P[f"Q:{arm}"][ho], P[f"Qshuf:{arm}"][ho] = v, q, qs
        P["V:mc"][ho] = train_mc(D, tr, a.steps, device, a.seed + k)(D["S"][ho])
    done = ~np.isnan(P[f"V:{arms[-1]}"])
    pr = d[KEY + ["value", "V_resnet", "T", "success"]].copy()
    for c, x in P.items():
        pr[c] = x
    t, T, ep = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.episode_index.to_numpy()
    for arm in arms + ["resnet"]:
        V = pr[f"V:{arm}"].to_numpy() if arm != "resnet" else pr.V_resnet.to_numpy()
        pr[f"Av:{arm}"] = advantage(np.nan_to_num(V), t, T, D["vT"], ep, a.n)[0]
        if arm.startswith("iql"):
            pr[f"Aq:{arm}"], pr[f"Aqshuf:{arm}"] = pr[f"Q:{arm}"] - V, pr[f"Qshuf:{arm}"] - V
    pr = pr[done].reset_index(drop=True)
    pr.to_parquet(out / "oof_predictions.parquet")

    pts = pd.read_csv(POINTS)[["ep", "t", "v", "A_g"]].rename(columns={"ep": "episode_index", "t": "frame_index"})
    m = pts.merge(pr, on=KEY, how="inner").assign(ep=lambda x: x.episode_index)  # stat() bootstraps over `ep`
    f = lambda s: f"{s['est']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
    rows, lines = [], [f"points matched {len(m)} of {len(pts)} (folds {a.folds}); n = {a.n}, steps {a.steps}", "",
                       "| arm | corr(V, V*) | corr(Q-V, A*) | corr(n-step A of V, A*) | MAE V vs MC | corr(Q-V, shuffled-a) |",
                       "|---|---|---|---|---|---|"]
    for arm in arms + ["resnet"]:
        V = "V_resnet" if arm == "resnet" else f"V:{arm}"
        r = {"arm": arm, "corr_V_V*": stat(m, lambda x: corr(x[V], x.v)),
             "corr_Av_A*": stat(m, lambda x: corr(x[f"Av:{arm}"], x.A_g)),
             "mae_V_mc": float(np.abs(pr[V] - pr.value).mean())}
        if arm.startswith("iql"):
            r["corr_Aq_A*"] = stat(m, lambda x: corr(x[f"Aq:{arm}"], x.A_g))
            r["corr_Aq_shuffled"] = float(corr(pr[f"Aq:{arm}"], pr[f"Aqshuf:{arm}"]))
        rows.append(r)
        lines.append(f"| {arm} | {f(r['corr_V_V*'])} | {f(r['corr_Aq_A*']) if 'corr_Aq_A*' in r else '—'} | "
                     f"{f(r['corr_Av_A*'])} | {r['mae_V_mc']:.3f} | "
                     f"{r['corr_Aq_shuffled']:.2f} |" if "corr_Aq_shuffled" in r else
                     f"| {arm} | {f(r['corr_V_V*'])} | — | {f(r['corr_Av_A*'])} | {r['mae_V_mc']:.3f} | — |" if arm == "mc" else
                     f"| {arm} (fine-tuned ResNet OOF, reference) | {f(r['corr_V_V*'])} | — | {f(r['corr_Av_A*'])} | "
                     f"{r['mae_V_mc']:.3f} | — |")
    (out / "summary.json").write_text(json.dumps({"args": vars(a), "points": int(len(m)), "arms": rows}, indent=2))
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
