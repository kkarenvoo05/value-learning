"""Action-imagination gate G1 (reports/2026-11-11-advantage-guidance/report.md §5; Karen 2026-11-12: "set up G1 ...
and run"): does a chunk-conditioned critic Q(s, a_{t:t+5}) rank candidate action chunks *within a state*?

    .venv/bin/python scripts/libero_iql_g1.py --out reports/2026-11-12-iql-g1 [--seeds 0 1] [--steps 20000]

Test (held out: val init states, never in any training set): the three steer probes (scripts/libero_steer_probe.py;
base pi05_libero, one base episode per val state, branch points t = 40 130 220 310). At each of the 120 points,
8 candidate chunks were sampled (noise scale 1.0 / 1.5 / 2.0 per probe), each candidate's first 5 actions executed,
then the base policy continued, 2 times per candidate -> p_c = success share (0, 0.5, 1).
  rank      within-state Spearman(score_c, p_c) over the 8 candidates, averaged over points where p_c varies
  pick      p[argmax score] - mean_c p_c, averaged over all points (what best-of-8 by this critic would gain)
  ceiling   a perfect critic's observable within-state corr is ~ sqrt(var_between / var_observed) with
            var_between = the probe's unbiased between-candidate variance (2 continuations only) [ours]
CIs: bootstrap over the 30 base episodes (points within an episode share a scene). Random scores are the null (0).

Critics (trained on train states only; s = frozen ImageNet ResNet-18 agentview | wrist features + proprio + t / 520,
as scripts/libero_iql.py; a = the 5 executed actions a_t..a_{t+4} = one decision [plan: chunk-conditioned];
transition s_t -> s_{t+20} on the every-4th-frame grid, reward -20/1040, terminal outcome 0 / -0.5, gamma 1 [ours:
the labels' units, so Q is "this chunk, then the behaviour policy for the rest of the 2 s"]):
  iql_tau{0.5,0.7,0.9}  IQL (Kostrikov et al. 2021) as libero_iql.train_iql, 2 x 128 heads [plan: small head]
  mcq                   [ours, control] Q(s, a) regressed on the Monte Carlo return-to-go (no bootstrapping)
  v_after               [ours, diagnostic, not usable for imagination] rank by IQL tau 0.9 V of the simulator's
                        state right after the candidate's 5 actions (the probe saved those frames)
Data sets [plan: "base rollouts plus the exploration rollouts and ... round 2 once collected"]:
  base  the 900 round-0 base rollouts (moka_fixed_fresh train)
  all   base + 450 exploration rollouts (moka_explore: noise x2.0 on chunks, 30 train states) + the 100 AG round-1
        rollouts (moka_agr_r1 round 2: the guided residual policy, w 9.26)
  The branching continuations of scripts/libero_branch_value.py cannot be used: they saved outcomes, not actions.
Seeds: each (data, critic) is fit with --seeds; scores are averaged over seeds before ranking [ours].
Writes <out>/{table.md, summary.json, scores.parquet}.
"""

from __future__ import annotations

import argparse
import functools
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
import libero_iql as L  # noqa: E402
from vl.libero_advantages import NORM, V_FAIL  # noqa: E402
from vl.libero_resnet import table_meta  # noqa: E402

TABLES = Path("/iris/u/ankile/karenvo/libero/tables")
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
PROBES = {1.0: "steer_probe_libero_10_18296012", 1.5: "steer_v2_scale1p5_libero_10_18329175",
          2.0: "steer_v2_scale2p0_libero_10_18329176"}
N, N_ACT, AGR_OFFSET = 20, 5, 100000  # agr_r1 round-2 ids (200000+) collide with moka_explore's; shift them
KEY = L.KEY
log = L.log


@functools.lru_cache(maxsize=None)
def _r18(device):
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    return net.to(device).eval().half()


@torch.no_grad()
def r18_features(pairs: np.ndarray, device) -> np.ndarray:
    """(n, 1024) fp16, agentview | wrist, exactly libero_tag_predictability.image_features' preprocessing.
    Input: uint8 (n, 2, 224, 224, 3) (HWC, as the raw files store it)."""
    net = _r18(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1).half()
    x_all = np.asarray(pairs)
    out = np.zeros((len(x_all), 1024), np.float16)
    for s in range(0, len(x_all), 256):
        x = torch.from_numpy(np.ascontiguousarray(x_all[s:s + 256].transpose(0, 1, 4, 2, 3))).to(device)
        x = x.reshape(-1, 3, 224, 224).half().div_(255).sub_(mean).div_(std)
        out[s:s + 256] = net(x).view(-1, 1024).float().cpu().numpy().astype(np.float16)
    return out


def arrays(d: pd.DataFrame, eps: pd.DataFrame, img: np.ndarray, n: int = N) -> dict:
    """libero_iql._build for any table: d = every-4th-frame rows (KEY, value), sorted, img = their features."""
    d = d.sort_values(KEY).reset_index(drop=True)
    d["T"] = d.episode_index.map(eps.policy_steps - 1).to_numpy()
    d["success"] = d.episode_index.map(eps.success).to_numpy()
    d["init_state"] = d.episode_index.map(eps.init_state).to_numpy()
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


def cached(path: Path, fn) -> dict:
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return dict(d=pd.read_parquet(path.with_suffix(".meta.parquet")), **{k: z[k] for k in ("S", "A", "nxt", "term", "R", "vT")})
    D = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".tmp.npz"), **{k: D[k] for k in ("S", "A", "nxt", "term", "R", "vT")})
    D["d"].to_parquet(path.with_suffix(".meta.parquet"))
    path.with_suffix(".tmp.npz").rename(path)
    return D


def explore_set() -> dict:
    """The 450 noise-x2.0 exploration rollouts; features = moka_explore's cache (rows = sorted explore_vb40 rows,
    checked 2026-11-12: its base rows equal moka_fixed_fresh's features to fp16 rounding)."""
    t = TABLES / "moka_explore"
    eps = pd.read_csv(t / "episodes.csv").set_index("episode_index")
    order = pd.read_parquet(t / "labels" / "explore_vb40" / "labels.parquet")[KEY]
    assert (order.values == order.sort_values(KEY).values).all()
    keep = order.episode_index.isin(eps.index[eps.run.str.startswith("explore_")]).to_numpy()
    ft = pd.read_parquet(t / "frame_targets.parquet", columns=KEY + ["value"])
    d = order[keep].merge(ft, on=KEY, how="left", validate="one_to_one").reset_index(drop=True)
    assert d.value.notna().all()
    img = np.load(t / "feat_r18imagenet_trainframes.npy", mmap_mode="r")[np.flatnonzero(keep)]
    return arrays(d, eps, img)


def agr_set(device) -> dict:
    """The 100 AG round-1 rollouts (moka_agr_r1 round 2); features computed here from the raw frames."""
    t = TABLES / "moka_agr_r1"
    eps = pd.read_csv(t / "episodes.csv")
    new = eps[(eps["round"] == 2) & (eps.split == "train")]
    meta = table_meta(t, 4)
    meta = meta[meta.episode_index.isin(new.episode_index)].sort_values(KEY).reset_index(drop=True)
    eps = eps.set_index("episode_index")
    feats = np.zeros((len(meta), 1024), np.float16)
    for e, idx in meta.groupby("episode_index").indices.items():
        with np.load(eps.raw[e]) as z:
            f = meta.frame_index.to_numpy()[idx]
            feats[idx] = r18_features(np.stack([z["image"][f], z["wrist_image"][f]], 1), device)
    D = arrays(meta[KEY + ["value"]].copy(), eps, feats)
    D["d"]["episode_index"] += AGR_OFFSET
    return D


def concat(parts: list[dict]) -> dict:
    out, off = {k: [] for k in ("S", "A", "nxt", "term", "R", "vT", "d")}, 0
    for p in parts:
        for k in ("S", "A", "term", "R", "vT", "d"):
            out[k].append(p[k])
        out["nxt"].append(np.where(p["term"], 0, p["nxt"] + off))
        off += len(p["S"])
    D = {k: (pd.concat(v, ignore_index=True) if k == "d" else np.concatenate(v)) for k, v in out.items()}
    assert not D["d"].duplicated(KEY).any()
    return D


def first_actions(D: dict, n_act: int = N_ACT) -> dict:
    """A = first n_act of the n cached actions + their mask (cache layout: n*7 actions, then n mask entries)."""
    A = D["A"]
    n = (A.shape[1]) // 8
    return {**D, "A": np.concatenate([A[:, :n_act * 7], A[:, n * 7:n * 7 + n_act]], 1)}


def probe_points(device) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Rows = (probe, episode, point, candidate); S (state at the branch point), A (candidate's 5 actions + mask),
    S_after (state after those 5 actions)."""
    rows, S, A, Sa = [], [], [], []
    for scale, run in PROBES.items():
        eps = pd.read_csv(RUNS / run / "episodes.csv")
        for r in eps.itertuples():
            z = np.load(RUNS / run / "probe" / r.file)
            rp = list(z["rp_t"])
            for i, t in enumerate(z["br_t"]):
                j = rp.index(int(t))
                f = r18_features(np.stack([z["rp_image"][j], z["rp_wrist_image"][j]])[None], device)[0]
                s = np.concatenate([f.astype(np.float32), z["rp_state"][j].astype(np.float32), [t / 520.0]])
                fa = r18_features(np.stack([z["br_after_image"][i], z["br_after_wrist"][i]], 1), device)
                for c in range(z["br_chunks"].shape[1]):
                    S.append(s)
                    A.append(np.concatenate([z["br_chunks"][i, c, :N_ACT].reshape(-1), np.ones(N_ACT)]).astype(np.float32))
                    Sa.append(np.concatenate([fa[c].astype(np.float32), z["br_after_state"][i, c].astype(np.float32),
                                              [(t + N_ACT) / 520.0]]))
                    rows.append(dict(scale=scale, run=run, ep=f"{scale}:{r.episode}", init_state=r.init_state,
                                     t=int(t), point=f"{scale}:{r.episode}:{t}", cand=c,
                                     p=float(z["br_success"][i, c].mean())))
    return pd.DataFrame(rows), np.stack(S).astype(np.float32), np.stack(A), np.stack(Sa).astype(np.float32)


def train_mcq(D, tr, steps, device, seed, width=128, bs=1024, lr=3e-4):
    """[ours, control] Q(s, a) -> Monte Carlo return-to-go, MSE."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    S, A = torch.as_tensor(D["S"], device=device), torch.as_tensor(D["A"], device=device)
    y = torch.as_tensor(D["d"].value.to_numpy(np.float32), device=device)
    ns, na = L.Std(D["S"][tr], device), L.Std(D["A"][tr], device)
    q = L.mlp(S.shape[1] + A.shape[1], width).to(device)
    o = torch.optim.Adam(q.parameters(), lr=lr)
    trt = torch.as_tensor(tr, device=device)
    for _ in range(steps):
        ix = trt[torch.randint(len(trt), (bs,), generator=g).to(device)]
        loss = ((q(torch.cat([ns(S[ix]), na(A[ix])], 1)).squeeze(-1) - y[ix]) ** 2).mean()
        o.zero_grad(set_to_none=True), loss.backward(), o.step()

    @torch.no_grad()
    def pred(Sx, Ax):
        s, a = ns(torch.as_tensor(Sx, device=device)), na(torch.as_tensor(Ax, device=device))
        return q(torch.cat([s, a], 1)).squeeze(-1).cpu().numpy()
    return pred


def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(a.corr(b)) if a.std() > 0 and b.std() > 0 else np.nan


def metrics(P: pd.DataFrame, col: str) -> dict:
    """Per point: within-state Spearman and the best-of-8 pick gain; means over points."""
    g = P.groupby("point")
    per = pd.DataFrame({"ep": g.ep.first(), "scale": g.scale.first(),
                        "rho": g.apply(lambda x: spearman(x[col], x.p)),
                        "pick": g.apply(lambda x: x.p.iloc[int(np.argmax(x[col].to_numpy()))] - x.p.mean()),
                        "qsd": g[col].std()})
    return per


def boot_mean(per: pd.DataFrame, c: str, n=2000, seed=0):
    eps = per.ep.unique()
    gr = {e: x[c].dropna().to_numpy() for e, x in per.groupby("ep")}
    rng = np.random.default_rng(seed)
    v = [np.concatenate([gr[e] for e in rng.choice(eps, len(eps))]) for _ in range(n)]
    v = np.array([x.mean() for x in v if len(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--data", nargs="+", default=["base", "all"])
    ap.add_argument("--cache-dir", default="/iris/u/karenvo/iql_cache")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cd = Path(a.cache_dir)
    log(f"device {dev}{' ' + torch.cuda.get_device_name() if dev == 'cuda' else ''}")
    P, Sp, Ap, Sap = probe_points(dev)
    log(f"probe: {P.point.nunique()} points x 8 candidates, {P.ep.nunique()} base episodes, val states "
        f"{sorted(P.init_state.unique())}")
    base = L.build(N, cd / f"iql_round0_n{N}.npz")
    sets = {"base": [base]}
    if "all" in a.data:
        ex = cached(cd / f"iql_explore_n{N}.npz", explore_set)
        log(f"explore: {len(ex['S'])} frames, {ex['d'].episode_index.nunique()} episodes")
        ag = cached(cd / f"iql_agr_r1_n{N}.npz", lambda: agr_set(dev))
        log(f"agr round 1: {len(ag['S'])} frames, {ag['d'].episode_index.nunique()} episodes")
        sets["all"] = [base, ex, ag]
    for parts in sets.values():  # held out: no val init state in any training set (fresh layouts are -1)
        for p in parts:
            assert not set(p["d"].init_state) & set(P.init_state), "a probe (val) state is in the training data"
    res = {"args": vars(a), "points": int(P.point.nunique()), "arms": []}
    for name in a.data:
        D = first_actions(concat(sets[name]) if len(sets[name]) > 1 else sets[name][0])
        tr = np.arange(len(D["S"]))
        log(f"data {name}: {len(tr)} frames, {D['d'].episode_index.nunique()} episodes, a {D['A'].shape[1]}-d")
        for arm in [f"iql_tau{t:g}" for t in a.taus] + ["mcq"]:
            qs, vs = [], []
            for sd in a.seeds:
                if arm == "mcq":
                    qs.append(train_mcq(D, tr, a.steps, dev, sd, a.width)(Sp, Ap))
                else:
                    f = L.train_iql(D, tr, float(arm.split("tau")[1]), a.steps, dev, sd, width=a.width)
                    qs.append(f(Sp, Ap)[0])
                    vs.append(f.V(Sap))
                log(f"  {name} {arm} seed {sd} done")
            P[f"{name}:{arm}"] = np.mean(qs, 0)
            for k, q in enumerate(qs):
                P[f"{name}:{arm}:s{a.seeds[k]}"] = q
            if arm == "iql_tau0.9":
                P[f"{name}:v_after"] = np.mean(vs, 0)
    P["random"] = np.random.default_rng(0).random(len(P))
    P.to_parquet(out / "scores.parquet")

    # ceiling from the probe's own between-candidate variance (2 continuations per candidate)
    g = P.groupby("point").p
    var_obs = g.var(ddof=1)
    # Var(p_hat) = p(1-p)/R, unbiasedly estimated by p_hat(1-p_hat)/(R-1); R = 2 continuations per candidate
    noise = P.assign(b=P.p * (1 - P.p)).groupby("point").b.mean() / (2 - 1)
    var_true = var_obs - noise  # unclipped per point (clipping each point would bias the mean up)
    ceiling = float(np.sqrt(max(var_true.mean(), 0) / var_obs.mean())) if var_obs.mean() > 0 else float("nan")
    cols = [c for c in P.columns if c.count(":") == 1 and c.split(":")[0] in a.data] + ["random"]
    lines = [f"G1 within-state ranking at {res['points']} held-out probe points (val states), 8 candidates each, "
             f"2 continuations per candidate; critics trained {a.steps} steps x seeds {a.seeds}, width {a.width}",
             f"points where p varies: {int((var_obs > 0).sum())}; ceiling corr ~ {ceiling:.2f} "
             f"(var_between {var_true.mean():.4f} / var_observed {var_obs.mean():.4f})", "",
             "| critic | within-state Spearman | best-of-8 pick gain | by noise scale 1.0 / 1.5 / 2.0 (rho) | "
             "Q spread within state / across states |", "|---|---|---|---|---|"]
    across = P.groupby("point")
    for c in cols:
        per = metrics(P, c)
        r = {"critic": c, "rho": float(per.rho.mean()), "rho_ci95": boot_mean(per, "rho"),
             "pick": float(per.pick.mean()), "pick_ci95": boot_mean(per, "pick"),
             "rho_by_scale": {str(k): float(x.rho.mean()) for k, x in per.groupby("scale")},
             "points_rho": int(per.rho.notna().sum()),
             "spread_ratio": float(per.qsd.mean() / across[c].mean().std()) if c != "random" else float("nan")}
        res["arms"].append(r)
        lines.append(f"| {c} | {r['rho']:+.3f} [{r['rho_ci95'][0]:+.3f}, {r['rho_ci95'][1]:+.3f}] | "
                     f"{r['pick']:+.3f} [{r['pick_ci95'][0]:+.3f}, {r['pick_ci95'][1]:+.3f}] | "
                     + " / ".join(f"{v:+.2f}" for v in r["rho_by_scale"].values())
                     + f" | {r['spread_ratio']:.2f} |")
    res["ceiling"] = {"corr": ceiling, "var_between": float(var_true.mean()), "var_observed": float(var_obs.mean())}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
