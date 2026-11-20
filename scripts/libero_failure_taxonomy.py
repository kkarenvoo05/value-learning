"""Failure taxonomy of the base vs residual evals from their videos (Karen 2026-11-12, idea 1; meeting 11-11: "how
close is the robot to succeeding?", "compare episode length").

    .venv/bin/python scripts/libero_failure_taxonomy.py --out reports/2026-11-12-failure-taxonomy

Inputs: the agentview video features and IQL progress curves of scripts/libero_failure_curves.py (curves.parquet,
iql_cache/curves_<set>.npz). The IQL curve (tau 0.9, success reward) is time-aware, so every failure ends near 0; it
gives *when* an episode went wrong (first frame its smoothed value falls below 0.5), not how far it got.
How far = pots on the stove. The raw rollouts record the task's goal predicates per step (goals[:, 0:2] = the two
pots on the stove; goals[:, 2] is true from the start). A pot detector [ours]: 2 x 128 MLP on the agentview features
(no time input), BCE on the two pot flags, trained on every 4th frame of the 1450 train rollouts (base, exploration,
AG round 1); validated on stage2 val (120 base rollouts on the val states: video-frame predictions vs the recorded
flags). pots(t) = number of pots on (smoothed over 5 grid frames, i.e. 2 s).
Per failure [ours]: pots_end (last 2 s), first time a pot is on, the most pots ever on, the IQL drop time. Types:
  no_pot          no pot ever on the stove
  lost_pot        a pot was on, none at the end (knocked off / pulled back)
  one_pot         one pot on at the end (the second pot not placed in time)
Paired: base vs residual on the same 330 trials (matched noise seed 1).
Writes <out>/{taxonomy.csv, episodes.csv, summary.json, report.md, *.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import libero_iql as L  # noqa: E402
import libero_iql_g1 as G  # noqa: E402

TABLES = Path("/iris/u/ankile/karenvo/libero/tables")
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
STAGE2_VAL = RUNS / "stage2_moka_18155849" / "val"
log = L.log


def goals_for(d: pd.DataFrame, raw_of: dict) -> np.ndarray:
    out = np.zeros((len(d), 2), np.float32)
    for e, idx in d.groupby("episode_index").indices.items():
        with np.load(raw_of[e]) as z:
            g = z["goals"]
        out[idx] = g[d.frame_index.to_numpy()[idx], :2]
    return out


def training_set(cd: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X agentview features, Y pot flags, episode ids) for the G1 training frames; Y cached."""
    f = cd / "pot_flags_train.npz"
    parts = [L.build(20, cd / "iql_round0_n20.npz"), G.cached(cd / "iql_explore_n20.npz", G.explore_set),
             G.cached(cd / "iql_agr_r1_n20.npz", lambda: None)]
    X = np.concatenate([p["S"][:, :512] for p in parts])
    ep = np.concatenate([p["d"].episode_index.to_numpy() for p in parts])
    if f.exists():
        return X, np.load(f)["Y"], ep
    e1 = pd.read_csv(TABLES / "moka_explore" / "episodes.csv")
    e2 = pd.read_csv(TABLES / "moka_agr_r1" / "episodes.csv")
    raw_of = dict(zip(e1.episode_index, e1.raw)) | {k + G.AGR_OFFSET: r for k, r in zip(e2.episode_index, e2.raw)
                                                     if k >= 200000}
    Y = np.concatenate([goals_for(p["d"], raw_of) for p in parts])
    np.savez(f, Y=Y)
    return X, Y, ep


def train_pots(X, Y, tr, device, steps=20000, width=128, seed=0, bs=1024, lr=3e-4):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Xt, Yt = torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)
    ns = L.Std(X[tr], device)
    f = nn.Sequential(nn.Linear(X.shape[1], width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                      nn.Linear(width, 2)).to(device)
    o = torch.optim.Adam(f.parameters(), lr=lr)
    trt = torch.as_tensor(tr, device=device)
    for _ in range(steps):
        ix = trt[torch.randint(len(trt), (bs,), generator=g).to(device)]
        loss = nn.functional.binary_cross_entropy_with_logits(f(ns(Xt[ix])), Yt[ix])
        o.zero_grad(set_to_none=True), loss.backward(), o.step()

    @torch.no_grad()
    def pred(Xx):
        return torch.cat([torch.sigmoid(f(ns(torch.as_tensor(Xx[i:i + 8192], device=device)))).cpu()
                          for i in range(0, len(Xx), 8192)]).numpy()
    return pred


def smooth(x, k=5):
    return pd.Series(x).rolling(k, min_periods=1, center=True).mean().to_numpy()


def per_episode(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("frame_index")
    t, pots, v = g.frame_index.to_numpy(), smooth(g.pots.to_numpy()), smooth(g.iql_p.to_numpy())
    on = pots >= 0.5
    below = np.flatnonzero(v < 0.5)
    return pd.Series({"pots_end": float(np.round(pots[-5:].mean())), "pots_max": float(np.round(pots.max())),
                      "t_first_pot": float(t[np.argmax(on)]) if on.any() else np.nan,
                      "t_two_pots": float(t[np.argmax(pots >= 1.5)]) if (pots >= 1.5).any() else np.nan,
                      "t_drop": float(t[below[0]]) if len(below) else np.nan})


def ftype(r) -> str:
    if r.success:
        return "success"
    if r.pots_max < 1:
        return "no_pot"
    if r.pots_end < 1:
        return "lost_pot"
    return "one_pot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--curves", default=None, help="curves.parquet (default <out>/curves.parquet)")
    ap.add_argument("--cache-dir", default="/iris/u/karenvo/iql_cache")
    a = ap.parse_args()
    out, cd = Path(a.out), Path(a.cache_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = pd.read_parquet(a.curves or out / "curves.parquet")
    X, Y, ep = training_set(cd)
    u = np.unique(ep)
    hold = np.isin(ep, np.random.default_rng(0).choice(u, len(u) // 10, replace=False))
    f = train_pots(X, Y, np.flatnonzero(~hold), dev)
    acc_in = float(((f(X[hold]) > 0.5) == (Y[hold] > 0.5)).mean())
    log(f"pot detector: {len(X)} frames, held-out-episode frame accuracy {acc_in:.3f}")
    f = train_pots(X, Y, np.arange(len(X)), dev)
    parts = []
    for name in ("base", "residual", "stage2_val"):
        z = np.load(cd / f"curves_{name}.npz")["S"][:, :512]
        rows = pd.read_parquet(cd / f"curves_{name}.parquet")
        p = f(z)
        rows = rows.assign(p_pot1=p[:, 0], p_pot2=p[:, 1], pots=(p > 0.5).sum(1))
        parts.append(rows)
    P = pd.concat(parts, ignore_index=True).merge(C[["set", "episode", "frame_index", "iql_p"]],
                                                    on=["set", "episode", "frame_index"], how="left")
    assert P.iql_p.notna().all()

    # validation on stage2 val: video predictions vs the recorded pot flags
    v = P[P.set == "stage2_val"]
    eps = pd.read_csv(STAGE2_VAL / "episodes.csv", dtype={"goals_end": str})
    true = []
    for r in eps.itertuples():
        with np.load(STAGE2_VAL / "raw" / r.raw) as z:
            g = z["goals"]
        idx = np.arange(0, r.policy_steps, 4)
        true.append(pd.DataFrame({"episode": r.episode, "frame_index": idx, "true_pots": g[idx, :2].sum(1)}))
    vt = v.merge(pd.concat(true), on=["episode", "frame_index"])
    E = P.groupby(["set", "episode"]).apply(per_episode).reset_index()
    meta = P.groupby(["set", "episode"])[["init_state", "repeat", "success", "policy_steps"]].first().reset_index()
    E = meta.merge(E, on=["set", "episode"])
    E["type"] = E.apply(ftype, axis=1)
    sv = E[E.set == "stage2_val"].merge(eps[["episode", "goals_end"]], on="episode")
    sv["true_end"] = sv.goals_end.str[:2].map(lambda s: s.count("1"))
    val = {"frame_acc_pots": float((vt.pots == vt.true_pots).mean()),
           "end_acc_failures": float((sv[sv.success == 0].pots_end == sv[sv.success == 0].true_end).mean()),
           "end_confusion": pd.crosstab(sv[sv.success == 0].true_end, sv[sv.success == 0].pots_end).to_dict()}
    ft = pd.read_csv(STAGE2_VAL / "failure_types.csv")
    val["vs_sim_types"] = pd.crosstab(sv.merge(ft[["episode", "type"]], on="episode", suffixes=("", "_sim")).type_sim,
                                      sv.merge(ft[["episode", "type"]], on="episode", suffixes=("", "_sim")).type).to_dict()
    log(f"stage2 val: frame accuracy {val['frame_acc_pots']:.3f}, failures' end-state accuracy {val['end_acc_failures']:.3f}")

    # base vs residual, paired on (init_state, repeat)
    B = E[E.set == "base"].set_index(["init_state", "repeat"])
    R = E[E.set == "residual"].set_index(["init_state", "repeat"])
    J = B.join(R, lsuffix="_b", rsuffix="_r")
    counts = pd.DataFrame({"base": B.type.value_counts(), "residual": R.type.value_counts()}).fillna(0).astype(int)
    trans = pd.crosstab(J.type_b, J.type_r)
    fixed = J[(J.success_b == 0) & (J.success_r == 1)].type_b.value_counts().to_dict()
    broken = J[(J.success_b == 1) & (J.success_r == 0)].type_r.value_counts().to_dict()
    lens = {k: {"success_steps_mean": float(x[x.success == 1].policy_steps.mean()),
                "success_steps_median": float(x[x.success == 1].policy_steps.median()),
                "t_first_pot_median": float(x.t_first_pot.median()),
                "t_two_pots_median_successes": float(x[x.success == 1].t_two_pots.median()),
                "t_drop_median_failures": float(x[x.success == 0].t_drop.median())} for k, x in (("base", B), ("residual", R))}
    both = J[(J.success_b == 1) & (J.success_r == 1)]
    d = (both.policy_steps_r - both.policy_steps_b).to_numpy()
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    lens["paired_both_succeed"] = {"n": int(len(d)), "residual_minus_base_steps": float(d.mean()),
                                   "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res = {"pot_detector": {"heldout_episode_frame_acc": acc_in, **val}, "counts": counts.to_dict(),
           "transitions": trans.to_dict(), "fixed_by_base_type": fixed, "broken_into_type": broken, "lengths": lens}
    E.to_csv(out / "episodes.csv", index=False)
    P.to_parquet(out / "pots_curves.parquet")
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    figures(P, E, B, R, out)
    print(json.dumps(res, indent=1, default=str))


def figures(P, E, B, R, out):
    order = ["success", "one_pot", "lost_pot", "no_pot"]
    cols = {"success": "#4c9a63", "one_pot": "#e0a43a", "lost_pot": "#c8553d", "no_pot": "#7a7a7a"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    left = np.zeros(2)
    for t in order:
        n = np.array([(B.type == t).sum(), (R.type == t).sum()])
        ax[0].barh(["base", "residual"], n, left=left, color=cols[t], label=t)
        for i in range(2):
            if n[i] >= 8:
                ax[0].text(left[i] + n[i] / 2, i, str(n[i]), ha="center", va="center", color="white", fontsize=9)
        left += n
    ax[0].set_xlabel("trials (330 each)"); ax[0].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax[0].set_title("outcome per trial")
    bins = np.arange(200, 530, 20)
    for k, x, c in (("base", B, "#7a7a7a"), ("residual", R, "#3b6ea5")):
        ax[1].hist(x[x.success == 1].policy_steps, bins=bins, alpha=0.6, color=c, label=f"{k} (median {x[x.success == 1].policy_steps.median():.0f})")
    ax[1].set_xlabel("episode length of successes (steps, 10 Hz)"); ax[1].legend(fontsize=8); ax[1].set_title("how fast successes finish")
    for k, x, c in (("base", B, "#7a7a7a"), ("residual", R, "#3b6ea5")):
        ax[2].hist(x.t_first_pot.dropna(), bins=np.arange(0, 530, 20), alpha=0.6, color=c, label=k)
    ax[2].set_xlabel("step when the first pot is on the stove"); ax[2].legend(fontsize=8); ax[2].set_title("first pot placed")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out / "overview.png", dpi=110); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    for j, t in enumerate(["one_pot", "lost_pot", "no_pot"]):
        for i, (k, x) in enumerate((("base", B), ("residual", R))):
            sel = x[x.type == t].reset_index().head(6)
            for r in sel.itertuples():
                g = P[(P.set == k) & (P.episode == r.episode)].sort_values("frame_index")
                ax[i, j].plot(g.frame_index, smooth(g.pots.to_numpy()), color=cols[t], alpha=0.7)
                ax[i, j].plot(g.frame_index, smooth(g.iql_p.to_numpy()) * 2, color="#3b6ea5", alpha=0.35, lw=1)
            ax[i, j].set_title(f"{k}: {t} ({(x.type == t).sum()})", fontsize=10)
            ax[i, j].set_ylim(-0.1, 2.3)
            ax[i, j].set_ylabel("pots on stove (orange/red/grey)\nIQL value x 2 (blue)", fontsize=8)
    for a_ in ax[-1]:
        a_.set_xlabel("step")
    plt.tight_layout(); plt.savefig(out / "failure_curves.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
