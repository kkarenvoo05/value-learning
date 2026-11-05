"""Follow-up to the milestone check (vl.libero_milestone): why does the π₀.₅ probe's P(success) dip after the first
pot lands? Two questions, both on the cached mean-pooled features (no GPU, no new extraction).

Usage:
    .venv/bin/python -m vl.libero_milestone_probe \
        --probe /iris/u/ankile/karenvo/libero/features/feats_every4_18163142_oc \
        --resnet /iris/u/ankile/karenvo/libero/runs/resnet18_18166232
        -> reports/libero_milestone_probe.md, reports/figures/libero_milestone_probe/

Step 1, decodability: can the features tell whether a pot is on the stove? Label per frame = frame >= milestone
(vl.libero_milestone.milestone; 0 for episodes that never place a pot). Heads and training exactly as vl.libero_probe
(linear / MLP, AdamW, early stopping on the 5 inner train init states) with 2 outputs; trained on train, scored on
the 10 val init states:
  auc           all val frames (the clock alone already does well: placed frames are later)
  auc_tm        time-matched: AUC within each 2 s bin from 10 to 40 s (bins with >= 20 frames per class), mean
                weighted by frames; the clock alone gets 0.5 by construction
  auc_win       the milestone-check window: frames in [m - 5 s, m) vs [m, m + 5 s) of the same val episodes
  auc_win_gap   same with the 1 s either side of m dropped (pot being lowered / just released are ambiguous)
Control: `clock`, P(placed | frame bin) from the train episodes. 95% intervals for auc_win: bootstrap over val scenes.
If the features decode placement well but the value probe still dips, the information is there and the value head
does not use it; if they do not, mean pooling (or the final layer) loses it.

Step 2, what does the dip line up with? Events per val episode with a milestone m, from the raw sim data (gripper
command = action[6], > 0 closes; pot p = the one placed first, q = the other) [ours]:
  release       first open command after the last close command before m (letting go of pot p)
  milestone     m
  near_q        first frame after release with the gripper within NEAR of pot q
  grasp_q       first close command at or after near_q
  lift_q        first frame after release with pot q LIFT above its height at release
Per event: P(success) (probe) and V (probe, ResNet) averaged on frames aligned to the event, and the lock-in
statistic: per episode the time of the lowest probe P(success) in [m - 2 s, m + 12 s]; its spread (MAD) relative to
each event across episodes. The event the dip is tied to gives the smallest spread and the deepest aligned dip.
Neighbours: for val frames in three windows (before m, the dip [m, m + 8 s), after it [m + 10 s, m + 15 s)) the
K nearest train frames (cosine, z-scored `all` pooling): what fraction have a pot on the stove, come from a failure,
or come from a stuck-at-pot-2 failure (types stuck_pot2 / stuck_placing_pot2 of scripts/libero_failure_types.py).
Calibration by phase: probe P(success) averaged per episode in windows around m vs the actual success rate of the
same episodes. Separation: the raw milestone jump (vl.libero_milestone.jump) split by outcome, and its AUC for success.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from vl.libero_milestone import DATASET, FPS, RUN, W, episodes, jump
from vl.libero_probe import features_for, fit, inner_split, load, predict
from vl.metrics import auc

FEATS = Path("/iris/u/ankile/karenvo/libero/features/feats_every4_18163142.npz")
NEAR, LIFT = 0.10, 0.02  # metres
TBIN = 20
GAP = 10  # steps dropped either side of m for auc_win_gap
K = 20
OFFSETS = np.arange(-100, 151, 4)
STUCK2 = ("stuck_pot2", "stuck_placing_pot2")
CAL_WINDOWS = (("[m-10, m-5 s)", -100, -50), ("[m-5 s, m)", -50, 0), ("[m, m+4 s)", 0, 40), ("[m+4, m+8 s)", 40, 80),
               ("[m+10, m+15 s)", 100, 150))
SEP_COLS = {"ps_all_mlp": "π₀.₅ probe P(success)", "ev_all_mlp": "π₀.₅ probe V (201 bins)", "ev_resnet": "ResNet-18 V"}


def events(raw: Path, m: float) -> dict:
    with np.load(raw) as z:
        a, goals, op = z["actions"][:, 6], z["goals"], z["object_pos"]
    out = {"milestone": m}
    if not np.isfinite(m):
        return out
    m = int(m)
    p = 0 if goals[m - 1, 0] else 1  # goals[t] is after action t; milestone() returns t + 1
    q = 1 - p
    closed = np.flatnonzero(a[:m] > 0)
    if not len(closed):
        return out
    after = np.flatnonzero(a[closed[-1]:] <= 0)
    if not len(after):
        return out
    rel = closed[-1] + after[0]
    out["release"] = rel
    d_q = np.linalg.norm(op[:, 9:12] if q == 1 else op[:, 3:6], axis=1)
    z_q = op[:, 8] if q == 1 else op[:, 2]
    for name, hit in (("near_q", d_q < NEAR), ("lift_q", z_q > z_q[rel] + LIFT)):
        k = np.flatnonzero(hit[rel:])
        if len(k):
            out[name] = rel + k[0]
    if "near_q" in out:  # a stray close command right after release is not a grasp
        k = np.flatnonzero(a[out["near_q"]:] > 0)
        if len(k):
            out["grasp_q"] = out["near_q"] + k[0]
    return out


def scene_boot(df: pd.DataFrame, fn, rng, n=1000) -> tuple[float, float]:
    states = df.init_state.unique()
    groups = {s: g for s, g in df.groupby("init_state")}
    bs = [fn(pd.concat([groups[s] for s in rng.choice(states, len(states))])) for _ in range(n)]
    return np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def tm_auc(d: pd.DataFrame, col: str) -> float:
    d = d[(d.frame_index >= 10 * FPS) & (d.frame_index < 40 * FPS)]
    num = den = 0.0
    for _, g in d.groupby(d.frame_index // TBIN):
        n1 = int(g.placed.sum())
        if n1 >= 20 and len(g) - n1 >= 20:
            num += auc(g.placed.values, g[col].values) * len(g)
            den += len(g)
    return num / den if den else np.nan


def win(d: pd.DataFrame, gap: int = 0) -> pd.DataFrame:
    r = d.frame_index - d.milestone
    return d[((r >= -W) & (r < -gap)) | ((r >= gap) & (r < W))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="vl.libero_probe outcome+clock dir (val_predictions.parquet)")
    ap.add_argument("--resnet", required=True, help="vl.libero_resnet output dir")
    ap.add_argument("--poolings", nargs="+", default=["all", "agentview", "wrist", "text"])
    ap.add_argument("--heads", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--report", default="reports/libero_milestone_probe.md")
    ap.add_argument("--fig", default="reports/figures/libero_milestone_probe")
    a = ap.parse_args()
    out = Path(a.fig)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.set_num_threads(4)

    # ---- labels
    F, meta = load(FEATS, DATASET)
    eps = pd.concat([episodes("train"), episodes("val")])
    meta = meta.merge(eps[["episode_index", "milestone"]], on="episode_index", validate="many_to_one")
    meta["placed"] = (meta.frame_index >= meta.milestone).astype(int)  # nan -> 0
    tr, va = (meta.split == "train").values, (meta.split == "val").values
    inner = inner_split(meta)
    fit_m = tr & ~inner
    vmeta = meta[va].reset_index(drop=True)

    # ---- step 1: decodability
    clock = meta[tr].groupby(meta.frame_index // TBIN).placed.mean()
    vmeta["clock"] = clock.reindex(vmeta.frame_index // TBIN).fillna(clock.iloc[-1]).values
    cols = ["clock"]
    for pool in a.poolings:
        X = features_for(F, pool)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        X = ((X - mu) / sd).astype(np.float32)
        y = meta.placed.values
        for head in a.heads:
            net, ce, ep = fit(head, X[fit_m], y[fit_m], X[inner], y[inner])
            c = f"{pool}_{head}"
            vmeta[c] = predict(net, X[va])[:, 1]
            cols.append(c)
            print(f"{c}: inner CE {ce:.3f} (epoch {ep})", flush=True)
        if pool == "all":
            Xall = X
    rows = []
    for c in cols:
        w, wg = win(vmeta.dropna(subset=["milestone"]), 0), win(vmeta.dropna(subset=["milestone"]), GAP)
        lo, hi = scene_boot(wg, lambda d: auc(d.placed.values, d[c].values), rng)
        rows.append({"model": c, "auc": auc(vmeta.placed.values, vmeta[c].values), "auc_tm": tm_auc(vmeta, c),
                     "auc_win": auc(w.placed.values, w[c].values),
                     "auc_win_gap": f"{auc(wg.placed.values, wg[c].values):.3f} [{lo:.3f}, {hi:.3f}]"})
    t1 = pd.DataFrame(rows)
    print(t1.to_string(), flush=True)

    # milestone-aligned P(placed) curves
    best = t1.iloc[1:].sort_values("auc_tm").model.iloc[-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vm = vmeta.dropna(subset=["milestone"]).assign(rel=lambda d: d.frame_index - d.milestone)
    for c, st in (("clock", dict(color="grey", ls="--")), ("all_mlp", dict(color="#2a78d6", lw=2)),
                  (best, dict(color="#eb6834", lw=2))):
        if c in vm:
            g = vm[(vm.rel >= -100) & (vm.rel < 150)].groupby((vm.rel // 4) * 4)[c].mean()
            ax.plot(g.index / FPS, g.values, label=c, **st)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("seconds from first pot on the stove")
    ax.set_ylabel("mean P(pot on stove)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "decode_aligned.png", dpi=130)

    # ---- step 2: event alignment on the value predictions
    pred = pd.read_parquet(Path(a.probe) / "val_predictions.parquet")[
        ["episode_index", "init_state", "frame_index", "success", "ps_all_mlp", "ev_all_mlp"]]
    r = pd.read_parquet(Path(a.resnet) / "val_predictions.parquet")[["episode_index", "frame_index", "ev_resnet"]]
    pred = pred.merge(r, on=["episode_index", "frame_index"], validate="one_to_one")
    val = eps[eps.split == "val"].set_index("episode_index")
    ev = pd.DataFrame([{"episode_index": e, **events(RUN / val.raw[e], val.milestone[e])} for e in val.index])
    ev = ev.set_index("episode_index")
    names = ["release", "milestone", "near_q", "grasp_q", "lift_q"]
    T = val.policy_steps
    treated = [e for e in ev.index if np.isfinite(ev.milestone[e]) and ev.milestone[e] >= W and ev.milestone[e] + W <= T[e]]
    by_ep = {e: g.set_index("frame_index") for e, g in pred.groupby("episode_index")}

    def aligned(col, ev_name, eps_):
        rowsv = []
        for e in eps_:
            t0 = ev.loc[e, ev_name]
            if not np.isfinite(t0):
                continue
            g = by_ep[e][col]
            rowsv.append([g.get(int(round((t0 + o) / 4) * 4), np.nan) for o in OFFSETS])
        return np.nanmean(np.array(rowsv, dtype=float), axis=0), len(rowsv)

    lock, ev_rows = [], []
    for e in treated:
        g = by_ep[e]
        m = ev.milestone[e]
        s = g.ps_all_mlp[(g.index >= m - 20) & (g.index <= m + 120)]
        tmin = s.idxmin()
        lock.append({"episode_index": e, "success": int(g.success.iloc[0]), "t_min": tmin, "dip": s.min() - s.iloc[0],
                     **{n: tmin - ev.loc[e, n] for n in names}})
        ev_rows.append({n: (ev.loc[e, n] - m) / FPS for n in names})
    lock = pd.DataFrame(lock)
    evrel = pd.DataFrame(ev_rows)
    t2 = []
    for n in names:
        d = lock[n].dropna()
        curve, k = aligned("ps_all_mlp", n, treated)
        i0 = np.flatnonzero(OFFSETS == 0)[0]
        t2.append({"event": n, "episodes": k, "event time vs m (s), median": evrel[n].median(),
                   "dip time vs event (s), median": d.median() / FPS, "MAD (s)": (d - d.median()).abs().median() / FPS,
                   "aligned dip depth": np.nanmin(curve[i0:i0 + 20]) - curve[i0 - 5:i0].mean()})
    t2 = pd.DataFrame(t2)
    print(t2.to_string(), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=False)
    colors = dict(zip(names, ["#e87ba4", "black", "#2a78d6", "#eb6834", "#008300"]))
    for ax, (col, lab) in zip(axes, (("ps_all_mlp", "π₀.₅ probe P(success)"), ("ev_all_mlp", "π₀.₅ probe V (201 bins)"),
                                     ("ev_resnet", "ResNet-18 V"))):
        for n in names:
            curve, k = aligned(col, n, treated)
            ax.plot(OFFSETS / FPS, curve, color=colors[n], lw=2 if n == "milestone" else 1.3, label=f"{n} (n={k})")
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("seconds from event")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "event_aligned.png", dpi=130)

    # ---- step 2c: calibration by phase, and does the within-episode change separate outcomes?
    pt = pred[pred.episode_index.isin(treated)].copy()
    pt["rel"] = pt.frame_index - pt.episode_index.map(ev.milestone)
    cal = []
    for wname, lo_, hi_ in CAL_WINDOWS:
        s = pt[(pt.rel >= lo_) & (pt.rel < hi_)].groupby("episode_index").agg(ps=("ps_all_mlp", "mean"), y=("success", "first"))
        cal.append({"window": wname, "episodes": len(s), "probe P(success)": s.ps.mean(), "actual success": s.y.mean(),
                    "probe | success": s.ps[s.y == 1].mean(), "probe | failure": s.ps[s.y == 0].mean()})
    cal = pd.DataFrame(cal)
    print(cal.to_string(), flush=True)
    jp = []
    for e in treated:
        g = by_ep[e]
        jp.append({"episode_index": e, "init_state": int(g.init_state.iloc[0]), "success": int(g.success.iloc[0]),
                   **{c: jump(g.index.values, g[c].values, ev.milestone[e]) for c in SEP_COLS}})
    jp = pd.DataFrame(jp)
    sep = []
    for c, lab in SEP_COLS.items():
        f = lambda d: d[d.success == 1][c].mean() - d[d.success == 0][c].mean()
        lo, hi = scene_boot(jp, f, rng)
        sep.append({"model": lab, "jump | success": jp[jp.success == 1][c].mean(), "jump | failure": jp[jp.success == 0][c].mean(),
                    "success − failure": f"{f(jp):+.3f} [{lo:+.3f}, {hi:+.3f}]", "AUC(jump → success)": auc(jp.success.values, jp[c].values)})
    sep = pd.DataFrame(sep)
    print(sep.to_string(), flush=True)

    # ---- step 2b: nearest train neighbours of val frames around the milestone
    ft = pd.read_csv(RUN / "train" / "failure_types.csv")
    teps = eps[eps.split == "train"][["episode_index", "source_episode"]]
    stuck = set(teps.merge(ft[ft.type.isin(STUCK2)], left_on="source_episode", right_on="episode").episode_index)
    tmeta = meta[tr].reset_index(drop=True)
    Xt = torch.nn.functional.normalize(torch.as_tensor(Xall[tr]), dim=1)
    Xv = torch.nn.functional.normalize(torch.as_tensor(Xall[va]), dim=1)
    vrel = (vmeta.frame_index - vmeta.milestone).values
    vtreat = vmeta.episode_index.isin(treated).values
    t3 = []
    for wname, lo_, hi_ in (("before [m-5 s, m)", -W, 0), ("dip [m, m+8 s)", 0, 80), ("after [m+10, m+15 s)", 100, 150)):
        sel = np.flatnonzero(vtreat & (vrel >= lo_) & (vrel < hi_))
        nn_ = torch.cat([(Xv[sel[i:i + 1024]] @ Xt.T).topk(K, dim=1).indices for i in range(0, len(sel), 1024)]).numpy()
        nb = tmeta.iloc[nn_.ravel()]
        own_rel = (nb.frame_index.values - nb.milestone.values)
        v_succ = vmeta.success.values[sel]
        row = {"window": wname, "val frames": len(sel),
               "nbr pot on stove": nb.placed.mean(), "nbr from failure": 1 - nb.success.mean(),
               "nbr from stuck-at-pot-2": nb.episode_index.isin(stuck).mean(),
               "nbr time − query time (s)": np.median(nb.frame_index.values.reshape(-1, K) - vmeta.frame_index.values[sel, None]) / FPS,
               "nbr in own [m, m+8 s)": np.nanmean((own_rel >= 0) & (own_rel < 80))}
        for ok, nm in ((1, "succ"), (0, "fail")):
            k = np.repeat(v_succ == ok, K)
            row[f"nbr from failure | val {nm}"] = 1 - nb.success.values[k].mean()
        t3.append(row)
    t3 = pd.DataFrame(t3)
    print(t3.to_string(), flush=True)
    # train base rates for the neighbour fractions
    base = {"pot on stove": tmeta.placed.mean(), "failure": 1 - tmeta.success.mean(),
            "stuck-at-pot-2": tmeta.episode_index.isin(stuck).mean()}

    lines = [
        "# Why does the π₀.₅ probe dip after the first pot lands?",
        "",
        "Generated by `python -m vl.libero_milestone_probe` (docstring has the definitions). Features = cached mean-pooled "
        "final-layer prefix tokens (every 4th frame); train = 30 train init states, scored on the 10 val init states.",
        "",
        "## Step 1: can the features tell whether a pot is on the stove?",
        "",
        t1.to_markdown(index=False, floatfmt=".3f"),
        "",
        "`auc_tm` = time-matched (clock = 0.5 by construction); `auc_win` = the ±5 s milestone window; "
        "`auc_win_gap` drops the 1 s either side, [95% bootstrap over val scenes].",
        "",
        "![decode](figures/libero_milestone_probe/decode_aligned.png)",
        "",
        "## Step 2: what is the dip lined up with?",
        "",
        f"{len(treated)} val episodes with a full ±5 s milestone window. Lock-in: time of the lowest probe P(success) "
        "in [m − 2 s, m + 12 s], relative to each event; smaller MAD = tighter alignment. Dip depth = lowest point of "
        "the event-aligned mean P(success) in the 8 s after the event minus its mean over the 2 s before.",
        "",
        t2.to_markdown(index=False, floatfmt=".2f"),
        "",
        "![events](figures/libero_milestone_probe/event_aligned.png)",
        "",
        "### Calibration by phase (probe P(success) vs the actual success rate of the same episodes)",
        "",
        "Aligned on a stopping time, a calibrated P(success) has the same mean before and after it, so a large rise at "
        "placement is not what a good V should show; the question is whether it is right in each phase.",
        "",
        cal.to_markdown(index=False, floatfmt=".3f"),
        "",
        "### Does the jump at placement separate successes from failures?",
        "",
        "Raw jump (mean over the 5 s after minus the 5 s before), [95% bootstrap over val scenes].",
        "",
        sep.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"### Nearest train neighbours (K = {K}, cosine, `all` pooling)",
        "",
        t3.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Train base rates: " + ", ".join(f"{k} {v:.3f}" for k, v in base.items()),
        "",
    ]
    Path(a.report).write_text("\n".join(lines))
    lock.to_csv(out / "lock_in.csv", index=False)
    ev.to_csv(out / "events.csv")
    vmeta.drop(columns=["T"]).to_parquet(out / "decode_val.parquet")


if __name__ == "__main__":
    main()
