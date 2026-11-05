"""Milestone check: does V rise when the first moka pot lands on the stove? (docs/libero_plan.md, Stage 4)

Usage:
    .venv/bin/python -m vl.libero_milestone \
        --probe /iris/u/ankile/karenvo/libero/features/feats_every4_18163142_oc \
        --resnet /iris/u/ankile/karenvo/libero/runs/resnet18_18166232
        -> reports/libero_milestone.md, reports/figures/libero_milestone/aligned.png

Milestone m [ours]: the first frame at which either pot satisfies On(pot, cook_region) and keeps it for HOLD
steps (goals[t] is evaluated after action t, so the placed pot is first visible in image t + 1). The policy
does not always place moka_pot_1 first, hence "either pot". Episodes that never get there have no milestone.

Per model and val episode with a milestone, jump J = mean V over frames [m, m + W) minus mean V over
[m - W, m). V rises over time in many episodes anyway, so J alone means little; two controls:
  - time-matched (difference in differences): the same window on the other val episodes that have no pot on
    the stove anywhere in [m - W, m + W + HOLD); excess = J - mean of their J.
  - within-episode placebo: the same J at every other frame tau of the episode (4-frame grid, |tau - m| >= 2W);
    excess = J - mean placebo J.
Reference: an oracle V that knows only the clock and whether the milestone has been reached, fitted on the
train episodes (mean target per 2 s time bin x reached / not), scored with the same statistic. Its excess is
the jump a V that reads the milestone perfectly, and nothing else, would show. A clock-only V has zero excess
on the time-matched control by construction. 95% intervals: bootstrap over the 10 val init states (2000
resamples; episodes within a scene are not independent).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path("/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849")
DATASET = Path("/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2")
FPS = 10
HOLD = 10  # steps a pot must stay on the stove (1 s)
W = 50  # half-window, steps (5 s)
TBIN = 20  # oracle time bin, steps (2 s)
MIN_EPS = 10  # oracle cells with fewer train episodes fall back to the clock (all episodes at that bin)
OFFSETS = np.arange(-150, 151, 4)  # aligned-curve offsets, steps


def milestone(goals: np.ndarray) -> float:
    """First frame showing a pot on the stove that stays there HOLD steps; nan if none."""
    on = goals[:, :2].any(axis=1)
    for t in np.flatnonzero(on):
        if on[t:t + HOLD].all():
            return float(t + 1)
    return np.nan


def episodes(split: str) -> pd.DataFrame:
    eps = pd.read_csv(DATASET / "episodes.csv")
    eps = eps[eps.split == split].copy()
    ms = []
    for raw in eps.raw:
        with np.load(RUN / raw) as z:
            ms.append(milestone(z["goals"]))
    eps["milestone"] = ms
    return eps


def oracle(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean target and success rate per (time bin, milestone reached), from the train episodes."""
    ft = pd.read_parquet(DATASET / "frame_targets.parquet", columns=["episode_index", "frame_index", "value"])
    ft = ft.merge(train[["episode_index", "milestone", "success"]], on="episode_index")
    ft["tb"] = ft.frame_index // TBIN
    ft["reached"] = (ft.frame_index >= ft.milestone).astype(int)  # nan milestone -> never reached
    nb = int(ft.tb.max()) + 1
    ev, ps = np.zeros((nb, 2)), np.zeros((nb, 2))
    clock = ft.groupby("tb").agg(v=("value", "mean"), s=("success", "mean"))
    cells = ft.groupby(["tb", "reached"]).agg(v=("value", "mean"), s=("success", "mean"), n=("episode_index", "nunique"))
    for b in range(nb):
        for r in (0, 1):
            c = cells.loc[(b, r)] if (b, r) in cells.index else None
            ok = c is not None and c.n >= MIN_EPS
            ev[b, r] = c.v if ok else clock.v.get(b, np.nan)
            ps[b, r] = c.s if ok else clock.s.get(b, np.nan)
    return ev, ps


def win_mean(f: np.ndarray, v: np.ndarray, lo: float, hi: float) -> float:
    k = (f >= lo) & (f < hi)
    return v[k].mean() if k.any() else np.nan


def jump(f: np.ndarray, v: np.ndarray, t: float) -> float:
    return win_mean(f, v, t, t + W) - win_mean(f, v, t - W, t)


def scene_ci(per_ep: pd.DataFrame, col: str, rng) -> tuple[float, float, float]:
    g = per_ep.groupby("init_state")[col]
    s, n = g.sum(), g.size()
    states = s.index.values
    bs = [s[p].sum() / n[p].sum() for p in (rng.choice(states, len(states)) for _ in range(2000))]
    return per_ep[col].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="vl.libero_probe outcome+clock dir (val_predictions.parquet)")
    ap.add_argument("--resnet", required=True, help="vl.libero_resnet output dir")
    ap.add_argument("--report", default="reports/libero_milestone.md")
    ap.add_argument("--fig", default="reports/figures/libero_milestone")
    a = ap.parse_args()

    from vl.libero_probe import CENTERS, clock_by_outcome, load
    pred = pd.read_parquet(Path(a.probe) / "val_predictions.parquet")
    _, meta = load(Path("/iris/u/ankile/karenvo/libero/features/feats_every4_18163142.npz"), DATASET)
    hs, hf = clock_by_outcome(meta[meta.split == "train"], int(meta.frame_index.max()) + 1)
    fr = pred.frame_index.values
    pred["ev_oc"] = pred.ps_all_mlp * (hs[fr] @ CENTERS) + (1 - pred.ps_all_mlp) * (hf[fr] @ CENTERS)
    r = pd.read_parquet(Path(a.resnet) / "val_predictions.parquet")[["episode_index", "frame_index", "ev_resnet"]]
    pred = pred.merge(r, on=["episode_index", "frame_index"], how="inner", validate="one_to_one")

    train, val = episodes("train"), episodes("val")
    ev_or, ps_or = oracle(train)
    pred = pred.merge(val[["episode_index", "milestone"]], on="episode_index", validate="many_to_one")
    tb = np.minimum(pred.frame_index.values // TBIN, len(ev_or) - 1)
    reached = (pred.frame_index >= pred.milestone).astype(int).values
    pred["ev_oracle"] = ev_or[tb, reached]
    pred["ps_oracle"] = ps_or[tb, reached]

    # (column, label, scale): value-scale models, then P(success)
    models = [("value", "target v (true)", "v"), ("ev_clock", "clock", "v"), ("ev_oracle", "oracle: clock + milestone", "v"),
              ("ev_all_mlp", "π₀.₅ probe, 201 bins", "v"), ("ev_oc", "π₀.₅ probe, P(success) × clock", "v"),
              ("ev_resnet", "ResNet-18", "v"),
              ("ps_oracle", "oracle P(success | clock, milestone)", "p"), ("ps_all_mlp", "π₀.₅ probe P(success)", "p")]

    by_ep = {e: g.sort_values("frame_index") for e, g in pred.groupby("episode_index")}
    ms = val.set_index("episode_index").milestone
    T = val.set_index("episode_index").policy_steps
    treated = [e for e in by_ep if np.isfinite(ms[e]) and ms[e] - W >= 0 and ms[e] + W <= T[e]]

    rows = []
    for e in treated:
        m, g = ms[e], by_ep[e]
        f = g.frame_index.values
        ctrl = [c for c in by_ep if c != e and T[c] >= m + W and not (ms[c] < m + W + HOLD)]
        taus = [t for t in f if t - W >= 0 and t + W <= T[e] and abs(t - m) >= 2 * W]
        row = {"episode_index": e, "init_state": int(g.init_state.iloc[0]), "success": int(g.success.iloc[0]),
               "milestone_s": m / FPS, "n_ctrl": len(ctrl), "n_placebo": len(taus)}
        for col, _, _ in models:
            v = g[col].values
            j = jump(f, v, m)
            jc = np.nanmean([jump(by_ep[c].frame_index.values, by_ep[c][col].values, m) for c in ctrl]) if ctrl else np.nan
            jp = np.nanmean([jump(f, v, t) for t in taus]) if taus else np.nan
            row[f"{col}/J"], row[f"{col}/dd"], row[f"{col}/pl"] = j, j - jc, j - jp
        rows.append(row)
    per = pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    table = []
    for col, label, scale in models:
        r = {"model": label, "scale": "V" if scale == "v" else "P(success)"}
        for key, name in (("J", "raw jump"), ("dd", "excess vs time-matched"), ("pl", "excess vs placebo")):
            sub = per.dropna(subset=[f"{col}/{key}"])
            mu, lo, hi = scene_ci(sub, f"{col}/{key}", rng)
            r[name] = f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        o = "ev_oracle" if scale == "v" else "ps_oracle"
        r["÷ oracle (time-matched)"] = per[f"{col}/dd"].mean() / per[f"{o}/dd"].mean()
        table.append(r)
    t = pd.DataFrame(table)

    # per-episode sign agreement for the models, time-matched excess
    pos = {label: (per[f"{col}/dd"] > 0).mean() for col, label, _ in models}

    # aligned curves: mean V at offset from the milestone (nearest 4-grid frame), treated episodes
    out = Path(a.fig)
    out.mkdir(parents=True, exist_ok=True)
    curves = {col: [] for col, _, _ in models}
    for e in treated:
        g = by_ep[e].set_index("frame_index")
        m = ms[e]
        for col in curves:
            vals = []
            for o in OFFSETS:
                k = int(round((m + o) / 4) * 4)
                vals.append(g[col].get(k, np.nan) if 0 <= k < T[e] else np.nan)
            curves[col].append(vals)
    curves = {c: np.nanmean(np.array(v, dtype=float), axis=0) for c, v in curves.items()}
    style = {"value": dict(color="black", lw=2), "ev_clock": dict(color="grey", ls="--", lw=1),
             "ev_oracle": dict(color="#008300", ls=":", lw=2), "ev_all_mlp": dict(color="#eb6834", lw=2),
             "ev_oc": dict(color="#2a78d6", lw=2), "ev_resnet": dict(color="#e87ba4", lw=2),
             "ps_oracle": dict(color="#008300", ls=":", lw=2), "ps_all_mlp": dict(color="#2a78d6", lw=2)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, scale in zip(axes, ("v", "p")):
        for col, label, s in models:
            if s == scale:
                ax.plot(OFFSETS / FPS, curves[col], label=label, **style[col])
        ax.axvline(0, color="grey", lw=0.6)
        for x in (-W / FPS, W / FPS):
            ax.axvline(x, color="grey", lw=0.4, ls=":")
        ax.set_xlabel("seconds from first pot on the stove")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("mean V")
    axes[1].set_ylabel("mean P(success)")
    fig.suptitle(f"Milestone-aligned value, {len(treated)} val episodes (10 unseen scenes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "aligned.png", dpi=130)

    n_none = int(val.milestone.isna().sum())
    lines = [
        "# Milestone check: does V rise when the first pot lands on the stove?",
        "",
        "Generated by `python -m vl.libero_milestone` (docstring has the definitions). Val = 10 unseen init states, "
        f"120 episodes; {len(treated)} reach the milestone with a full ±{W / FPS:.0f} s window "
        f"({int(per.success.sum())} successes, {int((1 - per.success).sum())} failures), {n_none} never place a pot. "
        f"Milestone at {per.milestone_s.median():.1f} s median [{per.milestone_s.min():.1f}, {per.milestone_s.max():.1f}]. "
        f"Time-matched controls per episode: median {int(per.n_ctrl.median())}.",
        "",
        f"Jump = mean V over the {W / FPS:.0f} s after minus the {W / FPS:.0f} s before; mean over episodes, "
        "[95% bootstrap over scenes].",
        "",
        t.to_markdown(index=False, floatfmt=".2f"),
        "",
        "Fraction of episodes with a positive time-matched excess: "
        + ", ".join(f"{k} {v:.2f}" for k, v in pos.items()),
        "",
        f"![aligned](figures/libero_milestone/aligned.png)",
        "",
    ]
    Path(a.report).write_text("\n".join(lines))
    per.to_csv(out / "per_episode.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
