"""Stage 6 first step: RECAP-style advantage labels from R0 on the val split, and whether they make sense.

Usage:
    .venv/bin/python -m vl.advantages        # -> reports/advantages.md, reports/figures/stage6/advantages/

No training, val only (R0 never trained on these episodes, so no out-of-fold V is needed here).

Advantage [paper: RECAP's n-step estimate, no discount]:
    A_t = sum_{k=t}^{t+n-1} r_k + V(o_{t+n}) - V(o_t),   r_k = -1/1800 per step (our normalization)
Terminal handling [ours, explicit]: if t + n >= T the lookahead hits the end, so
    A_t = -(T - t)/1800 + v_T - V(o_t),   v_T = 0 (success) or -0.5 (failure / timeout: the 900-step penalty)
With V = the training target itself every A_t is exactly 0 (checked below): advantages only carry what V
gets *wrong* about a straight countdown, which is why an in-sample V gives ~0 advantages.
Frames with t + n < T are "interior": there A depends on V alone. Frames with t + n >= T are "terminal":
there the known outcome enters, so any V looks informative.

Value functions compared:
  R0        3-seed ensemble of g0_r0_s{0,1,2}, centred 0.5 s moving average (the proposed offline
            recipe, reports/inference_choices.md). V is linear in the distribution, so averaging V equals
            averaging the distributions.
  R0 single g0_r0_s0, no smoothing (what post-processing buys)
  C2a       3-seed ensemble of c2a_stack2_s{0,1,2} (current + 2 s-old frame stacked), centred 0.5 s, as R0
  time+round [ours, control]: mean training target per (repo, elapsed second). Sees no image, only the
            round and the clock. If its labels match R0's, the images add nothing to the advantages.

Threshold [paper, ambiguous]: RECAP sets eps per task to "the 30% percentile"; the text says of predicted
values, the formulas apply it to advantages. We use the 30th percentile of the val advantages
(-> 70% positive, the literal reading) and also report the 70th percentile (-> 30% positive).
Lookaheads n = 1, 2, 5 s (15, 30, 75 frames) [ours].
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vl.config import load_config
from vl.dataset import load_split
from vl.history_gain import hold_groups
from vl.history_plots import stall_seconds
from vl.history_probe import roc_auc_score

RUNS = Path("data/runs")
OUT_MD = Path("reports/advantages.md")
FIG = Path("reports/figures/stage6/advantages")
FPS = 15
NORM = 1800.0
NS_S = [1, 2, 5]
PCTS = {"70% positive": 30, "30% positive": 70}
MAIN_N, MAIN_PCT = 2, "70% positive"
B = 1000
MID_S = 5   # [ours] "mid-episode" = lookahead ends at least this many seconds before T
KEY = ["repo", "episode_index", "frame_index"]
COL = {"success": "#2a78d6", "failure": "#eb6834", "timeout": "#1baf7a", "control": "#8a8985",
       "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e0"}
EXAMPLES = [("real-marker-d2-r04-eval", 39, "autonomous success"),
            ("real-marker-d2-r04-eval", 20, "autonomous failure"),
            ("real-marker-d2-r02-eval", 79, "timeout, stall at the holder")]


def centred_mean(x: np.ndarray, starts: np.ndarray, k: int) -> np.ndarray:
    """Within-episode centred moving average, same window as inference_choices.smooth (pandas center=True)."""
    out = np.empty_like(x)
    for a, b in zip(starts[:-1], starts[1:]):
        c = np.r_[0.0, np.cumsum(x[a:b])]
        i = np.arange(b - a)
        lo, hi = np.clip(i - k // 2, 0, b - a), np.clip(i - k // 2 + k, 0, b - a)
        out[a:b] = (c[hi] - c[lo]) / (hi - lo)
    return out


def advantage(V: np.ndarray, t: np.ndarray, T: np.ndarray, vT: np.ndarray, starts: np.ndarray, n: int):
    """n-step advantage per frame (frames sorted by episode, then frame; one row per frame 0..T)."""
    fut = np.empty_like(V)
    for a, b in zip(starts[:-1], starts[1:]):
        fut[a:b] = np.r_[V[a + n:b], np.full(min(n, b - a), np.nan)]
    term = t + n >= T
    A = np.where(term, -(T - t) / NORM + vT - V, -n / NORM + fut - V)
    return A, term


def load_val() -> pd.DataFrame:
    runs = [f"g0_r0_s{s}" for s in range(3)]
    ps = [pd.read_parquet(RUNS / r / "val_predictions.parquet") for r in runs]
    for p in ps[1:]:
        assert (p.episode_uid.values == ps[0].episode_uid.values).all() and (p.frame_index.values == ps[0].frame_index.values).all()
    d = ps[0].drop(columns=["V", "ce"]).copy()
    d["V_single"] = ps[0].V.values.astype(float)
    d["V_ens"] = np.mean([p.V.values.astype(float) for p in ps], 0)
    d = d.sort_values(["episode_uid", "frame_index"], kind="stable").reset_index(drop=True)
    starts = np.r_[0, np.flatnonzero(d.episode_uid.values[1:] != d.episode_uid.values[:-1]) + 1, len(d)]
    d["V_R0"] = centred_mean(d.V_ens.to_numpy(), starts, 8)
    cs = [pd.read_parquet(RUNS / f"c2a_stack2_s{s}" / "val_predictions.parquet") for s in range(3)]
    for c in cs:
        c = c.sort_values(["episode_uid", "frame_index"], kind="stable")
        assert (c.episode_uid.values == d.episode_uid.values).all() and (c.frame_index.values == d.frame_index.values).all()
    V_c2a = np.mean([c.sort_values(["episode_uid", "frame_index"], kind="stable").V.values.astype(float) for c in cs], 0)
    d["V_C2a"] = centred_mean(V_c2a, starts, 8)
    # time+round control from the training targets
    dcfg = load_config(load_config("configs/train.yaml")["data_config"])
    tr = load_split(dcfg, "train")
    tr = tr.assign(sec=tr.frame_index // FPS)
    table = tr.groupby(["repo", "sec"]).value.mean().rename("V_ctrl").reset_index()
    full = (table.groupby("repo").sec.agg(["min", "max"]).apply(lambda r: list(range(r["min"], 200)), axis=1)
            .explode().rename("sec").reset_index().astype({"sec": int}))
    table = full.merge(table, on=["repo", "sec"], how="left").sort_values(["repo", "sec"])
    table["V_ctrl"] = table.groupby("repo").V_ctrl.ffill()        # past the longest training episode: hold
    d = d.assign(sec=d.frame_index // FPS).merge(table, on=["repo", "sec"], how="left", validate="m:1")
    assert d.V_ctrl.notna().all(), "every val repo must have training episodes"
    d = d.sort_values(["episode_uid", "frame_index"], kind="stable").reset_index(drop=True)
    man = pd.read_parquet("data/manifest/episodes.parquet")[["episode_uid", "machine_failure_mode"]]
    d = d.merge(man, on="episode_uid", validate="m:1").merge(stall_seconds(), on=KEY, how="left", validate="1:1")
    assert d.stall_s.notna().all()
    d["t_s"], d["to_go"] = d.frame_index / FPS, (d["T"] - d.frame_index) / FPS
    d["vT"] = np.where(d.outcome == "success", 0.0, -0.5)
    return d.sort_values(["episode_uid", "frame_index"], kind="stable").reset_index(drop=True)


def ep_boot(ep_codes, num, den, rng):
    """Episode bootstrap of a pooled ratio sum(num)/sum(den); num/den per frame. Returns 2.5/97.5%."""
    n_ep = ep_codes.max() + 1
    N, D = np.bincount(ep_codes, num, n_ep), np.bincount(ep_codes, den, n_ep)
    idx = rng.integers(0, n_ep, (B, n_ep))
    r = N[idx].sum(1) / np.maximum(D[idx].sum(1), 1e-12)
    return np.percentile(r, [2.5, 97.5])


def diff_boot(ep_codes, pos, m1, m2, rng):
    n_ep = ep_codes.max() + 1
    s = [np.bincount(ep_codes, pos * m, n_ep) for m in (m1, m2)]
    c = [np.bincount(ep_codes, m.astype(float), n_ep) for m in (m1, m2)]
    idx = rng.integers(0, n_ep, (B, n_ep))
    with np.errstate(invalid="ignore"):                       # draws with no frame in a group are dropped
        r = s[0][idx].sum(1) / c[0][idx].sum(1) - s[1][idx].sum(1) / c[1][idx].sum(1)
    return np.nanpercentile(r, [2.5, 97.5])


def within_repo_auc(ep: pd.DataFrame, col: str) -> float:
    """Episode-level success AUC of `col`, averaged over eval repos that have both classes (round-free)."""
    aucs, w = [], []
    for _, g in ep[ep.repo.str.contains("-eval")].groupby("repo"):
        if g.succ.nunique() == 2 and g[col].notna().all():
            aucs.append(roc_auc_score(g.succ, g[col]))
            w.append(len(g))
    return float(np.average(aucs, weights=w))


def within_repo_spearman(ep: pd.DataFrame, col: str) -> float:
    rs, w = [], []
    for _, g in ep[ep.succ == 1].groupby("repo"):
        if len(g) >= 5 and g[col].nunique() > 1:
            rs.append(g.dur_s.rank().corr(g[col].rank()))
            w.append(len(g))
    return float(np.average(rs, weights=w))


def main():
    rng = np.random.default_rng(0)
    d = load_val()
    starts = np.r_[0, np.flatnonzero(d.episode_uid.values[1:] != d.episode_uid.values[:-1]) + 1, len(d)]
    t, T, vT = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.vT.to_numpy()
    A0, _ = advantage(d.value.to_numpy(float), t, T, vT, starts, 30)
    assert np.abs(A0).max() < 1e-9, "target-as-V must give zero advantages (arithmetic check)"
    ep_codes = pd.factorize(d.episode_uid)[0]
    hold = hold_groups(d)
    still_stall = d.index.isin(hold.index[(hold.group == "stall") & hold["still >= 4 s now"]])
    still_succ = d.index.isin(hold.index[hold.group.str.startswith("success") & hold["still >= 4 s now"]])
    moving_succ = d.index.isin(hold.index[hold.group.str.startswith("success") & ~hold["still >= 4 s now"]])
    succ = (d.outcome == "success").to_numpy()
    same_next = np.r_[d.episode_uid.values[1:] == d.episode_uid.values[:-1], False]

    sources = {"R0": "V_R0", "R0 single": "V_single", "C2a": "V_C2a", "time+round": "V_ctrl"}
    rows, labels = [], {}
    for name, col in sources.items():
        V = d[col].to_numpy(float)
        for ns in NS_S:
            n = ns * FPS
            A, term = advantage(V, t, T, vT, starts, n)
            d[f"A|{name}|{ns}"], d[f"term|{ns}"] = A, term
            for pname, pct in PCTS.items():
                thr = np.percentile(A, pct)
                pos = (A > thr).astype(float)
                labels[(name, ns, pname)] = pos
                inter = ~term
                r = {"V": name, "n_s": ns, "threshold": pname, "eps": thr,
                     "pos_terminal": pos[term].mean(), "pos_interior": pos[inter].mean()}
                for o in ["success", "failure", "timeout"]:
                    m = inter & (d.outcome == o).to_numpy()
                    r[f"int_{o}"] = pos[m].mean()
                m_s, m_f = inter & succ, inter & ~succ
                r["int_succ_minus_fail"] = pos[m_s].mean() - pos[m_f].mean()
                r["int_succ_minus_fail_ci"] = diff_boot(ep_codes, pos, m_s, m_f, rng).round(3).tolist()
                r["still_stall"], r["still_success"], r["moving_success"] = (pos[still_stall & inter].mean(),
                                                                             pos[still_succ & inter].mean(),
                                                                             pos[moving_succ & inter].mean())
                r["still_succ_minus_stall_ci"] = diff_boot(ep_codes, pos, still_succ & inter, still_stall & inter,
                                                           rng).round(3).tolist()
                flips = (pos[1:] != pos[:-1]) & same_next[:-1] & inter[1:] & inter[:-1]
                r["flips_per_s"] = flips.sum() / (same_next[:-1] & inter[1:] & inter[:-1]).sum() * FPS
                g = pd.DataFrame({"uid": d.episode_uid, "repo": d.repo, "succ": succ.astype(int), "pos": pos,
                                  "inter": inter, "dur_s": T / FPS})
                ep = (g[g.inter].groupby("uid").agg(repo=("repo", "first"), succ=("succ", "first"),
                                                    share=("pos", "mean"), dur_s=("dur_s", "first")))
                r["auc_within_round"] = within_repo_auc(ep, "share")
                r["speed_rho"] = within_repo_spearman(ep, "share")
                # mid-episode: the lookahead ends >= MID_S before T, so V never sees the final scene
                mid = (d.to_go > ns + MID_S).to_numpy()
                r["mid_succ"], r["mid_fail"] = pos[mid & succ].mean(), pos[mid & ~succ].mean()
                r["mid_succ_minus_fail_ci"] = diff_boot(ep_codes, pos, mid & succ, mid & ~succ, rng).round(3).tolist()
                g["inter"] = mid
                ep = (g[g.inter].groupby("uid").agg(repo=("repo", "first"), succ=("succ", "first"),
                                                    share=("pos", "mean"), dur_s=("dur_s", "first")))
                r["mid_auc_within_round"] = within_repo_auc(ep, "share")
                r["mid_still_success"], r["mid_still_stall"] = pos[still_succ & mid].mean(), pos[still_stall & mid].mean()
                r["mid_still_ci"] = diff_boot(ep_codes, pos, still_succ & mid, still_stall & mid, rng).round(3).tolist()
                # time-matched: label minus the mean label of mid frames in the same (repo, elapsed second), so
                # neither the round nor the clock (nor episode length) can produce a difference
                res_ = pd.Series(pos[mid]).groupby([d.repo.values[mid], d.sec.values[mid]]).transform("mean").to_numpy()
                resid = np.zeros(len(pos))
                resid[mid] = pos[mid] - res_
                r["tm_succ_minus_fail"] = resid[mid & succ].mean() - resid[mid & ~succ].mean()
                r["tm_succ_minus_fail_ci"] = diff_boot(ep_codes, resid, mid & succ, mid & ~succ, rng).round(3).tolist()
                r["tm_still"] = resid[still_succ & mid].mean() - resid[still_stall & mid].mean()
                r["tm_still_ci"] = diff_boot(ep_codes, resid, still_succ & mid, still_stall & mid, rng).round(3).tolist()
                if name != "R0":
                    r["agree_with_R0"] = (pos[inter] == labels[("R0", ns, pname)][inter]).mean()
                rows.append(r)
    res = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    res.to_csv(FIG / "advantage_checks.csv", index=False)
    figures(d, res, labels)
    write_md(d, res)


# ---------------------------------------------------------------- figures

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(COL["ink2"])
    ax.tick_params(colors=COL["ink2"], labelsize=9)
    ax.grid(axis="y", color=COL["grid"], lw=0.8)
    ax.set_axisbelow(True)


def figures(d, res, labels):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "axes.labelcolor": COL["ink2"], "figure.facecolor": "white"})
    n = MAIN_N
    term = d[f"term|{n}"].to_numpy()
    eps = res.query("V == 'R0' and n_s == @n and threshold == @MAIN_PCT").eps.item()

    # 1. distribution of interior advantages by outcome, R0 vs control
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    bins = np.linspace(-0.08, 0.08, 81)
    for ax, name in zip(axes, ["R0", "time+round"]):
        A = d[f"A|{name}|{n}"].to_numpy()
        for o in ["success", "failure", "timeout"]:
            m = ~term & (d.outcome == o).to_numpy()
            ax.hist(np.clip(A[m], bins[0], bins[-1]), bins, density=True, histtype="step", lw=2, color=COL[o],
                    label=f"{o} ({m.sum():,} frames)")
        e = res.query("V == @name and n_s == @n and threshold == @MAIN_PCT").eps.item()
        ax.axvline(e, color=COL["ink"], lw=1, ls="--")
        ax.text(e, ax.get_ylim()[1] * 0.95, f"  threshold ε = {e:+.4f}", fontsize=8, color=COL["ink"], va="top")
        ax.axvline(-n * FPS / NORM, color=COL["ink2"], lw=1, ls=":")
        ax.set_title(f"{name}: interior advantages, n = {n} s")
        ax.set_xlabel("advantage Â (normalized value units; tails clipped into the end bins)")
        style(ax)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].text(-n * FPS / NORM, axes[1].get_ylim()[1] * 0.55, "← cost of n s\n    (V unchanged)", fontsize=8,
                 color=COL["ink2"], ha="right")
    fig.suptitle("Where the advantages fall (val, frames whose lookahead stays inside the episode)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "01_distributions.png", dpi=150)
    plt.close(fig)

    # 2. the checks as a forest plot: raw vs time-matched differences in positive share, mid-episode frames
    checks = [("mid_succ", "mid_fail", "mid_succ_minus_fail_ci", "tm_succ_minus_fail", "tm_succ_minus_fail_ci",
               "Successes vs failures + timeouts"),
              ("mid_still_success", "mid_still_stall", "mid_still_ci", "tm_still", "tm_still_ci",
               "Gripper still ≥ 4 s at the holder:\npause before a seat vs stall")]
    srcs = ["R0", "R0 single", "C2a", "time+round"]
    sub = res[res.threshold == MAIN_PCT]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    ys, ticks = [], []
    for i, src in enumerate(srcs):
        for j, ns in enumerate(NS_S):
            ys.append(-(i * 4 + j))
            ticks.append(f"{src}, n = {ns} s")
    for ax, (a1, b1, ci1, tm, ci2, title) in zip(axes, checks):
        k = 0
        for src in srcs:
            for ns in NS_S:
                r = sub[(sub.V == src) & (sub.n_s == ns)].iloc[0]
                y = ys[k]
                raw = 100 * (r[a1] - r[b1])
                ax.plot(np.array(r[ci1]) * 100, [y + 0.18] * 2, color=COL["control"], lw=2)
                ax.scatter(raw, y + 0.18, s=44, facecolor="white", edgecolor=COL["control"], lw=2, zorder=3,
                           label="raw" if k == 0 else None)
                ax.plot(np.array(r[ci2]) * 100, [y - 0.18] * 2, color=COL["success"], lw=2)
                ax.scatter(100 * r[tm], y - 0.18, s=44, color=COL["success"], edgecolor="white", lw=1.5, zorder=3,
                           label="time-matched (same round, same second)" if k == 0 else None)
                k += 1
        ax.axvline(0, color=COL["ink2"], lw=1)
        ax.set_yticks(ys, ticks, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("difference in share labelled positive (percentage points)")
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=COL["grid"], lw=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, f"Mid-episode frames only (lookahead ends ≥ {MID_S} s before the end). Threshold: 30th "
             "percentile (70% positive). Bars: 95% episode-bootstrap intervals.\nRight of 0 = the label agrees "
             "with the eventual outcome. The control is 0 after time matching by construction.", fontsize=8,
             color=COL["ink2"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(FIG / "02_checks.png", dpi=150)
    plt.close(fig)

    # 3. example traces: V vs target, advantage, label strip
    fig, axes = plt.subplots(3, len(EXAMPLES), figsize=(13, 6.4), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1.6, 0.35]})
    for c, (repo, e, title) in enumerate(EXAMPLES):
        m = ((d.repo == repo) & (d.episode_index == e)).to_numpy()
        g = d[m]
        ts = g.t_s.to_numpy()
        ax = axes[0, c]
        ax.plot(ts, g.value, color=COL["ink2"], lw=1.5, ls="--", label="target (countdown)")
        ax.plot(ts, g.V_ctrl, color=COL["control"], lw=1.5, label="time+round control")
        ax.plot(ts, g.V_R0, color=COL["success"], lw=2, label="R0 V (ens ×3, 0.5 s)")
        ax.set_ylim(-1, 0)
        ax.set_title(f"{repo.split('-')[2]} ep {e}: {title}", fontsize=10)
        style(ax)
        ax = axes[1, c]
        A = g[f"A|R0|{n}"].to_numpy()
        tm = g[f"term|{n}"].to_numpy()
        ax.axhline(0, color=COL["ink2"], lw=0.8)
        ax.axhline(eps, color=COL["ink"], lw=1, ls="--")
        ax.plot(ts[~tm], np.clip(A[~tm], -0.12, 0.12), color=COL["success"], lw=1.5)
        ax.plot(ts[tm], np.clip(A[tm], -0.12, 0.12), color=COL["failure"], lw=1.5)
        ax.plot(ts, np.clip(g[f"A|time+round|{n}"], -0.12, 0.12), color=COL["control"], lw=1)
        ax.set_ylim(-0.12, 0.12)
        style(ax)
        ax = axes[2, c]
        pos = labels[("R0", n, MAIN_PCT)][m]
        ax.imshow(pos[None, :], aspect="auto", cmap=matplotlib.colors.ListedColormap([COL["failure"], COL["success"]]),
                  extent=(ts[0], ts[-1] + 1 / FPS, 0, 1), vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks([])
        ax.set_xlabel("seconds")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_ylabel("value")
    axes[1, 0].set_ylabel(f"Â, n = {n} s")
    axes[2, 0].set_ylabel("label", rotation=0, ha="right", va="center")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.01, 0.005, "Middle row: blue = R0 advantage (lookahead inside the episode), orange = lookahead "
             "reaches the end (known outcome used), gray = time+round control, dashed = threshold ε. "
             "Values clipped to ±0.12.", fontsize=8, color=COL["ink2"])
    axes[2, -1].text(1.0, -1.4, "label strip: blue = positive, orange = negative", transform=axes[2, -1].transAxes,
                     fontsize=8, color=COL["ink2"], ha="right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "03_examples.png", dpi=150)
    plt.close(fig)

    # 4. advantage vs time to the end, mean by outcome (shows where label information comes from)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    tb = np.arange(0, 30.5, 0.5)
    for ax, name in zip(axes, ["R0", "time+round"]):
        A = d[f"A|{name}|{n}"].to_numpy()
        for o in ["success", "failure", "timeout"]:
            g = pd.DataFrame({"A": A, "b": np.floor(d.to_go / 0.5) * 0.5})[(d.outcome == o).to_numpy()]
            mm = g[g.b <= 30].groupby("b").A.mean()
            ax.plot(mm.index, mm.values, color=COL[o], lw=2, label=o)
        ax.axvline(n, color=COL["ink2"], lw=1, ls=":")
        ax.text(n, 0.95, f" lookahead reaches\n the end (< {n} s)", transform=ax.get_xaxis_transform(),
                fontsize=8, color=COL["ink2"], va="top")
        ax.axhline(0, color=COL["ink2"], lw=0.8)
        ax.invert_xaxis()
        ax.set_xlabel("seconds before the episode ends")
        ax.set_title(f"{name}: mean Â by time to the end, n = {n} s")
        style(ax)
    axes[0].set_ylabel("mean advantage")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "04_time_to_end.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def write_md(d, res):
    def tbl(sub, cols, names):
        head = "| " + " | ".join(names) + " |\n|" + "---|" * len(names) + "\n"
        return head + "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in sub.iterrows())

    def fmt(x):
        if isinstance(x, (list, np.ndarray)):
            return f"[{x[0]:+.3f}, {x[1]:+.3f}]"
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else f"{x:.3f}" if abs(x) >= 0.001 else f"{x:+.5f}"
        return str(x)

    cols = ["V", "n_s", "eps", "pos_terminal", "int_success", "int_failure", "int_timeout", "int_succ_minus_fail",
            "int_succ_minus_fail_ci", "auc_within_round", "speed_rho", "still_success", "still_stall",
            "still_succ_minus_stall_ci", "moving_success", "flips_per_s", "agree_with_R0"]
    mcols = ["V", "n_s", "mid_succ", "mid_fail", "mid_succ_minus_fail_ci", "mid_auc_within_round", "mid_still_success",
             "mid_still_stall", "mid_still_ci", "tm_succ_minus_fail", "tm_succ_minus_fail_ci", "tm_still", "tm_still_ci"]
    mnames = ["V", "n (s)", "pos succ", "pos fail+timeout", "succ − fail 95% CI", "AUC within round", "still: succ",
              "still: stall", "succ − stall 95% CI", "time-matched succ − fail", "95% CI",
              "time-matched still succ − stall", "95% CI"]
    names = ["V", "n (s)", "ε", "pos, terminal", "pos succ", "pos fail", "pos timeout", "succ − fail",
             "95% CI", "AUC within round", "speed ρ", "still: succ", "still: stall", "succ − stall 95% CI",
             "moving succ", "flips/s", "agree w/ R0"]
    n_int = (~d[f"term|{MAIN_N}"]).sum()
    lines = [
        "# Advantage labels from R0: do they make sense? (Stage 6 first step, val only)",
        "",
        f"`.venv/bin/python -m vl.advantages` → this file, `{FIG}/` (figures, `advantage_checks.csv`).",
        "Definitions, controls and thresholds: the module docstring of `src/vl/advantages.py`. "
        f"{d.episode_uid.nunique()} val episodes, {len(d):,} frames ({n_int:,} interior at n = {MAIN_N} s). "
        "Intervals: episode bootstrap, 1000 draws. Arithmetic check passed: with the target as V every advantage is 0.",
        "",
        "Columns: *pos* = share of frames labelled positive; *interior* = lookahead stays inside the episode (A depends "
        "on V only); *AUC within round* = episode-level success AUC of the interior positive share, inside each eval "
        "round (round-free), weighted mean; *speed ρ* = Spearman of success duration vs interior positive share inside "
        "each repo (negative = faster successes get more positives); *still* = hold phase, gripper still ≥ 4 s now "
        "(`vl.history_gain.hold_groups`): pause before a seat (success) vs stall (timeout); *moving succ* = success "
        "hold frames while moving; *flips/s* = label changes per second on interior frames.",
        "",
    ]
    for pname in PCTS:
        lines += [f"## Threshold: {pname} (ε = {PCTS[pname]}th percentile of val advantages)", "",
                  tbl(res[res.threshold == pname], cols, names), "",
                  f"Mid-episode only (lookahead ends ≥ {MID_S} s before T, so V never sees the final scene):", "",
                  tbl(res[res.threshold == pname], mcols, mnames), ""]
    old = OUT_MD.read_text() if OUT_MD.exists() else ""
    reading = old[old.index("## Reading"):] if "## Reading" in old else "## Reading\n\n(to be written)\n"
    OUT_MD.write_text("\n".join(lines) + reading)       # the hand-written reading survives a re-run
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
