"""Stage 5 first step (docs/libero_plan.md): RECAP-style advantage labels on the LIBERO val rollouts from the
ResNet-18 value models, and whether they make sense. The LIBERO port of vl.advantages (station data).

Usage:
    .venv/bin/python -m vl.libero_advantages   # -> reports/libero_advantages.md, reports/figures/libero_advantages/

No training, val only: the ResNets were trained on the 30 train init states and never saw the 10 value-val ones,
so their val predictions are out-of-sample and no out-of-fold V is needed here. Labels for the *train* rollouts
(what Stage 6 arm C trains on) still need the out-of-fold V of the plan.

Advantage [paper: RECAP's n-step estimate, no discount], targets of vl.libero_targets (horizon 520 + failure
penalty 520, so one step costs 1/1040 and a failure ends at -0.5):
    A_t = -n/1040 + V(o_{t+n}) - V(o_t)                      if t + n < T   ("interior": depends on V only)
    A_t = -(T - t)/1040 + v_T - V(o_t),  v_T = 0 / -0.5      if t + n >= T  ("terminal": the outcome enters)
With V = the training target every A_t is 0 (checked in main).
Frames: the every-4th-frame grid of the val predictions (frame_index % 4 == 0), so n must be a multiple of 4:
n = 12, 20, 48 frames = 1.2, 2, 4.8 s [ours: the lags the history runs already use].

Value functions compared:
  ResNet       3-seed ensemble (resnet18_18166232, _s1_18174529, _s2_18174726), expected value of the raw
               softmax [ours: no smoothing; at 2.5 sampled frames/s a 0.5 s window is one frame]
  ResNet s0    resnet18_18166232 alone (what the ensemble buys)
  ResNet+2s    3-seed ensemble of the C2a arm (current + 2 s-old frame; c2a20_18167078, hist20_s1/_s2)
  B            3-seed ensemble of the Stage 3b π₀.₅ value expert, arm B (w256 + augmentation;
               value_expert_aug_w256_s{0,1,2}), same frames; added 2026-10-30 for the B vs ResNet comparison
  ResNet ff    3-seed ensemble of the ResNet trained on fixed + fresh layouts (825 fit episodes; added 2026-11-02:
               the Stage 6 candidates)
  B ff         3-seed ensemble of frozen-π₀.₅ arm B on fixed + fresh layouts (value_expert_fixedfresh_w256_s*)
  B s0         value_expert_aug_w256_s0_18186728 alone
  clock        [ours, control] mean train target per frame index. Sees no image. If its labels separate
               outcomes as well as the ResNet's, the images add nothing to the advantages.

Threshold [paper, ambiguous; as vl.advantages]: the 30th percentile of the val advantages (70% positive) and the
70th (30% positive), one per V and n.
Checks [ours]:
  - succ - fail: share positive on interior frames of successes minus failures; 95% CI by scene bootstrap
    (10 val init states, as the milestone checks: episodes of one scene are not independent)
  - AUC within scene: episode-level success AUC of the interior positive share inside each init state that has
    both outcomes, weighted mean (scene difficulty cannot produce it)
  - mid-episode: frames whose lookahead ends >= 5 s before T, so V never sees the final scene; time-matched =
    label minus the mean label of mid frames in the same scene and frame index, so neither the scene nor the clock
    can produce a difference; time-matched AUC = the within-scene AUC of each episode's mean time-matched label.
    The raw shares and AUCs are confounded by episode length (every failure runs the full 52 s, so its interior
    frames cover later times than a success's): the clock control alone gets AUC 1.0 on them. Read the
    time-matched columns
  - by failure type (scripts/libero_failure_types.py), terminal share by outcome, flips per second
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vl.advantages import diff_boot
from vl.metrics import auc

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
DATASET = Path("/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2")
OUT_MD = Path("reports/libero_advantages.md")
FIG = Path("reports/figures/libero_advantages")
FPS = 10
NORM = 1040.0
V_FAIL = -0.5
NS = [12, 20, 48]
PCTS = {"70% positive": 30, "30% positive": 70}
MAIN_N, MAIN_PCT = 20, "70% positive"
MID_S = 5
TM_BIN = 4  # time-matching bin, frames: one grid frame, so the clock control is exactly 0
ENSEMBLES = {"ResNet": ["resnet18_18166232", "resnet18_s1_18174529", "resnet18_s2_18174726"],
             "ResNet+2s": ["resnet18_c2a20_18167078", "resnet18_hist20_s1_18174530", "resnet18_hist20_s2_18174732"],
             "B": ["value_expert_aug_w256_s0_18186728", "value_expert_aug_w256_s1_18189370",
                   "value_expert_aug_w256_s2_18189371"],
             "ResNet ff": ["resnet18_fixedfresh_s0_18237895", "resnet18_fixedfresh_s1_18237896",
                           "resnet18_fixedfresh_s2_18237897"],
             "B ff": ["value_expert_fixedfresh_w256_s0_18237898", "value_expert_fixedfresh_w256_s1_18237899",
                      "value_expert_fixedfresh_w256_s2_18237900"]}
EV_COL = {"B": "ev_expert", "B ff": "ev_expert"}  # E[V] column of each ensemble's val_predictions (default ev_resnet)
SOURCES = ["ResNet", "ResNet s0", "ResNet+2s", "B", "B s0", "ResNet ff", "B ff", "clock"]
KEY = ["episode_index", "frame_index"]
COL = {"success": "#2a78d6", "failure": "#eb6834", "control": "#8a8985", "ink": "#0b0b0b", "ink2": "#52514e",
       "grid": "#e6e5e0"}


def advantage(V, t, T, vT, ep, n):
    """n-step advantage per frame; rows sorted by episode then frame on a grid of step 4 (n % 4 == 0)."""
    assert n % 4 == 0
    k = n // 4
    same = np.r_[ep[k:] == ep[:-k], np.zeros(k, bool)]
    fut = np.r_[V[k:], np.full(k, np.nan)]
    term = t + n >= T
    assert (same | term).all(), "an interior frame's lookahead must be on the grid"
    A = np.where(term, -(T - t) / NORM + vT - V, -n / NORM + fut - V)
    return A, term


def load_val() -> pd.DataFrame:
    base = None
    for name, runs in ENSEMBLES.items():
        ps = [pd.read_parquet(RUNS / r / "val_predictions.parquet").sort_values(KEY).reset_index(drop=True)
              for r in runs]
        if base is None:
            base = ps[0][KEY + ["init_state", "T", "success", "value"]].copy()
            base["V_ResNet s0"] = ps[0].ev_resnet.values.astype(float)
        col = EV_COL.get(name, "ev_resnet")
        for p in ps:
            assert (p[KEY].values == base[KEY].values).all()
        base[f"V_{name}"] = np.mean([p[col].values.astype(float) for p in ps], 0)
        if name == "B":
            base["V_B s0"] = ps[0][col].values.astype(float)
    ft = pd.read_parquet(DATASET / "frame_targets.parquet", columns=["episode_index", "frame_index", "value", "split"])
    clock = ft[ft.split == "train"].groupby("frame_index").value.mean().rename("V_clock")
    base = base.merge(clock, left_on="frame_index", right_index=True, how="left", validate="m:1")
    assert base.V_clock.notna().all()
    eps = pd.read_csv(DATASET / "episodes.csv")
    ftypes = pd.read_csv(RUNS / "stage2_moka_18155849" / "val" / "failure_types.csv")
    eps = eps[eps.split == "val"].merge(ftypes[["episode", "type"]], left_on="source_episode", right_on="episode",
                                        how="left")
    assert (eps.type.isna() == (eps.success == 1)).all()
    base = base.merge(eps[["episode_index", "type"]], on="episode_index", validate="m:1")
    base["type"] = base.type.fillna("success")
    base = base.sort_values(KEY).reset_index(drop=True)
    base["vT"] = np.where(base.success == 1, 0.0, V_FAIL)
    base["to_go_s"] = (base["T"] - base.frame_index) / FPS
    return base


def within_scene_auc(ep: pd.DataFrame) -> float:
    aucs, w = [], []
    for _, g in ep.groupby("init_state"):
        if g.succ.nunique() == 2:
            aucs.append(auc(g.succ.values, g.share.values))
            w.append(len(g))
    return float(np.average(aucs, weights=w))


def checks(d, rng):
    t, T, vT, ep = (d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.vT.to_numpy(),
                    d.episode_index.to_numpy())
    A0, _ = advantage(d.value.to_numpy(float), t, T, vT, ep, 20)
    assert np.abs(A0).max() < 1e-9, "target-as-V must give zero advantages (arithmetic check)"
    scene = pd.factorize(d.init_state)[0]
    succ = (d.success == 1).to_numpy()
    same_next = np.r_[ep[1:] == ep[:-1], False]
    tm_cell = d.init_state.values * 1000 + d.frame_index.values // TM_BIN
    rows, labels = [], {}
    for name in SOURCES:
        V = d[f"V_{name}"].to_numpy(float)
        for n in NS:
            A, term = advantage(V, t, T, vT, ep, n)
            d[f"A|{name}|{n}"], d[f"term|{n}"] = A, term
            inter = ~term
            for pname, pct in PCTS.items():
                thr = np.percentile(A, pct)
                pos = (A > thr).astype(float)
                labels[(name, n, pname)] = pos
                r = {"V": name, "n_s": n / FPS, "threshold": pname, "eps": thr,
                     "term_succ": pos[term & succ].mean(), "term_fail": pos[term & ~succ].mean(),
                     "int_succ": pos[inter & succ].mean(), "int_fail": pos[inter & ~succ].mean()}
                r["int_diff"] = r["int_succ"] - r["int_fail"]
                r["int_diff_ci"] = diff_boot(scene, pos, inter & succ, inter & ~succ, rng).round(3).tolist()
                g = pd.DataFrame({"e": ep, "init_state": d.init_state.values, "succ": succ.astype(int), "pos": pos})
                r["auc_scene"] = within_scene_auc(g[inter].groupby("e").agg(
                    init_state=("init_state", "first"), succ=("succ", "first"), share=("pos", "mean")))
                mid = (d.to_go_s > n / FPS + MID_S).to_numpy()
                r["mid_succ"], r["mid_fail"] = pos[mid & succ].mean(), pos[mid & ~succ].mean()
                r["mid_diff_ci"] = diff_boot(scene, pos, mid & succ, mid & ~succ, rng).round(3).tolist()
                r["mid_auc_scene"] = within_scene_auc(g[mid].groupby("e").agg(
                    init_state=("init_state", "first"), succ=("succ", "first"), share=("pos", "mean")))
                resid = np.zeros(len(pos))
                resid[mid] = pos[mid] - pd.Series(pos[mid]).groupby(tm_cell[mid]).transform("mean").to_numpy()
                r["tm_diff"] = resid[mid & succ].mean() - resid[mid & ~succ].mean()
                r["tm_diff_ci"] = diff_boot(scene, resid, mid & succ, mid & ~succ, rng).round(3).tolist()
                g["pos"] = resid
                r["tm_auc_scene"] = within_scene_auc(g[mid].groupby("e").agg(
                    init_state=("init_state", "first"), succ=("succ", "first"), share=("pos", "mean")))
                for ty in ["stuck_pot2", "stuck_placing_pot2", "near_miss", "stalled_earlier", "knocked_over",
                           "stuck_pot1", "slow"]:
                    m = inter & (d.type == ty).to_numpy()
                    r[f"type_{ty}"] = pos[m].mean() if m.any() else np.nan
                ok = same_next[:-1] & inter[1:] & inter[:-1]
                r["flips_per_s"] = ((pos[1:] != pos[:-1]) & ok).sum() / ok.sum() * FPS / 4
                if name != "ResNet":
                    r["agree_with_ResNet"] = (pos[inter] == labels[("ResNet", n, pname)][inter]).mean()
                rows.append(r)
    return pd.DataFrame(rows), labels


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
    # 1. checks forest: raw vs time-matched mid-episode succ - fail
    sub = res[res.threshold == MAIN_PCT]
    srcs = SOURCES
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    ys, ticks, k = [], [], 0
    for i, src in enumerate(srcs):
        for j, nn in enumerate(NS):
            y = -(i * 4 + j)
            ys.append(y)
            ticks.append(f"{src}, n = {nn / FPS:g} s")
            r = sub[(sub.V == src) & (sub.n_s == nn / FPS)].iloc[0]
            ax.plot(np.array(r.mid_diff_ci) * 100, [y + 0.18] * 2, color=COL["control"], lw=2)
            ax.scatter(100 * (r.mid_succ - r.mid_fail), y + 0.18, s=44, facecolor="white", edgecolor=COL["control"],
                       lw=2, zorder=3, label="raw" if k == 0 else None)
            ax.plot(np.array(r.tm_diff_ci) * 100, [y - 0.18] * 2, color=COL["success"], lw=2)
            ax.scatter(100 * r.tm_diff, y - 0.18, s=44, color=COL["success"], edgecolor="white", lw=1.5, zorder=3,
                       label="time-matched (same scene, same frame)" if k == 0 else None)
            k += 1
    ax.axvline(0, color=COL["ink2"], lw=1)
    ax.set_yticks(ys, ticks, fontsize=9)
    ax.set_xlabel("successes − failures, share labelled positive (percentage points)")
    ax.set_title("Do the labels agree with the outcome? (val, mid-episode frames)")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.01, f"Lookahead ends ≥ {MID_S} s before the end. Threshold: 30th percentile (70% positive). "
             "Bars: 95% scene bootstrap (10 init states).", fontsize=8, color=COL["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "01_checks.png", dpi=150)
    plt.close(fig)

    # 2. example traces: one success and the two commonest failure types, from one scene where possible
    ex = []
    for ty in ["success", "stuck_pot2", "near_miss"]:
        e = d[d.type == ty].episode_index.unique()
        if len(e):
            ex.append((int(e[0]), ty))
    eps_ = res.query("V == 'ResNet' and n_s == @n / @FPS and threshold == @MAIN_PCT").eps.item()
    fig, axes = plt.subplots(3, len(ex), figsize=(4.3 * len(ex), 6.2), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1.6, 0.35]}, squeeze=False)
    for c, (e, ty) in enumerate(ex):
        m = (d.episode_index == e).to_numpy()
        g = d[m]
        ts = g.frame_index.to_numpy() / FPS
        ax = axes[0, c]
        ax.plot(ts, g.value, color=COL["ink2"], lw=1.5, ls="--", label="target")
        ax.plot(ts, g.V_clock, color=COL["control"], lw=1.5, label="clock control")
        ax.plot(ts, g["V_ResNet"], color=COL["success"], lw=2, label="ResNet V (ens ×3)")
        ax.set_ylim(-1, 0)
        ax.set_title(f"ep {e} (scene {int(g.init_state.iloc[0])}): {ty}", fontsize=10)
        style(ax)
        ax = axes[1, c]
        A, tm = g[f"A|ResNet|{n}"].to_numpy(), g[f"term|{n}"].to_numpy()
        ax.axhline(0, color=COL["ink2"], lw=0.8)
        ax.axhline(eps_, color=COL["ink"], lw=1, ls="--")
        ax.plot(ts[~tm], np.clip(A[~tm], -0.15, 0.15), color=COL["success"], lw=1.5)
        ax.plot(ts[tm], np.clip(A[tm], -0.15, 0.15), color=COL["failure"], lw=1.5)
        ax.set_ylim(-0.15, 0.15)
        style(ax)
        ax = axes[2, c]
        pos = labels[("ResNet", n, MAIN_PCT)][m]
        ax.imshow(pos[None, :], aspect="auto", interpolation="nearest", vmin=0, vmax=1,
                  cmap=matplotlib.colors.ListedColormap([COL["failure"], COL["success"]]),
                  extent=(ts[0], ts[-1] + 0.4, 0, 1))
        ax.set_yticks([])
        ax.set_xlabel("seconds")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_ylabel("value")
    axes[1, 0].set_ylabel(f"Â, n = {n / FPS:g} s")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.01, 0.005, "Middle: blue = advantage with the lookahead inside the episode, orange = lookahead "
             "reaches the end (outcome used), dashed = threshold. Strip: blue = positive label, orange = negative.",
             fontsize=8, color=COL["ink2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "02_examples.png", dpi=150)
    plt.close(fig)


def fmt(x):
    if isinstance(x, (list, np.ndarray)):
        return f"[{x[0]:+.3f}, {x[1]:+.3f}]"
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(x) else f"{x:.3f}" if abs(x) >= 0.001 else f"{x:+.5f}"
    return str(x)


def tbl(sub, cols, names):
    head = "| " + " | ".join(names) + " |\n|" + "---|" * len(names) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in sub.iterrows())


def write_md(d, res):
    n_ep = d.episode_index.nunique()
    n_s = int(d.groupby("episode_index").success.first().sum())
    types = d.groupby("episode_index").type.first().value_counts()
    cols = ["V", "n_s", "eps", "term_succ", "term_fail", "int_succ", "int_fail", "int_diff", "int_diff_ci",
            "auc_scene", "flips_per_s", "agree_with_ResNet"]
    names = ["V", "n (s)", "ε", "terminal: succ", "terminal: fail", "pos succ", "pos fail", "succ − fail", "95% CI",
             "AUC within scene", "flips/s", "agree w/ ResNet"]
    mcols = ["V", "n_s", "mid_succ", "mid_fail", "mid_diff_ci", "mid_auc_scene", "tm_diff", "tm_diff_ci",
             "tm_auc_scene"]
    mnames = ["V", "n (s)", "pos succ", "pos fail", "succ − fail 95% CI", "AUC within scene",
              "time-matched succ − fail", "95% CI", "time-matched AUC within scene"]
    tcols = ["V", "n_s", "int_succ"] + [c for c in res.columns if c.startswith("type_")]
    tnames = ["V", "n (s)", "success"] + [c[5:] for c in tcols[3:]]
    lines = [
        "# LIBERO advantage labels from the ResNet and π₀.₅ (3b arm B) V: do they make sense? (Stage 5 first step, "
        "val only)", "",
        f"`.venv/bin/python -m vl.libero_advantages` → this file, `{FIG}/` (figures, `advantage_checks.csv`, "
        "`val_labels.parquet`). Definitions, controls and thresholds: the module docstring of "
        "`src/vl/libero_advantages.py`. "
        f"{n_ep} val episodes ({n_s} successes), {len(d):,} frames (every 4th), 10 init states. Intervals: scene "
        "bootstrap, 1000 draws. Arithmetic check passed: with the target as V every advantage is 0.", "",
        "Failure types (val): " + ", ".join(f"{k} {v}" for k, v in types.items() if k != "success") + ".", "",
        "Columns: *terminal* = lookahead reaches the end, so the known outcome enters; *pos succ / fail* = share "
        "positive on interior frames (A depends on V only); *AUC within scene* = episode-level success AUC of the "
        "interior positive share inside each init state with both outcomes, weighted mean; *flips/s* = label changes "
        "per second of interior frames.", ""]
    for pname in PCTS:
        sub = res[res.threshold == pname]
        lines += [f"## Threshold: {pname} (ε = {PCTS[pname]}th percentile of val advantages)", "",
                  tbl(sub, cols, names), "",
                  f"Mid-episode only (lookahead ends ≥ {MID_S} s before T, so V never sees the final scene):", "",
                  tbl(sub, mcols, mnames), "",
                  "Share positive on interior frames by failure type:", "", tbl(sub, tcols, tnames), ""]
    old = OUT_MD.read_text() if OUT_MD.exists() else ""
    reading = old[old.index("## Reading"):] if "## Reading" in old else "## Reading\n\n(to be written)\n"
    OUT_MD.write_text("\n".join(lines) + reading)       # the hand-written reading survives a re-run
    print(OUT_MD.read_text())


def main():
    rng = np.random.default_rng(0)
    d = load_val()
    res, labels = checks(d, rng)
    FIG.mkdir(parents=True, exist_ok=True)
    res.to_csv(FIG / "advantage_checks.csv", index=False)
    keep = KEY + ["init_state", "success", "type"] + [c for c in d.columns if c.startswith(("V_", "A|", "term|"))]
    out = d[keep].copy()
    for (name, n, pname), pos in labels.items():
        out[f"pos|{name}|{n}|{PCTS[pname]}"] = pos.astype(np.int8)
    out.to_parquet(FIG / "val_labels.parquet")
    figures(d, res, labels)
    write_md(d, res)


if __name__ == "__main__":
    main()
