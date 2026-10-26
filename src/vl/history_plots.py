"""Figures for the short-history probe: where does memory change V? [ours]

Reads the per-frame val predictions written by `vl.history_probe --save-preds` (mean over head seeds)
and the proprio table, and writes four figures to reports/figures/history/:
  1_psucc_vs_stall.png      p_succ vs seconds the gripper has been still, hold phase, stalls vs successes
  2_episode_scatter.png     per-episode hold-phase mean p_succ, single frame vs memory
  3_traces.png              V traces for the known worst stalls and slow successes
  4_error_by_time.png       |V - target| by time into the episode and outcome

Usage: .venv/bin/python -m vl.history_plots --preds /move/u/karenvo/vl-cache/history_local/aug_split_cropwrist_s0_stride6_preds.parquet
Error bars: 95% bootstrap over episodes (1,000 resamples).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {"ctrl": ("single frame", "#2a78d6", "o"),
          "masked_24": ("+ frame 24 s back", "#eb6834", "s"),
          "masked_8": ("+ frame 8 s back", "#1baf7a", "^")}
OUTCOME = {"stall": ("stall timeout", "#6250d6", "x"), "success": ("success", "#1baf7a", "o")}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"
STALLS = [("r04", 101), ("r02", 44), ("r02", 45), ("r01", 48)]
SLOW_SUCC = [("r03", 68), ("r02", 42), ("r01", 103)]
STILL_MPS = 0.01   # [ours] gripper "still" below 1 cm/s over the last 0.5 s, as in the proprio stall check


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def boot_ci(df: pd.DataFrame, col: str, n=1000, seed=0):
    """Mean of `col` over frames and its 95% CI from resampling whole episodes."""
    g = df.groupby("episode_uid")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(g), (n, len(g)))
    b = s[k].sum(1) / c[k].sum(1)
    return s.sum() / c.sum(), *np.percentile(b, [2.5, 97.5])


def stall_seconds() -> pd.DataFrame:
    """Seconds since the gripper last moved faster than STILL_MPS (causal, 15 Hz proprio)."""
    p = pd.read_parquet("data/proprio/frame_proprio.parquet").sort_values(["repo", "episode_index", "frame_index"])
    g = p.groupby(["repo", "episode_index"])
    speed = np.sqrt(sum((p[a] - g[a].shift(7)) ** 2 for a in "xyz")) / (7 / 15)
    still = speed.fillna(np.inf) < STILL_MPS
    key = [p.repo, p.episode_index]
    run = (~still).groupby(key).cumsum()
    p["stall_s"] = still.groupby(key + [run]).cumsum() / 15
    return p[["repo", "episode_index", "frame_index", "stall_s"]]


def pick(d, rd, e):
    return d[d.repo.str.contains(f"{rd}-eval") & (d.episode_index == e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="reports/figures/history")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(args.preds).merge(stall_seconds(), on=["repo", "episode_index", "frame_index"], how="left",
                                          validate="1:1")
    assert d.stall_s.notna().all()
    d["t"] = d.frame_index / 15
    d["kind"] = np.where(d.outcome == "success", "success",
                         np.where(d.machine_failure_mode == "timeout_holding_marker", "stall", "other"))
    hold = d[d.repo.str.contains("eval") & d.kind.isin(["stall", "success"]) & (d.t > 5) & (d["T"] / 15 - d.t > 2)]
    for m in MODELS:
        d[f"err_{m}"] = (d[f"V_{m}"] - d.value).abs()

    # 1. p_succ vs stall duration
    edges = [0, 1, 2, 4, 8, 16, 32, 1e9]
    labels = ["<1", "1–2", "2–4", "4–8", "8–16", "16–32", "32+"]
    hold = hold.assign(sb=pd.cut(hold.stall_s, edges, right=False, labels=labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind in zip(axes, ["stall", "success"]):
        h = hold[hold.kind == kind]
        for m, (name, col, mk) in MODELS.items():
            r = [boot_ci(h[h.sb == b], f"p_succ_{m}") if (h.sb == b).any() else (np.nan,) * 3 for b in labels]
            r = np.array(r)
            x = np.arange(len(labels))
            ax.plot(x, r[:, 0], color=col, lw=2, marker=mk, ms=7, label=name)
            ax.fill_between(x, r[:, 1], r[:, 2], color=col, alpha=0.12, lw=0)
        n = h.groupby("sb", observed=False).episode_uid.nunique()
        ax.set_xticks(range(len(labels)), [f"{b}\n(n={n[b]})" for b in labels])
        ax.set_title(f"{OUTCOME[kind][0]} episodes", color=INK, loc="left")
        ax.set_xlabel("seconds the gripper has been still (n = episodes)", color=INK)
        style(ax)
    axes[0].set_ylabel("mean predicted P(success)", color=INK)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Hold phase: does memory lower P(success) as a stall goes on, without touching successes?",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "1_psucc_vs_stall.png", dpi=150)
    plt.close(fig)

    # 2. per-episode scatter, single frame vs memory
    e = hold.groupby("episode_uid").agg(kind=("kind", "first"), a=("p_succ_ctrl", "mean"), b=("p_succ_masked_24", "mean"),
                                        dur=("t", "max"))
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for kind, (name, col, mk) in OUTCOME.items():
        q = e[e.kind == kind]
        lower = (q.b < q.a).mean()
        ax.scatter(q.a, q.b, s=np.clip(q.dur, 8, 50) * 2, color=col, marker=mk, lw=1.5 if mk == "x" else 0.8,
                   edgecolors="white" if mk == "o" else None, alpha=0.85,
                   label=f"{name} (n={len(q)}; {lower:.0%} moved down)")
    ax.set_xlabel("single frame: hold-phase mean P(success)", color=INK)
    ax.set_ylabel("+ frame 24 s back: hold-phase mean P(success)", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.97, 0.03, "below the line = memory\nlowered P(success)", ha="right", va="bottom", color=MUTED,
            transform=ax.transAxes, fontsize=9)
    ax.set_title("Per episode (marker size = hold-phase length)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "2_episode_scatter.png", dpi=150)
    plt.close(fig)

    # 3. traces
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharey=True)
    cases = [(c, "stall timeout") for c in STALLS] + [(c, "success") for c in SLOW_SUCC]
    for ax, ((rd, ep), what) in zip(axes.T.ravel(), cases):
        q = pick(d, rd, ep)
        if q.empty:
            ax.set_visible(False)
            continue
        still = q.stall_s.to_numpy() > 2
        ax.fill_between(q.t, -1, 0, where=still, color=GRID, alpha=0.9, lw=0, step="mid")
        if q.t.max() > 24:
            ax.axvspan(24, q.t.max(), color="#eb6834", alpha=0.06, lw=0)
        ax.plot(q.t, q.value, color=INK, lw=2, label="target")
        for m in ("ctrl", "masked_24"):
            name, col, _ = MODELS[m]
            ax.plot(q.t, q[f"V_{m}"], color=col, lw=2, label=name)
        mae = {m: q[f"err_{m}"].mean() for m in ("ctrl", "masked_24")}
        ax.set_title(f"{rd}/{ep} ({what}, {q.t.max():.0f} s): MAE {mae['ctrl']:.3f} → {mae['masked_24']:.3f}",
                     color=INK, loc="left", fontsize=10)
        ax.set_ylim(-1, 0)
        style(ax)
    axes[-1, -1].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("seconds into episode", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("V", color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Rectangle((0, 0), 1, 1, color=GRID), plt.Rectangle((0, 0), 1, 1, color="#eb6834", alpha=0.15)]
    l += ["gripper still > 2 s", "memory has a real past frame (> 24 s in)"]
    fig.legend(h, l, loc="lower right", bbox_to_anchor=(0.97, 0.06), frameon=False)
    fig.suptitle("Worst stall timeouts (left) and slow successes (right)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "3_traces.png", dpi=150)
    plt.close(fig)

    # 4. error by time into episode
    d["tb"] = pd.cut(d.t, [0, 8, 24, 1e9], right=False, labels=["0–8 s", "8–24 s", "> 24 s"])
    kinds = [("success", d.outcome == "success"), ("stall timeout", d.kind == "stall"),
             ("other failure / timeout", (d.outcome != "success") & (d.kind != "stall"))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    w = 0.26
    for ax, (name, mask) in zip(axes, kinds):
        q = d[mask]
        for j, (m, (mname, col, _)) in enumerate(MODELS.items()):
            r = np.array([boot_ci(q[q.tb == b], f"err_{m}") if (q.tb == b).any() else (np.nan,) * 3
                          for b in q.tb.cat.categories])
            x = np.arange(3) + (j - 1) * w
            ax.bar(x, r[:, 0], w * 0.92, color=col, label=mname)
            ax.errorbar(x, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], fmt="none", ecolor=INK, lw=1, capsize=2)
        n = q.groupby("tb", observed=False).episode_uid.nunique()
        ax.set_xticks(range(3), [f"{b}\n(n={n[b]})" for b in q.tb.cat.categories])
        ax.set_title(name, color=INK, loc="left")
        style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean |V − target|", color=INK)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Value error by time into the episode (n = episodes with frames in that bucket)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "4_error_by_time.png", dpi=150)
    plt.close(fig)

    # numbers behind the figures
    print("hold-phase episodes:", e.kind.value_counts().to_dict())
    print("share moved down (masked_24 < ctrl):", e.assign(dn=e.b < e.a).groupby("kind").dn.mean().round(3).to_dict())
    for name, mask in kinds:
        q = d[mask]
        print(name, {b: {m: round(q.loc[q.tb == b, f"err_{m}"].mean(), 4) for m in MODELS} for b in q.tb.cat.categories})


if __name__ == "__main__":
    main()
