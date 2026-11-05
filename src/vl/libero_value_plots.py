"""Value plots for the LIBERO Stage 3a probes (docs/libero_plan.md Stage 4 view), one figure per chosen val episode.

Usage:
    .venv/bin/python -m vl.libero_value_plots \
        --probe /iris/u/ankile/karenvo/libero/features/feats_every4_18163142_oc \
        --resnet /iris/u/ankile/karenvo/libero/runs/resnet18_18166232
        -> reports/figures/libero_stage3a/<k>_<kind>_ep<idx>.png  (filmstrip + curves), overview.png

Curves (every 4th frame, as the probes were scored): target v (black; straight by construction), E[V] of the
201-bin probe on the all-token pool (orange), E[V] of the outcome+clock probe (blue) and its P(success) (green,
right axis), the ResNet-18 control (purple), the clock (grey dashed). Dotted verticals: first step at which each
pot satisfies On(pot, stove cook region) (from the raw goal predicates). Any drop or rise off the black line
comes from the model.
Episodes, one per kind, fixed rule: successes and each Stage 2 failure type (scripts/libero_failure_types.py),
the episode whose MAE (201-bin probe) is the median of its kind; plus the worst-MAE episode.
overview.png: mean E[V] and P(success) against time, successes vs each failure type.
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


def episode_table() -> pd.DataFrame:
    eps = pd.read_csv(DATASET / "episodes.csv")
    ft = pd.read_csv(RUN / "val" / "failure_types.csv")[["episode", "type"]].rename(columns={"episode": "source_episode"})
    eps = eps[eps.split == "val"].merge(ft, on="source_episode", how="left")
    eps["kind"] = np.where(eps.success == 1, "success", eps.type)
    return eps


def choose(pred: pd.DataFrame, eps: pd.DataFrame) -> list[tuple[str, int]]:
    mae = (pred.ev_all_mlp - pred.value).abs().groupby(pred.episode_index).mean().rename("mae")
    e = eps.merge(mae, left_on="episode_index", right_index=True)
    out = []
    for kind, g in e.groupby("kind"):
        g = g.sort_values("mae")
        out.append((kind, int(g.episode_index.iloc[len(g) // 2])))
    out.append(("worst", int(e.loc[e.mae.idxmax(), "episode_index"])))
    return out


def figure(e: pd.DataFrame, raw: dict, title: str, path: Path, n_strip: int = 8):
    T = len(raw["actions"])
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, n_strip, height_ratios=[1, 1.8], hspace=0.25, wspace=0.03)
    for k, t in enumerate(np.linspace(0, T - 1, n_strip).astype(int)):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(raw["image"][t])
        ax.set_axis_off()
        ax.set_title(f"{t / FPS:.0f} s", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    s = e.frame_index / FPS
    ax.plot(s, e.value, color="black", lw=2, label="target v")
    ax.plot(s, e.ev_clock, color="grey", ls="--", lw=1, label="clock")
    if "ev_resnet" in e:
        ax.plot(s, e.ev_resnet, color="tab:purple", lw=1, alpha=0.8, label="ResNet-18")
    ax.plot(s, e.ev_all_mlp, color="tab:orange", lw=1.3, label="π₀.₅ probe, 201 bins")
    ax.plot(s, e.ev_oc, color="tab:blue", lw=1.3, label="π₀.₅ probe, P(success) × clock")
    ax.axhline(-0.5, color="grey", lw=0.6)
    g = raw["goals"]
    for i, name in enumerate(["pot 1", "pot 2"]):
        if g[:, i].any():
            t0 = int(np.argmax(g[:, i])) / FPS
            ax.axvline(t0, color="tab:green", ls=":", lw=1)
            ax.text(t0, -0.02, f" {name} on", color="tab:green", fontsize=8, va="top")
    ax.set(xlim=(0, 52), ylim=(-1.02, 0.02), xlabel="elapsed time (s)", ylabel="value")
    ax2 = ax.twinx()
    ax2.plot(s, e.ps_all_mlp, color="tab:green", lw=1, alpha=0.6, label="P(success)")
    ax2.set(ylim=(-0.02, 1.02), ylabel="P(success)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", ncol=3)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def overview(pred: pd.DataFrame, eps: pd.DataFrame, path: Path):
    d = pred.merge(eps[["episode_index", "kind"]], on="episode_index")
    d["sec"] = (d.frame_index // 20) * 2          # 2 s bins
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    order = ["success", "near_miss", "slow", "stalled_earlier", "stuck_pot2", "stuck_placing_pot2", "stuck_pot1",
             "knocked_over"]
    for kind in order:
        g = d[d.kind == kind]
        if g.empty:
            continue
        n = g.episode_index.nunique()
        m = g.groupby("sec")[["ev_oc", "ps_all_mlp"]].mean()
        kw = dict(lw=2.2, color="black") if kind == "success" else dict(lw=1.2)
        axes[0].plot(m.index, m.ev_oc, label=f"{kind} ({n})", **kw)
        axes[1].plot(m.index, m.ps_all_mlp, label=f"{kind} ({n})", **kw)
    axes[0].set(xlabel="elapsed time (s)", ylabel="mean E[V] (P(success) × clock)", ylim=(-1, 0))
    axes[1].set(xlabel="elapsed time (s)", ylabel="mean P(success)", ylim=(0, 1))
    axes[0].axhline(-0.5, color="grey", lw=0.6)
    axes[1].legend(fontsize=7)
    fig.suptitle("Val episodes (10 unseen init states) by Stage 2 failure type: π₀.₅ all-token MLP probe", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="vl.libero_probe output dir (val_predictions.parquet)")
    ap.add_argument("--resnet", default=None, help="vl.libero_resnet output dir")
    ap.add_argument("--out", default="reports/figures/libero_stage3a")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(Path(a.probe) / "val_predictions.parquet")
    # E[V] of the outcome+clock probe: P(success) times the per-outcome clock means at each frame
    from vl.libero_probe import CENTERS, clock_by_outcome, load
    _, meta = load(Path("/iris/u/ankile/karenvo/libero/features/feats_every4_18163142.npz"), DATASET)
    hs, hf = clock_by_outcome(meta[meta.split == "train"], int(meta.frame_index.max()) + 1)
    fr = pred.frame_index.values
    pred["ev_oc"] = pred.ps_all_mlp * (hs[fr] @ CENTERS) + (1 - pred.ps_all_mlp) * (hf[fr] @ CENTERS)
    if a.resnet:
        r = pd.read_parquet(Path(a.resnet) / "val_predictions.parquet")[["episode_index", "frame_index", "ev_resnet"]]
        pred = pred.merge(r, on=["episode_index", "frame_index"], how="left")
    eps = episode_table()
    for k, (kind, ep) in enumerate(choose(pred, eps)):
        row = eps[eps.episode_index == ep].iloc[0]
        with np.load(RUN / row.raw) as z:
            raw = {x: z[x] for x in ("image", "actions", "goals")}
        e = pred[pred.episode_index == ep].sort_values("frame_index")
        mae = (e.ev_all_mlp - e.value).abs().mean()
        title = (f"{kind}: val episode {ep} (init state {row.init_state}, {'success' if row.success else 'timeout'}, "
                 f"{row.policy_steps} steps), 201-bin probe MAE {mae:.3f}")
        figure(e, raw, title, out / f"{k}_{kind}_ep{ep}.png")
        print(out / f"{k}_{kind}_ep{ep}.png")
    overview(pred, eps, out / "overview.png")
    print(out / "overview.png")


if __name__ == "__main__":
    main()
