"""Stage 4 value plots: 3b arm B (π₀.₅ expert, w256 + aug) vs the ResNet-18 control, 3 seeds each [ours].

Usage: .venv/bin/python scripts/libero_b_vs_resnet_plots.py
    -> reports/figures/libero_stage4_b_vs_resnet/<k>_<kind>_ep<idx>.png, overview.png, aligned.png

Episodes, fixed rule (set before looking): per kind (success + each Stage 2 failure type), the episode whose MAE
averaged over both models and all seeds is the median of its kind, so neither model picks; plus the episode where
B beats the ResNet most and the one where the ResNet beats B most (seed-mean MAE difference).
Curves: target v (black), each seed of B (orange) and the ResNet (purple) as thin lines, the seed mean bold.
Dotted green: first step each pot is On(stove). aligned.png: V around the milestone (vl.libero_milestone), minus V
at the milestone, mean over val episodes with one, successes and failures separately.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/move/u/karenvo/Projects/value-learning/src")
from vl.libero_milestone import episodes  # noqa: E402
from vl.libero_value_plots import FPS, RUN, episode_table  # noqa: E402

R = Path("/iris/u/ankile/karenvo/libero/runs")
MODELS = {"B": ("tab:orange", "ev_expert", ["value_expert_aug_w256_s0_18186728", "value_expert_aug_w256_s1_18189370",
                                             "value_expert_aug_w256_s2_18189371"]),
          "ResNet": ("tab:purple", "ev_resnet", ["resnet18_18166232", "resnet18_s1_18174529", "resnet18_s2_18174726"])}
OUT = Path("/move/u/karenvo/Projects/value-learning/reports/figures/libero_stage4_b_vs_resnet")
LABEL = {"B": "π₀.₅ expert B", "ResNet": "ResNet-18"}


def load() -> pd.DataFrame:
    key = ["episode_index", "frame_index"]
    pred = None
    for m, (_, col, dirs) in MODELS.items():
        for s, d in enumerate(dirs):
            p = pd.read_parquet(R / d / "val_predictions.parquet")
            p = p[key + ["value", col]].rename(columns={col: f"{m}_s{s}"})
            pred = p if pred is None else pred.merge(p.drop(columns="value"), on=key)
    for m in MODELS:
        pred[m] = pred[[f"{m}_s{s}" for s in range(3)]].mean(1)
    return pred


def episode_mae(pred: pd.DataFrame) -> pd.DataFrame:
    g = pred.groupby("episode_index")
    return pd.DataFrame({m: g.apply(lambda e, m=m: np.mean([(e[f"{m}_s{s}"] - e.value).abs().mean()
                                                            for s in range(3)])) for m in MODELS})


def choose(mae: pd.DataFrame, eps: pd.DataFrame) -> list[tuple[str, int]]:
    e = eps.merge(mae, left_on="episode_index", right_index=True)
    e["both"], e["diff"] = (e.B + e.ResNet) / 2, e.B - e.ResNet
    out = []
    for kind, g in e.groupby("kind"):
        g = g.sort_values("both")
        out.append((kind, int(g.episode_index.iloc[len(g) // 2])))
    out.append(("B_best", int(e.loc[e["diff"].idxmin(), "episode_index"])))
    out.append(("ResNet_best", int(e.loc[e["diff"].idxmax(), "episode_index"])))
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
    for m, (c, _, _) in MODELS.items():
        for k in range(3):
            ax.plot(s, e[f"{m}_s{k}"], color=c, lw=0.7, alpha=0.45)
        ax.plot(s, e[m], color=c, lw=1.8, label=f"{LABEL[m]} (seed mean; thin = seeds)")
    ax.axhline(-0.5, color="grey", lw=0.6)
    g = raw["goals"]
    for i, name in enumerate(["pot 1", "pot 2"]):
        if g[:, i].any():
            t0 = int(np.argmax(g[:, i])) / FPS
            ax.axvline(t0, color="tab:green", ls=":", lw=1)
            ax.text(t0, -0.02, f" {name} on", color="tab:green", fontsize=8, va="top")
    ax.set(xlim=(0, 52), ylim=(-1.02, 0.02), xlabel="elapsed time (s)", ylabel="value")
    ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def overview(pred: pd.DataFrame, eps: pd.DataFrame, path: Path):
    d = pred.merge(eps[["episode_index", "kind"]], on="episode_index")
    d["sec"] = (d.frame_index // 20) * 2  # 2 s bins
    order = ["success", "near_miss", "slow", "stalled_earlier", "stuck_pot2", "stuck_placing_pot2", "stuck_pot1",
             "knocked_over"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, col, name in zip(axes, ["value", "B", "ResNet"], ["target v", LABEL["B"], LABEL["ResNet"]]):
        for kind in order:
            g = d[d.kind == kind]
            m = g.groupby("sec")[col].mean()
            kw = dict(lw=2.2, color="black") if kind == "success" else dict(lw=1.2)
            ax.plot(m.index, m, label=f"{kind} ({g.episode_index.nunique()})", **kw)
        ax.set(title=f"mean {name}", xlabel="elapsed time (s)", ylim=(-1, 0))
        ax.axhline(-0.5, color="grey", lw=0.6)
    axes[0].set_ylabel("value")
    axes[2].legend(fontsize=7)
    fig.suptitle("Val episodes (10 unseen init states) by Stage 2 failure type, seed-mean V", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def aligned(pred: pd.DataFrame, path: Path, w: int = 100):
    ms = episodes("val").set_index("episode_index").milestone
    rows = []
    for ep, e in pred.groupby("episode_index"):
        if not np.isfinite(ms[ep]):
            continue
        e = e.assign(rel=e.frame_index - ms[ep])
        e = e[e.rel.abs() <= w]
        at = e.iloc[(e.rel).abs().argsort()[:1]]
        for col in ["value", "B", "ResNet"]:
            e[col + "_d"] = e[col] - float(at[col].iloc[0])
        rows.append(e.assign(rel4=(e.rel // 4) * 4))
    d = pd.concat(rows).merge(episodes("val")[["episode_index", "success"]], on="episode_index")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (succ, name) in zip(axes, [(1, "successes"), (0, "failures")]):
        g = d[d.success == succ]
        for col, c, lab in [("value", "black", "target v"), ("B", MODELS["B"][0], LABEL["B"]),
                            ("ResNet", MODELS["ResNet"][0], LABEL["ResNet"])]:
            m = g.groupby("rel4")[col + "_d"].mean()
            ax.plot(m.index / FPS, m, color=c, lw=2 if col == "value" else 1.6, label=lab)
        ax.axvline(0, color="tab:green", ls=":", lw=1)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set(title=f"{name} with a milestone ({g.episode_index.nunique()} episodes)",
               xlabel="time from first pot on the stove (s)")
    axes[0].set_ylabel("V − V(milestone), mean over episodes")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does V react when the first pot lands? (seed-mean V; a clock alone would follow the black line)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = load()
    eps = episode_table()
    mae = episode_mae(pred)
    for k, (kind, ep) in enumerate(choose(mae, eps)):
        row = eps[eps.episode_index == ep].iloc[0]
        with np.load(RUN / row.raw) as z:
            raw = {x: z[x] for x in ("image", "actions", "goals")}
        e = pred[pred.episode_index == ep].sort_values("frame_index")
        title = (f"{kind}: val episode {ep} (init state {row.init_state}, {'success' if row.success else 'timeout'}, "
                 f"{row.policy_steps} steps), MAE B {mae.B[ep]:.3f} / ResNet {mae.ResNet[ep]:.3f}")
        figure(e, raw, title, OUT / f"{k}_{kind}_ep{ep}.png")
        print(OUT / f"{k}_{kind}_ep{ep}.png")
    overview(pred, eps, OUT / "overview.png")
    aligned(pred, OUT / "aligned.png")
    print(OUT / "overview.png", OUT / "aligned.png")


if __name__ == "__main__":
    main()
