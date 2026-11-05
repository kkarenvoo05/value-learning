"""What advantages did the value functions assign? Per-episode viewer for the arm C labels (2026-11-04, Karen).

For a handful of train rollouts, one figure each:
  (1) agentview frames along the episode, border red where the ResNet label is negative
  (2) the ResNet's full 201-bin value distribution per frame (heatmap) with its E[V], the π₀.₅ value expert's E[V],
      and the true Monte-Carlo target (vl.libero_targets: -(steps to go)/1040, failures end at -0.5)
  (3) the advantage A_t of both value functions with each one's 30th-percentile threshold
  (4) the labels the policy was trained on (ResNet = what arm C used; expert; oracle = rollout outcome) and the
      number of pots on the stove from the simulator's goal predicates (vl.libero_subtask.subtask_labels)

Values, advantages and labels are read from the label tables exactly as arm C was trained on them
(labels/oof_{resnet,expert}_n20_p30, labels/oracle_outcome). Only the 201-bin probabilities are recomputed: each
episode's out-of-fold ResNet (resnet18_oof_f<fold>_s0_*, the fold that never fit it) is re-run on its frames, and
p @ CENTERS is asserted to reproduce the stored V. The expert's OOF run saved E[V] only, so it has no heatmap.

Episode picks [ours], from the 900 train rollouts, ResNet labels:
  success_most_negative   success with the largest share of negative frames
  success_typical         success at the median negative share (reference)
  failure_most_positive   failure with the largest share of positive frames
  pot_falls               an episode where a pot leaves the stove (pots-on-stove count drops); the one with the
                          largest count drop held longest beforehand: does A go negative where the state got worse?
  resnet_vs_expert        the episode where ResNet and expert labels disagree on the most frames

Usage (repo venv, one GPU or CPU):
    .venv/bin/python scripts/libero_advantage_examples.py --out reports/2026-11-04-advantage-examples
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.libero_probe import CENTERS  # noqa: E402
from vl.libero_resnet import CAMS, evaluate, table_meta  # noqa: E402
from vl.libero_subtask import subtask_labels  # noqa: E402
from vl.model import ValueModel  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
LAB = TABLE / "labels"
FPS = 20  # LIBERO env control_freq: steps per real second (docs/libero_plan.md 2026-11-03; videos are written at 10)
# reference palette (dataviz skill): ResNet = slot 1 blue, expert = slot 2 orange; negative label = diverging red pole
BLUE, ORANGE, INK, MUTED, RED, NEUTRAL = "#1c5cab", "#eb6834", "#0b0b0b", "#8a8984", "#c4332f", "#e4e3df"
HEAT = LinearSegmentedColormap.from_list("blue_ramp", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def load_labels():
    keep = ["episode_index", "frame_index", "positive", "A", "term", "V", "success", "init_state", "fold"]
    r = pd.read_parquet(LAB / "oof_resnet_n20_p30" / "labels.parquet")[keep]
    e = pd.read_parquet(LAB / "oof_expert_n20_p30" / "labels.parquet")[["episode_index", "frame_index", "positive", "A", "V"]]
    o = pd.read_parquet(LAB / "oracle_outcome" / "labels.parquet")[["episode_index", "frame_index", "positive"]]
    d = r.merge(e, on=["episode_index", "frame_index"], suffixes=("_r", "_e"), validate="one_to_one")
    ps = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "ps_expert"])
                    for p in sorted(glob.glob(str(RUNS / "value_expert_oof_f*_*" / "oof_predictions.parquet")))])
    d = d.merge(ps.rename(columns={"ps_expert": "ps_e"}), on=["episode_index", "frame_index"], how="left",
                validate="one_to_one")
    assert d.ps_e.notna().all()
    d = d.merge(o.rename(columns={"positive": "positive_o"}), on=["episode_index", "frame_index"], validate="one_to_one")
    thr = {s: json.loads((LAB / f"oof_{s}_n20_p30" / "summary.json").read_text())["threshold"] for s in ("resnet", "expert")}
    return d, thr


def pots_on_stove(raw: str) -> np.ndarray:
    with np.load(raw) as z:
        return subtask_labels(z["goals"])


def debounce(k: np.ndarray, hold: int = 10) -> np.ndarray:
    """Pots-on-stove with changes shorter than `hold` steps (0.5 s) removed: the goal predicate flickers for 1-3 steps
    while a pot settles [ours]. A run shorter than hold keeps the previous value."""
    out, cur, t = k.copy(), k[0], 0
    while t < len(k):
        e = t
        while e < len(k) and k[e] == k[t]:
            e += 1
        if e - t >= hold or t == 0:
            cur = k[t]
        out[t:e] = cur
        t = e
    return out


def p_success(probs: np.ndarray) -> np.ndarray:
    """Mass above -0.5: successes land in (-0.5, 0], failures in [-1, -0.5] (vl.libero_targets)."""
    return probs[:, CENTERS > -0.5].sum(1)


def pick(d: pd.DataFrame, eps: pd.DataFrame) -> dict:
    g = d.groupby("episode_index").agg(success=("success", "first"), neg=("positive_r", lambda x: 1 - x.mean()),
                                       disagree=("positive_r", "size"))
    g["disagree"] = d.assign(x=d.positive_r != d.positive_e).groupby("episode_index").x.mean()
    s, f = g[g.success == 1], g[g.success == 0]
    out = {"success_most_negative": int(s.neg.idxmax()),
           "success_typical": int((s.neg - s.neg.median()).abs().idxmin()),
           "failure_most_positive": int(f.neg.idxmin())}
    best, key = None, None  # pot falls: a sustained (>= 10 steps, 0.5 s) drop in pots-on-stove; the longest-lasting drop
    raw = eps.set_index("episode_index").raw
    for e in g.index:
        k = debounce(pots_on_stove(raw[e]))
        for t in np.flatnonzero(np.diff(k) < 0):
            back = np.flatnonzero(k[t + 1:] >= k[t])
            gone = int(back[0]) if len(back) else len(k) - t - 1
            if key is None or gone > key:
                best, key = e, gone
    if best is not None:
        out["pot_falls"] = int(best)
    out["resnet_vs_expert"] = int(g.disagree.idxmax())
    return out, g


def resnet_probs(e: int, fold: int, meta: pd.DataFrame, arr, device) -> tuple[np.ndarray, np.ndarray]:
    run = sorted(glob.glob(str(RUNS / f"resnet18_oof_f{fold}_s0_*")))
    run = [p for p in run if (Path(p) / "oof_predictions.parquet").exists()]
    assert len(run) == 1, run
    model = ValueModel(CAMS, n_bins=len(CENTERS)).to(device)
    model.load_state_dict(torch.load(Path(run[0]) / "best.pt", map_location=device))
    idx = np.flatnonzero(meta.episode_index.values == e)
    p, _ = evaluate(model, arr, idx, meta.bin.values.astype(np.int64), device)
    return p, meta.frame_index.values[idx]


def figure(e, why, d, thr, meta, arr, eps, probs, frames, value_true, path):
    de = d[d.episode_index == e].sort_values("frame_index")
    assert (de.frame_index.values == frames).all()
    ev = probs @ CENTERS
    err = np.abs(ev - de.V_r.values).max()
    assert err < 0.01, f"ep {e}: recomputed E[V] differs from the stored OOF V by {err:.4f}"
    row = eps.set_index("episode_index").loc[e]
    k = debounce(pots_on_stove(row.raw))
    if row.success:
        k = np.append(k, 2)  # success = both pots on the stove after the last action (the image never shows it)
    t = frames / FPS
    ps = p_success(probs)
    T = row.policy_steps / FPS

    fig = plt.figure(figsize=(13, 12), dpi=110)
    gs = fig.add_gridspec(6, 8, height_ratios=[1.25, 2.4, 1.0, 1.5, 0.75, 0.55], hspace=0.45, wspace=0.06)
    # (1) frames
    rows_idx = np.flatnonzero(meta.episode_index.values == e)
    show = np.linspace(0, len(frames) - 1, 8).round().astype(int)
    for j, i in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        img = arr[rows_idx[i], 0].transpose(1, 2, 0)  # stored upright (libero_rollouts.py flips at record time)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        neg = not de.positive_r.values[i]
        for sp in ax.spines.values():
            sp.set_edgecolor(RED if neg else NEUTRAL); sp.set_linewidth(3 if neg else 1)
        ax.set_title(f"{t[i]:.1f} s · A={de.A_r.values[i]:+.3f}", fontsize=8, color=RED if neg else INK)
    # (2) distribution
    ax = fig.add_subplot(gs[1, :])
    edges_t = np.concatenate([t, [t[-1] + 4 / FPS]])
    half = (CENTERS[1] - CENTERS[0]) / 2
    ax.pcolormesh(edges_t, np.concatenate([CENTERS - half, [CENTERS[-1] + half]]), probs.T, cmap=HEAT,
                  norm=PowerNorm(0.5, vmin=0, vmax=probs.max()), shading="flat", rasterized=True)
    ax.plot(t, value_true, color=INK, lw=1.6, ls=(0, (4, 2)), label="true MC target")
    ax.plot(t, ev, color=BLUE, lw=2, label="ResNet E[V] (heatmap = its 201-bin probabilities)")
    ax.plot(t, de.V_e.values, color=ORANGE, lw=2, label="π₀.₅ value expert E[V]")
    changes = np.flatnonzero(np.diff(k) != 0) + 1
    ax.axhline(-0.5, color=MUTED, lw=0.8, ls="-")
    ax.text(0.2, -0.49, "success above / failure below", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_ylim(-1, 0.02); ax.set_xlim(0, T)
    ax.set_ylabel("value  (0 = done now)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, fontsize=8, frameon=False)
    # (2b) P(success)
    axp = fig.add_subplot(gs[2, :], sharex=ax)
    axp.plot(t, ps, color=BLUE, lw=1.8, label="ResNet: mass above −0.5")
    axp.plot(t, de.ps_e.values, color=ORANGE, lw=1.8, label="expert: its P(success) head")
    axp.axhline(float(row.success), color=INK, lw=1, ls=(0, (4, 2)))
    axp.set_ylim(-0.03, 1.03); axp.set_ylabel("P(success)")
    axp.legend(loc="lower left", bbox_to_anchor=(0, 0.98), ncol=2, fontsize=8, frameon=False)
    ax.set_title(f"episode {e} · init state {row.init_state} ({row.layout}) · "
                 f"{'SUCCESS' if row.success else 'FAILURE'} at {T:.1f} s · picked as: {why}", fontsize=10, loc="left")
    # (3) advantages
    ax3 = fig.add_subplot(gs[3, :], sharex=ax)
    ax3.axhline(0, color=MUTED, lw=0.8)
    ax3.plot(t, de.A_r.values, color=BLUE, lw=1.6, label="ResNet A")
    ax3.plot(t, de.A_e.values, color=ORANGE, lw=1.6, label="expert A")
    ax3.axhline(thr["resnet"], color=BLUE, lw=1, ls="--")
    ax3.axhline(thr["expert"], color=ORANGE, lw=1, ls="--")
    ax3.text(0.2, min(thr.values()), "dashed: label threshold (30th pct); below = negative", fontsize=7.5, color=MUTED, va="top")
    term = de.term.values.astype(bool)
    if term.any():
        ax3.axvspan(t[term][0], T, color=NEUTRAL, alpha=0.6, lw=0)
        ax3.text(t[term][0], ax3.get_ylim()[1], "terminal: outcome enters A ", fontsize=7.5, color=MUTED, va="top", ha="right")
    ax3.set_ylabel("A = V(t+20 steps) − V(t) − 20/1040\n(20 steps = 1 s)")
    ax3.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    # (4) labels
    ax4 = fig.add_subplot(gs[4, :], sharex=ax)
    for r, (name, col) in enumerate([("ResNet (arm C)", "positive_r"), ("expert", "positive_e"), ("oracle", "positive_o")]):
        neg = ~de[col].values.astype(bool)
        ax4.broken_barh([(a, 0.4) for a in t], (2 - r - 0.4, 0.8), facecolors=np.where(neg, RED, NEUTRAL))
        ax4.text(-0.3, 2 - r, f"{name}  {neg.mean():.0%} neg", ha="right", va="center", fontsize=8)
    ax4.set_yticks([]); ax4.set_ylim(-0.6, 2.6)
    for sp in ("left", "right", "top"):
        ax4.spines[sp].set_visible(False)
    # (5) pots on stove
    ax5 = fig.add_subplot(gs[5, :], sharex=ax)
    ax5.step(np.arange(len(k)) / FPS, k, where="post", color=INK, lw=1.4)
    ax5.set_ylim(-0.3, 2.3); ax5.set_yticks([0, 1, 2]); ax5.set_ylabel("pots on\nstove", fontsize=8)
    ax5.set_xlabel("time (s)")
    for a_ in (ax, axp, ax3):
        for tt in changes:
            a_.axvline(tt / FPS, color=MUTED, lw=1, ls=":")
    for tt in changes:
        ax5.text(tt / FPS, 2.25, f" {k[tt - 1]}→{k[tt]}", fontsize=7.5, color=MUTED, va="top")
    for a_ in (ax, axp, ax3, ax4):
        plt.setp(a_.get_xticklabels(), visible=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"episode": int(e), "picked_as": why, "success": int(row.success), "T_s": float(T),
            "init_state": int(row.init_state), "layout": row.layout,
            "neg_share": {"resnet": float(1 - de.positive_r.mean()), "expert": float(1 - de.positive_e.mean()),
                          "oracle": float(1 - de.positive_o.mean())},
            "label_disagreement_resnet_expert": float((de.positive_r != de.positive_e).mean()),
            "max_abs_ev_recompute_error": float(err),
            "corr_A_vs_dPsucc_resnet": dp_corr(de, ps),
            "pots_changes_debounced": [(float(i / FPS), int(k[i - 1]), int(k[i])) for i in changes]}


def dp_corr(de, ps, n=5):
    """Correlation of the interior ResNet A with P(success)(t + n grid frames) - P(success)(t): how much of A is
    the success odds moving (n = 5 grid frames = the 20-step lookahead, 1 s)."""
    interior = ~de.term.values.astype(bool)[:-n]
    a = de.A_r.values[:-n][interior]
    dp = (ps[n:] - ps[:-n])[interior]
    return float(np.corrcoef(a, dp)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", help="extra episodes to draw")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, thr = load_labels()
    eps = pd.read_csv(TABLE / "episodes.csv")
    meta = table_meta(TABLE, 4)
    arr = np.load(TABLE / "frames_every4.npy", mmap_mode="r")
    assert len(arr) == len(meta)
    picks, g = pick(d, eps)
    for e in a.episodes or []:
        picks[f"extra_{e}"] = e
    print("picks", picks, flush=True)
    summary = {"thresholds": thr, "episodes": []}
    for why, e in picks.items():
        fold = int(d[d.episode_index == e].fold.iloc[0])
        probs, frames = resnet_probs(e, fold, meta, arr, device)
        vt = meta[meta.episode_index == e].value.values
        info = figure(e, why.replace("_", " "), d, thr, meta, arr, eps, probs, frames, vt, out / f"ep{e:04d}_{why}.png")
        info["file"] = f"ep{e:04d}_{why}.png"; info["fold"] = fold
        summary["episodes"].append(info)
        print(json.dumps(info), flush=True)
    s, f = g[g.success == 1], g[g.success == 0]
    summary["all_train"] = {"episodes": int(len(g)), "neg_share_success_median": float(s.neg.median()),
                            "neg_share_failure_median": float(f.neg.median()),
                            "neg_share_success_p10_p90": [float(s.neg.quantile(.1)), float(s.neg.quantile(.9))],
                            "neg_share_failure_p10_p90": [float(f.neg.quantile(.1)), float(f.neg.quantile(.9))],
                            "label_disagreement_resnet_expert_mean": float(g.disagree.mean())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["all_train"]), flush=True)


if __name__ == "__main__":
    main()
