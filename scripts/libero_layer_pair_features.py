"""Describe selected pairs and cached pooled features; correlations are not causal evidence.

Run after libero_layer_diagnosis.py. Figures are local scientific diagnostics, not a public upload.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/move/u/karenvo/libero_features/layer_diagnosis_20261105"))
    ap.add_argument("--features", type=Path, default=Path("/move/u/karenvo/libero_features/p1c_hidden_18288157"))
    a = ap.parse_args()
    pairs = pd.read_csv(a.report / "pairs.csv")
    candidates = pd.read_parquet(a.cache / "all_candidates.parquet")
    cols = [c for c in candidates if c.startswith("dv_")]
    candidates[cols].corr().to_csv(a.report / "change_correlations.csv")
    condition = candidates.no_goal_transition & (candidates.pot_motion_m <= .005)
    grouped = []
    for label, mask in (("all_interior", np.ones(len(candidates), bool)), ("no_event_low_pot_motion", condition),
                        ("placement", candidates.placement)):
        q = candidates[mask]
        for col in cols:
            grouped.append({"subset": label, "model": col[3:], "pairs": len(q),
                            "episodes": q.episode_index.nunique(), "mean_abs_dv": q[col].abs().mean(),
                            "mean_dv": q[col].mean(), "rms_dv": np.sqrt(np.mean(q[col] ** 2))})
    pd.DataFrame(grouped).to_csv(a.report / "all_pair_scores.csv", index=False)

    meta = pd.read_parquet(a.features / "meta.parquet").reset_index(names="feature_row")
    lookup = meta.set_index(["episode_index", "frame_index"]).feature_row
    indices = np.array([[lookup.loc[(r.episode_index, t)] for t in (r.t0, r.t1)] for r in pairs.itertuples()])
    rows = []
    files = sorted(a.features.glob("llm_h*.npy"), key=lambda p: int(p.stem[5:]) if p.stem[5:].isdigit() else 99)
    for path in files:
        feature = np.load(path, mmap_mode="r")
        x = feature[indices].astype(np.float32)  # pair, endpoint, camera, hidden
        denom = (np.linalg.norm(x[:, 0], axis=-1) + np.linalg.norm(x[:, 1], axis=-1)) / 2
        relative = np.linalg.norm(x[:, 1] - x[:, 0], axis=-1) / np.maximum(denom, 1e-12)
        cosine = 1 - (x[:, 0] * x[:, 1]).sum(-1) / np.maximum(
            np.linalg.norm(x[:, 0], axis=-1) * np.linalg.norm(x[:, 1], axis=-1), 1e-12)
        for pi, pair in enumerate(pairs.itertuples()):
            for camera, name in enumerate(("agentview", "wrist")):
                rows.append({"pair_id": pair.pair_id, "cohort": pair.cohort, "layer": path.stem[4:],
                             "camera": name, "relative_l2": float(relative[pi, camera]),
                             "cosine_distance": float(cosine[pi, camera])})
        print(f"features {path.stem}: {len(pairs)} pairs", flush=True)
    features = pd.DataFrame(rows)
    features.to_csv(a.report / "pair_feature_changes.csv", index=False)

    scores = pd.read_csv(a.report / "audit_scores.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("mae", "noise_1s", "snr_1s"),
                                 ("MAE (lower is better)", "Residual change SD, 1 s", "Change SNR, 1 s")):
        for x, name in enumerate(("n4", "n9", "n18")):
            q = scores[scores.name.str.startswith(name + "_")]
            ax.scatter(np.full(len(q), x), q[metric], s=40, alpha=.8)
            ax.plot([x - .15, x + .15], [q[metric].mean()] * 2, color="black", lw=2)
        ax.set_xticks([0, 1, 2], ["4 layers\n3 seeds", "9 layers\n1 seed", "18 layers\n3 seeds"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(a.report / "depth_audit.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    order = [p.stem[4:] for p in files]
    for ax, camera in zip(axes, ("agentview", "wrist")):
        table = features[features.camera == camera].groupby(["layer", "cohort"]).relative_l2.mean().unstack().reindex(order)
        table.plot(ax=ax, marker="o", ms=3)
        ax.set_xticks(range(len(order)), order, rotation=45)
        ax.set_title(f"{camera}: pooled-feature change")
        ax.set_ylabel("Relative L2 (descriptive, not causal)")
        ax.grid(alpha=.2)
    fig.savefig(a.report / "pooled_feature_changes.png", dpi=160)
    plt.close(fig)

    packed = np.load(a.cache / "pairs.npz")
    lookup = {p: i for i, p in enumerate(packed["pair_id"].tolist())}
    # A compact review sheet: four largest selected jumps, both camera views and endpoints.
    shown = pairs[pairs.cohort == "jump"].head(4)
    fig, axes = plt.subplots(len(shown), 4, figsize=(11, 3 * len(shown)), squeeze=False)
    captions = []
    for row, pair in enumerate(shown.itertuples()):
        ix = lookup[pair.pair_id]
        for col, (camera, side) in enumerate((("image", 0), ("image", 1), ("wrist_image", 0), ("wrist_image", 1))):
            axes[row, col].imshow(packed[camera][ix, side])
            axes[row, col].axis("off")
            t = pair.t0 if side == 0 else pair.t1
            axes[row, col].set_title(f"{'Main' if camera == 'image' else 'Wrist'} · step {t}", fontsize=10)
        captions.append(f"Episode {pair.episode_index}; ΔV = {pair.dv_n18_s0:+.3f}; "
                        f"pot motion {1000 * pair.pot_motion_m:.2f} mm; arm {1000 * pair.eef_motion_m:.1f} mm")
    fig.suptitle("Candidate flicker pairs: 0.2 s apart; no nearby goal-predicate change", fontsize=13)
    fig.subplots_adjust(left=.015, right=.985, bottom=.04, top=.94, wspace=.04, hspace=.32)
    fig.canvas.draw()
    for row, caption in enumerate(captions):
        fig.text(.02, axes[row, 0].get_position().y0 - .022, caption, fontsize=10)
    fig.savefig(a.cache / "jump_contact_sheet.png", dpi=130)
    plt.close(fig)
    summary = {"candidate_pairs": len(candidates), "eligible_no_event_pairs": int(condition.sum()),
               "cohorts": pairs.groupby("cohort").size().to_dict(),
               "contact_sheet": str(a.cache / "jump_contact_sheet.png"),
               "note": "Pairs were selected using seed 0; feature distances and images are diagnostic observations, not causal proof."}
    (a.report / "pair_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
