"""Per-frame value targets from the episode manifest (no videos needed).

Usage:
    .venv/bin/python -m vl.targets --config configs/targets.yaml

For an episode with valid final frame T (0-indexed) and frame rate fps, each frame t = 0..T gets
    R_t = -(T - t) / fps - c_fail_s * [failed]      Monte-Carlo return in seconds, undiscounted
    v_t = R_t / (horizon_s + c_fail_s)              normalized value, ~[-1, 0]
    bin = nearest of n_bins evenly spaced centers on [v_min, v_max]
Frames after T are never targets. Values outside the support are flagged (bin = -1) and
reported, never clipped. Choices and their provenance are in configs/targets.yaml.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vl.config import load_config


def bin_centers(cfg: dict) -> np.ndarray:
    return np.linspace(cfg["v_min"], cfg["v_max"], cfg["n_bins"])


def value_to_bin(v: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Nearest bin index and in-support mask; out-of-support values get bin -1."""
    width = (cfg["v_max"] - cfg["v_min"]) / (cfg["n_bins"] - 1)
    idx = np.rint((v - cfg["v_min"]) / width).astype(int)
    ok = (idx >= 0) & (idx <= cfg["n_bins"] - 1)
    return np.where(ok, idx, -1), ok


def episode_targets(T: int, fps: float, outcome: str, cfg: dict) -> pd.DataFrame:
    if outcome in cfg["failure_outcomes"]:
        failed = True
    elif outcome in cfg["success_outcomes"]:
        failed = False
    else:
        raise ValueError(f"no target for outcome {outcome!r}")
    t = np.arange(int(T) + 1)
    time_to_go = (T - t) / fps
    ret = -time_to_go - cfg["c_fail_s"] * failed + 0.0   # + 0.0 turns -0.0 into 0.0
    value = ret / (cfg["horizon_s"] + cfg["c_fail_s"])
    b, ok = value_to_bin(value, cfg)
    return pd.DataFrame({"frame_index": t, "time_to_go_s": time_to_go, "return_s": ret,
                         "value": value, "bin": b, "in_support": ok, "failed": failed})


def build_targets(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Targets for every usable episode (in a split, not excluded), one row per valid frame."""
    use = manifest[(manifest.split != "audit_only") & (manifest.exclude_reason == "")]
    parts = []
    for ep in use.itertuples():
        e = episode_targets(int(ep.T), ep.fps, ep.outcome, cfg)
        parts.append(e.assign(episode_uid=ep.episode_uid, repo=ep.repo,
                              episode_index=ep.episode_index, split=ep.split,
                              outcome=ep.outcome))
    cols = ["episode_uid", "repo", "episode_index", "split", "outcome", "failed", "frame_index",
            "time_to_go_s", "return_s", "value", "bin", "in_support"]
    return pd.concat(parts, ignore_index=True)[cols]


def summarize(ft: pd.DataFrame) -> pd.DataFrame:
    """Frame counts, value ranges, and out-of-support counts per split and outcome."""
    return (ft.groupby(["split", "outcome"])
              .agg(episodes=("episode_uid", "nunique"), frames=("value", "size"),
                   value_min=("value", "min"), value_max=("value", "max"),
                   out_of_support=("in_support", lambda s: int((~s).sum())))
              .reset_index())


def plot_examples(ft: pd.DataFrame, path: Path, per_outcome: int = 4, seed: int = 0) -> None:
    """Target value vs elapsed time for a few random training episodes of each outcome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    tr = ft[ft.split == "train"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for outcome, color in {"success": "tab:blue", "failure": "tab:red", "timeout": "tab:orange"}.items():
        uids = tr[tr.outcome == outcome].episode_uid.unique()
        for k, uid in enumerate(rng.choice(uids, size=min(per_outcome, len(uids)), replace=False)):
            e = tr[tr.episode_uid == uid]
            elapsed = e.time_to_go_s.iloc[0] - e.time_to_go_s
            ax.plot(elapsed, e.value, color=color, lw=1.2, label=outcome if k == 0 else None)
    ax.axhline(-0.5, color="gray", ls="--", lw=0.8)
    ax.set(xlabel="elapsed time in episode (s)", ylabel="target value v = R / 120 s",
           ylim=(-1.02, 0.02), title="Monte Carlo value targets, random training episodes")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/targets.yaml")
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest_path = Path(load_config(args.dataset_config)["paths"]["manifest"])
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found: build it first (README, Setup and commands)")
    ft = build_targets(pd.read_parquet(manifest_path), cfg)
    out = Path(cfg["paths"]["frame_targets"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ft.to_parquet(out)
    s = summarize(ft)
    Path(cfg["paths"]["report"]).write_text(
        "# Target summary\n\nGenerated by `python -m vl.targets` from "
        f"`{manifest_path}` with `{args.config}`.\n\n"
        + s.to_markdown(index=False, floatfmt=".4f") + "\n")
    plot_examples(ft, Path(cfg["paths"]["figure"]))
    print(f"wrote {out}: {len(ft)} frames from {ft.episode_uid.nunique()} episodes")
    print(s.to_string(index=False))
    print("out-of-support frames:", int((~ft.in_support).sum()))


if __name__ == "__main__":
    main()
