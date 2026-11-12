"""Speed tags for the policy pre-training stage on the LIBERO demos [2026-11-06] (project venv, CPU, seconds).

Writes <demo table>/labels/speed_p<share>/labels.parquet (episode_index = DEMO_OFFSET + demo index, frame_index 0,
positive, task, policy_steps; every demo of all 40 tasks, fit and inner) + summary.md. Definition:
vl.libero_finetune_data.speed_labels (fastest `share` of each task's demos positive [RECAP App. A-F: ~30% of the
pre-training data positive, threshold per task]; episode-level, task baseline [ours]).

Usage: .venv/bin/python scripts/libero_speed_labels.py [--share 0.3] [--action-stats <parquet>]
--action-stats: optional per-episode action statistics (episode_index, trans, rot, still, grip_sw) to report how
positive and negative demos differ in their actions (is the tag predictable from the action at all?).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl import libero_finetune_data as fd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(fd.DEMO_TABLE))
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--action-stats", default=None)
    a = ap.parse_args()
    table = Path(a.table)
    eps = pd.read_csv(table / "episodes.csv")
    lab = fd.speed_labels(eps, a.share)
    out = table / "labels" / f"speed_p{round(a.share * 100)}"
    out.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(out / "labels.parquet")
    m = lab.assign(demo=lab.episode_index - fd.DEMO_OFFSET).merge(
        eps[["episode_index", "suite", "role"]].rename(columns={"episode_index": "demo"}), on="demo")
    frames = m.policy_steps.sum()
    per_role = m.groupby("role").agg(episodes=("demo", "size"), positive_eps=("positive", "mean"),
                                     positive_frames=("policy_steps", lambda s: (s * m.loc[s.index, "positive"]).sum()
                                                      / s.sum()))
    by_tag = m.groupby(["suite", "positive"]).policy_steps.mean().unstack().rename(columns={0: "neg", 1: "pos"})
    lines = [f"# Speed labels, positive share {a.share} per task (episodes)\n",
             f"`scripts/libero_speed_labels.py --share {a.share}`; definition `vl.libero_finetune_data.speed_labels`.\n",
             f"{len(m)} demos, {frames} frames; frame share positive "
             f"{(m.policy_steps * m.positive).sum() / frames:.3f}.\n",
             per_role.to_markdown(floatfmt=".3f"), "\nMean demo length (steps) by suite and tag:\n",
             by_tag.to_markdown(floatfmt=".1f")]
    if a.action_stats:
        st = pd.read_parquet(a.action_stats).rename(columns={"episode_index": "demo"})
        st = st.merge(m[["demo", "positive"]], on="demo", validate="one_to_one")
        lines += ["\nPer-episode action statistics (mean over demos; trans/rot = mean |action| of dims 0-2 / 3-5, "
                  "still = share of steps with every |a| < 0.05, grip_sw = gripper sign changes):\n",
                  st.groupby("positive")[["trans", "rot", "still", "grip_sw"]].mean().to_markdown(floatfmt=".4f")]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out / 'labels.parquet'}")


if __name__ == "__main__":
    main()
