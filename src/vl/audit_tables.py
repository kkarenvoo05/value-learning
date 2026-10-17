"""Regenerate the numeric tables of the audit report from the manifest.

Usage:
    .venv/bin/python -m vl.audit_tables --config configs/dataset.yaml   # -> reports/audit_tables.md
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from vl.config import load_config


def md(df: pd.DataFrame) -> str:
    return df.to_markdown() if hasattr(df, "to_markdown") else "```\n" + df.to_string() + "\n```"


def short(repo: str) -> str:
    return repo.replace("real-marker-d2-", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = pd.read_parquet(cfg["paths"]["manifest"])
    pins = json.loads((Path(cfg["paths"]["meta_root"]) / "pins.json").read_text())
    s = m[m.split != "audit_only"].copy()
    s["repo"] = s.repo.map(short)
    s["valid_s"] = s.valid_length / s.fps
    out = []

    pin_tab = pd.DataFrame([{"repo": short(k), "commit": v["sha"], "tag": v["tag"],
                             "GB": round(v["bytes_total"] / 1e9, 2)} for k, v in pins.items()])
    out += ["## Pinned revisions", md(pin_tab.set_index("repo")),
            f"Total if fully downloaded: {pin_tab.GB.sum():.1f} GB "
            f"(side_1+wrist_left videos + data only: see report)."]

    out += ["## Episodes by repository, split and outcome",
            md(pd.crosstab(s.repo, [s.split, s.outcome], margins=True))]
    out += ["## Episodes, groups and valid frames by split",
            md(s.groupby("split").agg(episodes=("episode_uid", "size"), groups=("group_id", "nunique"),
                                      valid_frames=("valid_length", "sum"),
                                      valid_hours=("valid_s", lambda x: round(x.sum() / 3600, 2))))]
    tp = s[s.split.isin(["train", "val"])]
    out += ["## Validation share per repository (episodes)",
            md(tp.groupby("repo").split.apply(lambda x: round((x == "val").mean(), 3)).to_frame("val_fraction"))]
    out += ["## Valid duration (s) by data source and outcome (training pool)",
            md(tp.groupby(["source_type", "outcome"]).valid_s.describe().round(1))]
    out += ["## Frames recorded after the reviewed endpoint (training pool)",
            md(tp.groupby("source_type").frames_after_endpoint.describe().round(1))]
    out += ["## Human control (training pool)",
            md(tp.groupby("source_type").agg(
                episodes=("episode_uid", "size"), with_human_frames=("has_human_frames", "sum"),
                counterfactual=("is_counterfactual", lambda x: int(x.fillna(False).sum())),
                human_frames=("n_human_frames", "sum"), intervention_onsets=("n_intervention_onsets", "sum")))]
    out += ["## Label sources", md(s.groupby(["repo", "label_source"]).size().unstack(fill_value=0))]
    out += ["## Soft truncation by outcome", md(pd.crosstab(s.repo, [s.outcome, s.soft_truncate]))]
    t = s[s.outcome == "timeout"]
    out += ["## Timeouts: soft-truncated / reaching the 800-step cap",
            md(pd.crosstab(t.repo, [t.soft_truncate.rename("soft_truncate"),
                                    (t.valid_length >= 800).rename("valid_len>=800")]))]
    out += ["## Machine (VLM) max stage by reviewed outcome — diagnostic only, NOT ground truth",
            md(pd.crosstab(tp.outcome, tp.machine_stage.fillna(-1).astype(int)))]

    # Group structure: which repos are linked through shared initial states.
    multi = s.groupby("group_id").repo.agg(lambda r: tuple(sorted(set(r))))
    pairs = {}
    for repos in multi:
        for a, b in combinations(repos, 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    out += ["## Groups spanning several repositories (count of groups per repo pair)",
            md(pd.Series(pairs, name="groups").sort_values(ascending=False).to_frame())]
    out += ["## Group size distribution (split repos)",
            md(s.groupby("group_id").size().value_counts().sort_index().to_frame("groups"))]

    a = m[m.split == "audit_only"]
    rows = []
    for _, r in a.iterrows():
        for col in ["content_fingerprint", "init_state_key"]:
            hit = s[s[col] == r[col]]
            if len(hit):
                rows.append({"audit_repo": short(r.repo), "link": col,
                             "split_repos": ",".join(sorted(set(hit.repo))),
                             "splits": ",".join(sorted(set(hit.split)))})
    if rows:
        out += ["## Audit-only repos linked to split repos",
                md(pd.DataFrame(rows).value_counts().to_frame("episodes"))]
    out += ["## Train-pool episodes sharing a group with test", str(int(s.test_overlap_conflict.sum()))]
    out += ["## Excluded episodes", md(s[s.exclude_reason != ""][["repo", "episode_index", "outcome",
                                                                  "label_source", "exclude_reason"]])]
    over = s[s.valid_length > 801]
    out += ["## Valid length above the documented 800-step cap",
            md(over[["repo", "episode_index", "outcome", "T", "raw_length", "soft_truncate"]])]
    cam = s[[c for c in s.columns if c.startswith("cam_") and not c.endswith("err_frames")]]
    out += ["## Camera availability (episodes)", md(cam.groupby(s.repo).sum())]
    bad = s[(s.filter(like="_span_err_frames").abs() >= 0.5).any(axis=1)]
    out += ["## Video span != frame count", md(bad[["repo", "episode_index", "T", "raw_length"] +
                                                   list(s.filter(like="_span_err_frames").columns)])]

    path = Path(cfg["paths"]["report_dir"]) / "audit_tables.md"
    path.write_text("# Audit tables (generated by `python -m vl.audit_tables`)\n\n" + "\n\n".join(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
