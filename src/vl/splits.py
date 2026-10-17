"""Group-aware, reproducible train/val/test assignment.

Usage:
    .venv/bin/python -m vl.splits --config configs/dataset.yaml

Episodes are linked (union-find) when they share any of:
  * the same physical initial state (pen_x, pen_y, pen_yaw, holder_x, holder_y, quantized),
  * the same content fingerprint (duplicated recording, whatever its metadata says),
  * the same (source_repo, source_episode_index) identity (derived copies).
Connected components are "episode groups". Validation takes ~fraction of training-pool
groups, stratified per (primary repo, group contains a non-success). Test repos keep
Lars's assignment; a group touching both train pool and test is reported, never moved.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from vl.config import load_config

STATE_COLS = ["init_pen_x", "init_pen_y", "init_pen_yaw", "init_holder_x", "init_holder_y"]
STATE_QUANT = 1e-4  # metres / radians; Sobol starts are continuous so exact replays match exactly


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def state_key(row) -> str:
    return ",".join(f"{round(row[c] / STATE_QUANT):d}" for c in STATE_COLS)


def build_groups(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["init_state_key"] = m.apply(state_key, axis=1)
    # Multi-session eval repos reuse one `source_repo` string with session-local episode indices,
    # so identity is keyed on the release block (the per-session recording), else the repo itself.
    block = m.release_block_repo.where(m.release_block_repo.notna(), m.repo_id)
    m["source_identity"] = block + ":" + m.source_episode_index.astype(int).astype(str)
    uf = UnionFind(m.episode_uid)
    link_counts = {}
    for col in ["init_state_key", "content_fingerprint", "source_identity"]:
        n = 0
        for _, uids in m.groupby(col).episode_uid:
            uids = list(uids)
            for u in uids[1:]:
                uf.union(uids[0], u)
                n += 1
        link_counts[col] = n
    m["group_id"] = m.episode_uid.map(uf.find)
    m.attrs["link_counts"] = link_counts
    return m


def assign_splits(m: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = m.copy()
    test_map = cfg["test"]
    pool = set(cfg["train_pool"])
    m["split"] = None
    m.loc[m.repo.isin(test_map), "split"] = m.repo.map(test_map)
    in_pool = m.repo.isin(pool)

    g = m.groupby("group_id")
    touches_test = g.repo.transform(lambda r: r.isin(test_map).any())
    touches_pool = g.repo.transform(lambda r: r.isin(pool).any())
    m["test_overlap_conflict"] = touches_test & touches_pool

    pool_groups = m[in_pool & ~m.test_overlap_conflict]
    info = pool_groups.groupby("group_id").agg(
        primary_repo=("repo", lambda r: r.value_counts().sort_index().idxmax()),
        any_nonsuccess=("outcome", lambda o: bool((o != "success").any())),
    ).sort_index()
    rng = np.random.default_rng(cfg["validation"]["seed"])
    frac = cfg["validation"]["fraction"]
    val_groups = set()
    for _, sub in info.groupby(["primary_repo", "any_nonsuccess"], sort=True):
        ids = sub.index.to_numpy()
        k = int(round(frac * len(ids)))
        val_groups.update(rng.permutation(ids)[:k].tolist())
    m.loc[in_pool, "split"] = np.where(m.loc[in_pool, "group_id"].isin(val_groups), "val", "train")
    # Conflicting pool episodes stay in train but are flagged; see the audit report.
    return m


def exclusions(m: pd.DataFrame) -> pd.Series:
    reason = pd.Series("", index=m.index)
    def add(mask, text):
        reason[mask] = np.where(reason[mask] == "", text, reason[mask] + ";" + text)
    add(m.outcome == "unknown", "unknown_outcome")
    add(m.endpoint_flags.str.contains("unknown|out_of_range", regex=True), "ambiguous_endpoint")
    add(m.test_overlap_conflict & m.split.isin(["train", "val"]), "shares_group_with_test")
    return reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mdir = Path(cfg["paths"]["manifest"]).parent
    m = pd.read_parquet(mdir / "episodes_unsplit.parquet")
    m = build_groups(m)
    links = m.attrs["link_counts"]
    in_split = m.repo.isin(list(cfg["train_pool"]) + list(cfg["test"]))
    # Grouping uses only the repos in the split; audit-only repos are grouped separately
    # below so they cannot merge two split groups through themselves.
    split_m = build_groups(m[in_split].drop(columns=["group_id"]))
    split_m = assign_splits(split_m, cfg)
    split_m["exclude_reason"] = exclusions(split_m)
    audit_m = m[~in_split].assign(split="audit_only", test_overlap_conflict=False, exclude_reason="not_in_split")
    out = pd.concat([split_m, audit_m], ignore_index=True)
    out.to_parquet(cfg["paths"]["manifest"])
    out.to_csv(cfg["paths"]["manifest_csv"], index=False)
    meta = {"link_counts_split_repos": split_m.attrs.get("link_counts"), "link_counts_all_repos": links,
            "validation": cfg["validation"], "state_quantization": STATE_QUANT}
    (mdir / "split_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(out.groupby(["split"]).agg(episodes=("episode_uid", "size"), groups=("group_id", "nunique")))
    print("excluded:", out[out.exclude_reason != ""].exclude_reason.value_counts().to_dict())


if __name__ == "__main__":
    main()
