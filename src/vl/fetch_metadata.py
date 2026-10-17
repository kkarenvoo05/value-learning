"""Resolve each configured repo to an immutable commit and download its metadata only.

Usage:
    .venv/bin/python -m vl.fetch_metadata --config configs/dataset.yaml

Writes data/meta/<repo>/... and data/meta/pins.json ({repo: {sha, tag, bytes...}}).
No videos and no trajectory parquet are downloaded here.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from vl.config import all_repos, load_config


def resolve_pin(api: HfApi, repo_id: str, tag: str) -> dict:
    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    tags = {t.name: t.target_commit for t in refs.tags}
    branches = {b.name: b.target_commit for b in refs.branches}
    if tag not in tags:
        raise RuntimeError(f"{repo_id}: tag {tag!r} not found (tags={list(tags)})")
    sha = tags[tag]
    info = api.dataset_info(repo_id, revision=sha, files_metadata=True)
    sizes = defaultdict(int)
    for s in info.siblings:
        f = s.rfilename
        kind = f.split("/")[1] if f.startswith("videos/") else f.split("/")[0]
        sizes[kind] += s.size or 0
    return {
        "repo_id": repo_id,
        "tag": tag,
        "sha": sha,
        "main_sha": branches.get("main"),
        "tag_equals_main": branches.get("main") == sha,
        "bytes_by_group": dict(sizes),
        "bytes_total": sum(sizes.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    api = HfApi()
    meta_root = Path(cfg["paths"]["meta_root"])
    meta_root.mkdir(parents=True, exist_ok=True)

    pins = {}
    for name in all_repos(cfg, include_audit_extra=True):
        repo_id = f"{cfg['hf_namespace']}/{name}"
        pin = resolve_pin(api, repo_id, cfg["revision_tag"])
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=pin["sha"],  # download by SHA, never by moving ref
            allow_patterns=cfg["metadata_patterns"],
            local_dir=meta_root / name,
        )
        pins[name] = pin
        print(f"{name:45s} {pin['sha'][:12]}  {pin['bytes_total'] / 1e9:6.2f} GB total")

    (meta_root / "pins.json").write_text(json.dumps(pins, indent=2))
    total = sum(p["bytes_total"] for p in pins.values())
    print(f"Full download of all {len(pins)} repos would be {total / 1e9:.1f} GB.")


if __name__ == "__main__":
    main()
