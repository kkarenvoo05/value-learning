from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def all_repos(cfg: dict, include_audit_extra: bool = False) -> list[str]:
    repos = list(cfg["train_pool"]) + list(cfg["test"])
    if include_audit_extra:
        repos += list(cfg.get("overlap_audit_extra", []))
    return repos
