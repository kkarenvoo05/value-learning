"""Build the standardized episode manifest from pinned metadata + frame summaries.

Usage:
    .venv/bin/python -m vl.manifest --config configs/dataset.yaml

Label precedence (from the release documentation, see reports/dataset_audit.md):
  1. Eval-round repos: meta/episode_provenance.parquet `outcome` and `num_steps`
     ("the reviewed values used by the paper"). Valid final frame T = num_steps - 1.
     Cross-checked against the latest human event in the matching .label_history.jsonl.
  2. Collection repos: latest human `outcome` event in .label_history.jsonl
     ("the latest human label wins"). T = outcome_frame.
  3. If the latest human event is {"action": "skip"}: fall back to the latest auto
     (collection-results) outcome event and record label_source = "auto_after_human_skip".
  4. Otherwise the outcome is "unknown" and the episode is excluded from targets (never
     silently turned into a failure).
Machine (VLM) stage predictions are carried only as `machine_*` diagnostics.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vl.config import all_repos, load_config

CAMERAS = ["observation.images.wrist_left", "observation.images.wrist_right",
           "observation.images.side_1", "observation.images.side_2"]


def round_of(name: str, axis: str) -> int | None:
    """Collection increment (axis='c') or evaluated model round (axis='r') from a repo name."""
    hit = re.search(rf"-{axis}(\d\d)-", name + "-")
    return int(hit.group(1)) if hit else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def latest_events(events: list[dict]) -> dict[tuple[int, str], dict]:
    """Latest event per (episode_index, source kind) for outcome labels; file order breaks ts ties."""
    out = {}
    for i, e in enumerate(events):
        if e.get("label_kind") != "outcome":
            continue
        key = (e["episode_index"], e["source"]["kind"])
        if key not in out or (e["ts"], i) >= (out[key]["ts"], out[key]["_i"]):
            out[key] = {**e, "_i": i}
    return out


def stage_events(events: list[dict]) -> dict[int, dict]:
    return {e["episode_index"]: e for e in events if e.get("label_kind") == "stage"}


def resolve_label(human: dict | None, auto: dict | None) -> dict:
    """Apply precedence rules 2-4 to one episode's label-history events."""
    if human is not None and "new_outcome" in human["payload"]:
        p = human["payload"]
        return {"outcome": p["new_outcome"], "T": p["outcome_frame"],
                "soft_truncate": bool(p.get("soft_truncate")), "label_source": "human",
                "label_tool": human["source"].get("tool"), "label_ts": human["ts"]}
    if auto is not None and "new_outcome" in auto["payload"]:
        p = auto["payload"]
        src = "auto_after_human_skip" if human is not None else "auto_only"
        return {"outcome": p["new_outcome"], "T": p["outcome_frame"],
                "soft_truncate": bool(p.get("soft_truncate")), "label_source": src,
                "label_tool": auto["source"].get("tool"), "label_ts": auto["ts"]}
    return {"outcome": "unknown", "T": None, "soft_truncate": False, "label_source": "none",
            "label_tool": None, "label_ts": None}


def load_review(repo_dir: Path, session: str | None):
    base = repo_dir / "meta/sessions" / session if session else repo_dir
    ev = read_jsonl(base / ".label_history.jsonl")
    return latest_events(ev), stage_events(ev)


def build_repo(name: str, cfg: dict, pins: dict, summary_dir: Path) -> pd.DataFrame:
    d = Path(cfg["paths"]["meta_root"]) / name
    info = json.loads((d / "meta/info.json").read_text())
    lineage = json.loads((d / "meta/dataset_lineage.json").read_text())
    eps = pd.read_parquet(next(d.glob("meta/episodes/*/*.parquet")))
    summ = pd.read_parquet(summary_dir / f"{name}.parquet").set_index("episode_index")
    prov_path = d / "meta/episode_provenance.parquet"
    prov = pd.read_parquet(prov_path).set_index("episode_index") if prov_path.exists() else None
    ledger = {}
    for lp in ["meta/protocol_quota_ledger.jsonl", "meta/teleop_manifest_ledger.jsonl"]:
        for e in read_jsonl(d / lp):
            ledger[e["episode_index"]] = e
    review_cache = {}

    rows = []
    for _, ep in eps.iterrows():
        i = int(ep.episode_index)
        s = summ.loc[i]
        r = {
            "repo": name,
            "repo_id": pins[name]["repo_id"],
            "commit": pins[name]["sha"],
            "episode_index": i,
            "episode_uid": f"{name}@{pins[name]['sha'][:12]}:{i}",
            "source_type": lineage.get("source_type"),
            "fps": info["fps"],
            "raw_length": int(ep.length),
        }
        # Camera availability: a camera counts as available only if the repo has the stream
        # AND the episode's video span matches its frame count.
        for cam in CAMERAS:
            short = cam.split(".")[-1]
            k0, k1 = f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"
            if k0 in eps:
                span = float(ep[k1] - ep[k0])
                r[f"cam_{short}"] = True
                r[f"cam_{short}_span_err_frames"] = round(span * info["fps"] - ep.length, 3)
            else:
                r[f"cam_{short}"] = False
                r[f"cam_{short}_span_err_frames"] = np.nan

        # ---- Provenance identity and label resolution
        if prov is not None:
            p = prov.loc[i]
            session = p.session_id
            src_ep = int(p.source_episode_index)
            r.update({
                "session_id": session,
                "source_repo": p.source_repo,
                "source_episode_index": src_ep,
                "release_block_repo": p.release_block_repo,
                "policy": p.policy, "policy_key": p.policy_key, "method": p.method,
                "eval_role": p.role, "actor_model_id": p.actor_model_id,
                "critic_model_id": p.critic_model_id, "start_id": int(p.start_id),
                "start_key": p.start_key, "manifest_idx": int(p.manifest_idx),
                "collection_round": None,
                "model_round": round_of(name, "r"),
            })
            multi = (d / "meta/sessions").exists()
            if (name, session if multi else None) not in review_cache:
                review_cache[(name, session if multi else None)] = load_review(d, session if multi else None)
            latest, stages = review_cache[(name, session if multi else None)]
            key_ep = src_ep if multi else i
            lab = resolve_label(latest.get((key_ep, "human")), latest.get((key_ep, "auto")))
            r["history_outcome"], r["history_T"], r["history_label_source"] = lab["outcome"], lab["T"], lab["label_source"]
            # Rule 1: provenance is authoritative for eval repos.
            r.update({"outcome": p.outcome, "T": int(p.num_steps) - 1,
                      "soft_truncate": lab["soft_truncate"],
                      "label_source": f"provenance(review={lab['label_source']})"})
            stage = stages.get(key_ep)
        else:
            led = ledger.get(i, {})
            latest, stages = review_cache.setdefault(name, load_review(d, None))
            lab = resolve_label(latest.get((i, "human")), latest.get((i, "auto")))
            r.update({
                "session_id": None, "source_repo": pins[name]["repo_id"], "source_episode_index": i,
                "release_block_repo": None,
                "policy": led.get("model_id") or ("teleop" if lineage.get("source_type") == "teleop" else None),
                "policy_key": led.get("arm_key") or led.get("manifest_source"),
                "method": lineage.get("source_type"), "eval_role": None,
                "actor_model_id": led.get("model_id"), "critic_model_id": None,
                "start_id": None, "start_key": None, "manifest_idx": led.get("manifest_idx"),
                "collection_round": round_of(name, "c"),
                "model_round": round_of(name, "r"),
                "outcome": lab["outcome"], "T": lab["T"], "soft_truncate": lab["soft_truncate"],
                "label_source": lab["label_source"],
                "history_outcome": lab["outcome"], "history_T": lab["T"],
                "history_label_source": lab["label_source"],
                "collection_time_outcome": led.get("outcome"),
                "ledger_steps": led.get("steps"),
                "is_counterfactual": led.get("is_counterfactual"),
                "ledger_had_intervention": led.get("had_intervention"),
                "ledger_intervention_count": led.get("intervention_count"),
                "arm": led.get("arm_key") or led.get("manifest_source"),
            })
            stage = stages.get(i)

        r["machine_stage"] = stage["payload"]["summary"].get("stage") if stage else None
        r["machine_failure_mode"] = stage["payload"]["summary"].get("failure_mode") if stage else None

        # ---- Frame-level facts
        for k in ["n_rows", "frame_index_contiguous", "ts_max_abs_err_s", "n_valid", "valid_is_prefix",
                  "first_invalid", "first_done", "done_is_suffix", "success_flag_any",
                  "first_success_flag", "n_intervention_onsets", "n_human_frames",
                  "first_human_frame", "content_fingerprint",
                  "init_pen_x", "init_pen_y", "init_pen_yaw", "init_holder_x", "init_holder_y"]:
            r[k] = s[k]
        r["init_state_constant"] = bool(all(s[f"{k}_constant"] for k in ["pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y"]))
        r["frame_policy_id"] = s.get("frame_policy_id")
        r["has_human_frames"] = bool(s.n_human_frames > 0)
        rows.append(r)
    return pd.DataFrame(rows)


def endpoint_checks(m: pd.DataFrame) -> pd.DataFrame:
    """Flag episodes whose resolved endpoint is unknown, out of range, or contradicts frame flags."""
    m = m.copy()
    T = m["T"]
    known = m.outcome.isin(["success", "failure", "timeout"]) & T.notna()
    m["valid_length"] = np.where(known, T + 1, np.nan)
    in_range = known & (T >= 0) & (T <= m.raw_length - 1)
    # Per-frame `done` marks the reviewed terminal for success/failure (verified in the audit);
    # `is_valid` marks the reviewed prefix for soft-truncated episodes.
    done_ok = np.where(m.outcome.isin(["success", "failure"]), m.first_done == T, True)
    trunc_ok = np.where(m.soft_truncate, m.n_valid == T + 1, True)
    hist_ok = (m.history_outcome == m.outcome) & (m.history_T == m["T"])
    reasons = []
    for k, ok_known, ok_rng, ok_done, ok_tr, ok_hist in zip(range(len(m)), known, in_range, done_ok, trunc_ok, hist_ok):
        r = []
        if not ok_known:
            r.append("unknown_outcome_or_endpoint")
        elif not ok_rng:
            r.append("endpoint_out_of_range")
        if ok_known and not ok_done:
            r.append("done_flag_disagrees_with_endpoint")
        if ok_known and not ok_tr:
            r.append("is_valid_disagrees_with_soft_truncate")
        if ok_known and not ok_hist:
            r.append("provenance_disagrees_with_label_history")
        reasons.append(";".join(r))
    m["endpoint_flags"] = reasons
    m["frames_after_endpoint"] = np.where(known, m.raw_length - 1 - T, np.nan)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pins = json.loads((Path(cfg["paths"]["meta_root"]) / "pins.json").read_text())
    summary_dir = Path(cfg["paths"]["manifest"]).parent / "frame_summary"
    parts = [build_repo(n, cfg, pins, summary_dir) for n in all_repos(cfg, include_audit_extra=True)]
    m = endpoint_checks(pd.concat(parts, ignore_index=True))
    out = Path(cfg["paths"]["manifest"]).parent / "episodes_unsplit.parquet"
    m.to_parquet(out)
    print(f"wrote {out}: {len(m)} episodes")
    print(m.groupby("repo").outcome.value_counts().unstack(fill_value=0))
    print("endpoint flags:", m.endpoint_flags.replace("", "ok").value_counts().to_dict())


if __name__ == "__main__":
    main()
