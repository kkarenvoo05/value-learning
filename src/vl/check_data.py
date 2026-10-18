"""Integrity of the downloaded training data (Stage 3.1).

Usage:
    .venv/bin/python -m vl.check_data            # -> reports/data_integrity.md

Checks, for every train-pool repo and each configured camera plus the trajectory parquet:
  1. every file's sha256 equals the LFS sha256 at the pinned commit (pins.json);
  2. each episode's video span holds a frame at from_timestamp + k / fps for every k = 0..T
     (packet timestamps only; no decoding), for both cameras -> frame k is row k in each.
     Frames missing only after T are listed but harmless (never sampled);
  3. the parquet has `length` rows per episode, frame_index 0..length-1, timestamp = k / fps,
     the same is_valid count the manifest was built from (`n_valid`, read remotely in Stage 1),
     and for soft-truncated episodes is_valid covers exactly 0..T.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from vl.config import load_config
from vl.frames import video_locations


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def check_hashes(repo_id: str, sha: str, root: Path, prefixes: list[str]) -> list[str]:
    """Problems found (empty = all files present and identical to the pinned commit)."""
    api, remote = HfApi(), {}
    for p in prefixes:
        for f in api.list_repo_tree(repo_id, repo_type="dataset", revision=sha, path_in_repo=p,
                                    recursive=True, expand=True):
            if getattr(f, "lfs", None) is not None:
                remote[f.path] = f.lfs.sha256
    local = {str(p.relative_to(root)) for pre in prefixes for p in (root / pre).rglob("*") if p.is_file()}
    problems = [f"missing {p}" for p in sorted(set(remote) - local)]
    problems += [f"not in pinned commit {p}" for p in sorted(local - set(remote))]
    common = sorted(set(remote) & local)
    with ThreadPoolExecutor(8) as ex:
        for p, h in zip(common, ex.map(lambda q: sha256(root / q), common)):
            if h != remote[p]:
                problems.append(f"sha256 mismatch {p}")
    return problems, len(common)


def packet_times(path: str) -> np.ndarray:
    with av.open(path) as c:
        s = c.streams.video[0]
        pts = [p.pts for p in c.demux(s) if p.pts is not None]
        return np.sort(np.array(pts, dtype=np.int64)) * float(s.time_base)


def check_video_spans(locs: pd.DataFrame, T: pd.Series, fps: float) -> pd.DataFrame:
    """Per (episode, camera): how many leading frames 0, 1, ... sit exactly at from_ts + k / fps."""
    rows = []
    for path, g in locs.groupby("path"):
        t = packet_times(path)
        for e in g.itertuples():
            # The episode's span ends at to_ts: anything later is the next episode in the file.
            end = min(e.from_ts + (e.length - 0.5) / fps, e.to_ts - 0.5 / fps)
            inside = t[(t >= e.from_ts - 0.5 / fps) & (t < end)]
            expect = e.from_ts + np.arange(e.length) / fps
            m = min(len(inside), e.length)
            good = np.abs(inside[:m] - expect[:m]) < 0.25 / fps
            leading = m if good.all() else int(np.argmin(good))
            rows.append({"repo": e.repo, "episode_index": e.episode_index, "cam": e.cam,
                         "length": e.length, "n_frames": len(inside), "leading_ok": leading,
                         "T": T.get(e.episode_index, np.nan)})
    out = pd.DataFrame(rows)
    out["covers_T"] = out["T"].isna() | (out.leading_ok >= out["T"] + 1)   # NaN T: not in a split
    out["complete"] = (out.leading_ok == out.length) & (out.n_frames == out.length)
    return out


def check_parquet(root: Path, man: pd.DataFrame, fps: float) -> pd.DataFrame:
    cols = ["episode_index", "frame_index", "timestamp", "is_valid"]
    d = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted((root / "data").rglob("*.parquet")))
    rows = []
    for ep, g in d.groupby("episode_index"):
        g = g.sort_values("frame_index")
        n = len(g)
        valid = g.is_valid.to_numpy().astype(bool)
        rows.append({"episode_index": ep, "n_rows": n, "local_n_valid": int(valid.sum()),
                     "frame_index_ok": (g.frame_index.to_numpy() == np.arange(n)).all(),
                     "timestamp_ok": np.abs(g.timestamp.to_numpy() - np.arange(n) / fps).max() < 1e-3})
    out = pd.DataFrame(rows).merge(man[["episode_index", "raw_length", "n_valid", "T", "soft_truncate"]],
                                   on="episode_index", how="left")
    out["rows_ok"] = out.n_rows == out.raw_length
    out["valid_ok"] = (out.local_n_valid == out.n_valid) & np.where(
        out.soft_truncate.fillna(False).astype(bool), out.local_n_valid == out["T"] + 1, True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--skip-hashes", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = load_config(cfg["dataset_config"])
    cams, vroot, meta = cfg["cameras"], Path(cfg["paths"]["video_root"]), Path(ds["paths"]["meta_root"])
    pins = json.loads((meta / "pins.json").read_text())
    manifest = pd.read_parquet(ds["paths"]["manifest"])
    lines, summary = [], []
    for repo in ds["train_pool"]:
        man = manifest[manifest.repo == repo]
        fps = float(man.fps.iloc[0])
        root = vroot / repo
        n_hashed, hash_problems = 0, []
        if not args.skip_hashes:
            hash_problems, n_hashed = check_hashes(f"{ds['hf_namespace']}/{repo}", pins[repo]["sha"], root,
                                                   [f"videos/observation.images.{c}" for c in cams] + ["data"])
        locs = video_locations(meta, vroot, repo, cams)
        T = man[man.split.isin(["train", "val"])].set_index("episode_index")["T"]
        spans = check_video_spans(locs, T, fps)
        pq = check_parquet(root, man, fps)
        bad_span = spans[~spans.covers_T]
        short_after_T = spans[spans.covers_T & ~spans.complete]
        bad_pq = pq[~(pq.rows_ok & pq.frame_index_ok & pq.timestamp_ok & pq.valid_ok)]
        summary.append({"repo": repo, "commit": pins[repo]["sha"][:12], "files_hashed": n_hashed,
                        "hash_problems": len(hash_problems), "episodes": len(pq),
                        "video_spans_checked": len(spans), "spans_missing_frames_le_T": len(bad_span),
                        "spans_short_after_T_only": len(short_after_T),
                        "bad_parquet_episodes": len(bad_pq)})
        lines += [f"- {repo}: {p}" for p in hash_problems]
        lines += [f"- {repo} ep{r.episode_index} {r.cam}: frames 0..{r.leading_ok - 1} of {r.length} "
                  f"present (T = {r.T:.0f}){'' if r.covers_T else ' -- MISSING FRAMES <= T'}"
                  for r in pd.concat([bad_span, short_after_T]).itertuples()]
        lines += [f"- {repo} ep{r.episode_index}: parquet {r._asdict()}" for r in bad_pq.itertuples()]
        print(summary[-1], flush=True)
    out = Path("reports/data_integrity.md")
    out.write_text("# Training-data integrity (Stage 3.1)\n\nGenerated by `python -m vl.check_data`. "
                   f"Cameras: {', '.join(cams)} + trajectory parquet.\n\n"
                   + pd.DataFrame(summary).to_markdown(index=False)
                   + "\n\n## Problems\n\n" + ("\n".join(lines) if lines else "None.") + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
