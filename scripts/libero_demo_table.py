"""Stage-1 demo table (docs/libero_demo_pretrain_draft.md): physical-intelligence/libero (LeRobot v2.0) -> raw files.

Writes the same per-episode raw .npz format as the moka rollouts (scripts/libero_rollouts.py), so the existing
loaders read it unchanged: image / wrist_image (T, 224, 224, 3) uint8 resized with openpi_client's
resize_with_pad (the rollouts' and the policy's resize; the dataset's 256 x 256 images are already rotated like
the rollouts', checked by eye 2026-11-02), state (T, 8), actions (T, 7), prompt (the task string). Plus
<out>/episodes.csv with the moka table's columns (split train, init_state -1, success 1, raw absolute) and
suite / task / role ('fit' or 'inner', vl.libero_demos.assign_roles). Targets: `python -m vl.libero_demos
--dataset <out>` in the project venv afterwards.

Usage (openpi env, CPU only):
    LIBERO_CONFIG_PATH=/iris/u/ankile/karenvo/libero/libero_cfg .venv/bin/python \
        <value-learning>/scripts/libero_demo_table.py --src <download dir> --out <table dir> [--exclude-task "..."]
"""

import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.libero_demos import MAX_STEPS, assign_roles  # noqa: E402  (pure pandas / numpy)

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]  # openpi's convert script, no libero_90


BDDL = Path("/iris/u/ankile/karenvo/pi05/openpi/third_party/libero/libero/libero/bddl_files")


def task_language(filename: str) -> str:
    """LIBERO's benchmark.grab_language_from_filename (the `libero` package is not in the openpi env): the task
    string comes from the .bddl file name, not its (:language ...) line, which differs for some tasks."""
    x = filename
    if x[0].isupper():
        language = " ".join(x[x.find("SCENE") + (8 if "SCENE10" in x else 7):].split("_"))
    else:
        language = " ".join(x.split("_"))
    return language[:language.find(".bddl")]


def suite_of_task() -> dict:
    """task language -> suite, over each suite's .bddl task files."""
    out = {}
    for s in SUITES:
        files = sorted((BDDL / s).glob("*.bddl"))
        assert len(files) == 10, (s, len(files))
        for f in files:
            lang = task_language(f.name)
            assert out.get(lang, s) == s, f"task {lang!r} in two suites"
            out[lang] = s
    return out


def convert(job):
    src, dst, task = job
    if Path(dst).exists():
        return dst
    from openpi_client import image_tools
    from PIL import Image
    df = pd.read_parquet(src)
    assert (df.frame_index.values == np.arange(len(df))).all(), src
    dec = lambda col: np.stack([np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB")) for x in df[col]])
    img, wr = dec("image"), dec("wrist_image")
    assert img.shape[1:] == (256, 256, 3) and wr.shape == img.shape, (src, img.shape)
    tmp = Path(dst).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, image=image_tools.resize_with_pad(img, 224, 224),
                        wrist_image=image_tools.resize_with_pad(wr, 224, 224),
                        state=np.stack(df.state.values).astype(np.float64),
                        actions=np.stack(df.actions.values).astype(np.float64), prompt=np.array(task))
    tmp.rename(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-task", action="append", default=[], help="task string to leave out (repeatable)")
    ap.add_argument("--inner-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "meta/info.json").read_text())
    assert info["fps"] == 10 and info["codebase_version"] == "v2.0", info
    eps = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    suite = suite_of_task()
    tasks = {e["tasks"][0] for e in eps}
    assert tasks <= set(suite), f"dataset tasks without a suite: {sorted(tasks - set(suite))[:3]}"
    rows = []
    for e in eps:
        (task,) = e["tasks"]
        if task in a.exclude_task:
            continue
        i = e["episode_index"]
        chunk = i // info["chunks_size"]
        rows.append(dict(episode_index=i, split="train", init_state=-1, repeat=0, source_episode=i,
                         policy_steps=e["length"], success=1, timeout=0, suite=suite[task], task=task,
                         src=str(src / info["data_path"].format(episode_chunk=chunk, episode_index=i)),
                         raw=str(out / "raw" / f"ep{i:05d}.npz")))
    missing = set(a.exclude_task) - {e["tasks"][0] for e in eps}
    assert not missing, f"--exclude-task not in the dataset: {missing}"
    ep = pd.DataFrame(rows)
    assert (ep.policy_steps <= ep.suite.map(MAX_STEPS)).all()
    ep["role"] = assign_roles(ep, a.inner_frac, 0)
    if a.limit:  # smoke: roles assigned on the full table first, so they match the full run
        ep = pd.concat([ep[ep.role == "fit"].head(a.limit - 2), ep[ep.role == "inner"].head(2)])
    with ProcessPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(convert, zip(ep.src, ep.raw, ep.task), chunksize=4)):
            if k % 100 == 0:
                print(f"{k}/{len(ep)} converted", flush=True)
    with np.load(ep.raw.iloc[0]) as z:  # spot check
        assert len(z["image"]) == ep.policy_steps.iloc[0] and z["image"].shape[1:] == (224, 224, 3)
    ep.drop(columns="src").to_csv(out / "episodes.csv", index=False)
    print(f"wrote {out / 'episodes.csv'}: {len(ep)} episodes, {ep.policy_steps.sum()} frames;",
          ep.groupby(["suite", "role"]).size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
