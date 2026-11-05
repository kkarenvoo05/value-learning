"""Stage 3a comparison table across probe and ResNet runs on the same val frames (docs/libero_plan.md).

Usage:
    .venv/bin/python -m vl.libero_compare --ref probe=<dir>:ev_all_mlp \
        resnet=<dir>:ev_resnet probe_hist=<dir>:ev_all_mlp ...

Each model is NAME=DIR:COLUMN, DIR holding val_predictions.parquet (vl.libero_probe or vl.libero_resnet), COLUMN
its E[V]. Rows are joined on (episode_index, frame_index), so all models are scored on the same frames.
Per model: MAE (all frames), MAE on success frames after 20 s (where the single-frame ResNet beat the probe),
MAE on failure frames, success AUC at 5 / 10 / 20 / 30 s, and the MAE difference to --ref with a 95% bootstrap
interval over the 10 val init states (2000 resamples; scenes are the unit, frames within a scene are not
independent). Writes nothing; prints a markdown table.
"""

import argparse

import numpy as np
import pandas as pd

from vl.metrics import auc

KEYS = ["episode_index", "frame_index"]
FPS = 10


def load(spec: str) -> tuple[str, pd.DataFrame]:
    name, rest = spec.split("=", 1)
    d, col = rest.rsplit(":", 1)
    p = pd.read_parquet(f"{d}/val_predictions.parquet")
    base = p[KEYS + ["init_state", "success", "value"]] if "init_state" in p else p[KEYS + ["success", "value"]]
    return name, base.assign(**{name: p[col].values})


def auc_at(d: pd.DataFrame, col: str, s: int) -> float:
    w = d[(d.frame_index - s * FPS).abs() <= FPS // 2]
    one = w.loc[(w.frame_index - s * FPS).abs().groupby(w.episode_index).idxmin()]
    return auc(one.success.values, one[col].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="NAME=DIR:COLUMN")
    ap.add_argument("--ref", required=True, help="NAME=DIR:COLUMN of the reference model")
    a = ap.parse_args()
    ref_name, d = load(a.ref)
    names = [ref_name]
    for spec in a.models:
        n, p = load(spec)
        d = d.merge(p[KEYS + [n]], on=KEYS, how="inner", validate="one_to_one")
        names.append(n)
    late = (d.success == 1) & (d.frame_index >= 20 * FPS)
    rng = np.random.default_rng(0)
    states = d.init_state.unique()
    rows = []
    for n in names:
        err = (d[n] - d.value).abs()
        r = {"model": n, "MAE": err.mean(), "MAE success >20 s": err[late].mean(), "MAE failure": err[d.success == 0].mean()}
        for s in (5, 10, 20, 30):
            r[f"AUC {s}s"] = auc_at(d, n, s)
        if n != ref_name:
            diff = (err - (d[ref_name] - d.value).abs()).groupby(d.init_state)
            per = diff.mean()
            cnt = diff.size()
            bs = []
            for _ in range(2000):
                pick = rng.choice(states, len(states))
                bs.append((per[pick] * cnt[pick]).sum() / cnt[pick].sum())
            r[f"ΔMAE vs {ref_name} [95% scenes]"] = (f"{err.mean() - (d[ref_name] - d.value).abs().mean():+.3f} "
                                                     f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
        rows.append(r)
    t = pd.DataFrame(rows)
    print(f"{len(d)} frames, {d.episode_index.nunique()} episodes, {len(states)} scenes\n")
    print(t.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
