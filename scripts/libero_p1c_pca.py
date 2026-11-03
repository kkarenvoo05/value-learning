"""P1c dimension control (docs/libero_plan.md 2026-11-03) [ours]: is the drop from frozen SigLIP (head SNR 0.122) to
the layer-1 attention values V (0.093) just V's 256-d projection? Per camera, PCA to 256 dims fit on the fit frames
(fit_fixed minus inner, as P1b's analyze), for siglip_pi05 and llm_final, written next to the P1b arrays as
<name>_pca256.npy (n, 2, 256) float16. Then analyze with --fit-subset fit_fixed.txt --tag pca.
Reading: siglip_pi05_pca256 near V1 (~0.09) -> the layer-1 gap is the projection; near 0.12 -> the early blocks add
noise. llm_final_pca256 vs llm_v17 (0.060) checks the same at the top.
Usage: PYTHONPATH=scripts:src .venv/bin/python scripts/libero_p1c_pca.py [--dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vl.libero_probe import inner_split

DATASET = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/iris/u/ankile/karenvo/libero/features/frozen_encoders_p1b")
    ap.add_argument("--encoders", nargs="+", default=["siglip_pi05", "llm_final"])
    ap.add_argument("--k", type=int, default=256)
    a = ap.parse_args()
    d = Path(a.dir)
    meta = pd.read_parquet(d / "meta.parquet").reset_index(drop=True)
    keep = {int(x) for x in (DATASET / "subsets/fit_fixed.txt").read_text().splitlines()
            if x.strip() and not x.startswith("#")}
    tr = np.flatnonzero((meta.split == "train").values & ~inner_split(meta) & meta.episode_index.isin(keep).values)
    for enc in a.encoders:
        X = np.load(d / f"{enc}.npy", mmap_mode="r")
        outs = []
        for c in range(X.shape[1]):
            x = np.asarray(X[:, c], np.float32)
            mu = x[tr].mean(0)
            _, S, Vt = np.linalg.svd(x[tr[::2]] - mu, full_matrices=False)
            share = float((S[:a.k] ** 2).sum() / (S ** 2).sum())
            outs.append(((x - mu) @ Vt[:a.k].T).astype(np.float16))
            print(f"{enc} cam {c}: {x.shape[1]} -> {a.k}, variance kept {share:.3f}", flush=True)
        np.save(d / f"{enc}_pca{a.k}.npy", np.stack(outs, 1))


if __name__ == "__main__":
    main()
