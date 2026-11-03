"""P1b (docs/libero_plan.md 2026-11-02): put frozen pi05 language-model features into the P1 table under P1's exact
protocol (same frames, same splits, same MLP head, same scoring) [ours].

P1 found frozen pi05 SigLIP at head SNR 0.172 (= ImageNet ResNet), but earlier heads on frozen pi05 LLM features
(scripts/libero_layer_value.py) gave 0.08-0.11 with a different head (AttnProbe on 33 grid tokens). P1b asks whether
that drop survives a matched head.

The cached LLM features exist only for the 570 Stage 2 ("fixed") episodes, so this writes a P1-format directory
restricted to those episodes (all 120 val episodes are among them; fit = subsets/fit_fixed.txt). Every encoder is
re-scored on that subset, so the rows compare with each other, not with the P1 table (fit there was 825 episodes).
Encoders written to <out>/<name>.npy, float16 (n, 2, d), cameras [agentview, wrist] as in P1:
  resnet_in, siglip_pi05, siglip_web, resnet_ft, siglip_pi05_ft   P1's arrays, rows subset
  llm_final   frozen pi05_libero final-layer prefix hidden states (Stage 3a, feats_every4_18163142.npz poolings 1 / 2:
              mean over each image's 256 tokens), 2048-d
  llm_v{1,5,9,13,17}  per-layer attention values V (progress_18191428/v_layers.npy, 256-d); each camera's 16 grid-4
              cells are equal-size blocks, so their mean = the mean over all 256 image tokens
Then: PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze --out <out> \
          --fit-subset fit_fixed.txt --encoders <names>
Usage: .venv/bin/python scripts/libero_p1b_prepare.py [--out DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEAT = Path("/iris/u/ankile/karenvo/libero/features")
P1 = FEAT / "frozen_encoders"
P1_ENC = ["resnet_in", "siglip_pi05", "siglip_web", "resnet_ft", "siglip_pi05_ft"]
LAYERS = [1, 5, 9, 13, 17]
KEY = ["split", "init_state", "episode", "frame_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FEAT / "frozen_encoders_p1b"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(P1 / "meta.parquet").reset_index(drop=True)

    z = np.load(FEAT / "feats_every4_18163142.npz")
    s = pd.DataFrame({k: z[k] for k in KEY + ["T", "success"]})
    assert list(z["poolings"][1:3]) == ["agentview", "wrist"], z["poolings"]
    pm = np.load(FEAT / "progress_18191428/meta.npz")
    assert list(pm["layers"]) == LAYERS, pm["layers"]
    for k in KEY:  # the per-layer cache uses the same frame order as Stage 3a
        assert (pm[k] == z[k]).all(), k

    s["src"] = np.arange(len(s))
    meta["p1"] = np.arange(len(meta))
    j = meta.merge(s, on=KEY, how="inner", suffixes=("", "_s")).sort_values("p1").reset_index(drop=True)
    assert len(j) == len(s), (len(j), len(s))  # every Stage 3a frame found in P1
    assert (j["T"] == j["T_s"]).all() and (j.success == j.success_s).all()
    assert (j.split == "val").sum() == (meta.split == "val").sum()  # all val frames kept
    rows, src = j.p1.values, j.src.values
    j[meta.columns.drop("p1")].to_parquet(out / "meta.parquet")
    print(f"{len(j)} frames ({j.episode_index.nunique()} episodes) of P1's {len(meta)}", flush=True)

    st = P1 / "state.npy"
    if st.exists():
        np.save(out / "state.npy", np.load(st)[rows])
    for k in P1_ENC:
        np.save(out / f"{k}.npy", np.load(P1 / f"{k}.npy", mmap_mode="r")[rows])
        print("wrote", k, flush=True)
    f = z["feats"][src][:, 1:3]  # (n, 2, 2048)
    np.save(out / "llm_final.npy", f.astype(np.float16))
    print("wrote llm_final", f.shape, flush=True)
    V = np.load(FEAT / "progress_18191428/v_layers.npy", mmap_mode="r")  # (n, 5, 33, 256)
    for i, L in enumerate(LAYERS):
        v = np.asarray(V[:, i], dtype=np.float32)[src]
        cams = np.stack([v[:, :16].mean(1), v[:, 16:32].mean(1)], 1)  # agentview cells, wrist cells
        np.save(out / f"llm_v{L}.npy", cams.astype(np.float16))
        print(f"wrote llm_v{L}", cams.shape, flush=True)
    # the fine-tuned models' val probabilities: val rows are all kept and stay in P1 order, so copy as-is
    for k in ("resnet_ft", "siglip_pi05_ft"):
        p = P1 / f"probs_{k}_val.npy"
        if p.exists():
            (out / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()
