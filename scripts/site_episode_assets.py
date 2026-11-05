"""Episode-player assets for the week-7 site page (scripts/build_site.py) [ours, 2026-11-03].

Val predictions of four fixed+fresh seed-0 value functions (ResNet 18237895, frozen B 18237898, LoRA 18258391,
SigLIP ft 18269859) -> per-episode MAE, flicker (sd of frame-to-frame change, 0.4 s grid) and jump at the first pot
placement (mean V 0.8-2.8 s after minus 0-2 s before). Writes reports/site/content/episodes.json (traces + metrics +
medians over all 120 val episodes) and frames/ep<N>_frames.jpg (sprite: one 144 px column per grid frame, row 0 agent
view, row 1 wrist) for the chosen episodes. Episode choice (2026-11-03): 454 = clearest flicker after placement,
534 = a typical episode where pi05 is more accurate, 469 = clear milestone step.
Usage: .venv/bin/python scripts/site_episode_assets.py [--episodes 454 534 469]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.libero_milestone import episodes  # noqa: E402
from vl.libero_resnet import table_meta  # noqa: E402

R = Path("/iris/u/ankile/karenvo/libero/runs")
D = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
OUT = Path(__file__).resolve().parents[1] / "reports/site/content"
ARMS = {"resnet": ("resnet18_fixedfresh_s0_18237895", "ev_resnet"), "b": ("value_expert_fixedfresh_w256_s0_18237898", "ev_expert"),
        "lora": ("value_lora_fixedfresh_lr1e4_s0_18258391", "ev_expert"), "siglip": ("siglip_pi05_fixedfresh_s0_18269859", "ev_resnet")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, nargs="+", default=[454, 534, 469]); a = ap.parse_args()
    val = episodes("val").set_index("episode_index")
    df = None
    for k, (run, col) in ARMS.items():
        p = pd.read_parquet(R / run / "val_predictions.parquet")[["episode_index", "frame_index", "success", "value", col]].rename(columns={col: k})
        df = p if df is None else df.merge(p, on=["episode_index", "frame_index", "success", "value"])
    df = df.sort_values(["episode_index", "frame_index"])
    met = {}
    for ep, g in df.groupby("episode_index"):
        ms, fi = val.milestone.get(ep, np.nan), g.frame_index.values
        met[ep] = {}
        for k in ARMS:
            v = g[k].values
            jump = (v[(fi > ms + 8) & (fi <= ms + 28)].mean() - v[(fi >= ms - 20) & (fi < ms)].mean()) if np.isfinite(ms) else np.nan
            met[ep][k] = {"mae": float(np.abs(v - g.value.values).mean()), "flicker": float(np.std(np.diff(v))), "jump": float(jump)}
    m = table_meta(D, 4); arr = np.load(D / "frames_every4.npy", mmap_mode="r"); P = 144
    out = {}
    for ep in a.episodes:
        idx = np.flatnonzero(m.episode_index.values == ep); fr = m.frame_index.values[idx]
        g = df[df.episode_index == ep].set_index("frame_index").loc[fr]
        sp = Image.new("RGB", (P * len(idx), 2 * P))
        for j, i in enumerate(idx):
            for c in range(2):
                sp.paste(Image.fromarray(arr[i][c].transpose(1, 2, 0)).resize((P, P), Image.LANCZOS), (j * P, c * P))
        (OUT / "frames").mkdir(parents=True, exist_ok=True); sp.save(OUT / "frames" / f"ep{ep}_frames.jpg", quality=72)
        out[str(ep)] = {"t": (fr / 10).round(1).tolist(), "target": g.value.round(4).tolist(), "milestone_s": float(val.milestone[ep]) / 10,
                        "success": int(g.success.iloc[0]), "init_state": int(val.init_state[ep]), "px": P,
                        "series": {k: g[k].round(4).tolist() for k in ARMS},
                        "metrics": {k: {q: round(x, 3) for q, x in met[ep][k].items()} for k in ARMS}}
    out["median_all_val"] = {k: {q: round(float(np.nanmedian([met[e][k][q] for e in met])), 3) for q in ("mae", "flicker", "jump")} for k in ARMS}
    (OUT / "episodes.json").write_text(json.dumps(out)); print(json.dumps(out["median_all_val"]))


if __name__ == "__main__":
    main()
