"""Scroll-through viewer data for the advantage examples (2026-11-04, Karen): per episode, an MP4 of every recorded
step (agentview | wrist, as the policy saw them, 20 fps = real time: LIBERO control_freq 20, docs/libero_plan.md
2026-11-03) and per-grid-frame values / advantages / labels / the ResNet's 201-bin distribution (the every-4th-step
grid V and the labels live on), for reports/<dir>/viewer/index.html (static page, no server).

Same sources and checks as scripts/libero_advantage_examples.py (whose helpers it imports): labels from the arm C
label tables, 201-bin probabilities from each episode's out-of-fold ResNet (asserted to reproduce the stored V).

Usage: .venv/bin/python scripts/libero_advantage_viewer.py --out reports/2026-11-04-advantage-examples/viewer
       [--episodes 139 87 ...] [--random 8]
Episodes: the hand-picked ones of ../summary.json (or --episodes), plus --random successes and as many failures
drawn uniformly from the 900 train rollouts (seed 0) [ours: so typical episodes sit next to the extreme picks].
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_advantage_examples as X  # noqa: E402


STEP_HZ = 20  # LIBERO env control_freq (real time); X.FPS is the step rate the figures' time axis uses


def r(x, n=4):
    return [round(float(v), n) for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--random", type=int, default=8, help="random successes and as many random failures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", default="reports/2026-11-04-label-rules",
                    help="dir with rule_labels.parquet + summary.json from scripts/libero_label_rules.py (or '')")
    ap.add_argument("--skip-video", action="store_true", help="keep existing videos (data refresh only)")
    ap.add_argument("--denoise", default="reports/2026-11-05-label-denoise",
                    help="dir with denoise_labels.parquet + denoise_table.csv from scripts/libero_label_denoise.py (or '')")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((out.parent / "summary.json").read_text())
    picked = {e["episode"]: e["picked_as"] for e in summ["episodes"]}
    eps_list = a.episodes or [139, 87, 1099, 1176, 1265]
    group = {e: "picked" for e in eps_list}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, thr = X.load_labels()
    eps = pd.read_csv(X.TABLE / "episodes.csv").set_index("episode_index")
    rng = np.random.default_rng(a.seed)
    train = eps[eps.split == "train"]
    for ok, name in ((1, "random success"), (0, "random failure")):
        pool = sorted(set(train.index[train.success == ok]) - set(eps_list))
        for e in sorted(rng.choice(pool, a.random, replace=False).tolist()):
            eps_list.append(int(e)); group[int(e)] = name; picked[int(e)] = name
    meta = X.table_meta(X.TABLE, 4)
    arr = np.load(X.TABLE / "frames_every4.npy", mmap_mode="r")
    index = {"thresholds": thr, "centers": r(X.CENTERS), "step_hz": STEP_HZ, "episodes": []}
    rules = None
    if a.rules:
        rules = pd.read_parquet(Path(a.rules) / "rule_labels.parquet")
        rs = json.loads((Path(a.rules) / "summary.json").read_text())
        index["rules"] = rs["rules"]; index["rule_thresholds"] = rs["resnet"]["thresholds"]
    # label sets for the viewer: threshold rules R0-R3 (A as trained; R3's threshold is on A + cost) and the denoising
    # variants (their own A); thr = [lo, hi] in that set's A units, lo == hi for binary tags, -1 label = untagged
    sets, den = {}, None
    if rules is not None:
        rt = index["rule_thresholds"]
        for k, nm in index["rules"].items():
            sets[k] = {"name": f"{k} {nm}", "thr": None if k == "R3" else [rt[k], rt[k]], "own_A": False}
    if a.denoise:
        den = pd.read_parquet(Path(a.denoise) / "denoise_labels.parquet")
        dt = pd.read_csv(Path(a.denoise) / "denoise_table.csv")
        for j, row_ in enumerate(dt.itertuples()):
            if row_.variant.startswith("current"):
                continue
            sets[f"D{j}"] = {"name": row_.variant, "col": row_.variant, "thr": [row_.thr_lo, row_.thr_hi], "own_A": True}
    index["labelsets"] = {k: {kk: vv for kk, vv in v.items() if kk != "col"} for k, v in sets.items()}
    for e in eps_list:
        de = d[d.episode_index == e].sort_values("frame_index")
        fold = int(de.fold.iloc[0])
        probs, frames = X.resnet_probs(e, fold, meta, arr, device)
        assert (de.frame_index.values == frames).all()
        ev = probs @ X.CENTERS
        assert np.abs(ev - de.V_r.values).max() < 0.01
        row = eps.loc[e]
        k = X.debounce(X.pots_on_stove(row.raw))
        rows_idx = np.flatnonzero(meta.episode_index.values == e)
        with np.load(row.raw) as z:
            video = np.concatenate([z["image"], z["wrist_image"]], 2)  # (T, 224, 448, 3), every step
        assert len(video) == row.policy_steps, (len(video), row.policy_steps)
        if a.skip_video and (out / "video" / f"ep{e:04d}.mp4").exists():
            video = None
        (out / "video").mkdir(exist_ok=True)
        if video is not None:
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{video.shape[2]}x{video.shape[1]}", "-r", str(STEP_HZ), "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-g", "10",
                                   "-movflags", "+faststart", str(out / "video" / f"ep{e:04d}.mp4")],
                                  stdin=subprocess.PIPE)
            ff.communicate(np.ascontiguousarray(video).tobytes())
            assert ff.returncode == 0
            mp4 = out / "video" / f"ep{e:04d}.mp4"  # + VP9 WebM for browsers without H.264 (open-source Chromium)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libvpx-vp9", "-b:v", "0",
                            "-crf", "34", "-row-mt", "1", "-deadline", "good", "-g", "10",
                            str(mp4.with_suffix(".webm"))], check=True)
        data = {"episode": int(e), "picked_as": picked.get(e, "extra"), "success": int(row.success),
                "T_s": float(row.policy_steps / STEP_HZ), "init_state": int(row.init_state), "layout": row.layout,
                "video": f"video/ep{e:04d}", "steps": int(row.policy_steps), "step_hz": STEP_HZ,
                "grid": 4, "frame_index": [int(f) for f in frames],
                "V_true": r(meta.value.values[rows_idx]), "V_r": r(ev), "V_e": r(de.V_e.values),
                "ps_r": r(X.p_success(probs), 3), "ps_e": r(de.ps_e.values, 3),
                "A_r": r(de.A_r.values), "A_e": r(de.A_e.values),
                "pos_r": de.positive_r.astype(int).tolist(), "pos_e": de.positive_e.astype(int).tolist(),
                "pos_o": de.positive_o.astype(int).tolist(), "term": de.term.astype(int).tolist(),
                "pots": [int(k[min(f, len(k) - 1)]) for f in frames],
                "probs": [r(p, 4) for p in probs]}
        if rules is not None:
            re_ = rules[rules.episode_index == e].sort_values("frame_index")
            assert (re_.frame_index.values == frames).all()
            assert (re_.resnet_R0.values == data["pos_r"]).all(), "rule R0 must equal the trained ResNet labels"
            data["rules"] = {k: re_[f"resnet_{k}"].astype(int).tolist() for k in index["rules"]}
            data["stall"] = re_.stall.astype(int).tolist()
        data["sets"] = {}
        for k, s_ in sets.items():
            if k.startswith("R"):
                data["sets"][k] = {"lab": data["rules"][k]}
            else:
                de_ = den[den.episode_index == e].sort_values("frame_index")
                assert (de_.frame_index.values == frames).all()
                data["sets"][k] = {"lab": de_[s_["col"]].astype(int).tolist(), "A": r(de_[s_["col"] + " A"].values)}
        (out / f"ep{e:04d}.json").write_text(json.dumps(data, separators=(",", ":")))
        index["episodes"].append({"episode": int(e), "picked_as": data["picked_as"], "success": data["success"],
                                  "group": group[e], "steps": data["steps"], "layout": data["layout"],
                                  "neg_resnet": round(1 - float(np.mean(data["pos_r"])), 3)})
        print(f"ep {e}: {len(frames)} frames", flush=True)
    (out / "episodes.json").write_text(json.dumps(index))


if __name__ == "__main__":
    main()
