"""Localized tag-usage test (reports/2026-11-06-tag-hypotheses/, E4): does a tagged policy respond to its tag at the
frames where the tag should matter, even if it ignores it on average?

The earlier test (scripts/libero_tag_usage.py, action test) drew random val frames, where the policy's next actions
are mostly fixed by the scene. In the data, + and − frames differ mostly in gripper timing
(reports/2026-11-06-tag-deep-dive/), and every failure is a timeout with stalls and gripper chatter. So the same test
is repeated on key frames (src/vl/keyframes.py): grip (the recorded gripper switches inside the executed window),
stall (end effector still for 20 steps), moving (the rest, comparison).

Per category and model, with the same noise draws for + / − / no tag (libero_tag_usage.sample_variants):
  tag / noise           RMS(a(+) − a(−)) / RMS(a(+, z1) − a(+, z2)), executed 5 steps × 7 dims, raw units
  gripper flips         share of executed steps where + and − disagree on open / close, vs two noise draws
  earlier switch (grip) mean of the signed gripper difference, oriented so > 0 = the + tag switches earlier
Every quantity is compared with a never-tagged arm A (the floor: what prompt text alone does). Per-frame values are
saved (frames.parquet) so later analyses need no GPU.

Usage (openpi env, one GPU; scripts/libero_tag_keyframes.sbatch):
    .venv/bin/python <vl>/scripts/libero_tag_keyframes.py --out <dir> [--val-eps 60 --k 4]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libero_tag_usage import DIMS, EXEC, TAGS, boot_ci, boot_ratio_ci, log, rms, sample_variants  # noqa: E402
from pi05_policy_finetune import load_finetuned  # noqa: E402
from vl import keyframes as K  # noqa: E402
from vl import libero_finetune_data as fd  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
MODELS = {  # name -> (run dir, checkpoint)
    "recap_vb40": (RUNS / "policy_armC_recap_vb40_s0_resume4000_18338571", "final"),
    "recap_scoped_vb40": (RUNS / "policy_armC_recap_scoped_vb40_s0_18338529", "final"),
    "A_frozensiglip": (RUNS / "policy_armA_frozensiglip_s0_18310771", "final"),  # floor, same recipe as RECAP
    "A_success_only": (RUNS / "policy_armA_s0_18288138", "final"),             # floor used by the deep dive
}


def select_frames(n_eps: int, seed: int = 1):
    eps = pd.read_csv(fd.TABLE / "episodes.csv")
    rows = eps[eps.split == "val"].sort_values("episode_index")
    rows = rows.iloc[np.random.default_rng(seed).permutation(len(rows))[:n_eps]]
    rng, out = np.random.default_rng(seed + 1), []
    for r in rows.itertuples():
        ep = fd.load_episode(r.raw, None, int(r.episode_index))
        for t, cat, d in K.key_frames(ep["state"], ep["actions"], rng):
            out.append(dict(ep=int(r.episode_index), t=int(t), cat=cat, grip_dir=int(d), success=int(r.success),
                            _ep=ep))
    return out


def measure(policy, frames, k, mb):
    trip = [(f["ep"], f["t"], f["_ep"]) for f in frames]
    variants = {name: (lambda ep, t, tag=tag: fd.sample(ep, t, tag)) for name, tag in TAGS.items()}
    _, raw = sample_variants(policy, trip, k, mb, variants)
    ex = {nm: raw[nm][:, :, :EXEC, :DIMS] for nm in TAGS}          # (k, n, EXEC, DIMS)
    g = {nm: ex[nm][..., 6] for nm in TAGS}
    d = pd.DataFrame([{kk: v for kk, v in f.items() if kk != "_ep"} for f in frames])
    d["tag"] = rms(ex["pos"] - ex["neg"], (2, 3)).mean(0)
    d["tag_none"] = rms(ex["pos"] - ex["none"], (2, 3)).mean(0)
    d["noise"] = rms(ex["pos"][:-1] - ex["pos"][1:], (2, 3)).mean(0)
    d["grip_flip_tag"] = (np.sign(g["pos"]) != np.sign(g["neg"])).mean((0, 2))
    d["grip_flip_noise"] = (np.sign(g["pos"][:-1]) != np.sign(g["pos"][1:])).mean((0, 2))
    d["earlier"] = K.earlier_switch(g["pos"], g["neg"], d.grip_dir.to_numpy()[None, :]).mean(0)
    d["earlier_noise"] = K.earlier_switch(g["pos"][:-1], g["pos"][1:], d.grip_dir.to_numpy()[None, :]).mean(0)
    return d


def summarize(d: pd.DataFrame) -> dict:
    res = {}
    for cat, c in list(d.groupby("cat")) + [("all", d)]:
        e = c.groupby("ep")
        r = {"frames": int(len(c)), "episodes": int(c.ep.nunique()),
             "tag": float(c.tag.mean()), "noise": float(c.noise.mean()),
             "ratio_tag_over_noise": float(c.tag.mean() / c.noise.mean()),
             "ratio_ci95": boot_ratio_ci(e.tag.sum(), e.noise.sum()),
             "grip_flip_tag": float(c.grip_flip_tag.mean()), "grip_flip_noise": float(c.grip_flip_noise.mean()),
             "grip_flip_ratio": float(c.grip_flip_tag.mean() / max(c.grip_flip_noise.mean(), 1e-9))}
        if cat == "grip":
            r["earlier_switch"] = float(c.earlier.mean())
            r["earlier_switch_ci95"] = boot_ci(e.earlier.mean())
            r["earlier_noise_abs"] = float(c.earlier_noise.abs().mean())
            for o, s in c.groupby("success"):
                r[f"earlier_switch_success{o}"] = float(s.earlier.mean())
        res[cat] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-eps", type=int, default=60)
    ap.add_argument("--k", type=int, default=4, help="noise draws per frame")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--max-frames", type=int, default=0, help="> 0: first n frames only (smoke test)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = select_frames(a.val_eps)
    if a.max_frames:
        frames = frames[:a.max_frames]
    cats = pd.Series([f["cat"] for f in frames]).value_counts().to_dict()
    log(f"{len(frames)} key frames from {a.val_eps} val episodes: {cats}")
    res, per = {"setup": vars(a), "categories": cats, "models": {}}, []
    for name in a.models:
        run, step = MODELS[name]
        log(f"model {name}: {run} ({step})")
        policy, _ = load_finetuned(run, step, seed=0)
        d = measure(policy, frames, a.k, a.mb)
        d["model"] = name
        per.append(d)
        res["models"][name] = summarize(d)
        log(f"  {json.dumps({c: {k: round(v, 4) for k, v in r.items() if isinstance(v, float)} for c, r in res['models'][name].items()})}")
        (out / "results.json").write_text(json.dumps(res, indent=2))
        pd.concat(per).to_parquet(out / "frames.parquet")
        del policy
        gc.collect()
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
