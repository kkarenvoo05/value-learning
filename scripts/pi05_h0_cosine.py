"""Direction check for the AG saturation question [ours, 2026-11-18]: is the data-behaviour residual (the learned
no-tag branch h(0), or the label-weighted mean residual p h(+) + (1 - p) h(-)) aligned with the guidance direction
h(+) - h(-)? If it is, uncond_mix / anchored_mix add guidance on top of a term that already contains some of it
(double counting), which would explain why they peak at a lower w and lower success than the standard mix.

On the contrast check's val frames, with the SAME noise draws z for every variant (libero_tag_usage.sample_variants),
each direction is an action difference vs base pi0.5 (mix (0, 0, 1)):
    G1    = a[(1, -1, 1)]          - a_base   one unit of contrastive guidance (w 1)
    G4    = a[(4, -4, 1)] / 4      - ...      (as a direction: linearity check vs G1)
    pos   = a[(1, 0, 0)]           - a_base   h(+)
    neg   = a[(0, 1, 0)]           - a_base   h(-)
    mean  = a[(p, 1 - p, 0)]       - a_base   label-weighted mean residual (p = vb40 positive share 0.4)
    h0    = a[(0, 0, 0, 1)]        - a_base   learned no-tag branch (runs trained with --uncond-branch only)
    noise = a_base(z_j) - a_base(z_j+1)       null: the policy's own sampling spread
Cosines over the executed 5 steps x 7 dims of the normalised actions ("all7") and the 6 arm dims ("arm6"):
    cos       mean over (frame, z) of the per-sample cosine with G1, episode-bootstrap 95% CI
    cos_frame cosine of the per-frame directions averaged over z (the systematic part), same CI
    proj      <d, G1> / <G1, G1> pooled: d's component along one unit of guidance (1.0 = one w unit)
    size      RMS(d) / RMS(noise)
Random 35-d directions give |cos| ~ 0.17; noise vs G1 is the empirical null.

Usage (openpi env, one GPU; scripts/pi05_h0_cosine.sbatch loops over runs, one process each):
    .venv/bin/python <vl>/scripts/pi05_h0_cosine.py --run <residual run> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import libero_tag_usage as tu  # noqa: E402
import pi05_ag_residual as ar  # noqa: E402
import pi05_tag_adarms as ta  # noqa: E402
from vl import libero_finetune_data as fd  # noqa: E402


def cos_rows(x, y):
    """Cosine along the last axis."""
    return (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-12)


def boot(per_frame, ep_ids, n=2000, seed=0):
    """Episode bootstrap 95% CI of the frame mean."""
    eps = np.unique(ep_ids)
    by = {e: per_frame[ep_ids == e] for e in eps}
    rng = np.random.default_rng(seed)
    m = [np.concatenate([by[e] for e in rng.choice(eps, len(eps))]).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", default="final")
    ap.add_argument("--val-eps", type=int, default=24)
    ap.add_argument("--per-ep", type=int, default=6)
    ap.add_argument("--k", type=int, default=3, help="noise draws per frame")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--p", type=float, default=0.4, help="positive share for the mean residual")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert ar.is_residual(a.run), "residual runs only"
    policy, _ = ar.load_residual(a.run, a.step)
    frames = tu.val_frames(a.val_eps, a.per_ep)
    ep_ids = np.array([f[0] for f in frames])
    uncond = bool(policy._model.uncond)
    tu.log(f"{len(frames)} val frames, k {a.k}, run {a.run} step {a.step}, uncond branch {uncond}")

    mixes = {"base": (0.0, 0.0, 1.0), "G1": ta.contrastive_mix(1.0), "G4": ta.contrastive_mix(4.0),
             "pos": (1.0, 0.0, 0.0), "neg": (0.0, 1.0, 0.0), "mean": (a.p, 1.0 - a.p, 0.0)}
    if uncond:
        mixes["h0"] = ar.uncond_mix(0.0)  # (0, 0, 0, 1) = v_ref + h(0)
    A = {}
    for name, mix in mixes.items():
        ta.set_mix(policy, mix)
        got, _ = tu.sample_variants(policy, frames, a.k, a.mb, {"m": lambda ep, t: fd.sample(ep, t, None)})
        A[name] = got["m"][:, :, :tu.EXEC, :tu.DIMS]  # (k, n, 5, 7) normalised
        tu.log(f"sampled {name} {mix}")
    base = A["base"]
    D = {n: A[n] - base for n in mixes if n != "base"}
    D["G4"] = D["G4"] / 4.0
    D["noise"] = base[:-1] - base[1:]

    res = {"setup": vars(a) | {"uncond": uncond, "n_frames": len(frames)}, "spaces": {}}
    pairs = [(x, "G1") for x in ("h0", "mean", "neg", "pos", "G4", "noise") if x in D]
    pairs += [(x, y) for x, y in (("h0", "mean"), ("h0", "neg")) if x in D]
    for space, dims in (("all7", 7), ("arm6", 6)):
        flat = {n: d[..., :dims].reshape(d.shape[0], d.shape[1], -1) for n, d in D.items()}
        noise_rms = tu.rms(flat["noise"], (0, 1, 2))
        r = {"size_vs_noise": {n: float(tu.rms(f, (0, 1, 2)) / noise_rms) for n, f in flat.items()}}
        for x, y in pairs:
            X, Y = flat[x], flat[y][:flat[x].shape[0]]
            c = cos_rows(X, Y).mean(0)  # (n,) per frame, mean over z
            cf = cos_rows(X.mean(0), Y.mean(0))  # (n,) per-frame mean directions
            r[f"{x}_vs_{y}"] = {"cos": float(c.mean()), "cos_ci": boot(c, ep_ids),
                                "cos_frame": float(cf.mean()), "cos_frame_ci": boot(cf, ep_ids),
                                "proj": float((X * Y).sum() / (Y * Y).sum())}
        res["spaces"][space] = r
    (out / "results.json").write_text(json.dumps(res, indent=2))
    # same frames (val_frames seed) and noise (PRNGKey 123) for every run: directions are comparable ACROSS runs,
    # e.g. cos(G1 of round 0, G1 of round 7) = did later rounds learn a new guidance direction?
    np.savez_compressed(out / "directions.npz", ep_ids=ep_ids, **D)
    for space, r in res["spaces"].items():
        print(f"\n[{space}] size vs noise: " + ", ".join(f"{n} {v:.2f}" for n, v in r["size_vs_noise"].items()))
        print(" pair | cos [95%] | cos of frame means [95%] | proj")
        for k_, v in r.items():
            if k_ == "size_vs_noise":
                continue
            print(f" {k_} | {v['cos']:+.3f} [{v['cos_ci'][0]:+.3f}, {v['cos_ci'][1]:+.3f}] | "
                  f"{v['cos_frame']:+.3f} [{v['cos_frame_ci'][0]:+.3f}, {v['cos_frame_ci'][1]:+.3f}] | {v['proj']:+.3f}")
    tu.log(f"done -> {out}")


if __name__ == "__main__":
    main()
