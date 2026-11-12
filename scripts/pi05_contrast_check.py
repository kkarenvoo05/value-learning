"""Offline check of contrastive guidance on an AdaRMS-tagged run, before any simulator eval [ours, 2026-11-09].

Contrastive guidance [Advantage Guidance, Zhou & Levine 2026 (arXiv 2610.19783), Eq. 4]:
    v = v_none + w (v_pos - v_neg)
implemented as the velocity mix (c_pos, c_neg, c_none) = (w, -w, 1) in pi05_tag_adarms.TagPi0. The unconditional
v_none is the tag-dropout branch of the same network [ours; the paper trains a separate reference policy].

(1) Self-check: the mixes (1, 0, 0), (0, 1, 0), (0, 0, 1) must reproduce plain sampling with the +, - and no tag
    (same noise; bf16 tolerance as pi05_guidance.py). This tests that each velocity row and the mixing are right.
(2) Response per w on val frames (states never trained on), same noise draws z for every variant:
      shift(w)  = RMS( a_w(z) - a_none(z) )            over the executed 5 steps x 7 real dims
      noise     = RMS( a_none(z_j) - a_none(z_j+1) )   (the policy's own sampling spread)
    ratio = shift / noise, the same un-squared ratio as scripts/libero_tag_usage.py (square it for the paper's
    Fig. 3 / 6 metric). Also: gripper sign flips vs none, the largest raw action (blow-up check), and shift / w
    (linearity). Plain + vs - is reported as the reference (18355677: 13.6% of noise at final).

Usage (openpi env, one GPU; MODE=contrastcheck in scripts/pi05_tag_adarms.sbatch):
    .venv/bin/python <vl>/scripts/pi05_contrast_check.py --out <dir> --run <adarms run> [--weights 1 2 4 8]
Exit 1 if the self-check fails.
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

TOL = 2e-2  # relative max error, bf16 (as pi05_guidance.py / pi05_tag_adarms selftest (5))


def variant(tag):
    return lambda ep, t: fd.sample(ep, t, tag)


def ex(a):
    return a[:, :, :tu.EXEC, :tu.DIMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", default="final")
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--val-eps", type=int, default=24)
    ap.add_argument("--per-ep", type=int, default=6)
    ap.add_argument("--k", type=int, default=3, help="noise draws per frame")
    ap.add_argument("--mb", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert ta.is_adarms(a.run) or ar.is_residual(a.run), "contrastive guidance: AdaRMS or residual runs only"
    # residual runs [2026-11-11]: same mix interface; "none" = the frozen base pi0.5
    policy, _ = ar.load_residual(a.run, a.step) if ar.is_residual(a.run) else ta.load_adarms(a.run, a.step)
    frames = tu.val_frames(a.val_eps, a.per_ep)
    tu.log(f"{len(frames)} val frames, k {a.k}, run {a.run} step {a.step}")

    plain, plain_raw = tu.sample_variants(policy, frames, a.k, a.mb, {n: variant(t) for n, t in tu.TAGS.items()})
    res = {"setup": vars(a), "selfcheck": {}, "weights": {}}

    # (1) one-hot mixes == plain tagged sampling (the request carries no tag: the mix builds the conditions)
    ok = True
    for name, mix in (("pos", (1.0, 0.0, 0.0)), ("neg", (0.0, 1.0, 0.0)), ("none", (0.0, 0.0, 1.0))):
        ta.set_mix(policy, mix)
        got, _ = tu.sample_variants(policy, frames, 1, a.mb, {"m": variant(None)})
        ref = plain[name][:1]
        err = float(np.abs(got["m"] - ref).max() / (np.abs(ref).max() + 1e-8))
        res["selfcheck"][name] = err
        ok &= err < TOL
        tu.log(f"self-check mix {mix} vs plain {name}: rel max err {err:.2e}")
    res["selfcheck"]["ok"] = bool(ok)
    if not ok:
        (out / "results.json").write_text(json.dumps(res, indent=2))
        tu.log("SELF-CHECK FAILED")
        sys.exit(1)

    def stats(A, R, ref_name="none"):
        """A, R: normalised / raw actions (k, n, ah, ad) of one variant, vs plain ref_name with the same noise."""
        r = {}
        for space, P, X in (("norm", plain, A), ("raw", plain_raw, R)):
            base = ex(P[ref_name])
            noise = tu.rms(base[:-1] - base[1:], (2, 3)).mean()
            shift = tu.rms(ex(X) - base, (2, 3)).mean()
            r[space] = {"shift": float(shift), "noise": float(noise), "ratio": float(shift / noise)}
        g_x, g_b = np.sign(ex(R)[..., 6]), np.sign(ex(plain_raw[ref_name])[..., 6])
        r["gripper_flip"] = float((g_x != g_b).mean())
        r["max_abs_raw"] = float(np.abs(ex(R)).max())
        return r

    nb = ex(plain_raw["none"])
    res["noise_gripper_flip"] = float((np.sign(nb[:-1, ..., 6]) != np.sign(nb[1:, ..., 6])).mean())
    res["max_abs_raw_none"] = float(np.abs(nb).max())
    res["plain_pos_vs_neg"] = stats(plain["pos"], plain_raw["pos"], "neg")
    res["plain_pos_vs_none"] = stats(plain["pos"], plain_raw["pos"], "none")
    tu.log(f"plain + vs -: {json.dumps(res['plain_pos_vs_neg'])}")
    for w in a.weights:
        ta.set_mix(policy, ta.contrastive_mix(w))
        A, R = tu.sample_variants(policy, frames, a.k, a.mb, {"w": variant(None)})
        r = stats(A["w"], R["w"])
        r["ratio_per_w"] = r["norm"]["ratio"] / w
        res["weights"][str(w)] = r
        tu.log(f"w {w}: {json.dumps(r)}")
        (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\n w | shift/noise (norm) | per w | gripper flips vs none | max |raw|")
    pn = res["plain_pos_vs_neg"]
    print(f" plain + vs - | {pn['norm']['ratio']:.3f} | | {pn['gripper_flip']:.3f} | {pn['max_abs_raw']:.3f}")
    for w, r in res["weights"].items():
        print(f" {w} | {r['norm']['ratio']:.3f} | {r['ratio_per_w']:.3f} | {r['gripper_flip']:.3f} | {r['max_abs_raw']:.3f}")
    print(f" noise gripper flips {res['noise_gripper_flip']:.3f}, max |raw| none {res['max_abs_raw_none']:.3f}")
    tu.log(f"done -> {out}")


if __name__ == "__main__":
    main()
