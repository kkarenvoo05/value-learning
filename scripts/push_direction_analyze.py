"""Score push vectors from scripts/pi05_push_direction.py: do round-1 pushes point the same way as round-0's? [2026-11-21]

Input: <root>/<name>/push.npz for every policy (same val frames, same noise z, same frozen base a_none).
Per policy: push P = a_w - a_none (k, n, 5, 7). Normalised action space (each dim scaled as the model sees it) for
directions, raw units for physical reading (xyz in cm: summed over the 5 executed steps x 0.75 cm per unit, the G0
calibration 0.0075 m per unit of summed action; open loop, approximate).
Per pair (a, b), per (frame, z), on the arm-motion push (xyz + rotation: 5 steps x 6 = 30-d; the main read) and also
on all 35 dims and on the gripper alone (pair["all"], pair["grip"]). Why split: in normalised units the gripper holds
~2/3 of the push energy, almost all of it on the few frames near grasp / release, so it swamps a 35-d score:
  cos      = <P_a, P_b> / (|P_a| |P_b|)                    (same direction = 1, unrelated ~ 0, opposite = -1)
  beta     = sum <P_a, P_b> / sum |P_b|^2                  (how much of b's push a contains: 1 = all of it)
  cos_w    = sum <P_a, P_b> / sqrt(sum |P_a|^2 sum |P_b|^2) (one cosine over all frames: big pushes count more; same
             weighting as beta, so the two agree in sign; cos above weights every frame equally)
  CIs: bootstrap over val episodes (frames of one episode share its trajectory).
Sanity: a_none must agree across policies (same frames + z + frozen base); ratio = RMS shift / noise ~ the target.
Output: <root>/analysis.json (everything the page / report needs), printed tables.
Usage: .venv/bin/python scripts/push_direction_analyze.py --root <dir> [--pairs a:b ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CM = 0.75  # cm per unit of summed raw action over one 5-step chunk (G0 calibration, 18329176)
DIMS = ["x", "y", "z", "rx", "ry", "rz", "grip"]
PHASES = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 520)]

# default pairs: (a, b, kind). kind: "r1 vs r0" = the question; "seed" = two retrains on the same data (how alike two
# pushes can be at best); "r2 vs r1" / "over" = context
DEFAULT_PAIRS = [
    ("900_r1", "900_r0", "r1 vs r0"), ("b100_r1_s0", "b100_r0", "r1 vs r0"), ("b100_r1_s1", "b100_r0", "r1 vs r0"),
    ("b100cv_r1", "b100cv_r0", "r1 vs r0"), ("b50cv_r1_s0", "b50cv_r0", "r1 vs r0"),
    ("b50cv_r1_s1", "b50cv_r0", "r1 vs r0"), ("mt_r1_s0", "mt_r0", "r1 vs r0"), ("mt_r1_s1", "mt_r0", "r1 vs r0"),
    ("mtos_r1", "mt_r0", "r1 vs r0"), ("mtos77_r1", "mt_r0", "r1 vs r0"),
    ("b100_r1_s1", "b100_r1_s0", "seed"), ("b50cv_r1_s1", "b50cv_r1_s0", "seed"), ("mt_r1_s1", "mt_r1_s0", "seed"),
    ("900_r2", "900_r1", "r2 vs r1"), ("b100_r2", "b100_r1_s0", "r2 vs r1"),
    ("b100_r0", "900_r0", "r0 vs r0"), ("b50cv_r0", "900_r0", "r0 vs r0"), ("mt_r0", "900_r0", "r0 vs r0"),
]


def boot(stat, eps, n=2000, seed=0):
    """95% CI of stat(idx) over a bootstrap of episodes (idx = frame indices of the resampled episodes)."""
    rng = np.random.default_rng(seed)
    ue = np.unique(eps)
    by = {e: np.flatnonzero(eps == e) for e in ue}
    vals = [stat(np.concatenate([by[e] for e in rng.choice(ue, len(ue))])) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pairs", nargs="*", default=None, help="a:b[:kind] (default DEFAULT_PAIRS)")
    ap.add_argument("--arrows", default="b50cv_r0,b50cv_r1_s0,b50cv_r1_s1,b100_r0,b100_r1_s0,b100_r1_s1,900_r0,900_r1,mt_r0,mt_r1_s0,mt_r1_s1,b100cv_r0,b100cv_r1,mtos_r1",
                    help="policies whose per-frame xyz pushes go into the json (for the arrow plots)")
    a = ap.parse_args()
    root = Path(a.root)
    D = {p.parent.name: dict(np.load(p)) for p in sorted(root.glob("*/push.npz"))}
    assert D, f"no push.npz under {root}"
    names = list(D)
    ref = D[names[0]]
    for nm, d in D.items():  # same frames and frozen base everywhere
        assert (d["ep"] == ref["ep"]).all() and (d["t"] == ref["t"]).all(), f"{nm}: different frames"
    none_diff = max(float(np.abs(d["raw_none"] - ref["raw_none"]).max()) for d in D.values())
    eps, t = ref["ep"], ref["t"]
    rms = lambda x, ax: np.sqrt((x ** 2).mean(ax))
    out = {"policies": {}, "pairs": [], "none_max_abs_diff": none_diff, "frames": int(len(t)),
           "episodes": int(len(np.unique(eps))), "k": int(ref["raw_w"].shape[0]), "dims": DIMS, "cm_per_unit": CM}
    P = {}
    for nm, d in D.items():
        pn = d["norm_w"] - d["norm_none"]; pr = d["raw_w"] - d["raw_none"]
        P[nm] = pn  # (k, n, 5, 7)
        noise = rms(d["norm_none"][:-1] - d["norm_none"][1:], (2, 3)).mean()
        shift = rms(pn, (2, 3)).mean()
        xyz = pr[..., :3].sum(2).mean(0) * CM  # (n, 3) cm, mean over z
        mag = np.linalg.norm(xyz, axis=1)
        phase = {f"{lo}-{hi}": float(mag[(t >= lo) & (t < hi)].mean()) for lo, hi in PHASES if ((t >= lo) & (t < hi)).any()}
        out["policies"][nm] = {
            "w": float(d["w"]), "ratio": float(shift / noise), "shift": float(shift), "noise": float(noise),
            "xyz_cm_mean": xyz.mean(0).round(4).tolist(),       # signed mean: a consistent direction survives
            "xyz_cm_abs": float(mag.mean()),                     # typical size
            "dim_mean_raw": pr.mean((0, 1, 2)).round(5).tolist(), "dim_rms_raw": rms(pr, (0, 1, 2)).round(5).tolist(),
            "grip_push_raw": float(pr[..., 6].mean()), "phase_xyz_cm": phase,
            # where the push energy sits (normalised units): gripper share, and share held by the 10% largest frames
            "grip_share": float((pn[..., 6] ** 2).sum() / (pn ** 2).sum()),
            "top10pct_share": float(np.sort(e := (pn ** 2).sum((0, 2, 3)))[::-1][:max(1, len(e) // 10)].sum() / e.sum())}
        if nm in a.arrows.split(","):
            out["policies"][nm]["arrows"] = {"xyz_cm": xyz.round(3).tolist(), "eef": d["state"][:, :3].round(4).tolist()}
    out["frames_meta"] = {"ep": eps.tolist(), "t": t.tolist()}
    pairs = [tuple(p.split(":")) + (("custom",) if p.count(":") == 1 else ()) for p in a.pairs] if a.pairs else DEFAULT_PAIRS
    for pa, pb, kind in pairs:
        if pa not in P or pb not in P:
            print(f"skip {pa} vs {pb} (missing)"); continue
        idx = np.arange(len(t))

        def score(sl, per_frame=True):
            A, B = P[pa][..., sl].reshape(*P[pa].shape[:2], -1), P[pb][..., sl].reshape(*P[pb].shape[:2], -1)
            dot = (A * B).sum(-1); na, nb = (A ** 2).sum(-1), (B ** 2).sum(-1)
            beta = lambda i: float(dot[:, i].sum() / (nb[:, i].sum() + 1e-12))
            cosw = lambda i: float(dot[:, i].sum() / np.sqrt(na[:, i].sum() * nb[:, i].sum() + 1e-12))
            g = {"beta": beta(idx), "beta_ci": boot(beta, eps), "cos_w": cosw(idx), "cos_w_ci": boot(cosw, eps)}
            if per_frame:
                cos_f = (dot / np.sqrt(na * nb + 1e-12)).mean(0)          # per frame, mean over z
                cos = lambda i: float(cos_f[i].mean())
                g.update(cos=cos(idx), cos_ci=boot(cos, eps), cos_hist=np.histogram(cos_f, bins=20, range=(-1, 1))[0].tolist(),
                         cos_phase={f"{lo}-{hi}": float(cos_f[(t >= lo) & (t < hi)].mean()) for lo, hi in PHASES
                                    if ((t >= lo) & (t < hi)).any()})
            return g

        xa = (D[pa]["raw_w"] - D[pa]["raw_none"])[..., :3].sum(2).mean(0)
        xb = (D[pb]["raw_w"] - D[pb]["raw_none"])[..., :3].sum(2).mean(0)
        cos_xyz = float(((xa * xb).sum(1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1) + 1e-12)).mean())
        # top level = arm motion (xyz + rotation, 30-d), the main read; "all" adds the gripper; "grip" = gripper alone
        # (1-d: a per-frame cosine there is only the sign of two tiny numbers, so weighted measures only)
        r = {"a": pa, "b": pb, "kind": kind, **score(slice(0, 6)), "cos_xyz": cos_xyz,
             "all": score(slice(0, 7)), "grip": score(slice(6, 7), per_frame=False)}
        out["pairs"].append(r)
    (root / "analysis.json").write_text(json.dumps(out, indent=1))
    print(f"{len(D)} policies, {out['frames']} frames ({out['episodes']} val eps) x k {out['k']}; "
          f"a_none max |diff| across policies {none_diff:.2e}")
    print("\npolicy | w | shift/noise | grip share | top-10% frames share | |xyz push| cm | mean xyz push (cm)")
    for nm, r in out["policies"].items():
        print(f"{nm} | {r['w']:.2f} | {r['ratio']:.2f} | {r['grip_share']:.2f} | {r['top10pct_share']:.2f} | {r['xyz_cm_abs']:.2f} | {r['xyz_cm_mean']}")
    print("\narm motion (30-d) | gripper | all 35-d\npair | kind | cos [CI] | cos_w [CI] | beta [CI] | cos xyz || grip cos_w, beta || all cos_w, beta")
    for r in out["pairs"]:
        print(f"{r['a']} vs {r['b']} | {r['kind']} | {r['cos']:.2f} [{r['cos_ci'][0]:.2f}, {r['cos_ci'][1]:.2f}] | "
              f"{r['cos_w']:.2f} [{r['cos_w_ci'][0]:.2f}, {r['cos_w_ci'][1]:.2f}] | {r['beta']:.2f} [{r['beta_ci'][0]:.2f}, {r['beta_ci'][1]:.2f}] | {r['cos_xyz']:.2f} || "
              f"{r['grip']['cos_w']:.2f}, {r['grip']['beta']:.2f} || {r['all']['cos_w']:.2f}, {r['all']['beta']:.2f}")


if __name__ == "__main__":
    main()
