"""Push vectors of residual AG policies on shared val frames, for direction comparisons [ours, 2026-11-21].

Question (reports/2026-11-21-push-direction; Karen: "write the push-direction check"): does a round-1 policy push in
the same direction as its round-0 policy at the same states? If r1's push ~ r0's push, the guided data (collected
WITH r0's push) re-teaches that push and the r1 policy applies it a second time on top of base.

For one residual run (pi05_ag_residual) at its eval push (w from its contrast check at FT_SHIFT_TARGET, the same
least-squares slope as scripts/pi05_tag_adarms.sbatch shift_to_w), on the table's val frames (states never trained
on; every moka table has the same 120 base val episodes), with fixed noise draws z:
    push(frame, z) = a_w(z) - a_none(z)     (executed 5 steps x 7 dims; raw and normalised)
a_none is the frozen base pi0.5, identical for every run, so pushes from different runs are directly comparable
(the analysis checks a_none agrees across runs). Frames, noise (PRNGKey 123 in libero_tag_usage.sample_variants)
and val-episode choice are deterministic, so every run sees the same (frame, z).
Output <out>/push.npz: raw_w, raw_none, norm_w, norm_none (k, n, 5, 7); ep, t, state (n, 8); w; + meta.json.
Scored and plotted by scripts/push_direction_analyze.py (CPU).

Usage (openpi env, one GPU; MODE=pushdir in scripts/pi05_tag_adarms.sbatch):
    .venv/bin/python <vl>/scripts/pi05_push_direction.py --out <dir> --run <residual run> --check <check dir>
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


def w_for_target(check: Path, target: float) -> float:
    """w = target / slope, slope = least squares through 0 of shift/noise (norm) vs w (as the sbatch shift_to_w)."""
    r = json.loads((check / "results.json").read_text())["weights"]
    p = [(float(w), v["norm"]["ratio"]) for w, v in r.items() if float(w) > 0]
    k = sum(w * x for w, x in p) / sum(w * w for w, x in p)
    return round(target / k, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--check", required=True, help="the run's contrast-check dir (results.json) -> eval w")
    ap.add_argument("--target", type=float, default=0.99, help="push target (shift / noise), as the evals")
    ap.add_argument("--step", default="final")
    ap.add_argument("--table", default="/iris/u/ankile/karenvo/libero/tables/moka_agr_r7")
    ap.add_argument("--val-eps", type=int, default=24)
    ap.add_argument("--per-ep", type=int, default=8)
    ap.add_argument("--k", type=int, default=2, help="noise draws per frame")
    ap.add_argument("--mb", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert ar.is_residual(a.run), "residual AG runs only"
    w = w_for_target(Path(a.check), a.target)
    fd.TABLE = Path(a.table)
    frames = tu.val_frames(a.val_eps, a.per_ep)
    tu.log(f"{len(frames)} val frames, k {a.k}, run {a.run}, w {w} (target {a.target})")
    policy, _ = ar.load_residual(a.run, a.step)
    variant = lambda ep, t: fd.sample(ep, t, None)
    ta.set_mix(policy, (0.0, 0.0, 1.0))
    n_none, r_none = tu.sample_variants(policy, frames, a.k, a.mb, {"m": variant})
    ta.set_mix(policy, ta.contrastive_mix(w))
    n_w, r_w = tu.sample_variants(policy, frames, a.k, a.mb, {"m": variant})
    ex = lambda x: np.asarray(x[:, :, :tu.EXEC, :tu.DIMS], np.float32)
    np.savez_compressed(out / "push.npz", raw_w=ex(r_w["m"]), raw_none=ex(r_none["m"]), norm_w=ex(n_w["m"]),
                        norm_none=ex(n_none["m"]), ep=np.array([e for e, _, _ in frames]),
                        t=np.array([t for _, t, _ in frames]),
                        state=np.stack([np.asarray(ep["state"][t], np.float32) for _, t, ep in frames]),
                        n_steps=np.array([len(ep["actions"]) for _, _, ep in frames]), w=np.array(w))
    push = ex(r_w["m"]) - ex(r_none["m"])
    meta = {**vars(a), "w": w, "frames": len(frames),
            "push_rms_raw": float(np.sqrt((push ** 2).mean())),
            "noise_rms_raw": float(np.sqrt(((ex(r_none["m"])[:-1] - ex(r_none["m"])[1:]) ** 2).mean())) if a.k > 1 else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    tu.log(f"done -> {out}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
