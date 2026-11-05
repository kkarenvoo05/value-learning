"""Can the arm C advantage labels be made less noisy? Smoothing V, longer lookahead, margin / no-tag middle (2026-11-05).

Follows reports/2026-11-04-label-rules: under any threshold, ordinary-progress frames carry A noise (IQR ±0.04)
four times larger than a stall's cost (−0.019). Three remedies, each on the out-of-fold V of both value functions:

  smooth k    V averaged over a centred window of k value frames (k·4 steps; edges use the frames that exist)
              before differencing [ours; offline labels, so non-causal is allowed]
  n = N       lookahead N steps instead of 20 (vl.libero_advantages.advantage; time cost N/1040; frames whose
              lookahead passes the end use the outcome)
  margin      negative = A < p30 (as now), positive = A > p_hi, frames in between untagged ("no tag" in the prompt)
  combo       smooth 5 + n = 80 + margin p30/p70
Threshold for smooth / n variants = p30 of that variant's A (70% positive, as trained), so only the noise changes.

Noise metrics [ours]. The ResNet and the π₀.₅ value expert are independently trained, so what their advantages share
is treated as signal and what differs as noise:
  corr_A        Pearson r of A_ResNet vs A_expert over all frames (and interior frames: lookahead inside the episode)
  label_agree   share of frames where both models give the same tag (frames both tag, for margins)
  kappa         Cohen's kappa of the two models' tags (chance-corrected agreement)
  flips_per_s   ResNet tag changes between consecutive TAGGED value frames per second (5 value frames/s)
Signal checks (ResNet; as label-rules): negative share on stalls, before pot falls, before a successful first
placement, on ordinary progress (success, moving, no event, interior), lift = event negative share / base negative.

Usage: .venv/bin/python scripts/libero_label_denoise.py --out reports/2026-11-05-label-denoise
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.libero_advantages import NORM, V_FAIL, advantage  # noqa: E402

LAB = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/labels")
TABLE = LAB.parent
RULES = Path("reports/2026-11-04-label-rules/rule_labels.parquet")  # stall + event flags per frame
STEP_HZ = 20


def smooth(V: np.ndarray, ep: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return V
    out = np.empty_like(V)
    h = k // 2
    for e in np.unique(ep):
        i = np.flatnonzero(ep == e)
        v = V[i]
        c = np.concatenate([[0.0], np.cumsum(v)])
        lo = np.maximum(np.arange(len(v)) - h, 0)
        hi = np.minimum(np.arange(len(v)) + h + 1, len(v))
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def adv(d, V, n):
    t, T, ep = d.frame_index.to_numpy(float), d["T"].to_numpy(float), d.episode_index.to_numpy()
    vT = np.where(d.success.to_numpy() == 1, 0.0, V_FAIL)
    return advantage(V, t, T, vT, ep, n)


def tags(A, lo_pct=30, hi_pct=None):
    """1 positive, 0 negative, -1 untagged. hi_pct None = binary at lo_pct (as trained)."""
    lo = np.percentile(A, lo_pct)
    if hi_pct is None:
        return np.where(A > lo, 1, 0), (lo, lo)
    hi = np.percentile(A, hi_pct)
    return np.where(A > hi, 1, np.where(A <= lo, 0, -1)), (lo, hi)


def kappa(a, b):
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe))


def flips(L, ep):
    s = pd.DataFrame({"L": L, "ep": ep})
    s = s[s.L >= 0]
    per = s.groupby("ep").L.apply(lambda x: (np.diff(x.to_numpy()) != 0).sum() / (len(x) / (STEP_HZ / 4)))
    return float(per.mean())


def evaluate(d, Ar, Ae, Lr, Le, term):
    tag_r, both = Lr >= 0, (Lr >= 0) & (Le >= 0)
    inter = ~term
    s = d.success.to_numpy() == 1
    neg = lambda m: float((Lr[m & tag_r] == 0).mean()) if (m & tag_r).any() else float("nan")
    untag = lambda m: float((Lr[m] < 0).mean())
    base = neg(np.ones(len(d), bool))
    ordinary = s & ~d.stall.to_numpy() & (d.event.to_numpy() == "") & inter
    stall, drop = d.stall.to_numpy(), d.event.to_numpy() == "drop"
    place = (d.event.to_numpy() == "place_first") & s
    out = {"corr_A": float(np.corrcoef(Ar, Ae)[0, 1]), "corr_A_interior": float(np.corrcoef(Ar[inter], Ae[inter])[0, 1]),
           "label_agree": float((Lr[both] == Le[both]).mean()), "kappa": kappa(Lr[both], Le[both]),
           "flips_per_s": flips(Lr, d.episode_index.to_numpy()), "tagged": float(tag_r.mean()),
           "base_neg": base, "neg_stall": neg(stall), "neg_drop": neg(drop), "neg_place_success": neg(place),
           "neg_ordinary": neg(ordinary), "untagged_stall": untag(stall), "untagged_ordinary": untag(ordinary),
           "untagged_drop": untag(drop), "terminal_share": float(term.mean())}
    out["lift_stall"], out["lift_drop"] = out["neg_stall"] / base, out["neg_drop"] / base
    out["lift_place_success"] = out["neg_place_success"] / base
    # ordinary-progress A spread (ResNet, interior): the noise the rule has to cut through
    q = np.quantile(Ar[ordinary], [0.25, 0.5, 0.75])
    out["ordinary_A_median"], out["ordinary_A_iqr"] = float(q[1]), float(q[2] - q[0])
    out["stall_A_median"] = float(np.median(Ar[stall & inter]))
    out["stall_minus_ordinary_over_iqr"] = (out["stall_A_median"] - out["ordinary_A_median"]) / out["ordinary_A_iqr"]
    return out


VARIANTS = (
    [("current (n=20)", dict(k=1, n=20))]
    + [(f"smooth {k} ({k * 4} steps)", dict(k=k, n=20)) for k in (3, 5, 9)]
    + [(f"n={n} ({n / STEP_HZ:g} s)", dict(k=1, n=n)) for n in (40, 80, 160)]
    + [("margin p30/p50 (20% untagged)", dict(k=1, n=20, hi=50)), ("margin p30/p70 (40% untagged)", dict(k=1, n=20, hi=70))]
    + [("combo: smooth 5 + n=80 + p30/p70", dict(k=5, n=80, hi=70))]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(TABLE / "episodes.csv").set_index("episode_index")
    key = ["episode_index", "frame_index"]
    r = pd.read_parquet(LAB / "oof_resnet_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    e = pd.read_parquet(LAB / "oof_expert_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    assert (r[key].values == e[key].values).all()
    f = pd.read_parquet(RULES)[key + ["stall", "event"]].sort_values(key).reset_index(drop=True)
    assert (f[key].values == r[key].values).all()
    d = r[key + ["success", "init_state"]].copy()
    d["T"] = d.episode_index.map(eps.policy_steps - 1).astype(float)
    d["stall"], d["event"] = f.stall.values, f.event.values
    Vr, Ve, ep = r.V.to_numpy(float), e.V.to_numpy(float), d.episode_index.to_numpy()
    A0, _ = adv(d, Vr, 20)
    assert np.allclose(A0, r.A.to_numpy()), "n=20 unsmoothed must reproduce the trained ResNet advantages"
    rows, perframe = [], d[key].copy()
    for name, v in VARIANTS:
        Ar, term = adv(d, smooth(Vr, ep, v["k"]), v["n"])
        Ae, _ = adv(d, smooth(Ve, ep, v["k"]), v["n"])
        Lr, thr = tags(Ar, 30, v.get("hi"))
        Le, _ = tags(Ae, 30, v.get("hi"))
        if name.startswith("current"):
            assert (Lr == r.positive.to_numpy()).all()
        res = evaluate(d, Ar, Ae, Lr, Le, term)
        rows.append({"variant": name, "thr_lo": thr[0], "thr_hi": thr[1], **res})
        perframe[name] = Lr.astype(np.int8)
        perframe[name + " A"] = Ar.astype(np.float32)
        print(name, json.dumps({k: round(x, 3) for k, x in res.items()}), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "denoise_table.csv", index=False)
    perframe.to_parquet(out / "denoise_labels.parquet")
    (out / "summary.json").write_text(json.dumps({"variants": rows}, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["variant", "corr_A_interior", "kappa", "flips_per_s", "tagged", "neg_ordinary", "neg_stall", "lift_stall",
            "lift_drop", "lift_place_success", "ordinary_A_iqr", "stall_minus_ordinary_over_iqr"]
    print(tab[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
