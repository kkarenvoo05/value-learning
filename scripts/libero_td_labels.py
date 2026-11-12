"""Item 3 (2026-11-07, Karen: build): better advantage labels from a fitted n-step TD critic and V ensembles.

    .venv/bin/python scripts/libero_td_labels.py targets   # -> labels/td_targets_n20/targets.parquet (CPU, seconds)
    .venv/bin/python scripts/libero_td_labels.py labels    # after the 5 TD OOF folds: candidate label sets

`targets`: y = vl.libero_td.td_targets(V0) with V0 = the OOF ResNet V of `oof_resnet_n20_p30` (the V behind the
current labels), n = 20 frames. The 5 TD fold runs (vl.libero_resnet --oof-fold k --targets <this>, same recipe and
folds as resnet18_oof_f{k}_s0_*) then give V1 = V_td.

`labels`: value-binned 40% labels (vl.libero_td.labels_from_v, the rule of `resnet_valuebinned_p40`) from
  td_vb40        V_td alone                                   [Dissecting: n-step TD critic]
  ens_vb40       mean of the OOF ResNet V and OOF pi05 value-expert V    [ours: ensemble of the two critics we have]
  td_ens_vb40    mean of V_td and the OOF value-expert V
and checks that the same code on V0 reproduces `resnet_valuebinned_p40` exactly. Which set the policy trains on is
decided offline by scripts/libero_tag_ideal_shift.py --candidates ... --gate (action information).
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl import libero_td as td  # noqa: E402

TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
LAB = TABLE / "labels"
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
TD_PATTERN = "resnet18_tdoof_f{k}_s0_*"
N = 20


def base() -> pd.DataFrame:
    """OOF ResNet labels (V0) + T, sorted; every train frame of the label grid."""
    r = pd.read_parquet(LAB / "oof_resnet_n20_p30" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    eps = pd.read_csv(TABLE / "episodes.csv").set_index("episode_index")
    r["T"] = r.episode_index.map(eps.policy_steps - 1).to_numpy()
    return r


def oof_v(pattern: str, col: str = "ev_resnet") -> pd.DataFrame:
    parts = []
    for k in range(5):
        hits = sorted(d for d in glob.glob(str(RUNS / pattern.format(k=k))) if (Path(d) / "oof_predictions.parquet").exists())
        assert len(hits) == 1, (k, hits)
        p = pd.read_parquet(Path(hits[0]) / "oof_predictions.parquet")
        assert (p.fold == k).all()
        parts.append(p[td.KEY + [col]].rename(columns={col: "V"}).assign(run=Path(hits[0]).name))
    return pd.concat(parts, ignore_index=True)


def write(name: str, lab: pd.DataFrame, info: dict):
    out = LAB / name
    out.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(out / "labels.parquet")
    (out / "summary.json").write_text(json.dumps({**info, "frames": len(lab), "positive": float(lab.positive.mean())},
                                                 indent=2))
    print(f"{name}: {len(lab)} frames, positive {lab.positive.mean():.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["targets", "labels", "oof", "hindsight"])
    ap.add_argument("--table", default=None, help="oof: episode table of the OOF runs")
    ap.add_argument("--pattern", default=None, help="oof: run-dir glob with {k}, e.g. resnet18_r2oof_f{k}_s0_*")
    ap.add_argument("--name", default=None, help="oof: label set name under <table>/labels/")
    ap.add_argument("--rule", choices=["vb", "thr"], default="vb",
                    help="oof: vb = value-binned (ours, Dissecting Eq. 4); thr = one threshold on A (RECAP exact)")
    ap.add_argument("--share-by", default=None, metavar="COL",
                    help="oof: take the 40%% cut within each value of this episodes.csv column (e.g. round)")
    a = ap.parse_args()
    if a.what == "oof":  # RECAP round 2 [2026-11-07]: value-binned 40% labels from any 5-fold OOF V on any table
        return oof_labels(Path(a.table), a.pattern, a.name, a.rule, a.share_by)
    r = base()
    v0 = td.labels_from_v(r, r.V.to_numpy(), N)
    cur = pd.read_parquet(LAB / "resnet_valuebinned_p40" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    assert (cur[td.KEY].values == v0[td.KEY].values).all() and (cur.positive.values == v0.positive.values).all(), \
        "vl.libero_td must reproduce the current labels"
    if a.what == "hindsight":  # coach test [2026-11-08]: is V the bottleneck? judge each frame by what happened
        return hindsight_sets(r, cur)
    if a.what == "targets":
        y = td.td_targets(r, r.V.to_numpy(), N)
        out = LAB / "td_targets_n20"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"episode_index": r.episode_index, "frame_index": r.frame_index, "value": y}).to_parquet(
            out / "targets.parquet")
        s = {"rule": "y = -n/1040 + V0(t+n) interior, MC outcome when the lookahead reaches T; V0 = oof_resnet_n20_p30 V",
             "n": N, "frames": len(r), "corr_with_V0": float(np.corrcoef(y, r.V)[0, 1]),
             "mean_abs_change_vs_V0": float(np.abs(y - r.V).mean())}
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=2), f"\n-> {out}")
        return
    e = pd.read_parquet(LAB / "oof_expert_n20_p30" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    assert (e[td.KEY].values == r[td.KEY].values).all()
    t = r[td.KEY].merge(oof_v(TD_PATTERN), on=td.KEY, how="left", validate="one_to_one")
    assert t.V.notna().all(), "TD OOF predictions must cover every train frame"
    v_td = t.V.to_numpy(float)
    stats = {"corr_Vtd_V0": float(np.corrcoef(v_td, r.V)[0, 1]), "corr_Vtd_expert": float(np.corrcoef(v_td, e.V)[0, 1]),
             "corr_V0_expert": float(np.corrcoef(r.V, e.V)[0, 1]), "td_runs": sorted(t.run.unique().tolist())}
    print(json.dumps(stats, indent=2))
    sets = {"td_vb40": (v_td, "V_td (5-fold OOF ResNet on fitted n-step TD targets)"),
            "ens_vb40": ((r.V.to_numpy() + e.V.to_numpy()) / 2, "mean of OOF ResNet V0 and OOF value-expert V"),
            "td_ens_vb40": ((v_td + e.V.to_numpy()) / 2, "mean of V_td and OOF value-expert V")}
    for name, (V, desc) in sets.items():
        lab = td.labels_from_v(r, V, N)
        kappa_cur = cohen_kappa(lab.positive.to_numpy(), cur.positive.to_numpy())
        write(name, lab, {"V": desc, "n": N, "rule": "value-binned (Dissecting Eq. 4), 40% positive (RECAP App. A-F)",
                          "kappa_vs_current": kappa_cur, **stats})


def hindsight_sets(r: pd.DataFrame, cur: pd.DataFrame):
    """Coach test [2026-11-08, ours]. Candidates for libero_tag_ideal_shift.py --candidates (no gate):
      hind_v0_vb40    A = G_t - V0(t): the realised return replaces V0's guess of the future (V0 = baseline only)
      hind_ens_vb40   same with the ResNet + value-expert mean as baseline (does the baseline matter?)
      look_n80_vb40 / look_n160_vb40   the current rule at 4 s / 8 s lookahead (between 1 s and the episode end)
    All value-binned 40% (the current rule). Reading [set before the result]: hindsight clearly above the current
    labels' action information -> a better V can help up to that ceiling; about equal or below -> V is not the
    bottleneck, the data (one policy, little action variety) is."""
    e = pd.read_parquet(LAB / "oof_expert_n20_p30" / "labels.parquet").sort_values(td.KEY).reset_index(drop=True)
    assert (e[td.KEY].values == r[td.KEY].values).all()
    v0, ens = r.V.to_numpy(float), (r.V.to_numpy(float) + e.V.to_numpy(float)) / 2
    sets = {"hind_v0_vb40": (td.hindsight_labels(r, v0), "A = realised return G_t - V0(t), V0 = OOF ResNet"),
            "hind_ens_vb40": (td.hindsight_labels(r, ens), "A = G_t - mean(OOF ResNet V, OOF value-expert V)"),
            "look_n80_vb40": (td.labels_from_v(r, v0, 80), "current rule, n = 80 frames (4 s), V0"),
            "look_n160_vb40": (td.labels_from_v(r, v0, 160), "current rule, n = 160 frames (8 s), V0")}
    for name, (lab, desc) in sets.items():
        write(name, lab, {"A": desc, "rule": "value-binned (Dissecting Eq. 4), 40% positive", "purpose": "coach test",
                          "kappa_vs_current": cohen_kappa(lab.positive.to_numpy(), cur.positive.to_numpy())})


def oof_labels(table: Path, pattern: str, name: str, rule: str = "vb", share_by: str | None = None):
    d = oof_v(pattern).sort_values(td.KEY).reset_index(drop=True)
    eps = pd.read_csv(table / "episodes.csv").set_index("episode_index")
    train = set(eps.index[eps.split == "train"])
    assert set(d.episode_index) == train, "OOF predictions must cover every train episode exactly"
    d["T"] = d.episode_index.map(eps.policy_steps - 1).to_numpy()
    d["success"] = d.episode_index.map(eps.success).to_numpy()
    groups = d.episode_index.map(eps[share_by]).to_numpy() if share_by else None
    lab = td.labels_from_v(d, d.V.to_numpy(float), N, binned=rule == "vb", groups=groups)
    out = table / "labels" / name
    out.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(out / "labels.parquet")
    s = {"pattern": pattern, "runs": sorted(d.run.unique().tolist()), "n": N, "frames": len(lab),
         "rule": ("value-binned (Dissecting Eq. 4)" if rule == "vb" else "one threshold on A (RECAP Eq. 5)")
         + ", 40% positive (RECAP App. A-F)" + (f", cut within each {share_by}" if share_by else ", pooled cut"),
         "positive": float(lab.positive.mean())}
    if share_by:
        s["positive_by_" + share_by] = {str(k): float(v) for k, v in lab.positive.groupby(groups).mean().items()}
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), f"\n-> {out}")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


if __name__ == "__main__":
    main()
