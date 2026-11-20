"""Round 0 of a fixed-budget AG chain from a subsample of the base pi0.5 rollouts [2026-11-16; mentor: "all rounds
can be the same, so 50/100/200 every time, including the first, instead of the 900"; Karen: "build the subsample
round 0"].

Instead of collecting round 0 again with base pi0.5, take it from moka_fixed_fresh (the 900 base train rollouts:
15 per train state on the 30 fixed train states + 450 fresh layouts), with the same fixed / fresh mix as later
rounds: --fixed-repeats R of the 15 repeats on every train state + --fresh F fresh layouts.
  - Nested [ours]: one rng-seeded permutation of the repeats per state and one of the fresh layouts, first R / F
    taken, so the 50 subsample is inside the 100 one, which is inside the 200 one.
  - Val rows (120 episodes on the 10 val states) are kept unchanged: the trainer's val loss reads them; never trained on.
  - Labels exactly as libero_round_labels.py does for round 0: V = the 5-fold OOF ResNet predictions (each episode from
    the fold that never saw its scene), value-binned 40% cut, n = 20, computed on the subsample's train episodes only.
    Caveat [stated]: those fold models were fit on all 900 base rollouts, so V has seen more data than the chain
    (option (a), "V pre-trained on 900 base rollouts"; the chain-only V comes with IQL).
Writes <out>/episodes.csv (round = 1), <out>/frame_targets.parquet (filtered copy), <out>/subsample.json and
<out>/labels/<label name>/{labels,v}.parquet + summary.json (the next round's --prev).
Check: --fixed-repeats 15 --fresh 450 must reproduce resnet_valuebinned_p40 exactly (--check).

    .venv/bin/python scripts/libero_round0_subsample.py --fixed-repeats 2 --fresh 40 --out <tables>/moka_agr_b100_r0 \
        --label-name agr_b100_vb40_r0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_round_labels import TABLE0, labels_all, oof_runs, round0_v  # noqa: E402
from vl import libero_td as td  # noqa: E402


def subsample(eps: pd.DataFrame, fixed_repeats: int, fresh: int, seed: int) -> pd.DataFrame:
    tr = eps[eps.split == "train"]
    fx, fr = tr[tr.init_state >= 0], tr[tr.init_state < 0]
    rng = np.random.default_rng(seed)
    keep = []
    for s in sorted(fx.init_state.unique()):  # sorted, so the rng draws are fixed by the seed alone
        e = fx[fx.init_state == s].sort_values("repeat")
        assert len(e) >= fixed_repeats, (s, len(e))
        keep += e.episode_index.to_numpy()[rng.permutation(len(e))[:fixed_repeats]].tolist()
    f = fr.sort_values("episode_index").episode_index.to_numpy()
    assert len(f) >= fresh, len(f)
    keep += f[rng.permutation(len(f))[:fresh]].tolist()
    return tr[tr.episode_index.isin(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-repeats", type=int, required=True)
    ap.add_argument("--fresh", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-name", required=True)
    ap.add_argument("--check", action="store_true", help="full sample: must reproduce resnet_valuebinned_p40")
    a = ap.parse_args()
    out = Path(a.out)
    lab_dir = out / "labels" / a.label_name
    lab_dir.mkdir(parents=True, exist_ok=True)

    eps0 = pd.read_csv(TABLE0 / "episodes.csv", dtype={"goals_end": str})
    sub = subsample(eps0, a.fixed_repeats, a.fresh, a.seed)
    n_states = eps0[(eps0.split == "train") & (eps0.init_state >= 0)].init_state.nunique()
    assert len(sub) == a.fixed_repeats * n_states + a.fresh, len(sub)
    eps = pd.concat([sub, eps0[eps0.split == "val"]]).sort_values("episode_index").assign(round=1)
    eps.to_csv(out / "episodes.csv", index=False)
    ft = pd.read_parquet(TABLE0 / "frame_targets.parquet")
    ft[ft.episode_index.isin(eps.episode_index)].to_parquet(out / "frame_targets.parquet")

    runs = oof_runs()
    v0 = round0_v(runs)
    v = v0[v0.episode_index.isin(sub.episode_index)].sort_values(td.KEY).reset_index(drop=True)
    assert set(v.episode_index) == set(sub.episode_index), "OOF V missing for some subsample episodes"
    v.to_parquet(lab_dir / "v.parquet")
    lab, _ = labels_all(sub.assign(round=1), v)
    lab.to_parquet(lab_dir / "labels.parquet")
    if a.check:
        ref = pd.read_parquet(TABLE0 / "labels" / "resnet_valuebinned_p40" / "labels.parquet").sort_values(td.KEY)
        lab_s = lab.sort_values(td.KEY)
        assert (lab_s[td.KEY].values == ref[td.KEY].values).all(), "frames differ"
        assert (lab_s.positive.values == ref.positive.values).all(), "labels differ"
        print("check: full sample reproduces resnet_valuebinned_p40 exactly")

    succ = lab.episode_index.map(sub.set_index("episode_index").success).to_numpy()
    info = {"source": str(TABLE0), "fixed_repeats": a.fixed_repeats, "fresh": a.fresh, "seed": a.seed,
            "episodes": int(len(sub)), "fixed": int((sub.init_state >= 0).sum()), "fresh_eps": int((sub.init_state < 0).sum()),
            "success": float(sub.success.mean()), "val_episodes": int((eps.split == "val").sum()),
            "episode_ids": sorted(int(x) for x in sub.episode_index)}
    (out / "subsample.json").write_text(json.dumps(info, indent=1))
    s = {"table": str(out), "round": 1, "prev": "round0 (subsample)", "oof_runs": [r.name for r in runs], "n": 20,
         "rule": "value-binned (Dissecting Eq. 4), 40% positive cut within the round (RECAP App. A-F)",
         "V": "round-0 OOF ResNet predictions (fold models fit on all 900 base rollouts)",
         "frames": int(len(lab)), "positive": float(lab.positive.mean()),
         "by_round": {"1": {"episodes": int(len(sub)), "success": float(sub.success.mean()), "frames": int(len(lab)),
                            "positive": float(lab.positive.mean()),
                            "positive_success_eps": float(lab.positive[succ == 1].mean()),
                            "positive_failure_eps": float(lab.positive[succ == 0].mean()),
                            "mean_V": float(lab.V.mean())}}}
    (lab_dir / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"{out}: {info['episodes']} train episodes ({info['fixed']} fixed, {info['fresh_eps']} fresh), success "
          f"{info['success']:.3f}, {len(lab)} labelled frames, positive {lab.positive.mean():.3f}")


if __name__ == "__main__":
    main()
