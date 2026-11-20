"""Advantage labels for one rollout round of the residual loop (scripts/ag_rounds.sh) [2026-11-11; mentor 11-11:
"run the current policy for more rounds"].

    .venv/bin/python scripts/libero_round_labels.py --table <round-r table> --round <r> --prev <label dir> --out <dir>

Round 0 = moka_fixed_fresh (base pi0.5 rollouts): V = the 5-fold OOF ResNet (resnet18_oof_f{k}_s0_*, the V behind
resnet_valuebinned_p40, the labels the 75.4% residual trained on). New rounds:
  V on a new episode = the OOF fold model that never saw its scene [ours: the same rule as round 0, so every round's
    labels come from one model per episode at round 0's noise level; a 5-model mean would also change the label
    noise, which would confound the round-over-round comparison]. Fixed-state episodes use the fold of their init
    state (oof_split's rng-0 deal of round 0's states); fresh layouts use fold = episode_index % 5 (no model saw any
    fresh layout of a later round). So every V is out-of-sample and no V is refit [ours; refit every 2-3 rounds is
    a separate decision].
  Labels = vl.libero_td.labels_from_v on every train episode of the table (all rounds), n = 20, value-binned, 40%
    positive with the cut taken within each round [RECAP App. A-F: ~40% of each round; value bins pooled], i.e.
    `libero_td_labels.py oof --share-by round` generalised to V from earlier rounds.
Writes <out>/labels.parquet (episode_index, frame_index, positive, A, z, V, term), <out>/v.parquet (V per frame, all
rounds, the next round's --prev input) and <out>/summary.json. --prev = the previous round's label dir, or "round0".
Checks: the round-0 frames' V equals the OOF predictions; with --check-round0 the same code reproduces
resnet_valuebinned_p40 exactly on round 0 alone.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vl import libero_td as td

ROOT = Path("/iris/u/ankile/karenvo/libero")
TABLE0 = ROOT / "tables" / "moka_fixed_fresh"
OOF_PATTERN = "resnet18_oof_f{k}_s0_*"
N, K_FOLDS = 20, 5


def oof_runs(pattern: str = OOF_PATTERN) -> list[Path]:
    runs = []
    for k in range(K_FOLDS):
        hits = sorted(d for d in glob.glob(str(ROOT / "runs" / pattern.format(k=k)))
                      if (Path(d) / "oof_predictions.parquet").exists())
        assert len(hits) == 1, (k, hits)
        runs.append(Path(hits[0]))
    return runs


def round0_v(runs: list[Path]) -> pd.DataFrame:
    p = pd.concat([pd.read_parquet(r / "oof_predictions.parquet") for r in runs], ignore_index=True)
    return p[td.KEY + ["ev_resnet"]].rename(columns={"ev_resnet": "V"})


def state_folds(eps0: pd.DataFrame) -> dict[int, int]:
    """Fold of each fixed train init state, exactly as vl.libero_oof.oof_split deals round 0's states."""
    tr = eps0[eps0.split == "train"]
    states = np.sort(tr.init_state[tr.init_state >= 0].unique())
    rng = np.random.default_rng(0)
    return {int(s): int(f) for s, f in zip(rng.permutation(states), np.arange(len(states)) % K_FOLDS)}


def new_episode_folds(new: pd.DataFrame, s_fold: dict[int, int]) -> pd.Series:
    """Fold model for each new episode: its init state's fold (fixed), episode_index % K_FOLDS (fresh layout)."""
    fixed = new.init_state >= 0
    missing = set(new.init_state[fixed]) - set(s_fold)
    assert not missing, f"init states not in round 0's train split: {sorted(missing)}"
    f = np.where(fixed, new.init_state.map(s_fold).fillna(-1), new.episode_index % K_FOLDS).astype(int)
    return pd.Series(f, index=new.episode_index.values)


def predict_new(table: Path, new_eps: list[int], folds: pd.Series, runs: list[Path], out: Path) -> pd.DataFrame:
    """V of every 4th frame of the new episodes from their fold's best.pt (raw softmax mean, as oof_predictions)."""
    import torch

    from vl.libero_probe import CENTERS
    from vl.libero_resnet import CAMS, evaluate, pack, table_meta
    from vl.model import ValueModel

    meta = table_meta(table, 4)
    meta = meta[meta.episode_index.isin(new_eps)].reset_index(drop=True)
    arr = pack(meta, out / "frames_every4.npy", table)
    bins = meta.bin.values.astype(np.int64)
    fold = meta.episode_index.map(folds).to_numpy()
    V = np.full(len(meta), np.nan)
    model = ValueModel(CAMS, n_bins=len(CENTERS)).to("cuda")
    for k, run in enumerate(runs):
        idx = np.flatnonzero(fold == k)
        if not len(idx):
            continue
        model.load_state_dict(torch.load(run / "best.pt"))
        p, ce = evaluate(model, arr, idx, bins, "cuda")
        V[idx] = p @ CENTERS
        print(f"fold {k} ({run.name}): {len(set(meta.episode_index[idx]))} episodes, {len(idx)} frames, CE {ce:.3f}",
              flush=True)
    assert np.isfinite(V).all()
    return meta[td.KEY + ["value"]].assign(V=V, fold=fold)


def labels_all(eps: pd.DataFrame, v: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """(labels, round per row): value-binned 40% labels over every train episode in v, the cut within each round."""
    d = v[td.KEY + ["V"]].sort_values(td.KEY).reset_index(drop=True)
    e = eps.set_index("episode_index")
    d["T"] = d.episode_index.map(e.policy_steps - 1).to_numpy()
    d["success"] = d.episode_index.map(e.success).to_numpy()
    rounds = d.episode_index.map(e["round"]).to_numpy()
    groups = rounds
    if "task" in e.columns and e.task.nunique() > 1:  # [2026-11-16, multi-task AG] the 40% cut within (round, task)
        tasks = d.episode_index.map(e.task).to_numpy()
        groups = pd.factorize(pd.Series(rounds).astype(str) + "|" + pd.Series(tasks).astype(str))[0]
    return td.labels_from_v(d, d.V.to_numpy(float), N, groups=groups), rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="episode table with all rounds (libero_round2_table.py output)")
    ap.add_argument("--round", type=int, required=True, help="the table's newest round (its new episodes get V)")
    ap.add_argument("--prev", required=True, help="previous round's label dir (its v.parquet), or 'round0'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-round0", action="store_true", help="also check round 0 alone reproduces vb40")
    a = ap.parse_args()
    table, out = Path(a.table), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(table / "episodes.csv", dtype={"goals_end": str})
    if "round" not in eps.columns:
        eps["round"] = 1
    eps = eps[eps.split == "train"]
    runs = oof_runs()
    v0 = round0_v(runs)
    eps0 = pd.read_csv(TABLE0 / "episodes.csv")
    if a.check_round0:
        lab0, _ = labels_all(eps0[eps0.split == "train"].assign(round=1), v0)
        ref = pd.read_parquet(TABLE0 / "labels" / "resnet_valuebinned_p40" / "labels.parquet").sort_values(td.KEY)
        assert (lab0[td.KEY].values == ref[td.KEY].values).all() and (lab0.positive.values == ref.positive.values).all()
        print("round-0 check: resnet_valuebinned_p40 reproduced exactly")
    # [2026-11-16] fixed-budget chains (libero_round0_subsample.py): round 0 may be a subsample of the 900, so the
    # round-0 checks below cover the round-0 episodes in this table (all 900 for the 100-per-round chain: unchanged)
    v0 = v0[v0.episode_index.isin(eps.episode_index)]
    prev = v0 if a.prev == "round0" else pd.read_parquet(Path(a.prev) / "v.parquet")[td.KEY + ["V"]]
    new = eps[eps["round"] == a.round]
    assert len(new), f"no round-{a.round} train episodes in {table}"
    old = eps[eps["round"] < a.round]
    assert set(prev.episode_index) == set(old.episode_index), "--prev must cover every earlier train episode exactly"
    assert set(v0.episode_index) <= set(prev.episode_index), "round-0 episodes missing from --prev"
    chk = prev.merge(v0, on=td.KEY, suffixes=("", "_0"))
    assert len(chk) == len(v0) and np.array_equal(chk.V.to_numpy(), chk.V_0.to_numpy()), "round-0 V changed"

    folds = new_episode_folds(new, state_folds(eps0))
    pn = predict_new(table, new.episode_index.tolist(), folds, runs, out)
    v = pd.concat([prev, pn[td.KEY + ["V"]]], ignore_index=True).sort_values(td.KEY).reset_index(drop=True)
    v.to_parquet(out / "v.parquet")
    lab, rounds = labels_all(eps, v)
    lab.to_parquet(out / "labels.parquet")
    (out / "frames_every4.npy").unlink(missing_ok=True)  # cache only needed for the prediction

    e = eps.set_index("episode_index")
    succ = lab.episode_index.map(e.success).to_numpy()
    per_round = {}
    for r in np.unique(rounds):
        m = rounds == r
        er = eps[eps["round"] == r]
        per_round[str(int(r))] = {"episodes": int(len(er)), "success": float(er.success.mean()),
                                  "frames": int(m.sum()), "positive": float(lab.positive[m].mean()),
                                  "positive_success_eps": float(lab.positive[m & (succ == 1)].mean()),
                                  "positive_failure_eps": float(lab.positive[m & (succ == 0)].mean()),
                                  "mean_V": float(lab.V[m].mean())}
    s = {"table": str(table), "round": a.round, "prev": a.prev, "oof_runs": [r.name for r in runs], "n": N,
         "rule": "value-binned (Dissecting Eq. 4), 40% positive cut within each round (RECAP App. A-F)",
         "V_new": "OOF fold model that never saw the episode's scene (fixed: init state's fold; fresh: ep % 5)",
         "new_mae_vs_mc": float(np.abs(pn.V - pn.value).mean()), "frames": int(len(lab)),
         "positive": float(lab.positive.mean()), "by_round": per_round}
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), f"\n-> {out}")


if __name__ == "__main__":
    main()
