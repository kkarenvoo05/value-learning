"""vb40 labels from a ResNet V fit on the chain's own episodes only [2026-11-16; Karen: "instead of using the v trained
on 900 episodes, should we use V trained on the respective amount so it just uses the data in its chain? since if we
want to port this over to zima, we will not have a v that is trained on that many episodes"; "set up chain-only V"].

Same V method as the frozen round-0 V (resnet18_oof_f{k}_s0_*: python -m vl.libero_resnet --every 4 --oof-fold k,
ImageNet ResNet-18 start, 8000 steps, early stop on inner CE), but fit FROM SCRATCH each round on every train episode
of --table (the chain's round-0 subsample + its later rounds), never on the other base rollouts:
  - 5 folds by episode (vl.libero_oof.oof_split; fresh-only tables: inner = 20% of the other folds' episodes); each
    frame's V comes from the fold model that neither fit nor early-stopped on its episode.
  - labels = libero_round_labels.labels_all (vl.libero_td.labels_from_v, n = 20, value-binned, 40% positive cut within
    each round) on every round of the table, i.e. vb40's rule. Every round is relabelled with the newest V (the vb40
    chains keep earlier rounds' frozen V).
  - val episodes (the 10 val states of the base rollouts) are only scored (final_eval), never fit or selected on.
Writes <out>/labels.parquet (episode_index, frame_index, positive, A, z, V, term), <out>/v.parquet, <out>/summary.json,
the fold runs under <out>/v_folds/f<k>. The frame cache <table>/frames_every4.npy is deleted at the end.

    .venv/bin/python scripts/libero_chainv_round_labels.py --table <tables>/moka_agr_b50_r0 --out <label dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_round_labels import N, labels_all  # noqa: E402
from vl import libero_td as td  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f} s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8000)  # = the round-0 OOF runs
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-cache", action="store_true")
    a = ap.parse_args()
    table, out = Path(a.table), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(table / "episodes.csv", dtype={"goals_end": str})
    if "round" not in eps.columns:
        eps["round"] = 1
    eps = eps[eps.split == "train"]
    assert (eps.init_state < 0).all(), "chain-only V is built for fresh-layout-only tables"
    log(f"{table.name}: {len(eps)} train episodes, rounds {sorted(eps['round'].unique().tolist())}")
    preds, folds = [], {}
    try:
        for k in range(a.folds):
            run = out / "v_folds" / f"f{k}"
            if not (run / "oof_predictions.parquet").exists():
                cmd = [sys.executable, "-m", "vl.libero_resnet", "--every", "4", "--dataset", str(table),
                       "--oof-fold", str(k), "--oof-folds", str(a.folds), "--steps", str(a.steps),
                       "--seed", str(a.seed), "--out", str(run)]
                log("fold %d: %s" % (k, " ".join(cmd)))
                subprocess.run(cmd, check=True)
            p = pd.read_parquet(run / "oof_predictions.parquet")
            r = json.loads((run / "results.json").read_text())
            folds[k] = {"episodes": int(p.episode_index.nunique()), "best_step": r["val"].get("best_step"),
                        "inner_ce": r["val"].get("inner_ce"), "val_mae": r["val"]["mae"],
                        "val_auc_20s": r["val"].get("auc_20s"), "oof_mae": float((p.ev_resnet - p.value).abs().mean())}
            log(f"fold {k}: {folds[k]}")
            preds.append(p)
    finally:
        if not a.keep_cache:
            (table / "frames_every4.npy").unlink(missing_ok=True)
    p = pd.concat(preds, ignore_index=True)
    assert not p.duplicated(td.KEY).any() and set(p.episode_index) == set(eps.episode_index), \
        "every train episode must get exactly one out-of-fold V"
    v = p[td.KEY + ["ev_resnet"]].rename(columns={"ev_resnet": "V"}).sort_values(td.KEY).reset_index(drop=True)
    v.to_parquet(out / "v.parquet")
    lab, rounds = labels_all(eps, v)
    lab.to_parquet(out / "labels.parquet")

    e = eps.set_index("episode_index")
    succ = lab.episode_index.map(e.success).to_numpy()
    mc = lab[td.KEY].merge(p[td.KEY + ["value"]], on=td.KEY).value.to_numpy()
    per_round = {}
    for r in np.unique(rounds):
        m = rounds == r
        er = eps[eps["round"] == r]
        per_round[str(int(r))] = {"episodes": int(len(er)), "success": float(er.success.mean()),
                                  "frames": int(m.sum()), "positive": float(lab.positive[m].mean()),
                                  "positive_success_eps": float(lab.positive[m & (succ == 1)].mean()),
                                  "positive_failure_eps": float(lab.positive[m & (succ == 0)].mean()),
                                  "mean_V": float(lab.V[m].mean()), "mae_V_mc": float(np.abs(lab.V[m] - mc[m]).mean())}
    s = {"table": str(table), "V": "ResNet-18 OOF (vl.libero_resnet --oof-fold), fit from scratch on this table's "
         "train episodes only", "steps": a.steps, "folds": folds, "n": N,
         "rule": "value-binned (Dissecting Eq. 4), 40% positive cut within each round (RECAP App. A-F)",
         "frames": int(len(lab)), "positive": float(lab.positive.mean()),
         "corr_V_mc": float(np.corrcoef(lab.V, mc)[0, 1]), "mae_V_mc": float(np.abs(lab.V - mc).mean()),
         "by_round": per_round}
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
