"""Audit depth results and prepare matched frame pairs for fixed-expert interventions.

Usage: .venv/bin/python scripts/libero_layer_diagnosis.py --out reports/2026-11-05-layer-interventions
See that report's protocol.md. Images are saved to a feature cache, never to tracked report files.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vl.layer_diagnosis import hierarchical_comparison, pair_metadata, per_seed_metrics

RUNS = {
    "n18_s0": "value_expert_fixedfresh_w256_s0_18237898",
    "n18_s1": "value_expert_fixedfresh_w256_s1_18237899",
    "n18_s2": "value_expert_fixedfresh_w256_s2_18237900",
    "n4_s0": "value_expert_trunc4_fixedfresh_w256_s0_18291640",
    "n4_s1": "value_expert_trunc4_fixedfresh_w256_s1_18309645",
    "n4_s2": "value_expert_trunc4_fixedfresh_w256_s2_18309646",
    "n9_s0": "value_expert_trunc9_fixedfresh_w256_s0_18291639",
}


def choose_pairs(candidates, n, seed=20261105):
    rng = np.random.default_rng(seed)
    eligible = candidates[candidates.no_goal_transition & (candidates.pot_motion_m <= .005)].copy()
    eligible["abs_jump"] = eligible.dv_n18_s0.abs()
    selected, used = [], []

    def available(row):
        return all(row.episode_index != e or abs(row.t1 - t) >= 20 for e, t in used)

    def add(row, cohort, match=None):
        d = row.to_dict()
        d.update(cohort=cohort, matched_to=match, pair_id=f"{cohort}_{sum(x['cohort'] == cohort for x in selected):02d}")
        selected.append(d)
        used.append((int(row.episode_index), int(row.t1)))
        return d["pair_id"]

    counts = {}
    for _, row in eligible.sort_values("abs_jump", ascending=False).iterrows():
        if counts.get(row.episode_index, 0) >= 2 or not available(row):
            continue
        add(row, "jump")
        counts[row.episode_index] = counts.get(row.episode_index, 0) + 1
        if len(selected) >= n:
            break
    jumps = list(selected)
    for jump in jumps:
        q = eligible[eligible.episode_index == jump["episode_index"]].copy()
        q = q[q.abs_jump <= q.abs_jump.quantile(.25)]
        q["match_cost"] = ((q.t1 - jump["t1"]).abs() / 80
                           + (q.eef_motion_m - jump["eef_motion_m"]).abs() / .01
                           + (q.gripper_motion - jump["gripper_motion"]).abs() / .005)
        for _, row in q.sort_values("match_cost").iterrows():
            if available(row):
                add(row, "steady", jump["pair_id"])
                break
    for cohort, pool in (("random", eligible), ("placement", candidates[candidates.placement])):
        seen, taken = set(), 0
        for idx in rng.permutation(pool.index):
            row = pool.loc[idx]
            if row.episode_index in seen or not available(row):
                continue
            add(row, cohort)
            seen.add(row.episode_index)
            taken += 1
            if taken >= n:
                break
    return pd.DataFrame(selected).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=Path("/iris/u/ankile/karenvo/libero/runs"))
    ap.add_argument("--dataset", type=Path, default=Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/move/u/karenvo/libero_features/layer_diagnosis_20261105"))
    ap.add_argument("--bootstraps", type=int, default=2000)
    ap.add_argument("--per-cohort", type=int, default=12)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--skip-bootstrap", action="store_true", help="Reuse a completed bootstrap after checking prediction hashes")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    keys = ["episode_index", "frame_index"]
    predictions, provenance, rows = {}, {}, []
    reference = None
    for name, directory in RUNS.items():
        run = a.runs_root / directory
        path = run / "val_predictions.parquet"
        p = pd.read_parquet(path).sort_values(keys).reset_index(drop=True)
        assert not p.duplicated(keys).any(), name
        if reference is None:
            reference = p
        else:
            for col in keys + ["value", "success", "init_state"]:
                assert np.array_equal(p[col], reference[col]), (name, col)
        predictions[name] = p
        r = json.loads((run / "results.json").read_text())
        provenance[name] = {"run": str(run), "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "setup": r["setup"], "best_step": r["best_step"], "params_M": r["params_M"]}
        rows.append({"name": name, "best_step": r["best_step"], "params_M": r["params_M"]})
    eps = reference.groupby("episode_index", sort=True).first()
    episodes = eps.index.to_numpy()
    frames = np.sort(reference.frame_index.unique())

    def wide(p, col):
        return p.pivot(index="episode_index", columns="frame_index", values=col).reindex(index=episodes, columns=frames).to_numpy()

    targets = wide(reference, "value")
    success, scenes = eps.success.to_numpy(), eps.init_state.to_numpy()
    values = {name: wide(p, "ev_expert") for name, p in predictions.items()}
    for row in rows:
        row.update({k: float(v[0]) for k, v in per_seed_metrics(values[row["name"]][None], targets, success, frames).items()})
    score = pd.DataFrame(rows)
    score.to_csv(a.out / "audit_scores.csv", index=False)
    if a.skip_bootstrap:
        previous = json.loads((a.out / "provenance.json").read_text())
        assert all(previous[k]["predictions_sha256"] == provenance[k]["predictions_sha256"] for k in RUNS)
        comparison = json.loads((a.out / "bootstrap.json").read_text())["metrics"]
    else:
        comparison = hierarchical_comparison(np.stack([values[f"n18_s{s}"] for s in range(3)]),
                                            np.stack([values[f"n4_s{s}"] for s in range(3)]),
                                            targets, success, frames, scenes, draws=a.bootstraps)
        (a.out / "bootstrap.json").write_text(json.dumps({"direction": "n4 minus n18", "draws": a.bootstraps,
            "resampling": "paired scenes then episodes; independent seeds per arm", "metrics": comparison}, indent=2))
    provenance["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    provenance["command"] = sys.argv
    (a.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(score.round(6).to_string(index=False), flush=True)
    print("N4 minus N18:", json.dumps(comparison, indent=2), flush=True)
    if a.audit_only:
        return

    table = pd.read_csv(a.dataset / "episodes.csv").set_index("episode_index")
    assert (table.loc[episodes, "split"] == "val").all()
    raw_paths = {}
    pairs = []
    indexed = {name: p.set_index(keys) for name, p in predictions.items()}
    for ei, episode in enumerate(episodes):
        p = Path(table.loc[episode, "raw"])
        raw_paths[int(episode)] = p if p.is_absolute() else a.runs_root / "stage2_moka_18155849" / p
        with np.load(raw_paths[int(episode)]) as z:
            goals, state = z["goals"], z["state"]
            names = [str(k) for k in z["object_pos_keys"]]
            pot_ix = [i for i, k in enumerate(names) if k in ("moka_pot_1_pos", "moka_pot_2_pos")]
            assert len(pot_ix) == 2, names
            pos = z["object_pos"].reshape(len(state), -1, 3)[:, pot_ix]
        ef = reference[reference.episode_index == episode].frame_index.to_numpy()
        for t0, t1 in zip(ef[:-1], ef[1:]):
            if t1 - t0 != 4 or t0 < 20 or t1 > 340:
                continue
            row = {"episode_index": int(episode), "init_state": int(eps.loc[episode, "init_state"]),
                   "success": int(eps.loc[episode, "success"]), "t0": int(t0), "t1": int(t1),
                   **pair_metadata(goals, pos, state, t0, t1)}
            for name, pp in indexed.items():
                va, vb = [float(pp.loc[(episode, t), "ev_expert"]) for t in (t0, t1)]
                row.update({f"va_{name}": va, f"vb_{name}": vb, f"dv_{name}": vb - va})
            pairs.append(row)
        if ei % 20 == 0:
            print(f"simulator metadata {ei + 1}/{len(episodes)} episodes", flush=True)
    candidates = pd.DataFrame(pairs)
    candidates.to_parquet(a.cache / "all_candidates.parquet", index=False)
    selected = choose_pairs(candidates, a.per_cohort)
    selected.to_csv(a.out / "pairs.csv", index=False)
    summary = []
    for cohort, group in selected.groupby("cohort"):
        for name in RUNS:
            summary.append({"cohort": cohort, "model": name, "n": len(group),
                            "mean_abs_dv": float(group[f"dv_{name}"].abs().mean()),
                            "mean_dv": float(group[f"dv_{name}"].mean())})
    pd.DataFrame(summary).to_csv(a.out / "cohort_scores.csv", index=False)
    # Read each raw image stream once and retain only the selected endpoint frames.
    images, wrists, states, prompts = {}, {}, {}, {}
    for episode, group in selected.groupby("episode_index"):
        times = np.unique(group[["t0", "t1"]].to_numpy())
        with np.load(raw_paths[int(episode)]) as z:
            for name, dest in (("image", images), ("wrist_image", wrists), ("state", states)):
                x = z[name]
                for t in times:
                    dest[(int(episode), int(t))] = x[t].copy()
            prompts[int(episode)] = str(z["prompt"])
        print(f"packed selected images for episode {episode}", flush=True)
    arrays = {}
    for name, src in (("image", images), ("wrist_image", wrists), ("state", states)):
        arrays[name] = np.stack([[src[(int(r.episode_index), int(t))] for t in (r.t0, r.t1)] for r in selected.itertuples()])
    arrays["prompt"] = np.array([prompts[int(r.episode_index)] for r in selected.itertuples()])
    arrays["pair_id"] = selected.pair_id.to_numpy(dtype=str)
    np.savez(a.cache / "pairs.npz", **arrays)
    print("pair cohorts:", selected.groupby("cohort").size().to_dict(), flush=True)
    print("cache:", a.cache / "pairs.npz", flush=True)


if __name__ == "__main__":
    main()
