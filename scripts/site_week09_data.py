"""Chart data for the week-9 progress page [2026-11-09] -> reports/site/content/week09_data.json.

Everything is recomputed from the run directories, so the page and its numbers agree:
  policy   every comparable policy eval (libero_10 task 8, val states x 33, env seed 7, matched noise seed 1) vs base
           18355478, paired by (init state, repeat), 5000 bootstrap draws; incomplete evals are flagged, not dropped
  tag      tag effect on the executed action (+ vs - tag, same noise) as a share of the policy's own sampling noise,
           from the tag-usage checks (final checkpoints), with planted-tag controls and the prompt-swap reference
  coach    ideal shift (how much the action could depend on the tag, given labels + data) per label source
  curve    action information (coach test, held-out) vs number of episodes: labels vs the PC1 positive control
  branch   the 3 failure-branching runs: V* (true success chance, 12 continuations) along each failed episode, with
           the vb40 label of each branch point; split-half reliability per run and pooled
Usage: .venv/bin/python scripts/site_week09_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
OUT = ROOT / "reports/site/content/week09_data.json"
BASE = "eval_base_noise1_libero_10_18355478"

# (eval dir, label, group, note)
POLICY = [
    ("eval_armA_r1_noise1_libero_10_18376075", "Successes only", "untagged", "516 successful rollouts, frozen vision encoder"),
    ("eval_armU_alluntagged_noise1_libero_10_18376077", "All rollouts, demos 2%", "untagged",
     "900 rollouts (successes + failures) + 29 demos, original lr"),
    ("eval_grid_d50_orig_s0_atfinal_noise1_libero_10_18396165", "All rollouts, demos 50%", "untagged", "original lr, final"),
    ("eval_grid_d75_orig_s0_atfinal_noise1_libero_10_18396169", "All rollouts, demos 75%", "untagged", "original lr, final"),
    ("eval_grid_d75_low_s0_at2000_noise1_libero_10_18396176", "Demos 75%, low lr, step 2000", "untagged",
     "10x lower lr, checkpoint at step 2000"),
    ("eval_grid_d75_low_s0_atfinal_noise1_libero_10_18396177", "Demos 75%, low lr, final", "untagged", "10x lower lr, step 5000"),
    ("eval_grid_d02_low_s0_atfinal_noise1_libero_10_18396977", "Demos 2%, low lr, final", "untagged", "10x lower lr, step 5000"),
    ("eval_armC_adarms_vb40_pos_noise1_libero_10_18355679", "Tag via AdaRMS", "tag",
     "all rollouts + demos, + tag; tag fed to the action expert's norm layers"),
    ("eval_armC_disc_vb40_pos_noise1_libero_10_18364108", "Tag in prompt + discrete loss", "tag", "+ tag; FAST-style token loss"),
    ("eval_armC_disc_vb40_speedpre_pos_noise1_libero_10_18364109", "… + speed-tag pre-training", "tag",
     "tag pre-trained on demos of all 40 LIBERO tasks"),
    ("eval_armC_recapfull_pos_noise1_libero_10_18376192", "RECAP-complete", "tag",
     "prompt tag + real FAST tokens + knowledge insulation + subtask text + dropout 0.3 + demos"),
    ("eval_armW_vbweights_noise1_libero_10_18376079", "Advantage weights (all rollouts)", "weights",
     "no tag; loss weight exp(advantage), all rollouts + demos"),
    ("eval_armCS_succtag_pos_noise1_libero_10_18387310", "Successes + tag", "success+", "+ tag, 516 successes"),
    ("eval_armWS_succweights_noise1_libero_10_18387308", "Successes + advantage weights", "success+",
     "loss weights on the 516 successes"),
]


def load(d):
    e = pd.read_csv(RUNS / d / "episodes.csv")
    return e.set_index(["init_state", "repeat"]).success


def policy_rows():
    base = load(BASE)
    rng = np.random.default_rng(0)
    rows = []
    for d, label, group, note in POLICY:
        if not (RUNS / d / "episodes.csv").exists():
            continue
        a = load(d)
        j = a.index.intersection(base.index)
        x = a[j].values - base[j].values
        bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(label=label, group=group, note=note, job=d.rsplit("_", 1)[1], n=int(len(a)),
                         success=round(float(a.mean()), 4), diff=round(float(x.mean()), 4),
                         lo=round(float(lo), 4), hi=round(float(hi), 4), complete=bool(len(a) == 330)))
    return dict(base=round(float(base.mean()), 4), base_job="18355478", rows=rows)


TAG = [  # (tag-usage dir, arm key, label, group)
    ("tag_usage_18310225", "tagged", "Oracle tag (true outcome)", "real"),
    ("tag_usage_18338865", "tagged", "Prompt tag (RECAP labels)", "real"),
    ("tag_usage_18352443", "tagged", "Prompt tag (subtask-scoped labels)", "real"),
    ("tag_usage_18359339", "tagged@final", "Prompt tag + discrete loss", "real"),
    ("tag_usage_18363755", "tagged@final", "… + speed-tag pre-training", "real"),
    ("tag_usage_18376191", "tagged@final", "RECAP-complete", "real"),
    ("tag_usage_adarms_vb40_18355677", "tagged@final", "AdaRMS tag (RECAP labels)", "real"),
    ("tag_usage_18338865", "A_success_only", "Never tagged (reference)", "floor"),
    ("tag_usage_18352947", "tagged@final", "Planted tag, small (prompt)", "control"),
    ("tag_usage_adarms_ctrl_shift0p02_18361554", "tagged@final", "Planted tag, small (AdaRMS)", "control"),
    ("tag_usage_18352945", "tagged@final", "Planted tag, large (prompt)", "control"),
]


def tag_rows():
    rows = []
    for d, k, label, group in TAG:
        r = json.loads((RUNS / d / "results.json").read_text())["arms"][k]
        raw = r["action"]["raw"]
        rows.append(dict(label=label, group=group, ratio=round(raw["ratio_tag_over_noise"], 4),
                         shift_raw=round(raw["tag_pos_vs_neg"], 5), pos_vs_none_raw=round(raw.get("tag_pos_vs_none", np.nan), 5),
                         loss_correct=r["loss"]["loss_correct"], loss_flipped=r["loss"].get("loss_flipped"), job=d))
    pt = json.loads((RUNS / "prompt_test_1106b/results.json").read_text())["arms"]["tagged"]["prompt"]["norm"]
    return dict(rows=rows, prompt_swap_far=round(pt["far"]["ratio_over_noise"], 3),
                prompt_swap_near=round(pt["near"]["ratio_over_noise"], 3))


def coach_rows():
    rows = []
    for f, names in [("reports/2026-11-08-coach-test/ideal_shift/ideal_shift_summary.json",
                      {"RECAP faithful (value-binned 40%)": "Current: ResNet value, 1 s advantage",
                       "candidate hind_v0_vb40": "Hindsight (what actually happened − V)",
                       "candidate look_n80_vb40": "Longer lookahead, 4 s", "candidate look_n160_vb40": "Longer lookahead, 8 s",
                       "candidate oracle_outcome": "Episode outcome (success / failure)",
                       "CONTROL- random 40%": "Random labels (floor)"}),
                     ("reports/2026-11-07-td-labels/ideal_shift/ideal_shift_summary.json",
                      {"candidate td_vb40": "TD critic", "candidate ens_vb40": "ResNet + π₀.₅ ensemble"})]:
        for r in json.loads((ROOT / f).read_text())["results"]:
            if r["labels"] in names:
                rows.append(dict(label=names[r["labels"]], value=round(r["ideal_shift_raw"], 4),
                                 lo=round(r["ideal_shift_ci95"][0], 4), hi=round(r["ideal_shift_ci95"][1], 4)))
    rules = []
    for d, lab in [("coach_explore", "Ours: value-binned 40%"), ("coach_explore_thr40", "RECAP: one threshold, 40%"),
                   ("coach_explore_vb40_round", "Ours, 40% per round"), ("coach_explore_thr40_round", "RECAP exact: threshold per round")]:
        s = json.loads((ROOT / "reports/2026-11-08-explore-pilot" / d / "ideal_shift_summary.json").read_text())
        for r in s["results"]:
            if r["labels"].startswith("explore"):
                rules.append(dict(label=lab, round=r["group"], value=round(r["ideal_shift_raw"], 4)))
            elif r["labels"].startswith("positive") and d == "coach_explore":
                rules.append(dict(label="PC1 positive control", round=r["group"], value=round(r["ideal_shift_raw"], 4)))
            elif r["labels"].startswith("random") and d == "coach_explore":
                rules.append(dict(label="Random labels (floor)", round=r["group"], value=round(r["ideal_shift_raw"], 4)))
    return dict(sources=rows, rules=rules)


def curve_rows():
    out = {}
    for key, d in [("old", "old_fixed"), ("explore", "explore_explore_vb40_pooled")]:
        c = pd.read_csv(ROOT / "reports/2026-11-08-coach-curve" / d / "curve.csv")
        c = c[c.kind.isin(["labels", "pc1"])]
        out[key] = [dict(kind=r.kind, n=int(r.n_sub), info=round(r.gain_action_over_state_mean, 4),
                         info_sd=None if pd.isna(r.gain_action_over_state_std) else round(r.gain_action_over_state_std, 4),
                         shift_over_random=round(r.shift_over_random, 4), mse_over_random=round(r.mse_gain_over_random, 4),
                         draws=int(r.subsamples)) for r in c.itertuples()]
    return out


def branch_rows():
    runs = ["branch_fail_a_libero_10_18395412", "branch_fail_b_libero_10_18395413", "branch_fail_c_libero_10_18395414"]
    d = pd.concat([pd.read_csv(RUNS / r / "points.csv").assign(run=r) for r in runs])
    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)
    rel = [dict(run=r.split("_")[2], job=r.rsplit("_", 1)[1],
                v=json.loads((RUNS / r / "analysis.json").read_text())["split_half_reliability"]["V*"],
                a=json.loads((RUNS / r / "analysis.json").read_text())["split_half_reliability"]["A*"]) for r in runs]
    e1 = json.loads((RUNS / "branch_value_libero_10_18355241/analysis.json").read_text())["split_half_reliability"]
    pooled = dict(v=corr(d.v_a.values, d.v_b.values), a=corr(d.A_g_a.values, d.A_g_b.values),
                  label_corr=corr(d.pos_resnet_valuebinned_p40.astype(float).values, d.A_g.values),
                  pos_when_up=round(float(d[d.A_g > 0].pos_resnet_valuebinned_p40.mean()), 3),
                  pos_when_down=round(float(d[d.A_g < 0].pos_resnet_valuebinned_p40.mean()), 3),
                  oofV_vs_Vstar=corr(d.V.values, d.v.values), points=int(len(d)), episodes=int(d.groupby("run").ep.nunique().sum()))
    eps = []
    for r, ep in [("branch_fail_c_libero_10_18395414", 7), ("branch_fail_b_libero_10_18395413", 183),
                  ("branch_fail_a_libero_10_18395412", 349), ("branch_fail_b_libero_10_18395413", 76)]:
        x = d[(d.run == r) & (d.ep == ep)].sort_values("t")
        eps.append(dict(ep=int(ep), job=r.rsplit("_", 1)[1], t=x.t.astype(int).tolist(), v=x.v.round(3).tolist(),
                        pos=[None if pd.isna(p) else int(p) for p in x.pos_resnet_valuebinned_p40]))
    return dict(reliability=rel, e1=dict(v=e1["V*"], a=e1["A*"]), pooled=pooled, episodes=eps)


def main():
    data = dict(policy=policy_rows(), tag=tag_rows(), coach=coach_rows(), curve=curve_rows(), branch=branch_rows())
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print("wrote", OUT)
    for r in data["policy"]["rows"]:
        print(f"{r['label']:34s} {r['success']:.3f} {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] n {r['n']}")


if __name__ == "__main__":
    main()
