"""Chart data for the "Since Nov 10" section of the week-9 page [2026-11-11] -> reports/site/content/week09_since_data.json.

Recomputed from the run directories, so the page and its numbers agree (rerun before every build; partial evals are
flagged, not dropped):
  ag        contrastive guidance v_ref + w (v+ - v-) (Advantage Guidance, Zhou & Levine) on the AdaRMS vb40 run and the
            residual run, plus the shuffled-label control: success vs base 18355478, paired by (init state, repeat),
            5000 bootstrap draws (as site_week09_data.py)
  shift     offline contrast checks: action shift / sampling noise per w (AdaRMS vb40, residual)
  followup  the auto-queued follow-ups on the "demos 75%, low lr" recipe (seed 1, tag +/-, success-only weights), paired
  dsrl      DSRL online (18434524): training success per cycle (each of the 30 train states once), by agent life (requeue restarted the agent); base on the
            same 30 train states from the round-1 rollouts (stage2_moka_18155849, 450 episodes)
  round2    filtered-BC round-2 collection (round-1 policy) vs base on the same states / layouts (unpaired)
  noise     exploration-noise pilot (reports/2026-11-09-noise-sweep/pilot_table.csv)
  vgt       learned values / labels vs simulator ground truth (reports/2026-11-10-value-vs-gt/summary.json)
Usage: .venv/bin/python scripts/site_since_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/episodes.csv")
OUT = ROOT / "reports/site/content/week09_since_data.json"
BASE = "eval_base_noise1_libero_10_18355478"

# (eval dir glob, label, group, note)
AG = [
    ("eval_adarms_vb40_none_noise1_libero_10_18409259", "No tag (w 0)", "adarms", "the checkpoint's own tag-dropout branch = v_ref"),
    ("eval_armC_adarms_vb40_pos_noise1_libero_10_18355679", "Plain + tag (RECAP-style)", "adarms", "v(+) only, no contrast"),
    ("eval_adarms_vb40_cg2_noise1_libero_10_18409260", "w 2", "adarms", "action shift 0.30 x noise"),
    ("eval_adarms_vb40_cg4_noise1_libero_10_18409262", "w 4", "adarms", "action shift 0.57 x noise"),
    ("eval_adarms_vb40_cg8_noise1_libero_10_18432563", "w 8", "adarms", "action shift 0.99 x noise"),
    ("eval_adarms_vb40_cg16_noise1_libero_10_*", "w 16", "adarms", "action shift ~2 x noise (job 18445301)"),
    ("eval_adarms_vb40_cgm4_noise1_libero_10_18409263", "w -4 (sign control)", "adarms", "guidance pointed the wrong way"),
    ("eval_ag_residual_vb40_s0_shift099_noise1_libero_10_*", "Residual, shift of AdaRMS w 8", "residual",
     "frozen pi0.5 + zero-init AdaLN head; w 9.26 gives shift 0.99 x noise"),
    ("eval_ag_residual_vb40_s0_cg16_noise1_libero_10_*", "Residual, w 16", "residual", "shift 1.64 x noise (job 18452438)"),
    ("eval_ag_residual_vb40_s0_shift057_noise1_libero_10_*", "Residual, shift of AdaRMS w 4", "residual", "shift 0.57 x noise"),
    ("eval_*shufvb40*_noise1_libero_10_*", "Shuffled labels, w 8", "control",
     "same recipe, vb40 labels permuted across episodes (demos still +)"),
]
DSRL_EVAL = [("eval_dsrl_s0_ep971_libero_10_*", "DSRL agent, episode 971", "dsrl",
              "frozen pi0.5, SAC actor picks the start noise (stochastic); same 330 held-out trials as base")]
FOLLOW = [
    ("eval_grid_d75_low_s0_at2000_noise1_libero_10_18396176", "Recipe, seed 0, step 2000 (page result)", "untagged", "demos 75%, 10x lower lr"),
    ("eval_fu_d75_low_seed1_s1_at2000_none_noise1_libero_10_18406244", "Recipe, seed 1, step 2000", "untagged", "same recipe, training seed 1"),
    ("eval_fu_d75_low_seed1_s1_atfinal_none_noise1_libero_10_18406245", "Recipe, seed 1, final", "untagged", "step 5000"),
    ("eval_fu_d75_low_tag_s0_at2000_positive_noise1_libero_10_18406247", "+ tag (dropout 0.3), sampled +, step 2000", "tag", "prompt tag"),
    ("eval_fu_d75_low_tag_s0_at2000_negative_noise1_libero_10_18406249", "same model, sampled −, step 2000", "tag",
     "the − tag should be worse if the tag were used"),
    ("eval_fu_d75_low_tag_s0_atfinal_positive_noise1_libero_10_18406248", "+ tag, sampled +, final", "tag", "step 5000"),
    ("eval_fu_d75_low_wsucc_s0_at2000_none_noise1_libero_10_18406251", "Success-only weights, step 2000", "weights",
     "516 successes weighted by exp(advantage), demos weight 1"),
    ("eval_fu_d75_low_wsucc_s0_atfinal_none_noise1_libero_10_18406252", "Success-only weights, final", "weights", "step 5000"),
    ("eval_armA_r1_noise1_libero_10_18376075", "Filtered BC round 1 (reference)", "untagged", "516 successes, frozen vision encoder"),
]


def load(d):
    return pd.read_csv(RUNS / d / "episodes.csv").set_index(["init_state", "repeat"]).success


def find(pattern):
    hits = sorted(p for p in RUNS.glob(pattern) if (p / "episodes.csv").exists())
    return hits[-1].name if hits else None


def paired(spec):
    base = load(BASE)
    rng = np.random.default_rng(0)
    rows = []
    for pat, label, group, note in spec:
        d = find(pat)
        if d is None:
            rows.append(dict(label=label, group=group, note=note, pending=True))
            continue
        a = load(d)
        if len(a) == 0:
            rows.append(dict(label=label, group=group, note=note, pending=True))
            continue
        j = a.index.intersection(base.index)
        x = a[j].values - base[j].values
        bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        g = pd.Series(x, index=j).groupby(level=0).mean()  # per init state
        rows.append(dict(label=label, group=group, note=note, job=d.rsplit("_", 1)[1], n=int(len(a)),
                         success=round(float(a.mean()), 4), diff=round(float(x.mean()), 4), lo=round(float(lo), 4),
                         hi=round(float(hi), 4), complete=bool(len(a) == 330), states_up=int((g > 0).sum()),
                         states_down=int((g < 0).sum()), pending=False))
    return dict(base=round(float(base.mean()), 4), rows=rows)


def residual_vs_adarms():
    r, a = find("eval_ag_residual_vb40_s0_shift099_noise1_libero_10_*"), "eval_adarms_vb40_cg8_noise1_libero_10_18432563"
    if r is None:
        return None
    x, y = load(r), load(a)
    j = x.index.intersection(y.index)
    d = x[j].values - y[j].values
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=int(len(j)), residual=round(float(x[j].mean()), 3), adarms=round(float(y[j].mean()), 3),
                diff=round(float(d.mean()), 3), lo=round(float(lo), 3), hi=round(float(hi), 3))


def shift():
    out = {}
    for key, d in [("AdaRMS vb40", "contrast_check_adarms_vb40"), ("Residual", "contrast_check_ag_residual_vb40_s0")]:
        p = RUNS / d / "results.json"
        if not p.exists():
            continue
        w = json.loads(p.read_text())["weights"]
        out[key] = [dict(w=float(k), ratio=round(v["norm"]["ratio"], 3), flip=round(v["gripper_flip"], 4))
                    for k, v in sorted(w.items(), key=lambda kv: float(kv[0])) if float(k) > 0]
    return out


def dsrl():
    d = RUNS / "dsrl_s0_libero_10_18434524"
    e = pd.read_csv(d / "episodes.csv")
    restart = 443  # episodes kept from agent life 1 when the requeued job started a fresh agent (slurm-18434524.out)
    e["life"] = np.where(e.index < restart, 1, 2)
    cycles = []  # one cycle = one repeat = each of the 30 train states once, so cycles are balanced across states
    for (life, rep), sub in e.groupby(["life", "repeat"]):
        if len(sub) >= 20:
            cycles.append(dict(life=int(life), cycle=int(rep), x=int(sub.index[-1]) + 1, y=round(float(sub.success.mean()), 3),
                               n=int(len(sub))))
    last3 = e[e.life == 2].groupby("repeat").filter(lambda s: len(s) >= 20)
    last3 = last3[last3.repeat.isin(sorted(last3.repeat.unique())[-3:])]
    t = pd.read_csv(TABLE)
    base = t[(t.split == "train") & (t.layout == "fixed")]
    log = [json.loads(x) for x in open(d / "dsrl_log.jsonl")]
    return dict(cycles=cycles, last3=round(float(last3.success.mean()), 3), last3_n=int(len(last3)), base_train=round(float(base.success.mean()), 3), base_n=int(len(base)), episodes=int(len(e)),
                restart=restart, life1=round(float(e.success[:restart].mean()), 3), life2=round(float(e.success[restart:].mean()), 3),
                updates=int(log[-1].get("updates", 0)))


def round2():
    t = pd.read_csv(TABLE)
    t = t[t.split == "train"]
    out = []
    for layout, jobs in [("fixed", ["fbc_r2_collect_fixed_a_libero_10_18433688", "fbc_r2_collect_fixed_b_libero_10_18433689"]),
                         ("fresh", ["fbc_r2_collect_fresh_a_libero_10_18433690", "fbc_r2_collect_fresh_b_libero_10_18433691"])]:
        r = pd.concat([pd.read_csv(RUNS / j / "episodes.csv") for j in jobs])
        b = t[t.layout == layout]
        out.append(dict(layout=layout, r1=round(float(r.success.mean()), 3), n=int(len(r)), base=round(float(b.success.mean()), 3),
                        base_n=int(len(b))))
    return out


def noise():
    p = pd.read_csv(ROOT / "reports/2026-11-09-noise-sweep/pilot_table.csv")
    return [dict(arm=r.arm, n=int(r.episodes), success=round(float(r.success), 3), spread=None if pd.isna(r.spread_ratio_arm)
                 else round(float(r.spread_ratio_arm), 1)) for r in p.itertuples()]


def vgt():
    s = json.loads((ROOT / "reports/2026-11-10-value-vs-gt/summary.json").read_text())
    rows = []
    for r in s["sources"]:
        a = r.get("corr_A_A*") or {}
        lab = r.get("label_corr_A*") or {}
        rows.append(dict(source=r["desc"], a=a.get("est"), a_ci=a.get("ci95"), label=lab.get("est"), label_ci=lab.get("ci95")))
    c = s["ceiling"]["A*"]
    return dict(rows=rows, ceiling=round(c["ceiling_sqrtR"], 3), split_half=round(c["split_half_r"], 3), points=s["points_with_A*"])


def main():
    out = dict(ag=paired(AG), res_vs_ada=residual_vs_adarms(), shift=shift(), followup=paired(FOLLOW), dsrl=dsrl(), dsrl_eval=paired(DSRL_EVAL)['rows'][0], round2=round2(), noise=noise(), vgt=vgt())
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for r in out["ag"]["rows"] + out["followup"]["rows"] + [out["dsrl_eval"]]:
        print(f'{r["label"][:48]:48s}', "pending" if r["pending"] else f'{r["success"]:.3f} {r["diff"]:+.3f} [{r["lo"]:+.3f}, {r["hi"]:+.3f}] n {r["n"]}')
    print("->", OUT)


if __name__ == "__main__":
    main()
