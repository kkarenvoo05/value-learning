"""When do the plateaued arms fail? Pot timing from the eval videos [2026-11-14; Karen: "start with the video review and
the first-pot timing on AG round 4 and DSRL agent_1300"].

    .venv/bin/python scripts/libero_failure_timing.py --out reports/2026-11-14-failure-timing

Every failure of the plateaued arms is a 520-step timeout. Question: are they slow (still progressing at the cap,
first pot placed late) or stuck (first pot placed on time, then no second pot)? Evals save only the agentview video,
so pot state per frame comes from the pot detector of scripts/libero_failure_taxonomy.py (same training frames,
same MLP, same smoothing), on ResNet-18 agentview features of every 4th video frame
(libero_failure_curves.episode_states, cached in --cache-dir as curves_<name>).
Check [ours]: the detector's end state (last 2 s) vs the simulator's goals_end, per arm.
Per episode: t_first_pot, t_two_pots, pots_end, pots_max, drops (smoothed pot count falls by >= 1), and motion
[ours] = mean |pixel diff| between consecutive sampled frames over the last 100 steps (stuck vs active at the cap).
Writes <out>/{episodes.csv, summary.json, report.md, timing.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_failure_curves import episode_states, video_frames  # noqa: E402
from libero_failure_taxonomy import smooth, train_pots, training_set  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
ARMS = {"base": RUNS / "eval_base_noise1_libero_10_18355478",
        "residual": RUNS / "eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187",
        "agr_r4": RUNS / "eval_agr_r4_s0_shift099_noise1_libero_10_18463162",
        "dsrl1300": RUNS / "eval_dsrl_s0_agent1300_libero_10_18500794"}


def per_episode(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("frame_index")
    t, pots = g.frame_index.to_numpy(), smooth(g.pots.to_numpy())
    r = np.round(pots)
    on = r >= 1
    return pd.Series({"pots_end": float(np.round(pots[-5:].mean())), "pots_max": float(r.max()),
                      "t_first_pot": float(t[np.argmax(on)]) if on.any() else np.nan,
                      "t_two_pots": float(t[np.argmax(r >= 2)]) if (r >= 2).any() else np.nan,
                      "drops": int((np.diff(r) < 0).sum())})


def motion(path: Path, last: int = 100) -> float:
    idx, fr = video_frames(path, every=4)
    fr = fr[idx >= idx[-1] - last].astype(np.float32)
    return float(np.abs(np.diff(fr, axis=0)).mean()) if len(fr) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="/iris/u/karenvo/iql_cache")
    a = ap.parse_args()
    out, cd = Path(a.out), Path(a.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y, _ = training_set(cd)
    f = train_pots(X, Y, np.arange(len(X)), dev)  # as the taxonomy's final detector

    E = []
    for name, run in ARMS.items():
        rows, S = episode_states(name, run, dev, cd)
        p = f(S[:, :512])
        rows = rows.assign(pots=(p > 0.5).sum(1))
        e = rows.groupby("episode").apply(per_episode).reset_index()
        eps = pd.read_csv(run / "episodes.csv")
        e = eps[["episode", "init_state", "repeat", "success", "policy_steps", "goals_end", "video"]].merge(e, on="episode")
        e["arm"] = name
        E.append(e)
    E = pd.concat(E, ignore_index=True)
    ge = E.goals_end.astype(float)  # float in some CSVs (011 -> 11.0); NaN = not recorded
    E["true_end"] = ge.map(lambda v: np.nan if np.isnan(v) else f"{int(v):03d}"[:2].count("1"))
    E["motion_end"] = [motion(ARMS[r.arm] / "videos" / r.video) if not r.success else np.nan for r in E.itertuples()]
    E.drop(columns="video").to_csv(out / "episodes.csv", index=False)

    S = {}
    for name, e in E.groupby("arm", sort=False):
        s, fl = e[e.success == 1], e[e.success == 0]
        k = fl.true_end.notna()  # failures only: a success ends the step the second pot lands, so its last 2 s show 1
        one = fl[fl.true_end == 1]
        gap90 = float((s.t_two_pots - s.t_first_pot).quantile(0.9))
        S[name] = {
            "n": int(len(e)), "success": round(float(e.success.mean()), 3), "failures": int(len(fl)),
            "detector_fail_end_acc": round(float((fl[k].pots_end == fl[k].true_end).mean()), 3) if k.any() else None,
            "succ_gap_p90": gap90,
            "one_pot_fail_had_p90_gap": round(float((one.t_first_pot <= 520 - gap90).mean()), 3) if len(one) else None,
            "no_pot_fail_never_placed": int((fl[fl.true_end == 0].pots_max < 1).sum()),
            "fail_end_pots_sim": fl.true_end.value_counts().sort_index().to_dict(),
            "succ_t_first_pot_median": float(s.t_first_pot.median()),
            "succ_t_two_pots_median": float(s.t_two_pots.median()),
            "succ_gap_median": float((s.t_two_pots - s.t_first_pot).median()),
            "succ_steps_p90": float(s.policy_steps.quantile(0.9)),
            "fail_t_first_pot_median": float(fl.t_first_pot.median()),
            "fail_first_pot_after_succ_p90": round(float((fl.t_first_pot > s.t_first_pot.quantile(0.9)).mean()), 3),
            "fail_no_first_pot": int(fl.t_first_pot.isna().sum()),
            "one_pot_fail_first_pot_median": float(one.t_first_pot.median()) if len(one) else None,
            "one_pot_fail_time_after_first_median": float((520 - one.t_first_pot).median()) if len(one) else None,
            "fail_ever_two_pots": int(fl.t_two_pots.notna().sum()),
            "fail_with_drop": int((fl.drops > 0).sum()),
            "fail_motion_end_median": round(float(fl.motion_end.median()), 2),
        }
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))

    fig, ax = plt.subplots(1, len(ARMS), figsize=(4 * len(ARMS), 3.2), sharey=True)
    for i, (name, e) in enumerate(E.groupby("arm", sort=False)):
        b = np.arange(0, 521, 20)
        ax[i].hist(e[e.success == 1].t_first_pot.dropna(), b, alpha=.6, label="success", color="#4c9a63")
        ax[i].hist(e[e.success == 0].t_first_pot.dropna(), b, alpha=.6, label="failure", color="#c8553d")
        ax[i].set_title(f"{name} ({S[name]['success']:.3f})"), ax[i].set_xlabel("step first pot on")
    ax[0].legend()
    fig.tight_layout(), fig.savefig(out / "timing.png", dpi=120)
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
