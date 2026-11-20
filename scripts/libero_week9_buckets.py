"""Start-state buckets: which start states stay hard, and what makes them hard [2026-11-17; meeting next step "Bucket
initial states and analyze persistent failure cases"].

Buckets:
  saved states  the 40 LIBERO saved init states used so far (10 val, 30 train). AG pools every AG-round trial on that
                state: val = r0-r7 evals (8 x 33 = 264 per state); train = r0 / r7 train-state evals (10 each) + the
                fixed-state collections of rounds 1-7 (2 per state each). Base = base eval / train-state eval / the
                base collection.
  fresh layouts the 100 continuous-eval layouts (moka_cont100, 3 repeats each), run by the 5 round-0 chain policies
                (b50cv, b100cv, b200cv, b100, b100iql) -> 15 trials per layout.
Start features from sim_init (MuJoCo state: [time, qpos 24, qvel 22]; qpos = 7 arm + 2 gripper joints, moka_pot_1
xyz + quat, moka_pot_2 xyz + quat, 1 scene joint): pot xy and the arm's start-joint offset from the default pose
(fresh layouts all start from the default pose; saved states have LIBERO's joint noise).
Over-dispersion: variance of the per-bucket success rates against the binomial variance if every bucket had the
pooled rate (the "is the bucket real or just chance" check).
Out: reports/2026-11-17-ag-rounds/{fig10_buckets.png, buckets/buckets.json}
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from libero_week9_gallery import EVALS, RUNS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-11-17-ag-rounds"
ROOT = RUNS.parent
TRAIN_STATE = {"base": "ps_base_train_noise1_libero_10_18541680", "r0": "ps_r0_train_shift099_noise1_libero_10_18541681",
               "r7": "ps_r7_train_shift099_noise1_libero_10_18541682"}
COLLECT_FIXED = ["agr_r1_collect_fixed_libero_10_18460302", "agr_r2_collect_fixed_libero_10_18460308",
                 "agr_r3_collect_fixed_libero_10_18460314", "agr_r4_collect_fixed_libero_10_18463157",
                 "agr_r5_collect_fixed_libero_10_18484602", "agr_r6_collect_fixed_libero_10_18509032",
                 "agr_r7_collect_fixed_libero_10_18524598"]
BASE_COLLECT = "distill_collect_base_libero_10_18507468"
CEVAL = {"b50cv": "ceval_agr_b50cv_r0_s0_shift099_libero_10_18546934", "b100cv": "ceval_agr_b100cv_r0_s0_shift099_libero_10_18546939",
         "b200cv": "ceval_agr_b200cv_r0_s0_shift099_libero_10_18546944", "b100": "ceval_agr_b100_r0_s0_shift099_libero_10_18544461",
         "b100iql": "ceval_agr_b100iql_r0_s0_shift099_libero_10_18545166"}
VAL = [9, 12, 13, 17, 25, 32, 40, 42, 44, 48]
DEFAULT_ARM = np.array([0, -0.161, 0, -2.445, 0, 2.227, 0.785])
CORNER = (-0.045, 0.24)  # pot 1 x > -0.045 and y < 0.24: the corner the hardest saved states share
C = {"blue": "#3b6fd8", "orange": "#e07b39", "grey": "#8a8f98", "green": "#3a9a5b"}


def rows(d: str):
    return csv.DictReader(open(RUNS / d / "episodes.csv"))


def feats(sim: np.ndarray) -> dict:
    q = sim[1:25]
    return {"p1x": float(q[9]), "p1y": float(q[10]), "p2x": float(q[16]), "p2y": float(q[17]),
            "arm_offset": float(np.linalg.norm(q[:7] - DEFAULT_ARM))}


def saved_states() -> dict:
    ag, base, sim = collections.defaultdict(list), collections.defaultdict(list), {}
    for k, d in EVALS.items():
        for r in rows(d):
            (base if k == "base" else ag)[int(r["init_state"])].append(int(float(r["success"])))
    for k, d in TRAIN_STATE.items():
        for r in rows(d):
            (base if k == "base" else ag)[int(r["init_state"])].append(int(float(r["success"])))
    for d in COLLECT_FIXED + [BASE_COLLECT]:
        for r in rows(d):
            s = int(r["init_state"])
            (base if d == BASE_COLLECT else ag)[s].append(int(float(r["success"])))
            if s not in sim and r.get("raw"):
                sim[s] = np.load(RUNS / d / "raw" / r["raw"], allow_pickle=True)["sim_init"]
    for f in sorted(RUNS.glob("*_libero_10_*/episodes.csv")):  # val states: any moka run that saved raw npz
        if all(s in sim for s in ag):
            break
        for r in csv.DictReader(open(f)):
            try:
                s = int(r["init_state"])
            except (KeyError, ValueError):
                break
            if r.get("task_id") == "8" and s in ag and s not in sim and r.get("raw") and (f.parent / "raw" / r["raw"]).exists():
                sim[s] = np.load(f.parent / "raw" / r["raw"], allow_pickle=True)["sim_init"]
    return {s: {"ag": ag[s], "base": base[s], "val": s in VAL, **feats(sim[s])} for s in sorted(ag) if s in sim}


def fresh_layouts() -> dict:
    out = {}
    for k, d in CEVAL.items():
        for r in rows(d):
            src = r["source_raw"]
            if src not in out:
                out[src] = {"trials": [], "by_policy": collections.defaultdict(list),
                            **feats(np.load(src, allow_pickle=True)["sim_init"])}
            out[src]["trials"].append(int(float(r["success"])))
            out[src]["by_policy"][k].append(int(float(r["success"])))
    return out


def dispersion(groups: list[list[int]]) -> dict:
    p = np.array([np.mean(g) for g in groups])
    n = np.array([len(g) for g in groups])
    m = np.concatenate(groups).mean()
    return {"var_obs": float(np.var(p)), "var_binomial": float(np.mean(m * (1 - m) / n)), "ratio": float(np.var(p) / np.mean(m * (1 - m) / n))}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> None:
    (OUT / "buckets").mkdir(exist_ok=True)
    S = saved_states()
    L = fresh_layouts()
    F = ["p1x", "p1y", "p2x", "p2y", "arm_offset"]
    st = list(S)
    y_ag = np.array([np.mean(S[s]["ag"]) for s in st])
    y_b = np.array([np.mean(S[s]["base"]) for s in st])
    lay = list(L)
    y_l = np.array([np.mean(L[k]["trials"]) for k in lay])
    corr = {"saved_ag": {f: float(np.corrcoef([S[s][f] for s in st], y_ag)[0, 1]) for f in F},
            "saved_base": {f: float(np.corrcoef([S[s][f] for s in st], y_b)[0, 1]) for f in F},
            "fresh": {f: float(np.corrcoef([L[k][f] for k in lay], y_l)[0, 1]) for f in F[:4]}}
    val = [s for s in st if S[s]["val"]]
    disp = {"val_states_ag_r0_r7": dispersion([S[s]["ag"] for s in val]),
            "val_states_r5": dispersion([[int(float(r["success"])) for r in rows(EVALS["r5"]) if int(r["init_state"]) == s] for s in val]),
            "val_states_r7": dispersion([[int(float(r["success"])) for r in rows(EVALS["r7"]) if int(r["init_state"]) == s] for s in val]),
            "fresh_layouts_pooled": dispersion([L[k]["trials"] for k in lay]),
            **{f"fresh_layouts_{p}": dispersion([L[k]["by_policy"][p] for k in lay]) for p in CEVAL}}
    # figure
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    o = np.argsort(y_ag)
    xs = np.arange(len(st))
    for i, j in enumerate(o):
        s = st[j]
        lo, hi = wilson(sum(S[s]["ag"]), len(S[s]["ag"]))
        col = C["blue"] if S[s]["val"] else C["green"]
        a.plot([i, i], [100 * lo, 100 * hi], color=col, alpha=0.35, lw=3, solid_capstyle="round")
        a.plot(i, 100 * y_ag[j], "o", color=col, ms=5)
        a.plot(i, 100 * y_b[j], "_", color="#333", ms=8, mew=1.6)
    a.set_xticks(xs, [str(st[j]) for j in o], fontsize=7, rotation=90)
    a.set_xlabel("saved start state (sorted by AG success)")
    a.set_ylabel("success (%)")
    a.set_ylim(-3, 103)
    a.axhline(50, color="#bbb", lw=0.8, ls=":")
    a.legend(handles=[plt.Line2D([], [], color=C["blue"], marker="o", ls="", label="val state, AG r0–r7 (264 trials)"),
                      plt.Line2D([], [], color=C["green"], marker="o", ls="", label="train state, AG rounds (34 trials)"),
                      plt.Line2D([], [], color="#333", marker="_", ls="", ms=8, mew=1.6, label="base")],
             fontsize=8, frameon=False, loc="lower right")
    a.set_title("Every saved start state, pooled over the AG rounds (bars: 95% interval)", fontsize=10, loc="left")
    # b: fresh layouts: histogram of per-layout success vs binomial null
    rng = np.random.default_rng(0)
    m = np.concatenate([L[k]["trials"] for k in lay]).mean()
    null = rng.binomial(15, m, size=(4000, len(lay))) / 15
    bins = np.arange(-0.5, 16.5) / 15
    b.hist(y_l, bins=bins, color=C["orange"], alpha=0.8, label="100 fresh layouts")
    h = np.array([np.histogram(n, bins=bins)[0] for n in null])
    b.step(bins[:-1] + 0.5 / 15, h.mean(0), where="mid", color="#333", lw=1.4, label=f"chance only (every layout {100 * m:.0f}%)")
    b.set_xlabel("layout success over 15 trials (5 policies x 3)")
    b.set_ylabel("layouts")
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    b.set_ylim(0, 29)
    b.legend(fontsize=8, frameon=False, loc="upper left")
    b.set_title("Fresh layouts: spread is what chance gives", fontsize=10, loc="left")
    # c: where moka pot 1 (the second pot picked) starts, colored by failure rate; the hard saved states share a corner
    cm = plt.get_cmap("Oranges")
    norm = matplotlib.colors.Normalize(0, 0.6)
    c.scatter([L[k]["p1x"] for k in lay], [L[k]["p1y"] for k in lay], c=[1 - v for v in y_l], cmap=cm, norm=norm, s=16,
              marker="s", alpha=0.75, lw=0, label="fresh layout (15 trials)")
    sc = c.scatter([S[s]["p1x"] for s in st], [S[s]["p1y"] for s in st], c=1 - y_ag, cmap=cm, norm=norm, s=70,
                   edgecolors="#222", linewidths=0.8, label="saved state (AG rounds)")
    for s, yy in zip(st, y_ag):
        if yy < 0.6:
            c.annotate(str(s), (S[s]["p1x"], S[s]["p1y"]), textcoords="offset points", xytext=(5, 3), fontsize=7.5, color="#333")
    c.add_patch(plt.Rectangle((CORNER[0], 0.2245), -0.0245 - CORNER[0], CORNER[1] - 0.2245, fill=False, ls="--", lw=1, ec="#555"))
    cb = fig.colorbar(sc, ax=c, fraction=0.05, pad=0.02)
    cb.set_label("failure rate")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    c.set_xlabel("moka pot 1 start x (m)")
    c.set_ylabel("moka pot 1 start y (m)")
    c.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0))
    c.set_ylim(0.222, 0.29)
    c.set_title("Where pot 1 starts (dashed: the hard corner)", fontsize=10, loc="left")
    c.grid(axis="x", color="#ddd", lw=0.6)
    for ax in (a, b, c):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#ddd", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_buckets.png", dpi=150, bbox_inches="tight")
    res = {"corr": corr, "dispersion": disp,
           "saved": {s: {"ag": [sum(S[s]["ag"]), len(S[s]["ag"])], "base": [sum(S[s]["base"]), len(S[s]["base"])], "val": S[s]["val"],
                         **{f: S[s][f] for f in F}} for s in st},
           "fresh_lowest": sorted(((float(y_l[i]), Path(lay[i]).stem) for i in range(len(lay))))[:10],
           "fresh_hist": np.histogram(y_l, bins=bins)[0].tolist(), "corner": {}}
    inc = lambda v: v["p1x"] > CORNER[0] and v["p1y"] < CORNER[1]
    for name, src, key in (("saved_ag", S, "ag"), ("saved_base", S, "base"), ("fresh", L, "trials")):
        for side in (True, False):
            t = sum((v[key] for v in src.values() if inc(v) == side), [])
            res["corner"][f"{name}_{'in' if side else 'out'}"] = [sum(t), len(t), sum(inc(v) == side for v in src.values())]
    for p in CEVAL:
        for side in (True, False):
            t = sum((v["by_policy"][p] for v in L.values() if inc(v) == side), [])
            res["corner"][f"fresh_{p}_{'in' if side else 'out'}"] = [sum(t), len(t)]
    (OUT / "buckets" / "buckets.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"corr": corr, "dispersion": disp}, indent=1))
    print("lowest saved", [(st[j], round(100 * y_ag[j]), round(100 * y_b[j]), len(S[st[j]]["ag"])) for j in o[:8]])
    print("corner", res["corner"])
    print("fresh lowest", res["fresh_lowest"][:5], "zero layouts", int((y_l == 0).sum()))


if __name__ == "__main__":
    main()
