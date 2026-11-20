"""Task 64 base rollouts: where base pi05 fails [2026-11-17; Karen: "can you add some of the new task rollouts to the
page? what are the majority of its failures?"].

Task 64 = libero_90 "stack the right bowl on the left bowl and place them in the tray". BDDL goal:
(On akita_black_bowl_2 akita_black_bowl_1) and (In akita_black_bowl_1 tray) -> goals digits (stacked, base bowl in
tray). Base pi05 puts the left bowl (bowl_1, the base) in the tray first, then stacks the right bowl (bowl_2) on it.
Data: the 120 base rollouts on fresh layouts that seed the task-64 chains (t64_base_A / B = 50 + 50 train, V = 20 val),
520 steps, raw npz with object positions.

Per failure (object_pos, goals):
  wrong bowl to tray   bowl_1 never in the tray before bowl_2 was carried off (bowl_2 lifted with no tray goal and ends
                       > 15 cm from bowl_1), or bowl_1 ends below bowl_2's height -> stacked in reverse
  first bowl pushed out  tray goal true at some point, false at the end
  second bowl tilted on the rim  base bowl in tray, bowl_2 lifted and released within 15 cm of bowl_1, not seated
  late grasp, timeout  bowl_2 still lifted (> 5 cm) at step 520
  second bowl never lifted / dropped away
Out: reports/2026-11-17-ag-rounds/{fig7_t64.png, t64/*.mp4, t64/t64.json}
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
from libero_week9_gallery import side_by_side  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-11-17-ag-rounds"
SETS = {"A": "t64_base_A_libero_90_18549968", "B": "t64_base_B_libero_90_18549969", "V": "t64_base_V_libero_90_18550715"}
C = {"blue": "#3b6fd8", "orange": "#e07b39", "grey": "#8a8f98", "green": "#3a9a5b"}
SECOND = ["second bowl tilted on the rim", "late grasp, timeout", "first bowl pushed out", "second bowl never lifted",
          "second bowl dropped away"]
CLIPS = {  # key: [(set, episode, panel label)]; chosen from the contact sheets
    "t64_second": [("A", 0, "success, step 242"), ("B", 25, "tilted on the rim"), ("B", 11, "late grasp, timeout")],
    "t64_first": [("A", 18, "wrong bowl to tray"), ("B", 16, "stacked in reverse"), ("B", 12, "first bowl pushed out")],
}


def episode(s: str, r: dict) -> dict:
    z = np.load(RUNS / SETS[s] / "raw" / r["raw"], allow_pickle=True)
    n = int(r["policy_steps"])
    g = z["goals"][:n]
    keys = list(z["object_pos_keys"])
    op = z["object_pos"][:n].reshape(n, -1, 3)
    b1, b2 = op[:, keys.index("akita_black_bowl_1_pos")], op[:, keys.index("akita_black_bowl_2_pos")]
    up = (b2[:, 2] - b2[0, 2]) > 0.05
    t_tr = int(np.argmax(g[:, 1])) if g[:, 1].any() else -1
    after = np.where(up[max(t_tr, 0):])[0]
    o = {"set": s, "ep": int(r["episode"]), "success": int(r["success"]), "steps": n, "video": r["video"],
         "end": "11" if int(r["success"]) else r["goals_end"], "t_tray": t_tr,
         "t_lift2": int(after[0] + max(t_tr, 0)) if len(after) else -1,
         "t_stack": int(np.argmax(g[:, 0])) if g[:, 0].any() else -1, "lifted2": bool(up.any()),
         "held_end": bool(up[-1]), "xy": float(np.linalg.norm(b2[-1, :2] - b1[-1, :2])), "dz": float(b2[-1, 2] - b1[-1, 2])}
    o["mode"] = mode(o)
    return o


def mode(o: dict) -> str:
    if o["success"]:
        return "success"
    if o["t_tray"] == -1 and o["lifted2"] and o["xy"] > 0.15:
        return "wrong bowl to tray"
    if o["end"] == "01" and o["dz"] < -0.005 and o["t_lift2"] == -1:
        return "wrong bowl to tray"  # bowl_2 went to the tray first, bowl_1 stacked on top of it
    if o["t_tray"] >= 0 and o["end"][1] == "0":
        return "first bowl pushed out"
    if not o["lifted2"]:
        return "second bowl never lifted"
    if o["held_end"]:
        return "late grasp, timeout"
    if o["xy"] > 0.15:
        return "second bowl dropped away"
    return "second bowl tilted on the rim"


def figure(eps: list[dict]) -> None:
    fail = [e for e in eps if not e["success"]]
    cnt = collections.Counter(e["mode"] for e in fail)
    order = [m for m in SECOND if cnt[m]] + ["wrong bowl to tray"] + [m for m in cnt if m not in SECOND + ["wrong bowl to tray"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(order))[::-1]
    cols = [C["blue"] if m in SECOND else C["orange"] for m in order]
    a.barh(y, [cnt[m] for m in order], color=cols, height=0.62)
    for yi, m in zip(y, order):
        a.text(cnt[m] + 0.4, yi, f"{cnt[m]}", va="center", fontsize=9)
    a.set_yticks(y, order, fontsize=9)
    a.set_xlabel(f"failed episodes (of {len(fail)})")
    a.set_title(f"Base π₀.₅ on task 64: {sum(e['success'] for e in eps)}/{len(eps)} succeed; how the rest fail", fontsize=10, loc="left")
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["blue"]), plt.Rectangle((0, 0), 1, 1, color=C["orange"])],
             labels=["after the first bowl is in the tray", "first stage"], fontsize=8, frameon=False, loc="center right")
    a.set_xlim(0, max(cnt.values()) + 4)
    # pause between "first bowl in tray" and "second bowl lifted"
    ok = [e["t_lift2"] - e["t_tray"] for e in eps if e["success"] and e["t_lift2"] > 0]
    bad = [e["t_lift2"] - e["t_tray"] for e in fail if e["t_tray"] >= 0 and e["t_lift2"] > 0 and e["mode"] in SECOND]
    bp = b.boxplot([ok, bad], orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False)
    for el, colr in zip(bp["boxes"], [C["green"], C["blue"]]):
        el.set(facecolor=colr, alpha=0.35, edgecolor=colr)
    for el in bp["medians"]:
        el.set(color="#222")
    rng = np.random.default_rng(0)
    for i, (v, colr) in enumerate([(ok, C["green"]), (bad, C["blue"])]):
        b.scatter(v, i + 1 + rng.uniform(-0.15, 0.15, len(v)), s=10, color=colr, alpha=0.7, lw=0)
    b.set_yticks([1, 2], [f"success (n={len(ok)})", f"second-stage failure (n={len(bad)})"], fontsize=9)
    b.set_xlabel("steps, first bowl in tray → second bowl lifted")
    b.set_title(f"Pause before the second bowl (median {int(np.median(ok))} vs {int(np.median(bad))})", fontsize=10, loc="left")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#ddd", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_t64.png", dpi=150, bbox_inches="tight")


def main() -> None:
    eps = [episode(s, r) for s in SETS for r in csv.DictReader(open(RUNS / SETS[s] / "episodes.csv"))]
    figure(eps)
    (OUT / "t64").mkdir(exist_ok=True)
    for key, panels in CLIPS.items():
        shown = []
        for s, ep, lab in panels:
            e = next(x for x in eps if x["set"] == s and x["ep"] == ep)
            shown.append({"policy": lab, "end": f"{s}{ep}", "video": str(RUNS / SETS[s] / "videos" / e["video"]), "steps": e["steps"]})
        side_by_side(shown, OUT / "t64" / f"{key}.mp4")
    fail = [e for e in eps if not e["success"]]
    summary = {"n": len(eps), "success": sum(e["success"] for e in eps), "by_set": {s: [sum(e["success"] for e in eps if e["set"] == s), sum(e["set"] == s for e in eps)] for s in SETS},
               "modes": collections.Counter(e["mode"] for e in fail), "median_success": {k: float(np.median([e[k] for e in eps if e["success"]])) for k in ("t_tray", "t_lift2", "t_stack")}}
    print(json.dumps(summary, indent=1))
    (OUT / "t64" / "t64.json").write_text(json.dumps({"summary": summary, "episodes": eps}, indent=1))


if __name__ == "__main__":
    main()
