"""Method gallery for the Week 9 Nov 17 page [2026-11-17; Karen: "can you add some qualitative rollouts for different
methods and levels of success rates?"].

Six policies on the same 330 held-out moka trials (val states x 33 repeats, env seed 7), weakest to strongest:
AG with shuffled labels (control), RECAP-style tag fine-tune, base pi05, AG round 0, AG round 5, DSRL.
All but DSRL use matched noise seed 1 (same start state and same sampling noise per trial); DSRL picks its own noise,
so it shares only the start state.
Per story: one 3x2 grid mp4 (top: shuffled AG / RECAP / base; bottom: AG r0 / AG r5 / DSRL; a finished episode holds
its last frame; 2x speed) + a contact sheet (every 60 steps) used to write the captions.
Also: per-state success table, base-failure fixes / base-success breaks per method, and the AG r5 x DSRL overlap.
Out: reports/2026-11-17-ag-rounds/methods/{<story>.mp4, <story>_sheet.png, methods.json}
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from libero_week9_gallery import FONT, sheet  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-11-17-ag-rounds" / "methods"
METHODS = {  # key: (label, eval run)
    "shuf": ("AG, shuffled labels", "eval_ag_residual_shufvb40_s0_shift099_noise1_libero_10_18472216"),
    "recap": ("RECAP tag", "eval_armC_recapfull_pos_noise1_libero_10_18376192"),
    "base": ("base", "eval_base_noise1_libero_10_18355478"),
    "ag0": ("AG r0", "eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187"),
    "ag5": ("AG r5", "eval_agr_r5_s0_shift099_noise1_libero_10_18484607"),
    "dsrl": ("DSRL", "eval_dsrl_s0_ep1800_libero_10_18452292"),
}
STORIES = [  # key, (val state, repeat); picked by outcome pattern over (shuf, recap, base, ag0, ag5, dsrl)
    ("m_fix", (42, 12)),       # 000111: base fails, the three strong methods succeed
    ("m_break", (44, 21)),     # 001111: base succeeds, the two weak fine-tunes fail
    ("m_ag_wins", (12, 9)),    # 000110: AG succeeds, DSRL fails
    ("m_dsrl_wins", (13, 18)),  # 000101: DSRL succeeds, AG r5 fails
    ("m_all_fail", (12, 10)),  # 000000: the only trial every method fails
]


def load() -> dict:
    ep = {}
    for k, (_, d) in METHODS.items():
        for r in csv.DictReader(open(RUNS / d / "episodes.csv")):
            ep[k, int(r["init_state"]), int(r["repeat"])] = {
                "success": int(float(r["success"])), "steps": int(r["policy_steps"]),
                "video": str(RUNS / d / "videos" / r["video"])}
    return ep


def grid(eps: list[dict], labels: list[str], path: Path, size: int = 300) -> None:
    n = max(e["steps"] for e in eps) + 30
    ins, filt = [], []
    for i, (e, lab) in enumerate(zip(eps, labels)):
        ins += ["-i", e["video"]]
        text = lab.replace(":", "\\:")
        filt.append(f"[{i}:v]tpad=stop_mode=clone:stop=-1,trim=end_frame={n},scale={size}:{size}:flags=lanczos,"
                    f"drawtext=fontfile={FONT}:text='{text}':x=6:y=6:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=4:expansion=none[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(6)) + "xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0,"
                f"drawtext=fontfile={FONT}:text='step %{{n}}':x=w-110:y=h-26:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=4,"
                "setpts=0.5*PTS,fps=20[out]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", ";".join(filt), "-map", "[out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart", str(path)], check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep = load()
    trials = sorted({(s, r) for (_, s, r) in ep})
    states = sorted({s for s, _ in trials})
    sr = {k: round(100 * sum(ep[k, s, r]["success"] for s, r in trials) / len(trials), 1) for k in METHODS}
    per_state = {s: {k: round(100 * sum(ep[k, s, r]["success"] for r in range(33)) / 33) for k in METHODS} for s in states}
    bf = [t for t in trials if not ep["base", *t]["success"]]
    bs = [t for t in trials if ep["base", *t]["success"]]
    fix = {k: [sum(ep[k, *t]["success"] for t in bf), len(bf), sum(1 - ep[k, *t]["success"] for t in bs), len(bs)] for k in METHODS}
    a = {t: ep["ag5", *t]["success"] for t in trials}
    d = {t: ep["dsrl", *t]["success"] for t in trials}
    overlap = {"both": sum(a[t] & d[t] for t in trials), "ag5_only": sum(a[t] & (1 - d[t]) for t in trials),
               "dsrl_only": sum((1 - a[t]) & d[t] for t in trials), "neither": sum((1 - a[t]) & (1 - d[t]) for t in trials)}
    overlap["neither_if_independent"] = round(sum(1 - a[t] for t in trials) * sum(1 - d[t] for t in trials) / len(trials), 1)
    meta = {"sr": sr, "per_state": per_state, "fix_break": fix, "ag5_x_dsrl": overlap, "stories": {}}
    for key, (s, r) in STORIES:
        eps = [ep[k, s, r] for k in METHODS]
        labels = [f"{METHODS[k][0]} ({sr[k]:.0f}%): {'success' if e['success'] else 'failed'}" for k, e in zip(METHODS, eps)]
        grid(eps, labels, OUT / f"{key}.mp4")
        sheet([{"policy": k, "video": e["video"]} for k, e in zip(METHODS, eps)], OUT / f"{key}_sheet.png", every=60)
        meta["stories"][key] = {"trial": [s, r], "outcome": {k: [e["success"], e["steps"]] for k, e in zip(METHODS, eps)}}
        print(key, (s, r), [(k, e["success"], e["steps"]) for k, e in zip(METHODS, eps)])
    (OUT / "methods.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("sr", "fix_break", "ag5_x_dsrl")}))


if __name__ == "__main__":
    main()
