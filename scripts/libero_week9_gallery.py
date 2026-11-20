"""Matched-trial video gallery for the Week 9 Nov 17 page [2026-11-17; Karen: "a video gallery of an example of: an
episode improving after the rounds, an episode failing after the rounds, an episode staying the same, second pot
improving after the rounds, first pot not improving after the rounds, etc"].

Each story is one held-out trial (val state, repeat): with matched noise seed 1, every policy sees the same start
state and the same sampling noise, so the clips differ only by the policy. Per story: one side-by-side mp4 (3
policies, agentview, upscaled 1.5x, step counter, played at 2x; a finished episode holds its last frame) and a
contact sheet (every 40 steps) used to write the captions.
Out: reports/2026-11-17-ag-rounds/gallery/{<story>.mp4, <story>_sheet.png, gallery.json}
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-11-17-ag-rounds" / "gallery"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
EVALS = {
    "base": "eval_base_noise1_libero_10_18355478",
    "r0": "eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187",
    "r1": "eval_agr_r1_s0_shift099_noise1_libero_10_18460307",
    "r2": "eval_agr_r2_s0_shift099_noise1_libero_10_18460313",
    "r3": "eval_agr_r3_s0_shift099_noise1_libero_10_18460319",
    "r4": "eval_agr_r4_s0_shift099_noise1_libero_10_18463162",
    "r5": "eval_agr_r5_s0_shift099_noise1_libero_10_18484607",
    "r6": "eval_agr_r6_s0_shift099_noise1_libero_10_18509037",
    "r7": "eval_agr_r7_s0_shift099_noise1_libero_10_18524603",
}
STORIES = [  # key, (val state, repeat), policies shown
    ("fixed_r0", (42, 5), ["base", "r0", "r7"]),
    ("second_fixed", (12, 11), ["base", "r2", "r5"]),
    ("traded", (13, 21), ["r2", "r5", "r7"]),
    ("flips", (44, 19), ["r4", "r5", "r6"]),
    ("worse", (17, 9), ["r5", "r6", "r7"]),
    ("hardest", (13, 13), ["base", "r5", "r7"]),
]
END = {"111": "success", "011": "second pot missing", "101": "first pot lost", "001": "no pot on", "": "failed"}


def episode(pol: str, trial: tuple[int, int]) -> dict:
    e = pd.read_csv(RUNS / EVALS[pol] / "episodes.csv", dtype={"goals_end": str})
    r = e[(e.init_state == trial[0]) & (e["repeat"] == trial[1])].iloc[0]
    g = "111" if r.success else (r.goals_end if isinstance(r.goals_end, str) else "")
    return {"policy": pol, "video": str(RUNS / EVALS[pol] / "videos" / r.video), "success": int(r.success),
            "steps": int(r.policy_steps), "end": END.get(g, g)}


def frames(path: str) -> list[np.ndarray]:
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def sheet(eps: list[dict], path: Path, every: int = 40) -> None:
    fs = [frames(e["video"]) for e in eps]
    n = max(len(f) for f in fs)
    ts = list(range(0, n, every)) + [n - 1]
    font = ImageFont.truetype(FONT, 13)
    W, H = 224, 224
    img = Image.new("RGB", (60 + W * len(ts), (H + 4) * len(eps)), "white")
    d = ImageDraw.Draw(img)
    for i, (e, f) in enumerate(zip(eps, fs)):
        d.text((4, i * (H + 4) + 100), e["policy"], fill="black", font=font)
        for j, t in enumerate(ts):
            fr = Image.fromarray(f[min(t, len(f) - 1)])
            img.paste(fr, (60 + j * W, i * (H + 4)))
            d.text((64 + j * W, i * (H + 4) + 4), f"{min(t, len(f) - 1)}", fill="yellow", font=font)
    img.save(path)


def side_by_side(eps: list[dict], path: Path) -> None:
    n = max(e["steps"] for e in eps) + 30
    ins, filt = [], []
    for i, e in enumerate(eps):
        ins += ["-i", e["video"]]
        label = f"{e['policy']}: {e['end']}".replace(":", "\\:")
        filt.append(f"[{i}:v]tpad=stop_mode=clone:stop=-1,trim=end_frame={n},scale=336:336:flags=lanczos,"
                    f"drawtext=fontfile={FONT}:text='{label}':x=8:y=8:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=4[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(len(eps))) + f"hstack=inputs={len(eps)},"
                f"drawtext=fontfile={FONT}:text='step %{{n}}':x=w-120:y=h-28:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=4,"
                "setpts=0.5*PTS,fps=20[out]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", ";".join(filt), "-map", "[out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart", str(path)],
                   check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for key, trial, pols in STORIES:
        eps = [episode(p, trial) for p in pols]
        sheet(eps, OUT / f"{key}_sheet.png")
        side_by_side(eps, OUT / f"{key}.mp4")
        row = {p: ("S" if episode(p, trial)["success"] else episode(p, trial)["end"]) for p in EVALS}
        meta[key] = {"trial": trial, "shown": eps, "all_policies": row}
        print(key, trial, [(e["policy"], e["end"], e["steps"]) for e in eps])
    (OUT / "gallery.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
