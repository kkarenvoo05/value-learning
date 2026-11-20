"""Side-by-side example videos and a page for the failure taxonomy (meeting 11-11: "share videos on Slack").

    .venv/bin/python scripts/libero_failure_media.py --out reports/2026-11-12-failure-taxonomy

Reads <out>/episodes.csv (scripts/libero_failure_taxonomy.py). Paired trials (same init state, repeat and matched
noise seed): base | residual, each with its outcome, type and the pot detector's count burned in. Picks [ours,
deterministic: one trial per val state in turn, first repeat]:
  fixed    base failed, residual succeeded (by base's failure type)
  broken   base succeeded, residual failed
  still    both failed (the residual's remaining failures)
Writes <out>/media/*.mp4 and <out>/index.html (figures + videos + numbers from summary.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
EVALS = {"base": RUNS / "eval_base_noise1_libero_10_18355478",
         "residual": RUNS / "eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187"}


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def caption(img: np.ndarray, lines: list[str], color) -> np.ndarray:
    im = Image.fromarray(img).resize((336, 336))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 336, 16 * len(lines) + 4], fill=(0, 0, 0))
    for i, s in enumerate(lines):
        d.text((4, 2 + 16 * i), s, fill=color if i == 0 else (255, 255, 255))
    return np.asarray(im)


def side_by_side(row_b, row_r, P, out: Path):
    vids = {}
    for k, r in (("base", row_b), ("residual", row_r)):
        ev = pd.read_csv(EVALS[k] / "episodes.csv").set_index("episode")
        vids[k] = (frames(EVALS[k] / "videos" / ev.video[r.episode]), r)
    n = max(len(v[0]) for v in vids.values())
    outf = []
    for t in range(n):
        cols = []
        for k in ("base", "residual"):
            fr, r = vids[k]
            img = fr[min(t, len(fr) - 1)]
            g = P[(P.set == k) & (P.episode == r.episode) & (P.frame_index <= t)]
            done = t >= len(fr) - 1
            pots = 2 if (done and r.success) else int(g.pots.iloc[-1]) if len(g) else 0  # success = both on
            status = ("SUCCESS" if r.success else "FAILED") if done else f"step {t}"
            col = (90, 200, 120) if (done and r.success) else (240, 90, 80) if done else (255, 255, 255)
            cols.append(caption(img, [f"{k}: {status}", f"pots on stove: {pots}   type: {r.type}"], col))
        outf.append(np.concatenate(cols, 1))
    with av.open(str(out), "w") as c:
        st = c.add_stream("libx264", rate=20)
        st.width, st.height, st.pix_fmt = outf[0].shape[1], outf[0].shape[0], "yuv420p"
        st.options = {"crf": "28"}
        for fr in outf:
            for pk in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "media").mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(out / "episodes.csv")
    P = pd.read_parquet(out / "pots_curves.parquet")
    B = E[E.set == "base"].set_index(["init_state", "repeat"])
    R = E[E.set == "residual"].set_index(["init_state", "repeat"])
    J = B.join(R, lsuffix="_b", rsuffix="_r").reset_index().sort_values(["repeat", "init_state"])
    groups = {
        "fixed: base had one pot": J[(J.type_b == "one_pot") & (J.success_r == 1)],
        "fixed: base placed no pot": J[(J.type_b == "no_pot") & (J.success_r == 1)],
        "fixed: base lost a pot": J[(J.type_b == "lost_pot") & (J.success_r == 1)],
        "broken: base succeeded, residual one pot": J[(J.success_b == 1) & (J.type_r == "one_pot")],
        "still failing: both one pot": J[(J.type_b == "one_pot") & (J.type_r == "one_pot")],
    }
    items = []
    for gname, g in groups.items():
        picked = g.drop_duplicates("init_state").head(a.per_group)
        for r in picked.itertuples():
            f = f"{gname.split(':')[0].replace(' ', '_')}_init{r.init_state:02d}_r{r.repeat}.mp4"
            rb = E[(E.set == "base") & (E.init_state == r.init_state) & (E.repeat == r.repeat)].iloc[0]
            rr = E[(E.set == "residual") & (E.init_state == r.init_state) & (E.repeat == r.repeat)].iloc[0]
            side_by_side(rb, rr, P, out / "media" / f)
            items.append((gname, f, r.init_state, r.repeat))
            print("wrote", f, flush=True)
    s = json.loads((out / "summary.json").read_text())
    html = [PAGE_HEAD.format(**fmt(s))]
    cur = None
    for gname, f, i, rep in items:
        if gname != cur:
            if cur is not None:
                html.append("</div>")
            html.append(f"<h3>{gname}</h3><div class='grid'>")
            cur = gname
        html.append(f"<figure><video src='media/{f}' controls muted loop preload='metadata'></video>"
                    f"<figcaption>val state {i}, repeat {rep} (left base, right residual)</figcaption></figure>")
    html.append("</div></main></body></html>")
    (out / "index.html").write_text("\n".join(html))


def fmt(s):
    c, L = s["counts"], s["lengths"]
    p = L["paired_both_succeed"]
    return {"b_succ": c["base"]["success"], "r_succ": c["residual"]["success"],
            "b_one": c["base"]["one_pot"], "r_one": c["residual"].get("one_pot", 0),
            "b_lost": c["base"]["lost_pot"], "r_lost": c["residual"].get("lost_pot", 0),
            "b_no": c["base"]["no_pot"], "r_no": c["residual"].get("no_pot", 0),
            "fixed": sum(s["fixed_by_base_type"].values()), "broken": sum(s["broken_into_type"].values()),
            "broken_one": s["broken_into_type"].get("one_pot", 0),
            "b_len": L["base"]["success_steps_median"], "r_len": L["residual"]["success_steps_median"],
            "d": p["residual_minus_base_steps"], "lo": p["ci95"][0], "hi": p["ci95"][1], "n": p["n"],
            "b_pot": L["base"]["t_first_pot_median"], "r_pot": L["residual"]["t_first_pot_median"],
            "acc": s["pot_detector"]["frame_acc_pots"], "end_acc": s["pot_detector"]["end_acc_failures"]}


PAGE_HEAD = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Moka pots: base vs residual</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#66645e;--card:#fff;--line:#e3e1db}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecebe6;--mut:#a3a19a;--card:#1f1f1d;--line:#33322f}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0}}
main{{max-width:1100px;margin:auto;padding:24px 16px}} h1{{font-size:24px;margin:0 0 4px}} .mut{{color:var(--mut)}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border-bottom:1px solid var(--line);padding:4px 12px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} img{{max-width:100%;background:#fff;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px}}
video{{width:100%;border-radius:4px}} figcaption{{font-size:13px;color:var(--mut)}}
</style></head><body><main>
<h1>Moka pots: base π0.5 vs advantage-guidance residual</h1>
<p class="mut">LIBERO-10 task 8, 10 held-out val states x 33 trials, same trials and sampling noise for both policies
(base eval 18355478, residual eval 18438187, 2026-11-11). Failure types from the agentview video with a pot-on-stove
detector (frame accuracy {acc:.0%} on 120 rollouts with recorded sim state, {end_acc:.0%} on their failures' end states).</p>
<table><tr><th>outcome</th><th>base</th><th>residual</th></tr>
<tr><td>success (both pots on the stove)</td><td>{b_succ}</td><td>{r_succ}</td></tr>
<tr><td>failed with one pot on the stove</td><td>{b_one}</td><td>{r_one}</td></tr>
<tr><td>failed after a placed pot came off</td><td>{b_lost}</td><td>{r_lost}</td></tr>
<tr><td>failed without placing any pot</td><td>{b_no}</td><td>{r_no}</td></tr></table>
<ul><li>The residual fixes {fixed} of base's failures and breaks {broken} of its successes ({broken_one} of those end with one pot on).</li>
<li>Almost every remaining failure has the first pot on the stove: the second pot is what is left to fix.</li>
<li>Faster: successes median {b_len:.0f} vs {r_len:.0f} steps; on the {n} trials both solve, the residual is
{d:+.1f} steps [{lo:+.1f}, {hi:+.1f}]. First pot on the stove: median step {b_pot:.0f} vs {r_pot:.0f}, and base's long
tail of late first pots (steps 270-520) is gone.</li></ul>
<img src="overview.png" alt="outcomes, success lengths, first-pot timing">
<img src="failure_curves.png" alt="pot count and IQL value over time for example failures">
"""

if __name__ == "__main__":
    main()
