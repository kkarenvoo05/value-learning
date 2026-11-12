"""Rollout gallery: base pi0.5 vs a guided policy on the same 330 held-out trials, side by side [2026-11-11].

"Always be visualizing" (mentor, Nov 11): which trials guidance fixes, which it breaks, where both fail, and how long
successful episodes take. Both evals use the same start state, env seed and matched noise seed (noise seed 1) per
(init state, repeat), so each pair starts from an identical scene and an identical noise stream until the policies
diverge.

Writes  reports/site/value-learning/gallery-<slug>/index.html  (+ gallery.json) and the last-frame posters to
        <stage>/posters/{base,cmp}/*.jpg (ffmpeg). Videos are not copied into the repo; --upload pushes videos, posters
        and the page to gs://zima-robot/reports/karenvo/value-learning/gallery-<slug>/.
Usage: .venv/bin/python scripts/site_gallery.py [--cmp RUN] [--slug residual] [--upload]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site import OUT, _js, _legend, e, inline, page  # noqa: E402
from gripper_report import MODE_TEXT  # noqa: E402
from gripper_report import build as gripper_build  # noqa: E402
from gripper_report import tables as gripper_tables  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
STAGE = Path("/iris/u/ankile/karenvo/libero/site_stage")
BASE = "eval_base_noise1_libero_10_18355478"
GCS = "gs://zima-robot/reports/karenvo/value-learning"
GCLOUD = "/iris/u/ankile/karenvo/google-cloud-sdk/bin/gcloud"
GROUPS = [  # key, title, base success, cmp success, colour
    ("fixed", "Fixed by guidance", 0, 1, "var(--good)"),
    ("broken", "Broken by guidance", 1, 0, "var(--crit)"),
    ("bothfail", "Both fail", 0, 0, "var(--fg3)"),
    ("bothok", "Both succeed", 1, 1, "var(--s1)"),
]


def load(run):
    return pd.read_csv(RUNS / run / "episodes.csv").set_index(["init_state", "repeat"])


def poster(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.3", "-i", str(src), "-frames:v", "1", "-q:v", "4", str(dst)],
                   check=True)


def hist(steps, lo=250, hi=520, w=15):
    edges = np.arange(lo, hi + w, w)
    c, _ = np.histogram(np.clip(steps, lo, hi - 1), edges)
    return [{"x": int(edges[i] + w / 2), "y": int(c[i])} for i in range(len(c))]


FEATURED = [  # (label, state, repeat, note) picked by watching the reruns, Nov 11
    ("base", 48, 0, "<b>Base freezes holding the pot.</b> It closes on the first pot's handle at step 110 and holds still until step ~275 "
                    "before lifting, then runs out of time on the second pot. Guided lifts about 12 steps after closing and succeeds."),
    ("base", 13, 1, "<b>Base freezes hovering.</b> After placing the first pot it holds still with open fingers at the second pot's handle "
                    "for ~200 steps and never closes. Guided succeeds."),
    ("guided", 42, 1, "<b>Guided tips a pot while placing it.</b> Still gripping the first pot over the stove, it tilts it onto its side "
                      "(steps ~225–265) and lets go; that pot is lying on its side for the rest of the episode."),
    ("guided", 9, 0, "<b>Guided tips the first pot, then hovers.</b> The first pot topples at the stove edge (steps ~220–235); the robot then "
                     "holds still with open fingers at the second pot for ~200 steps. Base succeeds on this rerun."),
]


def gripper_section(gdir: Path, cmp_run: str):
    """({"state-repeat": rerun pair}, HTML) for the wrist-view reruns, or ({}, "") if there are none."""
    if not (gdir / "gripper_metrics.csv").exists():
        return {}, ""
    pairs, st = gripper_build(gdir, BASE, cmp_run)
    tb = gripper_tables(pairs, st)
    wrist = {f"{int(x.init_state)}-{int(x.repeat)}": dict(
        s=int(x.init_state), r=int(x.repeat), bv=x.video_b, gv=x.video_g, bs=bool(x.success_b), gs=bool(x.success_g),
        bm=MODE_TEXT[x.mode_b], gm=MODE_TEXT[x.mode_g], bfc=int(x.freeze_closed_b), bfo=int(x.freeze_open_b),
        gfc=int(x.freeze_closed_g), gfo=int(x.freeze_open_g)) for x in pairs.itertuples()}
    b, g = st["base"], st["guided"]
    n = b["episodes"]
    bfail = pairs[~pairs.success_b.astype(bool)]
    bfail_closed = int((bfail.freeze_closed_b >= 60).sum())
    gtip_states = sorted({int(x.init_state) for x in pairs.itertuples() if x.tipped_g > 0})
    feats = "".join(f'<div class="wfeat" data-k="{s_}-{r_}" data-note="{e(note)}"></div>' for _, s_, r_, note in FEATURED
                    if f"{s_}-{r_}" in wrist)
    html = f"""
<h2 id="wrist">Wrist view: what guidance changes at the grasp</h2>
<p class="sub">The 224 px agent view is too small to see the gripper, so {n} trials (the 10 start states × repeats 0–2) were rerun with both
policies, the same start states and noise seeds, logging the wrist camera, the gripper command, the finger opening and the pots' poses at every step.
Matched-noise episodes are not bit-reproducible: base matched its gallery outcome on {b["agree_gallery"]}/{n} reruns and guided on
{g["agree_gallery"]}/{n}, so read this as how the two policies behave, not as an account of particular gallery trials.</p>
<div class="sumcard res"><ul>
<li><b>Guidance removes a freeze after the grasp, not before it.</b> Base often closes on a pot's handle and then holds still without lifting:
for 3 s or more in {b["eps_freeze_closed"]}/{n} episodes, including {bfail_closed} of its {n - b["success"]} failures. Guided never does
({g["eps_freeze_closed"]}/{n}). {b["hold_long"]} of base's {b["holds"]} lifts came 3 s or more after closing, against {g["hold_long"]} of {g["holds"]} for guided.</li>
<li><b>It is not faster or surer at grasping.</b> Both reach and close on the first pot at about the same time (median step {b["t_lift1"]:.0f} vs
{g["t_lift1"]:.0f}), and guided does not need fewer attempts on the second pot ({b["attempts2"]:.2f} vs {g["attempts2"]:.2f}).</li>
<li><b>Guided fails differently.</b> Of its {n - g["success"]} failures, {g["modes"]["tipped"]} left a pot lying on its side (states
{", ".join(map(str, gtip_states))}; base never did), mostly while placing a pot on the stove; {g["modes"]["freeze_open"]} held still with open
fingers at the second pot without closing, and {g["modes"]["slow"]} ran out of time. Open-hand holds of 3 s or more show up in
{g["eps_freeze_open"]} guided episodes against {b["eps_freeze_open"]} for base. Tipping on placement is the one failure that plausibly comes from
committing harder; the open-hand hover is base's own failure, which guidance does not remove.</li>
<li><b>Small numbers:</b> {n} pairs, {n - g["success"]} guided failures. The tip and hover counts need more reruns to be firm.</li>
</ul></div>
<div class="tw"><table><tr><th></th><th>base</th><th>guided</th></tr>{tb["summary"]}</table></div>
<p class="cap">"Held still" = the gripper moves less than 1 cm over the next 0.5 s while within 12 cm of a pot that is neither lifted nor on the stove.
A grasp attempt is a close command within 12 cm of a pot; "on air" means the fingers shut to under 1 cm. Tipped = the pot's up-axis ends below 0.5
(lying on its side), from the final simulator state.</p>
<h3>How each rerun episode ended</h3>
<div class="tw"><table><tr><th></th><th>base</th><th>guided</th></tr>{tb["modes"]}</table></div>
<h3>Examples</h3>
<p class="cap">Each video: third-person view | wrist camera, with a timeline below: gripper close command (grey), grasp attempts
(▼ <span style="color:#1f8a4c">lifted</span>, <span style="color:#d03b3b">closed on air</span>, <span style="color:#c98a00">held but no lift</span>),
a strip marking when the gripper holds still at a pot (<span style="color:#7b4bc9">open</span>, <span style="color:#c98a00">closed</span>),
finger opening and both pots' heights; dashed lines mark a pot landing on the stove. The other reruns open from the <b>wrist view</b> button on
their gallery card below.</p>
{feats}"""
    return wrist, html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", default="eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187")
    ap.add_argument("--cmp-label", default="Residual + guidance (w 9.26)")
    ap.add_argument("--note", default="", help="extra summary bullet (e.g. an eyeballed read of the failure sheets)")
    ap.add_argument("--slug", default="residual")
    ap.add_argument("--gripper", default=str(STAGE / "gripper-viz"),
                    help="scripts/gripper_viz.py output (wrist-view reruns of some trials); its videos are served from ../gripper-viz/")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    b, c = load(BASE), load(a.cmp)
    j = b.index.intersection(c.index)
    b, c = b.loc[j], c.loc[j]
    assert len(j) == 330, len(j)
    # matched trials: same env seed per (state, repeat)
    assert (b.env_seed == c.env_seed).all()
    stage = STAGE / f"gallery-{a.slug}"
    jobs = [(RUNS / run / "videos" / df.loc[k, "video"], stage / "posters" / side / (Path(df.loc[k, "video"]).stem + ".jpg"))
            for side, run, df in (("base", BASE, b), ("cmp", a.cmp, c)) for k in j]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: poster(*p), jobs))

    trials = []
    for k in j:
        bs, cs = int(b.loc[k, "success"]), int(c.loc[k, "success"])
        g = next(g for g, _, x, y, _ in GROUPS if (x, y) == (bs, cs))
        trials.append(dict(s=int(k[0]), r=int(k[1]), g=g, bs=bs, cs=cs, bn=int(b.loc[k, "policy_steps"]),
                           cn=int(c.loc[k, "policy_steps"]), bv=b.loc[k, "video"], cv=c.loc[k, "video"]))
    T = pd.DataFrame(trials)
    counts = T.g.value_counts().to_dict()
    wrist, wrist_html = gripper_section(Path(a.gripper), a.cmp)

    # per start state
    st = T.groupby("s").agg(base=("bs", "mean"), cmp=("cs", "mean"), fixed=("g", lambda x: (x == "fixed").sum()),
                            broken=("g", lambda x: (x == "broken").sum()), bothfail=("g", lambda x: (x == "bothfail").sum()))
    bsteps = T[T.bs == 1].groupby("s").bn.mean()
    csteps = T[T.cs == 1].groupby("s").cn.mean()
    srows = "".join(
        f'<tr><td><b>{s}</b></td><td>{100 * r.base:.0f}%</td><td>{100 * r.cmp:.0f}%</td>'
        f'<td style="color:var(--{"good" if r.cmp > r.base else "crit" if r.cmp < r.base else "fg2"})">{100 * (r.cmp - r.base):+.0f}</td>'
        f'<td>{r.fixed}</td><td>{r.broken}</td><td>{r.bothfail}</td>'
        f'<td>{bsteps.get(s, float("nan")):.0f}</td><td>{csteps.get(s, float("nan")):.0f}</td></tr>'
        for s, r in st.iterrows())

    both = T[T.g == "bothok"]
    d = both.cn - both.bn
    rng = np.random.default_rng(0)
    bsd = np.array([d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    lo, hi = np.percentile(bsd, [2.5, 97.5])
    lens = dict(base_mean=float(T[T.bs == 1].bn.mean()), cmp_mean=float(T[T.cs == 1].cn.mean()),
                base_med=float(T[T.bs == 1].bn.median()), cmp_med=float(T[T.cs == 1].cn.median()),
                both_n=int(len(both)), both_diff=float(d.mean()), both_lo=float(lo), both_hi=float(hi),
                faster=int((d < 0).sum()), slower=int((d > 0).sum()))
    hser = [{"name": "base π₀.₅", "color": "var(--fg3)", "pts": hist(T[T.bs == 1].bn.values)},
            {"name": a.cmp_label, "color": "var(--s3)", "pts": hist(T[T.cs == 1].cn.values)}]
    dh = hist(d.values, lo=-150, hi=150, w=15)
    dser = [{"name": "trials both solve", "color": "var(--s1)", "pts": dh, "endLabel": False},
            {"name": "no change", "color": "var(--fg3)", "dash": "5 4", "noMarkers": True, "endLabel": False,
             "pts": [{"x": 0, "y": 0}, {"x": 0, "y": max(p["y"] for p in dh)}]}]

    n = len(T)
    sb, sc = T.bs.mean(), T.cs.mean()
    out = OUT / f"gallery-{a.slug}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gallery.json").write_text(json.dumps(trials))
    chips = "".join(f'<button class="chip" data-g="{g}"><i style="background:{col}"></i>{e(t)} <b>{counts.get(g, 0)}</b></button>'
                    for g, t, _, _, col in GROUPS)
    states = "".join(f'<option value="{s}">state {s}</option>' for s in sorted(T.s.unique()))
    body = f"""
<h1>Rollout gallery: base π₀.₅ vs {e(a.cmp_label)}</h1>
<p class="sub">LIBERO-10 task 8 ("put both moka pots on the stove"), the 330 held-out trials (10 start states × 33 repeats).
Each pair has the same start state, env seed and matched noise seed, so the two runs start identically and differ only because the policy does.</p>
<div class="sumcard res"><ul>
<li><b>Success: base {100 * sb:.1f}% ({int(T.bs.sum())}/{n}) → {100 * sc:.1f}% ({int(T.cs.sum())}/{n}).</b>
Guidance fixes {counts.get("fixed", 0)} trials and breaks {counts.get("broken", 0)}; {counts.get("bothfail", 0)} fail under both.</li>
<li><b>Successes are slightly faster:</b> mean {lens["cmp_mean"]:.0f} vs {lens["base_mean"]:.0f} policy steps (median {lens["cmp_med"]:.0f} vs {lens["base_med"]:.0f}).
On the {lens["both_n"]} trials both solve, guidance is {lens["both_diff"]:+.1f} steps [{lens["both_lo"]:+.1f}, {lens["both_hi"]:+.1f}]
({lens["faster"]} faster, {lens["slower"]} slower). The limit is 520 steps.</li>
<li><b>Failures cluster by start state:</b> see the table. The poster on each video is its <i>last</i> frame, so failures show where the robot ended up.</li>
{f"<li>{inline(a.note)}</li>" if a.note else ""}
</ul></div>

<h2 id="len">Episode length</h2>
<div class="cap">Successful episodes only, by length (15-step bins). Failures all run to the 520-step limit.</div>
{_legend([(h["name"], h["color"]) for h in hser])}
{_js("lineChart", "glen", {"series": hser, "xlab": "policy steps", "xname": "steps", "ylab": "episodes", "width": 820, "height": 260})}
<div class="cap" style="margin-top:14px">Trials both solve: guided length minus base length (15-step bins). Left of the dashed line = guidance is faster.</div>
{_js("lineChart", "gdiff", {"series": dser, "xlab": "guided − base (policy steps)", "xname": "difference", "ylab": "trials", "width": 820, "height": 240})}

<h2 id="states">By start state</h2>
<div class="tw"><table><tr><th>start state</th><th>base</th><th>guided</th><th>change</th><th>fixed</th><th>broken</th><th>both fail</th>
<th>base steps (succ.)</th><th>guided steps (succ.)</th></tr>{srows}</table></div>

{wrist_html}
<h2 id="gallery">Videos</h2>
<p class="cap">Pick a group and a start state. ▶ plays both videos of a pair together (224 px agent view). LIBERO runs at 20 Hz and the eval saves one frame per control step, but writes the files at 10 fps; the page plays them at 2× so they run in real time. Videos load only when played. Cards marked <b>wrist view</b> also have a rerun with the wrist camera (see <a href="#wrist">Wrist view</a>).</p>
<div class="chips">{chips}<select id="st"><option value="">all states</option>{states}</select><select id="spd"><option value="2">real time (20 fps)</option><option value="1">half speed</option><option value="0.5">quarter speed</option></select></div>
<div id="grid" class="vgrid"></div>
<p class="src">Source: <code>runs/{BASE}</code> vs <code>runs/{e(a.cmp)}</code> (<code>episodes.csv</code>, <code>videos/</code>);
built by <code>scripts/site_gallery.py</code>. Wrist-view reruns: <code>scripts/gripper_viz.py</code>, <code>scripts/gripper_report.py</code>
(standalone page: <a href="../gripper-viz/index.html">gripper-viz</a>).</p>
<style>
.chips{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 14px}}
.chip{{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--grid);background:var(--surface);color:var(--fg);border-radius:999px;padding:4px 12px;font:inherit;font-size:13px;cursor:pointer}}
.chip i{{width:9px;height:9px;border-radius:50%;display:inline-block}}
.chip.on{{border-color:var(--fg);box-shadow:0 0 0 1px var(--fg) inset}}
#st,#spd{{font:inherit;font-size:13px;padding:4px 8px;border-radius:8px;border:1px solid var(--grid);background:var(--surface);color:var(--fg)}}
.vgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.pair{{background:var(--card);border-radius:10px;padding:8px}}
.pair .hd{{display:flex;justify-content:space-between;align-items:center;font-size:13px;margin-bottom:6px}}
.pair .vv{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
.pair video{{width:100%;aspect-ratio:1;border-radius:6px;background:#000;display:block}}
.pair .lb{{font-size:12px;color:var(--fg2);margin-top:3px}}
.ok{{color:var(--good);font-weight:600}}.no{{color:var(--crit);font-weight:600}}
.play{{border:1px solid var(--grid);background:var(--surface);color:var(--fg);border-radius:6px;padding:1px 10px;cursor:pointer;font:inherit;font-size:13px}}
.more{{margin:16px auto;display:block}}
.pair.wide{{grid-column:1/-1}}
.wbtn{{margin-left:6px}}
.wbox,.wfeat{{margin-top:10px}}
.wcap{{font-size:12.5px;color:var(--fg2);margin-bottom:6px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.wv{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
@media (max-width:760px){{.wv{{grid-template-columns:1fr}}}}
.wv figure{{margin:0}}.wv figcaption{{font-size:12.5px;margin-bottom:4px}}.wv .fz{{color:var(--fg3)}}
.wv video{{width:100%;border-radius:6px;background:#000;display:block}}
.wfeat{{background:var(--card);border-radius:10px;padding:10px;margin:10px 0 18px}}
.wkey i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}}
</style>
<script>
(function(){{
  const T = {json.dumps(trials)}, W = {json.dumps(wrist)};
  const grid = document.getElementById("grid"), sel = document.getElementById("st"), spd = document.getElementById("spd");
  let g = "fixed", shown = 0, list = [];
  const lbl = (ok, n) => `<span class="${{ok ? "ok" : "no"}}">${{ok ? "success" : "failure"}}</span> · ${{n}} steps`;
  function card(t) {{
    const d = document.createElement("div"); d.className = "pair";
    d.innerHTML = `<div class="hd"><span><b>state ${{t.s}}</b> · repeat ${{t.r}}</span><button class="play">▶ play both</button></div>
      <div class="vv"><div><video preload="none" muted playsinline poster="posters/base/${{t.bv.replace(".mp4", ".jpg")}}" src="v/base/${{t.bv}}"></video>
      <div class="lb">base: ${{lbl(t.bs, t.bn)}}</div></div>
      <div><video preload="none" muted playsinline poster="posters/cmp/${{t.cv.replace(".mp4", ".jpg")}}" src="v/cmp/${{t.cv}}"></video>
      <div class="lb">guided: ${{lbl(t.cs, t.cn)}}</div></div></div>`;
    const w = W[t.s + "-" + t.r];
    if (w) {{
      const wb = document.createElement("button"); wb.className = "play wbtn"; wb.textContent = "wrist view ▾";
      d.querySelector(".hd").appendChild(wb);
      wb.onclick = () => {{
        let box = d.querySelector(".wbox");
        if (box) {{ box.remove(); d.classList.remove("wide"); wb.textContent = "wrist view ▾"; return; }}
        d.classList.add("wide"); wb.textContent = "wrist view ▴";
        box = document.createElement("div"); box.className = "wbox"; box.innerHTML = wristPair(w); d.appendChild(box); wireWrist(box);
      }};
    }}
    const vs = d.querySelectorAll(".vv video"), btn = d.querySelector(".play");
    btn.onclick = () => {{
      const playing = !vs[0].paused || !vs[1].paused;
      vs.forEach(v => {{ if (playing) v.pause(); else {{ if (v.ended) v.currentTime = 0; v.playbackRate = +spd.value; v.play(); }} }});
      btn.textContent = playing ? "▶ play both" : "❚❚ pause";
    }};
    vs.forEach(v => v.onended = () => {{ if (vs[0].ended && vs[1].ended) btn.textContent = "↻ replay"; }});
    return d;
  }}
  function wristPair(w, note) {{
    const side = (lab, v, ok, m, fc, fo) => `<figure><figcaption><b>${{lab}}</b> rerun: <span class="${{ok ? "ok" : "no"}}">${{m}}</span>
      <span class="fz">· held still closed ${{fc}} / open ${{fo}} steps</span></figcaption>
      <video preload="none" muted playsinline controls src="../gripper-viz/${{v}}"></video></figure>`;
    return `<div class="wcap">${{note ? note + " " : ""}}Rerun of state ${{w.s}} · repeat ${{w.r}} (same start and noise seed) with the wrist camera logged; its outcome can differ from the gallery video.
      <button class="play wplay">▶ play both</button></div>
      <div class="wv">${{side("base", w.bv, w.bs, w.bm, w.bfc, w.bfo)}}${{side("guided", w.gv, w.gs, w.gm, w.gfc, w.gfo)}}</div>`;
  }}
  function wireWrist(el) {{
    const vs = el.querySelectorAll(".wv video"), b = el.querySelector(".wplay");
    b.onclick = () => {{ const playing = !vs[0].paused || !vs[1].paused;
      vs.forEach(v => {{ if (playing) v.pause(); else {{ if (v.ended) v.currentTime = 0; v.playbackRate = +spd.value / 2; v.play(); }} }});
      b.textContent = playing ? "▶ play both" : "❚❚ pause"; }};
  }}
  function more() {{
    list.slice(shown, shown + 24).forEach(t => grid.appendChild(card(t))); shown = Math.min(list.length, shown + 24);
    const old = document.querySelector(".more"); if (old) old.remove();
    if (shown < list.length) {{ const m = document.createElement("button"); m.className = "play more"; m.textContent = `show more (${{list.length - shown}} left)`; m.onclick = more; grid.after(m); }}
  }}
  function render() {{
    document.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c.dataset.g === g));
    list = T.filter(t => t.g === g && (!sel.value || t.s == sel.value)); grid.innerHTML = ""; shown = 0;
    if (!list.length) grid.innerHTML = '<p class="cap">No trials in this group for this state.</p>';
    more();
  }}
  document.querySelectorAll(".chip").forEach(c => c.onclick = () => {{ g = c.dataset.g; render(); }});
  spd.onchange = () => document.querySelectorAll(".vv video").forEach(v => v.playbackRate = +spd.value);
  document.querySelectorAll(".wfeat").forEach(el => {{ el.innerHTML = wristPair(W[el.dataset.k], el.dataset.note); wireWrist(el); }});
  sel.onchange = render; render();
}})();
</script>"""
    (out / "index.html").write_text(page("Rollout gallery", '<a href="../index.html">Value learning</a> / <a href="../week-09/index.html">Week 9</a> / Gallery', body, 1))
    print(json.dumps(dict(counts=counts, lens=lens), indent=1))
    print("->", out / "index.html")
    if a.upload:
        dst = f"{GCS}/gallery-{a.slug}"
        for side, run in (("base", BASE), ("cmp", a.cmp)):
            subprocess.run([GCLOUD, "storage", "rsync", "-r", str(RUNS / run / "videos"), f"{dst}/v/{side}"], check=True)
        subprocess.run([GCLOUD, "storage", "rsync", "-r", str(stage / "posters"), f"{dst}/posters"], check=True)
        subprocess.run([GCLOUD, "storage", "cp", "--cache-control=no-cache", str(out / "index.html"), str(out / "gallery.json"), f"{dst}/"],
                       check=True)
        print(f"https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/gallery-{a.slug}/index.html")


if __name__ == "__main__":
    main()
