"""Build the mentor-facing progress site: a hub page linking one page per project week [ours, 2026-11-03].

Inputs (reports/site/content/):
  week07.json      week-7 LIBERO findings + backbone table (schema: headline, glossary, findings[], backbone_table[])
  earlier.json     weeks 5-6 Mulligan findings (schema: overview, glossary, weeks[{week, dates, summary, findings[]}])
  jobs.json        in-progress section for the current week: [{job, what, why, after, out}] (state read live from sacct)
  episodes.json    per-episode value traces for the week-7 episode player (made by scripts/site_episode_assets.py)
Figures named in the content files are copied next to each page. Output: reports/site/value-learning/
  index.html, site.css, charts.js, week-05/ … week-08/ (each index.html + assets/)
Publish: scripts/publish_week_report.sh value-learning   (public bucket; review first)
"""

import html
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "reports" / "site"
CONTENT = SITE / "content"
OUT = SITE / "value-learning"
e = html.escape
SERIES = {"ResNet": "var(--s1)", "pi05 frozen": "var(--s2)", "pi05 LoRA": "var(--s3)", "SigLIP ft": "var(--s4)"}


def inline(s) -> str:
    s = e(str(s))
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", s)


def page(title: str, crumb: str, body: str, depth: int) -> str:
    up = "../" * depth
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>'
            f'<link rel="stylesheet" href="{up}site.css"><script src="{up}charts.js"></script></head><body><div class="wr">'
            f'<header class="top"><div class="crumb">{crumb}</div><button class="theme" id="theme">Light / dark</button></header>'
            f'{body}<p class="src" style="margin-top:40px">Built {datetime.now():%Y-%m-%d %H:%M} PT from '
            f'<code>reports/site/content/</code> by <code>scripts/build_site.py</code>.</p></div>'
            f'<script>themeToggle(document.getElementById("theme"))</script></body></html>')


def copy_fig(src: str, dest_dir: Path) -> str | None:
    p = Path(src)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print("missing figure", p)
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", "_".join(p.parts[-2:]))
    shutil.copy2(p, dest_dir / name)
    return f"assets/{name}"


def backbone_color(g: str):
    """Colour follows the backbone everywhere: ResNet blue, frozen π₀.₅ orange, LoRA aqua, SigLIP yellow."""
    s = g.lower()
    if "resnet" in s:
        return "var(--s1)"
    if "lora" in s:
        return "var(--s3)"
    if "siglip" in s:
        return "var(--s4)"
    if "pi05" in s or "π₀.₅" in s or s == "frozen b":
        return "var(--s2)"
    return None


def chart_html(c: dict, cid: str) -> str:
    if not c or c.get("type") != "bar" or not c.get("rows"):
        return ""
    rows, other = [], {}
    for r in c["rows"]:
        g = r.get("group") or ""
        col = backbone_color(g)
        if col is None:  # non-backbone groups: fixed order of the remaining-identity slots within this chart, gray first
            other.setdefault(g, ["var(--fg3)", "var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"][min(len(other), 4)])
            col = other[g]
        rows.append({"label": r["label"], "value": r["value"], "sd": r.get("sd"), "color": col,
                     "note": (f"{r['n_seeds']} seeds" if r.get("n_seeds") else "")})
    hb = c.get("higher_is_better")
    hint = " · higher is better" if hb is True else " · lower is better" if hb is False else ""
    return (f'<div class="ft" style="font-weight:600;font-size:13.5px;margin-top:10px">{inline(c.get("title", ""))}'
            f'<span style="color:var(--fg3);font-weight:400">{e(" (" + c["unit"] + ")" if c.get("unit") else "")}{hint}</span></div>'
            f'<div id="{cid}"></div><script>barChart(document.getElementById("{cid}"),'
            f'{json.dumps({"rows": rows, "unit": c.get("unit", "value"), "title": c.get("title", ""), "labelWidth": 250})})</script>')


def finding_html(f: dict, n: str, assets: Path) -> str:
    st = f.get("status", "settled")
    figs = ""
    for src in f.get("figures", [])[:2]:
        rel = copy_fig(src, assets)
        if rel:
            figs += f'<img class="figimg" loading="lazy" src="{rel}" alt="{e(f["title"])}"><div class="cap">{e(Path(src).name)}</div>'
    date = f.get("date", "")
    return (f'<div class="finding" id="{e(f.get("id", n))}"><h3>{n + ". " if n else ""}{inline(f["title"])}</h3>'
            f'<div class="meta"><span class="pill {e(st)}">{e(st)}</span> {e(date)}</div>'
            f'<p class="q">{inline(f.get("question", ""))}</p>'
            f'<ul>{"".join(f"<li>{inline(v)}</li>" for v in f.get("verdict", []))}</ul>'
            f'{chart_html(f.get("chart", {}), "c_" + re.sub(r"[^a-z0-9]", "", n.lower() + str(abs(hash(f["title"])) % 10**6)))}'
            f'{chart_html(f.get("chart2", {}), "d_" + re.sub(r"[^a-z0-9]", "", n.lower() + str(abs(hash(f["title"])) % 10**6)))}'
            f'{figs}{discussion_html(f.get("discussion", []))}<div class="src">Source: {inline(f.get("source", ""))}</div></div>')


def discussion_html(blocks: list) -> str:
    out = ""
    for b in blocks:
        out += f'<h4 class="disc">{inline(b["h"])}</h4>' if b.get("h") else ""
        out += "".join(f"<p>{inline(x)}</p>" for x in b.get("p", []))
        out += f'<ul>{"".join(f"<li>{inline(x)}</li>" for x in b["ul"])}</ul>' if b.get("ul") else ""
        out += "".join(f"<p>{inline(x)}</p>" for x in b.get("p_after", []))
    return f'<div class="discussion">{out}</div>' if out else ""


def glossary_html(g: dict) -> str:
    return ('<details class="card"><summary>Glossary: what the metrics mean</summary><table>'
            + "".join(f"<tr><th style='width:190px'>{inline(k)}</th><td>{inline(v)}</td></tr>" for k, v in g.items())
            + "</table></details>")


def job_states(ids):
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "-o", "JobID,State,Start,End,Elapsed,NodeList"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    st = {}
    for ln in out.splitlines():
        j, s, start, end, el, node = ln.split("|")
        st[j] = {"state": s.split()[0], "start": start, "end": end, "elapsed": el, "node": node}
    return st


def jobs_html(jobs: list) -> str:
    st = job_states([j["job"] for j in jobs])
    rows = ""
    for j in jobs:
        s = st.get(j["job"], {})
        state = s.get("state", "?")
        cls = {"RUNNING": "running", "PENDING": "queued", "COMPLETED": "settled", "FAILED": "failed",
               "CANCELLED": "failed", "TIMEOUT": "failed"}.get(state, "open")
        when = (f"running {s['elapsed']} on {s['node']}" if state == "RUNNING" else
                f"finished {s['end'][5:16].replace('T', ' ')}" if state in ("COMPLETED", "FAILED", "TIMEOUT") else
                f"waits for {j['after']}" if j.get("after") else "waiting for a GPU")
        rows += (f"<tr><td style='white-space:nowrap'><code>{e(j['job'])}</code></td><td><b>{inline(j['what'])}</b><br>"
                 f"<span style='color:var(--fg2)'>{inline(j.get('why', ''))}</span></td>"
                 f"<td><span class='pill {cls}'>{e(state.lower())}</span><br><span style='color:var(--fg3);font-size:12px'>{e(when)}</span></td></tr>")
    return f'<div class="tw"><table><tr><th>job</th><th>what it is, and why</th><th>state</th></tr>{rows}</table></div>'


def spotlight(c7: dict, eps: dict, assets: Path) -> str:
    """The ResNet vs π₀.₅ tradeoff: scatter of accuracy vs steadiness, milestone bars, and the episode player."""
    shutil.copy2(CONTENT / "episodes.json", assets / "episodes.json")
    for f in (CONTENT / "frames").glob("ep*_frames.jpg"):
        shutil.copy2(f, assets / f.name)
    bt = [r for r in c7["backbone_table"] if r.get("spotlight")]
    pts = [{"label": f"{r['arm']} ({r['n_seeds']} seed{'s' if r['n_seeds'] != 1 else ''})", "short": r.get("short", r["arm"]),
            "x": r["mae"], "xsd": r.get("mae_sd"), "y": r["snr"], "ysd": r.get("snr_sd"),
            "color": backbone_color(r["group"]) or "var(--fg3)", "note": f"milestone {r['milestone']:+.3f}",
            "dx": r.get("dx", 11), "dy": r.get("dy", -9), "anchor": r.get("anchor", "start")} for r in bt]
    ms = [{"label": r.get("short", r["arm"]), "value": r["milestone"], "sd": r.get("milestone_sd"),
           "color": backbone_color(r["group"]) or "var(--fg3)", "note": f"{r['n_seeds']} seeds"} for r in bt]
    table = ("<table><tr><th>value function</th><th class='n'>seeds</th><th class='n'>MAE ↓</th><th class='n'>change SNR ↑</th>"
             "<th class='n'>milestone ↑</th><th class='n'>MAE succ. &gt;20 s ↓</th><th class='n'>AUC 10 s ↑</th></tr>"
             + "".join(f"<tr><td>{inline(r['arm'])}</td><td class='n'>{r['n_seeds']}</td>"
                       f"<td class='n'>{r['mae']:.3f}{' ± %.3f' % r['mae_sd'] if r.get('mae_sd') else ''}</td>"
                       f"<td class='n'>{r['snr']:.3f}{' ± %.3f' % r['snr_sd'] if r.get('snr_sd') else ''}</td>"
                       f"<td class='n'>{r['milestone']:+.3f}{' ± %.3f' % r['milestone_sd'] if r.get('milestone_sd') else ''}</td>"
                       f"<td class='n'>{r['mae_succ20']:.3f}</td><td class='n'>{r['auc10']:.3f}</td></tr>" for r in c7["backbone_table"])
             + "</table>")
    sp = c7["spotlight"]
    return f"""
<h2 id="tradeoff">Spotlight: ResNet vs π₀.₅, accuracy vs flicker vs progress</h2>
<p class="lead">{inline(sp['intro'])}</p>
<div class="grid2">
 <div class="card"><div style="font-weight:600">Accuracy vs steadiness, validation set</div>
  <div class="cap">Each dot is one value function (mean over seeds, bars = ±1 sd). Further right = less accurate; higher = steadier,
  more meaningful changes. Hover a dot for details.</div><div id="sc"></div></div>
 <div class="card"><div style="font-weight:600">Does V react when the pot lands on the stove?</div>
  <div class="cap">Milestone response: rise in V around the first pot placement, minus the rise in time-matched episodes without one.
  Higher = V tracks task progress.</div><div id="msb"></div></div>
</div>
<script>
scatter(document.getElementById("sc"),{json.dumps({"pts": pts, "xlab": "MAE vs target (lower = more accurate)", "ylab": "change SNR (higher = steadier)",
  "xname": "MAE", "yname": "SNR", "better": "↖ better", "width": 500, "height": 380})});
barChart(document.getElementById("msb"),{json.dumps({"rows": ms, "unit": "milestone response", "labelWidth": 150, "width": 500})});
</script>
<ul class="take">{''.join(f'<li>{inline(t)}</li>' for t in sp['takeaways'])}</ul>
<div class="card" id="player">
 <div style="font-weight:600">One episode, four backbones: move the cursor across the chart to scrub the cameras</div>
 <div class="tabs"></div>
 <p class="epcap" style="color:var(--fg2);font-size:13.5px"></p>
 <div class="player">
  <div><div class="legend"></div><div class="epchart rel"></div></div>
  <div><div class="cams"><div><div class="cam a"></div><div class="camlbl">agent view</div></div><div><div class="cam w"></div><div class="camlbl">wrist</div></div></div>
   <div class="readout"></div></div>
 </div>
 <div class="epmetrics tw"></div>
</div>
<script>
fetch("assets/episodes.json").then(r=>r.json()).then(d=>episodePlayer(document.getElementById("player"),d,{json.dumps({
  "assetBase": "assets/", "episodes": sp["episodes"],
  "series": [{"key": "resnet", "label": "ResNet-18", "color": "var(--s1)"}, {"key": "b", "label": "π₀.₅ frozen", "color": "var(--s2)"},
             {"key": "lora", "label": "π₀.₅ LoRA", "color": "var(--s3)"}, {"key": "siglip", "label": "π₀.₅ SigLIP fine-tuned", "color": "var(--s4)"}]})}));
</script>
<details class="card"><summary>Full table: every value function scored with the same metrics</summary><div class="tw">{table}</div>
<div class="cap">{inline(sp.get('table_note', ''))}</div></details>
"""


def ev_links(b: dict) -> str:
    out = ""
    for k in ("ev", "ev2"):
        if b.get(k):
            out += f' <a class="ev" href="{e(b[k][0])}">{e(b[k][1])} →</a>'
    return out


def summary_html(sm: dict) -> str:
    def block(title, items, cls="", num=False):
        tag = "ol" if num else "ul"
        lis = "".join(f"<li>{inline(b['t'])}{ev_links(b)}</li>" for b in items)
        return f'<div class="sumcard {cls}"><div class="sumh">{title}</div><{tag}>{lis}</{tag}></div>'
    return (f'<div class="goal"><b>Goal.</b> {inline(sm["goal"])}</div>'
            + block("Previously (last meeting, Oct 27)", sm["previously"], "prev")
            + block("What we found this week", sm["results"], "res", num=True)
            + '<div class="sum3">' + block("Working on now", sm["working"])
            + block("Next steps", sm["next"]) + block("Questions and new ideas", sm["questions"]) + "</div>")


def minor_html(fs: list, assets: Path) -> str:
    rows = "".join(f'<tr><td><a href="#{e(f["id"])}">{inline(f["title"])}</a></td><td>{inline(f.get("oneline", ""))}</td>'
                   f'<td><span class="pill {e(f.get("status", "settled"))}">{e(f.get("status", "settled"))}</span></td></tr>' for f in fs)
    det = "".join(f'<details class="minor"><summary>{inline(f["title"])}</summary>{finding_html(f, "", assets)}</details>' for f in fs)
    return (f'<div class="tw"><table><tr><th>experiment</th><th>result in one line</th><th>status</th></tr>{rows}</table></div>'
            f'<div class="sumh" style="margin-top:14px">Details (click to open)</div>{det}')


def week7(c7: dict, jobs: list) -> str:
    d = OUT / "week-07"
    assets = d / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    eps = json.loads((CONTENT / "episodes.json").read_text())
    tier = lambda t: [f for f in c7["findings"] if f.get("tier", "main") == t]
    main = "".join(finding_html(f, str(i + 1), assets) for i, f in enumerate(tier("main")))
    nav = ('<nav class="jump"><a href="#summary">Summary</a><a href="#tradeoff">ResNet vs π₀.₅</a><a href="#player">Episode player</a>'
           '<a href="#progress">In progress</a><a href="#evidence">Evidence</a><a href="#minor">Smaller experiments</a>'
           '<a href="#setup">Setup</a></nav>')
    body = (f'<h1>Week 7 · Oct 27 – Nov 3, 2026</h1><div class="sub">LIBERO moka pots (libero_10 task 8): '
            f'value functions for π₀.₅ and advantage-conditioned fine-tuning</div>{nav}'
            f'<h2 id="summary" style="border:0;margin-top:18px">Summary</h2>' + summary_html(c7["summary"])
            + glossary_html(c7.get("glossary", {}))
            + spotlight(c7, eps, assets)
            + f'<h2 id="progress">In progress now</h2><p class="lead">{inline(c7["in_progress"]["intro"])}</p>'
            + "".join(f'<div class="card"><b>{inline(s["title"])}</b><p style="color:var(--fg2);margin:4px 0">{inline(s["text"])}</p></div>'
                      for s in c7["in_progress"].get("stages", []))
            + '<div class="note">This page is the snapshot from before the Nov 3 meeting. Live job status is on '
              '<a href="../week-08/index.html#progress">Week 8</a>.</div>'
            + '<h2 id="evidence">Evidence: main findings</h2><p class="lead">The findings behind the summary, in the order the '
              'story unfolded. Each card has the numbers, a chart and the log entry it comes from.</p>' + main
            + '<h2 id="minor">Smaller experiments that did not change the picture</h2><p class="lead">Each tried to close '
              'π₀.₅\'s steadiness gap from the head, the loss or the data side. Open a row for its numbers.</p>'
            + minor_html(tier("minor"), assets)
            + '<h2 id="setup">Setup: reproduction, task and data</h2>' + minor_html(tier("setup"), assets))
    (d / "index.html").write_text(page("Week 7 progress", '<a href="../index.html">Value learning</a> › Week 7', body, 1))


def scene_cal_html(sc: dict, txt: dict) -> str:
    cols = {"ResNet-18": "var(--s1)", "π₀.₅ frozen": "var(--s2)", "π₀.₅ LoRA": "var(--s3)"}
    tgt = sc["target"]["v"]
    panels, scripts, rows = "", "", ""
    for i, (k, col) in enumerate(cols.items()):
        b = sc[k]
        pts = [{"x": x, "y": y, "ysd": (b["sd"][j] if b.get("sd") else None), "color": col, "short": "",
                "label": f"{k}, scene {b['scenes'][j]}", "note": f"{round(x * 12)}/12 rollouts succeeded · target {tgt[j]:.2f}"}
               for j, (x, y) in enumerate(zip(b["succ"], b["v"]))]
        ref = [{"x": x, "y": t} for x, t in zip(b["succ"], tgt)]
        seeds = len(b["per_seed_pearson"])
        panels += (f'<div class="card"><div style="font-weight:600"><i style="display:inline-block;width:10px;height:10px;'
                   f'border-radius:50%;background:{col};margin-right:6px"></i>{e(k)}</div><div class="cap">'
                   f'{seeds} seed{"s" if seeds > 1 else ""}; r = {b["pearson"]:.2f}</div><div id="scal{i}"></div></div>')
        scripts += (f'scatter(document.getElementById("scal{i}"),{json.dumps({"pts": pts, "ref": ref, "xlab": "scene success rate", "ylab": "V in first 2 s", "xname": "success rate", "yname": "early V", "width": 340, "height": 300, "r": 6, "xticks": [0, 0.25, 0.5, 0.75, 1], "yticks": [-1, -0.8, -0.6, -0.4, -0.2]}, ensure_ascii=False)});')
        rows += (f'<tr><td>{e(k)}</td><td class="n">{b["pearson"]:.2f}</td><td class="n">{b["spearman"]:.2f}</td>'
                 f'<td class="n">{b["slope_vs_target"]:.2f}</td><td class="n">{b["scene_mae"]:.3f}</td></tr>')
    return (f'<div class="finding" id="scene-calibration"><h3>{inline(txt["title"])}</h3>'
            f'<div class="meta"><span class="pill settled">settled</span> 2026-11-03</div><p class="q">{inline(txt["question"])}</p>'
            f'<ul>{"".join(f"<li>{inline(v)}</li>" for v in txt["verdict"])}</ul>'
            f'<div class="legend"><span>● filled: predicted V (bar = ±1 sd over seeds)</span>'
            f'<span>○ hollow: the true target, same scene</span></div>'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px">{panels}</div>'
            f'<script>{scripts}</script><div class="tw"><table><tr><th>backbone</th><th class="n">r with success rate</th>'
            f'<th class="n">rank r</th><th class="n">spread (slope vs target; 1 = ideal)</th><th class="n">per-scene error</th></tr>'
            f'{rows}<tr><td>true target</td><td class="n">{sc["target"]["pearson"]:.2f}</td><td></td><td class="n">1.00</td>'
            f'<td class="n">0</td></tr></table></div><div class="src">Source: {inline(txt["source"])}</div></div>')


def week8(c8: dict, jobs: list):
    d = OUT / "week-08"
    d.mkdir(parents=True, exist_ok=True)
    pill = lambda st: f'<span class="pill {e(st)}">{e(st)}</span> '
    lis = lambda items: "".join(f"<li>{inline(b['t'])}{ev_links(b)}</li>" for b in items)
    nav = ('<nav class="jump"><a href="#summary">Summary</a><a href="#plan">Plan this week</a>'
           '<a href="#directions">New directions</a><a href="#progress">In progress</a><a href="#results">Results</a></nav>')
    body = (f'<h1>{e(c8["title"])}</h1><div class="sub">{inline(c8["sub"])}</div>{nav}'
            f'<h2 id="summary" style="border:0;margin-top:18px">Summary</h2><div class="goal">{inline(c8["headline"])}</div>'
            f'<div class="sumcard prev"><div class="sumh">From the Nov 3 meeting</div><ul>{lis(c8["meeting"])}</ul></div>'
            f'<h2 id="plan">Plan this week</h2><div class="sumcard res"><ol>'
            + "".join(f"<li>{pill(b.get('status', 'queued'))}{inline(b['t'])}{ev_links(b)}</li>" for b in c8["plan"])
            + '</ol></div><h2 id="directions">New directions to explore</h2>'
              '<p class="lead">From the meeting: SmolVLA (arXiv 2506.01844) builds on SmolVLM (arXiv 2504.05299) and drops '
              'the top half of the language model with little loss on LIBERO (reading only the first 16 layers: 78.5% '
              'success vs 80.3% with 32). Our π₀.₅ value function reads every layer, including the ones where the flicker grows. '
              'SeeQ (arXiv 2609.22085) scopes the value to the current subtask.</p>'
            + f'<div class="sumcard"><ul>{lis(c8["directions"])}</ul></div>'
            + '<h2 id="progress">In progress now</h2>' + jobs_html(jobs)
            + f'<h2 id="results">Results</h2><p class="lead">{inline(c8["results_note"])}</p>'
            + scene_cal_html(json.loads((CONTENT / "scene_calibration.json").read_text()), c8["scene_calibration"])
            + "".join(finding_html(f, "", d / "assets") for f in c8.get("findings", [])))
    (d / "index.html").write_text(page("Week 8 progress", '<a href="../index.html">Value learning</a> › Week 8', body, 1))


def earlier(ce: dict):
    for w in ce["weeks"]:
        n = int(w["week"])
        d = OUT / f"week-{n:02d}"
        assets = d / "assets"
        fs = "".join(finding_html(f, str(i + 1), assets) for i, f in enumerate(w["findings"]))
        body = (f'<h1>Week {n} · {e(w["dates"])}, 2026</h1><div class="sub">Mulligan Insert Marker (real-robot data)</div>'
                f'<p class="lead" style="margin-top:14px">{inline(w["summary"])}</p>'
                + (f'<div class="note">{inline(ce["overview"])}</div>' if n == max(int(x["week"]) for x in ce["weeks"]) else "")
                + glossary_html(ce.get("glossary", {})) + fs)
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(page(f"Week {n} progress", f'<a href="../index.html">Value learning</a> › Week {n}', body, 1))


def hub(c7: dict, ce: dict, c8: dict):
    cards = (f'<a class="wk" href="week-08/index.html"><div class="wn">Week 8 · Nov 3 – Nov 10 · current</div>'
             f'<div class="wt">LIBERO: first policy results, label strategy, and which π₀.₅ layers a value function needs</div>'
             f'<div style="color:var(--fg2)">{inline(c8["headline"])}</div></a>'
             f'<a class="wk" href="week-07/index.html"><div class="wn">Week 7 · Oct 27 – Nov 3</div>'
             f'<div class="wt">LIBERO: ResNet vs π₀.₅ value functions, and the first advantage-conditioned fine-tunes</div>'
             f'<div style="color:var(--fg2)">{inline(c7["headline"][0])}</div></a>')
    for w in sorted(ce["weeks"], key=lambda w: -int(w["week"])):
        cards += (f'<a class="wk" href="week-{int(w["week"]):02d}/index.html"><div class="wn">Week {w["week"]} · {e(w["dates"])}</div>'
                  f'<div class="wt">Mulligan Insert Marker</div><div style="color:var(--fg2)">{inline(w["summary"])}</div></a>')
    body = ('<h1>Value learning for robot policies</h1><div class="sub">Karen Vo · weekly progress</div>'
            '<p class="lead" style="margin-top:14px">Can a small RECAP-style value function V(o) learn real task progress from robot '
            'camera images, and can it improve a policy? Weeks 5–6 worked on real-robot data (Mulligan Insert Marker). '
            'From week 7 the work moved to simulation (LIBERO) with π₀.₅, where V can be tested by fine-tuning the policy and '
            'running new rollouts.</p><div class="weeks">' + cards + "</div>")
    (OUT / "index.html").write_text(page("Value learning progress", "Value learning", body, 0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONTENT / "site.css", OUT / "site.css")
    shutil.copy2(CONTENT / "charts.js", OUT / "charts.js")
    c7 = json.loads((CONTENT / "week07.json").read_text())
    ce = json.loads((CONTENT / "earlier.json").read_text())
    jobs = json.loads((CONTENT / "jobs.json").read_text())
    c8 = json.loads((CONTENT / "week08.json").read_text())
    week7(c7, jobs)
    week8(c8, jobs)
    earlier(ce)
    hub(c7, ce, c8)
    print("->", OUT)


if __name__ == "__main__":
    main()
