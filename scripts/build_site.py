"""Build the mentor-facing progress site: a hub page linking one page per project week [ours, 2026-11-03].

Inputs (reports/site/content/):
  week07.json      week-7 LIBERO findings + backbone table (schema: headline, glossary, findings[], backbone_table[])
  earlier.json     weeks 5-6 Mulligan findings (schema: overview, glossary, weeks[{week, dates, summary, findings[]}])
  jobs.json        in-progress section for the current week: [{job, what, why, after, out}] (state read live from sacct)
  episodes.json    per-episode value traces for the week-7 episode player (made by scripts/site_episode_assets.py)
Figures named in the content files are copied next to each page. Output: reports/site/value-learning/
  index.html, site.css, charts.js, week-05/ … week-09/ (each index.html + assets/)
  week09.json + week09_data.json (scripts/site_week09_data.py) → week-09/ [2026-11-09]
  week09.json["since"] + week09_since_data.json (scripts/site_since_data.py) → "Since Nov 10" block on week-09 [2026-11-11]
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


def _js(fn: str, cid: str, spec: dict) -> str:
    return f'<div id="{cid}"></div><script>{fn}(document.getElementById("{cid}"),{json.dumps(spec, ensure_ascii=False)})</script>'


def _legend(items) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:{c};height:10px;width:10px;border-radius:50%"></i>{e(t)}</span>'
                                            for t, c in items) + "</div>"


def _card(cid: str, c: dict, body: str, status="settled", date="") -> str:
    return (f'<div class="finding" id="{cid}"><h3>{inline(c["title"])}</h3>'
            f'<div class="meta"><span class="pill {status}">{status}</span> {e(date)}</div><p class="q">{inline(c["question"])}</p>'
            f'<ul>{"".join(f"<li>{inline(v)}</li>" for v in c["verdict"])}</ul>{body}'
            f'<div class="src">Source: {inline(c["source"])}</div><p class="cap"><a href="#summary">↑ back to summary</a></p></div>')


def w9_policy(d: dict) -> str:
    col = {"untagged": "var(--s1)", "tag": "var(--s2)", "weights": "var(--s3)", "success+": None}
    heads = {"untagged": "No tag: which data, demo share and learning rate", "tag": "All rollouts + a tag or advantage weights",
             "success+": "Successful rollouts only, + a tag or weights"}
    rows, seen = [], set()
    for r in d["rows"]:
        g = "tag" if r["group"] == "weights" else r["group"]
        if g not in seen:
            seen.add(g); rows.append({"head": True, "label": heads[g]})
        c = col[r["group"]] or ("var(--s2)" if r["label"].endswith("tag") else "var(--s3)")
        note = f'{r["success"] * 100:.1f}% success ({r["n"]}/330 trials{"" if r["complete"] else ", still running"}) · {r["note"]} · job {r["job"]}'
        rows.append({"label": r["label"] + ("" if r["complete"] else " (running)"), "value": r["diff"], "lo": r["lo"], "hi": r["hi"],
                     "color": c, "note": note, "dim": not r["complete"], "hollow": not r["complete"],
                     "right": f'{r["success"] * 100:.1f}%  ({"+" if r["diff"] >= 0 else ""}{r["diff"] * 100:.1f})'})
    spec = {"rows": rows, "unit": "change vs base (points)", "pct": True, "labelWidth": 260, "width": 820, "domain": [-0.25, 0.15],
            "zeroLabel": f'base π₀.₅ {d["base"] * 100:.1f}%', "title": "Change in success vs base"}
    return (_legend([("no tag", "var(--s1)"), ("tag", "var(--s2)"), ("advantage weights", "var(--s3)")])
            + '<div class="cap">x axis: change in success rate vs base, in points. Right column: success rate (change). Hover a row for details.</div>'
            + _js("ciPlot", "w9pol", spec))


def w9_tag(d: dict) -> str:
    col = {"real": "var(--s2)", "floor": "var(--fg3)", "control": "var(--s1)"}
    rows = [{"label": r["label"], "value": round(100 * r["ratio"], 1), "color": col[r["group"]],
             "note": f'raw shift {r["shift_raw"]:.4f} · loss with correct / flipped tag {r["loss_correct"]:.5f} / {r["loss_flipped"]:.5f}'}
            for r in d["rows"]]
    rows.append({"label": "Swap the task instruction (reference)", "value": round(100 * d["prompt_swap_far"], 1), "color": "var(--s3)",
                 "note": "a different LIBERO instruction in the prompt, RECAP-tagged model"})
    return (_legend([("real labels", "var(--s2)"), ("never tagged", "var(--fg3)"), ("planted tag (control)", "var(--s1)"),
                     ("instruction swap", "var(--s3)")])
            + '<div class="cap">Action change when the tag flips, as % of the change between two noise draws (100% = as large as the policy\'s own randomness).</div>'
            + _js("barChart", "w9tag", {"rows": rows, "unit": "% of sampling noise", "labelWidth": 270, "width": 820, "digits": 1}))


def w9_coach(d: dict) -> str:
    src = [{"label": r["label"], "value": r["value"],
            "color": "var(--fg3)" if "Random" in r["label"] else "var(--s2)" if r["label"].startswith("Current") else "var(--s1)",
            "note": f'95% CI {r["lo"]:.4f} to {r["hi"]:.4f} (bootstrap over frames)'} for r in d["sources"]]
    src.sort(key=lambda r: -r["value"])
    src.append({"label": "Learned by the prompt-tag policy", "value": 0.0008, "color": "var(--s4)", "note": "RECAP-faithful run 18338571, tag check"})
    src.append({"label": "Learned by the AdaRMS-tag policy", "value": 0.006, "color": "var(--s4)", "note": "run 18355677, tag check"})
    rules = []
    for r in d["rules"]:
        rules.append({"label": f'{r["label"]} · {"round 1" if r["round"] == "1" else "explore"}', "value": r["value"],
                      "color": "var(--fg3)" if "Random" in r["label"] else ("var(--s1)" if r["round"] == "1" else "var(--s4)")})
    return (_legend([("current labels", "var(--s2)"), ("other value functions / lookaheads", "var(--s1)"), ("random floor", "var(--fg3)"),
                     ("what trained policies learned", "var(--s4)")])
            + '<div style="font-weight:600;margin-top:8px">Label source (all 900 rollouts)</div><div class="cap">Ideal shift: RMS action difference between + and − (raw action units). Higher = more signal the tag could carry.</div>'
            + _js("barChart", "w9coach", {"rows": src, "unit": "ideal shift", "labelWidth": 290, "width": 820, "digits": 4})
            + '<div style="font-weight:600;margin-top:16px">Label rule, on round 1 vs the exploration round (450 rollouts each)</div>'
            + _legend([("round 1 (normal sampling)", "var(--s1)"), ("explore round (2× noise)", "var(--s4)"), ("random floor", "var(--fg3)")])
            + _js("barChart", "w9rules", {"rows": rules, "unit": "ideal shift", "labelWidth": 290, "width": 820, "digits": 4}))


def w9_curve(d: dict) -> str:
    out = '<div class="grid2">'
    for key, title in [("old", "Round 1 data (450 rollouts, 30 scenes)"), ("explore", "Round 1 + exploration round pooled (900)")]:
        ser = []
        for kind, name, c in [("pc1", "PC1 positive control", "var(--s1)"), ("labels", "our labels", "var(--s2)")]:
            ser.append({"name": name, "color": c, "pts": [{"x": r["n"], "y": r["info"], "sd": r["info_sd"],
                        "note": f'{r["draws"]} draw{"s" if r["draws"] > 1 else ""}'} for r in d[key] if r["kind"] == kind]})
        xt = sorted({r["n"] for r in d[key]})
        out += (f'<div class="card"><div style="font-weight:600">{e(title)}</div>'
                + _js("lineChart", f"w9curve{key}", {"series": ser, "xlab": "episodes used (log scale)", "xname": "episodes",
                      "ylab": "action information", "xticks": xt, "xlog": True, "width": 460, "height": 280,
                      "yticks": [0, 0.05, 0.1, 0.15, 0.2] if key == "old" else [0, 0.1, 0.2, 0.3]})
                + "</div>")
    return out + '</div><div class="cap">Action information = share of the uncertainty in the tag that the action explains beyond the state, on held-out episodes (0.08 = 8%). Bars = ±1 sd over 5 subsamples.</div>'


def w9_branch(d: dict) -> str:
    out = ('<div class="legend"><span><i style="background:var(--s2)"></i>V* (true chance of success, 12 continuations)</span>'
           '<span><i style="background:var(--s3);height:10px;width:8px"></i>labelled + by our pipeline</span>'
           '<span><i style="background:var(--grid);height:10px;width:8px"></i>labelled −</span></div>'
           '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px">')
    for i, ep in enumerate(d["episodes"]):
        spec = {"series": [{"name": "V*", "color": "var(--s2)", "pts": [{"x": t, "y": v} for t, v in zip(ep["t"], ep["v"])], "endLabel": False}],
                "xlab": "env step (20 Hz)", "xname": "step", "ylab": "V*", "yticks": [0, 0.25, 0.5, 0.75, 1], "xticks": [0, 100, 200, 300, 400, 500],
                "width": 460, "height": 230, "strip": {"label": "label", "pts": [{"x": t, "on": p} for t, p in zip(ep["t"], ep["pos"])],
                                                       "onColor": "var(--s3)", "onLabel": "+", "offLabel": "−"}}
        out += f'<div class="card" style="margin:0"><div style="font-weight:600">Failed episode {ep["ep"]} <span style="color:var(--fg3);font-weight:400">· job {ep["job"]}</span></div>' + _js("lineChart", f"w9br{i}", spec) + "</div>"
    p = d["pooled"]
    rel = [{"label": "True value V*: split-half", "value": round(p["v"], 2), "color": "var(--s1)", "note": "6 vs 6 continuations, 299 points"},
           {"label": "True advantage A*: split-half", "value": round(p["a"], 2), "color": "var(--s2)", "note": "per run: " + ", ".join(f'{r["a"]:.2f}' for r in d["reliability"])},
           {"label": "A* reliability, all 12 continuations", "value": round(2 * p["a"] / (1 + p["a"]), 2), "color": "var(--s2)", "note": "Spearman-Brown from the split-half"},
           {"label": "Our labels vs A*: correlation", "value": round(p["label_corr"], 2), "color": "var(--s4)", "note": f'+ share when A* > 0: {p["pos_when_up"]:.2f}; when A* < 0: {p["pos_when_down"]:.2f}'},
           {"label": "Our value V vs V*: correlation", "value": round(p["oofV_vs_Vstar"], 2), "color": "var(--s4)", "note": "out-of-fold ResNet"}]
    return (out + '</div><div style="font-weight:600;margin-top:10px">Reliability and agreement (12 failed episodes pooled; 1 = perfect)</div>'
            + _js("barChart", "w9rel", {"rows": rel, "unit": "correlation", "labelWidth": 290, "width": 820, "digits": 2, "domain": [0, 1]}))


def _pct(x, sign=False):
    return f'{"+" if sign and x >= 0 else ""}{100 * x:.1f}'


def _dsrl_eval(r, base) -> str:
    """Held-out eval of the DSRL agent snapshot, paired with base on the 330 trials [2026-11-11]."""
    if not r or r["pending"]:
        return " Held-out eval of the episode-971 snapshot on the 330 trials (job 18445300) is running."
    state = "" if r["complete"] else f' (partial, {r["n"]} of 330 trials; too early to call)'
    return (f' **On the 330 held-out trials, the episode-971 agent gets {_pct(r["success"])}% vs base {_pct(base)}%, '
            f'{_pct(r["diff"], True)} points [{_pct(r["lo"], True)}, {_pct(r["hi"], True)}]**{state}; start states up / down '
            f'{r["states_up"]} / {r["states_down"]}.')


def since_dyn(key: str, sd: dict) -> str:
    """Headline sentences whose numbers change while jobs run [2026-11-11]."""
    if key == "residual":
        r = next(x for x in sd["ag"]["rows"] if x["label"].startswith("Residual, shift of AdaRMS w 8"))
        if r["pending"]:
            return "**Residual parameterisation (the paper's fix):** trained; eval at the action shift of AdaRMS w 8 is queued."
        c = sd["res_vs_ada"]
        state = "final" if r["complete"] else f'partial, {r["n"]} of 330 trials, all 10 states'
        return (f'**Residual parameterisation (the paper\'s fix; {state}): {_pct(r["success"])}% vs base, '
                f'{_pct(r["diff"], True)} points [{_pct(r["lo"], True)}, {_pct(r["hi"], True)}].** On the same trials AdaRMS w 8 has '
                f'{_pct(c["adarms"])}%, so the residual is {_pct(c["diff"], True)} [{_pct(c["lo"], True)}, {_pct(c["hi"], True)}] above it at the same action shift'
                + ("." if r["complete"] else "; too early to call."))
    if key == "dsrl":
        d = sd["dsrl"]
        return (f'**DSRL (the suggested RL baseline) is flat near base for ~400 episodes, then rising:** the last 3 cycles through the 30 train '
                f'states average {_pct(d["last3"])}% vs base {_pct(d["base_train"])}% on the same states ({d["episodes"]} episodes so far; '
                'training-time success).' + _dsrl_eval(sd.get("dsrl_eval"), sd["ag"]["base"]))
    return ""


def since_html(cs: dict, sd: dict) -> str:
    """'Since Nov 10' block under the week-9 summary: Nov 10 feedback, AG + DSRL experiments, running jobs, next steps."""
    pill = {"settled": "done", "running": "running", "open": "open", "queued": "queued"}
    head = "".join(f"<li>{inline(since_dyn(b['dyn'], sd) if 'dyn' in b else b['t'])}{ev_links(b)}</li>" for b in cs["headline"])
    lars = "".join(f'<tr><td><b>{inline(r["ask"])}</b></td><td>{inline(r["did"])}</td><td>{inline(r["result"])}{ev_links(r)}</td>'
                   f'<td><span class="pill {r["status"]}">{pill[r["status"]]}</span></td></tr>' for r in cs["lars"])
    cd = cs["cards"]

    def ci_rows(rows, colors, labels=None):
        out, seen = [], set()
        for r in rows:
            if r["pending"]:
                continue
            if labels and r["group"] not in seen:  # heading only for groups with at least one evaluated row
                seen.add(r["group"]); out.append({"head": True, "label": labels[r["group"]]})
            part = "" if r["complete"] else f' (running, {r["n"]}/330)'
            out.append({"label": r["label"] + part, "value": r["diff"], "lo": r["lo"], "hi": r["hi"], "color": colors[r["group"]],
                        "dim": not r["complete"], "hollow": not r["complete"],
                        "note": f'{_pct(r["success"])}% success ({r["n"]}/330 trials) · {r["note"]} · start states up / down: '
                                f'{r["states_up"]} / {r["states_down"]} · job {r["job"]}',
                        "right": f'{_pct(r["success"])}%  ({_pct(r["diff"], True)})'})
        return out

    agc = {"adarms": "var(--s2)", "residual": "var(--s3)", "control": "var(--fg3)"}
    agl = {"adarms": "Contrastive guidance on the AdaRMS-tag checkpoint (zero training)",
           "residual": "Residual parameterisation (frozen π₀.₅ + zero-init tag head)", "control": "Control"}
    pend = [r["label"] for r in sd["ag"]["rows"] if r["pending"]]
    ag_body = (_legend([("AdaRMS checkpoint", "var(--s2)"), ("residual model", "var(--s3)"), ("control", "var(--fg3)")])
               + '<div class="cap">x axis: change in success vs base, points. Hollow = eval still running. Hover a row for details.</div>'
               + _js("ciPlot", "w9ag", {"rows": ci_rows(sd["ag"]["rows"], agc, agl), "unit": "change vs base (points)", "pct": True,
                                         "labelWidth": 300, "width": 820, "domain": [-0.35, 0.4],
                                         "zeroLabel": f'base π₀.₅ {_pct(sd["ag"]["base"])}%', "title": "Guidance vs base"})
               + (f'<div class="cap">Not yet evaluated: {e(", ".join(pend))}.</div>' if pend else ""))
    ser = [{"name": k, "color": "var(--s2)" if k.startswith("AdaRMS") else "var(--s3)",
            "pts": [{"x": p["w"], "y": p["ratio"], "note": f'gripper flips {100 * p["flip"]:.1f}%'} for p in v]} for k, v in sd["shift"].items()]
    ag_body += ('<div style="font-weight:600;margin-top:16px">How far guidance moves the action (offline check)</div>'
                '<div class="cap">Action shift caused by guidance at weight w, as a multiple of the shift between two noise draws '
                '(1 = as large as the policy\'s own randomness). The residual needs w 9.3 for the shift AdaRMS reaches at w 8.</div>'
                + _js("lineChart", "w9shift", {"series": ser, "xlab": "guidance weight w", "xname": "w", "ylab": "shift / noise",
                                               "xticks": [1, 2, 4, 8, 16], "xlog": True, "width": 620, "height": 260,
                                               "yticks": [0, 0.5, 1, 1.5, 2]}))
    d = sd["dsrl"]
    xmax = max(p["x"] for p in d["cycles"])
    dser = [{"name": f"agent life {life}", "color": "var(--s2)" if life == 1 else "var(--s3)",
             "pts": [{"x": p["x"], "y": p["y"], "note": f'cycle {p["cycle"]}, {p["n"]} episodes'} for p in d["cycles"] if p["life"] == life]}
            for life in (1, 2)]
    dser.append({"name": "base π₀.₅", "color": "var(--fg3)", "dash": "5 4", "noMarkers": True,
                 "pts": [{"x": 0, "y": d["base_train"]}, {"x": xmax, "y": d["base_train"]}]})
    dcard = dict(cd["dsrl"])
    dcard["verdict"] = [v.replace("{last3}", f'{_pct(d["last3"])}%').replace("{base}", f'{_pct(d["base_train"])}%') for v in dcard["verdict"]]
    dsrl_body = ('<div class="cap">Training success per cycle (each of the 30 train states once). Life 2 = fresh agent after the requeue at '
                 f'episode {d["restart"]}. Dashed: base π₀.₅ on the same states ({d["base_n"]} rollouts). One cycle has ±9 points of noise.</div>'
                 + _js("lineChart", "w9dsrl", {"series": dser, "xlab": "episode", "xname": "episode", "ylab": "success",
                                               "yticks": [0, 0.25, 0.5, 0.75, 1], "width": 820, "height": 300}))
    fuc = {"untagged": "var(--s1)", "tag": "var(--s2)", "weights": "var(--s3)"}
    fu_body = (_legend([("no tag", "var(--s1)"), ("prompt tag", "var(--s2)"), ("advantage weights", "var(--s3)")])
               + _js("ciPlot", "w9fu", {"rows": ci_rows(sd["followup"]["rows"], fuc), "unit": "change vs base (points)", "pct": True,
                                        "labelWidth": 330, "width": 820, "domain": [-0.15, 0.2],
                                        "zeroLabel": f'base π₀.₅ {_pct(sd["followup"]["base"])}%', "title": "Follow-ups vs base"}))
    nz = "".join(f'<tr><td>{e(r["arm"][1:].replace("_all", "") + "×" if r["arm"].startswith("x") else "none (base)")}'
                 f'{" on every chunk" if r["arm"].endswith("_all") else ""}</td><td>{r["n"]}</td><td>{_pct(r["success"])}%</td>'
                 f'<td>{"—" if r["spread"] is None else str(r["spread"]) + "×"}</td></tr>' for r in sd["noise"])
    v = sd["vgt"]
    vrows = [{"label": r["source"], "value": round(r["a"] if r["a"] is not None else r["label"], 3),
              "color": "var(--fg3)" if "random" in r["source"] else "var(--s1)" if r["a"] is not None else "var(--s2)",
              "note": "learned advantage vs true advantage" if r["a"] is not None else "binary labels vs true advantage"} for r in v["rows"]]
    vrows.append({"label": "Ceiling (noise in the ground truth)", "value": v["ceiling"], "color": "var(--s4)",
                  "note": f'split-half reliability {v["split_half"]} over 12 continuations'})
    r2 = "".join(f'<tr><td>{e(r["layout"])}</td><td>{_pct(r["r1"])}% ({r["n"]})</td><td>{_pct(r["base"])}% ({r["base_n"]})</td></tr>'
                 for r in sd["round2"])
    other_body = ('<div class="grid2"><div class="card"><div style="font-weight:600" id="noise">Exploration-noise sweep (train states)</div>'
                  '<div class="tw"><table><tr><th>start-noise scale</th><th>episodes</th><th>success</th><th>action spread vs base</th></tr>'
                  + nz + '</table></div></div><div class="card"><div style="font-weight:600">Round-2 collection: round-1 policy vs base</div>'
                  '<div class="tw"><table><tr><th>layouts</th><th>round-1 policy</th><th>base π₀.₅</th></tr>' + r2 + '</table></div>'
                  '<div class="cap">Same states and layouts, different random draws (not paired).</div></div></div>'
                  f'<div style="font-weight:600;margin-top:14px">Learned values and labels vs the simulator\'s true advantage ({v["points"]} branch points)</div>'
                  '<div class="cap">Correlation with the true advantage A* (1 = perfect). The ceiling is the best any labeller could score, given the noise in A*.</div>'
                  + _js("barChart", "w9vgt", {"rows": vrows, "unit": "correlation with A*", "labelWidth": 300, "width": 820, "digits": 2, "domain": [0, 0.8]}))
    fut = "".join(f'<tr><td><b>{inline(f["exp"])}</b></td><td>{inline(f["why"])}</td><td style="white-space:nowrap">{inline(f["cost"])}</td>'
                  f'<td><span class="pill {"running" if f["pri"] == "high" else "queued" if f["pri"] == "medium" else "open"}">{e(f["pri"])}</span></td></tr>'
                  for f in cs["future"])
    return (f'<h2 id="since">{inline(cs["title"])}</h2><p class="lead">{inline(cs["lead"])}</p>'
            f'<div class="sumcard res"><ul>{head}</ul></div>'
            f'<h3 id="lars" style="margin-top:22px">Feedback, and what came of it</h3><div class="tw"><table><tr><th>feedback</th>'
            f'<th>what I did</th><th>result</th><th></th></tr>{lars}</table></div>'
            + _card("ag", cd["ag"], ag_body, "running", "2026-11-10 – 11-11")
            + _card("dsrl", dcard, dsrl_body, "running", "2026-11-11")
            + _card("followup", cd["followup"], fu_body, "settled", "2026-11-09 – 11-10")
            + _card("other", cd["other"], other_body, "settled", "2026-11-09 – 11-11")
            + '<h3 id="running" style="margin-top:22px">Running and queued</h3><p class="lead">State is read from Slurm when the page is built.</p>'
            + jobs_html(cs["jobs"])
            + f'<h3 id="future" style="margin-top:22px">Next experiments to discuss</h3><div class="tw"><table><tr><th>experiment</th>'
              f'<th>why</th><th>cost</th><th>priority</th></tr>{fut}</table></div>'
            + f'<h3 style="margin-top:22px">Questions</h3><div class="sumcard"><ol>{"".join(f"<li>{inline(q)}</li>" for q in cs["ask"])}</ol></div>'
            + '<h2 style="margin-top:34px">The Nov 10 update</h2><p class="lead">The summary at the top of the page and everything below this line are the update as sent for the Nov 10 meeting.</p>')


def week9(c9: dict, data: dict, since: dict | None = None):
    d = OUT / "week-09"
    d.mkdir(parents=True, exist_ok=True)
    lis = lambda items: "".join(f"<li>{inline(b['t'])}{ev_links(b)}</li>" for b in items)
    nav = ('<nav class="jump"><a href="#summary">Summary</a><a href="#since">Since Nov 10</a><a href="#ag">AG</a><a href="#dsrl">DSRL</a>'
           '<a href="#future">Next experiments</a><a href="#story">How I got here</a><a href="#gaps">RECAP gaps</a>'
           '<a href="#checks">Discrepancies</a><a href="#progress">Running now</a><a href="#questions">Questions</a>'
           '<a href="#evidence">Charts</a><a href="#minor">Smaller results</a></nav>')
    gaps = "".join(f'<tr><td><b>{inline(g["gap"])}</b></td><td>{inline(g["did"])}</td><td>{inline(g["result"])}'
                   f'{ev_links(g)}</td><td><span class="pill {g["status"]}">{"tried" if g["status"] == "settled" else "open"}</span></td></tr>'
                   for g in c9["gaps"])
    cd = c9["cards"]
    body = (f'<h1>{e(c9["title"])}</h1><div class="sub">{inline(c9["sub"])}</div>{nav}'
            f'<h2 id="summary" style="border:0;margin-top:18px">Summary</h2><div class="goal">{inline(c9["goal"])}</div>'
            f'<div class="sumcard res"><ul>{lis(c9["summary"])}</ul></div>'
            + (since_html(c9["since"], since) if since else "")
            + f'<h2 id="story">How I got here</h2><div class="sumcard"><ol>{lis(c9["story"])}</ol></div>'
            f'<h2 id="gaps">Gaps vs RECAP, and what I did</h2><div class="tw"><table><tr><th>RECAP has</th><th>what I did</th>'
            f'<th>result</th><th></th></tr>{gaps}</table></div>'
            f'<h2 id="checks">Discrepancies found while checking the results</h2><div class="note"><ul>'
            + "".join(f"<li>{inline(x)}</li>" for x in c9["discrepancies"]) + "</ul></div>"
            + '<h2 id="progress">Running now</h2><p class="lead">The last policy fine-tuning runs of the recipe grid (untagged, '
              'different demo shares and learning rates), then automatic follow-ups on the best recipe. State is read from Slurm when the page is built.</p>'
            + jobs_html(c9["jobs"])
            + f'<h2 id="questions">Questions</h2><div class="sumcard"><ol>{"".join(f"<li>{inline(q)}</li>" for q in c9["questions"])}</ol></div>'
            + '<h2 id="evidence">Charts behind the claims</h2>'
            + _card("policy", cd["policy"], w9_policy(data["policy"]), "running", "2026-11-09")
            + _card("tag-effect", cd["tag-effect"], w9_tag(data["tag"]), "settled", "2026-11-04 – 11-08")
            + _card("coach", cd["coach"], w9_coach(data["coach"]), "settled", "2026-11-06 – 11-09")
            + _card("curve", cd["curve"], w9_curve(data["curve"]), "settled", "2026-11-09")
            + _card("branching", cd["branching"], w9_branch(data["branch"]), "settled", "2026-11-09")
            + '<h2 id="minor">Smaller results since Nov 3</h2>' + minor_html(c9["minor"], d / "assets"))
    (d / "index.html").write_text(page("Week 9 progress", '<a href="../index.html">Value learning</a> › Week 9', body, 1))


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


def hub(c7: dict, ce: dict, c8: dict, c9: dict):
    cards = (f'<a class="wk" href="week-09/index.html"><div class="wn">Week 9 · update for Nov 10, follow-up Nov 11 · current</div>'
             f'<div class="wt">LIBERO: a fine-tune recipe that keeps base performance, and why the advantage tag goes unused</div>'
             f'<div style="color:var(--fg2)">{inline(c9["summary"][0]["t"])}</div></a>'
             f'<a class="wk" href="week-08/index.html"><div class="wn">Week 8 · Nov 3 – Nov 10</div>'
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
    c9 = json.loads((CONTENT / "week09.json").read_text())
    sp = CONTENT / "week09_since_data.json"  # scripts/site_since_data.py [2026-11-11]
    week9(c9, json.loads((CONTENT / "week09_data.json").read_text()), json.loads(sp.read_text()) if sp.exists() else None)
    earlier(ce)
    hub(c7, ce, c8, c9)
    print("->", OUT)


if __name__ == "__main__":
    main()
