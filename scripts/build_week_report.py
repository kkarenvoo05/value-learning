"""Build the weekly progress page from the report folders (docs/INDEX.md layout) [ours, draft 2026-11-03].

Collects every reports/YYYY-MM-DD-<slug>/ whose date falls in [--start, --end] and that has a summary.json
(title, status, question, verdict[], tiles[], charts[] with bars {label, value, sd, highlight}); its report.md
is rendered (small Markdown subset) inside a collapsible section. Writes one standalone page,
reports/site/<name>/index.html: inline CSS + inline SVG charts, light/dark, no external assets, so it can be
uploaded alone (scripts/publish_week_report.sh). Numbers come only from the summary.json files, which are written
from the report's own data files, so the page cannot drift from the evidence.

Usage: .venv/bin/python scripts/build_week_report.py --start 2026-10-28 --end 2026-11-03 [--name 2026-11-03-week]
"""

import argparse
import html
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FOLDER = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")
e = html.escape

CSS = """
:root{--bg:#fcfcfb;--fg:#141413;--fg2:#5e5d59;--fg3:#8a8984;--grid:#e4e3df;--card:#f3f2ee;--bar:#5b8def;--hi:#d97757;--sd:#141413}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--fg3:#8f8e88;--grid:#33332f;--card:#262624;--bar:#6f9cf2;--hi:#e08a6c;--sd:#f0efec}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--fg3:#8f8e88;--grid:#33332f;--card:#262624;--bar:#6f9cf2;--hi:#e08a6c;--sd:#f0efec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wr{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:22px;margin:0 0 2px}.sub{color:var(--fg2);font-size:13px;margin-bottom:18px}
h2{font-size:18px;margin:36px 0 6px;padding-top:10px;border-top:1px solid var(--grid)}h3{font-size:15px;margin:18px 0 6px}
.status{display:inline-block;font-size:12px;color:var(--fg2);background:var(--card);border-radius:6px;padding:1px 8px;margin-bottom:6px}
.q{color:var(--fg2);margin:4px 0 10px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin:10px 0}
.tile{background:var(--card);border-radius:8px;padding:10px 12px}.tl{font-size:12px;color:var(--fg2)}.tv{font-size:24px;font-weight:650}.ts{font-size:12px;color:var(--fg3)}
.fig{margin:16px 0}.ft{font-weight:600;font-size:13.5px;margin-bottom:4px}svg{display:block;width:100%;height:auto}
svg text{fill:var(--fg2);font-size:12px}svg .v{fill:var(--fg)}ul{padding-left:20px}li{margin:3px 0}
details{margin:12px 0}summary{cursor:pointer;color:var(--fg2)}.md{font-size:13px}.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--grid)}
th{color:var(--fg2);font-weight:600}code{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}
"""


def inline(s: str) -> str:
    s = e(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", s)


def md(text: str) -> str:
    """Markdown subset used by our reports: #-headings, paragraphs, - lists (one nesting level), pipe tables."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("#"):
            lvl = min(len(ln) - len(ln.lstrip("#")) + 2, 5)  # report # -> h3
            out.append(f"<h{lvl}>{inline(ln.lstrip('#').strip())}</h{lvl}>")
            i += 1
        elif ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            out.append('<div class="tw"><table><tr>' + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</table></div>")
        elif ln.lstrip().startswith("- "):
            items = []
            while i < len(lines) and (lines[i].lstrip().startswith("- ") or lines[i].startswith("  ")):
                if lines[i].lstrip().startswith("- "):
                    items.append([len(lines[i]) - len(lines[i].lstrip()), lines[i].lstrip()[2:]])
                else:
                    items[-1][1] += " " + lines[i].strip()
                i += 1
            html_, depth = "<ul>", 0
            for ind, t in items:
                d = 1 if ind >= 2 else 0
                html_ += "<ul>" if d > depth else "</ul>" if d < depth else ""
                depth = d
                html_ += f"<li>{inline(t)}</li>"
            out.append(html_ + "</ul>" * (depth + 1))
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ")):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def bar_chart(c: dict) -> str:
    bars = c["bars"]
    top = max(b["value"] + b.get("sd", 0) for b in bars) * 1.15
    lw, w, rh = 190, 760, 26  # label column, total width, row height
    h = rh * len(bars) + 26
    x = lambda v: lw + (w - lw - 60) * v / top
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{e(c["title"])}">']
    for k in range(5):  # light gridlines
        gv = top * k / 4
        parts.append(f'<line x1="{x(gv):.1f}" x2="{x(gv):.1f}" y1="0" y2="{h - 20}" stroke="var(--grid)"/>'
                     f'<text x="{x(gv):.1f}" y="{h - 4}" text-anchor="middle">{gv:.2f}</text>')
    for j, b in enumerate(bars):
        y = j * rh + 4
        col = "var(--hi)" if b.get("highlight") else "var(--bar)"
        parts.append(f'<text x="{lw - 8}" y="{y + 15}" text-anchor="end">{e(b["label"])}</text>'
                     f'<rect x="{lw}" y="{y + 3}" width="{x(b["value"]) - lw:.1f}" height="{rh - 10}" rx="3" fill="{col}"/>')
        if b.get("sd"):
            lo, hi = x(b["value"] - b["sd"]), x(b["value"] + b["sd"])
            parts.append(f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{y + rh / 2 - 1}" y2="{y + rh / 2 - 1}" '
                         f'stroke="var(--sd)" stroke-width="1.5"/>')
        parts.append(f'<text class="v" x="{x(b["value"] + b.get("sd", 0)) + 6:.1f}" y="{y + 15}">'
                     f'{b["value"]:.3f}{" ± " + format(b["sd"], ".3f") if b.get("sd") else ""}</text>')
    parts.append("</svg>")
    return f'<div class="fig"><div class="ft">{e(c["title"])}</div>{"".join(parts)}</div>'


def section(n: int, folder: Path, s: dict) -> str:
    rel = folder.relative_to(ROOT)
    tiles = "".join(f'<div class="tile"><div class="tl">{e(t["label"])}</div><div class="tv">{e(t["value"])}</div>'
                    f'<div class="ts">{e(t.get("sub", ""))}</div></div>' for t in s.get("tiles", []))
    rep = folder / "report.md"
    details = (f"<details><summary>Full report (<code>{e(str(rel))}/report.md</code>)</summary>"
               f'<div class="md">{md(rep.read_text())}</div></details>') if rep.exists() else ""
    return (f'<h2 id="{e(folder.name)}">{n} · {e(s["title"])}</h2>'
            f'<div class="status">{e(folder.name[:10])} · {e(s.get("status", ""))}</div>'
            f'<p class="q">{inline(s.get("question", ""))}</p>'
            + (f'<div class="tiles">{tiles}</div>' if tiles else "")
            + "<ul>" + "".join(f"<li>{inline(v)}</li>" for v in s.get("verdict", [])) + "</ul>"
            + "".join(bar_chart(c) for c in s.get("charts", [])) + details)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--name", default=None, help="output folder under reports/site/ (default <end>-week)")
    ap.add_argument("--title", default="Value learning")
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    found = []
    for f in sorted(REPORTS.iterdir()):
        m = FOLDER.match(f.name)
        if f.is_dir() and m and start <= date.fromisoformat(m.group(1)) <= end and (f / "summary.json").exists():
            found.append((f, json.loads((f / "summary.json").read_text())))
    if not found:
        raise SystemExit(f"no report folders with summary.json between {start} and {end}")
    toc = "".join(f'<li><a href="#{e(f.name)}">{e(s["title"])}</a> · {e(s.get("status", ""))}</li>' for f, s in found)
    body = "".join(section(i + 1, f, s) for i, (f, s) in enumerate(found))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{e(a.title)} week</title><style>{CSS}</style></head><body><div class=\"wr\">"
            f"<h1>{e(a.title)} · {start:%b %d} – {end:%b %d, %Y}</h1>"
            f'<div class="sub">LIBERO libero_10 task 8 (moka pots) · built from <code>reports/*/summary.json</code> '
            f"on {date.today():%Y-%m-%d}</div><ul>{toc}</ul>{body}</div></body></html>")
    out = REPORTS / "site" / (a.name or f"{end}-week")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    print(f"{len(found)} report(s) -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
