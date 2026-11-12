"""Paired base vs guided gripper report: failure modes, second-pot grasp stats, side-by-side wrist videos [2026-11-11].

Reads <dir>/gripper_metrics.csv from scripts/gripper_viz.py (labels base and guided, same init states and repeats with
matched noise) and writes <dir>/index.html next to the videos. The gallery labels (fixed / broken / both) come from
the 330-trial evals (--gallery-base, --gallery-guided episodes.csv); the reruns need not reproduce them exactly
(matched-noise episodes are not bit-reproducible, scripts/libero_rollouts.py), so both are shown.

Failure mode of a failed episode (first match; FREEZE = 60 steps = 3 s):
  tipped          a pot ended on its side (sim_end orientation) or dipped below its start height
  freeze_closed   held a pot still, gripper closed, without lifting it, for >= FREEZE steps in total
  freeze_open     hovered still at a pot, gripper open, without closing, for >= FREEZE steps in total
  slow            none of these: lifted things but ran out of time or missed the goal
Usage: .venv/bin/python scripts/gripper_report.py --dir <gripper_viz out> --gallery-base <run> --gallery-guided <run>
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
MODES = ["success", "freeze_closed", "freeze_open", "tipped", "slow"]
FREEZE = 60
MODE_TEXT = {"success": "success", "freeze_closed": "froze holding a pot (closed, no lift)",
             "freeze_open": "froze hovering at a pot (open, no close)", "tipped": "tipped a pot onto its side",
             "slow": "other (ran out of time)"}


def mode(r):
    if r.success:
        return "success"
    if r.tipped > 0:
        return "tipped"
    if max(r.freeze_closed, r.freeze_open) >= FREEZE:
        return "freeze_closed" if r.freeze_closed >= r.freeze_open else "freeze_open"
    return "slow"


def group(b, g):
    return {(1, 1): "both succeed", (0, 1): "fixed by guidance", (1, 0): "broken by guidance", (0, 0): "both fail"}[(int(b), int(g))]


def build(out: Path, gallery_base: str, gallery_guided: str):
    """(pairs, stats) from <out>/gripper_metrics.csv and the two gallery runs' episodes.csv."""
    df = pd.read_csv(out / "gripper_metrics.csv")
    df["mode"] = [mode(r) for r in df.itertuples()]
    key = ["init_state", "repeat"]
    gb = pd.read_csv(RUNS / gallery_base / "episodes.csv")[key + ["success"]].rename(columns={"success": "gal_base"})
    gg = pd.read_csv(RUNS / gallery_guided / "episodes.csv")[key + ["success"]].rename(columns={"success": "gal_guided"})
    b = df[df.label == "base"].set_index(key)
    g = df[df.label == "guided"].set_index(key)
    pairs = b.join(g, lsuffix="_b", rsuffix="_g", how="inner").reset_index()
    pairs = pairs.merge(gb, on=key).merge(gg, on=key)
    pairs["group"] = [group(x.success_b, x.success_g) for x in pairs.itertuples()]
    pairs["gal_group"] = [group(x.gal_base, x.gal_guided) for x in pairs.itertuples()]

    # per-policy stats; second-pot stats over episodes that lifted a first pot
    stats = {}
    for lab, d in (("base", b), ("guided", g)):
        d2 = d[d.attempts2.notna()]
        lifted = d2[d2.lifted2.astype(bool)]
        stats[lab] = dict(
            episodes=len(d), success=int(d.success.sum()),
            modes={m: int((d["mode"] == m).sum()) for m in MODES},
            lifted_first=len(d2),
            attempts2=float(d2.attempts2.mean()) if len(d2) else np.nan,
            missed2=float(d2.missed2.mean()) if len(d2) else np.nan,
            first_try2=int(((d2.attempts2 == 1) & d2.lifted2.astype(bool)).sum()),
            lifted2=len(lifted),
            wait2=float(lifted.wait2.median()) if lifted.wait2.notna().any() else np.nan,
            t_lift1=float(d2.t_lift1.median()) if len(d2) else np.nan,
            freeze_closed=float(d.freeze_closed.mean()), freeze_open=float(d.freeze_open.mean()),
            eps_freeze_closed=int((d.freeze_closed >= FREEZE).sum()), eps_freeze_open=int((d.freeze_open >= FREEZE).sum()),
            tipped=int((d.tipped > 0).sum()),
            hold=float(pd.concat([d.hold1, d.hold2]).median()),
            hold_long=int((pd.concat([d.hold1, d.hold2]) >= FREEZE).sum()), holds=int(pd.concat([d.hold1, d.hold2]).notna().sum()),
            agree_gallery=None)
    stats["base"]["agree_gallery"] = int((pairs.success_b.astype(int) == pairs.gal_base).sum())
    stats["guided"]["agree_gallery"] = int((pairs.success_g.astype(int) == pairs.gal_guided).sum())
    return pairs, stats


def tables(pairs, stats):
    """HTML <tr> rows: summary (freezes, lifts, grasps), modes (how episodes ended), gtable (rerun vs gallery)."""
    f = lambda x, n=1: "–" if pd.isna(x) else f"{x:.{n}f}"
    rows = [("Success", lambda s: f"{s['success']} / {s['episodes']}"),
            ("Episodes with a closed freeze ≥ 3 s (holding a pot, not lifting)", lambda s: f"{s['eps_freeze_closed']}"),
            ("Episodes with an open freeze ≥ 3 s (hovering at a pot, not closing)", lambda s: f"{s['eps_freeze_open']}"),
            ("Episodes with a pot tipped onto its side", lambda s: f"{s['tipped']}"),
            ("Steps from closing on a pot to lifting it (median)", lambda s: f(s["hold"], 0)),
            ("…lifts that took ≥ 60 steps (3 s)", lambda s: f"{s['hold_long']} / {s['holds']}"),
            ("Grasp attempts on the 2nd pot (mean)", lambda s: f(s["attempts2"], 2)),
            ("…of which closed on air (mean)", lambda s: f(s["missed2"], 2)),
            ("Lifted the 2nd pot on the first try", lambda s: f"{s['first_try2']} / {s['lifted_first']}"),
            ("Same outcome as the gallery run", lambda s: f"{s['agree_gallery']} / {s['episodes']}")]
    summary = "".join(f"<tr><th>{k}</th><td>{fn(stats['base'])}</td><td>{fn(stats['guided'])}</td></tr>" for k, fn in rows)
    modes = "".join(f"<tr><th>{MODE_TEXT[m]}</th><td>{stats['base']['modes'][m]}</td><td>{stats['guided']['modes'][m]}</td></tr>"
                    for m in MODES)
    gtab = pd.crosstab(pairs.gal_group, pairs.group)
    gcols = list(gtab.columns)
    gtable = ("<tr><th>gallery ↓ / rerun →</th>" + "".join(f"<td><b>{html.escape(c)}</b></td>" for c in gcols) + "</tr>" +
              "".join(f"<tr><th>{html.escape(i)}</th>" + "".join(f"<td>{gtab.loc[i, c]}</td>" for c in gcols) + "</tr>"
                      for i in gtab.index))
    return dict(summary=summary, modes=modes, gtable=gtable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gallery-base", required=True)
    ap.add_argument("--gallery-guided", required=True)
    a = ap.parse_args()
    out = Path(a.dir)
    pairs, stats = build(out, a.gallery_base, a.gallery_guided)
    (out / "report_stats.json").write_text(json.dumps(stats, indent=1, default=float))
    pairs.to_csv(out / "pairs.csv", index=False)
    tb = tables(pairs, stats)
    summary, modes, gtable = tb["summary"], tb["modes"], tb["gtable"]

    order = {"fixed by guidance": 0, "broken by guidance": 1, "both fail": 2, "both succeed": 3}
    pairs = pairs.sort_values(by=["group", "init_state", "repeat"], key=lambda s: s.map(order) if s.name == "group" else s)
    cards = []
    for x in pairs.itertuples():
        def side(lab, suf):
            r = getattr(x, "video" + suf), getattr(x, "mode" + suf), getattr(x, "attempts2" + suf), getattr(x, "missed2" + suf)
            fz = getattr(x, "freeze_closed" + suf), getattr(x, "freeze_open" + suf)
            att = f" · freeze closed {int(fz[0])} / open {int(fz[1])} steps"
            return (f'<figure><figcaption><b>{lab}</b> · {MODE_TEXT[r[1]]}{att}</figcaption>'
                    f'<video src="{r[0]}" controls muted preload="none" playsinline></video></figure>')
        cards.append(f'<section class="pair" data-group="{x.group}"><h3>state {x.init_state} · repeat {x.repeat} '
                     f'<span class="tag">{x.group}</span> <span class="muted">gallery: {x.gal_group}</span>'
                     f'<button onclick="playPair(this)">play both</button></h3>'
                     f'<div class="vids">{side("base", "_b")}{side("guided", "_g")}</div></section>')
    groups = "".join(f'<button onclick="filt(\'{k}\')">{k} ({int((pairs.group == k).sum())})</button>' for k in order)
    page = TEMPLATE.replace("%SUMMARY%", summary).replace("%MODES%", modes).replace("%GTABLE%", gtable) \
        .replace("%GROUPS%", groups).replace("%CARDS%", "\n".join(cards)).replace("%N%", str(len(pairs)))
    (out / "index.html").write_text(page)
    print(json.dumps(stats, indent=1, default=float))


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gripper view: base vs guided</title>
<style>
:root{--bg:#f9f9f7;--fg:#0b0b0b;--muted:#52514e;--line:#dcdbd6;--card:#fff;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ececea;--muted:#a3a29d;--line:#3a3936;--card:#1e1e1c;--accent:#6aa6ee}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1100px;margin:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}h3{font-size:14px;margin:0 0 8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
p,li{color:var(--muted);max-width:75ch}table{border-collapse:collapse;margin:8px 0}th,td{padding:4px 12px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}
th{font-weight:500}thead td{font-weight:600}.pair{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}
.vids{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media (max-width:760px){.vids{grid-template-columns:1fr}}
video{width:100%;border-radius:4px;background:#000}figure{margin:0}figcaption{font-size:13px;color:var(--muted);margin-bottom:4px}
.tag{background:var(--accent);color:#fff;border-radius:10px;padding:0 8px;font-size:12px}.muted{color:var(--muted);font-weight:400;font-size:12px}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:3px 10px;cursor:pointer}
.bar{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.key span{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;border-radius:2px}
</style></head><body>
<h1>Gripper view: base vs residual guidance</h1>
<p>Moka-pot task (LIBERO-10 task 8), %N% matched-noise pairs: 10 start states × repeats 0–2, rerun with the wrist camera and gripper logged.
Base = pi05 base; guided = residual guidance, shift 0.99 (the gallery's policies). Each video: third-person view | wrist camera, and below it a timeline of the
gripper close command (grey), grasp attempts (▼ <span style="color:#1f8a4c">lifted</span>, <span style="color:#d03b3b">closed on air</span>,
<span style="color:#c98a00">held but no lift</span>), a bottom strip marking freezes (<span style="color:#7b4bc9">hovering open</span>, <span style="color:#c98a00">holding closed</span>), finger opening and both pots' heights. Dashed lines mark when a pot lands on the stove.</p>
<h2>Freezes, lifts and grasps</h2><table><thead><tr><td></td><td>base</td><td>guided</td></tr></thead>%SUMMARY%</table>
<p>A freeze step: the gripper moves under 1 cm over the next 0.5 s while within 12 cm of a pot that is neither lifted nor on the stove.
An attempt is a close command within 12 cm of a pot; "on air" means the fingers shut to under 1 cm. Second-pot rows count episodes that lifted a first pot.
Tipped: the pot's up-axis ends below 0.5 (lying on its side), from the final simulator state.</p>
<h2>How each episode ended</h2><table><thead><tr><td></td><td>base</td><td>guided</td></tr></thead>%MODES%</table>
<h2>Rerun vs gallery outcome</h2><p>Matched-noise episodes are not bit-reproducible, so a rerun pair can land in a different group from its gallery pair.</p><table>%GTABLE%</table>
<h2>Pairs</h2><div class="bar"><button onclick="filt('')">all</button>%GROUPS%</div>
%CARDS%
<script>
function filt(g){document.querySelectorAll('.pair').forEach(e=>e.style.display=(!g||e.dataset.group===g)?'':'none')}
function playPair(b){const v=b.closest('.pair').querySelectorAll('video');v.forEach(x=>{x.currentTime=0;x.play()})}
</script></body></html>"""

if __name__ == "__main__":
    main()
