"""Build the push-direction page: page_template.html + analysis.json + page_text.json -> push_direction.html [2026-11-21].
Usage: python3 reports/2026-11-21-push-direction/build_page.py <analysis.json>"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads(Path(sys.argv[1]).read_text())
T = json.loads((HERE / "page_text.json").read_text())

LABELS = {"900_r0": "900 r0", "900_r1": "900 r1", "900_r2": "900 r2", "b100_r0": "b100 r0", "b100_r1_s0": "b100 r1 seed 0",
          "b100_r1_s1": "b100 r1 seed 1", "b100_r2": "b100 r2", "b100cv_r0": "b100cv r0", "b100cv_r1": "b100cv r1",
          "b50cv_r0": "b50cv r0", "b50cv_r1_s0": "b50cv r1 seed 0", "b50cv_r1_s1": "b50cv r1 seed 1",
          "mt_r0": "mt100cv r0", "mt_r1_s0": "mt100cv r1 seed 0", "mt_r1_s1": "mt100cv r1 seed 1",
          "mtos_r1": "mt100cv r1, 91% base", "mtos77_r1": "mt100cv r1, 77% base"}
CHAINS = [
    {"id": "b50cv", "label": "b50cv · 50 base", "r0": "b50cv_r0", "r1": ["b50cv_r1_s0", "b50cv_r1_s1"],
     "note": "Drops at round 1 with both seeds."},
    {"id": "mt", "label": "mt100cv · 50 + 50 base", "r0": "mt_r0", "r1": ["mt_r1_s0", "mt_r1_s1"],
     "note": "Two-task chain, moka frames shown. Drops with both seeds."},
    {"id": "b100", "label": "b100 · 100 base", "r0": "b100_r0", "r1": ["b100_r1_s0", "b100_r1_s1"],
     "note": "Seed 0 dropped, seed 1 did not."},
    {"id": "b100cv", "label": "b100cv · 100 base", "r0": "b100cv_r0", "r1": ["b100cv_r1"], "note": "Small drop, one seed."},
    {"id": "900", "label": "900 chain", "r0": "900_r0", "r1": ["900_r1"], "note": "No drop."},
]
KIND = {"r1 vs r0": "Round 1 vs its own round 0", "seed": "Two retrains on the same round-1 data",
        "r2 vs r1": "Round 2 vs round 1", "r0 vs r0": "Round 0 of a chain vs the 900 chain's round 0"}
order = list(KIND)
A["pairs"].sort(key=lambda r: order.index(r["kind"]) if r["kind"] in order else 99)
for r in A["pairs"]:
    r["eval"] = T["evals"].get(f"{r['a']}|{r['b']}", "")
A.update(chains=CHAINS, labels=LABELS, kindLabels=KIND, text=T)
for p in A["policies"].values():  # keep the page small
    p.pop("dim_mean_raw", None); p.pop("dim_rms_raw", None)
for r in A["pairs"]:
    r.pop("cos_hist", None); r["all"].pop("cos_hist", None)
html = (HERE / "page_template.html").read_text().replace("/*DATA*/null", json.dumps(A, separators=(",", ":")))
(HERE / "push_direction.html").write_text(html)
print(f"wrote {HERE / 'push_direction.html'} ({len(html) / 1024:.0f} KB)")
