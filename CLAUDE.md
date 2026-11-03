Read README.md first (project goal, ground rules, commands), then docs/CURRENT.md (live jobs,
open decision, resume steps), then docs/INDEX.md to find the one document a question needs.
docs/findings.md holds settled conclusions. Dated reports and log entries are evidence, not live status.

Documentation (layout in docs/INDEX.md):
- End of every session: overwrite docs/CURRENT.md (do not append) and add a short dated entry to
  docs/libero_plan.md `## Log` that points to the report folder.
- Each finished experiment: reports/YYYY-MM-DD-<slug>/ with report.md (question, setup, tables,
  verdict with observations and interpretations separated, caveats, next), its data files, and a
  summary.json for the weekly page. If a conclusion changes, correct docs/findings.md in place and
  update docs/INDEX.md.
- PROGRESS.md is the Mulligan stage checklist; keep it updated when a step is finished.
- Weekly page: `.venv/bin/python scripts/build_week_report.py --start <date> --end <date>`, then
  `scripts/publish_week_report.sh <end>-week` (public bucket: review first; only under
  reports/karenvo/value-learning/).

- Act as a research collaborator: explain key concepts, verify that results mean what we think.
- Use `.venv/bin/python`; run `pytest -q` after changing the pipeline.
- Ask before changing labels, splits, or the research objective; make routine choices yourself
  and document them, marking each as paper-specified or our choice.
