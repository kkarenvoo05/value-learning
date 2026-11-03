#!/usr/bin/env bash
# Upload a built weekly page to the PUBLIC bucket (anyone with the link can read it).
# Usage: scripts/publish_week_report.sh <name>      e.g. 2026-11-03-week
# Writes only gs://zima-robot/reports/karenvo/value-learning/<name>/. Review the page first: no credentials,
# no private paths beyond lab storage, no real-robot frames unless cleared.
set -euo pipefail
NAME=${1:?usage: $0 <name>}
SRC=$(cd "$(dirname "$0")/.." && pwd)/reports/site/$NAME
[ -f "$SRC/index.html" ] || { echo "build first: $SRC/index.html missing"; exit 1; }
G=$(command -v gcloud || echo /iris/u/ankile/karenvo/google-cloud-sdk/bin/gcloud)
"$G" storage cp -r --cache-control=no-cache "$SRC"/* "gs://zima-robot/reports/karenvo/value-learning/$NAME/"
echo "https://storage.googleapis.com/zima-robot/reports/karenvo/value-learning/$NAME/index.html"
