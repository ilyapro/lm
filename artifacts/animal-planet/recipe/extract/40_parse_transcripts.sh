#!/usr/bin/env bash
# Step 40: stream-parse the alt transcripts for living-memory MCP events.
#
# The parser script is sent to alt on stdin (`ssh alt python3 -`) so nothing
# is ever written on alt; its stdout (JSONL) is captured straight into the
# private staging dataset. The run summary line is duplicated into
# parse_summary.json locally.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

out="$AP_STAGING/transcripts/lm_events.jsonl"
capture_started=$(utc_now)
ssh -o BatchMode=yes "$AP_ALT_HOST" python3 - < transcript_parser.py > "$out"
capture_finished=$(utc_now)

OUT="$out" CAPTURE_STARTED="$capture_started" CAPTURE_FINISHED="$capture_finished" \
SUMMARY="$AP_STAGING/transcripts/parse_summary.json" python3 - <<'PY'
import json, os
run_summary = None
kinds = {}
with open(os.environ["OUT"], encoding="utf-8") as fh:
    for line in fh:
        record = json.loads(line)
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
        if record["kind"] == "run_summary":
            run_summary = record
doc = {
    "capture_started": os.environ["CAPTURE_STARTED"],
    "capture_finished": os.environ["CAPTURE_FINISHED"],
    "record_kinds": kinds,
    "run_summary": run_summary,
}
with open(os.environ["SUMMARY"], "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
print(json.dumps(doc["run_summary"], indent=2))
PY
