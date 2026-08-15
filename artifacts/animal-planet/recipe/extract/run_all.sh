#!/usr/bin/env bash
# Rebuild the ENTIRE private staging dataset from the live sources in one
# command:
#
#   bash artifacts/animal-planet/recipe/extract/run_all.sh
#
# Requires: ssh alias `alt` (read-only access), python3, and the local LM DB
# under ~/.local/share/living-memory/. Writes ONLY to $AP_STAGING
# (default /home/sfx/.cache/ap-audit/staging) and $AP_TMP (default
# /tmp/ap-audit). Never writes anything on alt. See ../01-extract.md.
#
# Sources are LIVE: full-table totals and post-window activity drift between
# runs; the window-bound cross-checks (W1/W2/W3) are the stable ones.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 10: snapshot alt LM DB =="
bash 10_snapshot_alt_db.sh
echo "== 20: export alt DB audit slice =="
python3 20_export_alt_db.py
echo "== 30: transcript inventory =="
bash 30_transcript_inventory.sh
echo "== 40: parse transcripts =="
bash 40_parse_transcripts.sh
echo "== 50: snapshot local LM DB =="
bash 50_snapshot_local_db.sh
echo "== 60: export local holdout workloads =="
python3 60_export_local_holdout.py
echo "== 70: match + cross-check + METADATA.json =="
python3 70_build_metadata.py
