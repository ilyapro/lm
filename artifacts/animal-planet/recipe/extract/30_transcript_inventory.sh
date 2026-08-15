#!/usr/bin/env bash
# Step 30: per-file inventory of the alt animal-planet transcripts.
#
# Runs entirely over ssh with inline commands (nothing is written on alt):
# path, size, mtime via stat; content hash via remote sha256sum. Output is
# converted locally to JSONL + a summary. Transcripts are live files, so the
# inventory is a moment-in-time capture; the capture timestamp is recorded.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

capture_started=$(utc_now)
ssh -o BatchMode=yes "$AP_ALT_HOST" bash -s <<'REMOTE' > "$AP_TMP/transcript_inventory.tsv"
set -euo pipefail
cd ~/.claude/projects
shopt -s nullglob
for f in *animal-planet*/*.jsonl; do
  stat_out=$(stat -c '%s %Y' -- "$f")
  hash=$(sha256sum -- "$f" | cut -d' ' -f1)
  printf '%s\t%s\t%s\n' "$f" "$stat_out" "$hash"
done
REMOTE
capture_finished=$(utc_now)

CAPTURE_STARTED="$capture_started" CAPTURE_FINISHED="$capture_finished" \
TSV="$AP_TMP/transcript_inventory.tsv" ALT_HOST="$AP_ALT_HOST" \
OUT_DIR="$AP_STAGING/transcripts" python3 - <<'PY'
import json, os
from datetime import datetime, timezone

tsv = os.environ["TSV"]
out_dir = os.environ["OUT_DIR"]
files = []
total_bytes = 0
with open(tsv, encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        path, rest, sha = line.split("\t")
        size, mtime = rest.split(" ")
        record = {
            "path": path,
            "bytes": int(size),
            "mtime_epoch": int(mtime),
            "mtime_iso": datetime.fromtimestamp(int(mtime), tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "sha256": sha,
        }
        total_bytes += record["bytes"]
        files.append(record)
files.sort(key=lambda r: r["path"])
with open(os.path.join(out_dir, "inventory.jsonl"), "w", encoding="utf-8") as fh:
    for record in files:
        fh.write(json.dumps(record) + "\n")
summary = {
    "source_host": os.environ["ALT_HOST"],
    "source_glob": "~/.claude/projects/*animal-planet*/*.jsonl",
    "capture_started": os.environ["CAPTURE_STARTED"],
    "capture_finished": os.environ["CAPTURE_FINISHED"],
    "file_count": len(files),
    "total_bytes": total_bytes,
    "dir_count": len({r["path"].split("/")[0] for r in files}),
    "hash_algorithm": "sha256 (remote sha256sum)",
}
with open(os.path.join(out_dir, "inventory_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
    fh.write("\n")
print(json.dumps(summary, indent=2))
PY
