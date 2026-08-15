#!/usr/bin/env bash
# Step 10: snapshot the alt Living Memory SQLite DB to local /tmp.
#
# Method (precedent: artifacts/baseline.md, pinned in LM trace
# 01KZW27BVVSZHZFH3J73PB4RE3): scp db+wal+shm of the LIVE db to a local
# copy, run PRAGMA wal_checkpoint(TRUNCATE) on the COPY, verify
# PRAGMA integrity_check, then record SHA-256 of the checkpointed copy plus
# the capture timestamps. alt is never written to; there is no sqlite3 CLI
# on alt and none is needed — the checkpoint runs locally via python3.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

DEST="$AP_TMP/alt"
DB="$DEST/global.sqlite3"

capture_started=$(utc_now)
remote_stat=$(ssh -o BatchMode=yes "$AP_ALT_HOST" \
  "stat -c '%n|%s|%Y' $AP_ALT_DB_DIR/global.sqlite3 $AP_ALT_DB_DIR/global.sqlite3-wal $AP_ALT_DB_DIR/global.sqlite3-shm 2>/dev/null || true")

rm -f "$DB" "$DB-wal" "$DB-shm"
scp -q "$AP_ALT_HOST:$AP_ALT_DB_DIR/global.sqlite3" "$DB"
scp -q "$AP_ALT_HOST:$AP_ALT_DB_DIR/global.sqlite3-wal" "$DB-wal" || echo "note: no -wal file on alt"
scp -q "$AP_ALT_HOST:$AP_ALT_DB_DIR/global.sqlite3-shm" "$DB-shm" || echo "note: no -shm file on alt"
capture_finished=$(utc_now)

wal_bytes=$(stat -c %s "$DB-wal" 2>/dev/null || echo 0)

checkpoint_report=$(python3 - "$DB" <<'PY'
import json, sqlite3, sys
path = sys.argv[1]
conn = sqlite3.connect(path)
journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
ckpt = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
conn.close()
print(json.dumps({
    "journal_mode": journal_mode,
    "wal_checkpoint": list(ckpt),
    "integrity_check": integrity,
}))
if integrity != "ok":
    raise SystemExit(f"integrity_check failed: {integrity}")
PY
)

sha=$(sha256sum "$DB" | cut -d' ' -f1)
db_bytes=$(stat -c %s "$DB")

REMOTE_STAT="$remote_stat" CAPTURE_STARTED="$capture_started" \
CAPTURE_FINISHED="$capture_finished" CHECKPOINT_REPORT="$checkpoint_report" \
SHA256="$sha" DB_BYTES="$db_bytes" WAL_BYTES_BEFORE_CKPT="$wal_bytes" \
SNAPSHOT_PATH="$DB" ALT_HOST="$AP_ALT_HOST" ALT_DB_DIR="$AP_ALT_DB_DIR" \
OUT="$AP_STAGING/alt-db/snapshot.json" python3 - <<'PY'
import json, os
remote_files = []
for line in os.environ["REMOTE_STAT"].splitlines():
    if not line.strip():
        continue
    name, size, mtime = line.split("|")
    remote_files.append({"path": name, "bytes": int(size), "mtime_epoch": int(mtime)})
doc = {
    "source_host": os.environ["ALT_HOST"],
    "source_dir": os.environ["ALT_DB_DIR"],
    "source_files_at_capture": remote_files,
    "capture_started": os.environ["CAPTURE_STARTED"],
    "capture_finished": os.environ["CAPTURE_FINISHED"],
    "method": "scp db+wal+shm of live DB -> local copy; PRAGMA wal_checkpoint(TRUNCATE) on the copy; PRAGMA integrity_check; sha256 of checkpointed copy",
    "snapshot_path": os.environ["SNAPSHOT_PATH"],
    "snapshot_bytes": int(os.environ["DB_BYTES"]),
    "wal_bytes_before_checkpoint": int(os.environ["WAL_BYTES_BEFORE_CKPT"]),
    "checkpoint": json.loads(os.environ["CHECKPOINT_REPORT"]),
    "sha256": os.environ["SHA256"],
}
with open(os.environ["OUT"], "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"snapshot ok: {doc['snapshot_bytes']} bytes sha256={doc['sha256'][:16]}... integrity={doc['checkpoint']['integrity_check']}")
PY
