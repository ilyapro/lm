#!/usr/bin/env bash
# Step 50: snapshot the LOCAL Living Memory SQLite DB (holdout workload
# source) into /tmp, checkpoint the copy, record sha256 + capture time.
# Same method as step 10 but with a local cp instead of scp.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

DEST="$AP_TMP/local"
DB="$DEST/global.sqlite3"
SRC="$AP_LOCAL_DB_DIR/global.sqlite3"

capture_started=$(utc_now)
src_stat=$(stat -c '%n|%s|%Y' "$SRC" "$SRC-wal" "$SRC-shm" 2>/dev/null || true)
rm -f "$DB" "$DB-wal" "$DB-shm"
cp "$SRC" "$DB"
cp "$SRC-wal" "$DB-wal" 2>/dev/null || echo "note: no -wal file"
cp "$SRC-shm" "$DB-shm" 2>/dev/null || echo "note: no -shm file"
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

SRC_STAT="$src_stat" CAPTURE_STARTED="$capture_started" \
CAPTURE_FINISHED="$capture_finished" CHECKPOINT_REPORT="$checkpoint_report" \
SHA256="$sha" DB_BYTES="$db_bytes" WAL_BYTES_BEFORE_CKPT="$wal_bytes" \
SNAPSHOT_PATH="$DB" SRC_PATH="$SRC" \
OUT="$AP_STAGING/local-db/snapshot.json" python3 - <<'PY'
import json, os
src_files = []
for line in os.environ["SRC_STAT"].splitlines():
    if not line.strip():
        continue
    name, size, mtime = line.split("|")
    src_files.append({"path": name, "bytes": int(size), "mtime_epoch": int(mtime)})
doc = {
    "source_host": "local",
    "source_path": os.environ["SRC_PATH"],
    "source_files_at_capture": src_files,
    "capture_started": os.environ["CAPTURE_STARTED"],
    "capture_finished": os.environ["CAPTURE_FINISHED"],
    "method": "cp db+wal+shm of live DB -> local copy; PRAGMA wal_checkpoint(TRUNCATE) on the copy; PRAGMA integrity_check; sha256 of checkpointed copy",
    "snapshot_path": os.environ["SNAPSHOT_PATH"],
    "snapshot_bytes": int(os.environ["DB_BYTES"]),
    "wal_bytes_before_checkpoint": int(os.environ["WAL_BYTES_BEFORE_CKPT"]),
    "checkpoint": json.loads(os.environ["CHECKPOINT_REPORT"]),
    "sha256": os.environ["SHA256"],
}
with open(os.environ["OUT"], "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"local snapshot ok: {doc['snapshot_bytes']} bytes sha256={doc['sha256'][:16]}... integrity={doc['checkpoint']['integrity_check']}")
PY
