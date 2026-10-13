# Living Memory S3 backup

The user timer `living-memory-backup.timer` runs at 03:00 Asia/Novosibirsk.
Every run snapshots the active SQLite database through the SQLite backup API,
checks its integrity, uploads a private staging object, downloads and checks
that object, then replaces `s3://lm-private-backup-sfx/global.sqlite3`. It
downloads and checks the final object before recording success. The staging
object is removed after each attempt. Bucket versioning must stay disabled.

Credentials live only in `~/.config/living-memory/s3cmd-backup.conf` (mode 0600).
The bucket and this file must be prepared before enabling the timer. The
script writes `~/.local/state/living-memory-backup/status.json` with the last
successful UTC time, size, checksum, and last error. Inspect timer and service
with `systemctl --user status living-memory-backup.timer` and
`journalctl --user -u living-memory-backup.service`.

To recover after losing local data, stop the Living Memory service on the
replacement host, download `global.sqlite3` from the bucket using the private
S3 configuration, check its SHA-256 against `status.json` if that file survived,
run `PRAGMA integrity_check`, and place it at
`~/.local/share/living-memory/global.sqlite3` with the service user's ownership.
Remove any old `global.sqlite3-wal` and `global.sqlite3-shm` on the replacement
host before starting the service. Never copy WAL files from an unrelated DB.
