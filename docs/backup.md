# Living Memory S3 backup

The ae loop goal `daily-living-memory-backup` runs at 03:00 Asia/Novosibirsk
and invokes `python3 /home/sfx/p/lm/scripts/backup_living_memory_s3.py`.
Every run snapshots the active SQLite database through the SQLite backup API,
checks its integrity, uploads a private staging object, downloads and checks
that object, then replaces `s3://lm-private-backup-sfx/global.sqlite3.gz`. It
downloads and checks the final object before recording success. The staging
object is removed after each attempt. Bucket versioning must stay disabled.
The gzip archive reduces transfer time; checks run against the unpacked SQLite
database as well as the downloaded archive.

Credentials live only in `~/.config/living-memory/s3cmd-backup.conf` (mode 0600).
The bucket and this file must be prepared before running the backup. The
script writes `~/.local/state/living-memory-backup/status.json` with the last
successful UTC time, size, checksum, and last error. The loop goal's schedule
is stored in `~/p/ae/projects/ae/state/goals/daily-living-memory-backup/loop_schedule`.

To recover after losing local data, stop the Living Memory service on the
replacement host, download `global.sqlite3.gz` from the bucket using the private
S3 configuration, unpack it, check the SQLite file's SHA-256 against
`status.json` if that file survived, run `PRAGMA integrity_check`, and place it at
`~/.local/share/living-memory/global.sqlite3` with the service user's ownership.
Remove any old `global.sqlite3-wal` and `global.sqlite3-shm` on the replacement
host before starting the service. Never copy WAL files from an unrelated DB.
