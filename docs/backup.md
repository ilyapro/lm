# Living Memory S3 backup

The existing ae loop goal `daily-living-memory-backup` is the sole owner of real
uploads and external reconciliation. Its schedule is 03:00 Asia/Novosibirsk,
stored in `~/p/ae/projects/ae/state/goals/daily-living-memory-backup/loop_schedule`.
The ordinary command, with no arguments, is:

```sh
python3 /home/sfx/p/lm/scripts/backup_living_memory_s3.py
```

The command owns preservation, replacement, verification, rollback and remote
cleanup under `~/.local/state/living-memory-backup/backup.lock`. Invoke the
command directly: no generated wrapper, imported `run()`, second lock owner or
post-success remote verification is required. In particular, a consumer must
not roll back a verified FINAL because a later cleanup/listing failed.

Credentials live only in `~/.config/living-memory/s3cmd-backup.conf` (mode 0600).
The bucket and configuration must already exist. Admission checks require the
bucket's private owner-only ACL, no bucket policy, disabled versioning, and an
inventory containing only FINAL (or no objects for the first backup), with no
incomplete multipart uploads. Unknown objects, retained safety from an older
wrapper, unavailable metadata and failed listings block admission; they are
never permission to delete, overwrite or repeat an unknown upload. All created
objects use `--acl-private`, and downloaded objects' privacy is checked.

Metadata reads use the installed s3cmd Python transport directly, in a bounded
subprocess, instead of parsing `s3cmd info` display text. Versioning admission
requires HTTP 200 with an empty `VersioningConfiguration`; enabled, suspended,
denied, malformed and unavailable responses block backup. Only HTTP 404 with
`NoSuchBucketPolicy` confirms no policy. Every privacy check still requires
exactly the owner's `sfx: FULL_CONTROL` grant, with its canonical ID matching
the ACL owner, and its own bucket-policy read.

The command reads HEAD metadata only for the previous FINAL/safety comparison;
download verification checks ACL/policy again and verifies the bytes directly.
A valid original MD5 in `s3cmd-attrs` takes precedence over ETag; missing or
malformed MD5, including a multipart ETag without original MD5, fails closed.
It does not query requester-pays, ownership controls, public-access-block,
lifecycle or CORS. Location discovery needed by signing and endpoint redirects
remain in s3cmd's transport. With a configured region and no redirects, the
ordinary replacement path uses 15 metadata reads (one versioning, six ACL, six
policy, two HEAD), without sharing privacy checks between stages.

The script snapshots the live SQLite database through the SQLite backup API with
a read-only source connection. It neither replaces the working database nor
stops Living Memory. It checks snapshot integrity and compresses it with gzip.
Before replacing an existing `s3://lm-private-backup-sfx/global.sqlite3.gz`
(FINAL), it copies FINAL to a unique private `.previous-*.sqlite3.gz` safety
object. It checks the copy's ACL, size and content MD5 against the previous
object's metadata, downloads it, checks that content MD5, decompresses it and
runs SQLite `integrity_check`. Its archive and SQLite SHA-256 values are retained
for rollback verification. Missing content MD5 (including an opaque multipart
ETag without s3cmd's content-MD5 metadata) blocks replacement.

The new snapshot is uploaded to `.incoming.sqlite3.gz`, downloaded, checked for
both archive and SQLite SHA-256, and checked with `integrity_check`. Only then
is it copied privately to FINAL. The same checks, including privacy and an
actual download, verify FINAL. When a confirmed replacement fails verification,
the script restores the verified safety copy and downloads/checks the restored
FINAL before allowing cleanup. Failed restoration preserves safety and staging.
If there was no original FINAL, that is recorded separately: failed verification
retains checked staging and the unverified FINAL without inventing a rollback.

`status.json` in the state directory is private and atomically persisted, with
file and directory fsync. It retains `last_success`, SQLite `sha256`, `bytes`,
archive `archive_sha256`, `last_attempt`, `last_error`, and the current `result`.
The prior success is historical: after a failed or unknown replacement it does
not by itself identify the current remote bytes. Read `result`, `rollback`,
`cleanup`, and any retained `transaction` together.

A verified FINAL is recorded **before** cleanup. Remote cleanup deletes staging,
lists multipart uploads once, aborts only uploads for this operation's staging
key, then deletes safety and checks the remaining object inventory. Completion
requires exactly FINAL, or the original empty inventory after a failed first
attempt. The command does not repeat its multipart or FINAL verification after
completion. A cleanup read failure preserves the verified result and remaining
copies; safety may already be deleted if the last object listing fails. It does
not claim that cleanup completed. No outer rollback should undo this result.

| Exit | Meaning |
| --- | --- |
| `0` | FINAL verified and remote cleanup complete. This can also complete cleanup of an earlier verified attempt; check `last_success`. |
| `1` | No new verified result, admission/lock failure, or failed rollback verification. The status explains whether FINAL was unchanged, rollback was verified, or recovery remains unresolved. A busy lock leaves status unchanged. |
| `2` | FINAL verified; cleanup unverified. Success/checksums remain valid facts of this attempt. |
| `3` | UNKNOWN mutation or interrupted/unresolved transaction. If `result=verified`, only cleanup is unresolved; the verified result remains recorded. |

Before each remote mutation, the script persists `transaction.pending` with its
operation and target (and upload ID for an abort). It clears that intent only
with a confirmed response and the corresponding checkpoint. Transport errors,
timeouts and crashes can leave an applied effect with no confirmed response.
They do not trigger automatic rollback, deletion or replay. s3cmd retries are
disabled; reads have a 120-second limit and transfers/copies a 3600-second limit.

Transport failures add `transport` to the existing `last_error`, `cleanup_error`
or `rollback_error` record. The human-readable `message` includes the same safe
cause. `transport.operation` identifies the command (or `metadata` check),
`exit_code` is present when the child exited, and `chain` contains at most eight
exception entries, outermost first. Each entry contains only a local `class`
(at most 64 ASCII identifier characters), optional signed 32-bit integer `errno`
and optional integer HTTP `status` (100–599). For example:

```json
{"operation":"put","exit_code":2,"chain":[{"class":"S3UploadError"},{"class":"TimeoutError","errno":110}]}
```

The installed s3cmd CLI runs inside the existing bounded subprocess. Observers
on its request/upload/multipart/download methods retain the escaping exception
chain before CLI error handling discards it. Signing, redirects and transfer
decisions stay in s3cmd. A separate private descriptor carries only this typed
data; raw stderr, exception messages/arguments, URIs, headers, bodies,
configuration and credentials are never included in the diagnostic. The parent
validates its shape and bounds before recording it.

An empty `chain` and `cause unavailable` mean no valid diagnostic was received;
exit 2 alone never establishes a timeout. A root class alone establishes no
underlying cause: s3cmd can erase the HTTP status when replacing a transfer's
503 error. Available `__cause__` takes precedence over `__context__`, including
context suppressed from displayed tracebacks. Cycles and chains beyond eight
entries stop collection. `TimeoutExpired` identifies the whole-subprocess
deadline; `TimeoutError` identifies an observed library exception. Neither
localizes a network/server phase or diagnoses an earlier attempt. These fields
grant no retry or recovery permission and do not establish a fresh backup.

On the next ordinary invocation, a verified/rolled-back/unchanged transaction
with no pending mutation resumes only unfinished cleanup, preserving its
original success time and completed cleanup checkpoints. It does not take
another snapshot, upload again or download FINAL again. A pending mutation,
failed rollback or other interrupted transaction stops before any remote call.
The existing backup owner must reconcile the specific recorded effect with the
recipient, preserving status and safety until the outcome and necessary copies
are established. Do not clear status, overwrite safety, or rerun the mutation
just because a transport timed out or a later read sees no object. Unreadable
status is also retained and blocks execution. There is no automatic recovery
poller or second backup controller.

For read-only diagnosis of an unresolved transaction, the existing backup owner
can check the current FINAL through the same transport under `backup.lock`:

```sh
python3 - <<'PY'
import fcntl, json, os, runpy, tempfile
from pathlib import Path
b = runpy.run_path('/home/sfx/p/lm/scripts/backup_living_memory_s3.py')
os.umask(0o077)
try:
    with (b['STATE_DIR'] / 'backup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='read-only-', dir=b['STATE_DIR']) as work:
            observed = b['verify_download'](b['FINAL'], Path(work), 'observed')
    print(json.dumps({'read_only_final': observed}))
except Exception as error:
    detail = getattr(error, 'transport', {'chain': b['exception_chain'](error)})
    print(json.dumps({'read_only_error': detail}))
    raise SystemExit(1)
PY
```

Exit 0 verifies only the existing FINAL's privacy and downloaded bytes; exit 1
reports the safe failure. This reads no live database, changes no remote object
or status, and does not replay a pending mutation. It cannot reconstruct an old
failure or prove a new upload succeeded. Reconciliation still requires fresh
object/multipart inventory and the specific transaction's retained hashes;
the owner's `objects()` and `multipart()` reads can use the same locked boundary.
Retain their private results locally. Never reuse an old upload ID without a
current inventory or call `run()` to diagnose an UNKNOWN effect.

The standalone script requires no LM service reload. Its offline regression is
private source alongside the script:

```sh
scripts/test.sh scripts/test_backup_living_memory_s3.py
```

It executes the real `__main__`/lock/status path with disposable SQLite files and
fake S3 transport. It forbids network and non-fixture SQLite access and uses no
production credentials. Metadata scenarios pass raw HTTP responses through the
installed s3cmd request/signing/error handling, including rejected 403/5xx and
malformed responses, confirmed absence, redirects and region discovery. The
Transfer scenarios execute the installed CLI and upload/multipart/download/
request exception paths with fake HTTP, distinguishing timeout, connection,
HTTP and unavailable causes. A socket-forbidden real subprocess checks the
diagnostic descriptor. Sensitive fixture messages stay out of status and
reported output. The tests need the installed `S3` Python package, as do the
transport subprocesses.
This proves local semantics; the existing loop confirms
fresh real backup bytes during its next ordinary work.

To recover after losing local data, stop the Living Memory service on the
replacement host, download `global.sqlite3.gz` from the bucket using the private
S3 configuration, unpack it, check the SQLite file's SHA-256 against a verified
matching record in `status.json` if available, run `PRAGMA integrity_check`, and
place it at `~/.local/share/living-memory/global.sqlite3` with the service user's
ownership. If status describes a failed/unknown replacement, reconcile FINAL
and its retained safety first. Remove any old `global.sqlite3-wal` and
`global.sqlite3-shm` on the replacement host before starting the service. Never
copy WAL files from an unrelated DB.
