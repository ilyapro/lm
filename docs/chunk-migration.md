# Chunk-embedding migration runbook

Moving embedding storage from one JSON vector per node (`nodes.embedding`) to many
float32 BLOB chunk vectors per node (`node_chunk_embeddings`, schema v6), on a
live database, without an outage.

The whole procedure is **additive first, destructive last**. Every step up to and
including the restart leaves `nodes.embedding` in place, so at every point before
the final step the previous release can be rolled back by restarting it.

Tool: `scripts/backfill_chunk_embeddings.py` (`backfill`, `verify`,
`drop-embedding-column`). It never defaults to a database — `--db` is always
explicit.

## Why the order is what it is

| Step | Server can be | Because |
| --- | --- | --- |
| backup | up | the sqlite3 backup API is consistent under a concurrent writer |
| backfill | up | it only INSERTs into `node_chunk_embeddings`; a SQLite authorizer denies every other write, so `nodes.embedding` cannot change |
| deploy + restart | restarting | the running process holds the old modules until it restarts |
| second backfill pass | up | nodes written during the window above were written by code that did not chunk |
| verify | up | read-only |
| drop the column | up (but see the gate) | irreversible; only safe once nothing reads the column |

**The gate on the last step.** `drop-embedding-column` must not run until the
deployed code reads *chunks* for the vector channel. As of this branch,
`retrieval.py` still scans `nodes.embedding` (`iter_embedding_rows`, superseded
but still the vector channel's source), and that method is written to yield
*nothing* once the column is gone. So a premature drop does not crash anything —
it silently turns recall into bm25 + graph + trigger with an empty vector
channel, which is far worse than a crash. Drop only after the max-pool vector
channel is deployed and verified.

## Measured numbers

Everything below was measured on 2026-08-17/18 on a `sqlite3`-backup-API snapshot
of the live database (12,862 active nodes, 16,683 total, 532 MB file + 159 MB WAL),
on this CPU box, and re-measured end to end by a full real run against that
snapshot.

| Quantity | Value |
| --- | --- |
| snapshot copy (backup API, server live) | **0.77–0.80 s**, 532,647,936 bytes |
| active nodes to chunk | 12,862 |
| chunks produced | **60,530** (4.71 per node, max 485 in one node) |
| chunk BLOB bytes | **92,974,080** (88.7 MiB) |
| backfill wall clock (full run, batch 64, encode batch 32) | **12m 12s** (732.2 s) |
| encode throughput | 82.7 chunks/s (17.6 nodes/s) |
| database file growth | 532,647,936 → 664,080,384 bytes (**+125.3 MiB**) |
| `nodes.embedding` JSON today | 134,788,051 bytes (128.5 MiB) over 16,606 rows |
| net embedding storage after the drop | 93.0 MB vs 134.8 MB — **31% less**, with 4.7× as many vectors |
| dry run (read-only rehearsal, real tokenizer) | 7.7 s |
| re-run after completion | 0.03 s of work (0 pending), plus ~5 s of model load |

The dry run predicted the real run exactly: 60,530 chunks and 92,974,080 bytes,
both. Use it as the plan, not as an estimate.

Disk during the migration: one backup (532 MB) plus the database's own growth
(+125 MiB, i.e. 89 MiB of BLOBs plus pages and index). The WAL stays small —
observed at 3–4 MB, because each node commits separately and autocheckpoint keeps
up. Budget **700 MB** free, of which 532 MB is the backup; put the backup on
another filesystem and it is ~130 MB.

The chunk count is higher than the 45k estimated when this work was planned
(measured 4.71 chunks per node against an estimated 3.53), so the BLOB total is
89 MiB rather than ~69 MB. It is still a third less than the JSON it replaces.
One node produces 485 chunks — worth knowing for the max-pool vector channel,
because a node with 485 windows gets 485 chances to match.

## Preconditions

* The chunk-writing code (schema v6: `storage.py` + `chunking.py`) is merged into
  the checkout that will be deployed.
* `python3 -c "import sentence_transformers"` works and the model is in the local
  HuggingFace cache. The backfill **refuses to start** if the model does not load,
  rather than filling the table with hash-fallback vectors from a different vector
  space. (`--allow-fallback-embeddings` exists for fixtures; never use it here.)
* ~1.1 GB free disk, plus room for the backup.
* You can restart the server (`systemctl --user restart living-memory.service`,
  see [deployment.md](deployment.md)).

## 0. Preflight — rehearse, read-only

```sh
DB=~/.local/share/living-memory/global.sqlite3

python3 scripts/backfill_chunk_embeddings.py verify --db "$DB"
python3 scripts/backfill_chunk_embeddings.py backfill --db "$DB" --dry-run \
    --json /tmp/chunk-dry-run.json
```

`verify` before the migration exits **1** and reports `chunk_table_present: false`
— that is the expected pre-migration state, not a failure. `--dry-run` opens the
database `mode=ro` with `PRAGMA query_only`, never constructs a `MemoryStore`
(whose constructor migrates), and reports the node count, chunk count and byte
total a real run would write. It needs no backup because it writes nothing.

Check that the dry run's `tokenizer` is `model:paraphrase-multilingual-MiniLM-L12-v2`
and not `heuristic`. Chunk boundaries depend on the tokenizer, so the backfill and
the query path must use the same one.

## 1. Backup — mandatory, and `cp` is not one

```sh
python3 scripts/backfill_chunk_embeddings.py backfill --db "$DB" \
    --backup ~/.local/share/living-memory/global.pre-chunk-$(date +%Y%m%d).sqlite3 \
    --json /tmp/chunk-backfill.json --progress-interval 30
```

The backup is taken by the same command that then does the work, through
`sqlite3.Connection.backup()` from a read-only source connection. That API copies
the committed database as one consistent snapshot **including everything sitting
in the WAL**, while the server keeps writing.

> `cp global.sqlite3 backup.sqlite3` is **not** a valid backup of this database.
> The live file carries a ~159 MB WAL holding the newest transactions; a copy
> without the `-wal` is a valid SQLite file that is silently missing them. The
> script rejects an `--existing-backup` holding less than 99% of the source's
> nodes for exactly this reason (`--min-backup-fraction`).

To reuse a backup you already have, pass `--existing-backup PATH` instead; it is
validated (`PRAGMA quick_check` plus the node-count comparison) before any write.

## 2. Backfill — with the server up

The command in step 1 is the backfill. It is safe to leave running while the
server serves reads and writes:

* it only ever INSERTs and DELETEs rows of `node_chunk_embeddings`; immediately
  after the migration a SQLite **authorizer** is installed that denies every
  write outside that table, so `nodes.embedding` physically cannot be rewritten
  or dropped by this process;
* it commits **one node per transaction** (`replace_node_chunks`), so the write
  lock is held for milliseconds at a time instead of for the whole run, and the
  WAL stays checkpointable;
* `PRAGMA busy_timeout` is raised to 30 s and each chunk write retries with
  backoff on `SQLITE_BUSY`;
* the first thing it does is the additive v5 → v6 migration: `CREATE TABLE
  node_chunk_embeddings` plus its indexes. Nothing about `nodes` changes, and
  nothing in the running server reads `metadata.schema_version` at request time
  (every migration is self-guarding on the actual DDL), so the running old
  process is unaffected by the version bump.

Expect **~12 minutes** and a progress line every `--progress-interval` seconds
with processed/total, rate and ETA, then a summary of nodes, chunks, bytes and
elapsed time (also written to `--json`).

**If it is interrupted** (Ctrl-C, SIGKILL, machine reboot, disk full), re-run the
same command with `--existing-backup` pointing at the backup you already took. It
resumes: progress lives in the database, not in a state file. A chunk row records
the `content_fingerprint` of the content it was cut from, so
`list_unchunked_nodes` reports exactly the nodes that are missing or stale, and
because each node commits atomically no node can be left holding a partial or
mixed-generation chunk set. Re-running after a *complete* run writes nothing.

If a node's content is edited by the server while the backfill is encoding it, the
chunks written are stamped with the fingerprint they were cut from, so that node
comes back as pending and is re-chunked from the new content. The summary reports
this as `nodes_rechunked_after_content_change`.

This was exercised against a real snapshot, not only against fixtures: a run
`SIGKILL`ed 66 seconds in left 1,408 nodes covered by 5,396 chunks with
`inconsistent_nodes: 0` and `nodes.embedding` byte-identical (SHA-256 over all
16,683 `(id, embedding)` pairs) to the pre-run backup; the next invocation
reported 11,454 pending — exactly the 12,862 minus what had committed — and
carried on from there without touching what was already done.

## 3. Deploy the new code

```sh
git -C /home/sfx/p/lm fetch origin && git -C /home/sfx/p/lm pull --ff-only
```

Nothing is served differently yet: the running process holds the old modules
until it restarts.

## 4. Restart the server

```sh
systemctl --user restart living-memory.service
systemctl --user is-active living-memory.service living-memory-tls.service
curl -fsS http://127.0.0.1:8765/health     # boot_id must differ from before
```

From here every write path chunks what it writes (`create_node`, `update_node`,
consolidation), and stale chunks are invalidated on content change.

## 5. Second backfill pass — catch the window

Nodes the old server wrote between step 2 and step 4 have no chunks: the code that
wrote them did not know about the chunk table. This pass is short (seconds to a
minute, proportional to the window).

```sh
python3 scripts/backfill_chunk_embeddings.py backfill --db "$DB" \
    --existing-backup ~/.local/share/living-memory/global.pre-chunk-YYYYMMDD.sqlite3
```

## 6. Verify

```sh
python3 scripts/backfill_chunk_embeddings.py verify --db "$DB" --json /tmp/chunk-verify.json
```

Exit code 0 requires all three of:

* `pending_nodes: 0` — every active node with non-whitespace content has current
  chunks;
* `inconsistent_nodes: 0` — no node holds two chunk generations, a gap in its
  ordinals, or two vector widths;
* one value in `dimensions` — a single vector width across the corpus, so the
  reader can reshape the scan into one matrix.

Also worth reading: `chunk_bytes` against `legacy_embedding_bytes` (the storage
win), and `zero_vector_chunks` (should be 0; a non-zero count means some content
encoded to a zero vector).

Then check recall itself — the vector channel is what this changes:

```sh
curl -fsS -H "Authorization: Bearer $LM_AUTH_TOKEN" \
     -H 'Content-Type: application/json' http://127.0.0.1:8765/mcp/ \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
          "params":{"name":"memory_recall","arguments":{"query":"chunk migration runbook"}}}'
```

## 7. Drop the JSON column — only now, and only behind the gate

Re-read **"The gate on the last step"** at the top: this is safe only once the
deployed vector channel reads chunks. It is the one irreversible step.

```sh
python3 scripts/backfill_chunk_embeddings.py drop-embedding-column --db "$DB" \
    --backup ~/.local/share/living-memory/global.pre-drop-$(date +%Y%m%d).sqlite3 \
    --yes --json /tmp/chunk-drop.json
```

`--yes` is mandatory, a backup is mandatory (take a fresh one — the pre-backfill
backup is now old), and the command refuses to proceed unless `verify` passes
(`--force` overrides that, at the cost of losing the un-chunked nodes from the
vector channel). The drop itself is `ALTER TABLE nodes DROP COLUMN embedding`
plus dropping the partial index that depended on the column.

The freed pages stay in the file until a `VACUUM`, which rewrites the whole
database and needs an exclusive lock — so it does **not** belong in the same
breath as the drop:

```sh
systemctl --user stop living-memory.service
python3 -c "import sqlite3; sqlite3.connect('$DB').execute('VACUUM')"
systemctl --user start living-memory.service
```

Consolidation's `similarity_search` changes behaviour at this point by design:
while the column existed it scored nodes by their single legacy vector
(bit-identical to v5, because its thresholds were calibrated against those
scores), and once the column is gone it scores them by max-pool over chunks.
Watch the first automatic consolidation pass after the drop for over-eager
merging.

## Rollback

| After step | How to roll back | Cost |
| --- | --- | --- |
| 1 (backup) | nothing to undo | — |
| 2 (backfill) | nothing to undo: the old code ignores `node_chunk_embeddings` entirely. To reclaim the space, `DELETE FROM node_chunk_embeddings` + `VACUUM`. `metadata.schema_version` is 6, which no code gates on; an old-code restart rewrites it to 5 and leaves the table alone. | seconds |
| 3 (deploy) | `git -C /home/sfx/p/lm reset --hard <previous>` | seconds |
| 4 (restart) | check out the previous revision and restart. `nodes.embedding` is still intact and current, so the old vector channel works exactly as before. **This is the last fully reversible point.** | one restart |
| 5–6 | same as 4 | one restart |
| 7 (drop) | restore the pre-drop backup: stop the server, move the live file aside, copy the backup into place, start the server. Everything written after the backup is lost. | minutes + data loss |

Because step 7 is the only irreversible one, the safe habit is to let steps 1–6
sit for a day of real traffic before running it. Nothing degrades while both
representations coexist: the extra cost is 89 MiB of disk.

## Failure modes seen or guarded against

| Symptom | Cause | What to do |
| --- | --- | --- |
| `refusing to run without a backup` | neither `--backup` nor `--existing-backup` | pass one; `--dry-run` needs neither |
| `backup … holds N nodes but the database has M` | the "backup" was a `cp` without the `-wal`, or is simply old | retake it with `--backup` |
| `the embedding model … did not load` | model not in the local cache, or `LIVING_MEMORY_EMBEDDING_BACKEND=hash` in the environment | fix the model cache; never pass `--allow-fallback-embeddings` against a real database |
| `database busy writing chunks …, retrying` | the server holds the write lock | expected under load; it retries with backoff. Persistent failure means something holds a long write transaction |
| run ends with `N nodes are still pending` | nodes written by the live server during the run | re-run (step 5) |
| `verify` reports `inconsistent_nodes > 0` | a chunk set was written by something other than `replace_node_chunks` | re-chunk those nodes: `DELETE FROM node_chunk_embeddings WHERE node_id IN (…)` then re-run the backfill |
| vector channel returns nothing after step 7 | the column was dropped before the chunk-reading code was deployed | deploy the chunk-reading vector channel; the chunks are all still there |

## Tests

`tests/test_backfill_chunk_embeddings.py` covers this against temporary fixture
databases only — never the live one. Notably: an interrupted run (a real
`SIGKILL` of a real subprocess) plus a re-run produces byte-identical chunk state
to an uninterrupted run; re-running writes nothing and rewrites no row; a
concurrent content change ends with exactly one chunk generation matching the new
content; `--dry-run` leaves the file byte-identical; the backfill leaves
`nodes.embedding` byte-identical and never drops the column; and the backup gate
rejects a missing, truncated or self-referential backup.

```sh
bash scripts/test.sh tests/test_backfill_chunk_embeddings.py
```
