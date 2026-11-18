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

## Refresh an existing chunk index in the serving encoder space

For a service already reading chunks, use the maintenance command while it serves
traffic. Select the actual database, serving model, tokenizer environment and
producer ID. A different encoder of the same width uses different coordinates.
For example, with the default offline MiniLM and a backup on a writable volume:

```sh
tool=scripts/backfill_chunk_embeddings.py
space=st:paraphrase-multilingual-MiniLM-L12-v2:384
python3 "$tool" verify --db "$DB" --encoder-id "$space"
python3 "$tool" backfill --db "$DB" --encoder-id "$space" --dry-run
python3 "$tool" backfill --db "$DB" --encoder-id "$space" --backup "$BACKUP" --limit 64
python3 "$tool" backfill --db "$DB" --encoder-id "$space" --existing-backup "$BACKUP"
python3 "$tool" verify --db "$DB" --encoder-id "$space"
```

Verify exits 1 while active chunkable sources are pending. A read-only command
can inspect an explicit space without loading its encoder. With no explicit ID,
it resolves the encoder locally and reports its ID. On writes the producer must
load and match the selected space. Use the same `--model` on each invocation for
a non-default model. A legacy table lacking producer IDs needs fresh encoding
of every active chunkable source. Use `--node-id ID` on verify, dry-run and backfill to inspect and repair one
source. Coverage follows the selected ID, scope and level.

The source-attempt limit bounds each pass, including lost races. A single source
may contain many windows. Budget for a validated SQLite backup, database and
WAL growth, CPU and memory for the model and for one page of windows. The
dry-run counts replacement vector bytes. Disk growth depends on page reuse,
index writes and WAL checkpointing under concurrent readers. Estimate duration
from window count divided by measured throughput on a limited pass, plus backup,
validation, model load and lock waits. Repeat verify after the last pass.

## Why the order is what it is

| Step | Server can be | Because |
| --- | --- | --- |
| backup | up | the sqlite3 backup API is consistent under a concurrent writer |
| backfill | up | after normal store migrations the authorizer permits chunk replacement and its schema-owned revision effects |
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

* Opening the writable store runs normal schema migrations. During the subsequent
  backfill a SQLite authorizer permits chunk writes, source fingerprint updates and
  schema-owned cache revisions. When a stored nonempty fingerprint differs from
  SHA-256 of the current UTF-8 body, the writer corrects it in the same transaction
  that replaces all windows.
* Each source commits in one transaction. SQLite waits up to the configured busy
  timeout and retries chunk writes with backoff on lock contention.
* The encoder runs outside that transaction. Before applying the result the writer
  checks source content, fingerprint, scope, level, active status, pending status
  and the observed chunk generation. A concurrent edit or replacement can win.
* Progress lives in the database. After interruption, resume with a validated
  `--existing-backup` and the same encoder. A ready source is skipped. A source
  that keeps changing can remain pending for the next pass.

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

A chunk's nullable `encoder_id` identifies the space that produced its
coordinates. Recall scores chunks with the prepared query's encoder identity
and width. Custom encoders can return `EncodedVector(values, encoder_id)`
for both source and query vectors. Plain vectors have unknown provenance.

Opening a database preserves historical chunk rows. Writable recall uses its
ordinary scope-aware repair budget (up to 16 sources per call) to encode missing,
stale or incompatible chunks from the source text. With a larger backlog,
same-width unknown vectors can nominate sources for repair. An FTS search for
the fallback encoder's synonyms can also nominate sources across a backend
switch. The queue keeps at least one slot for cyclic repair. Only compatible
windows scored after repair supply vector evidence. A source without a useful
discovery signal may still wait for its turn in the queue. Other retrieval channels
remain available while repair is pending. Read-only recall skips the repair.

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

Exit code 0 requires a chunk table and a complete, consistent set for every
selected active chunkable source. Each set must have dense ordinals, the current
source fingerprint, the target producer ID, and correctly sized float32 BLOBs.
The report also lists table-wide sizes and widths, including inactive sources.

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

**Restart the server again immediately after the drop.** `MemoryStore` caches
whether the legacy column exists (`_node_embedding_column_present`,
`storage.py`) once per process; a server that started while the column was
present keeps `cached=True` and every `INSERT INTO nodes` now fails with
`table nodes has no column named embedding` — reads keep working, but
`memory_remember`/`memory_teach` are broken until the restart. Observed live
on the 2026-08-18 rollout. Then smoke-test a **write** (e.g. a `memory_teach`
you needed anyway), not only a recall:

```sh
systemctl --user restart living-memory.service
```

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
| `verify` reports `inconsistent_nodes > 0` | a chunk set was written by something other than `replace_node_chunks` | run the backfill for the selected space to replace inconsistent sets |
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
