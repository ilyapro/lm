# Query-anchor backfill runbook

`scripts/backfill_query_anchors.py` reconstructs grounded query evidence from
consumed recall history. It uses the [current anchor identity and replacement
contract](query-anchors.md). Every command requires an explicit `--db` path.

Backfill is optional maintenance for an authorized database. It is not a
prerequisite for loading the anchor implementation and does not deploy code or
restart a service. Deployment and process uptake follow the applicable
[deployment procedure](deployment.md) and current host permissions.

## Preconditions and capacity

Use a checkout containing the current `storage.py`, `query_anchors.py` and
backfill script. The configured embedding model must be available locally and
match the vector space used by retrieval. The script refuses an unavailable
model; `--allow-fallback-embeddings` is for deterministic fixtures.

Reserve space for a complete SQLite backup plus the anchor tables. Vector
storage is approximately `anchors × dimensions × 4` bytes, with additional
space for query text, weighted edges and indexes. Runtime depends on consumed
events, grounded queries and the encoder; the command reports its rate and
elapsed time.

## Read-only rehearsal

```sh
DB=/path/to/memory.sqlite3

python3 scripts/backfill_query_anchors.py verify --db "$DB"
python3 scripts/backfill_query_anchors.py backfill --db "$DB" --dry-run \
    --json /tmp/anchor-dry-run.json --progress-interval 20
```

These commands open `mode=ro` with `PRAGMA query_only`; they do not construct a
`MemoryStore` or run migrations. A database without anchor tables reports
`anchor_tables_present: false`, and `verify` exits 1.

The dry run grounds the selected history without loading an embedding model.
`projected_anchors` counts distinct `(scope, fingerprint)` identities from
grounded events. The real run uses that identity but can skip queries with
unusable vectors or no live grounded target. `projected_edges` and event counts
describe the selected history and cutoff.

## Backup and write

```sh
BACKUP=/path/to/memory-pre-anchors.sqlite3

python3 scripts/backfill_query_anchors.py backfill --db "$DB" \
    --backup "$BACKUP" --json /tmp/anchor-backfill.json --progress-interval 30
```

The command takes a consistent backup with `sqlite3.Connection.backup()` before
writing. This includes committed WAL contents while a server continues to use
the database. Copying only the SQLite main file can omit WAL transactions.

To reuse a backup, pass `--existing-backup "$BACKUP"`; the script checks its
SQLite integrity and node-count coverage before writing. Neither write mode
runs without a backup.

Opening the store ensures its schema is present. The backfill then installs a
SQLite authorizer that permits writes only to `query_anchors`,
`query_anchor_edges` and their trigger-maintained cache revision. It reads
nodes, connections and recall history without modifying them. Anchor and edge
writes commit incrementally; busy writes retry with bounded backoff and a
30-second SQLite busy timeout.

The pass ends with an anchor-edge migration sweep. Genuine corrections and
compatible exact-copy bookkeeping can supply replacements; heuristic
similarity cannot. The sweep does not rewrite knowledge relations or repair
historical side effects elsewhere in the corpus.

## Resume or catch up

```sh
python3 scripts/backfill_query_anchors.py backfill --db "$DB" \
    --existing-backup "$BACKUP" --json /tmp/anchor-catchup.json
```

Rerun the same selection after an interruption or to process additional history.
Progress is recorded by each exact anchor's clock and target coverage. Events
are replayed oldest first using their own timestamps. An event is skipped when
its anchor already has every live grounded target and is fresh as of that event.
An anchor written by the live path keeps its newer freshness while missing
historical edges are added.

The same completed history is idempotent: it does not reinforce its anchors
again. A resumed pass still examines the selected events, so elapsed time is
not proportional only to the amount of new history. Review `events_applied`,
`events_already_anchored`, skipped-target counts and the final verification.

## Evaluation snapshots

Use a SQLite backup snapshot for evaluation. Pass `--until CUTOFF` to include
only events with `created_at < CUTOFF`; events exactly at the cutoff are excluded.
Use an ISO-8601 timestamp or date in UTC.

```sh
python3 scripts/backfill_query_anchors.py backfill --db "$SNAPSHOT" \
    --until "$CUTOFF" --backup "$SNAPSHOT_BACKUP" --json /tmp/anchor-eval-build.json
```

Scored queries must be outside the building window. Freeze the snapshot and
relevance labels for both evaluation arms, and report exact query repeats
separately from unseen query identities. Backfill can reconstruct query evidence;
it does not supply relevance ground truth.

## Verification and limits

```sh
python3 scripts/backfill_query_anchors.py verify --db "$DB" \
    --json /tmp/anchor-verify.json
```

`verify` is a backfill diagnostic. Exit 0 requires zero duplicate identities,
unusable vectors, edgeless anchors, missing targets, decayed targets with a live
replacement, and anchors whose identity is absent from consumed history in the
same scope. `edges_to_superseded_nodes` is the report's field for decayed targets
with a known replacement, including exact-copy replacements.

Decayed targets without a replacement are reported separately and do not fail
the check: those edges are inert at match time. A live corpus can also contain
valid edgeless anchors created by negative feedback, or anchors from events that
are not in the consumed-history selection. Consequently this diagnostic is not
a general proof of corpus validity or a license to remove every reported row.

Investigate a reported scope mismatch at its writer. A further backfill pass can
repair anchor edges with known replacements through its closing sweep. Do not
infer a replacement from similar text or an embedding score.

Rollback of deployed code follows the deployment procedure. Keep the backup;
do not drop anchor tables as a generic rollback step, because query feedback and
other consumers use their identities.

## Fixture checks

`tests/test_backfill_query_anchors.py` uses temporary synthetic databases. It
covers exact-identity replay, identical-vector distinct queries, scope isolation,
actual subprocess interruption and resume, backup requirements, restricted
writes, cutoff boundaries, read-only behavior and historical relation handling.

```sh
python3 -m pytest tests/test_backfill_query_anchors.py
```
