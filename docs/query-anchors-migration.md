# Query-anchor migration runbook

Giving the memory graph an entry from query space on a live database, without an
outage: the additive v6 → v7 schema (`query_anchors`, `query_anchor_edges`), the
retro backfill that turns five months of consumed recall history into anchors,
and the restart that puts the live write and read paths in service.

The whole procedure is **additive first, destructive last** — and here there is
no destructive step at all. Nothing is dropped, nothing is rewritten: two new
tables appear and fill up, and every step before the restart is undone by
restarting the previous release.

Tool: `scripts/backfill_query_anchors.py` (`backfill`, `verify`). It never
defaults to a database — `--db` is always explicit.

## Why the order is what it is

| Step | Server can be | Because |
| --- | --- | --- |
| preflight dry run | up | opens `mode=ro` with `PRAGMA query_only`, constructs no `MemoryStore`, writes nothing |
| backup | up | the sqlite3 backup API is consistent under a concurrent writer |
| backfill | up | it only writes `query_anchors`/`query_anchor_edges`; a SQLite authorizer denies every other write, so nodes, connections and recall history cannot change |
| deploy + restart | restarting | the running process holds the old modules until it restarts |
| catch-up backfill pass | up | consumptions between the backfill and the restart were handled by code that did not write anchors |
| verify | up | read-only |

**Why the backfill comes before the deploy.** It is the same reason as the chunk
rollout: the old server ignores tables it has never heard of, so filling them
early costs nothing and means the anchor channel is useful from the first
restart instead of starting empty. `metadata.schema_version` becomes 7, which no
code gates on (every migration is self-guarding on the actual DDL), so an
old-code restart in between simply rewrites it to 6 and leaves the tables alone.

## Measured numbers

Everything below was measured on 2026-08-18 against a `sqlite3`-backup-API
snapshot of the live database (501 MB + a ~4 MB WAL, 16,7xx nodes, 143k
connections, 54,820 recall events of which **9,117** are consumed), on this CPU
box, by a full real run end to end.

| Quantity | Value |
| --- | --- |
| snapshot copy (backup API, server live) | **0.83 s**, 501,157,888 bytes |
| consumed recall events in range | 9,117 (**9,082** processed; 35 have no consuming trace left) |
| grounded consumptions | **3,682** — 40.5% of processed |
| events applied | 3,661 (21 had no live target left) |
| **anchors written** | **3,337** (324 reinforcements, of which 101 merged by cosine) |
| **anchor edges written** | **7,942** rows from 8,436 edge writes — 2.38 per anchor |
| dry run (read-only rehearsal, no model load) | **41 s** (219 events/s) |
| full run wall clock | **1 m 25 s** (106 events/s), model load included |
| second pass over the same history | **57 s**, 0 events applied, 0 rows written |
| edge-migration sweep | 2,111 distinct targets examined, 0 moved |
| `verify` | exit 0 |

The second pass is the evidence for two claims at once. It wrote nothing —
3,661 events came back as `events_already_anchored`, and anchors, edges and the
sum of `hits` were unchanged at 3,337 / 7,942 / 8,436 — so the resume predicate
really is idempotent on real history, not only on fixtures. And a SHA-256 over
every node, connection and recall event was **identical** between the
pre-backfill backup and the database after both runs (16,726 nodes), so the
authorizer's "anchor tables only" is observable from outside the process, not
just asserted inside it.

Against the feasibility estimate (~3,760 anchors, ~8,550 edges, 2.28 edges per
anchor, extrapolated from a 600-event sample): anchors came in 11% low and edges
7% low, with a slightly *higher* edge-per-anchor ratio. Both gaps are the two
rules the estimate could not model — cosine dedup merged 101 near-identical
questions into existing anchors, and 263 grounded targets plus 21 whole events
were dropped because every node they used is gone with nothing replacing it.

The **sweep moved zero edges**, and that is the expected result rather than a
disappointment: `upsert_anchor` resolves a target through `supersedes` *before*
writing the edge, so the 219 grounded targets that had been superseded since
were already written as edges to their replacements. The sweep is the repair
path for edges written before that resolution existed, and it runs at the end of
every pass because "the write path resolved it" is a property worth verifying
rather than assuming.

Disk during the migration: one backup (501 MB) plus the database's own growth
(the anchor tables are ~7 MB — 3,337 × 384 float32 vectors is 5.1 MB, the rest
is rows and two indexes). Budget **550 MB**, of which 501 MB is the backup; put
the backup on another filesystem and it is ~10 MB.

## Preconditions

* The anchor code (schema v7: `storage.py` + `query_anchors.py`, the retrieval
  entry, the live write path in `feedback.py`) is merged into the checkout that
  will be deployed.
* `python3 -c "import sentence_transformers"` works and the model is in the local
  HuggingFace cache. The backfill **refuses to start** if the model does not
  load, rather than filling `query_anchors` with hash-fallback vectors from a
  different vector space — in which no anchor would ever match anything.
  (`--allow-fallback-embeddings` exists for fixtures; never use it here.)
* ~550 MB free disk, most of it for the backup.
* You can restart the server (`systemctl --user restart living-memory.service`,
  see [deployment.md](deployment.md)).

**`alt` is not part of this procedure.** Nothing automated touches that host — no
agent connects to it, deploys to it, restarts it, or runs this backfill against
its database. When `alt` is to be updated, the operator does it by hand, and only
on an explicit instruction.

## 0. Preflight — rehearse, read-only

```sh
DB=~/.local/share/living-memory/global.sqlite3

python3 scripts/backfill_query_anchors.py verify --db "$DB"
python3 scripts/backfill_query_anchors.py backfill --db "$DB" --dry-run \
    --json /tmp/anchor-dry-run.json --progress-interval 20
```

`verify` before the migration exits **1** and reports
`anchor_tables_present: false` — that is the expected pre-migration state, not a
failure. `--dry-run` opens the database `mode=ro` with `PRAGMA query_only`, never
constructs a `MemoryStore` (whose constructor migrates), labels the whole history
for real, and reports the anchors and edges a real run would write. It needs no
backup because it writes nothing, and it loads no model.

Read `projected_anchors` and `projected_edges` against
`artifacts/anchors/estimate.json`. They are an **upper bound**: the projection
counts distinct `(scope, fingerprint)` identities, and cosine dedup can only
merge further. A projection wildly below the estimate means the grounding
threshold or the history is not what the estimate measured — stop and find out
why before writing anything.

## 1. Backup — mandatory, and `cp` is not one

```sh
python3 scripts/backfill_query_anchors.py backfill --db "$DB" \
    --backup ~/.local/share/living-memory/global.pre-anchors-$(date +%Y%m%d).sqlite3 \
    --json /tmp/anchor-backfill.json --progress-interval 30
```

The backup is taken by the same command that then does the work, through
`sqlite3.Connection.backup()` from a read-only source connection. That API copies
the committed database as one consistent snapshot **including everything sitting
in the WAL**, while the server keeps writing.

> `cp global.sqlite3 backup.sqlite3` is **not** a valid backup of this database.
> The live file carries a large WAL holding the newest transactions; a copy
> without the `-wal` is a valid SQLite file that is silently missing them. The
> script rejects an `--existing-backup` holding less than 99% of the source's
> nodes for exactly this reason (`--min-backup-fraction`).

To reuse a backup you already have, pass `--existing-backup PATH` instead; it is
validated (`PRAGMA quick_check` plus the node-count comparison) before any write.

## 2. Backfill — with the server up

The command in step 1 is the backfill. It is safe to leave running while the
server serves reads and writes:

* it only ever writes rows of `query_anchors` and `query_anchor_edges`;
  immediately after the migration a SQLite **authorizer** is installed that
  denies every write outside those two tables, so this process physically cannot
  modify a node, a connection, or a recall event. It reads all three;
* it commits **one anchor at a time**, so the write lock is held for
  milliseconds at a time instead of for the whole run, and the WAL stays
  checkpointable;
* `PRAGMA busy_timeout` is raised to 30 s and each anchor write retries with
  backoff on `SQLITE_BUSY`;
* the first thing it does is the additive v6 → v7 migration: `CREATE TABLE
  query_anchors` and `query_anchor_edges` plus their two indexes. The `nodes` and
  `connections` DDL — with the baked-in `level` and `type` CHECK constraints a
  500 MB database cannot cheaply rebuild — stays byte-identical, and nothing in
  the running old process reads the new tables.

Expect **~1.5 minutes** and a progress line every `--progress-interval` seconds,
then a summary of events, anchors, edges and elapsed time (also written to
`--json`).

**If it is interrupted** (Ctrl-C, SIGKILL, machine reboot, disk full), re-run the
same command with `--existing-backup` pointing at the backup you already took. It
resumes: progress lives in the database, not in a state file. Anchors are stamped
with their *source event's* `created_at` rather than the wall clock, and the loop
replays history oldest-first, so an anchor's `last_matched_at` is the timestamp
of the last event it absorbed. An event is skipped only when its anchor already
carries an edge to every live node it grounded on *and* is already fresh as of
that event. A resumed run therefore lands on exactly the anchors, weights and hit
counts an uninterrupted run produces, and a re-run after a complete run writes
nothing.

Two consequences of the historical stamping worth knowing before you see them:

* an anchor built from a May consumption looks like an anchor last used in May,
  because it was. That is what makes `--until` produce exactly the anchor set
  that existed at a cutoff, and it is why `ANCHOR_TTL_DAYS` (90) will eventually
  age out the oldest of them once a decay sweep runs on anchors. Anchor decay is
  soft — the row and its edges survive, and the next consumption of that question
  revives it with everything it had learned;
* where the live path has already anchored a question, the retro pass adds the
  edges history knows about **without** moving that anchor's freshness backwards.

### Building a leak-free evaluation set

`--until CUTOFF` restricts the run to events with `created_at < CUTOFF`. That is
not a convenience flag. Goldset queries are drawn from this same history, so
anchors built from the very events being scored would show a fictitious gain; a
scored run without the cutoff is not evidence. Evaluation builds run against a
**snapshot**, never the live file:

```sh
SNAPSHOT=/tmp/eval-snapshot.sqlite3
python3 - "$DB" "$SNAPSHOT" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
src.execute("PRAGMA query_only = 1")
src.backup(sqlite3.connect(sys.argv[2]))     # WAL-correct, server can stay up
PY
python3 scripts/backfill_query_anchors.py backfill --db "$SNAPSHOT" \
    --until 2026-08-01T00:00:00Z --backup /tmp/eval-snapshot-pre.sqlite3
```

## 3. Deploy the new code

```sh
git -C /home/sfx/p/lm fetch origin && git -C /home/sfx/p/lm pull --ff-only
```

Nothing is served differently yet: the running process holds the old modules
until it restarts. **Do not stop here.** The previous rollout on this system left
the server three hours on old code with a merged, ready master — the anchors were
in the database, the code was on disk, and every client kept getting the old
behaviour because nobody restarted. A deploy without a restart is not a deploy.

## 4. Restart the server, and smoke-test a WRITE

```sh
systemctl --user restart living-memory.service
systemctl --user is-active living-memory.service living-memory-tls.service
curl -fsS http://127.0.0.1:8765/health     # boot_id must differ from before
```

From here `memory_recall` matches the query against anchor vectors and injects
the matched anchor's edges into the graph channel, and every content-grounded
consumption creates or reinforces an anchor.

**The smoke test must be a write** — a `memory_remember` or a `memory_teach`, not
a recall. This is the lesson the chunk rollout paid for and it applies unchanged
here, because the cause is the same: `MemoryStore` resolves the shape of the
schema **once per process** and caches it (`_node_embedding_column_cache`,
`_chunk_table_cache`, `_anchor_tables_cache`). On that rollout a server that had
started while `nodes.embedding` still existed kept `cached=True` after the column
was dropped, and every `INSERT INTO nodes` failed — while reads kept working, so
a recall-only smoke test passed with flying colours against a server whose writes
were broken.

The anchor equivalent is quieter, which is worse: a `MemoryStore` that resolved
`_anchor_tables_cache` to `False` against a pre-v7 file answers every anchor read
with "no anchors" and raises on the write path — silently degrading recall to its
four old channels rather than failing loudly. The order in this runbook keeps
that from happening (the tables exist before the restart, and a restarting server
runs the migration itself), but the check is a write, every time:

```sh
# The real thing, through MCP — a write that also creates the first live anchor.
curl -fsS -H "Authorization: Bearer $LM_AUTH_TOKEN" -H 'Content-Type: application/json' \
     http://127.0.0.1:8765/mcp/ \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
          "params":{"name":"memory_remember","arguments":{
            "content":"query-anchor rollout smoke: writes work after the restart",
            "context":{"scope":"project:lm"}}}}'
```

A `memory_remember` that returns a node id is the proof. Then confirm the process
serves this checkout's protocol texts:

```sh
python3 /home/sfx/p/lm/scripts/check_deployed_protocol.py
```

## 5. Catch-up backfill pass — the window between steps 2 and 4

Consumptions between the backfill and the restart were handled by code that did
not write anchors, so their signal is missing. This pass picks it up:

```sh
python3 scripts/backfill_query_anchors.py backfill --db "$DB" \
    --existing-backup ~/.local/share/living-memory/global.pre-anchors-YYYYMMDD.sqlite3 \
    --json /tmp/anchor-catchup.json
```

It costs about as much as the first pass (**57 s** measured, against 1 m 25 s)
whatever the size of the window: the pending predicate is per-event, so finding
the tail means re-labeling the whole history. What it *writes* is only the tail —
everything already anchored comes back as `events_already_anchored`, and
`events_applied` is the size of the window. On the snapshot, where there was no
window at all, it applied **0** events and wrote **0** rows.

## 6. Verify

```sh
python3 scripts/backfill_query_anchors.py verify --db "$DB" --json /tmp/anchor-verify.json
```

Exit code 0 requires all of:

* `edges_to_missing_nodes: 0` and `edges_to_superseded_nodes: 0` — no anchor edge
  points at a node that is gone, or at a decayed node that has an active
  replacement reachable through `supersedes`. An edge into a decayed node with
  **nothing** replacing it is reported separately
  (`edges_to_decayed_nodes_without_replacement`) and is *not* a failure: the node
  simply aged out, the edge is inert, and the match offers one target fewer.
  Treating ordinary decay as corruption would make this check unusable;
* `anchors_outside_source_scope: 0` — every anchor's `(scope, fingerprint)` is
  the identity of a consumed recall event *in that scope*. Since the fingerprint
  is `recall_fingerprint(query, scope)`, that equality is exactly "no anchor
  crosses the scope its question was asked in";
* `anchors_without_edges: 0` and `anchors_with_unusable_embedding: 0` — every
  anchor has a vector of its recorded width that is not all zeros, and at least
  one edge. An anchor that matches queries and offers nothing is dead weight;
* `duplicate_identities: 0` — one anchor per `(scope, fingerprint)`.

Also worth reading: `edges_per_anchor` (2.38 on the snapshot run) and `scopes`,
which shows the anchor corpus is dominated by the scopes the operator actually
works in.

Then check recall itself — the anchor entry is what this changes:

```sh
curl -fsS -H "Authorization: Bearer $LM_AUTH_TOKEN" -H 'Content-Type: application/json' \
     http://127.0.0.1:8765/mcp/ \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
          "params":{"name":"memory_recall","arguments":{"query":"поревьювь мердж-реквест"}}}'
```

A jargon query of that shape is the class anchors exist for: it should now reach
nodes that its own words never matched.

## Rollback

| After step | How to roll back | Cost |
| --- | --- | --- |
| 1 (backup) | nothing to undo | — |
| 2 (backfill) | nothing to undo: the old code ignores `query_anchors`/`query_anchor_edges` entirely. To reclaim the space, `DROP TABLE query_anchor_edges; DROP TABLE query_anchors;` — no other table references them. `metadata.schema_version` is 7, which no code gates on; an old-code restart rewrites it to 6 and leaves the tables alone. | seconds |
| 3 (deploy) | `git -C /home/sfx/p/lm reset --hard <previous>` | seconds |
| 4 (restart) | check out the previous revision and restart. The anchor tables stay; the old code cannot see them. | one restart |
| 5–6 | same as 4 | one restart |

There is no irreversible step in this migration. The anchors are a derived,
rebuildable structure: with the history and this script they can be reconstructed
from scratch at any time, which is the strongest rollback story any of these
rollouts has had.

## Failure modes seen or guarded against

| Symptom | Cause | What to do |
| --- | --- | --- |
| `refusing to run without a backup` | neither `--backup` nor `--existing-backup` | pass one; `--dry-run` needs neither |
| `backup … holds N nodes but the database has M` | the "backup" was a `cp` without the `-wal`, or is simply old | retake it with `--backup` |
| `the embedding model … did not load` | model not in the local cache, or `LIVING_MEMORY_EMBEDDING_BACKEND=hash` in the environment | fix the model cache; never pass `--allow-fallback-embeddings` against a real database |
| `--until is not a timestamp I can parse` | a cutoff like `last tuesday` | ISO-8601 instant or a bare date, UTC |
| `database busy writing the anchor for event …, retrying` | the server holds the write lock | expected under load; it retries with backoff. Persistent failure means something holds a long write transaction |
| `verify` reports `anchors_outside_source_scope > 0` | an anchor was written with a scope other than its event's | do not "fix" the anchors: find the writer. The scope is what keeps one project's questions out of another's answers |
| `verify` reports `edges_to_superseded_nodes > 0` | edges written before the `supersedes` hook, or a supersede written by raw SQL | run one more `backfill` pass — its closing sweep re-points them |
| recall returns no anchor-sourced results after the restart | a process cached `_anchor_tables_cache = False` against a pre-v7 file | restart it; check `verify` reports `anchor_tables_present: true` |

## Tests

`tests/test_backfill_query_anchors.py` covers this against temporary fixture
databases only — never the live one. Notably: an interrupted run (a real
`SIGKILL` of a real subprocess) plus a re-run produces anchor and edge state
identical to an uninterrupted run, timestamps and hit counts included; re-running
after a complete run writes nothing and rewrites no row; the authorizer refuses
every write to `nodes`, `connections` and `recall_events`; `--until` excludes the
event stamped exactly at the cutoff; `--dry-run` leaves the file byte-identical
and runs no migration; the backup gate rejects a missing, truncated or
self-referential backup; and `verify` fails on each defect it claims to catch and
passes once the sweep repairs the repairable one.

```sh
bash scripts/test.sh tests/test_backfill_query_anchors.py
```
