# Live query-anchor backfill — deploy log

**2026-08-18**, against `~/.local/share/living-memory/global.sqlite3`, following
[docs/query-anchors-migration.md](../../docs/query-anchors-migration.md) **steps 0–2 and 6**.
The server stayed **up** for all of it (`boot_id d291137fe5c040eb95f256c59017f39a`);
nothing was restarted and no code was deployed — see [Pending](#pending-after-the-merge).

## What was run

| Step | Command | Result |
| --- | --- | --- |
| 0 | `verify --db $DB` | **exit 1**, `anchor_tables_present: false` — the documented pre-migration state |
| 0 | `backfill --db $DB --dry-run` | **exit 0**, 29.8 s (305 events/s), projects 3,430 anchors / 8,072 edges |
| 1+2 | `backfill --db $DB --backup <path below>` | **exit 0**, backup 1.7 s + backfill 59.3 s, **65.9 s wall** |
| 6 | `verify --db $DB` | **exit 0**, every gate zero |

**Backup:** `~/.local/share/living-memory/global.pre-anchors-20260818.sqlite3`
— 501,354,496 bytes, 16,734 nodes on both sides, `PRAGMA quick_check: ok`, taken in
1.7 s through `sqlite3.Connection.backup()` from a `mode=ro` source connection.
A `cp` would have been invalid: the live file carried a **503 MB WAL** at the time.

## Counts

| Quantity | Value |
| --- | --- |
| consumed recall events in range | 9,122 (9,087 processed; 35 have no consuming trace left) |
| grounded consumptions | 3,682 — 40.5% |
| events applied | 3,661 (21 had no live target left) |
| **anchors** | **3,337** (324 reinforcements, 101 merged by cosine) |
| **anchor edges** | **7,942** rows from 8,436 writes — **2.38** per anchor |
| edge-migration sweep | 2,111 targets examined, 0 moved |
| `SQLITE_BUSY` retries under the live server | 0 |
| database growth | 501,264,384 → 510,300,160 bytes (+8.6 MB; anchors 6.54 MB, edges 0.87 MB) |

### Against the estimates

| Source | Anchors | Edges | vs. actual |
| --- | --- | --- | --- |
| root goal (600-event extrapolation) | ~3,700 | ~8,500 | −9.8% / −6.6% |
| `artifacts/anchors/estimate.json` | 3,450 | 8,335 | −3.3% / −4.7% |
| this run's dry run (upper bound) | 3,430 | 8,072 | −2.7% / −1.6% |
| **written** | **3,337** | **7,942** | — |

**Anchors.** The 93 between the dry-run bound and reality is **cosine dedup**: 101
questions with distinct `(scope, fingerprint)` identities but near-identical vectors
merged into an existing anchor. The projection counts identities and cannot model
that, which is why it is documented as an upper bound. Against the root goal's
~3,700 the shortfall is the same two rules a 600-event sample could not see — that
dedup, plus 21 grounded events whose every target node is gone with no replacement.

**Edges.** Exact arithmetic, no residual: 8,699 grounded targets − **263** decayed
with no heir (skipped rather than written as dead edges) = 8,436 writes; 8,436 −
**494** that landed on an `(anchor, target)` pair that already existed and
accumulated weight instead of adding a row = **7,942**. `sum(hits)` over
`query_anchor_edges` is 8,436, which confirms the collapse rather than assuming it.
`estimate.json` counted the 5.7% decayed-target share as *present* rather than
skipped and could not model the extra collapse the 101 merges caused. The ratio
moved the other way (2.38 vs 2.28) because merging concentrates edges on fewer
anchors.

**These are the snapshot rehearsal's numbers exactly** (3,337 / 7,942 / 8,436 — the
runbook's "Measured numbers"). Expected, not a coincidence: the live file has grown
from 9,117 to 9,122 consumed events since the rehearsal and all 5 new ones labeled
ungrounded — `events_grounded` is 3,682 in both runs while `events_processed` went
9,082 → 9,087.

## `verify` — exit 0

`edges_to_missing_nodes: 0`, `edges_to_superseded_nodes: 0`,
`anchors_outside_source_scope: 0`, `anchors_without_edges: 0`,
`anchors_with_unusable_embedding: 0` (all 384-dim), `duplicate_identities: 0`.
`edges_to_decayed_nodes_without_replacement` is also 0 — it is reported separately
and would not have been a failure, but the backfill skipped those 263 targets
rather than writing inert edges.

The corpus sits in the scopes the operator actually works in: `project:x` 1,327,
`project:octopus` 657, `project:online` 652, `project:ae` 392, `project:lm` 154,
`global` 81.

## The node graph was not touched — checked from outside the process

SHA-256 over ordered projections of `nodes(id, content, level, scope, created_at)`,
`connections(source_id, target_id, type)` and `recall_events(id, query, scope,
created_at)`, compared between the pre-backfill backup and the live database
afterwards: **identical on all three** (16,734 / 143,340 / 54,824 rows, delta 0),
`nodes.updated_at` moved on 0 rows. The server received no write during the 70 s
window, so the comparison is unconfounded. The authorizer's "anchor tables only" is
therefore observable from outside, not merely asserted inside.

Anchors carry their **source event's** timestamp, not the wall clock — freshness
spans `2026-05-15T07:25:32Z … 2026-08-17T18:49:28Z`, 0 decayed.

## Pending after the merge

**Not performed here, by contract:** the shared checkout `/home/sfx/p/lm` is
read-only for this node, so the code deploy, the restart and the post-restart WRITE
smoke test are left ready to run **from merged master**. Until step 4 runs, the
anchors sit in the database unread — the running process holds the old modules.

### The restart is mandatory, not advisable — two independent reasons

1. **`MemoryStore` caches the schema shape once per process.**
   `storage.py:478-487` resolves `_anchor_tables_cache` on first use and returns
   the cached answer forever after (`_invalidate_schema_shape_cache` only runs on
   the store's own migrations). The server process booted **before** the anchor
   tables existed, so it has cached `False`: every anchor read degrades to "no
   anchors" and every anchor write raises. No amount of new data changes that —
   only a new process does.
2. **This deploy also changes retrieval behaviour, not just table presence.**
   Since the backfill was run, two constants and one code path landed:
   `ANCHOR_MATCH_COSINE_THRESHOLD` **0.80 → 0.60** (`query_anchors.py`), and the
   graph-floor fix in `retrieval.py` that keeps an anchor activation from lowering
   a candidate's score. Both are module-level and process-resident. Measured
   effect of the threshold alone: live match rate **0.52% → 37.8%**
   (`artifacts/anchors/latency.json`), i.e. the running process and the merged
   code do not behave the same on roughly a third of all recalls.

Expected cost after the restart: paired `memory_recall` p50 **+3.777 ms**
against a +5 ms budget, anchor scan 0.560 ms p50
(`artifacts/anchors/latency.json`).

```sh
DB=~/.local/share/living-memory/global.sqlite3
BK=~/.local/share/living-memory/global.pre-anchors-20260818.sqlite3

# 3 — deploy the merged code
git -C /home/sfx/p/lm fetch origin && git -C /home/sfx/p/lm pull --ff-only

# 4 — restart, then confirm a NEW boot_id (must differ from d291137fe5c040eb95f256c59017f39a)
systemctl --user restart living-memory.service
systemctl --user is-active living-memory.service living-memory-tls.service
curl -fsS http://127.0.0.1:8765/health

# 4 — the smoke test MUST be a WRITE, never a recall: a MemoryStore that resolved
#     _anchor_tables_cache against a pre-v7 file answers reads with "no anchors"
#     and raises on writes, degrading recall silently. A returned node id is the proof.
curl -fsS -H "Authorization: Bearer $LM_AUTH_TOKEN" -H 'Content-Type: application/json' \
     http://127.0.0.1:8765/mcp/ \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
          "params":{"name":"memory_remember","arguments":{
            "content":"query-anchor rollout smoke: writes work after the restart",
            "context":{"scope":"project:lm"}}}}'
python3 /home/sfx/p/lm/scripts/check_deployed_protocol.py

# 5 — catch-up pass: consumptions between the backfill and the restart were handled
#     by code that wrote no anchors. Reuses the backup already taken; ~1 min.
python3 scripts/backfill_query_anchors.py backfill --db "$DB" \
    --existing-backup "$BK" --json /tmp/anchor-catchup.json

# 6 — re-verify; must exit 0 again
python3 scripts/backfill_query_anchors.py verify --db "$DB" --json /tmp/anchor-verify-post.json
```

**`alt` was not touched and is not part of this procedure.** Nothing automated
connects to, deploys to, restarts, or backfills that host; it is updated by the
operator by hand, only on an explicit instruction.

## Rollback

Nothing here is irreversible. The old code ignores `query_anchors` /
`query_anchor_edges` entirely, so there is nothing to undo; to reclaim the ~8.6 MB,
`DROP TABLE query_anchor_edges; DROP TABLE query_anchors;` — no other table
references them. `metadata.schema_version` is now 7, which no code gates on (every
migration is self-guarding on the actual DDL); an old-code restart rewrites it to 6
and leaves the tables alone. Full state before the run is in the backup above.

Rolling back the *code* is a separate and equally simple action: the anchor
entry is off when `MemoryRecallService(anchor_seeding=False)`, and reverting to
a pre-anchor build leaves the tables inert. What the merged code buys, measured:
`0.5812 / 0.3596` hit@5 / MRR on the frozen goldset against `0.5769 / 0.3540`
without anchors — and **no measured gain on queries that are not repeats**
(`result.md` §4). Roll back on cost, not on regression: there is none.

Machine-readable records: [live-backfill.json](live-backfill.json),
[live-verify.json](live-verify.json). Evaluation of the code being deployed:
[eval.md](eval.md), [latency.json](latency.json), and `result.md`.
