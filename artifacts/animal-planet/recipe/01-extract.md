# 01 — Extract: animal-planet evidence → private staging dataset

This is the first section of the animal-planet audit/replay-packet recipe.
It documents the re-runnable extraction of all raw evidence into a private
local staging dataset. The scripts under [`extract/`](extract/) are exactly
what was run; rebuilding staging from scratch is **one command**:

```bash
bash artifacts/animal-planet/recipe/extract/run_all.sh
```

Prerequisites: ssh alias `alt` (BatchMode key auth, read-only use), local
`python3`, local Living Memory DB at `~/.local/share/living-memory/`.
No sqlite3 CLI is needed anywhere (alt has none); all SQL runs through
Python stdlib `sqlite3`.

## Destinations (never tracked, never committed)

- `AP_STAGING` = `/home/sfx/.cache/ap-audit/staging/` — the staging dataset.
  Holds raw private rows (queries, node contents, transcript-derived
  events). Lives outside every repo.
- `AP_TMP` = `/tmp/ap-audit/` — DB snapshot copies.

Both are environment-overridable (see `extract/common.sh`). Nothing under
either path may ever be committed; tracked files carry only aggregates,
identifiers, lengths and hashes.

## Sources (strictly read-only)

1. **alt LM DB** `alt:/home/user/.local/share/living-memory/global.sqlite3`
   — scp db+wal+shm to `$AP_TMP/alt/`, `PRAGMA wal_checkpoint(TRUNCATE)` on
   the COPY, `PRAGMA integrity_check`, then all queries via
   `file:...?mode=ro` URI (precedent: `artifacts/baseline.md`). SHA-256 of
   the checkpointed copy + capture timestamps recorded in
   `staging/alt-db/snapshot.json`.
2. **alt transcripts** `alt:~/.claude/projects/*animal-planet*/*.jsonl`
   — per-file inventory (path/size/mtime/sha256 via remote `sha256sum`),
   then a stream-parse executed remotely as `ssh alt python3 - <
   transcript_parser.py` (script on stdin; nothing is ever written on alt).
   Beware: the mangled project dir names start with `-`; every remote file
   command needs `--` before the path.
3. **local LM DB** `~/.local/share/living-memory/global.sqlite3` — cp trio
   to `$AP_TMP/local/`, checkpoint the copy, `mode=ro`; export holdout
   workloads for non-animal-planet scopes.

## Step map

| Step | Script | Output (under staging/) |
| --- | --- | --- |
| 10 | `10_snapshot_alt_db.sh` | `alt-db/snapshot.json` (+ checkpointed copy in `$AP_TMP/alt/`) |
| 20 | `20_export_alt_db.py` | `alt-db/recall_events.jsonl` (export window = W1.start..W2.end, 1576 events), `alt-db/nodes.jsonl` (807 nodes referenced by results + feedback_trace_id), `alt-db/connections_typed.jsonl` (all 93 supersedes + 85 contradicts), `alt-db/nodes_typed_edges.jsonl`, `alt-db/connections_among_exported.jsonl`, `alt-db/game_schema_nodes.jsonl` + `alt-db/game_schema_source_nodes.jsonl`, `alt-db/outcome_nodes.jsonl`, `alt-db/retrieval_weights.json`, `alt-db/schema.json`, `alt-db/summary.json` |
| 30 | `30_transcript_inventory.sh` | `transcripts/inventory.jsonl` (336 files, 126 dirs, ~465MB), `transcripts/inventory_summary.json` |
| 40 | `40_parse_transcripts.sh` + `transcript_parser.py` | `transcripts/lm_events.jsonl` (702 LM tool_use/tool_result pairs + per-result size records for ALL tools), `transcripts/parse_summary.json` |
| 50 | `50_snapshot_local_db.sh` | `local-db/snapshot.json` |
| 60 | `60_export_local_holdout.py` | `local-db/holdout_<scope>.recall_events.jsonl` + `.nodes.jsonl` for `project:octopus`, `project:online`, `project:x` (≤1500 most recent events each) |
| 70 | `70_build_metadata.py` | `transcripts/matched.jsonl` (deterministic transcript↔DB match), `METADATA.json` (windows, pinned predicates, per-source counts, provenance, expected-vs-measured cross-check table, discrepancies, sha256 of every staging file) |

Node `embedding` columns are excluded from every export (bulky model
vectors; BM25/vector re-execution is out of scope for the packet).

## Windows (pinned in `extract/windows.json`)

- **W1** `[2026-08-06T20:00:00Z, 2026-08-09T11:00:23Z]` — goal start to the
  original audit node `01KZK2WMP07CNTYDQXFTS23F39`. Inclusive both ends.
- **W2** `[2026-08-06T20:00:00Z, 2026-08-12T21:48:45Z]` — end = net-value
  trace `01KZVZ5ZTE1WSXFHCK6091H0RE`; the 42.4%/17.6% linkage citation
  lives here (scope `project:game` only).
- **W3** `[2026-08-09T11:34:00Z, 2026-08-12T14:37:00Z]` — payload follow-up
  trace `01KZV6VCVGXVKPF6PSFXBH00EM` (boundaries at minute precision).

## Pinned definitions (derived, recorded in METADATA.json)

- **Transcript↔DB match**: recall tool_results (is_error=false) joined to
  `recall_events` by exact query equality, nearest
  `|created_at − tool_result.timestamp| ≤ 600s`, greedy one-to-one by
  ascending delta. 366/366 transcript recalls match (W1: 202/202).
- **Serialized payload measure**: `len_json_content` =
  `len(json.dumps(tool_result block content))`; identical to
  `len(json.dumps(toolUseResult))` on this corpus. The 08-09 audit's
  "avg 12.6KB" and "18.8% of tool-result volume" reproduce on the
  `len_text` basis (sum of text-block lengths) over ALL 397 W1 LM
  tool-results: 12,642.8 avg, 0.1876 share.
- **Organic vs automatic (W2, project:game)**: automatic =
  `agent IS NULL` → 1188 events / 209 feedback (**exact on both**);
  organic = complement → 161/71.
- **Auto-OUTCOME marker** (from alt `~/p/ae/node.sh`,
  `_node_lm_record_outcome`): `content LIKE 'OUTCOME pass: %' OR
  content LIKE 'OUTCOME fail: %'` with `agent='ae'`; written via
  `lm_client.py remember` — these calls never appear in transcripts.
- **Deterministic pre-recalls (W1)**: query contains `reopen_lesson` /
  `architectural_decision` → 152 + 152 (**exact**).

## Cross-check outcome (2026-08-13 capture)

15 of 19 checks exact / exact-within-rounding — including every number the
stage was required to pin: W1 727 recall_events, 9 supersedes, W2 game
1349, automatic 1188/209, game schemas 9 (as `created_at ≤ W1.end`), W1 LM
calls 202/141/46/8, avg 12.6KB, share 18.8%, holdout totals
12127/7976/9080. Two "near": full-table `related` edges (live DB keeps
growing) and total W1 tool_results 10667 vs 10669.

Three explicit discrepancies (full detail in `METADATA.json
.discrepancies[]`; never patched per-event):

1. **Organic 161/71 vs cited 165/70.** The citation is internally
   inconsistent: 165+1188 ≠ 1349 window total and 70+209 ≠ 280 total
   feedback. No column-level predicate can produce it; the automatic side
   is exact.
2. **Auto-OUTCOME W1: 78 pinned (89 broad) vs cited 105.** The 2026-08-13
   snapshot is a lower bound: identical retry contents dedup at remember
   time and traces can decay/be forgotten between audit and snapshot.
3. **W3 payload: 164 matched vs cited 278; survivor stats median 26,792 /
   p90 30,614 / mean 26,405 vs cited 26,496/35,237/26,543.** The cited
   population decomposes exactly as 164 surviving transcript recalls + 114
   W3 DB events from `/root`-agent manual sessions (session_id set,
   starting 2026-08-11T14:43:16Z) whose transcripts sit under
   `/root/.claude/projects` (permission denied) or were removed — the
   newest surviving transcript LM event is 2026-08-11T08:20:23Z and no
   user-readable transcript on alt contains later W3 recalls. Their
   serialized sizes cannot be re-measured from surviving evidence.

## Re-run caveats

The sources are live; a rebuild re-captures them. Window-bound checks
(W1/W2/W3 are all in the past) are stable across reruns; full-table totals
and all-time counts drift upward; snapshot SHA-256s change per capture and
are re-recorded in `snapshot.json` / `METADATA.json`. The frozen packet
pins one specific capture via the manifest (see later recipe sections).
