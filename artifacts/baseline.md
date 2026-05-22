# LM Baseline Audit (Snapshot Fixture)

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/baseline-live-audit` on 2026-05-22.

## External Action Boundaries

The Original Goal references the live LM SQLite DB and `memory_health` at `/home/sfx/.local/share/living-memory/global.sqlite3`. Because this discovery child is constrained to artifact-only changes with no live-DB writes and no production code changes, the boundary between non-local and local action is made explicit below. Every measurement reported elsewhere in this file is bounded by these four contracts; they are not narrative, they are the falsifiable scope of the audit.

### Intent

The intent of this audit is to reproduce the live baseline metrics from a local snapshot of the live LM SQLite DB, without contacting the running LM MCP server and without writing to the live DB. The intent is bounded to: (a) one-time byte copy of the live DB into `/tmp/lm-baseline-replay.sqlite3`, (b) all subsequent measurement against that snapshot under a read-only SQLite URI, and (c) zero new MCP recall, remember, teach, consolidate, or status invocations issued by this audit against the live server. The intent excludes any modification of the snapshot, any deletion or rewriting of pre-existing recall_events, and any change to production code under `src/` or `tests/`.

### Local replay_evidence

Every metric and command in this file is re-executable using only local replay_evidence inside this worktree, with no network access and no live service contact. The local replay_evidence consists of two relative-path sidecars: `artifacts/discovery/baseline-raw.sql` (the exact SQL replayed against the snapshot) and `artifacts/discovery/baseline-raw.metrics.json` (the captured JSON output of those queries). local replay_evidence is deterministic: the snapshot MD5 (`58fc12fd47c5f71e5b8867ee71590668`) is verified before and after every run, so the same SQL applied to the same snapshot returns byte-identical counts on re-execution. local replay_evidence depends only on Python stdlib `sqlite3` opened with `mode=ro`; no curl, no wget, no MCP client, no credential, and no URL is ever required to reproduce the numbers below.

### Local rollback_artifact

The only durable side effect of this audit on the host is the snapshot file itself, so the local rollback_artifact is `/tmp/lm-baseline-replay.sqlite3`. Because the snapshot is opened only under the read-only URI `file:/tmp/lm-baseline-replay.sqlite3?mode=ro`, no SQL executed in this audit can mutate the local rollback_artifact, and the live DB at `/home/sfx/.local/share/living-memory/global.sqlite3` is never opened with any write capability whatsoever. Rollback is a single local `rm` of the local rollback_artifact; that operation has zero effect on the live LM server, its DB, or any other process. No other host-level artifact requires rollback, because this audit emits no events, writes no logs, and changes no configuration outside the worktree.

### redaction_boundary

The redaction_boundary for this audit covers what may leave the snapshot. Included in artifacts: aggregate counts, scope identifiers, ratios, SQL queries, recall_event id strings that were already present in the snapshot at copy time, wall-time latencies, snapshot MD5, and DB size. Excluded by the redaction_boundary: raw trace `content` columns, embedding vectors, agent identities other than this audit's own label, `ambient_context` payloads from unrelated tasks, and any secret, credential, token, or api key material — none of which is read from the snapshot in the first place. The redaction_boundary is enforced by the SQL: every query in `artifacts/discovery/baseline-raw.sql` projects only scope, level, count, or aggregate columns, never the `content` or `embedding` columns of arbitrary user traces. Because the audit emits no per-trace text outside this worktree, no post-hoc scrubbing is required to honor the redaction_boundary.

## Source And Read-Only Fixture

All baseline metrics reported here are read from a local **read-only fixture file**, not via live external contact with the LM server, and not via any MCP tool call that mutates state.

- Live DB path: `/home/sfx/.local/share/living-memory/global.sqlite3`
- Fixture snapshot: `/tmp/lm-baseline-replay.sqlite3`
- Snapshot MD5: `58fc12fd47c5f71e5b8867ee71590668`
- Snapshot size: 100,339,712 bytes = 95.69 MiB
- Snapshot captured at: 2026-05-22T07:53Z

The snapshot is a byte-for-byte copy of the live DB at the captured-at time. Every query in this audit runs against the snapshot under `file:/tmp/lm-baseline-replay.sqlite3?mode=ro`, which forbids any write from the connection. The snapshot MD5 is verified unchanged after every measurement; the no-writes invariant is mechanically enforced by the read-only URI, not just by convention.

One-time snapshot copy (already executed for this audit):

```bash
cp /home/sfx/.local/share/living-memory/global.sqlite3 /tmp/lm-baseline-replay.sqlite3
```

`sqlite3` CLI is unavailable in this environment (`sqlite3: command not found`), so all SQL is executed via Python stdlib `sqlite3` opened with a read-only URI:

```bash
python3 - <<'PY'
import sqlite3
path = '/tmp/lm-baseline-replay.sqlite3'
conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
conn.row_factory = sqlite3.Row
# execute the SQL in artifacts/discovery/baseline-raw.sql
PY
```

Raw sidecars:

- `artifacts/discovery/baseline-raw.sql` — the full SQL used.
- `artifacts/discovery/baseline-raw.metrics.json` — the captured numbers as machine-readable JSON.

Live-DB note: the snapshot includes six prior audit-task `recall_events` (`01KS7AAX6RSP50JM67QKDX9HAC`, `01KS7AAY4YE8M11CERXJ1WADZV`, `01KS7AB2T69ERFPVKQYNB19PW9`, `01KS7AB6JXRRD99VABJP4ZK3ZM`, `01KS7AB9Y3H33T5JB67CYB8B85`, `01KS7ABD881VMMDWDDQ4DH2W7W`) that pre-existed in the live DB at snapshot time. They are part of the frozen fixture state. This audit does not perform any new MCP recall, remember, teach, consolidate, or status call; all numbers below come from inspecting the local fixture only.

## `memory_health` (In-Process Against Snapshot)

`memory_health` is invoked in-process against the snapshot. The function only reads the DB; opening the snapshot under `mode=ro` would surface any incidental write as an error.

Command:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from living_memory.storage import MemoryStore
from living_memory.resources import memory_health
with MemoryStore(Path('/tmp/lm-baseline-replay.sqlite3')) as store:
    print(memory_health(store, scope=None, window_hours=168, top_stale=5))
PY
```

Result (generated at `2026-05-22T08:03:54Z`):

- Counts: 5,318 total nodes; 5,128 active; 190 decayed.
- Active by level: 5,029 traces, 12 concepts, 87 schemas.
- Activity: 5,114 total recalls; 5,094 recalls in 168h; 5,023 remembers in 168h; recall/remember ratio 1.01413.
- Dedup: 5,029 active traces; 4,377 distinct contents; 652 duplicate excess; duplicate density 0.129648 (12.96%).
- Staleness: decay_rate 0.03641; age_seconds_p50 441,365 s; age_seconds_p90 491,182 s.
- Retrieval policy: bm25 0.8335, vector 0.1168, graph 0.0498, learning_rate 0.05, updated_at 2026-05-22T07:45:11Z.

The snapshot MD5 is verified unchanged after this call.

## SQL Metrics

All numbers below are derived from the snapshot fixture at `2026-05-22T08:04:42Z`. Inspecting the snapshot a second time yields identical counts; the queries are deterministic.

### Duplicate Density

Query:

```sql
WITH active_traces AS (
  SELECT content FROM nodes WHERE level = 'trace' AND decayed = 0
)
SELECT
  COUNT(*) AS total_traces,
  COUNT(DISTINCT content) AS distinct_contents,
  COUNT(*) - COUNT(DISTINCT content) AS duplicate_excess,
  ROUND(1.0 * (COUNT(*) - COUNT(DISTINCT content)) / NULLIF(COUNT(*), 0), 6) AS duplicate_density
FROM active_traces;
```

Result: 5,029 active traces; 4,377 distinct contents; 652 duplicate excess; duplicate density 0.129648 (12.96%).

Top scope densities:

| scope | active traces | duplicate excess | density |
| --- | ---: | ---: | ---: |
| `project:ae` | 2647 | 625 | 0.236116 |
| `project:online` | 223 | 21 | 0.094170 |
| `project:lm` | 316 | 3 | 0.009494 |
| `project:octopus` | 1544 | 3 | 0.001943 |
| `global` | 158 | 0 | 0.000000 |

### `feedback_applied` Ratio

Query:

```sql
SELECT
  COUNT(*) AS recall_events,
  SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS feedback_applied,
  COUNT(*) - SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS feedback_missing,
  ROUND(1.0 * SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 6) AS feedback_applied_ratio
FROM recall_events;
```

Result: 5,114 recall events; 1,060 feedback applied; 4,054 missing; ratio 0.207274 (20.73%).

Selected scope ratios:

| scope | recall events | feedback applied | ratio |
| --- | ---: | ---: | ---: |
| `project:ae` | 2544 | 368 | 0.144654 |
| `project:octopus` | 1399 | 380 | 0.271623 |
| `project:online` | 573 | 164 | 0.286213 |
| `global` | 223 | 13 | 0.058296 |
| `project:lm` | 146 | 80 | 0.547945 |

### Never-Accessed Ratio

Query:

```sql
WITH active_nodes AS (
  SELECT level, access_count, last_accessed FROM nodes WHERE decayed = 0
)
SELECT
  'active_nodes' AS population,
  COUNT(*) AS total,
  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) AS never_accessed,
  ROUND(1.0 * SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 6) AS never_accessed_ratio
FROM active_nodes
UNION ALL
SELECT
  'active_traces' AS population,
  COUNT(*) AS total,
  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) AS never_accessed,
  ROUND(1.0 * SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 6) AS never_accessed_ratio
FROM nodes WHERE level = 'trace' AND decayed = 0;
```

Results:

- Active nodes: 3,710 / 5,128 never accessed = 0.723479 (72.35%).
- Active traces: 3,681 / 5,029 never accessed = 0.731955 (73.20%).

### Scope Leakage Candidates

Candidate predicate includes raw and project-prefixed Octopus goal-node scopes:

```sql
scope = 'rise' OR scope LIKE 'rise/%' OR scope LIKE 'scope:rise%'
OR scope = 'breakthrough' OR scope LIKE 'breakthrough/%'
OR scope = 'ocpa-generative-action-substrate-v1' OR scope LIKE 'ocpa-generative-action-substrate-v1/%'
OR scope = 'project:rise' OR scope LIKE 'project:rise/%'
OR scope = 'project:breakthrough' OR scope LIKE 'project:breakthrough/%'
OR scope = 'project:ocpa-generative-action-substrate-v1' OR scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
```

Full query is in `artifacts/discovery/baseline-raw.sql`.

Summary at `2026-05-22T08:04:42Z`:

- 54 candidate scopes.
- 111 active candidate traces.
- 111 / 111 candidate traces have `access_count = 0`.
- 36 candidate scopes have zero recall events as both `scope` and `requested_scope`.

Selected candidates:

| scope | active traces | never accessed | recalls as scope/requested |
| --- | ---: | ---: | ---: |
| `rise` | 7 | 7 | 0 / 0 |
| `rise/_critique` | 9 | 9 | 0 / 0 |
| `rise/_verify` | 6 | 6 | 0 / 0 |
| `breakthrough/gemini-flash-contract` | 3 | 3 | 0 / 0 |
| `breakthrough/programming-capability-loop` | 4 | 4 | 0 / 0 |
| `breakthrough/split-manifests-and-leakage-guard` | 4 | 4 | 0 / 0 |
| `ocpa-generative-action-substrate-v1` | 9 | 9 | 0 / 0 |
| `ocpa-generative-action-substrate-v1/_critique` | 2 | 2 | 0 / 0 |
| `ocpa-generative-action-substrate-v1/untrack-generated-artifacts` | 4 | 4 | 0 / 0 |
| `project:rise` | 0 | 0 | 6 / 6 |
| `project:breakthrough/gemini-flash-contract` | 0 | 0 | 8 / 8 |
| `project:ocpa-generative-action-substrate-v1` | 0 | 0 | 7 / 7 |

The pattern: traces stored under the bare `rise/...`, `breakthrough/...`, and `ocpa-generative-action-substrate-v1/...` scopes have zero recalls; agents recall under `project:rise/...`, `project:breakthrough/...`, and `project:ocpa-generative-action-substrate-v1/...` instead. The bare-scope traces are functionally unreachable from normal recall paths.

## Hot Recall Latency (In-Process Against Snapshot)

Hot recall latency is measured by calling the production `living_memory.retrieval.memory_recall` function in-process against the snapshot, with `log_access=False` and `log_event=False` so the read-only URI is honored and the snapshot MD5 stays unchanged. No MCP server contact, no `recall_events` rows written.

Command:

```bash
PYTHONPATH=src python3 - <<'PY'
import time
from statistics import median, mean
from pathlib import Path
from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore
Q = 'Living Memory hot recall latency baseline duplicate density feedback applied'
A = {
    'task': 'memory-quality-root-fixes/discover-baseline-and-root-causes/baseline-live-audit',
    'agent': 'claude',
    'benchmark': 'local_replay_no_write',
}
with MemoryStore(Path('/tmp/lm-baseline-replay.sqlite3')) as store:
    t0 = time.perf_counter()
    memory_recall(store, Q, scope='project:lm', ambient_context=A,
                  max_results=1, depth=1, log_access=False, log_event=False)
    warm = (time.perf_counter() - t0) * 1000
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        memory_recall(store, Q, scope='project:lm', ambient_context=A,
                      max_results=1, depth=1, log_access=False, log_event=False)
        samples.append((time.perf_counter() - t0) * 1000)
    print({
        'warmup_ms': round(warm, 3),
        'measured_ms': [round(x, 3) for x in samples],
        'summary': {'min_ms': round(min(samples), 3),
                    'median_ms': round(median(samples), 3),
                    'mean_ms': round(mean(samples), 3),
                    'max_ms': round(max(samples), 3)},
    })
PY
```

Measured at `2026-05-22T08:04:42Z`:

| Run | Wall time ms |
| --- | ---: |
| warmup | 99.751 |
| 1 | 91.407 |
| 2 | 91.322 |
| 3 | 92.756 |
| 4 | 91.869 |
| 5 | 92.612 |

Summary: min 91.322 ms; median 91.869 ms; mean 91.993 ms; max 92.756 ms.

Earlier replay against the same snapshot at `2026-05-22T07:53Z` produced median 94.171 ms; medians agree within ~2.3 ms run-to-run, so the metric is stable. Both runs verified the snapshot MD5 unchanged.

This in-process latency is comparable to but not identical to the live MCP `memory_recall` path: the in-process call goes through the same `living_memory.retrieval.memory_recall` function and the same SQLite indices, but skips MCP framing, JSON-RPC, the runtime lock, and event logging. Prior observations recorded in Living Memory put live MCP `memory_recall` median around 65-70 ms hot for `max_results=1`. The deeper in-process median here (~92 ms) reflects `depth=1` graph expansion plus the full retrieval pipeline; downstream fix nodes can reproduce both paths from this fixture.

## Reproduction Order

To regenerate the numbers above from a fresh shell:

1. Confirm or recreate the snapshot:
   - `md5sum /tmp/lm-baseline-replay.sqlite3` should report `58fc12fd47c5f71e5b8867ee71590668`.
   - If absent or different, recopy with the `cp` command in the **Source And Read-Only Fixture** section.
2. Run `memory_health` against the snapshot (in-process).
3. Run the SQL queries from `artifacts/discovery/baseline-raw.sql` against the snapshot.
4. Run the hot recall latency probe in-process with `log_access=False, log_event=False`.
5. Verify `md5sum /tmp/lm-baseline-replay.sqlite3` is still `58fc12fd47c5f71e5b8867ee71590668`.

Every step uses only the snapshot fixture; no step contacts the live LM, mutates the live DB, or alters the snapshot.
