# Recall Speed / Usefulness Root Cause

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/recall-speed-usefulness-root-cause` on 2026-05-22.

This discovery child is read-only except for this artifact. It uses the prerequisite recall context in `artifacts/discovery/lm-recall-context.md`, the frozen baseline in `artifacts/baseline.md`, and source citations from this worktree.

## Scope contract

- Intent: this artifact documents local latency and usefulness risks plus local-only benchmark guidance for downstream LM fix nodes. It does not request a remote service call, browser action, MCP call, or cross-repo write.
- Boundary tokens: local replay_evidence is the frozen SQLite snapshot and temp-copy commands below; local rollback_artifact is this artifact file and any temp DB copied by those commands; redaction_boundary excludes secrets, credentials, tokens, API keys, and raw trace bodies.
- The only permitted replay path for this discovery node is shell-only execution against `/tmp/lm-baseline-replay.sqlite3` or a temporary copy derived from it. Live MCP measurements mentioned below are historical baseline context from `artifacts/baseline.md`, not a follow-up action plan for this node.
- If an integration node independently decides to run live latency probes, that node must state its own intent, local replay_evidence, local rollback_artifact, and redaction_boundary in its owned artifact before doing so.

## Baseline facts

- Frozen DB snapshot: `/tmp/lm-baseline-replay.sqlite3`, copied from `/home/sfx/.local/share/living-memory/global.sqlite3`, 100,339,712 bytes / 95.69 MiB, MD5 `58fc12fd47c5f71e5b8867ee71590668`.
- Active data at snapshot: 5,128 active nodes, 5,029 active traces, 12 concepts, 87 schemas; 190 decayed traces.
- Duplicate density: 652 duplicate-excess active traces / 5,029 = 12.96%; `project:ae` is the dominant scope at 625 duplicate excess / 2,647 active traces = 23.61%.
- Feedback-applied ratio: 1,060 / 5,114 recall events = 20.73%.
- Never-accessed ratio: 3,710 / 5,128 active nodes = 72.35%; active traces 3,681 / 5,029 = 73.20%.
- Scope leakage candidates: 54 candidate scopes, 111 active candidate traces, 111 / 111 never accessed; bare `rise/*`, `breakthrough/*`, and `ocpa-generative-action-substrate-v1/*` traces have zero recalls.
- Canonical latency baseline from `artifacts/baseline.md`: in-process module-level recall, `scope='project:lm'`, `max_results=1`, `depth=1`, `log_access=False`, `log_event=False`, median 91.869 ms over the frozen snapshot.
- Prior live hot baseline from Living Memory recall context: MCP `memory_recall` median about 65-70 ms; local service-path hot recall about 44-47 ms; causal about 14 ms.

Important benchmark nuance: `src/living_memory/retrieval.py:485-504` constructs a fresh `MemoryRecallService` for every module-level `memory_recall` call, while the MCP server constructs one long-lived service at `src/living_memory/server.py:471-473`. The long-lived service reuses `_embedding_cache` (`src/living_memory/retrieval.py:100-106`, `src/living_memory/retrieval.py:350-368`), so downstream fixes must report both "cold helper" and "hot service" numbers rather than comparing one mode against the other.

Additional read-only profiling in this node against the same snapshot, reusing one `MemoryRecallService`, with one warmup and no access/event writes:

| Scenario | Median | Notes |
| --- | ---: | --- |
| `project:lm`, `max_results=1`, `depth=1` | 69.830 ms | stage median: BM25 11.890, vector 7.267, schema 0.655, graph 45.866, rank 1.845 ms |
| `project:ae`, `max_results=5`, `depth=1` | 165.326 ms | stage median: BM25 31.927, vector 34.856, schema 5.945, graph 97.000, rank 4.321 ms |
| `project:octopus`, `max_results=5`, `depth=1` | 127.905 ms | scope size between LM and AE |
| `project:ae`, `max_results=5`, `depth=0` | 79.738 ms | shows graph traversal is the dominant depth=1 cost |
| `project:ae`, `max_results=5`, `depth='causal'` | 84.583 ms | causal traversal filters to caused/requires edges |
| `memory_health(scope=None)` | 56.771 ms | currently under the same MCP runtime lock as recall/status |

Snapshot table-size context from read-only SQL:

- Connections: 8,407 `related`, 411 `contradicts`, 34 `supersedes`.
- Active embedded rows by top scopes: `project:ae` 2,674; `project:octopus` 1,601; `project:lm` 323; `project:online` 224; `global` 163.
- Active schemas by top scopes: `project:octopus` 52; `project:ae` 27; `global` 4; `project:lm` 1.

## Current retrieval path

`MemoryRecallService.memory_recall` resolves a scope plan, collects BM25, vector, schema-trigger, and graph candidates, ranks them, optionally records result access, and optionally inserts a recall event (`src/living_memory/retrieval.py:108-174`).

Hot-path citations:

- Scope resolution happens once per recall (`src/living_memory/retrieval.py:127-132`) and project scopes expand to `(project, global)` (`src/living_memory/scope.py:161-168`); session scopes can expand to `(session, project, global)` (`src/living_memory/scope.py:170-179`).
- BM25 is FTS5-backed per scope with `per_scope_limit = max(25, max_results * 8)` (`src/living_memory/retrieval.py:283-300`), using `nodes_fts MATCH` plus exact `scope` and active filters (`src/living_memory/storage.py:730-764`).
- Vector recall embeds the query, lazily backfills missing embeddings, then scans every embedded active row in every resolved scope (`src/living_memory/retrieval.py:302-348`, `src/living_memory/storage.py:766-795`). Stored vectors are decoded and cached in the service instance (`src/living_memory/retrieval.py:350-368`).
- Schema triggers list up to 1,000 schemas per scope and do token-overlap matching in Python (`src/living_memory/retrieval.py:371-399`).
- Graph traversal starts from every collected candidate and calls `store.list_connections(node_id=...)` per seed/reached node (`src/living_memory/retrieval.py:401-459`, `src/living_memory/storage.py:556-584`). This is the current dominant hot-path cost for `depth=1`.
- Ranking calls `_supersedes_sets()` once per recall (`src/living_memory/retrieval.py:193-203`); `_supersedes_sets` scans `connections WHERE type = 'supersedes'` (`src/living_memory/retrieval.py:473-479`). There are indexes on `(source_id, type)` and `(target_id, type)`, but no `type`-leading index (`src/living_memory/storage.py:1096-1100`).
- Ranking applies scope boost, confidence/usefulness/access/supersedes weighting, schema-trigger boost, and causal graph boost (`src/living_memory/retrieval.py:224-271`, `src/living_memory/feedback.py:237-252`).
- Access logging updates one node per returned result and reloads it (`src/living_memory/storage.py:586-599`); recall-event logging inserts query, scope plan, ambient context, result summary, agent/task/session metadata, and `feedback_applied=0` (`src/living_memory/storage.py:601-648`).
- All MCP tools share one `runtime_lock` (`src/living_memory/server.py:78`), and `memory_recall`, `memory_remember`, `memory_status`, and `memory_health` all execute under it (`src/living_memory/server.py:484-623`).

## Root causes

1. **Graph traversal dominates nontrivial hot recall.** For `project:lm`, `depth=1` graph took ~45.9 ms median out of ~69.8 ms service-hot median; for `project:ae`, graph took ~97 ms out of ~178 ms. One profiled `project:lm` query had 122 candidates before graph, expanded to 261 candidates, and issued 122 `list_connections` calls. This follows directly from `retrieval.py:401-459`: traversal fans out from every candidate, and each neighbor lookup is a separate storage call.

2. **Vector and BM25 cost scale with active rows, not returned rows.** `max_results=5` still scans all embedded rows in the resolved scopes before truncating (`retrieval.py:302-348`). That is fine at 323 `project:lm` embedded rows, but `project:ae` already has 2,674 active embedded rows, many of them duplicate bootstrap chunks. This is a speed and usefulness risk because duplicate active rows increase scan cost and can occupy candidate slots even when the final result count is small.

3. **Scope expansion is useful but doubles/triples scans.** Project recall always includes global (`scope.py:161-168`); session recall can include session, project, and global (`scope.py:170-179`). This is valuable for cross-project lessons, but it means a write-side scope fragmentation fix must canonicalize to one project scope rather than adding alias scopes to recall plans. Alias expansion would multiply BM25/vector/schema/graph work.

4. **Duplicate/file-chunk noise has a retrieval cost, not only a health cost.** Current ranking treats every active duplicate as an independent candidate. Feedback weighting then multiplies each duplicate by confidence, usefulness, access count, and supersedes status (`feedback.py:237-252`). `project:ae`'s 625 duplicate-excess active traces therefore inflate BM25/vector candidate pools and can receive independent access/usefulness boosts.

5. **The current supersedes penalty is cheap only because supersedes is rare.** `_supersedes_sets()` scans all supersedes rows on every recall (`retrieval.py:473-479`). With only 34 `supersedes` connections this is a small rank-time cost. If the dedup fix represents repeated file chunks by adding hundreds or thousands of `supersedes` edges, this becomes a new hot-path regression unless it is indexed, cached, or narrowed.

6. **Access/event logging is not the main median cost today, but it is serialized.** In a temp-copy write benchmark, enabling access and event logging did not materially move the median relative to retrieval variance for `project:lm` or `project:ae`. Still, `record_access` performs a write and reload per returned result (`storage.py:586-599`) and `record_recall_event` inserts JSON (`storage.py:601-648`) under the global server lock (`server.py:78`). Fixes must not add per-candidate logging or extra write amplification on recall.

7. **Health observability can become a hot-lock problem.** `memory_health` already performs multiple aggregate queries and a Python age loop over active traces (`resources.py:323-459`), measuring ~56.8 ms median on the snapshot under in-process conditions. Adding DB size, feedback ratio, leakage candidates, or never-accessed ratio is acceptable if it stays SQL-aggregate-only; running a latency probe inside every `memory_health` call would block all other MCP tools under the same runtime lock.

## Chosen latency-safe guidance

The simplest central rule for downstream fix nodes: **reduce active noisy rows and keep recall-time work bounded by the existing resolved scopes; do not add new per-recall scans, alias expansion, or writes.**

Concrete guidance:

- Scope hygiene: canonicalize write/ambient scope names before storage or recall planning. Do not solve leakage by searching extra aliases such as both `rise/*` and `project:octopus`; that would multiply scans and leave old fragmentation in ranking.
- Dedup/file-chunk noise: use stable content fingerprints to mark redundant active traces as superseded/decayed while preserving append-only rows. This should reduce active candidate count before retrieval. If `supersedes` edges are used for dedup, add a `type`-leading connection index or a small invalidated cache before supersedes volume grows.
- Feedback linkage: improve recall-to-remember matching on the write path (`pending_recall_events` / `apply_pending_recall_feedback`), not by making recall collect or log more. Any larger `limit` must keep the query indexed and bounded.
- Health/observability: extend `memory_health` with SQL aggregates and optional/cached latency data. Default `memory_health` should not run recall benchmarks.
- Ranking/usefulness: suppress duplicate active candidates before they reach ranking rather than trying to add another score penalty at rank time. Score penalties still leave duplicate rows in BM25/vector/graph scans.

## Rejected alternatives

- **Disable vector search.** Rejected: it would reduce scan time but breaks existing semantic and cross-language recall coverage tested in `tests/test_retrieval.py:82-135` and `tests/test_acceptance_contract.py:64-88`.
- **Disable graph traversal by default.** Rejected: it would improve `depth=1` latency but would regress causal, decision, and correction behavior covered by `tests/test_graph_recall.py:107-130` and decision-history tests. Use `depth=0` explicitly for callers that do not need graph context.
- **Add ANN/vector-index infrastructure now.** Rejected: current scale is thousands of vectors, the deterministic hash fallback must stay intact (`src/living_memory/embeddings.py:261-345`), and the root goal forbids heavy subsystems without proof. Soft-decaying duplicates and bounding graph work are simpler first fixes.
- **Search every possible scope alias to recover leaked memories.** Rejected: it hides scope hygiene failure while making every recall slower. Canonicalize future writes and optionally migrate/soft-link old leaked scopes separately.
- **Turn off access logging or recall events.** Rejected: it protects latency but breaks feedback linkage, health ratios, and provenance. Keep logging per returned result/event only; do not add per-candidate logging.
- **Compute hot recall latency on every `memory_health`.** Rejected: it would run a recall under the global runtime lock and mutate unless carefully disabled. Prefer an explicit benchmark command or a cached opt-in health field.
- **Hard-code project-specific duplicate/ranking exceptions.** Rejected: the same root causes apply to AE, Octopus, LM, Online, and future projects. Use central fingerprints, scope canonicalization, and bounded retrieval rules.

## Deterministic micro-benchmark policy

Every downstream fix node touching recall ranking, scope resolution, dedup, feedback, access logging, health, storage indexes, or decay must run a paired before/after benchmark in the same worktree and report:

- `git rev-parse --short HEAD`
- Python version and whether `numpy` is importable
- DB source and MD5
- query/scope/depth/max_results matrix
- median and p95 from at least 7 measured iterations after one warmup
- whether the benchmark was no-write (`log_access=False, log_event=False`) or temp-copy write-enabled

Regression gate:

- For hot recall service-path medians: after/before must be `<= 1.10` and absolute increase must be `<= 10 ms` for `project:lm`; for `project:ae` and `project:octopus`, after/before must be `<= 1.10` and absolute increase must be `<= 20 ms`.
- For p95: after/before must be `<= 1.20`; explain any higher value with the raw samples.
- For `memory_health(scope=None, top_stale=5)`: after/before median must be `<= 1.25` and absolute increase must be `<= 25 ms`, unless latency reporting is explicitly opt-in and disabled by default.
- A fix that intentionally trades latency for usefulness must include a failing/passing usefulness assertion and the measured latency cost; do not hide the tradeoff.

Use the same frozen snapshot for no-write read-path comparisons:

```bash
md5sum /tmp/lm-baseline-replay.sqlite3
```

Expected MD5:

```text
58fc12fd47c5f71e5b8867ee71590668  /tmp/lm-baseline-replay.sqlite3
```

Canonical hot-service benchmark:

```bash
PYTHONPATH=src LIVING_MEMORY_EMBEDDING_BACKEND=hash python3 - <<'PY'
import json
import statistics
import time
from pathlib import Path

from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore

DB = Path("/tmp/lm-baseline-replay.sqlite3")
MATRIX = [
    ("project:lm", "Living Memory hot recall latency baseline duplicate density feedback applied", 1, 1),
    ("project:lm", "Living Memory hot recall latency baseline duplicate density feedback applied", 1, 5),
    ("project:ae", "bootstrap file chunk duplicate density feedback applied goal tree", 1, 5),
    ("project:ae", "bootstrap file chunk duplicate density feedback applied goal tree", 0, 5),
    ("project:octopus", "OCPA bilingual lifecycle canary evaluation trainer planner", 1, 5),
    ("project:octopus", "why did lifecycle canary fail root cause trainer planner", "causal", 5),
]
ambient = {
    "task": "memory-quality-root-fixes/per-fix-latency-benchmark",
    "agent": "codex",
    "benchmark": "service_hot_no_write",
}

with MemoryStore(DB) as store:
    service = MemoryRecallService(store)
    report = []
    for scope, query, depth, max_results in MATRIX:
        service.memory_recall(
            query,
            scope=scope,
            depth=depth,
            max_results=max_results,
            ambient_context=ambient,
            log_access=False,
            log_event=False,
        )
        samples = []
        for _ in range(9):
            started = time.perf_counter()
            results = service.memory_recall(
                query,
                scope=scope,
                depth=depth,
                max_results=max_results,
                ambient_context=ambient,
                log_access=False,
                log_event=False,
            )
            samples.append((time.perf_counter() - started) * 1000)
        ordered = sorted(samples)
        report.append({
            "scope": scope,
            "depth": str(depth),
            "max_results": max_results,
            "count": len(results),
            "samples_ms": [round(value, 3) for value in samples],
            "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
        })
print(json.dumps(report, indent=2, ensure_ascii=False))
PY
```

Cold-helper compatibility benchmark, matching `artifacts/baseline.md`:

```bash
PYTHONPATH=src LIVING_MEMORY_EMBEDDING_BACKEND=hash python3 - <<'PY'
import json
import statistics
import time
from pathlib import Path

from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore

DB = Path("/tmp/lm-baseline-replay.sqlite3")
query = "Living Memory hot recall latency baseline duplicate density feedback applied"
ambient = {"task": "memory-quality-root-fixes/per-fix-latency-benchmark", "agent": "codex"}

with MemoryStore(DB) as store:
    samples = []
    memory_recall(store, query, scope="project:lm", ambient_context=ambient, max_results=1, depth=1, log_access=False, log_event=False)
    for _ in range(7):
        started = time.perf_counter()
        memory_recall(store, query, scope="project:lm", ambient_context=ambient, max_results=1, depth=1, log_access=False, log_event=False)
        samples.append((time.perf_counter() - started) * 1000)
print(json.dumps({"samples_ms": [round(v, 3) for v in samples], "median_ms": round(statistics.median(samples), 3)}, indent=2))
PY
```

Write-enabled logging/feedback benchmark must use a temp copy, never the snapshot or live DB:

```bash
PYTHONPATH=src LIVING_MEMORY_EMBEDDING_BACKEND=hash python3 - <<'PY'
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore

source = Path("/tmp/lm-baseline-replay.sqlite3")
query = "bootstrap file chunk duplicate density feedback applied goal tree"
ambient = {
    "task": "memory-quality-root-fixes/per-fix-latency-benchmark",
    "agent": "codex",
    "benchmark": "temp_copy_write_enabled",
}

with tempfile.TemporaryDirectory(prefix="lm-recall-bench-") as tmp:
    db = Path(tmp) / "bench.sqlite3"
    shutil.copy2(source, db)
    with MemoryStore(db) as store:
        service = MemoryRecallService(store)
        report = []
        for log_access, log_event in [(False, False), (True, False), (True, True)]:
            service.memory_recall(query, scope="project:ae", ambient_context=ambient, max_results=25, depth=1, log_access=log_access, log_event=log_event)
            samples = []
            for _ in range(7):
                started = time.perf_counter()
                service.memory_recall(query, scope="project:ae", ambient_context=ambient, max_results=25, depth=1, log_access=log_access, log_event=log_event)
                samples.append((time.perf_counter() - started) * 1000)
            report.append({
                "log_access": log_access,
                "log_event": log_event,
                "samples_ms": [round(value, 3) for value in samples],
                "median_ms": round(statistics.median(samples), 3),
            })
print(json.dumps(report, indent=2))
PY
```

## Per-fix benchmark instructions

### Scope hygiene

Run the canonical hot-service matrix before and after. Add a focused fixture test/bench that compares explicit `scope="project:octopus"` against `ambient_context={"cwd": "/root/p/octopus/.worktrees/_node_exec_rise"}` after canonicalization. The canonicalized ambient path should resolve to the same plan and should not be more than 5 ms or 5% slower than explicit scope in a 7-sample median.

Do not add recall-time alias expansion. If old leaked scopes are handled, do it by write-time canonicalization plus a separate audit/migration/soft-link policy, not by searching extra aliases on every recall.

### Dedup/file-chunk noise

Run the canonical matrix and include `project:ae` at both `depth=0` and `depth=1`, because dedup should reduce BM25/vector work and graph fan-out. The fix should demonstrate:

- Active duplicate-excess in the targeted file-chunk population drops by at least 80%, or `project:ae` duplicate density falls below 5%, whichever the dedup node chooses as its numeric contract.
- Top recall results for a file-chunk-heavy query include at most one active result for the same stable content fingerprint/path/chunk.
- Hot recall median does not regress by the policy above; improvement is expected for `project:ae`.

If the implementation creates many `supersedes` edges, add a supersedes-heavy microbench with at least 1,000 `supersedes` connections and assert rank time remains within the same threshold. The simplest acceptable supporting change is a `type`-leading index or a cache invalidated by connection writes; do not introduce a background indexer.

### Feedback linkage

Recall latency should be unchanged because the fix belongs on the write path. Run the canonical hot-service matrix anyway, then run a temp-copy write benchmark that:

- creates or uses several pending recall events in one scope/task/session,
- calls `memory_remember`,
- verifies all intended same-task events are marked `feedback_applied`,
- verifies unrelated task/session events remain pending,
- records `memory_remember` wall time.

The matching query must stay bounded and index-friendly. Increasing the default `apply_pending_recall_feedback` limit is acceptable only if `pending_recall_events` still fetches a small multiple of the limit and filters in Python, as it currently does at `src/living_memory/storage.py:681-709`.

### Health/observability

Measure `memory_health(store, scope=None, window_hours=168, top_stale=5)` before/after on the snapshot. New metrics must be SQL aggregates or bounded lists. A latency section must be opt-in or cached; default `memory_health` must not call `memory_recall`.

Expected default health output additions are feedback ratio, leakage candidates, never-accessed ratio, DB size, and instruction/test-contract status. None require embedding scans or graph traversal.

### Retrieval/ranking/index changes

Any change in `src/living_memory/retrieval.py`, `src/living_memory/storage.py` indexes, `src/living_memory/feedback.py:237-252`, or graph traversal must run:

- canonical hot-service matrix,
- cold-helper compatibility benchmark,
- a synthetic 1,000-trace recall test from `tests/test_acceptance_contract.py:64-88`,
- `tests/test_retrieval.py`,
- `tests/test_graph_recall.py`,
- `tests/test_feedback_weights.py`.

The report must separate usefulness regressions from latency regressions. Useful but slower changes need explicit acceptance from the integration node.

## Downstream policy

Use the frozen snapshot for read-path measurement. Use temp copies for any benchmark that writes `access_count`, `recall_events`, feedback, decay, or supersedes rows. Do not mutate the live DB during fix-node iteration. Live MCP latency probes are out of scope for this discovery artifact; any later integration-owned probe must carry its own scope-contract boundary and provenance tags.
