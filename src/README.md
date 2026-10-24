# Living Memory source

`living_memory` contains the SQLite store, deterministic materializers,
retrieval service and MCP interface.

`memory_teach` appends a corrective trace, records the original's correction,
and refreshes its recorded derivatives in one transaction. Procedural schemas
and evidence carriers, clustered concepts and cross-scope promotions use their
existing writers. Search chunks encode the current content. A cluster or
promotion can keep a source representative in the legacy node embedding for
consolidation similarity; writers pass it as `representative_embedding`,
separately from a precomputed embedding of the content. A changed digest
encodes its own chunks atomically with its content and sources. Unchanged
digests keep their current chunks even when their representative changes.
Writers without a vector invalidate changed content for lazy encoding.

Ordinary recall repairs at most 16 pending nodes per call, sharing that budget
among the searched scopes and taking the oldest pending content first. The
limit bounds nodes, not the duration of encoding one large node. Lexical and
graph retrieval remain available for pending content. Opening an older store
once invalidates only single-chunk cluster/promotion concepts: their vectors
may describe a source instead of their digest. Multi-chunk content, other
materializers, node identities, sources and validity are preserved. Repair
uses the same recall queue and configured encoder, without a full
consolidation or an additional language-model call.

Materializer provenance and explicit `derived_from` edges establish dependency.
An ordinary trace's `source_traces` can contain recall history and does not make
it dependent on everything it read. Names, similar text and shared scope also
do not establish dependency. Each recorded input follows its explicit
replacement independently, preserving untouched steps and constraints. Later
materialization reads current inputs at its write boundary.

Writer-declared procedural steps can form a binding schema. Case records stay
whole in a nonbinding evidence carrier. An explicit derivative without a known
materializer becomes decayed when its source changes. Exact `memory_lookup`
preserves original sources, corrections and historical content with their
correction or decay validity.

`memory_remember` and `memory_teach` share one background consolidation
scheduler. The fixed cadence is every 100 admitted trace writes per scope;
`LM_AUTO_CONSOLIDATE_POLICY=adaptive` uses every 5 below 50 writes, every 25
below 500, then every 100. Replacements and rejected alternatives count as
writes. A batch crossing an interval schedules one pass. Retained trace rows
supply the count through a covering index, including forgotten and decayed
traces; no separate progress state is needed. The indexed read is linear in
the scope's retained trace count.

Active traces supply the material and determine the existing merge floors in
`consolidation.py`, independently of cadence. Scheduling follows the required
knowledge/feedback commit and is nonfatal. The scheduler coalesces requests,
backs off after failures and resumes durable pending scopes on server start.
