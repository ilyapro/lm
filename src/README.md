# Living Memory source

`living_memory` contains the SQLite store, deterministic materializers,
retrieval service and MCP interface.

`memory_teach` appends a corrective trace, records the original's correction,
and refreshes its recorded derivatives in one transaction. Procedural schemas
and evidence carriers, clustered concepts and cross-scope promotions use their
existing writers. The store invalidates changed content's chunks; ordinary
recall reads the current content without a full consolidation or an additional
language-model call.

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
