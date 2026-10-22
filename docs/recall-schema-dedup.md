# Recall: copies of a recorded procedure take one slot

`src/living_memory/schema_dedup.py` runs on the ranked recall results before the
quality gate and `max_results` cut. `LM_RECALL_SCHEMA_DEDUP=1` (also
`true`, `yes`, or `on`) enables it; other values leave the ranked list intact.

The stage keeps the highest-ranked copy only when all of these agree:

- scope and node level;
- the recorded procedural materializer and its normalized group key;
- the complete recorded source set;
- the complete content.

The group key is `procedure_key`, falling back to `task_pattern`, then
`procedure_id`. A shared title or trigger is a retrieval label, not evidence
that two groups are the same. Different scopes, groups, inputs or content stay
visible. Unknown legacy derivations stay visible too.

Freed slots go to the next ranked results. This stage changes no stored node,
source attribution or feedback identity; exact `memory_lookup` remains available.

Consolidation uses recorded source membership and supersedes chains to continue
cluster and cross-scope derivations. Their mutable lexical labels and lossy
digests do not identify owners. A partial pass preserves an owner's other current
sources. Multiple overlapping owners remain separate, with no mass collapse.
