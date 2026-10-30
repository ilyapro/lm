# Query anchors

Query anchors connect remembered questions to memory nodes that a grounded
consumption actually used. Retrieval can match a new query to a remembered
question and follow its weighted edges into the memory graph.

Code: `src/living_memory/query_anchors.py`. Storage:
`query_anchors` and `query_anchor_edges` in `storage.py`.

## Query identity

An anchor is identified by `recall_fingerprint(query, scope)`: the exact query
with whitespace collapsed, in its normalized scope. Case, word order,
identifiers, permission and conditions remain significant. The same query in
two scopes has independent anchors and evidence.

`upsert_anchor` reinforces an existing exact identity or creates a new anchor.
It preserves the existing anchor's query, fingerprint and embedding.
`resolve_query_anchor` uses the same identity without reinforcing it or adding
edges; explicit negative feedback uses this path.

Similarity never merges query identities. For example, “restart is allowed
before approval” and “restart is forbidden before approval” remain independent
even if their embeddings are identical. Similarity remains useful for finding
both anchors at retrieval time.

## Evidence and matching

A grounded consumption adds weight to the anchor's used-node edges. The default
contribution is `ANCHOR_EDGE_WEIGHT = 0.25`, saturating at 1.0. Repeated evidence
increments the edge's hit count. Targets resolve to current replacements before
writing; missing or decayed targets without a live replacement are skipped.

`match_anchors` selects and ranks live anchors by descending cosine similarity,
breaking ties by anchor ID before applying the limit. Scope order does not
change membership or ordering for the same admitted anchors. Its defaults are
a `0.60` similarity floor and a limit of five anchors. It compares vectors only
at the same recorded width and respects the retrieval scope plan. Returned
targets are live nodes ordered by edge weight;
an anchor with no live targets can match with an empty target list.

The retrieval path can use these targets as graph seeds. Related queries can
also receive ranking adjustments from
[query-relative irrelevance evidence](query-irrelevance.md). Neither a semantic
match nor a ranking adjustment proves that two questions or memory assertions
are identical, or changes an assertion's validity.

## Replacements and lifecycle

Anchor targets follow genuine `supersedes` corrections, including
`memory_teach`. Compatible exact-copy bookkeeping supplies a separate
replacement route: current bytes, scope, level and recorded source identity
must match. Mutable correction history does not break an otherwise valid copy
lineage, so an older copy can still reach the genuine correction of its current
copy. A technical duplicate edge is not a semantic correction.

Known inferred-only supersedes metadata, including `rule=R1c` or
`kind=content_correction`, cannot redirect an anchor. Explicit correction
contributions in mixed metadata keep their meaning. Unknown
supersedes metadata retains ordinary correction semantics. Consolidation can
also provide an explicit source replacement. The write hook and
`sweep_anchor_edge_migration` use the same replacement policy; the sweep repairs
only anchor edges with a known live destination.

Replacement traversal stops on a cycle or after 32 steps. A missing or decayed
replacement is not followed. These limits do not erase nodes, source provenance
or known supersession from the memory lookup interface.

Anchors decay after 90 days without a match, using their first-seen time when
no match is recorded. Decay preserves their edges. A later exact query upsert
revives the anchor with its accumulated evidence.

## Storage, bounds and verification

The anchor tables keep query evidence separate from node-to-node relations.
Retrieval caches prepared anchor vectors using the store's transactional anchor
revision; stores without that revision use uncached matching. This cache covers
membership and vector changes, while edge targets are read at match time.

Matching limits bound the number of returned anchors; the ordinary recall
limit bounds delivered memory results. Distinct questions or facts may fall
outside those limits without being declared duplicates.

[The backfill runbook](query-anchors-migration.md) documents read-only rehearsal,
backup, exact-identity replay, interruption recovery and verification. Backfill
uses the same identity and replacement rules as the live path. It does not
reconstruct historical similarity merges or authorize corpus cleanup.

Targeted tests cover exact repeats, distinct queries with identical vectors,
scope boundaries, semantic matching, negative-feedback identity, genuine teach,
copy replacement, historical heuristic edges, cache invalidation and resumable
backfill. Retrieval evaluations should use one fixed snapshot and goldset for
both arms, keep scored events outside the anchor-building window, and report
exact repeats separately from questions absent from that window. A successful
anchor match alone establishes neither retrieval quality nor semantic validity.
