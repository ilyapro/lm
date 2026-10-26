# Recall quality gate

Code: `src/living_memory/score_gate.py` (gate), `src/living_memory/delivery.py`
(stub shaping). Tests: `tests/test_recall_score_gate.py`.

## Why

The quality gate prevents weak ranked results from taking full delivery slots.
It is optional and uses a configured threshold on current retrieval evidence.
The map shares this qualification while retaining its own grouping and budget.

## Valves

| env | values | default |
| --- | --- | --- |
| `LM_RECALL_MIN_SCORE` | float threshold on the gate score | unset = off |
| `LM_RECALL_GATE_FORM` | `drop` or `stub` | `drop` |

The gate is off when `LM_RECALL_MIN_SCORE` is unset, empty, unparsable, zero or
negative, NaN or infinite. In that case `apply_score_gate` is the plain cut
(`ranked[:max_results]`, `ranked[max_results:]`), and no result carries
`withheld`. `LM_RECALL_GATE_FORM` is read only while the gate is on. Any value
other than `stub` (case-insensitive) reads as `drop`.

## Where it runs

`MemoryRecallService.memory_recall` ranks the results, collapses schema
duplicates, and then calls
`apply_score_gate(ranked, max_results, plan=, demotions=, causal_mode=,
corrections_by_superseded=)`, which
returns `(delivered, residual)`. `delivered` is what the recall returns and
records in its event. `residual` includes both passing overflow and rejected
entries. The optional map uses the same `QueryRelevance` object to qualify that
residual, with the full ranking available for correction inheritance. It never
uses residual membership alone as evidence of applicability.

## Forms

Every rank uses the same evidence gate for inline content. A found source that
does not qualify can occupy a spare slot as a content-less, addressable reference.
This preserves bounded access without claiming that the source answers the query.

- **drop**: walk `ranked` in order and deliver results that pass, until
  `max_results` results are delivered. If capacity remains, fill it with failing
  results in ranked order, marked `withheld="below_threshold"`. Every ranked
  result that is not delivered goes into `residual` in ranked order: results
  that failed the gate, and passing results past the cap. A passing result
  from below the plain cut can move up into a freed slot; references never
  displace passing results. When none pass, the bounded cut consists entirely
  of references, with no inline content.
- **stub**: keep the plain cut. Results inside it that fail the gate are
  delivered as `dataclasses.replace(result, withheld="below_threshold")`, and
  `residual` is the plain-cut residual. Tail slots are not filled from below.
  `withheld` goes into the recall event summary through `RecallResult.to_dict`
  and `_recall_result_summary`.

The service passes the same `supersedes` edges to ranking and final selection.
When both endpoints remain in the ranked list and the original passes the gate,
its correction also qualifies: matching the original establishes relevance for
the correction that replaces it. This propagates through correction chains and
works in both forms. A correction absent from the ranked list cannot qualify
through an edge; unrelated results use their own gate scores. The valve-off
plain cut is unchanged.

## Stub shape (delivery)

`shape_recall_results` renders a result with `withheld` set as
`delivery: "below_threshold"` (`DELIVERY_BELOW_THRESHOLD`), checked before
every other rule:

- `node.content` is `""`. The key is kept, so the node dict keeps its full key set.
- `content_ref` is `{node_id, full_content_chars}`, plus `fetch` when the entry
  is not sparse. `memory_lookup(node_id=…)` returns the whole node.
- Provenance and context are dieted the same way as on any other non-full entry.
- Every score field stays.
- It takes no snippet-ladder slot and never counts as a twin bearer. Its text
  was not delivered, so a later result with the same content is delivered on
  its own terms.

The withheld delivery class applies only while the gate is enabled.

A withheld node retains its selected event entry and lookup reference, but its
empty response body does not establish content delivery. If a later recall
selects it above the gate, session dedup permits its current text unless a
separate full/snippet response already supplied that content.

## Qualification evidence

Ranking and qualification share `_blend_candidate_score`, with different inputs
and feedback treatment. Ranking keeps learned scope weights, query anchors and
historical feedback. `QueryRelevance` uses the fixed `REFERENCE_WEIGHTS`
(lexical 0.4, vector 0.4, graph 0.2) and current query/source evidence:

```
gate_score(result, demotions, *, plan=None, causal_mode=False)
  = _blend_candidate_score(lexical_score, vector, source_graph_score, trigger,
                           REFERENCE_WEIGHTS, plan, causal_mode,
                           superseded=result.superseded, superseding=False,
                           qualification=True)
    * demotions.get(node_id, 1.0)
```

`lexical_score` is the fraction of distinct normalized query terms present in
the source's current content. Empty query terms give zero support. The shared
tokenizer retains its splitting, synonyms and stemming; evidence normalization
also keeps numbers, including identifier components, and conditions/negation.
The same normalization supplies trigger overlap, query coverage and name matches.
The default embedding and search-expansion features remain unchanged.

Reciprocal FTS rank (`bm25_score`) still discovers and orders candidates. Neither
it nor raw corpus-dependent BM25 scores qualify source content. Lexical coverage
is computed for every found candidate, including vector, graph and anchor
discoveries. A recognized schema name supplies complete lexical support under
the existing name semantics. Coverage is textual correspondence, not semantic
entailment or an interpretation of a condition's polarity.

The same calculation retains the strong-vector floor (`vector >= 0.65`),
graph-weight floor (0.25, or 0.75 in causal mode), scope boost and causal boost.
Ordinary trigger overlap contributes with its existing content support and
query coverage. The individual score excludes the correction's ranking boost;
a present correction can also inherit its original's qualifying evidence.

Qualification keeps low source confidence, the superseded penalty (0.2), and
query-relative/hub demotions. It ignores access and the **entire** net usefulness
accumulator, including negative values: lookup credit can raise that accumulator
without adding source evidence or cancelling confirmed query-relative
irrelevance. Positive confidence does not lift the score. These restrictions
apply to both qualification scales; positive ranking history cannot offset them.
Confirmed `used` and grounded consumption retain their existing effects on
query-relative negative evidence. Lookup retains once-per-event/node credit
and anchors without acquiring those cancellation effects.

`RecallResult.source_graph_score` is transient evidence, separate from the
ranker's effective `graph_score`. Its bounded seeds are selected independently
by `max(lexical_score, vector_score)`, with source id breaking ties. Zero support
does not seed source activation. It excludes anchor activation and uses genuine
source relations:

- Pure `co_access` edges contribute no source evidence.
- Cluster membership retains its fixed membership contribution, without
  access/usefulness/confidence reinforcement of that edge.
- Known content-reference, derivation and shared-source annotations retain
  their writer's source contribution, capped by the stored weight. A merged
  historical `basis` does not discard independent `content_reference` or
  `derived_from` evidence or turn the row's historical maximum into evidence.
- Unrecognized metadata keeps its ordinary relation semantics. Corrections,
  explicit relations and scope/decay restrictions remain effective.

The existing bounded graph reader applies source projection **before** ordering
and its per-node adjacency limit. Historical edges therefore cannot spend source
slots or displace a genuine source relation merely by gaining weight. The same
walker computes source paths with independent best-path state when source roots
or relation contributions differ from discovery. Only identical roots and
pure-source walks reuse ordinary activation. Ranking keeps its own traversal,
bounded neighbors, anchors and order.

**Offline use:** `gate_score`, `trigger_gate_score` and `passes_gate` accept a
`RecallResult` and recall demotions without a store. Supply the scope plan for
its boost; without one it is 1.0. Missing `lexical_score` or `source_graph_score`
means zero contribution from that source channel; discovery scores cannot
substitute for either. These transient fields are not added to the wire or
recall-event history. Diagnostic
ranking scores continue to describe discovery, not qualification.

## Trigger qualification

An ordinary trigger match has `trigger_score = 0.95 + 0.05 * overlap` after
clearing the existing overlap requirement. Its separate availability scale is:

```
trigger_gate_score = feedback_weighted_score(node, trigger_score / 0.95,
                                             superseded=..., qualification=True)
                    * coverage_factor * demotion
passes if trigger_gate_score >= 0.5 or gate_score >= LM_RECALL_MIN_SCORE
```

For schema recipes, `coverage_factor` is 1: a short instruction can remain
available as one clause of a larger query. For other procedural carriers it is
squared query coverage. Traces and results without a trigger have no separate
trigger scale. The scale retains low source confidence, correction penalties
and query/hub demotions; global usefulness and access do not affect it.
Scope and causal boosts belong only to the channel scale.

Under `LM_RECALL_SCHEMA_TRIGGER=name`, there is no trigger scale. Every result
uses the channel qualifier; a recognized schema name supplies lexical evidence.
See [schema trigger behavior](recall-schema-trigger.md).

## Invariants

- With the main gate off, main selection is the plain cut. The optional map
  still qualifies its residual through the shared evidence object.
- Every inline content bearer qualifies, regardless of rank.
- Unchanged current query/source evidence has unchanged qualification after
  inspection or changes in access, usefulness or anchor reinforcement.
- FTS competitors can change discovery order without changing the lexical
  support of a source that remains found. Source graph seeds and paths use
  source evidence throughout. Bounded discovery can still omit candidates or
  relations; this does not promise invariance under every corpus mutation.
- Source changes, genuine relations/corrections and confirmed negative
  applicability affect qualification through their own evidence.
- `delivered + residual` partitions `ranked`; both gate forms preserve bounded
  references and exact lookup. Only qualified residuals enter the map.
- Lookup is optional and returns complete source content. It does not resolve
  contradictions, certify relevance or promise usefulness. Consumers can narrow
  a query or stop when only weak sources were found.

These are eligibility and source-access guarantees. They do not establish
semantic precision, autonomous source selection or total token savings.
