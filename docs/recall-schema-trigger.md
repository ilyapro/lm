# Recall schema triggers

Code: `retrieval.py` (`schema_trigger_by_name`, `_collect_schema_triggers`,
`_named_schemas_first`) and `score_gate.py` (`QueryRelevance`,
`trigger_gate_score`).

## Ordinary trigger discovery

With `LM_RECALL_SCHEMA_TRIGGER` unset or any value other than `name`, trigger
collection requires half of a saved trigger's terms, including for a short
instruction within a compound query. A trigger adds a candidate and can admit
it across scopes. Ranking retains the content's vector similarity and the
existing graph weight as a small discovery floor. There is no unconditional
trigger multiplier.

The collector also records the fraction of meaningful query terms covered by
the trigger. A schema recipe or procedural carrier citing source records gets
a channel contribution proportional to squared query coverage. An arbitrary
concept with only a trigger does not get that contribution. Vector, BM25,
graph, scope, feedback and corrections retain their ranking roles.

## Qualification

Main recall and the optional map share `QueryRelevance`. The channel scale
uses current query/source evidence with fixed weights. The separate trigger
scale divides trigger overlap by 0.95 and uses squared query coverage for
case-evidence carriers. Schema recipes retain short-instruction availability
with a coverage factor of 1. A result qualifies if either scale passes.

Both scales preserve low source confidence, superseded penalties and
query-relative/hub restrictions. Access, the net usefulness accumulator and
anchors affect discovery and ranking, not content qualification. A lookup
cannot qualify unchanged content by reinforcing those historical signals.
See the [shared qualification contract](recall-score-gate.md).

## Name mode

`LM_RECALL_SCHEMA_TRIGGER=name` is case-insensitive and disabled by default.
It changes no provider, model or threshold.

- Trigger overlap neither discovers nor scores candidates. Every node enters
  through BM25/vector/graph and uses the channel qualifier.
- Ranking uses the fixed `REFERENCE_WEIGHTS` rather than learned scope weights,
  and omits the positive correction prior. Historical ranking feedback and
  anchor discovery remain; sharing weights does not make ranking history
  qualification evidence. Corrections retain structural precedence.
- If the query's token set equals a ranked schema's trigger token set, the
  schema receives BM25 1.0: the whole query names its title. It moves first if
  that current evidence passes the channel qualifier, or unconditionally when
  the main gate is off. Named schemas retain their relative ranking order.
- The name match is reported as `trigger_score=1.0` and `trigger` in diagnostic
  methods. This marker does not restore the separate trigger scale.
- The trigger scan reads `id` and `context.trigger` after ranking. Search scope,
  source identity, feedback attribution and exact lookup remain unchanged.

In both modes, a source that fails qualification can remain a bounded
`below_threshold` reference. Rank or a matching label alone does not establish
procedure authority or semantic applicability.
