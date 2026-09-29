# Recall quality gate

Code: `src/living_memory/score_gate.py` (gate), `src/living_memory/delivery.py`
(stub shaping). Tests: `tests/test_recall_score_gate.py`. Goal: recall-precision,
node score-gate.

## Why

A `memory_recall` answer always fills up to `max_results`, however weak the
tail is. Agent marks (`recall_feedback_marks`) show that the final score
predicts usefulness well: below 0.3, 4–6% of results are marked `used`; at 0.6
and above, 42–67%. The gate stops weak results from taking a full slot. It is
off by default. Choosing the threshold is not part of this change: the
precision-measure-handoff node calibrates it on snapshots.

## Valves

| env | values | default |
| --- | --- | --- |
| `LM_RECALL_MIN_SCORE` | float threshold on the gate score | unset = off |
| `LM_RECALL_GATE_FORM` | `drop` or `stub` | `drop` |

The gate is off when `LM_RECALL_MIN_SCORE` is unset, empty, unparsable, zero or
negative, NaN or infinite. In that case `apply_score_gate` is exactly the old cut
(`ranked[:max_results]`, `ranked[max_results:]`), and no result carries
`withheld`. `LM_RECALL_GATE_FORM` is read only while the gate is on. Any value
other than `stub` (case-insensitive) reads as `drop`.

## Where it runs

`MemoryRecallService.memory_recall` ranks the results, collapses schema
duplicates, and then calls
`apply_score_gate(ranked, max_results, plan=, demotions=, causal_mode=)`, which
returns `(delivered, residual)`. `delivered` is what the recall returns and
records in its event. `residual` is what `recall_map` describes.

## Forms

**Rank 1 is always delivered in full**, whatever its gate score. An answer is
therefore never empty just because the gate is on.

- **drop**: walk `ranked` in order and deliver rank 1 plus each following
  result that passes, until `max_results` results are delivered. Every ranked
  result that is not delivered goes into `residual` in ranked order: results
  that failed the gate, and passing results past the cap. A passing result
  from below the old cut can move up into a freed slot. The answer can be
  shorter than `max_results`.
- **stub**: keep the old cut. Results inside it that fail the gate are
  delivered as `dataclasses.replace(result, withheld="below_threshold")`, and
  `residual` is the old residual. Tail slots are not filled from below.
  `withheld` goes into the recall event summary through `RecallResult.to_dict`
  and `_recall_result_summary`.

## Stub shape (delivery)

`shape_recall_results` renders a result with `withheld` set as
`delivery: "below_threshold"` (`DELIVERY_BELOW_THRESHOLD`), checked before
every other rule:

- `node.content` is `""`. The key is kept, so the node dict keeps its full key set.
- `content_ref` is `{node_id, full_content_chars}`, plus `fetch` when the entry
  is not sparse. `memory_lookup(node_id=…)` returns the whole node.
- Provenance and context are dieted the same way as on any other non-full entry.
- Every score field stays.
- It takes no snippet-ladder slot, and it never counts as a twin or
  near-duplicate bearer. Its text was not delivered, so a later result with the
  same content, or one that the near-dup map maps onto it, is delivered on its
  own terms.

This is an added delivery class, and the existing classes keep their meaning.
Without the gate, nothing changes. `below_threshold` is not in the server's
`_DROPPABLE_TRAILING_STUBS`, so the repeat-gating trailing drop leaves it alone.

Known interaction: a stubbed node is recorded in the event like any delivered
node. `store.delivered_node_ids` therefore counts it, and within the same
transport session a later recall can render it as `session_duplicate` (a
one-line preview) instead of `full`. `memory_lookup` still reaches it. Measure
this before turning `stub` on together with session dedup.

## Gate score

The ranker's `score` uses the store's per-scope retrieval weights
(`store.get_retrieval_weights`). These act like a moving average of the last
~100 credits, so a threshold on `score` would mean something different next
week. The gate therefore computes its own score:

```
gate_score(result, demotions, *, plan=None, causal_mode=False)
  = _blend_candidate_score(bm25, vector, graph of the result,
                           REFERENCE_WEIGHTS, plan, causal_mode,
                           superseded=result.superseded, superseding=False)
    * demotions.get(node_id, 1.0)
```

- `REFERENCE_WEIGHTS = bm25 0.4, vector 0.4, graph 0.2` is a fixed module
  constant: the seeded `global` family weights, the only default that gives
  every channel weight. It never reads the store.
- It calls the ranker's own blend function, so it applies the same multipliers
  as the ranker:
  - the strong-vector floor (`vector ≥ 0.65` → base ≥ vector)
  - the graph-weight floor (0.25, or 0.75 in causal mode)
  - the scope boost (`1 + 0.2 · (len(plan.scopes) − rank − 1)`)
  - the node feedback multiplier (confidence × usefulness × access, capped at 1.4)
  - the superseded penalty 0.2×
  - the causal boost 1.5× with graph evidence
  - the per-node demotion from `demotions` (query-relative irrelevance and hub
    suppression, merged by the service)
- `graph_score` is the one the ranker kept (`effective_graph`), so the
  anchors-off fallback is respected.
- The trigger channel is left out of this blend. See the next section.
- One multiplier is left out: the 1.2× boost for a node that *corrects*
  another is not visible on a `RecallResult`. The gate is 1/1.2 stricter on
  corrections, and the ranker's correction-dominance pass still orders them.
- Without `plan` (offline), the scope boost is 1.0.

**Offline use:** `gate_score`, `trigger_gate_score` and `passes_gate` take a
`RecallResult` (or any object with `node`, `bm25_score`, `vector_score`,
`graph_score`, `trigger_score`, `superseded`) and a demotions mapping. They have
no store and no env, so a harness can score snapshot results at any threshold.
Note that the node's feedback fields (`confidence`, `usefulness_score`,
`access_count`) are read from `result.node`, so replay against the snapshot's
nodes.

## Trigger-found schemas

A schema reached by its `context.trigger` gets
`trigger_score = 0.95 + 0.05·overlap`. It has already cleared the 0.5 overlap
threshold, and the ranker scores it as `max(base, trigger_score) · 1.8`, which
is a different scale from the channel blend. On the channel scale such a
schema often has little or no bm25/vector evidence, so a threshold tuned for
traces would cut useful procedures.

Such schemas are gated on their own scale:

```
trigger_gate_score = feedback_weighted_score(node, trigger_score / 0.95,
                                              superseded=…) · demotion
passes if trigger_gate_score ≥ SCHEMA_TRIGGER_GATE_MIN (0.5)
       or gate_score ≥ LM_RECALL_MIN_SCORE
```

A clean trigger hit reads about 1.0, so it passes regardless of
`LM_RECALL_MIN_SCORE`. The trigger match is the relevance evidence, and
trigger results are as useful as bm25 ones (a non-goal of this project
leaves them alone). Only negative evidence that at least halves the schema
takes it out: a hub or query demotion, negative usefulness, or a correction.
The scope and causal boosts do not enter, because a trigger match depends on
neither. Results that are not schemas, or have `trigger_score == 0`, are gated
on the channel scale only.

## Invariants

- Valve off: output is byte-identical to the plain cut
  (`test_off_is_identical_output`, `test_invalid_or_zero_threshold_is_off`).
- Rank 1 is never withheld or dropped.
- `delivered + residual` is a partition of `ranked` in drop form. In stub form,
  `delivered` is the old cut.
- Changing store retrieval weights does not change any gate score
  (`test_gate_score_does_not_read_the_store_weights`).
- The MCP response gains only a delivery class value, and only while
  `stub` is on.
