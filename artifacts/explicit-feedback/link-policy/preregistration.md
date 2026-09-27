# LM_IMPLICIT_LINK_POLICY replay: preregistration

This file was committed before any replay result was computed. `report.md` quotes it verbatim. Any later
deviation is recorded in `report.md` as a deviation. This file is not edited.

## Question

`apply_pending_recall_feedback` links the closing trace with a `related` edge
(`metadata.basis = implicit_recall_feedback`) to EVERY resolved delivered node (policy `all`, today).
The valve `LM_IMPLICIT_LINK_POLICY=credited` would link only the nodes that hold a
`recall_credit_ledger` row for that event. Which value do we recommend, per host?

## Stores (each measured separately, never merged)

- **sfx**: `~/.local/share/living-memory/global.sqlite3`, frozen with
  `living_memory.retrieval_harness.create_snapshot` (source `file:...?mode=ro`).
- **alt**: the same path on host `alt`, frozen on alt with the same function, then copied to `/tmp`.

## Cutoff (time split)

`T = 2026-09-20T00:00:00Z`. Both arms are built from the same snapshot, then:

1. **Both arms:** delete every `connections` row with `created_at >= T` (every edge type), every
   `query_anchor_edges` row with `created_at >= T`, and every `query_anchors` row with
   `created_at >= T`. This removes the eval events' own closing links and any other graph or anchor
   state learned from them, so neither arm sees the future.
2. **Arm `all`:** nothing else. This is the graph as the current policy built it up to T.
3. **Arm `credited`:** also delete every pre-T `implicit_recall_feedback` edge (source trace S →
   node N) that had no credit **at link time**:
   - Ledger era (edge `created_at` at or after the first ledger row): the edge is kept iff a
     `recall_credit_ledger` row exists for (event E, N), where E is an event closed by S
     (`recall_events.feedback_trace_id = S`, or the edge's `metadata.recall_event_id`), and that row has
     `credited_at <= edge.created_at + 60 s`. A lookup credit that arrives only after the close would not
     have existed when a `credited` valve decided, so it does not count.
   - Pre-ledger era: no ledger exists. The edge is kept iff the live grader
     `living_memory.grounding.ground_results(S.content, {results of E}, min_containment=RECALL_CREDIT_MIN_CONTAINMENT)`
     marks N grounded for some such E. This is exactly the verdict that would have claimed a
     `grounded` row. Lookup credit cannot be reconstructed here, so this part slightly
     under-links (a conservative bias against `credited`).

## Eval traffic (conservative A/B filter)

Replay set: recall events with `created_at >= T`, a non-empty recorded result list, and a surviving
transport session. A whole **transport session** is excluded (along with every event that has no
transport id and trips a rule) if any of its events or written nodes trips any rule:

- event `scope` or `requested_scope` is `project:target|project:repo|project:x|project:benchmark`;
- event `query` (case-insensitive) matches
  `ledger|billing|tree-context|fixture|kit acceptance|acceptance kit|evaluation-live|development-live`;
- a node whose `context.transport_session_id` is that session has a `context` containing `fixture`,
  `tree-context-ab`, or a `run` value matching `-live-|--repaired-|baseline-r\d|ab-`,
  or has a scope from the list above.

This over-excludes (for example, LM work that mentions the credit ledger). That is intended:
conservative. The sibling `effect-daily-metric` owns the robust filter. It is not used here.

Each event is replayed through the real retrieval path
(`MemoryRecallService.memory_recall(log_access=False, log_event=False)`, as `retrieval_harness.run_item`
does) with the recorded query, `requested_scope`, `depth` and `max_results`. `ambient_context` has
`transport_session_id` stripped, so per-session delivery state recorded after the event cannot touch the
ranking. Results created after the event are dropped before ranks are computed (same in both arms).

## Labels

A **credited node** of an eval event is a node that has a `recall_credit_ledger` row (basis `grounded`
or `lookup`) for that event. All such rows come from events after T, so they are disjoint from the
edges the arms differ on. **Labelled events** are eval events with at least one credited node that
exists in the snapshot.

## Metrics (per host, per arm)

- `mrr`, `hit@1`, `hit@3` of credited nodes over labelled events (reciprocal rank of the first
  credited node; 0 if absent).
- `graph hits`: returned results with `graph` among their methods, split into credited and
  uncredited. `uncredited_graph_hits_per_event` is taken over all replayed events. Also reported:
  `graph_only` (graph is the only channel).
- Latency p50/p95 of the replayed recalls; implicit and total edge counts per arm.
- Paired bootstrap 95% CI (2000 resamples, seed 7) for the credited − all deltas. Reported only;
  the rule does not use it.

## Decision rule (fixed before any result)

Per host, recommend `credited` iff ALL of the following hold, else `all`:

- **R0** at least 100 labelled events (otherwise the verdict is "insufficient", which means `all`);
- **R1** Δmrr ≥ −0.01 **and** Δhit@3 ≥ −0.01 (credited − all, absolute, labelled events);
- **R2** `uncredited_graph_hits_per_event` falls by at least 5% relative (credited vs all);
- **R3** latency p50 credited ≤ 1.10 × p50 all.

Overall recommendation for `LM_IMPLICIT_LINK_POLICY`: `credited` if both hosts pass, `all` if
both fail. If the hosts split, each host gets its own value and the code default stays `all`.
