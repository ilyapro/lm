# LM_IMPLICIT_LINK_POLICY: snapshot replay report

## Preregistration (verbatim, committed before any result)

> # LM_IMPLICIT_LINK_POLICY replay: preregistration
>
> This file was committed before any replay result was computed. `report.md` quotes it verbatim. Any later
> deviation is recorded in `report.md` as a deviation. This file is not edited.
>
> ## Question
>
> `apply_pending_recall_feedback` links the closing trace with a `related` edge
> (`metadata.basis = implicit_recall_feedback`) to EVERY resolved delivered node (policy `all`, today).
> The valve `LM_IMPLICIT_LINK_POLICY=credited` would link only the nodes that hold a
> `recall_credit_ledger` row for that event. Which value do we recommend, per host?
>
> ## Stores (each measured separately, never merged)
>
> - **sfx**: `~/.local/share/living-memory/global.sqlite3`, frozen with
>   `living_memory.retrieval_harness.create_snapshot` (source `file:...?mode=ro`).
> - **alt**: the same path on host `alt`, frozen on alt with the same function, then copied to `/tmp`.
>
> ## Cutoff (time split)
>
> `T = 2026-09-20T00:00:00Z`. Both arms are built from the same snapshot, then:
>
> 1. **Both arms:** delete every `connections` row with `created_at >= T` (every edge type), every
>    `query_anchor_edges` row with `created_at >= T`, and every `query_anchors` row with
>    `created_at >= T`. This removes the eval events' own closing links and any other graph or anchor
>    state learned from them, so neither arm sees the future.
> 2. **Arm `all`:** nothing else. This is the graph as the current policy built it up to T.
> 3. **Arm `credited`:** also delete every pre-T `implicit_recall_feedback` edge (source trace S →
>    node N) that had no credit **at link time**:
>    - Ledger era (edge `created_at` at or after the first ledger row): the edge is kept iff a
>      `recall_credit_ledger` row exists for (event E, N), where E is an event closed by S
>      (`recall_events.feedback_trace_id = S`, or the edge's `metadata.recall_event_id`), and that row has
>      `credited_at <= edge.created_at + 60 s`. A lookup credit that arrives only after the close would not
>      have existed when a `credited` valve decided, so it does not count.
>    - Pre-ledger era: no ledger exists. The edge is kept iff the live grader
>      `living_memory.grounding.ground_results(S.content, {results of E}, min_containment=RECALL_CREDIT_MIN_CONTAINMENT)`
>      marks N grounded for some such E. This is exactly the verdict that would have claimed a
>      `grounded` row. Lookup credit cannot be reconstructed here, so this part slightly
>      under-links (a conservative bias against `credited`).
>
> ## Eval traffic (conservative A/B filter)
>
> Replay set: recall events with `created_at >= T`, a non-empty recorded result list, and a surviving
> transport session. A whole **transport session** is excluded (along with every event that has no
> transport id and trips a rule) if any of its events or written nodes trips any rule:
>
> - event `scope` or `requested_scope` is `project:target|project:repo|project:x|project:benchmark`;
> - event `query` (case-insensitive) matches
>   `ledger|billing|tree-context|fixture|kit acceptance|acceptance kit|evaluation-live|development-live`;
> - a node whose `context.transport_session_id` is that session has a `context` containing `fixture`,
>   `tree-context-ab`, or a `run` value matching `-live-|--repaired-|baseline-r\d|ab-`,
>   or has a scope from the list above.
>
> This over-excludes (for example, LM work that mentions the credit ledger). That is intended:
> conservative. The sibling `effect-daily-metric` owns the robust filter. It is not used here.
>
> Each event is replayed through the real retrieval path
> (`MemoryRecallService.memory_recall(log_access=False, log_event=False)`, as `retrieval_harness.run_item`
> does) with the recorded query, `requested_scope`, `depth` and `max_results`. `ambient_context` has
> `transport_session_id` stripped, so per-session delivery state recorded after the event cannot touch the
> ranking. Results created after the event are dropped before ranks are computed (same in both arms).
>
> ## Labels
>
> A **credited node** of an eval event is a node that has a `recall_credit_ledger` row (basis `grounded`
> or `lookup`) for that event. All such rows come from events after T, so they are disjoint from the
> edges the arms differ on. **Labelled events** are eval events with at least one credited node that
> exists in the snapshot.
>
> ## Metrics (per host, per arm)
>
> - `mrr`, `hit@1`, `hit@3` of credited nodes over labelled events (reciprocal rank of the first
>   credited node; 0 if absent).
> - `graph hits`: returned results with `graph` among their methods, split into credited and
>   uncredited. `uncredited_graph_hits_per_event` is taken over all replayed events. Also reported:
>   `graph_only` (graph is the only channel).
> - Latency p50/p95 of the replayed recalls; implicit and total edge counts per arm.
> - Paired bootstrap 95% CI (2000 resamples, seed 7) for the credited − all deltas. Reported only;
>   the rule does not use it.
>
> ## Decision rule (fixed before any result)
>
> Per host, recommend `credited` iff ALL of the following hold, else `all`:
>
> - **R0** at least 100 labelled events (otherwise the verdict is "insufficient", which means `all`);
> - **R1** Δmrr ≥ −0.01 **and** Δhit@3 ≥ −0.01 (credited − all, absolute, labelled events);
> - **R2** `uncredited_graph_hits_per_event` falls by at least 5% relative (credited vs all);
> - **R3** latency p50 credited ≤ 1.10 × p50 all.
>
> Overall recommendation for `LM_IMPLICIT_LINK_POLICY`: `credited` if both hosts pass, `all` if
> both fail. If the hosts split, each host gets its own value and the code default stays `all`.

## Results per host (measured separately, never merged)

| metric | sfx all | sfx credited | alt all | alt credited |
|---|---|---|---|---|
| events | 2335 | 2335 | 1039 | 1039 |
| labelled_events | 508 | 508 | 347 | 347 |
| mrr | 0.3592 | 0.3615 | 0.4302 | 0.4317 |
| hit@1 | 0.2008 | 0.2028 | 0.3055 | 0.3084 |
| hit@3 | 0.5079 | 0.5098 | 0.5331 | 0.5389 |
| credited_node_recall | 0.4896 | 0.4896 | 0.5148 | 0.5115 |
| graph_hits_credited | 155 | 144 | 20 | 19 |
| graph_hits_uncredited | 3955 | 3075 | 435 | 275 |
| graph_only_uncredited | 212 | 178 | 0 | 0 |
| uncredited_graph_hits_per_event | 1.6938 | 1.3169 | 0.4187 | 0.2647 |
| graph_share_of_credited_hits | 0.2987 | 0.2775 | 0.0637 | 0.0609 |
| graph_share_of_uncredited_hits | 0.3616 | 0.2823 | 0.1476 | 0.0933 |
| latency_p50_ms | 179.7681 | 163.7028 | 193.6440 | 178.7654 |
| latency_p95_ms | 439.1204 | 378.2591 | 512.5216 | 498.8215 |
| edges: implicit | 105978 | 16500 | 45296 | 4417 |
| edges: related | 229246 | 139768 | 104426 | 63547 |
| edges: connections | 234118 | 144640 | 106449 | 65570 |

### Counterfactual edge plan

| field | sfx | alt |
|---|---|---|
| cutoff | 2026-09-20T00:00:00Z | 2026-09-20T00:00:00Z |
| dropped_ledger_era | 24882 | 9999 |
| dropped_pre_ledger | 64596 | 30880 |
| dropped_total | 89478 | 40879 |
| implicit_pre_cutoff | 105978 | 45296 |
| kept_ledger_era | 4691 | 1328 |
| kept_pre_ledger_grounded | 11809 | 3089 |
| ledger_credit_only_after_link | 26 | 25 |
| ledger_start | 2026-09-07T07:06:34Z | 2026-09-07T07:07:46Z |
| traffic: post_cutoff | 4469 | 1176 |
| traffic: excluded_ab | 2134 | 137 |
| traffic: empty_results | 0 | 0 |
| traffic: kept | 2335 | 1039 |
| traffic: labelled | 508 | 347 |

### Paired bootstrap (credited − all, 95% CI; reported, not in the rule)

| delta | sfx | alt |
|---|---|---|
| mrr | 0.0024 [-0.0038, 0.0082] | 0.0014 [-0.0010, 0.0050] |
| hit@3 | 0.0020 [-0.0098, 0.0138] | 0.0058 [0.0000, 0.0144] |
| uncredited_graph_hits_per_event | -0.3769 [-0.4116, -0.3439] | -0.1540 [-0.1781, -0.1299] |

### Decision rule applied

| check | sfx | alt |
|---|---|---|
| R0_labelled>=100 | yes | yes |
| R1_quality | yes | yes |
| R2_uncredited_graph_drop>=5% | yes | yes |
| R3_latency | yes | yes |
| delta_mrr | 0.0024 | 0.0014 |
| delta_hit@3 | 0.0020 | 0.0058 |
| uncredited_graph_relative_drop | 0.2225 | 0.3678 |
| latency_ratio | 0.9106 | 0.9232 |
| verdict → recommendation | pass → `credited` | pass → `credited` |

## Recommendation

- Per host: sfx: `LM_IMPLICIT_LINK_POLICY=credited`, alt: `LM_IMPLICIT_LINK_POLICY=credited`
- Overall: **credited**; suggested valve default for the owner of `LM_IMPLICIT_LINK_POLICY` (explicit-marks-core / operator): `credited`. No code default is changed by this replay.

## Interpretation and caveats (written after the result)

- **Quality does not drop; it rises slightly on both hosts.** Credited-node MRR goes up by +0.0024 on
  sfx and +0.0014 on alt, and hit@3 by +0.0020 and +0.0058. Every bootstrap CI for these deltas
  includes or touches 0, so this is "no loss" rather than "a gain". On alt, credited-node recall
  moves 0.5148 → 0.5115 (−0.3 pp, about 1 node in 300), which is within noise.
- **The graph channel stops carrying unrelated nodes.** Uncredited graph hits per replayed event fall
  22% on sfx (1.69 → 1.32) and 37% on alt (0.42 → 0.26), with CIs far from 0. The graph share of
  uncredited hits falls from 0.36 to 0.28 on sfx and from 0.15 to 0.09 on alt. Credited graph hits
  fall only 155 → 144 (sfx) and 20 → 19 (alt).
- **Edges and latency.** Pre-cutoff implicit edges drop 84% on sfx (105,978 → 16,500) and 90% on alt
  (45,296 → 4,417); all connections drop 38%. Replay p50 latency falls about 9% on sfx and 8% on
  alt. Arms were interleaved per event, so machine load hit both arms equally.
- **Link-time vs eventual credit barely differ.** Only 26 (sfx) and 25 (alt) ledger-era edges get
  a ledger row after the 60 s link window. A `credited` valve that decides at close time loses almost
  nothing to lookups that arrive later.
- **Bias against `credited`.** Pre-ledger edges (before 2026-09-07) were kept only if the live
  grader marks them grounded; lookup credit cannot be reconstructed there. So the `credited` arm
  under-links, and it still passes.
- **Leakage shared by both arms.** Node `usefulness_score`/`access_count` and pre-cutoff anchors
  reinforced after T carry post-T learning into both arms equally. Edges and anchors created after T
  were removed from both arms.
- **Traffic filter.** It is conservative: sfx excludes 2,134 of 4,469 post-cutoff events (48%) and
  alt excludes 137 of 1,176. That includes LM work that mentions the ledger. The robust filter is
  the effect-daily-metric sibling's job, and this result does not depend on it.
- **Scope of the claim.** The replay measures ranking of later-credited nodes and graph noise.
  `credited` also means the closing trace's provenance edges to unused nodes are not created. The
  `recalled_nodes`/`source_traces` provenance on the trace is unaffected (it is separate from edges).
  Once explicit `used` marks exist (P2), counting them as credit for this valve is consistent with
  this result, but it was not measured here.
