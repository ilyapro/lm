# Query-anchor match entry — calibration

**Chosen: `ANCHOR_MATCH_COSINE_THRESHOLD = 0.60`, `ANCHOR_MATCH_LIMIT = 5`**, written into `src/living_memory/query_anchors.py`. These are the values the scored temporal-holdout run is executed with, **unchanged** — nothing downstream may retune them against the scored goldset.

Calibration of the query-anchor match entry (ANCHOR_MATCH_COSINE_THRESHOLD, ANCHOR_MATCH_LIMIT) on a calibration set built strictly inside the scored run's anchor training window.

---

## 1. The split, and why it is not the answer key

```
|------- calibration anchors -------|-- calibration queries --|--- SCORED ---|
<                    2026-06-10                    2026-07-15               >
```

Anchors: `events < 2026-06-10T00:00:00Z` — **1607 live anchors / 3321 edges**, built by `backfill_query_anchors.py --until 2026-06-10T00:00:00Z` into a copy of the anchor-free base snapshot `1584382ca3afc933…`. The live database was never opened.

Calibration queries are the events that BUILD the scored run's anchor corpus, so they are disjoint from the scored, post-cutoff items by construction. The two untimestamped strata are newly authored and their disjointness is asserted by the purity filter above.

**The scored sets were opened once, to exclude.** Opened once by load_exclusions to derive fingerprints, source event ids, normalised-query hashes and query embeddings, used only to DROP overlapping calibration items. No scored query text, relevance label or score was read, stored or reported.

| scored items scanned | calibration candidates | dropped | kept |
|---|---|---|---|
| 577 | 258 | **24** | 234 |

Drop reasons: `near_duplicate_query` 1, `normalized_query` 3, `source_event_id` 20. That 24 of 258 candidates collided is the point of running the filter: the shipped goldset's `content_grounded` stratum was itself built at cutoff 2026-06-10, so it overlaps this window and would have leaked.

After the filter, the closest any calibration query sits to any scored query is **0.92579** cosine (p50 0.63623), under the 0.95 near-duplicate cut.

**Re-verified at publication**, against the file that actually ships: 234 calibration items checked against 577 scored items on recall fingerprint, normalised query and source event id — **0 collisions**. `publish` refuses to write the artifact if that number is not zero, so the leak claim is enforced rather than reported.

Temporal assertion: **156** timestamped items, earliest `2026-06-10T02:58:32Z`, latest `2026-07-13T16:13:08Z`, **0 outside the window**. The other 78 carry no timestamp by construction and are bound by the fingerprint/near-duplicate test instead.

## 2. The calibration set

**234 items** — `content_grounded` 156, `cross_lingual` 34, `role_query` 44 — sha256 `ccef2013ffb14f25…`.

`content_grounded` is the shipped `build_content_grounded` label (IDF containment against the consuming trace) over 1513 eligible in-window events, capped at 180 and sampled with `random.Random(20260818).sample over query_id-sorted in-window items`.

`cross_lingual` and `role_query` are **newly authored** — `artifacts/harness/seed-queries.json` holds exactly the 38 + 36 items already inside the scored goldset, so there was no surplus to borrow. 35 jargon/paraphrase pairs were authored under the recorded `cross_lingual_rule` and re-verified mechanically on this snapshot (paraphrase top-1 vector ≥ 0.40, node predominantly Latin, jargon strictly lower on that node and not already ranking it first): **34 kept**. 44 role/procedural queries were authored against active `level:schema` nodes carrying `context.trigger`, re-validated by the shipped `build_role_query`.

## 3. Does a floor calibrated here transfer?

The one distribution that decides it is the nearest in-scope anchor cosine per query — the quantity the floor is compared against. Calibration (1,607 anchors) against scored (3,071 anchors, `result.md` §5):

| stratum | calibration p50 / p90 / max | scored p50 / p90 / max |
|---|---|---|
| `content_grounded` | 0.579 / 0.688 / 0.957 | 0.558 / 0.710 / 1.000 |
| `cross_lingual` | 0.508 / 0.632 / 0.838 | 0.455 / 0.589 / 0.663 |
| `role_query` | 0.451 / 0.568 / 0.593 | 0.529 / 0.697 / 0.912 |

The calibration queries face a 1,607-anchor corpus and the scored queries a 3,071-anchor one, so the two nearest-anchor distributions are the thing that decides whether a floor calibrated here transfers. They sit in the same band, and they do not shift in one direction: cross_lingual is higher here (p50 0.508 vs 0.455), role_query lower (0.451 vs 0.529), content_grounded within 0.02.

## 4. The sweep

Coverage, precision, quality and cost at every swept floor, at `ANCHOR_MATCH_LIMIT = 5`. `anchor-prec` is the share of *matched anchors* holding an edge to a relevant node; `item-prec` is the share of *matched queries* with at least one such anchor.

| floor | coverage | cross_lingual | role_query | anchor-prec | item-prec | hit@5 | MRR | rank changes | measured p50 Δ (traffic-weighted) |
|---|---|---|---|---|---|---|---|---|---|
| -1.00 | 234/234 (100.0%) | 34/34 | 44/44 | 0.034188 | 0.098291 | 0.4701 | 0.2413 | 11 (+0/−1) | **+25.93 ms** ⚠ over |
| 0.40 | 217/234 (92.7%) | 29/34 | 33/44 | 0.038191 | 0.096774 | 0.4701 | 0.2413 | 9 (+0/−1) | — |
| 0.45 | 192/234 (82.1%) | 23/34 | 22/44 | 0.04248 | 0.104167 | 0.4701 | 0.2413 | 8 (+0/−1) | — |
| 0.50 | 168/234 (71.8%) | 19/34 | 15/44 | 0.055306 | 0.119048 | 0.4701 | 0.2413 | 7 (+0/−1) | — |
| 0.55 | 125/234 (53.4%) | 13/34 | 6/44 | 0.084541 | 0.152 | 0.4701 | 0.2414 | 2 (+0/−0) | **+4.17 … +8.31 ms** ⚠ over |
| 0.60 **←** | 69/234 (29.5%) | 7/34 | 0/44 | 0.132597 | 0.217391 | 0.4701 | 0.2414 | 0 (+0/−0) | **+2.07 … +2.51 ms** |
| 0.65 | 29/234 (12.4%) | 3/34 | 0/44 | 0.21875 | 0.275862 | 0.4701 | 0.2414 | 0 (+0/−0) | **+0.12 … +0.37 ms** |
| 0.70 | 12/234 (5.1%) | 1/34 | 0/44 | 0.4 | 0.416667 | 0.4701 | 0.2414 | 0 (+0/−0) | — |
| 0.75 | 5/234 (2.1%) | 1/34 | 0/44 | 0.714286 | 0.8 | 0.4701 | 0.2414 | 0 (+0/−0) | — |
| 0.80 | 3/234 (1.3%) | 1/34 | 0/44 | 0.75 | 0.666667 | 0.4701 | 0.2414 | 0 (+0/−0) | **-0.37 … +1.13 ms** |
| 0.85 | 1/234 (0.4%) | 0/34 | 0/44 | 1.0 | 1.0 | 0.4701 | 0.2414 | 0 (+0/−0) | — |
| 0.90 | 1/234 (0.4%) | 0/34 | 0/44 | 1.0 | 1.0 | 0.4701 | 0.2414 | 0 (+0/−0) | — |

Anchors-off baseline on the same set: hit@1 0.0812, hit@5 **0.4701**, hit@10 0.6368, MRR **0.2414**.

### The limit

`ANCHOR_MATCH_LIMIT` does not move coverage at all — coverage asks whether *any* anchor clears the floor, which no limit changes. It moves precision and risk:

| floor | L=3 item-prec | L=5 item-prec | L=10 item-prec | L=10 regressions |
|---|---|---|---|---|
| 0.55 | 0.128 | 0.152 | 0.16 | 1 |
| 0.60 | 0.188406 | 0.217391 | 0.217391 | 1 |
| 0.65 | 0.206897 | 0.275862 | 0.275862 | 0 |

`L = 5` dominates `L = 3` on item precision at every floor, and over the in-budget part of the grid (floors ≥ 0.55) it regressed **0** items against `L = 10`'s **2**. It stays at **5**. Below 0.55 every limit regresses items, which is a second reason not to go there.

## 5. Cost

`artifacts/anchors/latency.json` measured the matched stratum on **two** queries. A floor is chosen partly on what it costs, so it was re-measured here with the same paired design (arms alternated every iteration, per-query medians differenced, 9 iterations after 2 warmups, reads that log neither access nor event) on up to 25 matched and 25 unmatched calibration queries per floor.

| floor | matched share | matched paired p50 | unmatched paired p50 | traffic-weighted p50 | +5 ms budget |
|---|---|---|---|---|---|
| -1.00 | 100.0% | +25.93 ms | — (none) | **+25.93 ms** | **over** |
| 0.55 | 53.4% | +13.00 ms | -0.41 ms | **+4.17 ms** | inside |
| 0.60 | 29.5% | +22.73 ms | +0.60 ms | **+2.51 ms** | inside |
| 0.65 | 12.4% | +6.50 ms | -0.24 ms | **+0.37 ms** | inside |
| 0.80 | 1.3% | +4.20 ms | +1.13 ms | **+1.13 ms** | inside |

The cost is not the anchor scan; it is the **second graph walk** a matched anchor opens. It therefore scales with the walk it duplicates: on small-scope queries (~90 ms baseline) a match costs +5 to +10 ms, and on the deep large-scope queries (200–1000 ms baseline) it costs +100 to +500 ms. That is why the budget is crossed by *match rate*, not by threshold as such.

### The cost measurement is noisy, so it was run twice

Two independent paired runs at different sample sizes. The matched paired delta is heavy-tailed (p25 ~+3 ms, p75 ~+130 ms), so the traffic-weighted p50 is itself noisy; both replicates are published so the spread is visible rather than hidden behind one run.

| floor | replicate 1 (n=20) | replicate 2 (n=25) | both inside +5 ms? |
|---|---|---|---|
| -1.00 | — | +25.93 ms | **no** — straddles or over |
| 0.55 | +8.31 ms | +4.17 ms | **no** — straddles or over |
| 0.60 **←** | +2.07 ms | +2.51 ms | **yes** |
| 0.65 | +0.12 ms | +0.37 ms | **yes** |
| 0.80 | -0.37 ms | +1.13 ms | **yes** |

The two runs disagree at 0.55 by nearly 2× and land on opposite sides of the budget. That is not a defect in either run — it is what a heavy-tailed delta does to a p50 at n≈20 — and it is the reason 0.55 is rejected as *uncertifiable* rather than as *measured over*. 0.60 came back inside the budget on both.

## 6. Why quality is flat, measured rather than asserted

Coverage moves by two orders of magnitude across the grid and the number of calibration items whose ranking changes stays at 0. The funnel separates the two possible causes — the edges point nowhere useful, or the ranker will not surface what they point at:

| floor | matched | …with an edge to a relevant node | …that the baseline missed at 5 | …rescued into top 5 |
|---|---|---|---|---|
| 0.80 | 3 | 2 | 0 | **0** |
| 0.60 | 69 | 15 | 3 | **0** |
| 0.55 | 125 | 19 | 5 | **0** |
| -1.00 | 234 | 23 | 8 | **0** |

**Both leaks are real, and the first is the larger.** Read the bottom row, which is the mechanism at its theoretical maximum: with **no floor at all** every one of the 234 queries matches an anchor, and still only **23** of them match an anchor whose edges lead to anything the item marks relevant. Of those, most were already answered at rank ≤ 5 without anchors, leaving **8** items the entry could possibly have rescued — and it rescued **0**. `role_query` is the sharpest case: all 44 of its queries match at no floor and **not one** matched anchor carries an edge to a relevant node.

Where the seeded node did not surface, the funnel records why: it did not enter the returned list at all (`seeded_relevant_node_rank: null`), at an activation of 0.14–0.21 — a cosine times an edge weight — against a learned graph channel weight that cannot lift it past a bm25/vector-dominated ranking.

So the floor is **not** the last thing standing between this mechanism and its targets. Lowering it is still right — it is measurably mis-set, and it is the only thing this node owns — but the next lever is the edge yield per anchor and the activation an anchor seed carries into the ranker, neither of which is in scope here.

## 7. The choice

**0.60 at limit 5.** The reasoning, in the order the evidence forces:

1. **0.80 is measurably wrong.** It admits 3 of 234 calibration queries (1.3%), which reproduces the 13 of 343 (3.8%) the scored run measured. A floor that never fires cannot be evaluated, and was never calibrated — it was picked to sit under the 0.95 dedup constant.
2. **Quality cannot choose.** All 36 cells — every floor from **no floor at all** (-1.00, where all 234 queries match) up to 0.90, at limits 3, 5 and 10 — return the anchors-off hit@1/hit@5/hit@10 exactly, with **0 items improved anywhere in the grid**. §6 says why. So the choice is coverage against cost, not quality against cost.
3. **A wrong match is cheap in quality and expensive in time.** Since the graph-floor monotonicity fix an anchor activation cannot lower a candidate's score, so the cost of admitting a bad anchor is the walk it opens — which is exactly what the +5 ms p50 budget already prices. That makes the budget, not precision, the binding constraint.
4. **0.60 is the widest floor certified inside the budget.** Traffic-weighted paired p50, two independent replicates: 0.55 gave **+8.31 ms and +4.17 ms** — landing on opposite sides of the hard +5 ms budget, so it cannot be certified at all — while 0.60 gave **+2.07 ms and +2.51 ms**, inside on both. No floor at all costs **+25.93 ms**. Width is the whole point, so take the widest floor the budget actually certifies.
5. **It buys real width on the class in question.** `cross_lingual` coverage goes from 1 of 34 at 0.80 to **7 of 34** at 0.60. `role_query` stays at 0 of 44 here — its calibration items sit at nearest-anchor p50 0.451 / max 0.593 against the smaller calibration corpus — but the scored `role_query` population sits at p50 0.529 / p90 0.697, so 0.60 reaches roughly its top third rather than the 1 of 36 that 0.80 reached.

Rejected: **0.55**, which buys the most coverage (53.4%, and the only `role_query` matches on this set) but whose cost straddles the budget across replicates; **0.65**, which costs almost nothing (+0.12 / +0.37 ms) but gives up more than half of 0.60's coverage (12.4% vs 29.5%) to buy precision that — per §6 — no longer converts into quality; **0.70+**, which is 0.80's failure mode with a smaller number on it; and **no floor at all**, which is the cleanest disproof of the premise: it fires on 100% of queries, improves nothing, and costs +25.93 ms.

## 8. What this predicts, and how to falsify it

**At 0.6 the entry will fire on far more of the scored holdout than the 13 of 343 (3.8%) it fired on at 0.80 -- this calibration measures 29.5% -- and will still not move hit@5 on either target stratum.**

Measured here over 36 cells: coverage rises 23x from the shipped floor, and hit@1/hit@5/hit@10 come back identical to the anchors-off arm in EVERY cell -- including at no floor at all, where all 234 queries match. Zero items improved anywhere in the grid. The funnel names the binding leak, and it is not the floor: at no floor only 23 of 234 matched anchors carry an edge to a relevant node, and of the 8 items whose relevant node WAS seeded and WAS missing from the anchors-off top 5, 0 entered it.

*Falsifier.* A scored run at 0.6 that lifts cross_lingual hit@5 or role_query hit@1 refutes the second half; one that leaves coverage near 3.8% refutes the first. Either outcome means the calibration set's anchor corpus is not representative of the scored one, which is the assumption the external-validity section exposes.

---

Method note: the sweep reuses an item's anchors-off ranking for every cell in which that item matches nothing. That reuse is checked rather than assumed — 12 items were re-run for real with anchors ON at an unreachable floor and 12 came back identical (`inert_path_check.ok = True`).

Full record: `artifacts/anchors/calibration.json`.
