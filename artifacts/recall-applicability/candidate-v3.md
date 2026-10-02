# Candidate v3: source-backed trigger applicability and procedure retention

This receipt describes the production source frozen in `candidate-v3.json`:
candidate commit `4d76ba10bf262b45e8c6bfc5715e128353755cec`, `src` tree
`1acf7e843243a19cc93e88e3afd1d128fd2d20e0`. Baseline is the accepted
`b9769d84e3188ee1e646627ebe1b4d454f69d6f9`. The v2 candidate receipt,
paired report, and their source hashes remain historical evidence for their
own source tree; none certifies this candidate. The parent still owes a fresh
declared-suite gate and one-shot successor holdout measurement.

## Mechanism

Trigger collection still uses the saved trigger's overlap with the query. It
also records the fraction of the *query's* meaningful tokens covered by that
trigger. The common rank blend now gives a source-backed procedural carrier or
binding schema a lexical trigger contribution proportional to the square of
that fraction. It takes the maximum with content/graph evidence, so one claim
is not counted twice. A concept with only a matching label and no procedural
source provenance receives no new rank claim. Scope boosts, feedback,
query-relative irrelevance, corrections and supersedes still act on the
resulting score. The gate reconstructs that same blend. Its separate trigger
availability scale uses query coverage for nonbinding carriers; binding
schemas retain the scale for short instructions inside compound questions.
The name valve, default scope, embedding backend and providers are unchanged.

This brings the two intact legacy-schema-turned carriers back within their
original public recall limits, where a natural `content_ref` lookup yields
whole current group evidence. It does not mark the case history as binding
steps. A complete saved trigger helps because it covers the question, not
because the code recognizes any fixture name or project word.

## Synthetic checks

The hash backend passes the two positive read chains at their unchanged
queries, fixture states and limits (`max_results=5` and `10`); the installed
encoder also passes both. The existing positive title-dedup test remains
green. Its group-identity mutant now asserts the actual dedup effect: nine
distinct history carriers consume nine of ten slots, while the directly
applicable recipe stays reachable. Its old recipe-suppression assertion became
false when the v2 rank repair made the direct recipe stronger. The legacy ID
and trigger corruption mutants still fail their positive oracles. Additional
contrasts cover two trigger labels and phrasings of a compound question with unrelated
same-trigger history, a source-backed same-trigger crowd with a short direct
recipe, a short binding instruction inside a compound question, correction
dominance, cross-project access, feedback, and full-byte lookup.

The large-group delivery checks remain executable. A relevant buried fact in
a 24-record carrier arrives through one bounded recall with less than one
quarter of the uniform full-content response characters. A second large
carrier puts a needed fact outside the snippet: the natural lookup returns
the exact full content, and recall plus lookup characters are at most 1.2
times the uniform full-content chain. These assertions count every necessary
response, including the lookup. They preserve the accepted reading gain
without treating an unreachable field case as evidence.

## Frozen development comparison

Only the **four original development cases**, extracted by their `split`
field from the original input, ran through the unchanged paired runner. The
fourteen original holdout entries were not written to the development case
file. Both arms used the original frozen snapshot, alternating order, fresh
copy/process/session, identical query parameters and the installed encoder.
Private input and paired output stay under
`/home/sfx/p/ae/artifacts/recall-applicability/candidate-v3-development/`;
the paired output SHA-256 is
`39195f15500f25da5ef104e443576949cd1e46bd6ff0c9e6c7cc893d1b73ce09`.
No question, fact, node ID or response text is copied here.

| Case | Baseline → candidate reached clauses | Necessary calls | Necessary bytes | All response bytes | Irrelevant top four | Warm latency ms |
| --- | --- | --- | --- | --- | --- | --- |
| d01 | 0 → 0 | no sufficient chain | n/a | 16,213 → 14,702 | 4 → 4 | 1,949 → 1,734 |
| d02 | 2 → 2 | 5 → 1 | 31,750 → 6,284 | 31,750 → 14,853 | 3 → 3 | 1,950 → 1,932 |
| d03 | 1 → 1 | 1 → 1 | 7,366 → 7,366 | 18,612 → 18,612 | 3 → 3 | 1,937 → 1,952 |
| d04 | 0 → 0 | no sufficient chain | n/a | 27,132 → 23,262 | 4 → 4 | 1,747 → 1,841 |

All **three baseline-reachable development clauses** remain reachable; there
are zero clause losses. d01 and d04 were insufficient in both arms. The
development top-four relevance count did not improve, so it cannot establish
P1's independent holdout threshold. The runner's source delta against the
baseline is retrieval.py 2,809 → 2,844 lines (+45/−10 changed lines) and
score_gate.py 238 → 248 (+20/−10). This is a localized ranking/gate change,
not an added provider, model, tier or mandatory LLM call.

## Rejected alternatives and assertion changes

- Restoring the old unconditional trigger floor or 1.8 multiplier would put
  generic histories back in top slots. A complete-trigger-only rescue was
  insufficient as an evidence rule: it would award the same priority to an
  arbitrary concept with a matching label. The source-backed condition and
  query coverage make the applicability claim reviewable.
- A group-identity dedup key recreates ten title-equivalent history slots.
  The mutant now proves that slot-count defect without requiring a useful
  direct recipe to disappear. The positive title-dedup oracle is unchanged.
- The compound applicability test previously expected exactly two relevant
  results with `max_results=4` and no score gate, although trigger-collected
  candidates were rankable in the other two slots. It now supplies four
  independently relevant facts and asserts all four fill the original limit.
  The new ungrounded-history contrast expects such nodes to stay unranked:
  collection alone is not score evidence.
- Token-count cutoffs, case-specific words, name-valve activation, deleting
  history, narrowing scope and an extra LLM call were not used.

## Targeted commands

```bash
timeout 600 npm test -- -q tests/test_procedure_case_separation.py tests/test_recall_applicability.py tests/test_recall_score_gate.py tests/test_retrieval_cross_scope_gate.py tests/test_schema_trigger_by_name.py tests/test_recall_delivery_chain.py tests/test_recall_carrier_omission.py tests/test_recall_feedback_loop.py tests/test_explicit_feedback.py tests/test_transport_feedback_closure_e2e.py tests/test_retrieval_feedback_amplification.py
LIVING_MEMORY_EMBEDDING_BACKEND=auto PYTHONPATH=.cache/python-deps timeout 240 python3 -m pytest -q tests/test_procedure_case_separation.py::test_taught_recipe_arrives_whole_through_its_legacy_schema_turned_carrier tests/test_procedure_case_separation.py::test_group_node_keeps_the_trigger_it_is_found_by tests/test_procedure_case_separation.py::test_crowding_oracle_detects_the_group_identity_mutant
timeout 1200 python3 scripts/recall_applicability_eval.py --snapshot /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3 --cases /home/sfx/p/ae/artifacts/recall-applicability/candidate-v3-development/cases.json --out /home/sfx/p/ae/artifacts/recall-applicability/candidate-v3-development/paired.json
```

The focused suite and installed-encoder selectors passed. The parent will
run the whole declared suite before consuming the independent v3 packet;
targeted success alone is neither P1 nor the parent's P4 completion.
