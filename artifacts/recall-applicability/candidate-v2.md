# Applicability candidate v2

## Mechanism

The first repair used a query-word-count admission rule but retained a near-constant trigger rank floor and 1.8x boost. Its archived holdout lost two instruction clauses and improved only one topic group. That is a counterexample to treating a trigger label as either a complete relevance score or a reason to exclude a short instruction from a compound question.

Ordinary recall now collects a concept or schema when half of its trigger words match the query, including a short valid trigger inside a longer question. The trigger keeps that node available and can admit it across scopes. In ranking, the trigger contributes in proportion to the node content's existing vector similarity; the existing graph weight provides a small floor if content evidence is sparse. The unconditional 1.8x boost is removed. The gate's channel score reconstructs this same contribution, while its separate feedback-sensitive trigger scale can still admit an instruction with sparse content evidence. Feedback demotions and supersedes remain in the existing paths. Full lookup and delivery are unchanged. The disabled `LM_RECALL_SCHEMA_TRIGGER=name` channel stays disabled.

This separates discovery from rank priority. A historical carrier may be collected by a generic label, but that label alone cannot give it a near-perfect rank score. A short instruction remains a candidate even when other words in a compound query dilute its trigger overlap with the whole query. No project-specific exceptions, extra LLM call, provider change, scope narrowing, or memory deletion were added.

## Development observations

The four original development cases were extracted by their `split` value from the original local gold set. They alone were measured on the original frozen SQLite snapshot through the paired runner, with separate writable copies, fresh worker processes, alternating arm order, the same recall parameters, and full lookup of returned references. The case input and raw paired result are private under `/home/sfx/p/ae/artifacts/recall-applicability/candidate-development/`; no holdout cases or measurements were inspected.

Both arms reached 3 of 6 required clauses; the candidate lost zero baseline-reachable clauses. Irrelevant top-four slots were 14 in both arms. The compound schema-channel case retained both clauses, moved its source from rank 4 to rank 1, and used four rather than five reading calls. The reported Russian-auditory case remained unreached in both arms. Across development cases, total read calls were 19 to 18, response bytes 93,711 to 71,433, and summed warm latency 16,838.1 to 16,922.1 ms. These are development observations, not field success.

Synthetic public `memory_recall` plus natural `content_ref` lookup checks use renamed two-word instruction triggers in concrete and compound queries, five irrelevant historical carriers, four competing applicable answers, a cross-project fact, and a long instruction carrier whose full clause requires lookup. The tests assert source reachability, absence of those carriers from the top four, and exact full-content lookup. Existing focused tests cover used/irrelevant feedback, correction dominance and supersedes, name-channel isolation, gate feedback, and the accepted one-call large-carrier reading gain. Earlier assertions that a generic carrier never entered the candidate pool were replaced: collection is deliberately broad now, and the public delivery result is the safety property.

## Checks and production delta

The contract's focused pytest command passed: 133 tests across applicability, score gate, cross-scope gate, name channel, delivery chain, carrier omission, feedback loop, explicit feedback, transport closure, and feedback amplification. The source tree was tested after the final source comment update. The parent runs the declared `npm run check` suite after integration.

Against accepted baseline `b9769d84e3188ee1e646627ebe1b4d454f69d6f9`, the two production files change by retrieval.py +18/-10 lines and score_gate.py +10/-7 lines, including explanatory comments. Ranking and gate continue to share `_blend_candidate_score`. The receipt JSON pins the full candidate source commit, source tree, and both file hashes.

Rejected alternatives: another query-token cutoff would repeat the short-instruction loss; a fixed trigger boost would preserve historical-carrier priority; suppressing all trigger-only candidates would discard applicable instructions; enabling the name valve would alter an intentionally disabled channel; hardcoded project or term exclusions would not generalize.
