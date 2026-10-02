# Applicability recall: sealed local corpus and baseline receipt

Frozen on 2026-10-02 (UTC) before any candidate result was inspected. The
private SQLite backup, questions, required facts, source IDs, and rank-by-case
baseline remain on sfx under
`/home/sfx/p/ae/artifacts/recall-applicability/`. They are not repository
artifacts and must not be copied to alt or a remote repository.

| Local artifact | SHA-256 |
| --- | --- |
| `sfx-frozen.sqlite3` | `3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef` |
| `goldset.json` | `c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c` |
| `baseline-outcomes.json` | `35dccb9896e3c99c63e1d53c38fbb812667a784a237ca87a0ef1039e236a0318` |

The snapshot was made by SQLite's backup API from a read-only connection to
the local sfx store at accepted code `b9769d84e3188ee1e646627ebe1b4d454f69d6f9`.
It contains 28,609 nodes, is 1,193,484,288 bytes, and passed
`PRAGMA quick_check`. The private directory is mode 700; the three sealed
inputs/results are mode 400. The corpus validator in
`scripts/recall_applicability_corpus.py` checks source clauses against the
snapshot, disjointness, recall parameters, and required category coverage.
Both hashes are input identity: a changed byte requires a new registration.

## Population and oracle

Four development questions include the reported P5_RUSSIAN failure and three
other topic groups. The fourteen holdout questions have distinct queries and
answer groups from development and from one another. Holdout has two cases
each for concrete facts, compound facts, applicable instructions, corrections,
cross-project transfer, absent knowledge, and large evidence groups. The
absent questions contain nonce compounds absent from the frozen node corpus.
Cases were authored from source claims in the frozen snapshot before running
the accepted baseline. Every positive case pins one or more literal claim
clauses and acceptable source IDs; IDs are oracle-only and never added to a
retrieval query. An absent case has no answer source. A source ID by itself
does not prove the answer: all required clauses must be available in delivered
content or through a necessary full lookup. For a correction, score the new
claim and its repudiation of the old one together. Do not count an obsolete
claim quoted as false as a correct current answer.

All cases request `depth=1`, `max_results=4`; scope is either omitted or the
case's fixed explicit scope. Broad recall remains broad. Do not enable
`LM_RECALL_SCHEMA_TRIGGER=name`, alter providers/models/tiers, add project or
word exceptions, or shorten a failed query. Do not tune on the sealed holdout
or replace failures after candidate measurement.

## Pre-candidate baseline record

The baseline rank capture finished at `2026-10-02T00:43:10Z` using the
accepted checkout, a writable copy of the same snapshot, fresh recall
service per case, and disabled access/event logging. It records every ranked
result ID and the rank of any oracle source in the private file. At four
slots, oracle sources appeared for 2 of 4 development cases and 8 of 14
holdout cases (8 of 12 positive holdout cases). These are source-rank
observations, not delivery sufficiency or a candidate verdict. The P5_RUSSIAN
development source did not occupy a top-four slot in this frozen replay.

## Paired verdict rule

Use the paired runner on fresh separate copies of these exact snapshot bytes,
with identical environment and recall arguments, fresh sessions, controlled
embedding warmup, and alternating arm order. For each result, assess topic
relevance independently of its score. If required clauses are absent from a
delivered snippet and a relevant result offers `content_ref`, perform the
needed `memory_lookup` and count its complete response. Record first
sufficient knowledge, all necessary recall and lookup calls, top-four
irrelevant slots, answer losses, total response bytes, warm latency, and
production-code complexity. A fact present only in the database but absent
from all naturally returned sources is not reachable. Report baseline and
candidate by category and separately identify new instruction, correction,
and cross-project losses.

Success requires at least two independent holdout topic groups to lose
irrelevant top-four results, no loss of any baseline-reachable required fact,
and no loss of baseline-reachable instruction, correction, or cross-project
knowledge after necessary lookups. Also check the large-group reading gain.
Failure stays failure; do not tune on this holdout. The aggregate paired
report may be committed, but private questions, facts, node IDs, raw responses,
and the snapshot stay local.
