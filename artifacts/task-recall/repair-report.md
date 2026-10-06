# P3 repair observations

The previous root result correctly left P3 open, but did not make delivery
executable through the native completion boundary. These observations extend
the earlier development report; they do not replace its baseline or claim
publication, gate success, or live uptake.

## Concrete gaps

The journal now contains the original structured P3 promise with empty
verification. The root has no accepted check contract or completion contract.
The earlier operator request has type `decision`; the native completion
consumer requires a current addressed `clarification`. Its draft also named
the work branch as publication ref, whereas the root's ordinary parent is
`refs/heads/master`.

Initial binding requires two native operations. First accept an exact
`CONTRACT_REVISION` that preserves the P3 promise while adding product checks
and separating delivery evidence. Then re-read the bound predicate and consume
the exact initial `AMEND_COMPLETION`. The combined revision/amendment path
requires an existing nonnull completion grant, which this root does not have.
The concrete pending proposals are in `native-contract-repair.json`; those
bytes are neither an operator answer nor an applied contract.

The original observer emitted prose, while `verify_command` parses fresh JSON
stdout and validates the consumer process and loaded-source projection.
The observer now has an optional native mode that compares a separately
imported published candidate's function digest with the live reader, checks
the authenticated process incarnation and installed source, and emits the
required JSON. Its original mode remains available. Five focused tests passed,
including a live isolated reader and refusal of different loaded functions.
Source hashes remain a projection supported by function identity, editable
import resolution, source timestamps, and stable installed bytes; the receipt
explicitly does not claim that function identity attests every source byte or
module global.

## Existing-population check before publication

A fresh SQLite backup opened read-only at the source was tested through public
MCP recall. The copy's node, connection and recall-event counts survived
migration; `quick_check` returned `ok`, and the task index was present. The
copy was then deleted. No source memory was rewritten.

Both original development queries still missed their target at `max_results=6`
on this candidate. The short request took 519.1 ms and the contextual request
459.7 ms. This is a diagnostic on an existing private population, not an
independent holdout or live-reader delivery evidence. Only aggregate outcomes
are recorded here; private inputs and memory identifiers stay outside Git.

Inspection found that FTS admitted the target at lexical ranks 7 and 44, but
reciprocal-rank scoring reduced its lexical evidence to 0.142857 and 0.022727.
Final ranks were 145 and 14. The short query's vector shortlist omitted the
target's content similarity; neither decay nor supersession caused the miss.
The resulting repair preserves compound identifiers as additional ordinary FTS
phrase evidence while retaining their individual words, and retains content
similarity already computed for lexically admitted candidates even when they
fall below the vector discovery shortlist. It introduces no task-equality
priority or scope filter and changes no weights, gates, or candidate limits.

On a fresh backup using the committed code, both original public queries now
reach their target at ranks **6 and 5**, still at `max_results=6`. No per-case
timings were retained for that confirmation. A synthetic regression with 160
reordered-token distractors fails independently when either repair is removed
in subprocess memory; it uses public retrieval and computed embeddings.

## Comparable synthetic result

The original fixture, six inputs, oracle, seed and limit are unchanged. This
single repair comparison is development evidence. The original baseline and
first candidate's report remain intact; `repair-after.json` records the final
candidate, with no new holdout population.

| Request | Baseline target rank | Repaired target rank | Recall + needed lookup | Bytes before → after | Time ms before → after |
| --- | ---: | ---: | --- | ---: | ---: |
| Short identifier | absent | 4 | 1 + 0 → 1 + 1 | 2,264 → 7,234 | 463.312 → 601.873 |
| Identifier and context | 2 | 1 | 1 + 1 → 1 + 1 | 8,389 → 9,850 | 33.308 → 31.797 |
| Explicit scope | 1 | 1 | 1 + 1 → 1 + 1 | 8,500 → 8,519 | 27.279 → 29.374 |
| Correction | 1 | 1 | 1 + 0 → 1 + 0 | 4,294 → 4,294 | 16.714 → 19.468 |
| Cross-project lesson | 2 | 2 | 1 + 0 → 1 + 0 | 4,095 → 4,096 | 15.592 → 19.279 |
| Missing fact | absent | absent | 1 + 0 → 1 + 0 | 1,541 → 1,541 | 13.552 → 13.376 |
| Total | | | 6 + 2 → 6 + 3 | 29,083 → 35,534 | 569.757 → 715.167 |

The missing-fact request still returns unrelated records; none satisfies the
oracle. The short ambiguous identifier does not give the desired record
unconditional priority over records with the same task. Context makes the
desired fact first. The added lookup and bytes obtain the newly reachable long
fact. Cold-start timing is included, and no general acceleration is claimed.

The existing FTS/BM25/vector retrieval pipeline is reused: **zero additional
searches, indexes, registries, services, or task-ID prose copies**. A caller who
knows the task no longer needs a mandatory content-word reformulation. Normal
snippet/content_ref lookup remains. Across the whole product change relative
to `4e0753b`, the two production files total **85 added / 15 removed lines**;
this is size evidence, not a deletion quota. The bounded observer adapts the
existing native delivery interface; it is not a new publisher or daemon.

## Targeted verification

All commands completed successfully. The groups overlap and are not summed as
distinct tests:

- `npm test -- tests/test_retrieval_task.py tests/test_storage.py tests/test_retrieval_lexical_recall.py -q`: 80 tests.
- `npm test -- tests/test_retrieval_correction_dominance.py tests/test_recall_applicability.py tests/test_retrieval_cross_scope_gate.py tests/test_scopeless_recall_mcp.py tests/test_retrieval_chunk_maxpool.py -q`: 58 tests.
- Final joined root command `timeout 180 env PYTHONPATH=src python3 -m pytest -q tests/test_retrieval_task.py tests/test_storage.py tests/test_live_task_delivery.py`: 77 tests.

The exact proposed contract passes pure shape/predicate/shell validation against
the current journal. This validates the proposal only; it does not authenticate
an operator answer or mutate native state. The native request classifier
confirms the new clarification belongs to the current root incarnation.

## Delivery status and native blocker

The supervisor supplied the genuine stage1 response at 06:31:10Z. The native
request reader authenticated the current incarnation and exact response, then
refused the revision at artifact-provenance admission. This root had already
published a ledger whose artifact declaration is empty. The proposed revision
adds the two source files and live receipt to `required_artifacts`, while native
policy permits a revision to retire frozen declarations but not add them. The
native refusal is `provenance_bound` / `contract revision cannot extend frozen
artifact authority`. The historical answer and DONE provenance remain intact;
the current journal still has empty P3 verification and no completion contract.

Removing those declarations from a corrected revision would avoid this first
refusal, but the native `completion_delivery` path selects its loaded-code
population only from root `required_artifacts` (unless a separate prospective
consumer grant exists) and requires that population to be nonempty. This root
has no such grant. A command-only delivery selector would therefore fail with
`delivery_loaded_paths_missing` after publication. The observer itself accepts
explicit loaded paths and checks live function identity, but the native proof
boundary also needs an accepted source population. That path must be repaired
in the native completion runtime before this root can safely attach its exact
delivery contract and publish through the same-owner completion path. An
operator answer cannot override either provenance or verification guard.

P3 remains open with its exact original promise. Product acceptance, the ordinary
gate, normal publication, service update, and fresh native delivery verification
remain required. No private query, memory body, memory identifier, or database
snapshot is a repository artifact. No provider, tier, schema-name trigger,
retired host, or operator pause has been changed.
