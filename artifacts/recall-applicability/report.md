# Sealed applicability recall: paired measurement

**Verdict: FAIL.** The candidate loses two baseline-reachable clauses from an applicable instruction and reduces irrelevant top-four results in only one independent holdout topic group. The preregistered threshold requires at least two groups and zero fact losses. No candidate tuning or second holdout run was performed.

## Provenance and method

- Frozen SQLite SHA-256: `3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef`; sealed gold set: `c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c`; pre-candidate rank capture: `35dccb9896e3c99c63e1d53c38fbb812667a784a237ca87a0ef1039e236a0318`.
- Private shape-only runner input SHA-256: `770445b4af22220f61f7f021d0b26be481e781080e04ee938530a679ceafe238`; private raw paired result SHA-256: `45ccf975ac3a1430d01425cf7fca3a6f21b177e45667da16d008c7994cbba88c`. These stay under `/home/sfx/p/ae/artifacts/recall-applicability/` and contain queries, clauses and node IDs; this tracked report contains aggregates only.
- Compared accepted `b9769d8` with candidate `e9637b64e52c816923819c7e7841b976ab0c70b8` on 18 cases (4 development, 14 sealed holdout), using depth 1 and max_results 4. Separate writable SQLite backups and fresh worker processes isolate each arm/case; arm order alternates. The same environment was inherited, except each arm's source root. Name-trigger mode was not enabled.
- The gold file stores clauses as strings. A private shape-only conversion wrapped each original clause in the runner’s `{text, acceptable_source_ids}` form without changing the sealed file, query, scope, clause or source ID. The worker saw only query/scope/depth/max_results. Every returned `content_ref` was fully looked up. A fact counts only when an allowed source supplies its exact clause; the first sufficient call defines the necessary prefix. Unreached cases exhausted returned references and have no sufficient prefix.
- Warm latency sums timed recall and full lookup calls after model initialization; first-query vector/index work remains timed. Bytes include complete JSON responses. The runner looked up every offered reference, so total calls/bytes can exceed the first sufficient prefix.

## Sealed holdout by category

| Category | Facts reached B→C | Sufficient cases B→C | Irrelevant top 4 B→C | Lost facts | Calls B→C (lookups) | Total bytes B→C | Warm ms B→C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| absent-knowledge (2) | 0/0 → 0/0 | 0 → 0 | 8 → 8 | 0 | 9 (7) → 9 (7) | 39,164 → 39,164 | 4,064.4 → 3,046.7 |
| compound (2) | 2/4 → 2/4 | 1 → 1 | 7 → 7 | 0 | 9 (7) → 9 (7) | 33,016 → 33,016 | 6,267.0 → 5,425.8 |
| concrete (2) | 0/3 → 1/3 | 0 → 1 | 8 → 7 | 0 | 10 (8) → 9 (7) | 73,752 → 32,790 | 4,271.9 → 4,164.2 |
| correction (2) | 4/4 → 4/4 | 2 → 2 | 6 → 6 | 0 | 9 (7) → 8 (6) | 33,144 → 27,148 | 3,635.1 → 3,966.0 |
| cross-project (2) | 1/3 → 1/3 | 1 → 1 | 7 → 7 | 0 | 9 (7) → 9 (7) | 45,532 → 34,514 | 3,317.9 → 3,995.0 |
| instruction (2) | 2/5 → 0/5 | 1 → 0 | 7 → 8 | 2 | 9 (7) → 9 (7) | 113,340 → 98,186 | 11,984.5 → 13,351.8 |
| large-group (2) | 0/4 → 0/4 | 0 → 0 | 8 → 8 | 0 | 9 (7) → 9 (7) | 129,653 → 130,938 | 3,287.3 → 3,543.7 |
| **Total (14)** | **9/23 → 8/23** | **5 → 5** | **51 → 51** | **2** | **64 (50) → 62 (48)** | **467,601 → 395,756** | **36,828.0 → 37,493.1** |

The net irrelevant-slot count is unchanged: one concrete topic group improves by one slot; one instruction group worsens by one. The concrete group gains one fact; the instruction group loses both of its previously reachable clauses. Both correction cases retain all four clauses; one cross-project case retains its reachable clause. The other cross-project case remains unreached in both arms. Both absent-knowledge questions receive four irrelevant results per arm; neither arm produces an empty answer. Large-group facts remain unreachable in both arms (0/4); those cases do not establish the accepted reading gain.

### Necessary reading chain

All five baseline sufficient holdout cases and all five candidate sufficient cases have `memory_recall` as their first sufficient call. Across those sufficient cases, necessary bytes are **42,250 → 33,798** and necessary calls **5 → 5**. The instruction loss changes which cases contribute to these totals; it cannot be treated as an efficiency win. Each arm made 14 recall calls and then 50 → 48 full lookups of returned references; all 9 unreached cases per arm exhausted their returned references without a sufficient answer. The aggregate necessary sequence distribution, including categories, is in `report.json`. Warm latency median per case is **1,831.7 → 1,969.0 ms**; the 14-case summed latency is **36,828.0 → 37,493.1 ms**.

## Development and code/regression checks

Development reached 3/6 → 3/6 clauses, with 14 → 14 irrelevant top-four slots. The reported P5_RUSSIAN example remains unreachable in both arms; it is not evidence of success. The development compound case retains both clauses and moves its first sufficient point from recall plus four lookups to the initial recall.

Production diff against `b9769d8`: retrieval.py +14/−3 lines (2,809 → 2,820); score_gate.py +0/−0 (238 → 238). The candidate changes trigger eligibility before the existing ranking, cross-scope admission and score gate use the trigger score. It does not enable `LM_RECALL_SCHEMA_TRIGGER=name`, narrow scope, remove memory, add an LLM call, or alter model/provider/tier code.

Targeted checks passed: 64 tests for applicability, score gate, cross-scope, name channel and runner; 31 tests for delivery chain, carrier omission and procedure separation. A further 62 feedback-loop, explicit-feedback, transport-closure and feedback-amplification tests passed. The accepted large-carrier reading gain is exercised by `tests/test_recall_carrier_omission.py::test_latest_rejected_revision_counterexample`: the fact requires recall plus lookup in the earlier rejected revision, but is delivered in one recall call by the candidate. The sealed large-group cases remained unreachable in both arms, so they do not independently demonstrate a field gain. The repository's declared `npm run check` suite passed: 3,369 passed, 84 skipped.

## SPEC verdict

| Item | Verdict | Evidence |
| --- | --- | --- |
| P1 | **FAIL** | One improved holdout group; two instruction facts lost. |
| P2 | **PASS** | Paired state/parameters/isolation and category measures above and in JSON. |
| P3 | **PASS** | Production diff limited to shared trigger eligibility; forbidden changes absent. |
| P4 | **PASS** | Focused checks and declared suite pass; the accepted large-carrier reading regression retains its one-call gain. |

The report contract now checks sealed hashes, aggregate arithmetic, SPEC verdicts, and absence of private queries, clauses, and node IDs. This validation does not change the failed holdout verdict.
