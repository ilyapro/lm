# Parent integration verification

The original goal is **not complete**. The frozen v2 field result verifies,
but the declared suite still exposes procedure-retention and query-anchor
oracle failures. P4 remains unmet; a successful field sample does not waive it.

## Verified evidence

- `python3 scripts/recall_applicability_verdict.py verify --report artifacts/recall-applicability/report-v2.json --require-success` exits 0 in the parent checkout.
- The v2 report still records three improved independent holdout groups and
  zero baseline-reachable fact losses. Its SHA-256 is
  `d6bc3d9af1a3a313737e08238332f27bb1485422b89b69083413b910fdcac3d5`.
- The candidate source tree remains
  `a054ce1b7e3e83a1ac543718c28d67ed442302e6`, as frozen in candidate-v2.json.
  No production file or paired runner was changed during parent verification.
- `git diff --exit-code 9c1adbc -- src scripts/recall_applicability_eval.py artifacts/recall-applicability/report-v2.json artifacts/recall-applicability/report-v2.md artifacts/recall-applicability/candidate-v2.json` exits 0.
- `git diff --exit-code df445b7 -- artifacts/recall-applicability/report.json artifacts/recall-applicability/report.md artifacts/recall-applicability/prereg.md scripts/recall_applicability_corpus.py` exits 0. The first failed measurement remains unchanged.
- No consumed holdout was rerun. Private packets, raw measurements and
  consumption records were only read by the verifier, and their hashes match.

## Bounded integration repairs already committed

Commit `0a12a24` fixes evidence relocation after the harness removes a child
execution worktree. The frozen report and private run receipt name the old
absolute candidate-receipt and runner paths. The verifier now accepts a missing
candidate receipt at the same repository-relative path in the current checkout
only with its original SHA-256. Both recorded paths must identify the same
historical checkout, existing edited files are rejected, and the runner and
source hashes remain mandatory. Report and receipt paths/bytes are preserved.
No metric, threshold, population, source identity or loss arithmetic changed.

Commit `a32e9e0` isolates synthetic preregistration paths. Before integration,
the verdict tests had no field preregistration in their checkout; after the
corpus sibling merged, their synthetic hashes were compared to the real
prereg-v2.md. Each synthetic packet now has its own preregistration, including
the CLI subprocess. An edited-preregistration regression still checks rejection.
`python3 -m pytest -q tests/test_recall_applicability_verdict.py` passes all 11
tests, including removed-worktree, stale hash, wrong path and existing edited
artifact cases.

## Declared-suite result

The first `timeout 2400 npm run check` run returned 14 failed, 3366 passed,
84 skipped. Nine failures were the synthetic-preregistration collision above.
After the verifier integration repairs, a fresh identical command returned
**5 failed, 3377 passed, 84 skipped in 196.12 seconds** (exit 1). The remaining
failures are:

1. `tests/test_procedure_case_separation.py::test_taught_recipe_arrives_whole_through_its_legacy_schema_turned_carrier`
2. `tests/test_procedure_case_separation.py::test_crowding_oracle_detects_the_group_identity_mutant`
3. `tests/test_procedure_case_separation.py::test_group_node_keeps_the_trigger_it_is_found_by`
4. `tests/test_retrieval_query_anchors.py::test_cold_start_ranking_is_byte_identical_to_pre_anchor_code`
5. `tests/test_retrieval_query_anchors.py::test_unresembled_query_class_ranks_identically_to_pre_anchor_code`

The first procedure check loses the whole taught current recipe through the
ordinary read chain; the third does not return the trigger-preserving carrier.
The second is different: its mutant now leaves the recipe reachable, so the
negative assertion fails. These must not all be dismissed as stale score tests.

The three procedure failures reproduce in isolation with `npm test -- -q`
and their complete node selectors. The exact same three unchanged fixtures
pass against actual archived accepted baseline
`b9769d84e3188ee1e646627ebe1b4d454f69d6f9`, using the same hash embedding backend
as scripts/test.sh. The baseline process inserted the archived src directory
at sys.path[0], imported retrieval before invoking pytest, printed and checked
its archive import provenance, and overrode pytest's pythonpath. Merely setting
PYTHONPATH was insufficient in this environment: an explicit import-origin
assertion caught the current checkout being imported, before any test ran.
The cause is the checkout-root living_memory/__init__.py shim preceding
PYTHONPATH under `python -c`. A separate import probe using the paired runner's
script-directory search path correctly selected the archived src: the runner's
workers do not have this root-shim collision.

A minimal baseline reproduction uses a temporary `git archive b9769d8 src`,
then the following child program with that archive's src as argv[1] and the
three test selectors as pytest arguments:

```python
import sys
sys.path.insert(0, sys.argv.pop(1))
import living_memory.retrieval
print(living_memory.retrieval.__file__)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
```

Set `LIVING_MEMORY_EMBEDDING_BACKEND=hash` and pass
`-q -o pythonpath=<archived-src>` along with the selectors. This uses only
synthetic test databases, not the sealed field packets.

The two anchor checks each report the existing backup-snapshot trigger query
as their sole divergence from the historical pre-anchor reference. Its trigger
rank semantics predate the intentional v2 change. This suggests an oracle
compatibility problem, but the replacement must still detect anchor-induced
changes and retain the original query matrix and independent controls.

## Remaining scope

The source and all sealed v2 evidence stay frozen while two independent children
work: a bounded causal diagnosis of procedure retention, and a test-only repair
of anchor reference compatibility. The diagnosis is not a P4 pass. If restoring
the positive procedure promises needs production edits, the parent must first
address which source-bound evidence and acceptance obligations become stale;
it must not reuse the v2 PASS for another source tree or silently retune against
the consumed packet. No new field corpus is authorized by this decomposition.
