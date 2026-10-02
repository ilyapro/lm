# Parent verification after procedure diagnosis and anchor repair

The original goal remains incomplete. The frozen v2 field result still verifies,
but two positive procedure read chains lose applicable knowledge and one obsolete
mutant suppression expectation fails. The anchor reference repair is green in
the fresh declared suite. A production repair requires distinct source-bound
evidence; the v2 PASS cannot certify a changed source tree.

## Executed checks

Verification started at parent commit
`b266f5a78546bc8c79c184d9fec5deacd25aad03`. Its production tree is still
`a054ce1b7e3e83a1ac543718c28d67ed442302e6`. This turn changes only documentation;
production, existing tests, runner, verifier and all original/v2 sealed evidence
remain unchanged.

`python3 scripts/recall_applicability_verdict.py verify --report
artifacts/recall-applicability/report-v2.json --require-success` exited 0. The
report retains three improved independent holdout groups and zero losses of
baseline-reachable facts. Both v2 large-group cases still miss in both arms and
cannot establish the accepted reading gain.

`timeout 600 npm run check` exited 1 with **3 failed, 3379 passed, 84 skipped
in 369.09 seconds**. The only failures were:

- `tests/test_procedure_case_separation.py::test_taught_recipe_arrives_whole_through_its_legacy_schema_turned_carrier`
- `tests/test_procedure_case_separation.py::test_crowding_oracle_detects_the_group_identity_mutant`
- `tests/test_procedure_case_separation.py::test_group_node_keeps_the_trigger_it_is_found_by`

The first and third remain positive public recall/lookup losses. The second
expects a useful direct recipe to disappear when dedup is broken; its replacement
must measure the dedup identity/slot effect while retaining recipe availability.
The bounded diagnosis records actual archived baseline provenance, the first
loss stage and installed-encoder comparisons. Nothing in the field PASS cancels
these executable losses.

## Additional synthetic verifier counterexample

The verifier does not enforce the promised non-vacuous protected baseline
controls. A synthetic packet with zero baseline-reachable instruction facts
still passes `verify(..., True)` if two other holdout groups improve. This does
not change the v2 field arithmetic: its protected baseline controls are present.
It does expose a guard needed before spending a new packet. Reproduction uses
only the existing synthetic test fixture and temporary files:

```bash
python3 - <<'PY'
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path('tests').resolve()))
import test_recall_applicability_verdict as t
with tempfile.TemporaryDirectory() as d:
    p = Path(d)
    t.verdict.PREREG = p / 'prereg-v2.md'
    fixture = t.packet(p)
    raw = t.verdict.read(fixture['raw'])
    for case in fixture['cases']:
        if case['split'] == 'holdout' and case['category'] == 'instruction':
            raw['cases'][case['case_id']]['baseline'] = t.row(
                case, 'baseline', found=False)
    t.write(fixture['raw'], raw)
    t.refresh(fixture)
    report = t.verdict.build_data(
        fixture['seal'], fixture['raw'], fixture['receipt'], fixture['candidate'])
    output = t.write(p / 'report.json', report)
    t.verdict.verify(output, True)
    print(report['results']['holdout']['categories']['instruction']['baseline']['reached_facts'])
    print(report['verdict']['P1']['pass'])
PY
```

Observed output: `0`, then `True`. No field packet or production service was
queried by this reproduction.

## Remaining work and evidence boundary

The decision in `replan-retention.md` assigns three independent leaves: a
cohesive procedure-retention repair, a bounded unseen packet on the original
snapshot, and an extension of the existing verifier with synthetic mutation
controls. The original parent P1–P4 statements, populations and thresholds stay
unchanged. Their completion must be established for the final candidate.

The completed `retrieval-revision` and `paired-measurement-v2` nodes are retired
from the active decomposition rather than rebound to another source. All their
artifacts remain immutable historical evidence. Future source/regression writes
belong to `procedure-retention-repair`; future verdict script/test writes belong
to `successor-verdict-contract`. No old predicate is amended or claimed to pass
on the new source.

After integration, the parent must run the whole declared suite **before**
creating the new consumption record. Only after that gate and seal/source
preflight pass will it run the existing paired runner once, commit a new raw/run
receipt binding and data-free aggregate, and execute `verify --require-success`.
This avoids consuming independent evidence on a candidate with known executable
losses. Neither consumed field packet is rerun; a negative successor result must
remain negative. No private question, fact, source ID, response or database
content enters tracked files, Living Memory, alt or a remote repository.
