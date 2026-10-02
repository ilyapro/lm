# Parent verification of the v4 applicability run

The accepted source is candidate generation 3 (`candidate-v3.json`), and the
independently sealed corpus is generation 4 (`prereg-v4.md`). The original FAIL,
the v2 PASS, and the insufficient unconsumed v3 packet remain historical
evidence. This is the single paired run on v4; no corpus or query was selected
after candidate output was seen.

## Executed checks

From the parent worktree, before writing the v4 consumption record:

```text
python3 scripts/recall_applicability_verdict.py validate-seal --seal /home/sfx/p/ae/artifacts/recall-applicability/v4/seal.json
  exit 0
npm run check
  exit 0; 3443 passed, 84 skipped in 209.83s
```

The candidate receipt's `src_tree` equaled `HEAD:src`, and the two recorded
production source hashes matched the current files. The paired runner then
completed 18 cases once, using the exact shape-only v4 input, a fresh writable
copy of the same frozen SQLite snapshot and a fresh worker per case and arm.
Each arm used its case's same scope, depth 1, max 4, ordinary trigger mode, and
the installed embedding backend. The private consumption record predates the
run. The raw result, input, command, source hashes, and timestamps are bound by
the private run receipt under `/home/sfx/p/ae/artifacts/recall-applicability/v4-run/`.

```text
python3 scripts/recall_applicability_verdict.py verify --report artifacts/recall-applicability/report-v4.json --require-success
  exit 0
```

## Acceptance reading

The [public aggregate](report-v4.json) records two improved independent
holdout groups, zero newly lost baseline-reachable facts in holdout or
development, and baseline-delivered protected facts in instruction (3),
correction (3), and cross-project (3). Candidate delivery reaches 22 of 32
holdout clauses versus 17 of 32 for the baseline. The two improved categories
are instruction and correction. Concrete and compound cases show no top-four
noise reduction in this packet; compound facts were already fully reachable
in both arms. The baseline missed 15 holdout clauses, including all five in
the large-group cases; those cases cannot establish a field reading gain.

The aggregate includes per-category relevant and irrelevant ranks, necessary
and exhausted recall/lookup sequences, response bytes, warm latency, and
production source complexity. Across holdout cases, baseline and candidate
respectively used 14/49 and 14/44 recall/lookup calls, delivered 430400 and
254619 response bytes, and summed 28132.831 and 28371.826 ms of warm
latency. Production deltas against the accepted baseline are +45/-10 lines in
`retrieval.py` and +20/-10 in `score_gate.py`.

The production diff reuses trigger collection, the existing rank blend, and
the existing gate. It reduces the rank contribution of a generic trigger to
the content-supported and query-covered evidence while retaining the binding
instruction availability path. The `name` mode remains a separate optional
branch; no default scope restriction, history removal, mandatory LLM call,
provider, model, or tier change appears in the diff. The passing declared
suite includes the focused applicability, procedure retention, correction,
feedback, lookup, and large-carrier delivery regressions. The field
large-group cases are an explicit limitation of this run, not the evidence
for the already accepted large-carrier delivery behavior.

The report's P4 field is intentionally `false` because the verifier does not
execute the repository suite. The fresh `npm run check` result above supplies
that parent-level verification.
