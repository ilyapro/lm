# Successor applicability verdict contract

The existing `scripts/recall_applicability_verdict.py` consumes the paired runner's
immutable evidence. It does not invoke recall, lookup, or a candidate arm. The
parent runs the declared repository checks before the single field measurement.

## Commands

Run these from the repository checkout. Paths under `PRIVATE` are local evidence
paths; the verifier writes only the two public aggregate outputs during `build`.

```sh
python3 scripts/recall_applicability_verdict.py validate-seal --seal PRIVATE/v3/seal.json
python3 scripts/recall_applicability_verdict.py build \
  --seal PRIVATE/v3/seal.json --raw PRIVATE/v3-run/paired-raw.json \
  --run-receipt PRIVATE/v3-run/run-receipt.json \
  --candidate-receipt artifacts/recall-applicability/candidate-v3.json \
  --output artifacts/recall-applicability/report-v3.json \
  --markdown artifacts/recall-applicability/report-v3.md
python3 scripts/recall_applicability_verdict.py verify \
  --report artifacts/recall-applicability/report-v3.json --require-success
python3 -m pytest -q tests/test_recall_applicability_verdict.py
```

`verify` without `--require-success` accepts a truthful failed aggregate after
recomputation. `--require-success` rejects a failed threshold. Neither command
reruns the holdout. The existing v2 `build` and `verify` arguments remain valid.

## v3 seal and receipts

The v3 seal has `schema_version: 1`, `generation: 3`, the original
`baseline_commit` and `baseline_src_tree`, and exactly the four `files` entries
`snapshot`, `goldset`, `previous_goldset`, and `baseline_outcomes`, each with
`path` and `sha256`. It adds `prior_seals: [{"path": ".../v2/seal.json",
"sha256": "..."}]`. The previous gold set is the **original** gold set;
the prior seal binds the separate v2 population. The frozen SQLite snapshot and
original previous gold set retain their original hashes. The v3 preregistration
must contain the four file hashes, the prior seal hash, the original failed
report hash, and the v2 report hash. Verification checks the prior seal's bound
files and its match to the v2 report before using either population.

`validate-seal` checks the frozen snapshot's literal oracle clauses using the
existing corpus validator in SQLite read-only mode. It checks original and v2
query, source-ID, and topic-group disjointness, unchanged development case
values, category coverage, the accepted baseline commit and source tree, and
baseline-outcome input and case provenance. It does not read candidate output.

The v3 candidate receipt uses the full v2 field set:
`baseline_commit`, `candidate_commit`, `src_tree`, and `source_sha256` for both
production source files. The run receipt retains the v2 schema: timestamped
single-consumption record, seal/raw/input/candidate hashes, runner hash,
identical arm parameters, pre/post source-tree and file hashes, and recorded
runner command. For a removed measurement worktree, the verifier resolves only
the hash-bound v3 candidate receipt at
`artifacts/recall-applicability/candidate-v3.json`; v2 resolves its v2 path.
The current checkout's `src` tree and file hashes must still match, so an old
report cannot certify a later source revision.

## Verdict and public report

The verifier recomputes fact losses from `baseline.reached_fact_indexes` minus
`candidate.reached_fact_indexes` for every case. A baseline miss is counted
separately. Success requires at least two **distinct** improved holdout topic
groups, zero new holdout and development fact losses, and at least one
baseline-reached required fact in each holdout instruction, correction, and
cross-project category. A category's presence or source rank does not meet
that control. Necessary lookup can recover a fact and counts in reachability.

The v3 JSON report has `schema_version: 3`, all existing split and category
aggregates, plus `development.baseline_missed_facts`,
`development.new_fact_losses`, and
`holdout.baseline_reached_protected_facts`. Its evidence adds
`historical_v2_report_sha256` and `prior_seal_sha256` alongside the original
failed-report hash. Original FAIL and v2 PASS are historical provenance, not
evidence that the current source passes. An unversioned v2 seal retains report
schema version 2 and the original JSON layout and aggregate arithmetic; the
non-vacuous and development-loss checks still govern its success bit.

Both generations retain exact shape-only runner input, isolated arm settings,
ordinary trigger mode, every response/call/necessary-prefix/exhausted-chain
metric, complexity arithmetic, and the query/fact/source-ID privacy check.
