# Fourth-generation applicability verdict compatibility

`recall_applicability_verdict.py` consumes a generation-4 corpus paired with the
existing generation-3 candidate. It only validates and aggregates recorded
synthetic or field evidence; the paired runner and production retrieval are
unchanged. The parent runs the declared whole suite before a one-shot field
measurement. Synthetic success here establishes interface compatibility, not
field improvement or P4.

## Seal and public receipt

The private v4 seal has `schema_version: 1`, `generation: 4`, and
`candidate_generation: 3`. Its `files` mapping still has exactly `snapshot`,
`goldset`, `previous_goldset`, and `baseline_outcomes`, each with `path` and
`sha256`. `snapshot` and `previous_goldset` bind the original frozen snapshot
and original gold set. The seal carries the accepted baseline commit and its
actual `src` tree. `prior_seals` has two `{path, sha256}` entries in order: the
immutable v2 seal, then the failed v3 seal. The public
`artifacts/recall-applicability/prereg-v4.md` must contain the four current
file hashes, both prior seal hashes and all eight files those seals bind, plus
the original failed report and v2 report hashes. The v3 seal hash is pinned to
its published archived preregistration (`e7c0ac6`); no v3 field report exists.

Preflight verifies each historical seal and its bound files, the v3 to v2
lineage, baseline commit and input provenance, and original development cases.
It excludes v4 holdout queries, answer source IDs, and topic groups from the
original, v2, and v3 populations. It uses the existing corpus validator for
literal fact clauses in the read-only frozen SQLite snapshot. The failed v3
packet remains an exclusion population, including its inadequate baseline
cross-project control. Historical integrity does not require that failed
packet to satisfy current v4 controls or pretend it was measured.

Current v4 holdout cases use `scope: null`, `depth: 1`, and `max_results: 4`.
For each baseline outcome, `source_ranks` must equal the ranks of acceptable
oracle IDs actually present in `ranked_ids`. The baseline must rank an oracle
source in the top four for at least one holdout case in each protected category:
instruction, correction, and cross-project. Rank is a preflight source control;
paired necessary recall and lookup calls separately establish delivered fact
retention.

## Commands

Run from the checkout after the private v4 packet and public receipt exist:

```sh
python3 scripts/recall_applicability_verdict.py validate-seal \
  --seal /home/sfx/p/ae/artifacts/recall-applicability/v4/seal.json
python3 scripts/recall_applicability_verdict.py build \
  --seal /home/sfx/p/ae/artifacts/recall-applicability/v4/seal.json \
  --raw PRIVATE/v4-run/paired-raw.json \
  --run-receipt PRIVATE/v4-run/run-receipt.json \
  --candidate-receipt artifacts/recall-applicability/candidate-v3.json \
  --output artifacts/recall-applicability/report-v4.json \
  --markdown artifacts/recall-applicability/report-v4.md
python3 scripts/recall_applicability_verdict.py verify \
  --report artifacts/recall-applicability/report-v4.json --require-success
python3 -m pytest -q tests/test_recall_applicability_verdict.py \
  tests/test_recall_applicability_verdict_v4.py
```

`verify` without `--require-success` accepts a truthful failed aggregate;
`--require-success` rejects it. The v4 report uses schema version 4 and retains
the v3 aggregates. Its evidence adds the v3 seal hash. The
candidate receipt is still the hash-bound `candidate-v3.json` identity, even
when its measurement worktree was removed. Its commit, current source tree,
source file hashes, runner and input hashes, pre/post source receipts, and
single timestamped consumption must still match. Identical isolated arm
settings, ordinary trigger mode, privacy, volume, calls, necessary and exhausted
read chains, latency, and production complexity checks remain in force. V2 and
v3 CLI arguments and report serialization remain unchanged.

Success still needs two distinct improved holdout groups, no loss of a
baseline-reachable holdout or development fact, and a baseline-delivered
required fact in each protected holdout category. Necessary lookup may recover
a fact. The parent retains responsibility for the fresh declared suite before
field measurement and for P4.
