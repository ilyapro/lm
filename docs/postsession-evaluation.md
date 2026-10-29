# Post-session insight evaluation

`scripts/postsession_insights.py` evaluates the gate and proposes insight
operations from an explicitly supplied corpus index. It never executes the
proposed operations. Calibration reads `train` and `eval`; extraction reads
`eval`. The reserved split stays outside this evaluator.

```sh
python3 scripts/postsession_insights.py --run \
  --index /path/to/corpus-index.jsonl --manifest /path/to/corpus.json \
  --db /path/to/memory.sqlite3 --out /path/to/private/report.json
```

`--db` names an existing Living Memory database. Extraction opens it with
SQLite `mode=ro`, without schema initialization, migrations, fingerprint
backfills or seeding. No preliminary copy is required. Normal SQLite WAL
visibility and sidecar requirements apply; this is not a frozen snapshot.
The extraction operation closes its reader after all workers finish, both on
success and on failure, before report writing.

The helper is a context manager: use `with build_deduper(path) as deduper`
and finish all workers inside that context. `StoreMemoryIndex` borrows the
store and propagates search errors. The helper checks the search schema even
when no proposals are produced. Missing paths, invalid SQLite files and
unsupported search schemas fail explicitly; they are never created or repaired
and do not produce a successful evaluation report. SQLite failures return
CLI exit code 2.

Without `--db`, dedup against existing memory is explicitly unavailable.
When extraction outcomes are present, the report records
`dedup.index_available: false` and unavailable verdicts. `--calibrate` alone
never opens memory, even if a `--db` argument is supplied. `--report` combines
calibration and extraction and opens memory only for extraction.

Only equal content, scope and recorded extraction source (transcript hash and
source span) suppress a proposal. Related or independently sourced claims
remain proposals. Candidate retrieval stays bounded; judge selection, budgets,
split restrictions and report fields retain their ordinary meaning.

Disposable checks run without a real corpus or external models:

```sh
python3 -m pytest tests/test_postsession_insights_cli.py
```

These checks cover reader lifetime, unchanged schema/data including a NULL
fingerprint, missing and unsupported databases, concurrent verdicts against a
synthetic index, fake-judge reporting and WAL visibility. They do not measure
production frequency, copy costs, fleet savings or judge quality.
