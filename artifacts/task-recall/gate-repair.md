# Task-recall merge-gate repair

The rejected candidate had three schema-compatibility failures, plus a schema
dedup fixture failure also seen on the base. The base comparison additionally
exposed a timestamp-dependent response-size assertion. This repair changes
tests and their shared compatibility assertion; production retrieval, weights,
thresholds, source inventory, and the recorded synthetic comparison are unchanged.

## Concrete causes and repairs

- The ledger tests required every existing SQL definition to remain byte-identical
  to `master`, which excluded the intentional task-index migration. Their shared
  assertion now permits only the known definitions for six derived objects:
  `nodes_fts`, its content shadow table, its three triggers, and its vocabulary
  table. Formatting is normalized only for those objects; unexpected definitions
  still fail. Every unrelated object retains byte identity, and the existing
  explicit list of allowed additive objects is unchanged. Four negative controls
  reject unrelated table changes, a reduced FTS definition, a broken update
  trigger, and a vocabulary reverting to row-level frequencies. The reopen test
  additionally compares complete stored node/event rows and reads both task and
  content evidence. Existing fixed-legacy-index tests still exercise new and
  edited tasks independently of the moving `master` comparison.
- The dedup fixture delivered only one schema with hash embeddings: the other
  two schemas scored about 1.025, below four traces at about 1.028. The installed
  MiniLM backend happened to satisfy the crowding premise. Concise procedure
  steps and longer supporting notes now establish that premise through actual
  retrieval with the deterministic hash backend. The test checks exact refill
  order and newly admitted records, not only the result count.
- The paired-run fixture could create and recall nodes within the same second
  in one worker but not the other. Sparse delivery then added two `updated_at`
  fields, reproducing the exact reported difference: 1,251 versus 1,323 bytes.
  The fixture now predates both workers and records a recent sweep so measured
  sweep duration cannot enter this equality check. Strict response-byte equality
  remains; the runner still measures actual serialized responses without
  subtracting or normalizing their fields.

## Verification

All four targeted groups passed; they contain 124 distinct tests:

| Command | Result |
| --- | --- |
| `timeout 180 env PYTHONPATH=src python3 -m pytest -q tests/test_retrieval_task.py tests/test_storage.py tests/test_live_task_delivery.py` | 77 passed |
| `timeout 180 env LIVING_MEMORY_EMBEDDING_BACKEND=hash PYTHONPATH=src python3 -m pytest -q tests/test_transcript_ledger.py tests/test_lookup_credit.py` | 34 passed |
| `python3 -m pytest -q tests/test_recall_schema_dedup.py` | 10 passed, also verified with explicit hash environment |
| `python3 -m pytest -q tests/test_recall_applicability_runner.py` | 3 passed |

Both accepted artifact checks and `git diff --check` passed. The two transcript
schema tests have descriptive new names reflecting the precise migration
allowance; their master-built database comparisons remain active.

No production mechanism or memory copy was added by this repair. It does not
rerun or replace the earlier bounded synthetic measurement and does not claim a
performance improvement.

## Completion boundary

The current native typed contract contains the authorized completion amendment
with the 31-file loaded-source population. Read-only inspection of the native
acceptance journal still shows the original P3 predicate open and the delivery
command awaiting publication. Earlier reports describe prior contract blockers;
they remain historical evidence. Required artifacts, accepted commands, completed
children, and negative history were not changed.

The whole suite belongs to the ordinary merge gate after this repair result.
This targeted verification is not a full-gate PASS. Normal product acceptance,
successful gate and publication must precede the same owner's reader update and
independent loaded-code/public-behavior observation. P3 remains `met: false`
until that delivery evidence succeeds.
