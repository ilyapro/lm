# Curated failure cases — animal-planet audit packet

De-identified failure cases for the four baseline failure families, curated
from the tracked replay corpus (`../corpus/dev.jsonl`, `../corpus/eval.jsonl`)
only — `holdout.jsonl` is sealed and was never opened (see `../corpus/POLICY.md`;
holdout absence of every referenced event id is additionally guaranteed by the
recorded split hash rule and was grep-verified mechanically).

- `index.json` — family definitions, honest recorded-footprint notes
  (including zero-findings), per-split aggregates, case list, verification.
- `cases/<case-id>.json` — 16 cases, 4 per family, dev and eval represented in
  every family. Case schema: `{id, family, split, event_ids, node_ids,
  selection_rule, description, expected_behavior, metric_hook, evidence}`.
  All content is structural (ids, scopes, levels, ranks, scores, timestamps,
  char counts); no surrogate or original text.
- `curate/curate_failures.py` — deterministic generator; every embedded number
  is recomputed from the corpus. Verify current files:

```bash
python3 artifacts/animal-planet/failures/curate/curate_failures.py --check
```

Families → `ap_baseline.py compare` metric hooks: `cross_scope` → cross_scope,
`auto_dup` → auto_recall (payload secondary), `correction_ordering` →
correction_dominance, `mixed_era` → payload / correction_dominance (see each
case's `metric_hook_note`; the never-delivered mixed-era schema is a
node-level, prospective case by design).
