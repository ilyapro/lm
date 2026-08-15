# 03 — Calculator: `scripts/ap_baseline.py`

Third section of the animal-planet audit/replay-packet recipe. The calculator
is the single reproduction and falsification tool for the frozen baseline: it
derives every metric mechanically from the tracked corpus
(`artifacts/animal-planet/corpus/`), never from stored constants. The cited
audit numbers (42.4 / 17.6 / 26,496 / 35,237 / 727 / 9 / …) exist in exactly
one place — the expected-value/tolerance table (`EXPECTED` in the script,
mirrored into the report's `expected`/`comparison`/`discrepancies` blocks) —
and are used only for comparison, never emitted as computed output.

```bash
python3 scripts/ap_baseline.py report                      # write baseline-report.json
python3 scripts/ap_baseline.py verify                      # gate the frozen packet (exit 0 iff clean)
python3 scripts/ap_baseline.py compare --split dev         # before/after JSON on stdout
```

All subcommands accept `--corpus-root` (default
`artifacts/animal-planet/corpus`) so tamper tests can point at a modified
copy; `verify` resolves `manifest.json` and `baseline-report.json` from the
corpus root's parent directory. Everything is deterministic (no timestamps,
sorted keys, stable ordering): identical corpus bytes → identical outputs.
Stdlib-only, except that `compare` imports `living_memory` from `./src`;
no network, no database, no MCP.

## `report` — the frozen baseline

Windows come verbatim from `splits.json` (W1/W2/W3, inclusive both ends).
One generic pass aggregates all three split files identically; the sealed
holdout is only ever touched by this mechanical whole-file aggregation
(`corpus/POLICY.md`).

Metric definitions (all over alt-source events):

- **`organic_linkage_pct`** — W2 `project:game` events with `class=organic`
  (splits.json `event_class_predicate`: automatic iff `agent IS NULL`) that
  have `feedback_applied` set, as a share of W2 organic events. Computed
  161-event population, 71 linked. `feedback_applied` is a
  **lower-bound proxy for observed use, not ground-truth relevance**
  (recorded verbatim in the report's caveats).
- **`automatic_linkage_pct`** — same over `class=automatic` (1188 events,
  209 linked; both exact vs the extract-stage cross-check).
- **`payload_median_chars` / `payload_p90_chars` / `payload_mean_chars`** —
  over `transcript_serialized_chars` (the pinned `len_json_content` measure
  of the matched MCP recall tool-result) of transcript-matched events created
  in W3. Median is `statistics.median`, p90 is nearest-rank
  (`sorted[ceil(0.9·n)−1]`), matching the extract-stage measurement.
- **Cross-checks** — W1 recall-event count; W1 supersedes edges (relations
  deduped by `(source, target, created_at)` across split files); the
  deterministic pre-recall template counts (`reopen_lesson`,
  `architectural_decision`) and their W1 share; W2 `project:game`
  population/class/feedback counts; `w1_auto_outcome_traces` and
  `w1_avg_recall_kb` are recorded as `null` — not derivable from the
  de-identified corpus (see discrepancies d2/d4).

Documented tolerances: linkage ±0.5 pp vs the rounded cited values; payload
±1% vs the cited values (covers the 26.5k/35.2k roundings); counts exact.

Every deviation is reported explicitly, never patched: the
`discrepancies[]` block carries the four entries copied/extended from the
packet metadata (`recipe/01-extract.md` cross-check outcome, mirrored from
staging `METADATA.json.discrepancies[]`) with dynamically computed values —
d1 organic population (cited 165/70 is internally inconsistent; the pinned
predicate yields 161/71 → 44.1% vs cited 42.4%), d2 auto-OUTCOME (content
template not recomputable from surrogates; extract pinned 78/89 vs cited
105), d3 W3 payload population (164 survivors reproduce exactly; the cited
278 includes 114 unrecoverable `/root`-session tool-results), d4 the
12.6KB/recall figure (len_text basis over all W1 tool-results, not carried
in the corpus). `report` warns loudly if any deviation is NOT covered by a
recorded discrepancy.

## `verify` — the packet gate

Exit 0 iff all three hold:

1. **(a) Reproduction** — a full recomputation of the report equals the
   stored `baseline-report.json` (any drift lists JSON paths).
2. **(b) Citations** — every metric in the comparison table is within its
   documented tolerance of the cited value, or its deviation is covered by a
   recorded discrepancy entry.
3. **(c) Integrity** — SHA-256 of `corpus/{dev,eval,holdout}.jsonl` +
   `splits.json` match `manifest.json`. A missing manifest exits 2 with a
   clear message (the manifest is written by the freeze step).

Manifest contract (for the freeze step): the manifest must associate each of
the four corpus files with its SHA-256. The reader is shape-tolerant — a hash
binds to a file when the key holding it, an ancestor key, or a sibling string
value names the file — but the recommended canonical shape is:

```json
{"files": {"corpus/dev.jsonl": {"sha256": "…"}, "corpus/eval.jsonl": {"sha256": "…"},
           "corpus/holdout.jsonl": {"sha256": "…"}, "corpus/splits.json": {"sha256": "…"}}}
```

Tamper behavior (exercised during development): byte tampering fails (c);
tampering + rehashing fails (a); a full adversarial re-freeze (tampered
corpus + regenerated report + rehashed manifest) still fails (b) because the
recomputed counts leave the exact cited values with no covering discrepancy.

## `compare` — before/after replay path

`compare --split {dev,eval,holdout} [--metrics
payload,cross_scope,correction_dominance,auto_recall]` prints one JSON
document to stdout and exits 0 on successful computation — thresholds and
pass/fail judgments are the caller's contract, not the calculator's.
Holdout is sealed until the packet freeze; during development `compare` runs
on dev/eval only (a warning is printed for holdout).

**BEFORE** is the recorded behavior in the corpus: recorded candidate ranks,
scopes and per-method scores, recorded `transcript_serialized_chars`, and
recorded per-session repetition of automatic recalls. **AFTER** is the
current code applied to the same events:

- **Delivery replay** (`payload`, `auto_recall`): events are grouped by
  `transport_session_id` and replayed in `(created_at, id)` order.
  `RecallResult` + `Node` objects are reconstructed from corpus records
  (surrogate content preserves exact lengths, whitespace positions and
  content-equality classes — precisely the invariants
  `delivery.shape_recall_results` depends on) and shaped through
  `shape_recall_results` with the current defaults
  (`LM_DELIVERY_*` env knobs honored and echoed in the output), with
  `already_delivered_ids` accumulated per session over a 200-event horizon
  mirroring `storage.delivered_node_ids`. Replayed payload is the char
  length of the reconstructed current-server envelope
  (`{query, scope, recall_event_id, count, results, auto_decay}`,
  `ensure_ascii=False`). Node provenance content is not in the corpus;
  a shape-filler reconstructs it to the recorded per-key sizes and pads to
  the recorded `serialized_chars` (residuals are counted in the output).
  The payload family additionally reports `replayed_unshaped` — a
  legacy-renderer emulation (all results full, provenance verbatim) — so
  `unshaped − before` quantifies reconstruction bias (snapshot-time
  provenance/stats growth, envelope approximation, escaping) while
  `after − unshaped` isolates the pure shaping effect.
- **Recorded-candidate rerank** (`cross_scope`, `correction_dominance`):
  a `living_memory.replay`-style rerank through the real
  `MemoryRecallService.rank_candidates` (via `replay.make_ranking_service`)
  over synthetic `_Candidate`s built from the recorded per-method scores
  plus corpus node flags (decayed, confidence/usefulness/access stats) and
  the split file's supersedes pairs, under current default weights
  (`floor_default_weights` with assumed evidence), truncated to the event's
  `max_results`. Cross-scope = result node scope ≠ requested scope.
  Correction-dominance violations = a superseded node ordered above its
  superseding correction when both are candidates of one event; because
  supersedes edges are snapshot-time and mostly post-date the recorded
  recalls, this pair population can be 0 — the family therefore also tracks
  `superseded_deliveries` (results whose node is a supersedes target) and
  `superseded_top1_events` before/after.

Replay fidelity limits (also in `--help` and in every `compare` output):

- BM25/vector/graph **re-execution is out of scope** — reranks re-mix the
  recorded candidate scores through the current ranking code; candidates the
  historical ranking excluded can never (re)appear.
- **Node stats and supersedes edges are snapshot-time**, not event-time.
- Content is surrogate: snippet boundaries may cut differently and
  causal-query markers are undetectable (mode derives from the recorded
  `depth` enum with a neutral query).
- BEFORE payload chars are historical wire text of the originals; AFTER
  chars measure the replayed surrogate response with the current envelope.
- Session replay sees only the selected split's slice of each transport
  session (splits are hash-partitioned by event id); this applies to BEFORE
  and AFTER equally.
