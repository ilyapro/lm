# Usage-signal work plan (goal feedback-usage-signal) — state at the pause

Written when the parent node was paused after building the shared measurement
tooling. Everything below is the decomposition the parent intends to emit next;
nothing under "children" has been started.

## Done in this branch

- `scripts/usage_signal_corpus.py` — closed recall events (`feedback_trace_id`
  set) since a cutoff, with results + current node content, closing trace,
  script class (`lat`/`cyr`/`mixed`, pair class `lat/lat` vs `cyr-any`),
  encoder-cosine relatedness (query↔trace per closure, node↔trace per pair) and
  the `recall_lookup_events` that followed (same-transport flag, lag).
- `scripts/usage_signal_replay.py` — re-grades a corpus under the checkout's
  tokenizer at a threshold sweep; reports grounding by script and relatedness,
  closure coverage, lookup overlap/union. Pinned by
  `tests/test_usage_signal_replay.py`.
- Corpora (not tracked): `~/.cache/living-memory-harness/usage-signal/{sfx,alt}-closures.jsonl`
  built from the read-only snapshots `{sfx,alt}-2026-09-07.sqlite3` in the same
  directory (SHA256SUMS there), `--since 2026-09-04T00:00:00Z`; ~2 min each with
  the real encoder. Regenerate with the corpus script if missing.
- Baseline artifact: `artifacts/grounding/usage-signal-baseline.{md,json}`
  (master tokenizer). Pooled 844 closures / 6806 pairs: at 0.25, 3.9% of pairs
  and 19.0% of closures ground (sfx 2.7% / 14.2%; lat/lat 4.7% vs cyr-any 1.2%
  on sfx). Same-transport lookups within 24 h reach 26.2% of closures on their
  own; only 58 of those 395 pairs also ground at 0.25; union coverage 38.3%
  (sfx 14.2% → 33.0%).
- Full suite green at the branch point (`bash scripts/test.sh`, exit 0).

## Intended children (parallel unless noted)

1. **lookup-credit** (decompose). Live mechanism: additive ledger table
   `recall_credit_ledger(recall_event_id, node_id, basis, source_id,
   credited_at, UNIQUE(recall_event_id, node_id))` in storage.py;
   `feedback.apply_lookup_credit(store, lookup_event_id, node_ids,
   transport_session_id)` called from `server.memory_lookup` right after
   `record_lookup_event`; join = newest `recall_events` row with the same
   `transport_session_id`, `created_at` before the lookup, whose results name
   the node; credit exactly as a grounded result (recorded per-channel scores
   through `apply_retrieval_feedback`, `signal = max(0.2, 1/(rank+1))`,
   `scope = event.scope`, anchor via `_reinforce_query_anchors(store,
   [(event, (node_id,))])`); the closure path in
   `apply_pending_recall_feedback` writes ledger rows for grounded nodes and
   skips pairs already credited (no double count); nodes not looked up earn
   nothing. Env: `LM_LOOKUP_CREDIT_POLICY=delivered|off`. Replay side:
   `ReplayResult.looked_up`, credit rule `grounded_or_lookup`, extra arm in
   `scripts/credit_rule_ab.py`; artifact `artifacts/grounding/lookup-credit.md`
   with coverage before/after from `usage_signal_replay` and the holdout A/B.
   Owns feedback.py, storage.py (ledger only), server.py (lookup wiring),
   replay.py, credit_rule_ab.py, new tests.
2. **cyrillic-tokenizer** (execute). Russian light stemmer (Snowball-style
   endings) inside `embeddings._stem` for Cyrillic tokens only; Latin and
   identifier output byte-identical to master (fixture generated from
   `git show master:src/living_memory/embeddings.py`); the 69 Cyrillic
   `_SYNONYMS` forms keep mapping (direct lookup precedes stemming);
   `retrieval._expanded_query` emits prefix terms for Cyrillic stems so the
   unstemmed `unicode61` FTS index still matches inflections; recall-map
   labels use their own `_TERM_RE` (verify unaffected), chunks/anchors are
   model-based (no reindex) — state that with evidence. Env flag to disable.
   Before/after: `usage_signal_replay` (cyr-any pairs grounded) and a goldset
   run (`retrieval_harness` over
   `~/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3` with
   `recalib/goldset-recalibration.jsonl`, hit@5/MRR not below baseline).
   Artifact `artifacts/grounding/cyrillic-tokenizer.md`. Owns embeddings.py,
   retrieval.py, tests.
3. **threshold-recalibration** (execute, depends on 2 — containment values
   change with the tokenizer). `scripts/grounding_recalibration.py`: sweep
   0.05–0.30 step 0.01 over both corpora with the replay machinery; explicit
   criterion (e.g. lowest threshold whose unrelated-pair grounded share stays
   ≤ 5% and pair signal/noise ratio ≥ 5 at both relatedness cuts 0.5/0.3 and
   0.6/0.4, per host and pooled); set `DEFAULT_MIN_CONTAINMENT` in
   grounding.py with env override `LM_GROUNDING_MIN_CONTAINMENT`; update the
   pin at `tests/test_grounding.py:181`; credit-ab holdout replay on the sfx
   snapshot (cutoff 2026-09-04) old vs new threshold (add `--min-containment`
   to `scripts/credit_rule_ab.py`). Artifact
   `artifacts/grounding/recalibration-2026-09.md`. Does not touch feedback.py
   (its stale "held at 0.25" comment is fixed by the parent at integration).

Constraints carried by every child: live databases read-only, no deploys or
restarts, changes reversible via env, tests green.
