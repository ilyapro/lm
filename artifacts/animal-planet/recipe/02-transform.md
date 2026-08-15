# 02 — Transform: private staging → de-identified replay corpus

Second section of the animal-planet audit/replay-packet recipe. It documents
the deterministic de-identification transform and the pre-declared splits.
Rebuilding the corpus from staging (01-extract.md) and re-verifying it is two
commands:

```bash
python3 artifacts/animal-planet/recipe/transform/build_corpus.py
python3 artifacts/animal-planet/recipe/transform/verify_corpus.py   # exit 0 iff clean
```

Inputs: the private staging dataset `$AP_STAGING` (default
`/home/sfx/.cache/ap-audit/staging/`, never tracked). Outputs (tracked):
`artifacts/animal-planet/corpus/{dev,eval,holdout}.jsonl`, `corpus/splits.json`
(+ the hand-written `corpus/POLICY.md`). A private id/originals map is also
written INTO staging (`$AP_STAGING/deid_map/`) for the failures-curation
child; it is never tracked.

## De-identification transform

Every private text field — node `content`, `context`/`ambient_context` string
values, `query` text, `task` labels, `agent` names, `session_id` labels,
`decay_reason`, correction `old`/`new` texts — is replaced by a surrogate
preserving exactly three invariants:

1. **Exact character length** (`len` in Python characters).
2. **Whitespace positions**: space, tab, newline and carriage return are kept
   verbatim at their original positions; every other character (including
   exotic Unicode whitespace and control characters) becomes a pseudo-random
   lowercase `a-z` letter, so every surrogate matches `^[a-z \t\n\r]*$`.
3. **Content-equality classes**, global across the whole build: identical
   originals map to identical surrogates, distinct originals to distinct
   surrogates (collisions between short distinct originals are re-derived
   with an attempt counter until unique).

These are precisely the properties `living_memory.delivery.shape_recall_results`
depends on: twin dedup compares `node.content` strings byte-for-byte
(equality classes), snippeting/preview classification cuts by character
length, and `_one_line_preview` splits on the first newline (whitespace
positions). Per-field character counts of the originals are additionally
recorded (`*_chars`; for context maps per top-level key with delivery
semantics — `len` for strings, `len(json.dumps(value, ensure_ascii=False,
default=str))` for lists/objects, mirroring `delivery._json_chars`).

Surrogate letters derive from HMAC-SHA256 over the original string keyed by an
**ephemeral salt** generated at build time (`os.urandom(32)`) and **not
recorded anywhere** — the mapping is non-invertible once the build exits.
The transform is deterministic *within* one build (fixed salt, fixed
processing order). **Byte-level corpus identity is guaranteed by the frozen
tracked files** (hash-pinned in the packet manifest), **not by re-runs**: a
rebuild draws a fresh salt and yields different letters with identical
lengths, whitespace, equality classes — and therefore identical metrics; all
packet metrics are salt-independent.

### Kept real (never surrogated)

ULIDs and event/node/trace ids, `results` candidate arrays verbatim (ranks,
per-method scores, node ids, levels, scopes, methods, graph paths of ULIDs),
scope names (values from the staged scope vocabulary; also
`requested_scope`/`resolved_scopes`), ISO timestamps, scores, levels, flags
(`feedback_applied`, `decayed`, `max_results`, `depth` parameter enums),
per-field char counts, context **key names**, `transport_session_id`
(validated opaque hex32), node stats (`access_count`, `usefulness_score`,
`confidence`, `unique_agents`, `last_accessed`, pattern-validated
`temporal_hint`), and the recognized automatic-recall template fingerprints:
`class` (`automatic` iff `agent IS NULL`, else `organic`) and `template_id`
(query contains `reopen_lesson` / `architectural_decision`), both computed
from the originals via the METADATA-pinned predicates. Inside context trees,
generic string leaves are kept only when they match a kept-real class (known
scope name, ULID, UUID, digit-bearing hex id, ISO timestamp, pure number,
`true/false/yes/no`); everything else is surrogated.

### Dropped (neither kept nor surrogated)

- `content_fingerprint` — an unsalted content hash would invite dictionary
  attacks; replay does not use it.
- node `provenance` content — `prior_recalls[].query` is raw private text;
  the corpus records `provenance_shape` instead: per-key kind/count/chars and
  the `serialized_chars` of the full provenance dict exactly as
  `resources.node_to_dict` ships it (merged with `source_traces` and
  `corrections`), so payload accounting stays exact.
- connection `metadata` — relations carry type/direction/other_id/weight/
  created_at only.
- (from extract already) node `embedding` vectors.

### Known micro-divergences, by design

- The `". "` truncation boundary of `delivery._truncate_at_boundary` is not
  preserved (a period becomes a letter): a replayed snippet may cut at a
  different clean boundary within the same window. Snippet/full/twin/
  session-duplicate *classification* depends only on the three preserved
  invariants.
- `json.dumps` char counts of surrogated values differ where originals
  contain `"`, `\` or non-`\t\n\r` control characters (escape sequences
  shrink to one letter). The recorded original `*_chars` values are
  authoritative where exactness matters.
- `str.strip()` behavior differs for exotic Unicode whitespace (surrogated to
  letters). Space/tab/newline/CR behavior is exact.

## Corpus record design

Two record types per split file, nodes first (sorted by id), then events
(sorted by `created_at, id`); every file is self-contained — it embeds every
node its events reference via `results[].node_id` or `feedback_trace_id`.

`{"type":"event",...}`: all recorded recall_event fields — real `id`,
`source` (`alt`|`local`), `created_at`, scopes, `depth`, `max_results`,
verbatim `results` with per-candidate scores, feedback fields
(`feedback_applied`, `feedback_trace_id`, `feedback_applied_at`); pinned
`class` + `template_id`; surrogate+chars for `query`/`agent`/`task`/
`session_id`; `ambient_context_surrogate` map + per-key chars; recomputed
`supersedes_bearing` flag; `transcript_matched` +
`transcript_serialized_chars` (the pinned `len_json_content` measure from
`transcripts/matched.jsonl`, alt events only).

`{"type":"node",...}`: real `id`/`level`/`scope`/timestamps/`decayed`/stats/
`source_traces`; `included_via` (`results`, `feedback`,
`typed_edge_partner`, `consolidation_evidence`); surrogate+chars for
`content`/`agent`/`task`/`decay_reason`; `context_surrogate` + per-key
chars; `corrections` with structure kept and `old`/`new` surrogated (+
`corrections_chars`); `provenance_shape`; `relations` — all
supersedes/contradicts edges among nodes of the same file (direction
`out`/`in` per endpoint).

Node inclusion per file: event-referenced nodes, plus one hop of typed-edge
partners (supersedes/contradicts, alt DB only — the extract stage exported no
local-DB connections, so local nodes carry no relations), plus — in dev and
eval only — the mixed-era consolidation evidence pack (`project:game`
`level=schema` nodes and their source-trace nodes, `included_via:
consolidation_evidence`).

## Pre-declared splits

Declared in advance by rule, recorded verbatim in `splits.json` — never by
inspection:

- `source=local` (all local-DB non-animal-planet workloads:
  `project:octopus`, `project:online`, `project:x`) → **holdout** only.
- `source=alt`: `bucket = int.from_bytes(sha256(event_id)[:8], 'big') % 100`;
  `bucket < 15` → holdout (~15% holdout-AP slice); `15 <= bucket < 66` → dev
  (51/85 = 60% of the remainder); else eval (34/85 = 40%).

Nothing is stratified by outcome. The only allowed check (documented in
`splits.json.family_check`): correction/supersedes-bearing events — events
whose `results` reference an endpoint of a supersedes connection — must land
in both dev and eval; with the declared constants the check held (dev 406,
eval 256), so no boundary adjustment was needed. `splits.json` records the
rule text, hash constants, per-split/per-scope/per-class/per-template counts,
window bounds (bounds only — the METADATA window `*_source` annotations quote
private node text and are deliberately excluded), and the sha256 of each
split file.

## Self-checks (`verify_corpus.py`, all must pass)

1–11: schema + byte re-serialization identity; kept-real fields equal staging
byte-for-byte; every `*_chars` equals the original's length; surrogate
length/whitespace/purity per field; global equality classes
(functional + injective, jointly across files); pinned class/template
predicates recomputed; split rule recomputed, ids disjoint and exhaustive
over the staged export, splits.json counts and hashes fresh; per-file
self-containment; ≥2 non-AP project scopes in holdout; family check;
transcript match flags/chars.

12–13, no-leak evidence: sampled ≥12-char n-grams of all staged private text
(field originals + raw staging lines) are searched across the corpus files;
hits are adjudicated positionally against a sentinel-aligned copy — windows
overlapping surrogate spans by ≤3 characters are structural coincidences
(JSON syntax + kept-real key names + up to three pseudo-random letters,
p ≤ 26⁻¹…³ per site; the observed k-histogram decays ≈ ×1/26 per extra
letter), any ≥4-letter overlap or unexplained hit fails. A residual ≥12-char
leak below that bar would need ≥9 characters carried by non-surrogate
material, which checks 1–11 audit field-by-field. Separately, EVERY 24-gram
of the authored tracked files (splits.json, POLICY.md, this file, the
transform code) is searched exhaustively against the full staging text; hits
must be fully explained by kept-real vocabulary (field/key names, scope
names, template fingerprints, timestamps). Leak diagnostics, if any, are
written to the private staging area only.

`verify_corpus.py` touches `holdout.jsonl` only mechanically (parsing and
invariant recomputation) and never prints record content — see
`corpus/POLICY.md` for the holdout seal.

## Private id/originals map (staging only)

`$AP_STAGING/deid_map/surrogates.jsonl` holds one `{surrogate, original}`
pair per unique de-identified string; corpus record ids are real, so staging
exports resolve any record back to raw text by id. `build_info.json` records
the build timestamp, corpus hashes and counts — and, deliberately, no salt.

## Re-run caveats

Staging is reproducible from the live sources per 01-extract.md (window-bound
counts stable; snapshot hashes drift). Re-running the transform on the same
staging preserves every count, split assignment, id, structure and metric,
but re-randomizes surrogate letters (fresh salt). The frozen packet is the
tracked files themselves, hash-pinned by the later manifest section.
