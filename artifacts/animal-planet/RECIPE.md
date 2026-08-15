# Animal-planet audit & replay packet — end-to-end recipe

This is the umbrella recipe of the frozen animal-planet baseline packet: how
every artifact in this directory was produced, how to re-run the pipeline, and
how to verify the frozen state. Detailed per-stage documents:
[`recipe/01-extract.md`](recipe/01-extract.md) (evidence → private staging),
[`recipe/02-transform.md`](recipe/02-transform.md) (staging → de-identified
corpus), [`recipe/03-calculator.md`](recipe/03-calculator.md) (the baseline
calculator). The frozen state itself is pinned by
[`manifest.json`](manifest.json) — the immutable manifest written by the
freeze step; it records the SHA-256 of every tracked packet file (everything
except the manifest itself).

The packet is the evaluation contract of the `retrieval-signal-and-context-cost`
goal: the baseline every later Living Memory change is measured against, plus
the replay corpus that measurement runs on. The originally cited baseline
numbers (42.4% / 17.6% linkage, 26.5k median / 35.2k p90 payload chars, 727
W1 recalls, …) come from audit node `01KZK2WMP07CNTYDQXFTS23F39`
(scope `project:ae`), net-value trace `01KZVZ5ZTE1WSXFHCK6091H0RE` and payload
follow-up trace `01KZV6VCVGXVKPF6PSFXBH00EM` (scope `project:lm`).

## Packet layout

| Path | Role |
| --- | --- |
| `manifest.json` | Immutable source manifest: windows, provenance, predicates, per-source counts, SHA-256 of every packet file, discrepancies. |
| `RECIPE.md` | This document (hashed in the manifest). |
| `recipe/01-extract.md` + `recipe/extract/*` | Re-runnable extraction: live evidence → private staging. |
| `recipe/02-transform.md` + `recipe/transform/*` | De-identification transform + pre-declared splits. |
| `recipe/03-calculator.md` | Calculator contract (`scripts/ap_baseline.py` at repo root). |
| `corpus/{dev,eval,holdout}.jsonl`, `corpus/splits.json` | Privacy-safe replay corpus; holdout SEALED per `corpus/POLICY.md`. |
| `failures/` | 16 curated de-identified failure cases, 4 per family, dev+eval in every family. |
| `baseline-report.json` | Frozen baseline numbers + expected/tolerance table + explicit discrepancies. |

## Windows (inclusive both ends; each pins specific metrics)

| Window | Range | Pins |
| --- | --- | --- |
| W1 | `2026-08-06T20:00:00Z` … `2026-08-09T11:00:23Z` | 727 recall_events, 9 supersedes edges, deterministic pre-recall templates 152+152 (~42%), auto-OUTCOME trace count (cited 105 — discrepancy d2), avg 12.6KB/recall + 18.8% share (len_text basis — d4), LM call counts 202/141/46/8. |
| W2 | `2026-08-06T20:00:00Z` … `2026-08-12T21:48:45Z` | Organic/automatic linkage 42.4%/17.6% over `scope='project:game'` (1349 events; automatic 1188/209 exact; organic cited 165/70 vs measured 161/71 — d1). |
| W3 | `2026-08-09T11:34:00Z` … `2026-08-12T14:37:00Z` | Payload chars of matched recall tool-results: cited median 26,496 / p90 35,237 / mean 26,543 over n=278 (survivors 164 — d3). |
| export | `2026-08-06T20:00:00Z` … `2026-08-12T21:48:45Z` | Alt recall_events export range (= W1.start … W2.end, covers W1/W2/W3). |

W1.end = `created_at` of the audit node; W2.end = `created_at` of the
net-value trace; W3 bounds come from the payload follow-up trace (minute
precision). DB events are windowed by `recall_events.created_at`, transcript
events by the tool_result timestamp.

## Stage 1 — Extract (live evidence → private staging)

One command rebuilds the whole private staging dataset (see
`recipe/01-extract.md` for the full step map):

```bash
bash artifacts/animal-planet/recipe/extract/run_all.sh
```

Destinations (never tracked, never committed): `AP_STAGING`
(default `/home/sfx/.cache/ap-audit/staging/`) and `AP_TMP`
(default `/tmp/ap-audit/`). All sources are strictly read-only; nothing is
ever written on the evidence host `alt` and nothing there is stopped,
restarted or mutated.

### Alt LM DB snapshot (step 10)

```bash
ssh -o BatchMode=yes alt "stat -c '%n|%s|%Y' /home/user/.local/share/living-memory/global.sqlite3{,-wal,-shm}"
scp -q alt:/home/user/.local/share/living-memory/global.sqlite3     /tmp/ap-audit/alt/global.sqlite3
scp -q alt:/home/user/.local/share/living-memory/global.sqlite3-wal /tmp/ap-audit/alt/global.sqlite3-wal
scp -q alt:/home/user/.local/share/living-memory/global.sqlite3-shm /tmp/ap-audit/alt/global.sqlite3-shm
# on the LOCAL COPY only (alt has no sqlite3 CLI; python3 stdlib everywhere):
python3 -c "import sqlite3; c=sqlite3.connect('/tmp/ap-audit/alt/global.sqlite3'); \
  print(c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone(), \
        c.execute('PRAGMA integrity_check').fetchone())"
sha256sum /tmp/ap-audit/alt/global.sqlite3   # recorded in staging alt-db/snapshot.json
```

All subsequent queries open the checkpointed copy via a read-only URI:
`sqlite3.connect("file:/tmp/ap-audit/alt/global.sqlite3?mode=ro", uri=True)`.

### Alt DB export SQL (step 20, `20_export_alt_db.py`)

```sql
-- audit slice of recall events (params: export window = W1.start, W2.end)
SELECT * FROM recall_events WHERE created_at >= ? AND created_at <= ?
  ORDER BY created_at, rowid;                          -- 1,576 events
-- nodes referenced by those events (results[].node_id + feedback_trace_id)
SELECT * FROM nodes WHERE id IN (...);                  -- 807 nodes
-- all typed edges (the table is `connections`, column `type`)
SELECT * FROM connections WHERE type IN ('supersedes','contradicts')
  ORDER BY created_at, rowid;                           -- 93 + 85
-- plain `related` edges among exported nodes
SELECT * FROM connections WHERE type NOT IN ('supersedes','contradicts');
-- mixed-era consolidation evidence
SELECT * FROM nodes WHERE scope = 'project:game' AND level = 'schema'
  ORDER BY created_at;                                  -- 11 (9 by W1.end)
-- auto-OUTCOME candidates + ranking weights
SELECT * FROM nodes WHERE content LIKE '%OUTCOME%' ORDER BY created_at;
SELECT * FROM retrieval_weights ORDER BY scope;
```

Node `embedding` columns are excluded from every export.

### Transcripts (steps 30–40)

Inventory runs entirely over ssh (nothing written on alt); the mangled
project dir names start with `-`, so every remote file command needs `--`:

```bash
ssh -o BatchMode=yes alt 'for f in ~/.claude/projects/*animal-planet*/*.jsonl; do
  stat -c "%s|%Y" -- "$f"; sha256sum -- "$f"; done'   # 336 files, 126 dirs, ~465 MB
```

The stream parser is sent to alt on stdin so no file lands there:

```bash
ssh -o BatchMode=yes alt python3 - \
  < artifacts/animal-planet/recipe/extract/transcript_parser.py \
  > "$AP_STAGING/transcripts/lm_events.jsonl"
```

It extracts every `mcp__living-memory__*` tool_use/tool_result pair (702)
plus per-result size records for ALL tools (20,796).

**Transcript↔DB match rule** (step 70, recorded in staging `METADATA.json`):
recall tool_results with `is_error=false` join `recall_events` by exact query
string equality, nearest `|created_at − tool_result.timestamp| ≤ 600s`,
greedy one-to-one by ascending time delta (ties: DB id, transcript order).
366/366 transcript recalls matched (W1: 202/202).

**Serialized payload measure** (pinned): `len_json_content` =
`len(json.dumps(<content of the tool_result block>, ensure_ascii=False))`;
identical to `len(json.dumps(toolUseResult))` on this corpus. The 08-09
audit's "avg 12.6KB / 18.8% share" figures reproduce only on the `len_text`
basis (sum of text-block lengths over ALL W1 LM tool-results): 12,642.8 avg,
0.1876 share (discrepancy d4).

### Local LM DB (steps 50–60)

Same snapshot method with `cp` instead of `scp`
(`~/.local/share/living-memory/global.sqlite3` → `/tmp/ap-audit/local/`,
checkpoint the copy, `mode=ro`), then per holdout scope:

```sql
SELECT * FROM recall_events WHERE scope = ?
  ORDER BY created_at DESC, rowid DESC LIMIT 1500;
```

for `project:octopus` (12,127 total), `project:online` (7,976),
`project:x` (9,080) — three non-animal-planet workloads, holdout-only.

### Classification predicate & markers (pinned at extract, kept verbatim since)

- **Event class**: `automatic` iff `agent IS NULL`, else `organic`
  (verbatim `event_class_predicate` in `corpus/splits.json`). Linkage
  population: `scope='project:game'`, `created_at` in W2; linkage =
  share with `feedback_applied != 0` (domain here is {0,1}).
- **Auto-OUTCOME marker** (from alt `~/p/ae/node.sh`,
  `_node_lm_record_outcome`, goal cites line 1204): template
  `OUTCOME {pass|fail}: {task}[ — {goal.md first line}]`, written via
  `lm_client.py remember --agent ae` — these writes never appear in
  transcripts. Pinned SQL:
  `(content LIKE 'OUTCOME pass: %' OR content LIKE 'OUTCOME fail: %') AND
  agent='ae'`, node `created_at` in W1 → 78 (broad any-agent variant 89) vs
  cited 105 (discrepancy d2).
- **Deterministic pre-recall templates**: query contains `reopen_lesson` /
  `architectural_decision` (substring) → 152 + 152 in W1 (exact).

## Stage 2 — Transform (staging → de-identified corpus)

```bash
python3 artifacts/animal-planet/recipe/transform/build_corpus.py
python3 artifacts/animal-planet/recipe/transform/verify_corpus.py   # exit 0 iff clean
```

**De-identification invariants** (full spec: `recipe/02-transform.md`).
Every private text field (node `content`, context string values, `query`,
`task`, `agent`, `session_id`, `decay_reason`, correction `old`/`new`) is
replaced by a surrogate preserving exactly:

1. **exact character length**;
2. **whitespace positions** — space/tab/newline/CR verbatim at their
   original positions, every other character becomes a pseudo-random
   lowercase `a-z` letter (so surrogates match `^[a-z \t\n\r]*$`);
3. **global content-equality classes** — identical originals ↦ identical
   surrogates, distinct ↦ distinct.

These are precisely the invariants `living_memory.delivery.shape_recall_results`
depends on (twin dedup, snippeting, one-line previews). Original per-field
char counts are recorded alongside (`*_chars`). Kept real: ids/ULIDs,
`results` candidate arrays verbatim (ranks, per-method scores, scopes,
levels), scope names, timestamps, flags, node stats, `class`/`template_id`
fingerprints. Dropped entirely: `content_fingerprint`, node `provenance`
content (a structural `provenance_shape` with exact serialized char counts
replaces it), connection `metadata`, `embedding` vectors.

Surrogate letters derive from HMAC-SHA256 keyed by an **ephemeral salt**
(`os.urandom(32)`) that is **intentionally recorded nowhere** — the mapping
is non-invertible once the build exits. A rebuild draws a fresh salt: same
lengths, whitespace, equality classes, counts and metrics, different letters.
**Byte identity of the corpus is guaranteed only by the frozen tracked files
+ the manifest hashes, never by re-runs.**

**Split rule** (verbatim in `corpus/splits.json`, declared in advance, never
by inspection): `source=local` → holdout only;
`source=alt` → `bucket = int.from_bytes(sha256(event_id)[:8],'big') % 100`;
`bucket < 15` → holdout, `15 ≤ bucket < 66` → dev, else eval. The only
allowed check (`family_check`): correction/supersedes-bearing events must
land in both dev and eval — held (406/256) with no boundary adjustment.
Holdout is SEALED: see `corpus/POLICY.md` (no per-case inspection, no tuning;
first semantic read belongs to the post-freeze shadow-eval stage).

## Stage 3 — Calculator

```bash
python3 scripts/ap_baseline.py report                # recompute baseline-report.json
python3 scripts/ap_baseline.py verify                # gate the frozen packet (exit 0 iff clean)
python3 scripts/ap_baseline.py compare --split dev   # before/after replay JSON
```

`report` derives every metric mechanically from the corpus (cited constants
live only in the expected/tolerance table). Tolerances: linkage ±0.5 pp,
payload ±1%, counts exact. `verify` exits 0 iff the report reproduces
byte-identically, every metric is within tolerance **or covered by a recorded
discrepancy**, and the corpus hashes match `manifest.json`. `compare` replays
delivery (`shape_recall_results`) and a recorded-candidate rerank; BM25/vector
re-execution is out of scope (recorded candidate scores only). Details:
`recipe/03-calculator.md`.

## Stage 4 — Freeze (this packet state)

`manifest.json` records: `frozen: true` + freeze timestamp, the windows and
what each pins, per-source provenance (host, DB paths, snapshot SHA-256s +
capture times, transcript inventory count/bytes/digest, local-DB holdout
scopes/slices), the pinned predicates, per-source counts, the four carried
discrepancies, and the SHA-256 of **every tracked packet file** — corpus,
splits, POLICY, failures, recipe docs and code, baseline-report, and this
RECIPE.md. The only packet file without a recorded hash is `manifest.json`
itself (it cannot contain its own digest). The calculator's at-freeze hash is
recorded separately (`calculator` block): later bug-fixes to the *tooling*
may change that file, while packet files stay immutable.

Verify the frozen state at any time:

```bash
python3 scripts/ap_baseline.py verify        # exit 0 iff the packet is intact
# independent re-hash of every manifest-listed file (jq + sha256sum):
cd "$(git rev-parse --show-toplevel)" \
  && jq -r '(.files + .packet_files) | to_entries[] | "\(.value.sha256)  artifacts/animal-planet/\(.key)"' \
     artifacts/animal-planet/manifest.json | sha256sum -c --quiet && echo HASHES-OK
```

Corpus surrogate-purity re-scan (corpus-only, no staging needed): every
`*_surrogate` field must match `^[a-z \t\n\r]*$`, and every string leaf
inside surrogated containers (`context_surrogate`, `ambient_context_surrogate`,
`corrections`) must match it too unless it is a kept-real leaf per
`recipe/transform/deid.py::keeps_real` (scope names, ULIDs/UUIDs/hex ids,
timestamps, numbers, yes/no). Runnable via:

```bash
python3 - <<'PY'
import json, sys
from pathlib import Path
root = Path("artifacts/animal-planet")
sys.path.insert(0, str(root / "recipe" / "transform"))
from deid import PURITY_RE, keeps_real
FLAT = ("query_surrogate","agent_surrogate","task_surrogate","session_id_surrogate",
        "content_surrogate","decay_reason_surrogate")
scopes, records = set(), []
for split in ("dev","eval","holdout"):
    for line in (root/"corpus"/f"{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(line); records.append(r)
        for key in ("scope","requested_scope"): 
            if r.get(key): scopes.add(r[key])
        scopes.update(r.get("resolved_scopes") or [])
bad = 0
def walk(v):
    global bad
    if isinstance(v, str):
        if not (PURITY_RE.match(v) or keeps_real(v, scopes)): bad += 1
    elif isinstance(v, dict):
        for x in v.values(): walk(x)
    elif isinstance(v, list):
        for x in v: walk(x)
for r in records:
    for f in FLAT:
        v = r.get(f)
        if v is not None and not PURITY_RE.match(v): bad += 1
    for f in ("context_surrogate","ambient_context_surrogate","corrections"):
        if r.get(f) is not None: walk(r[f])
print(f"purity re-scan: {len(records)} records, impure leaves: {bad}")
sys.exit(1 if bad else 0)
PY
```

## Drift caveats (read before any re-run)

- **The sources are live.** The alt DB, its nodes/edges and the local DB keep
  mutating; transcripts appear and disappear. A re-extraction re-captures
  them: window-bound metrics (W1/W2/W3 all lie in the past) reproduce;
  full-table totals, all-time counts and snapshot SHA-256s drift and are
  re-recorded per capture. The frozen packet pins exactly one capture
  (2026-08-12/13) via `manifest.json`.
- **Byte identity is guaranteed only by the frozen tracked corpus + manifest
  hashes.** A transform re-run draws a fresh ephemeral de-id salt
  (intentionally unrecorded): identical structure, counts and metrics,
  different surrogate letters — so rebuilt files hash differently by design.
- **Unrecoverable populations stay unrecoverable.** The 114 `/root`-session
  W3 tool-results (d3) and the live-time auto-OUTCOME count 105 (d2) cannot
  be re-measured from surviving evidence; the packet records the survivor
  populations plus explicit discrepancy entries instead of patched numbers.
- **`feedback_applied` is a lower-bound proxy** for observed use, not
  ground-truth relevance (recorded verbatim in `baseline-report.json`).

## Hard rules

- Raw transcripts and SQLite databases are private read-only sources: never
  committed, never pasted into prompts. Staging (`$AP_STAGING`, `$AP_TMP`)
  lives outside every repo and is never tracked. Tracked files carry only
  aggregates, identifiers, lengths, hashes and surrogates.
- `alt` is read-only evidence: nothing there is stopped, restarted or
  mutated; remote parsing runs with the script on stdin.
- `corpus/holdout.jsonl` is sealed (`corpus/POLICY.md`).
- The files at the `artifacts/` repo root (`ANALYSIS.md`, `baseline.md`,
  `backfill/`, `discovery/`, `replay/`, `latency_*.json`, …) are a
  **2026-05-22 prior-era audit of the local DB**. They are not part of this
  packet and are never used as its evidence; this packet's evidence base is
  exclusively the captures listed in `manifest.json` (the extraction scripts
  reuse the *snapshot method* precedent documented there, nothing more).
