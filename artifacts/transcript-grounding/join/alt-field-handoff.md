# Alt-field operator handoff — transcript-grounding phase 1 join

Goal `transcript-grounding`, phase 1 (corpus join), **alt field**. This pack is
self-sufficient: everything needed to run the join on the alt host is in this
file.

> **Status: executed 2026-08-29.** This is no longer a blind handoff — the
> procedure below is the one that actually ran, with the three things the
> first execution changed folded in (a mandatory `--verify --report` in step 4,
> the new `--self-verify-report` flag in step 5, and the mandatory
> `selection_bias` measurement in step 6). What it measured is recorded in
> section 8. The copy set in step 2 and the python3-only snapshot protocol in
> step 3 ran unmodified and are confirmed correct.

What the run produces: `coverage-alt.json` and `match-table-alt.jsonl.gz` —
per-tier coverage of recall_events against the alt post-session transcript
corpus, with the phase-1 60% falsifier evaluated **for the alt field on its
own** (each field gets its own verdict; the local verdict does not transfer).

Hard boundaries:

- The **live** alt store is never opened by anything in this protocol — not
  even read-only. The runner only ever sees a pinned snapshot copy.
- The local sealed manifest `artifacts/post-session/corpus.json` is **never**
  used on the alt host. The alt field gets its own freshly built
  manifest+index at NEW paths (step 3).
- A `falsifier_60pct.verdict` of `"fail"` is a valid, recorded measurement.
  Do not tune inputs to flip it.

Host facts (prior art, traces `01M1577PRQ54C8BMT66DZJJNDP` and
`01KZW27BVVSZHZFH3J73PB4RE3`): alt store at
`/home/user/.local/share/living-memory/global.sqlite3` (WAL mode — `-wal` /
`-shm` siblings may exist), transcripts under `/home/user/.claude/projects`,
AE project at `/home/user/p/ae`, **no sqlite3 CLI on the alt host — python3
only** (every command below is pure python3/coreutils).

---

## 1. Preflight on the alt host

```sh
python3 --version
# corpus builder needs >= 3.11 (tomllib, typing.Self, datetime.UTC);
# the join runner alone runs on >= 3.8. If python3 < 3.11 stop and report.

ls -la /home/user/.local/share/living-memory/global.sqlite3*
ls /home/user/.claude/projects | head
df -h /home/user     # need free space >= the store trio (db + wal)
```

The corpus builder also scans, when present: `~/.codex/sessions`,
`~/.gigacode/projects`, `~/.deepseek/sessions`, and
`<ae-root>/projects/*/state/codex_home/sessions`. Absent roots are fine —
they just contribute nothing.

Confirmed on the 2026-08-29 run: python3 3.12.3, store 218,804,224 B with
live `-wal` (28 MB) and `-shm` siblings, 550 project dirs, 337 G free,
`/home/user/p/ae` present, `~/.codex/sessions` present, no `~/.gigacode` or
`~/.deepseek`. Wall clock end to end was minutes, not hours: the corpus build
took 11.5 s and the join 68 s.

## 2. Ship the tools

The builder is not a single file: `scripts/postsession_corpus.py` inserts
`<root>/src` into `sys.path` (root = parent of its `scripts/` dir) and imports
`living_memory.postsession.*`. The layout below is therefore mandatory and is
the proven-minimal copy set (verified by running both tools from exactly this
tree, nothing else, on a clean directory):

```
~/tj/scripts/postsession_corpus.py
~/tj/scripts/transcript_join.py
~/tj/src/living_memory/          # the whole package tree (.py files)
```

From the local repo checkout (replace `<alt>` with the alt ssh alias):

```sh
ssh <alt> 'mkdir -p ~/tj/scripts ~/tj/src ~/tj/work ~/tj/out'
scp scripts/postsession_corpus.py scripts/transcript_join.py <alt>:tj/scripts/
scp -r src/living_memory <alt>:tj/src/living_memory
```

Smoke-test on the alt host (offline, no store needed):

```sh
python3 ~/tj/scripts/transcript_join.py --self-test
# expected: self-test OK: 13 denominator events, 9 matched, every tier, both traps
python3 ~/tj/scripts/postsession_corpus.py --help   # proves the imports load
```

## 3. Pin the store snapshot (python3-only protocol)

Copy the db+wal+shm trio to a working location, checkpoint the **copy**,
SHA-pin it. The live files are only ever read by `cp`.

```sh
python3 - <<'PY'
import hashlib, os, shutil, sqlite3
SRC = "/home/user/.local/share/living-memory/global.sqlite3"
DST = os.path.expanduser("~/tj/work/snapshot.sqlite3")
for ext in ("", "-wal", "-shm"):
    if os.path.exists(SRC + ext):
        shutil.copy2(SRC + ext, DST + ext)
con = sqlite3.connect(DST)                     # read-write on the COPY only
con.execute("PRAGMA wal_checkpoint(TRUNCATE)") # fold the copied WAL in
row = con.execute("PRAGMA quick_check").fetchone()
con.close()
assert row == ("ok",), "quick_check failed: re-copy the trio and retry"
h = hashlib.sha256()
with open(DST, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print("snapshot sha256:", h.hexdigest())
PY
```

**Record the printed sha.** If `quick_check` fails (the live server was
mid-write during the copy), delete the copies and re-run the block.

The runner opens `--store` strictly via the sqlite3 `file:` URI with
`mode=ro` — it cannot write to it — but `--store` must still always point at
`~/tj/work/snapshot.sqlite3`, never at the live path. The runner writes the
snapshot's own sha256 into `coverage-alt.json` (`store.sha256`); after the
run, check it equals the sha pinned here — that proves the measured snapshot
is the pinned one.

## 4. Build the ALT corpus manifest + index (NEW paths)

```sh
python3 ~/tj/scripts/postsession_corpus.py --build \
    --home /home/user \
    --ae-root /home/user/p/ae \
    --manifest ~/tj/work/corpus-alt.json \
    --index ~/tj/work/corpus-index-alt.jsonl \
    --progress
```

- `--ae-root` is **mandatory** on the alt host: the builder's default is the
  local-only path `/home/sfx/p/ae`; omitting the flag silently drops every
  AE-side transcript (ae_chat, ae_node_result, AE-spawned codex sessions).
- `--manifest` / `--index` must be NEW paths. Never reuse or overwrite the
  local sealed `artifacts/post-session/corpus.json`.
- `--build` is read-only with respect to everything it scans; it writes only
  the two files named above.

Record the manifest sha256:

```sh
python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
    ~/tj/work/corpus-alt.json
```

Then re-verify the freshly built pair, **writing the report** — this step is
mandatory, not optional: the goal requires the alt manifest's drift on the
record, and step 5 feeds this JSON to `--self-verify-report`.

```sh
python3 ~/tj/scripts/postsession_corpus.py --verify \
    --manifest ~/tj/work/corpus-alt.json \
    --index ~/tj/work/corpus-index-alt.jsonl \
    --report ~/tj/work/verify-alt.json
```

Run it immediately before step 5, so what it measures is exactly the drift
between building the corpus and joining against it. Live transcripts grow
constantly, so a nonzero `append` count is normal and harmless (the join
reads only the sealed prefix); `rewrite` and `missing` counts are the ones
worth re-building for. The 2026-08-29 run verified 2983/2983 byte-identical
with zero drift in every category.

## 5. Run the join

Real `scripts/transcript_join.py --help`, quoted verbatim (every flag below
is verified to exist there):

```
usage: transcript_join.py [-h] [--store STORE] [--manifest MANIFEST]
                          [--index INDEX] [--out-dir OUT_DIR]
                          [--field-label FIELD_LABEL]
                          [--sealed-manifest SEALED_MANIFEST]
                          [--drift-report DRIFT_REPORT]
                          [--self-verify-report SELF_VERIFY_REPORT]
                          [--store-origin STORE_ORIGIN] [--self-test]

Join postsession-corpus transcripts to recall_events and measure per-tier
coverage of the silent share. Store is opened read-only (sqlite file: URI,
mode=ro); the index must hash to the index_sha256 of whichever manifest
--manifest was handed, so a freshly built self-sealed pair verifies like the
sealed one. Writes coverage-<field-label>.json and match-table-<field-
label>.jsonl.gz into --out-dir.

options:
  -h, --help            show this help message and exit
  --store STORE         sqlite store (a pinned snapshot on live hosts)
  --manifest MANIFEST   corpus manifest sealing --index (a fresh build, or
                        corpus.json)
  --index INDEX         corpus-index.jsonl rebuilt for this host
  --out-dir OUT_DIR     directory for the two output artifacts
  --field-label FIELD_LABEL
                        artifact suffix, e.g. 'local' or 'alt'
  --sealed-manifest SEALED_MANIFEST
                        sealed artifacts/post-session/corpus.json, compared
                        against and recorded only (never read for rows, never
                        relaxes the gate)
  --drift-report DRIFT_REPORT
                        postsession_corpus.py --verify --report JSON,
                        summarised into coverage.drift_vs_sealed_manifest
  --self-verify-report SELF_VERIFY_REPORT
                        postsession_corpus.py --verify --report JSON for the
                        manifest handed to --manifest itself, summarised into
                        coverage.drift_vs_own_manifest (how far the freshly
                        built corpus moved between build and join)
  --store-origin STORE_ORIGIN
                        live store path a --store snapshot was copied from,
                        for the record
  --self-test           run the offline synthetic fixture suite and exit (no
                        store needed)
```

The alt invocation:

```sh
python3 ~/tj/scripts/transcript_join.py \
    --store ~/tj/work/snapshot.sqlite3 \
    --store-origin /home/user/.local/share/living-memory/global.sqlite3 \
    --manifest ~/tj/work/corpus-alt.json \
    --index ~/tj/work/corpus-index-alt.jsonl \
    --self-verify-report ~/tj/work/verify-alt.json \
    --out-dir ~/tj/out \
    --field-label alt
```

`--sealed-manifest` and `--drift-report` are deliberately **absent** here.
They only describe a run's relationship to the *local* sealed corpus under
`artifacts/post-session/`, which — per the hard boundaries above — never
travels to the alt host. Omitting them is correct and is recorded as such:
`coverage-alt.json` gets `sealed_corpus.compared: false` and
`drift_vs_sealed_manifest.measured: false`, so the artifact states plainly
that the alt run makes no claim about the local seal rather than leaving a
reader to guess.

`--self-verify-report` is the flag that fills the gap those two leave. The
goal still wants the fresh alt manifest's drift on the record, and that
drift is against *itself* — the corpus as built in step 4 versus the corpus
as it sits on disk at join time — not against a local seal it has no
relationship to. It lands under its own key, `drift_vs_own_manifest`, so the
two questions are never conflated. `--store-origin` is optional and records
only the live path the step-3 snapshot was copied from.

Semantics to know:

- The runner refuses to start if the index does not hash to the `index_sha256`
  of the manifest handed to `--manifest` — the step-4 pair seals itself, and
  that is exactly what the gate checks. It does **not** require the pair to be
  any particular corpus, so the alt build passes it on its own terms; what the
  gate forbids is reading rows the handed manifest does not describe.
- Exit code 0 covers **both** falsifier verdicts (`fail` is a measurement,
  not an error); 2 means bad inputs; the run prints the verdict on stdout.
- Matching tiers, highest wins: `event_id_echo` (verbatim recall-event ULID
  found in the transcript's sealed prefix, time-window validated) →
  `transport_session` (inherit via shared `transport_session_id`) →
  `cli_session` (`session_id` vs index `cli_session_id`/`linked_session_ids`)
  → `time_cwd_window` (±300 s window + cwd equality, unique candidate
  required). At most `index.records` records are read per transcript, so the
  result is stable even though live transcripts keep growing.

## 6. Expected outputs

Both land in `--out-dir`:

`coverage-alt.json` — keys: `field_label`; `store.{path,sha256,origin,access}`
(`sha256` must equal the step-3 pin);
`manifest.{path,sha256,index_sha256,generated_at,sessions,is_sealed_corpus}`
(`sha256` must equal the step-4 recording; `is_sealed_corpus` is `null` on the
alt field, where no sealed manifest is handed);
`index.{path,sha256,rows,verified_against_manifest,gate,`
`equals_sealed_corpus_index_sha256}` (the last is `null` on the alt field);
`sealed_corpus` and `drift_vs_sealed_manifest` (both recording `compared` /
`measured` `false` on the alt field); `drift_vs_own_manifest` (the step-4
`--verify --report` summarised — `rows_checked`, `rows_byte_identical`,
`appended`, `rewritten`, `missing`); `denominator` (pinned definition: all
recall_events with `created_at` inside the corpus window `[min started_at,
max ended_at]` over the index rows actually used, embedded verbatim, and
never narrowed to events carrying a `transport_session_id`);
`coverage.{total_events,matched_events,share,by_method}`;
`fallback_precision` (blinded tier-3/4 audit against the tier-1/2 gold
subset; flagged low-confidence when the gold subset is under 300 events);
`selection_bias` (see below); `falsifier_60pct.{threshold,share,verdict}`;
`match_table.{path,rows,sha256}`; `rules`; `diagnostics`.

**`selection_bias` is not optional, and it decides as much as the share.**
The coverage number alone cannot tell a neutral join from one that
preferentially finds events that were already consumed — and on the local
field that is exactly what was hiding under it. The runner therefore reports
the consumed share (`recall_events.feedback_applied = 1`) `store_wide`, over
the `denominator`, among `matched` and `unmatched`, per tier under
`by_method`, and — the control that rules out matched events merely being
newer — inside each calendar month separately under `within_month_control`
(a month counts only when both its cells hold at least 30 events). A field
whose coverage is high *because* the join detects consumption does not
support the goal's downstream claim. Stores predating the column report
`measured: false` rather than a silent zero.

Two `diagnostics` keys carry the structural ceiling: `events_with_any_echo`
and `denominator_events_with_any_echo` count events whose id appears in any
transcript at all, before any time or ambiguity check. They separate "the
join is weak" from "the transcripts are gone" — if matched ≈ ceiling, the
join is already extracting everything the corpus physically contains and no
better tier can help.

`match-table-alt.jsonl.gz` — one row per denominator event, matched and
unmatched alike: `event_id, created_at, matched, method, corpus_session_id,
transcript_path, transcript_sha256, delivery_record_index`. No query text, no
node content — phase 2 needs `delivery_record_index` to slice the
post-delivery remainder and nothing else.

Alt-specific expectations (not errors):

- The prediction that `time_cwd_window` contributes ~0 held, and so did the
  same for `cli_session`: the 2026-08-29 run matched entirely on the first
  two tiers (`event_id_echo` 4257, `transport_session` 169, the other two 0).
  Note this does *not* mean tiers 3–4 are broken — the blinded audit shows
  `time_cwd_window` predicting 116 gold events at precision 1.0. They score
  zero because tier 1 had already claimed every event they could reach.
  Report `by_method` as measured.
- `diagnostics.missing_transcripts > 0` means index rows whose files vanished
  between steps 4 and 5; if the count is more than a handful, rebuild the
  manifest+index (step 4) and re-run.
- The falsifier is evaluated per-field: `falsifier_60pct.verdict` in
  `coverage-alt.json` is the alt field's own phase-1 verdict, independent of
  the local one.
- **The local field measured `"fail"`, and that is not a reason to doubt your
  own run.** Local result (`coverage-local.json`): 6798 / 59074 = `0.115076`,
  `by_method` `{event_id_echo: 6360, transport_session: 434, cli_session: 0,
  time_cwd_window: 4}`. The cause is structural, not a runner defect: only
  7471 of the 59074 local events echo their id in *any* transcript still on
  disk, because 85% of local events are anonymous (no `agent`, no
  `session_id`) and concentrated in May–June, while the rolling CLI window
  leaves no Claude transcript older than July. Where transcripts do exist the
  join is near-exact — 92.9% of `codex` events and 96.7% of `/root` events
  matched, and the blinded tier-3/4 audit scored precision 1.0 on a 6794-event
  gold subset. Expect the alt share to track the alt host's transcript
  retention the same way. Report what you measure; do not tune to 0.60.

## 7. Bring the artifacts home and commit

From the local repo checkout:

```sh
scp <alt>:tj/out/coverage-alt.json \
    <alt>:tj/out/match-table-alt.jsonl.gz \
    <alt>:tj/work/corpus-alt.json \
    <alt>:tj/work/verify-alt.json \
    artifacts/transcript-grounding/join/

git check-ignore artifacts/transcript-grounding/join/coverage-alt.json \
    artifacts/transcript-grounding/join/match-table-alt.jsonl.gz \
    || echo "not ignored — good"
git add artifacts/transcript-grounding/join/coverage-alt.json \
        artifacts/transcript-grounding/join/match-table-alt.jsonl.gz \
        artifacts/transcript-grounding/join/corpus-alt.json \
        artifacts/transcript-grounding/join/verify-alt.json
git commit -m "transcript-grounding phase 1: alt-field join coverage (falsifier: <verdict>)"
```

The manifest (`corpus-alt.json`, ~3 KB) and the verify report
(`verify-alt.json`, ~1.5 KB) are both committed: they are small, carry no
transcript rows, and are what lets phase 2 re-verify the alt seal without
re-running the build. The index itself (2.3 MB of absolute home paths) stays
on the alt host.

(`artifacts/transcript-grounding/join/` is verified not gitignored in this
repo; `git check-ignore` printing nothing is the expected outcome.)

Quote in the commit message or run notes: the snapshot sha256 (step 3), the
alt manifest sha256 (step 4), and the falsifier verdict. All three are also
embedded in `coverage-alt.json` itself.

Cleanup on the alt host after the artifacts are safely committed:
`rm -rf ~/tj/work` (the snapshot trio is large); keep `~/tj/scripts` +
`~/tj/src` for a re-run if phase 2 asks for one.

---

## 8. What the first execution measured (2026-08-29)

Provenance of that run, all three values also embedded in `coverage-alt.json`:

| | |
|---|---|
| snapshot sha256 | `9e7b5e2a95055b114fafa1c9e1e21ff4ced5c4ee178b699bcacc9df63968b0f1` |
| snapshot origin | `/home/user/.local/share/living-memory/global.sqlite3` (218,951,680 B after checkpoint) |
| alt manifest sha256 | `df722d7e2b5f719d32686efbe7ee4ab4c6580d7a8d3b181ee8bfa76e7141b74a` |
| alt index sha256 | `2a9a43c0c4ae2aa0a79afab7b390ede5530750f8cf934f2828be704d188ad7f4` (2983 sessions) |
| match table sha256 | `be7375c6e5185ff5988f961b93e8de91963928ad3e65393e9715aaa5de2973ee` (16098 rows) |
| corpus window (UTC) | 2026-07-26T20:08:14Z → 2026-08-29T08:24:57Z |

**Coverage: 4426 / 16098 = `0.274941` → `falsifier_60pct.verdict = "fail"`.**
The alt field fails the 60% falsifier on its own terms, as the local field
did at `0.115076`. Both fields now carry a recorded negative phase-1 verdict.

That number is *at the structural ceiling*, and this is the part that matters
more than the share. Only 4403 of the 16098 denominator events (`0.2735`)
have their id echoed in any surviving transcript at all, so the join at
`0.2749` — slightly above the ceiling, because tier 2 propagates to events
whose own id never echoed — is already extracting essentially everything the
corpus physically contains. Unlike the local field, this cannot be blamed on
transcripts rolling off disk: the corpus covers the whole window, 16098 of
the store's 16102 events fall inside it, `missing_transcripts` is 0, and the
corpus verified 2983/2983 byte-identical. Better join tiers cannot move this
number.

**The selection bias replicates on the alt field, more sharply than locally.**
Consumed share (`feedback_applied = 1`):

| population | events | consumed | share |
|---|---:|---:|---:|
| store-wide | 16102 | 2437 | `0.151348` |
| matched | 4426 | 2387 | `0.539313` |
| unmatched | 11672 | 50 | `0.004284` |
| tier `event_id_echo` | 4257 | 2317 | `0.544280` |
| tier `transport_session` | 169 | 70 | `0.414201` |

Matched events are **3.56x** enriched over the store-wide base rate (local:
3.8x) and **126x** over unmatched events. The within-month control holds in
both comparable months — 2026-07 matched `0.876` vs unmatched `0.010`,
2026-08 matched `0.511` vs unmatched `0.004` — so this is not matched events
merely being newer. The local diagnosis therefore reproduces on an
independent host: `event_id_echo` behaves closer to a consumption detector
than a neutral join, because the dominant reason an event id is written into
a transcript is the `memory_remember` provenance that consumes it.

One difference worth carrying forward: on the local field `transport_session`
sat at the store base rate (`0.1820` vs `0.1717`) and was the one arguably
unbiased tier. On alt it does **not** — `0.414201` against a `0.151348` base
rate. The transport tier inherits its session from an echo-matched event, so
on this field it inherits that event's bias too. Nothing in either field's
numbers supports treating any current tier as a neutral sample of the silent
remainder.

Both halves of the phase-1 two-field boundary have now been measured, and
both fail. Reopening the goal needs prospective transcript retention —
joining recalls while their transcripts are still on disk — not better join
tiers, and not another field.
