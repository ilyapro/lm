# Recall effect metric: baseline ("before"), 2026-09-07..2026-09-27

Script: `scripts/recall_effect_daily.py`. Tests: `tests/test_recall_effect_daily.py`.
Each host is its own field. The two stores are never merged.

| file | source |
|---|---|
| `sfx-before.md` / `.json` | sfx live store `~/.local/share/living-memory/global.sqlite3`, read as `file:...?mode=ro`, 2026-09-27 ~11:17Z |
| `alt-before.md` / `.json` | alt snapshot `/tmp/lm-alt-effect-snapshot.sqlite3` (sha256 `d89f3df778c5a5099edab7c4bed9de165136141f579a4ac49ac652f4f2ed55b6`). It was made on alt with the SQLite backup API from `file:/home/user/.local/share/living-memory/global.sqlite3?mode=ro` at 2026-09-27 13:11 alt local time and copied here with scp. The alt live DB was never opened for writing. |
| `after-procedure.md` | exact commands for the operator-gated "after" report |

27.09 is a partial day in both reports (data up to ~11:05Z).

## Definitions

- **used**: a `recall_credit_ledger` row for the (event, node) pair with basis `grounded` or `lookup`.
- **rank1_used / top3_used**: share of recall events (with ≥1 delivered node) whose rank-1 result / any of ranks 1–3 was used.
- **useful_node_share**: used delivered (event, node) pairs divided by all delivered pairs.
- Every block is split by `grounded`, `lookup` and `explicit`, and into closed (`feedback_applied=1`) vs never-closed events.
- **explicit** counts ledger rows with basis `explicit`, or rows in any `*explicit*credit*` side table. `marked_used` counts accepted `used` rows in `recall_feedback_marks`. Both appear automatically once the tables exist. The default `used` definition stays grounded+lookup, so before and after stay comparable. `--used-bases grounded,lookup,explicit` gives the combined view, and the `used_with_explicit` / `used_or_marked` blocks are always emitted.

## What leaked on 2026-09-23 and how it is caught now

The AE tree-context A/B harness (`ae/artifacts/tree-context-ab/live-20260923T*`, arms `baseline` and `candidate`) ran fixture goals against the **live** sfx LM through a recording proxy. Its recalls were made in `requested_scope='global'` (and some in `project:mm`), with queries that repeat fixture task text, e.g. "development-packaging routing accepted kits route only" and "ev-calibration reuse retired quadratic calibration fit". Neither the scope list `project:target|repo|x` nor the keywords ledger/billing/tree-context/fixture/kit acceptance match these queries, so 250 events in 213 sessions leaked into 23.09.

With the legacy filter, 23.09 shows r1 38.2%, top3 53.2% and useful nodes 42.3%. With the session-evidence filter it shows r1 5.6%, top3 16.1% and useful nodes 7.1%, which is in line with the neighbouring days.

The robust key is the harness's own receipt. Every capture header `**/capture-memory-*/NNNNNN.headers.json` records the `mcp-session-id` response header, and that value equals `recall_events.transport_session_id`. 1604 ids were found under `ae/artifacts`, and 925 sessions in the window match (17, 19, 20 and 23.09). All 250 leaked events on 23.09 are receipt-proven, and 228 of them are independently caught by fixture provenance.

Exclusion is per transport session: one piece of evidence excludes every event of that session. The layers are listed from strongest to weakest, and the report shows each layer's count per day:

1. `receipt`: the session id is in the harness capture headers (or in a `--receipt-ids` file).
2. `fixture_provenance`: a node written in the session has `context.fixture`, `context.project=target`, a harness `context.run` (`*--live--r3*`, `live-c-r3`, `tree-context-ab-*`), or the "AE fixture" content marker.
3. `fixture_task`: the event task or node task/node_path starts with a fixture goal path from the harness `corpus_manifest.json` files (non-audit splits: development-*, ev-*). This catches the 16–17.09 baseline-evidence runs, which predate the capture proxy.
4. `synthetic_scope`: `project:_chat_inject_*` scopes from AE chat end-to-end tests (223 events on 16.09).
5. `fixture_query_family`: never-closed sessions without a node, where every query names a manifest fixture family as a whole token.
6. `fixture_scope`: `project:target`, but only within 30 min of proven harness traffic. Live supervisors also use `project:target` (18, 19 and 26.09), so the scope alone is not enough.

The legacy filter is computed only for comparison. It misses harness traffic (`legacy leak` column: 331/217/28/54/3/250 events on 16/17/19/20/22/23.09). It also drops live work every day (`legacy over-exclude`, 2–146 events/day, mostly `project:x`, `project:repo` and queries about ledgers or fixtures during real development).

Residual risk: a harness session is still missed if it has no receipt, wrote no node, has no fixture task, and uses at least one query that names no fixture family. Future harness runs that keep the capture proxy are fully covered by layer 1.

alt has no harness traffic: 0 capture directories under alt's `~/p/ae/artifacts`, 0 fixture-provenance nodes, and scopes `project:game`/`project:ae`. Nothing is excluded there. alt's store has **no recall events and no nodes from 2026-09-12 to 2026-09-22** (only 6 on 23.09), so its baseline covers 07–11.09 and 23–27.09.

## Baseline numbers (kept live traffic)

| host | window | events with results | r1 used | top3 used | useful nodes | never-closed |
|---|---|---:|---:|---:|---:|---:|
| sfx | 07–13.09 | 3476 | 8.7% | 19.9% | 11.2% | 19.3% |
| sfx | 14–20.09 | 3310 | 9.0% | 20.6% | 8.9% | 22.4% |
| sfx | 21–27.09 | 3107 | 5.1% | 16.7% | 8.0% | 28.1% |
| sfx | 24–27.09 | 1746 | 3.3% | 16.2% | 9.5% | 26.3% |
| sfx | 07–27.09 total | 9893 (2604 of 12497 events excluded) | 7.7% | 19.1% | 9.3% | 23.1% |
| alt | 07–11.09 | 3499 | 11.4% | 22.1% | 11.1% | 13.0% |
| alt | 23–27.09 | 1176 | 13.0% | 25.3% | 12.9% | 5.0% |
| alt | 07–27.09 total | 4675 (0 excluded) | 11.8% | 22.9% | 11.6% | 11.0% |

These numbers are over all kept events, both closed and never-closed. The operator's 27.09 figures (r1 3–6% on 24–27.09, useful 9–13%) were computed on closed events only. The closed-only daily view is in each report's "Closed vs never-closed" table. The sfx before-reports contain no `explicit` basis and no `recall_feedback_marks` table.
