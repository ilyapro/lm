# Recall effect by day — alt (2026-09-07..2026-09-27)

- DB: `/tmp/lm-alt-effect-snapshot.sqlite3` opened as `file:/tmp/lm-alt-effect-snapshot.sqlite3?mode=ro`; generated 2026-09-27T11:18:33Z.
- Used = recall_credit_ledger row for (event, node) with basis in ['grounded', 'lookup']. Shares are over events with ≥1 delivered node.
- Ledger bases present: grounded, lookup; explicit credit present: False; recall_feedback_marks present: False.
- A/B exclusion per transport session. Receipt ids scanned: 1604 (matched sessions in window: 0). Fixture task families from harness corpus manifests: development-battery, development-intake, development-ledger, development-packaging, development-transfer, development-water, ev-calibration, ev-flood, ev-labels, ev-parcel, ev-recovery, ev-source-malformed, ev-source-missing, ev-source-unreadable, ev-stock, ev-timetable.

## Window total (kept traffic)

Events 4675, kept 4675, excluded 0. rank-1 used 11.8%, top-3 used 22.9%, useful nodes 11.6% (2187/18837).

## Daily metric (kept traffic)

| day | events | excl | kept | closed | never | r1 used | top3 used | useful nodes | grounded r1/top3/nodes | lookup r1/top3/nodes | explicit r1/top3/nodes | legacy-filter r1 / useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 2026-09-07 | 630 | 0 | 630 | 580 | 50 | 7.6% | 15.7% | 7.7% | 6.2% / 10.6% / 3.1% | 1.4% / 5.4% / 4.6% | 0.0% / 0.0% / 0.0% | 7.6% / 7.7% |
| 2026-09-08 | 1766 | 0 | 1766 | 1448 | 318 | 12.1% | 23.2% | 11.7% | 10.3% / 15.6% / 6.0% | 1.8% / 8.6% / 5.7% | 0.0% / 0.0% / 0.0% | 11.9% / 11.9% |
| 2026-09-09 | 1067 | 0 | 1067 | 983 | 84 | 12.3% | 23.6% | 12.1% | 8.7% / 15.2% / 5.8% | 3.6% / 9.5% / 6.3% | 0.0% / 0.0% / 0.0% | 12.9% / 12.4% |
| 2026-09-10 | 24 | 0 | 24 | 23 | 1 | 16.7% | 37.5% | 12.9% | 8.3% / 16.7% / 4.3% | 8.3% / 25.0% / 8.6% | 0.0% / 0.0% / 0.0% | 16.7% / 12.9% |
| 2026-09-11 | 12 | 0 | 12 | 11 | 1 | 8.3% | 33.3% | 14.6% | 8.3% / 25.0% / 6.8% | 0.0% / 8.3% / 7.8% | 0.0% / 0.0% / 0.0% | 8.3% / 14.6% |
| 2026-09-12 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-13 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-14 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-15 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-16 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-17 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-18 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-19 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-20 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-21 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-22 | 0 | 0 | 0 | 0 | 0 | — | — | — | — / — / — | — / — / — | — / — / — | — / — |
| 2026-09-23 | 6 | 0 | 6 | 5 | 1 | 0.0% | 16.7% | 1.9% | 0.0% / 0.0% / 0.0% | 0.0% / 16.7% / 1.9% | 0.0% / 0.0% / 0.0% | 0.0% / 1.9% |
| 2026-09-24 | 734 | 0 | 734 | 696 | 38 | 13.4% | 23.8% | 12.0% | 11.8% / 19.1% / 7.8% | 1.5% / 5.5% / 4.2% | 0.0% / 0.0% / 0.0% | 13.7% / 12.1% |
| 2026-09-25 | 144 | 0 | 144 | 140 | 4 | 11.8% | 24.3% | 14.1% | 10.4% / 20.1% / 8.3% | 1.4% / 4.9% / 5.9% | 0.0% / 0.0% / 0.0% | 12.1% / 14.4% |
| 2026-09-26 | 167 | 0 | 167 | 158 | 9 | 13.8% | 34.1% | 18.1% | 13.8% / 25.1% / 7.1% | 0.0% / 11.4% / 11.1% | 0.0% / 0.0% / 0.0% | 13.8% / 18.1% |
| 2026-09-27 | 125 | 0 | 125 | 118 | 7 | 12.0% | 23.2% | 9.6% | 12.0% / 16.8% / 4.7% | 0.0% / 7.2% / 4.9% | 0.0% / 0.0% / 0.0% | 11.6% / 9.6% |

## Closed vs never-closed (kept traffic)

| day | closed n | closed r1 | closed top3 | closed useful | never n | never r1 | never top3 | never useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-07 | 580 | 7.9% | 16.6% | 7.9% | 50 | 4.0% | 6.0% | 3.7% |
| 2026-09-08 | 1448 | 14.5% | 26.9% | 13.1% | 318 | 1.3% | 6.0% | 3.6% |
| 2026-09-09 | 983 | 13.0% | 25.2% | 12.6% | 84 | 3.6% | 4.8% | 3.2% |
| 2026-09-10 | 23 | 17.4% | 39.1% | 13.2% | 1 | 0.0% | 0.0% | 0.0% |
| 2026-09-11 | 11 | 9.1% | 36.4% | 15.3% | 1 | 0.0% | 0.0% | 0.0% |
| 2026-09-12 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-13 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-14 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-15 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-16 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-17 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-18 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-19 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-20 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-21 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-22 | 0 | — | — | — | 0 | — | — | — |
| 2026-09-23 | 5 | 0.0% | 20.0% | 2.0% | 1 | 0.0% | 0.0% | 0.0% |
| 2026-09-24 | 696 | 14.1% | 25.0% | 12.5% | 38 | 0.0% | 2.6% | 1.6% |
| 2026-09-25 | 140 | 12.1% | 25.0% | 14.3% | 4 | 0.0% | 0.0% | 0.0% |
| 2026-09-26 | 158 | 14.6% | 34.8% | 18.4% | 9 | 0.0% | 22.2% | 13.3% |
| 2026-09-27 | 118 | 12.7% | 24.6% | 9.9% | 7 | 0.0% | 0.0% | 0.0% |

## Leak check 2026-09-23

Events on 2026-09-23 excluded by the new filter that the legacy filter (scope project:target|repo|x + ledger/billing/tree-context/fixture/kit acceptance) let through: **0** in 0 sessions; by reason {}.

| filter | kept events | r1 used | top3 used | useful nodes |
|---|---:|---:|---:|---:|
| legacy (scope/keywords) | 6 | 0.0% | 16.7% | 1.9% |
| this script (session evidence) | 6 | 0.0% | 16.7% | 1.9% |


## Exclusions per day

| day | excluded events | sessions | by reason (events; a session may carry several) | legacy would drop | legacy leak | legacy over-exclude |
|---|---:|---:|---|---:|---:|---:|
| 2026-09-07 | 0 | 0 | {} | 52 | 0 | 52 |
| 2026-09-08 | 0 | 0 | {} | 158 | 0 | 158 |
| 2026-09-09 | 0 | 0 | {} | 122 | 0 | 122 |
| 2026-09-10 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-11 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-12 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-13 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-14 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-15 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-16 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-17 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-18 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-19 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-20 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-21 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-22 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-23 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-24 | 0 | 0 | {} | 24 | 0 | 24 |
| 2026-09-25 | 0 | 0 | {} | 4 | 0 | 4 |
| 2026-09-26 | 0 | 0 | {} | 0 | 0 | 0 |
| 2026-09-27 | 0 | 0 | {} | 4 | 0 | 4 |

### Excluded sessions (up to 15 per day, leaked-by-legacy first; full list in the JSON)

## Legacy over-exclusion sample (kept by the new filter)

- 2026-09-07 `33af79b539f1` project:game: nectomite terminal nautilex receipt predecessor committed ledger terminal digest binding
- 2026-09-07 `a0ec1735bd50` project:game: grand-world entry-result binding aliases selfZeroedField canonical proto validation testing fixtures
- 2026-09-08 `deb75a5dc2d3` project:game: biome-successors parent tests read-only scope server fixtures physics beast-runs operator authorizat
- 2026-09-08 `018a8a8dd9e4` project:game: server fixture activity ACT_CAP BEAST_WINDOW quiet water movement test scouting physics discoverFigh
- 2026-09-09 `819f8603d449` project:game: biome phase5 additional actor fixtures world-aquatic-motion-scene-repair 38ce2b33 parent native repl
- 2026-09-09 `c4e21d28b23f` global: grand-world successor slot08 admission-instrument-repair-v2 failure phase5 actor fixtures delivery i
- 2026-09-24 `b6c6b3e3195e` project:game: steamkite runtime-invariants-v3 contract field immutable changed baseline published ledger fresh nod
- 2026-09-24 `e448a11e77a2` global: physical bounds sulfur namespace controls private full history fixture master branch lineage four pr
- 2026-09-25 `2ff76c4c019a` project:game: full-history disposable fixtures runVerifier timeout malformed JSON lod semantic measurements protec
- 2026-09-25 `4523e0915f2c` project:game: slot07 common adapter canonical FIRST original legacy strict result source.common historical pins se
- 2026-09-27 `e6ce38fcf68a` project:game: one-sided reachability instrument transitive imports symlink byte-alias sweep self-test disposable f
- 2026-09-27 `a4ef33b41873` project:game: full history isolation directory sweep fixture unresolved foreign evidence use own namespace
