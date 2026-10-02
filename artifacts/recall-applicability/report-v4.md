# Recall applicability v4

P1: PASS; 2 improved independent groups; 0 new fact losses; 15 baseline misses.

| Holdout category | Baseline reached / required | Candidate reached / required | Irrelevant top four (baseline → candidate) | New losses | Baseline misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| absent-knowledge | 0 / 0 | 0 / 0 | 8 → 8 | 0 | 0 |
| compound | 6 / 6 | 6 / 6 | 6 → 6 | 0 | 0 |
| concrete | 2 / 4 | 2 / 4 | 7 → 7 | 0 | 2 |
| correction | 3 / 5 | 5 / 5 | 7 → 6 | 0 | 2 |
| cross-project | 3 / 6 | 3 / 6 | 7 → 7 | 0 | 3 |
| instruction | 3 / 6 | 6 / 6 | 7 → 6 | 0 | 3 |
| large-group | 0 / 5 | 0 / 5 | 8 → 8 | 0 | 5 |

Per-category call, lookup, necessary prefix, exhausted chain, byte, latency and production complexity aggregates are in the JSON report.
P4 full-suite success is established by the parent, not this report.

Development new fact losses: 0; baseline misses: 3.
Protected baseline-reached facts: correction=3, cross-project=3, instruction=3
