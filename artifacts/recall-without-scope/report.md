# Report: recall without `scope` searches the whole store (goal recall-without-scope-is-broad)

`before.json` / `after.json` come from `scripts/scopeless_recall_measure.py` run
against two fresh copies of one snapshot of the sfx store (taken 2026-09-29
~09:29Z with the SQLite backup API from a `mode=ro` connection; sha256 prefix
`8f1de3f0fd495e05`). `before` ran the code at `684d3e9`, `after` this branch,
both through the in-process MCP client with the live server's `LM_*` env (score
gate 0.35, drop). Each arm used its own MCP session, so session dedup does not
leak between arms. Only aggregates are committed; queries, node texts and scope
names stay out of the repo.

Cases: 40 recorded recalls that named a `project:*` scope and whose top result
was a live node of it (seeded sample); the same query is asked again with no
scope (`no_ambient`), with only an ambient `session_id` (`session_id_only`), and
with the target's own scope (`explicit_target_scope`, the restricted upper
bound). `reported` holds the three reported queries (two misses and the short
query that found the fact under an explicit scope). `negative` holds 10 seeded
nonsense queries.

| arm | target reached before → after | results delivered | outside target scope + global | p50 / p95 ms | mean response chars |
|---|---|---|---|---|---|
| no_ambient | 11 → **20** / 40 | 109 → 185 | 19 → 76 | 225 → 364 / 401 → 540 | 5924 → 10274 |
| session_id_only | 0 → **20** / 40 | 91 → 185 | 0 → 75 | 212 → 366 / 377 → 571 | 5461 → 10266 |
| explicit_target_scope | 26 → 26 / 40 | 175 → 175 | 0 → 0 | 245 → 245 / 412 → 422 | 9884 → 9915 |
| reported, no scope | 0 → 1 / 3 | 4 → 12 | 0 → 3 | 167 → 326 | 4103 → 7332 |
| reported, explicit | 1 → 1 / 3 | 15 → 15 | 0 → 0 | 241 → 247 | 8001 → 8001 |
| negative (nonsense) | — | 10 → 10 | — | — | — |

Reading:

- Scope-less recall now reaches 20 of the 26 facts the target's own scope
  reaches (before: 11 without ambient context, 0 with a `session_id`). The
  explicit arm is unchanged: an explicit scope restricts exactly as before.
- The nonsense controls deliver one result per query before and after -- the
  top result the score gate always keeps -- so the wider search adds no noise
  there. The extra delivered results on real queries all pass the same 0.35
  gate; 76 of 185 now come from projects other than the target's, which is
  the point of a scope-less search and also the price in context volume.
- The first reported query does not reach its fact in the top 5 under any
  scope: with that query the fact ranks 54th scope-less and outside the top 60
  under `project:ae`. That is ranking, not scope admission; the short query
  reaches it scope-less now.
- Cost: ~+140 ms p50 per scope-less call and ~+375 MB peak RSS in this
  process. The whole-store vector scan itself is ~5 ms; the largest named share
  is the steady-state "unchunked nodes" probe over every active node (~35 ms,
  3 ms for `global` alone), the rest is the larger candidate/residual pool
  (graph walk, recall map). Explicit-scope calls cost the same as before.
