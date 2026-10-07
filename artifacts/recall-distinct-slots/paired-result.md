# Frozen paired MCP result

The unchanged ten-node synthetic corpus and three fixed queries have SHA-256
`64ba90bf6183751e3acd3b3d0739b7b5de96ba795a26ae0f550fb44063ab5742`.
The baseline is `baseline.json` (commit `21e5a08`); the candidate is
`candidate.json`. Each arm uses a fresh SQLite store and the normal MCP handlers.
The reported elapsed time measures the calls after corpus setup and chunk
seeding, not model loading or agent reasoning. The baseline is a frozen host
measurement; the candidate now pins the hash encoder and uses separate fixed
seconds for creation and access. The recorded milliseconds are from different
encoder environments and cannot establish a speedup. Call and byte counts
measure the complete responses required by this fixture.

| Required work for the inventory question | Baseline | Candidate |
| --- | ---: | ---: |
| Recognized repeat slots in `max_results=2` | 1 | 0 |
| Independent inventory fact in first answer | no | yes |
| Necessary recall calls | 2 | 1 |
| Necessary lookup calls | 0 | 0 |
| Complete response plus lookup bytes | 4,790 | 1,869 |
| Necessary call execution time, ms | 29.037 | 2.660 |

Baseline needs the fixed `max_results=3` reference call after the first
answer; the candidate does not. The candidate's first answer is `survey_a` and
`inventory`. The independent inventory fact is 70 degrees from the survey
vectors and is full content. The superseded `old_rule` stays suppressed and
the corrected `new_rule` stays first in the distinction query. The Monday
condition, opposite beacon polarities, and PX-17/PX-18 permit identifiers
remain distinct. The candidate distinction response is 3,734 bytes, versus
3,752 in the frozen baseline. The 18-byte difference comes from the setup
recall's encoder-dependent prior access scores, not a lost fact or field.
The fixed fixture clock keeps `updated_at` in the complete wire response
whether the run crosses a real second or not.
The reference query still returns both Tuesday paraphrases and inventory, so
the original repeat's identifier remains available. Direct MCP `memory_lookup`
on that original id returns its complete content, as the targeted test checks.

For all necessary user work here (inventory question plus the fixed distinction
query), response bytes fall from 8,542 to 5,603. The recorded timed sums are
41.796 and 5.320 ms, respectively, but are not comparable speed measurements.
The fixed reference query is a diagnostic that both arms
run, but it is not a necessary second call for the candidate. On that
diagnostic, the candidate returns the same three ids and delivery classes at
2,773 bytes. Event result ids match the actual selected ids; explicit feedback
on a delivered id is accepted. A subsequent same-session recall can still
return that id as `session_duplicate` for intentional refetch.

The change applies the existing `near_dup` map to a bounded extra candidate
window before access and event writes, only when a repeat can be replaced by
an independent result. Its identifier, length, and direct-bearer safeguards
are unchanged. A score-gated cut stays on its existing path, since its
residual may contain rejected candidates. No live memory was changed and this
is not a claim that the four distinct private-commit snippets in supervisor
event `01M4AFBNW2DQAQB70XRCB8DRC3` are collapsible.

Targeted verification: `env PYTHONPATH=src python3 -m pytest -q tests/test_recall_distinct_slots.py
tests/test_recall_near_dup_delivery.py` passed (37 tests). Post-publication
loaded-code and real MCP consumer verification belongs to the parent under
`publication-method.md`; this worktree measurement is not that receipt.
The candidate runner also passed with
`env PYTHONPATH=src python3 tests/fixtures/recall_distinct_slots.py --candidate`:
it reruns the frozen normal-MCP path and compares all stable result and cost
fields with `candidate.json`, while allowing measured time to vary.
The focused test repeats that byte comparison with ambient `hash` and `auto`
settings across a real second boundary; an isolated empty HOME run also
passed with the same 1,869/2,773/3,734 response-byte totals.
