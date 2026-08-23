# Re-measure after the honest-bearer fix

`dup-slot-measurement.md` reported, under *Spot check*, that **8 collapsed slots
pointed at a bearer they never matched** — 4 distinct pairs whose cosine to the
node named in their `content_ref` was as low as **0.904395**, below the 0.95
collapse threshold and inside the 0.85–0.95 band the root goal describes as
holding *different* facts.

That was not a cost of the collapse, it was a defect in it.
`build_duplicate_map` resolved a match through a node that had itself become a
stub and recorded the pair against the chain's root. Cosine is not transitive,
so the resulting claim — "this slot repeats that node" — was one the function
never measured. The rule is now: **every recorded pair clears the threshold
against the bearer it names**; when only the intermediate stub matched, the scan
moves on and the candidate keeps its text.

## What changed, on the same frozen query set

Both runs replay the identical selection (`digest 19877a4383b2e2d2`, 2000
traffic-weighted + 500 distinct-query real `recall_events`), each against its own
read-only snapshot of the live database. Snapshots differ by a few hours of
traffic (13,451 vs 13,456 active nodes) — which is itself a control: the
`before_rollback` arm reproduces to the digit on both.

| | before fix (`3b335021b8de`) | after fix |
| --- | ---: | ---: |
| duplicate-slot share, rollback arm | 2.3917% | 2.3917% |
| duplicate-slot share, shipped default | 0.9507% | **0.9733%** |
| …distinct-query stratum | 3.26% → 1.55% | 3.26% → 1.63% |
| repeat slots occupied, before → after | 317 → 126 | 317 → **129** |
| collapsed slots | 256 | **250** |
| chars replaced by a stub | 571,519 | **517,521** |
| **pairs whose bearer is below threshold** | **4** | **0** |
| **slots whose bearer is below threshold** | **8** | **0** |
| min cosine to the recorded bearer | 0.904395 | **0.95043** |
| pairs flagged for human reading | 22 | 19 |
| pairs losing ≥3 identifiers (slots) | 14 (60) | **11 (54)** |
| token Jaccard, p25 / median | 0.5673 / 0.7333 | 0.6029 / 0.7426 |

The headline holds: **2.39% → 0.97%**, a 59.3% relative reduction against 60.3%
before. The fix buys back 6 slots of collapse and 54k characters, and in exchange
every `content_ref` an agent is handed now names a node the collapsed slot was
actually measured against. Three of the fourteen identifier-losing pairs
disappeared with it — they were chained pairs.

## What it does not fix

The larger precision finding of the original measurement is untouched and still
stands: 33 of the collapsed slots joined **two different facts**, 8 of those 9
pairs `schema`/`schema`, where consolidation's rigid `Procedure: …` template
dominates the mean-pooled vector. That is a level-aware problem, not a threshold
problem — moving the cosine does not separate those pairs from the correct ones —
and it remains a follow-up. Operator recourse is unchanged and immediate:
`LM_RECALL_NEAR_DUP_COSINE=0` restores byte-only delivery without a revert.

## Reproducing

```bash
PYTHONPATH=src python3 scripts/recall_dup_slot_measure.py \
    --live ~/.local/share/living-memory/global.sqlite3 \
    --workdir /tmp/lm-dupslot-verify \
    --selection artifacts/near-dup/dup-slot-measurement.json \
    --json artifacts/near-dup/dup-slot-measurement-honest-bearer.json \
    --md /tmp/lm-dupslot-verify/honest-bearer.md
```

The live database was opened `mode=ro` and copied out with `VACUUM INTO`; it was
never written, and neither the MCP server nor the dashboard was signalled.
`live_database_written: false` and `corpus_stability.constant_through_both_arms:
true` are recorded in the JSON. The JSON's `repository_commit` field reads
`e9484600b419`: the run measured the fix as a working-tree change, committed
immediately afterwards as the commit that adds this file.
