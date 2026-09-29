# memory_remember no longer waits for consolidation — measurements

Corpus: a copy of live `~/.local/share/living-memory/global.sqlite3` taken
2026-09-29 08:07 +07 (`project:ae`: 6603 active traces). The live DB was only read.
Every run: one heavy process at a time, `OMP/OPENBLAS/MKL_NUM_THREADS=4`,
`nice -n 10`, `LM_DECAY_SWEEP_INTERVAL_SEC=0` (the hourly sweep would fire in
the first write on a copy and retire TTL traces), the live env
(`LM_AUTO_CONSOLIDATE_POLICY=adaptive`, …) and config. The harness
(`latency_harness.py`) builds an in-process server, pads the scope to
count ≡ 99 (mod 100) with 96 identical filler traces, then issues the due
100th write while a second thread runs a recall every second.

## Due write and recall during the pass

| run | code | load (1 min) | due write returns | pass | recall during pass p50 / max | recall outside pass p50 | longest lock hold |
|---|---|---|---|---|---|---|---|
| before | 622df40 | 6.9 → 8.4 | **1137.7 s** (pass inline) | 1137.7 s | **1139 s** (1 recall, blocked for the whole pass) | 0.31 s | whole pass |
| after #1 | this branch (pre-fairness fix) | 68.6 → 34.3 | 0.09 s | 55.1 s | 1.23 s / 13.1 s | 0.67 s | 5.3 s |
| after #2 | this branch | 5.2 → 47.5 (ae gate shards started) | 4.04 s | 154.2 s | 2.67 s / 25.9 s | 0.45 s | 22.9 s (0.5 s step stretched by CPU starvation of the niced process) |
| after #3 | this branch (final) | 3.4 → 3.8 | **0.04 s** | **17.1 s** | **0.65 s / 1.72 s** | 0.23 s | 0.58 s |

"before" and "after #3" ran in the same low-load window (09:27–09:59 +07).
After-cache runs start from the corpus copy plus a filled
`consolidation_embeddings` table, i.e. the steady state from the second pass
after deploy on. The first pass after deploy still encodes every trace once
(1600 s measured below, niced), in the background.

## Where the minutes went (project:ae, 6603 traces)

`cluster_equiv.py` → `equiv_ae.json`:

| cost | before | after |
|---|---|---|
| whole-content vector per trace (nodes have no vector column; every pass re-encoded every trace) | 1600 s (cold encode) | 0.13 s (cache read) |
| greedy clustering (pure-Python cosine per trace × cluster, ~3M pairs, 919 clusters) | 514 s | 2.19 s |

Indexed vs reference clustering on the same vectors: identical membership,
strategies and cluster keys for all 919 clusters; embedding sums
bit-identical (max diff 0.0).

Lock steps by site after the change (`step_labels.py`, load 4): pass 14.7 s,
12.1 s of it under the lock in 167 steps; the longest steps are one cluster
merge (≤ 0.54 s) or one procedural-schema group (≤ 0.53 s).

## Same input, same result

`compare_before_after3.json` (and the same for after #1 and #2): clusters
921, traces 6700, concepts created 1 / updated 13, schemas created 3 /
updated 105, decayed 73 in every run; all 14 touched concepts match
before vs after with identical content, source traces and confidence.
