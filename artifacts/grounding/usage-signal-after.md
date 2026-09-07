# Usage signal after the three vectors (snapshots 2026-09-07, closures since 2026-09-04)

Parent-level closing measurement for goal `feedback-usage-signal`. The three
vectors were each measured in isolation by their own artifacts —
`artifacts/grounding/lookup-credit.md` (lookup credit, master tokenizer at
0.25), `artifacts/grounding/cyrillic-tokenizer.md` (Russian stemming at 0.25)
and `artifacts/grounding/recalibration-2026-09.md` (threshold 0.25 → 0.22 under
the new tokenizer). This artifact replays the same closure corpora once more on
the integrated checkout, so the "after" column is the live gate as it will ship:
`embeddings.tokenize` with Cyrillic stemming (`LM_TOKENIZE_CYRILLIC_STEM=on`),
`DEFAULT_MIN_CONTAINMENT = 0.22` (`LM_GROUNDING_MIN_CONTAINMENT`), and a
same-transport `memory_lookup` of a delivered node credited once per
(event, node) through `recall_credit_ledger` (`LM_LOOKUP_CREDIT_POLICY=delivered`).

Corpora: `~/.cache/living-memory-harness/usage-signal/{sfx,alt}-closures.jsonl`
built by `scripts/usage_signal_corpus.py` from the read-only snapshots
`{sfx,alt}-2026-09-07.sqlite3` (cutoff `2026-09-04T00:00:00Z`; 421 + 423
closures, 2959 + 3847 delivered-result/closing-trace pairs). Relatedness is the
encoder cosine recorded in the corpus (related ≥ 0.5, unrelated ≤ 0.3); a
lookup counts when it is same-transport and within 86400 s of the delivery.
"Before" is `artifacts/grounding/usage-signal-baseline.md` (master tokenizer,
live gate 0.25, no lookup credit); "after" is this checkout at its default
0.22. Same corpus, same grader (`scripts/usage_signal_replay.py`), only the
checkout differs.

## Closures that credit at least one delivered node

The headline of the goal: a closure "credits" when the reinforcement loop
moves usefulness, the scope's weights and an anchor for at least one delivered
node. Before, only text grounding could do that; after, grounding or a
same-session lookup can, deduplicated by the ledger (the overlap column is
what the ledger collapses to one credit).

| host | closures | before: grounded @0.25 (master) | after: grounded @0.22 | lookup closures | overlap (before → after) | before: grounded only | after: grounded ∪ lookup |
|---|---|---|---|---|---|---|---|
| sfx | 421 | 60 (14.2%) | 104 (24.7%) | 102 (24.2%) | 22 → 31 | 60 (14.2%) | 165 (39.2%) |
| alt | 423 | 100 (23.6%) | 238 (56.3%) | 119 (28.1%) | 36 → 93 | 100 (23.6%) | 278 (65.7%) |
| pooled | 844 | 160 (19.0%) | 342 (40.5%) | 221 (26.2%) | 58 → 124 | 160 (19.0%) | 443 (52.5%) |

Pooled, the share of closures that credit nothing falls from 81.0% to 47.5%
(sfx 85.8% → 60.8%, alt 76.4% → 34.3%).

## Where the lift comes from (pooled, same checkout)

The after-sweep carries a 0.25 row, which isolates the tokenizer from the
threshold; the lookup union is the third step.

| step | pairs grounded | cyr-any pairs | lat/lat pairs | closures credited |
|---|---|---|---|---|
| before: master tokenizer, 0.25, grounding only | 268/6806 (3.9%) | 201/5500 (3.6%) | 67/1306 (5.1%) | 160/844 (19.0%) |
| + Cyrillic stemming (0.25) | 445/6806 (6.5%) | 378/5500 (6.9%) | 67/1306 (5.1%) | 240/844 (28.4%) |
| + threshold 0.22 | 718/6806 (10.5%) | 624/5500 (11.3%) | 94/1306 (7.2%) | 342/844 (40.5%) |
| + lookup credit (grounded ∪ lookup) | — | — | — | 443/844 (52.5%) |

Latin/Latin pairs are byte-identical between the two tokenizers at equal
threshold (67/1306 at 0.25 in both), as the tokenizer artifact requires; the
Cyrillic-any share nearly doubles at 0.25 and triples at 0.22.

## Signal against noise, before → after (live gate)

Pair-level, the criterion the recalibration was chosen under: grounded share
among encoder-related pairs versus encoder-unrelated pairs.

| host | related pairs grounded | unrelated pairs grounded | pair S/N |
|---|---|---|---|
| sfx | 74/1058 (7.0%) → 127/1058 (12.0%) | 0/253 (0.0%) → 0/253 (0.0%) | inf → inf |
| alt | 171/2569 (6.7%) → 502/2569 (19.5%) | 2/140 (1.4%) → 5/140 (3.6%) | 4.66 → 5.47 |
| pooled | 245/3627 (6.8%) → 629/3627 (17.3%) | 2/393 (0.5%) → 5/393 (1.3%) | 13.2 → 13.7 |

Closure-level the picture is less flattering and is stated as such: pooled,
closures grounded among unrelated closures rise from 6/86 (7.0%) to 19/86
(22.1%) and closure S/N drops from 3.43 to 2.20 (sfx 8.65 → 3.30, alt 2.32 →
1.79). A closure is "unrelated" by its query↔trace cosine, while grounding is
per node; a wide recall (many results) grounds one of them more easily under a
lower gate even when the closing trace is off-topic to the query. The pair-level
criterion, which is what reinforcement acts on (credit is per node, not per
closure), was the pre-stated one and holds: unrelated pairs grounded stay ≤ 5%
on every host and pooled S/N does not fall.

## What replay cannot see

- Lookup credit lands live at lookup time, on closed and unclosed events
  alike; the corpus holds closed events only, so the union column understates
  the live lookup volume (the lookup-credit artifact records 173 follows on
  the sfx snapshot whose only in-window lookup fell after the cutoff).
- The 24 h lookup window and same-transport join are the live defaults
  (`LM_LOOKUP_CREDIT_WINDOW_SECONDS`); a cross-transport lookup credits
  nothing, by design, and is not counted here.
- Learned-weight effects of the union were A/B'd on the sfx holdout in
  `artifacts/grounding/lookup-credit.md` (non-degenerate, no significant
  regression under any label); the 0.22 gate's holdout A/B is in
  `artifacts/grounding/recalibration-2026-09.md`. This artifact is coverage
  only and adds no new holdout claim.

## Reproduce

```
python3 scripts/usage_signal_replay.py \
  --corpus ~/.cache/living-memory-harness/usage-signal/sfx-closures.jsonl \
  --corpus ~/.cache/living-memory-harness/usage-signal/alt-closures.jsonl \
  --sweep 0.10,0.15,0.20,0.22,0.25 \
  --report artifacts/grounding/usage-signal-after.json \
  --markdown /tmp/usage-signal-after.md
```

Sources pinned in `usage-signal-after.md.src.sha256`; the JSON is the script's
full report (per host and pooled, every threshold in the sweep).

## Full sweep on this checkout

## pooled

| thr | pairs grounded | lat/lat | cyr-any | pairs related | pairs unrelated | pair S/N | closures grounded | closures related | closures unrelated | closure S/N | lookup pairs | lookup closures | overlap | union closures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.100 | 3799/6806 (55.8%) | 652/1306 (49.9%) | 3147/5500 (57.2%) | 2661/3627 (73.4%) | 81/393 (20.6%) | 3.56 | 744/844 (88.1%) | 448/489 (91.6%) | 69/86 (80.2%) | 1.142 | 395/6806 (5.8%) | 221/844 (26.2%) | 321 | 759/844 (89.9%) |
| 0.150 | 2097/6806 (30.8%) | 287/1306 (22.0%) | 1810/5500 (32.9%) | 1665/3627 (45.9%) | 31/393 (7.9%) | 5.819 | 582/844 (69.0%) | 368/489 (75.3%) | 47/86 (54.6%) | 1.377 | 395/6806 (5.8%) | 221/844 (26.2%) | 230 | 632/844 (74.9%) |
| 0.200 | 996/6806 (14.6%) | 126/1306 (9.7%) | 870/5500 (15.8%) | 851/3627 (23.5%) | 10/393 (2.5%) | 9.236 | 405/844 (48.0%) | 275/489 (56.2%) | 22/86 (25.6%) | 2.199 | 395/6806 (5.8%) | 221/844 (26.2%) | 152 | 494/844 (58.5%) |
| 0.220 | 718/6806 (10.5%) | 94/1306 (7.2%) | 624/5500 (11.3%) | 629/3627 (17.3%) | 5/393 (1.3%) | 13.654 | 342/844 (40.5%) | 238/489 (48.7%) | 19/86 (22.1%) | 2.203 | 395/6806 (5.8%) | 221/844 (26.2%) | 124 | 443/844 (52.5%) |
| 0.250 | 445/6806 (6.5%) | 67/1306 (5.1%) | 378/5500 (6.9%) | 401/3627 (11.1%) | 3/393 (0.8%) | 14.553 | 240/844 (28.4%) | 173/489 (35.4%) | 11/86 (12.8%) | 2.766 | 395/6806 (5.8%) | 221/844 (26.2%) | 96 | 367/844 (43.5%) |

## alt

| thr | pairs grounded | lat/lat | cyr-any | pairs related | pairs unrelated | pair S/N | closures grounded | closures related | closures unrelated | closure S/N | lookup pairs | lookup closures | overlap | union closures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.100 | 2652/3847 (68.9%) | 20/31 (64.5%) | 2632/3816 (69.0%) | 2080/2569 (81.0%) | 45/140 (32.1%) | 2.519 | 411/423 (97.2%) | 279/285 (97.9%) | 41/44 (93.2%) | 1.051 | 227/3847 (5.9%) | 119/423 (28.1%) | 205 | 415/423 (98.1%) |
| 0.150 | 1587/3847 (41.2%) | 18/31 (58.1%) | 1569/3816 (41.1%) | 1348/2569 (52.5%) | 17/140 (12.1%) | 4.322 | 365/423 (86.3%) | 252/285 (88.4%) | 31/44 (70.5%) | 1.255 | 227/3847 (5.9%) | 119/423 (28.1%) | 157 | 382/423 (90.3%) |
| 0.200 | 772/3847 (20.1%) | 12/31 (38.7%) | 760/3816 (19.9%) | 690/2569 (26.9%) | 9/140 (6.4%) | 4.177 | 277/423 (65.5%) | 199/285 (69.8%) | 18/44 (40.9%) | 1.707 | 227/3847 (5.9%) | 119/423 (28.1%) | 115 | 311/423 (73.5%) |
| 0.220 | 561/3847 (14.6%) | 10/31 (32.3%) | 551/3816 (14.4%) | 502/2569 (19.5%) | 5/140 (3.6%) | 5.473 | 238/423 (56.3%) | 174/285 (61.1%) | 15/44 (34.1%) | 1.791 | 227/3847 (5.9%) | 119/423 (28.1%) | 93 | 278/423 (65.7%) |
| 0.250 | 344/3847 (8.9%) | 7/31 (22.6%) | 337/3816 (8.8%) | 313/2569 (12.2%) | 3/140 (2.1%) | 5.692 | 165/423 (39.0%) | 123/285 (43.2%) | 9/44 (20.4%) | 2.111 | 227/3847 (5.9%) | 119/423 (28.1%) | 70 | 220/423 (52.0%) |

## sfx

| thr | pairs grounded | lat/lat | cyr-any | pairs related | pairs unrelated | pair S/N | closures grounded | closures related | closures unrelated | closure S/N | lookup pairs | lookup closures | overlap | union closures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.100 | 1147/2959 (38.8%) | 632/1275 (49.6%) | 515/1684 (30.6%) | 581/1058 (54.9%) | 36/253 (14.2%) | 3.859 | 333/421 (79.1%) | 169/204 (82.8%) | 28/42 (66.7%) | 1.243 | 168/2959 (5.7%) | 102/421 (24.2%) | 116 | 344/421 (81.7%) |
| 0.150 | 510/2959 (17.2%) | 269/1275 (21.1%) | 241/1684 (14.3%) | 317/1058 (30.0%) | 14/253 (5.5%) | 5.418 | 217/421 (51.5%) | 116/204 (56.9%) | 16/42 (38.1%) | 1.492 | 168/2959 (5.7%) | 102/421 (24.2%) | 73 | 250/421 (59.4%) |
| 0.200 | 224/2959 (7.6%) | 114/1275 (8.9%) | 110/1684 (6.5%) | 161/1058 (15.2%) | 1/253 (0.4%) | 38.05 | 128/421 (30.4%) | 76/204 (37.2%) | 4/42 (9.5%) | 3.913 | 168/2959 (5.7%) | 102/421 (24.2%) | 37 | 183/421 (43.5%) |
| 0.220 | 157/2959 (5.3%) | 84/1275 (6.6%) | 73/1684 (4.3%) | 127/1058 (12.0%) | 0/253 (0.0%) | inf | 104/421 (24.7%) | 64/204 (31.4%) | 4/42 (9.5%) | 3.295 | 168/2959 (5.7%) | 102/421 (24.2%) | 31 | 165/421 (39.2%) |
| 0.250 | 101/2959 (3.4%) | 60/1275 (4.7%) | 41/1684 (2.4%) | 88/1058 (8.3%) | 0/253 (0.0%) | inf | 75/421 (17.8%) | 50/204 (24.5%) | 2/42 (4.8%) | 5.149 | 168/2959 (5.7%) | 102/421 (24.2%) | 26 | 147/421 (34.9%) |

