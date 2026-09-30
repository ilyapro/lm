# Group-identity candidate regression protocol

Candidate production code `e3cb245` is frozen before this run (supervisor
#1382 direction). It replaces title-based recall dedup plus peer carrier
links with one group identity (scope, group key) on write and on read.

Data, queries and items are unchanged: the design set, holdout 2 and
holdout 3 from the committed protocols, on fresh copies of the same local
read-only sfx snapshot, against the saved original `24e9009` measurements
(holdout 3: the corrected baseline). These sets were examined during repair:
they are regression data, not an independent holdout. All eligible items;
baseline misses are counted separately from new losses.

Reader: the frozen `replay.py`/`holdout.py` reader, unchanged. It re-fetches a
cut schema, carrier or wanted record, and looks up every evidence id a
delivered carrier lists. Its `Related group carriers` step stays in the code
and is inert (the candidate writes no peer ids). Nothing undelivered is
searched for; all delivered and forced lookup characters with metadata count.
Each measuring process asserts the imported `living_memory` path.

Run two ordinary procedural passes per fresh copy before recall, then census.
Accept only if: zero newly lost full current instructions/corrections in all
three sets; zero unconfirmed binding steps in census and deliveries; at most
one active node per procedural group; the second pass creates nothing; net
production lines against `24e9009` (`git diff --numstat 24e9009 -- src`) ≤ 0.
Report volume against both the original and the `47ec513` exhaustive-peer
measurements.
