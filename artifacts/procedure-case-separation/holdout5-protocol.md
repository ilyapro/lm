# New alt-only holdout after the carrier repair

Candidate production code is frozen at `47ec513`; reader code at `e26e914`.
This protocol is fixed before the new alt snapshot or its results are read.
Take one read-only online-backup snapshot of alt LM and transfer it alt→sfx.
Never transfer sfx data to alt. Exclude any item whose id or stripped-content
SHA-256 appears anywhere in the previously examined alt snapshot or in the
sfx design snapshot. Report the excluded count and verify zero retained
overlap. All remaining eligible items enter; there is no sampling or tuning.

Use the two families, classification, queries and original `24e9009` versus
candidate replay from `holdout-protocol.md`. `holdout4.py items` accepts both
examined snapshots; `measure`, `pass`, `census` and `report` remain the same.
The reader also follows the `Related group carriers` IDs a delivered concept
lists through `memory_lookup`, then looks up every evidence id in those
carriers. It counts complete lookup response characters, including metadata.
The result is obtained only through this delivered path, never by searching a
missing id in the database.

Accept if there are zero newly lost whole instruction/correction items in
both query families, zero unconfirmed binding steps in the after census and
deliveries, and no duplicate active nodes per procedural group. Report
baseline misses, case items, latency and full delivered-plus-forced-lookup
volume separately. Fewer than 10 eligible instruction/correction items means
the corpus is too small for an independent generalization claim; still report
all observed results and do not use this set to change the candidate.
