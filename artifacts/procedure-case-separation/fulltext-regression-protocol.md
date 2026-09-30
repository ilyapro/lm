# Full-text carrier regression protocol (frozen before the replay)

Operator direction after the refuted group-identity candidate: a group
carrier may hold a fuller nonbinding representation of its source records;
the 600-character lead is not a requirement. Candidate production code is
pinned at `7cb9436` (`src/` identical to the measured copy): a carrier holds
each current record whole, by id, as evidence (never steps); recall's live
dedup stage identifies schemas and carriers by title again (base rule).

Data, queries and items are unchanged: the design set (23 prose recipes, 40
schema topics), holdout 2 (33 cases) and holdout 3 (items + 60 collateral
queries) from the committed protocols, on fresh copies of the same local
read-only pristine sfx snapshot. These sets were examined during repair:
regression data, not independent validation. All items; baseline misses are
counted separately from new losses.

Reader (fixed now, before any measurement; `replay.read`): it sees only the
query and the real recall output. Every delivered result is read; a result
delivered cut (delivery other than `full`) is re-fetched whole by its id with
`memory_lookup`; nothing else is looked up — no target id, no ids a carrier
lists, no peers, nothing undelivered. The previous reader's target-id
re-fetch and evidence expansion are removed. Because the reader changed, the
original `24e9009` is re-measured with this same reader on fresh pristine
copies (`fulltext_run.py`, which imports and asserts the requested code
before the reader); saved numbers from the old reader are shown only as
history.

Candidate procedure: one fresh pristine copy, two ordinary procedural passes,
census, one non-matching warm-up recall per searched scope (recall's own
drain embeds the rewritten carriers), then one copy of that state per set.

Accept only if: zero newly lost full current instructions/corrections in all
three sets (itemwise against the original under the same reader); zero
unconfirmed binding steps in the census (orphans included) and in deliveries;
at most one active node per procedural group; the second pass creates
nothing; production net lines against `24e9009` ≤ 0. Report, without a pass
threshold, stored procedural-node characters and delivered + forced-lookup
characters against the original; a bounded increase is reported as an exact
delta with its mechanism, not hidden.

## Amendment after the first run (criteria, reader and data unchanged)

The pinned `7cb9436` was measured once: design and holdout 3 had 0 new
losses; holdout 2 had 2 new instruction losses, both records of one group
whose legacy schema, turned carrier in place, was relabeled by the pass (its
records' majority procedure_id replaced the trigger the query matched). The
fix `c25a6cd` keeps a group node's trigger while a current record carries
it. It is measured once more by exactly this protocol on a new fresh copy.
Both runs are reported; the data had been examined, so neither is
independent validation.
