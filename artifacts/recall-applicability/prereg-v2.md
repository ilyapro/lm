# Applicability recall: independent holdout v2

Sealed on 2026-10-02 UTC after the original holdout was consumed and failed.
The original snapshot and every original sealed artifact remain unchanged. This
receipt contains no private query, answer clause, node ID, or raw response.
The private packet is local to sfx under
`/home/sfx/p/ae/artifacts/recall-applicability/v2/` (directory mode 700;
packet, baseline outcome, and seal files mode 400).

| Sealed input or outcome | SHA-256 |
| --- | --- |
| Original frozen SQLite snapshot | `3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef` |
| New private v2 gold set | `1b1aa4290bc92968f356a82f58555c1fa26591d067d090fe61018166499541e2` |
| Previous private gold set | `c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c` |
| New private v2 baseline outcomes | `de75ca46f71f07cb08e5f684d19bb5307be5daaaa19b945c4302e5a0168dff3c` |

The private `seal.json` fixes these absolute paths and digests, schema version 1,
the UTC seal time, accepted baseline commit
`b9769d84e3188ee1e646627ebe1b4d454f69d6f9`, and its `src` tree
`8bba3ecdd76777031347e5c0dc6e7441d8c0b19e`. The snapshot entry points
to the original frozen database; no new database was made.

## Independent authoring and family exclusion

The four original development cases were copied byte for byte as JSON values.
Fourteen new holdout cases were authored on the original snapshot before any
baseline capture. Two cases each cover concrete facts, compound facts, short
trigger applicable instructions, corrections, cross-project transfer, absent
knowledge, and large evidence groups. Oracle IDs are never inserted into the
recall questions. Every positive literal required clause was checked as a
substring of its frozen source before the packet was written and then checked
again by the existing corpus validator. Absent questions use nonce compounds
with no match in the snapshot.

Exclusion used the old development **and** holdout queries, source IDs, and
semantic topics. Candidate answer sources were checked against direct source
lineage (`source_traces` and carrier evidence), correction and `supersedes`
relationships, and source content. A carrier and its child evidence count as
one answer family; so do obsolete and correcting claims about the same subject.
The author also rejected semantically equivalent subjects even if their group
names differed. Broad multipurpose carriers can mention unrelated cases, so
their mere common ancestor was not treated as topic equivalence; the underlying
claim and its provenance were checked instead. The new queries, sources, and
topic groups are disjoint from every original case. Cases were not selected by
candidate behavior, and no candidate code or results were inspected here.

## Baseline provenance and fixed environment

The baseline rank capture imported **actual archived source** from accepted
`b9769d8`: `git archive` extracted that commit's `src` into an isolated local
directory. The corpus capture script did not exist at that commit, so a copy
of the existing capture script was placed beside the archive; its hardcoded
`code_head` field was not used as evidence of executable provenance. The
archived `src` tree hash above and the extraction command establish it.

For every case, recall used its frozen broad scope (`null`), `depth=1`, and
`max_results=4`. The capture used a writable copy of the exact snapshot,
fresh recall service per case, and disabled access and event logging. The
environment retained the session's `LM_RECALL_NEAR_DUP_COSINE=0.97`, used the
installed default sentence embedding backend without a backend override, and
left `LM_RECALL_SCHEMA_TRIGGER=name` disabled. It did not alter providers,
models, or tiers. The later paired runner must give both arms identical recall
arguments and environment, separate fresh snapshot copies and sessions, its
controlled embedding warmup, and alternating arm order.

At baseline top-four source-rank capture, one new holdout instruction source,
one correction source, and two cross-project sources were reachable. This is
enough to test retention in all protected categories. Source rank alone does
not establish delivered clause sufficiency: the paired measurement must
perform any necessary `memory_lookup` and count the resulting response.

## One-shot measurement rule

Run the revised frozen candidate once against this exact packet with the
paired runner, after the candidate and verdict tool are frozen. Measure topic
relevance, source and required-clause reachability through all necessary recall
and lookup calls, irrelevant top-four slots, response volume, call counts,
warm latency, and production-code complexity. Require **at least two
independent holdout topic groups** to lose irrelevant top-four results and
**zero losses of baseline-reachable required facts**, including every
baseline-reachable instruction, correction, and cross-project fact. Preserve
large-group reading. Report failure as failure; do not revise this packet,
rerun for favorable outcomes, or erase the original failed report. No
multi-day collection is a prerequisite.
