## Interpretation and caveats (written after the result)

- **Quality does not drop; it rises slightly on both hosts.** Credited-node MRR goes up by +0.0024 on
  sfx and +0.0014 on alt, and hit@3 by +0.0020 and +0.0058. Every bootstrap CI for these deltas
  includes or touches 0, so this is "no loss" rather than "a gain". On alt, credited-node recall
  moves 0.5148 → 0.5115 (−0.3 pp, about 1 node in 300), which is within noise.
- **The graph channel stops carrying unrelated nodes.** Uncredited graph hits per replayed event fall
  22% on sfx (1.69 → 1.32) and 37% on alt (0.42 → 0.26), with CIs far from 0. The graph share of
  uncredited hits falls from 0.36 to 0.28 on sfx and from 0.15 to 0.09 on alt. Credited graph hits
  fall only 155 → 144 (sfx) and 20 → 19 (alt).
- **Edges and latency.** Pre-cutoff implicit edges drop 84% on sfx (105,978 → 16,500) and 90% on alt
  (45,296 → 4,417); all connections drop 38%. Replay p50 latency falls about 9% on sfx and 8% on
  alt. Arms were interleaved per event, so machine load hit both arms equally.
- **Link-time vs eventual credit barely differ.** Only 26 (sfx) and 25 (alt) ledger-era edges get
  a ledger row after the 60 s link window. A `credited` valve that decides at close time loses almost
  nothing to lookups that arrive later.
- **Bias against `credited`.** Pre-ledger edges (before 2026-09-07) were kept only if the live
  grader marks them grounded; lookup credit cannot be reconstructed there. So the `credited` arm
  under-links, and it still passes.
- **Leakage shared by both arms.** Node `usefulness_score`/`access_count` and pre-cutoff anchors
  reinforced after T carry post-T learning into both arms equally. Edges and anchors created after T
  were removed from both arms.
- **Traffic filter.** It is conservative: sfx excludes 2,134 of 4,469 post-cutoff events (48%) and
  alt excludes 137 of 1,176. That includes LM work that mentions the ledger. The robust filter is
  the effect-daily-metric sibling's job, and this result does not depend on it.
- **Scope of the claim.** The replay measures ranking of later-credited nodes and graph noise.
  `credited` also means the closing trace's provenance edges to unused nodes are not created. The
  `recalled_nodes`/`source_traces` provenance on the trace is unaffected (it is separate from edges).
  Once explicit `used` marks exist (P2), counting them as credit for this valve is consistent with
  this result, but it was not measured here.
