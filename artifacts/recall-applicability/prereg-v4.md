# Applicability recall: independent replacement holdout v4

Sealed on 2026-10-02 UTC after the original failed measurement, v2 PASS, and inadequate v3 packet. This receipt is data-free: it contains no question, required clause, source or node ID, or raw response. The original snapshot and every earlier sealed byte remain unchanged. The failed v3 packet is an exclusion and provenance input, not a successful retention control. Its diagnostic is archived at commit `e7c0ac6808380ced1e5668a7c55a1f5ba9e928df` in `artifacts/recall-applicability/prereg-v3.md`; even broad-scope diagnostic recall ranked neither cross-project source in its top four.

The private v4 packet is under `/home/sfx/p/ae/artifacts/recall-applicability/v4/`, with directory mode 700 and the goldset, baseline outcomes, and seal files mode 400.

| Bound item | SHA-256 |
| --- | --- |
| v4 seal | `f7a3d2a0152e62c15f38fc225d37d648239fc62d1c02e15d1398d3ab77f2cc4c` |
| original failed report | `fcd16e6e2e556f0bdeaa04f83ea111eac95bb40f6c97192439f77728b6de8942` |
| v2 PASS report | `d6bc3d9af1a3a313737e08238332f27bb1485422b89b69083413b910fdcac3d5` |
| v4 snapshot | `3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef` |
| v4 goldset | `fd09263d7afab743d12fa8e4881f6c3e52c39424e630f96c34ac255df57cdda5` |
| v4 previous_goldset | `c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c` |
| v4 baseline_outcomes | `a5b142aab3f3b859f5826dc4b390d2bedfde977e227dbd3ebe449b17b84d598e` |
| v2 seal | `58c6d33dd929775cf53c84cd99dc8c503e8994267adf1b5ef444df48fc1a7181` |
| v2 snapshot | `3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef` |
| v2 goldset | `1b1aa4290bc92968f356a82f58555c1fa26591d067d090fe61018166499541e2` |
| v2 previous_goldset | `c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c` |
| v2 baseline_outcomes | `de75ca46f71f07cb08e5f684d19bb5307be5daaaa19b945c4302e5a0168dff3c` |
| v3 seal | `f27268596fe4dc4a0f7c8a98dcf735eb01df9d235ed4ef8b6dcd0a8e2a932cca` |
| v3 snapshot | `3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef` |
| v3 goldset | `3deb21f938e5991d60cf3cdff595ea8740136c15fcd07c5912909d6f47ebb42f` |
| v3 previous_goldset | `c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c` |
| v3 baseline_outcomes | `03e9fdbc63e35f6182490c1c3de7831a57931e6168093f8615325eed8955bfdc` |

The seal uses `schema_version=1`, `generation=4`, `candidate_generation=3`, a UTC sealing timestamp, accepted baseline commit `b9769d84e3188ee1e646627ebe1b4d454f69d6f9`, and its archived `src` tree `8bba3ecdd76777031347e5c0dc6e7441d8c0b19e`. Its four file entries point to the original snapshot, the v4 gold set, the original previous gold set, and the v4 baseline outcomes. `prior_seals` binds exactly the immutable v2 and v3 seals in that order, including their bound ancestry.

## Independent freeze and exclusions

I independently authored one bounded set of fourteen holdout cases on the unchanged original snapshot, two each for concrete, compound, instruction, correction, cross-project, absent-knowledge, and large-group questions. The four original development JSON case values were copied unchanged. Every new question uses broad null scope, depth 1, and max_results 4. The instruction set includes applicable saved-trigger carrier knowledge using ordinary task vocabulary, without adopting the repair's synthetic fixture terms. Every positive literal required clause was checked against its complete SQLite source and the existing corpus validator passed; absent nonce compounds had no snapshot content match.

Exclusion compared all original, v2, and v3 queries, source IDs, and semantic answer subjects, not just labels. It considered carrier evidence, child `source_traces`, correction and supersedes families, and the underlying claim before admitting a source. Direct lineage intersection with earlier answer sources was checked. The holdout categories cover distinct topics, and no prior topic was relabeled as a new family.

Questions, sources, and literal clauses were frozen in the private gold set at 2026-10-02T03:53:15Z, hash `fd09263d7afab743d12fa8e4881f6c3e52c39424e630f96c34ac255df57cdda5`, before baseline capture recorded at 2026-10-02T03:54:43Z. No question or oracle was replaced after seeing baseline ranks. No candidate implementation, development measurement, candidate output, or verifier decision was used to select cases. No candidate or consumed packet was run.

## Actual archived baseline and controls

Actual `b9769d84e3188ee1e646627ebe1b4d454f69d6f9` `src` was extracted with `git archive`; `git rev-parse <commit>:src` returned `8bba3ecdd76777031347e5c0dc6e7441d8c0b19e`. The baseline process ran from a separate scratch directory and asserted that `living_memory.config`, `living_memory.retrieval`, and `living_memory.storage` each imported from that archive. This avoided the checkout-root Python shim. The installed default sentence-transformers encoder loaded during the completed capture. The capture script's hardcoded `code_head` was treated only as a schema field, not as proof of executable provenance.

For each case, the capture created a fresh writable copy of the original frozen snapshot, a new store and recall service, then disabled event and access logging. Recall used the frozen broad null scope, depth 1, and max_results 4, ordinary trigger mode with `LM_RECALL_SCHEMA_TRIGGER` unset, and `LM_RECALL_NEAR_DUP_COSINE=0.97`. The remaining non-secret session settings were unchanged; no provider, model, or tier was changed. An initial Python `-I` smoke invocation excluded the installed user-site encoder and was interrupted before writing outcomes; the completed capture used normal Python with asserted archived module origins and the installed encoder.

The baseline `source_ranks` equal the recorded ranking positions of acceptable oracle sources. At least one holdout source reached the top four for each protected category: instruction 1 case, correction 1 case, cross-project 2 cases. These are non-vacuous source-rank preflight controls. They do not claim delivered fact sufficiency: final retention must count required clauses actually delivered through ordinary recall and any natural `content_ref` lookup. The successor verifier still rejects baseline-reachable protected fact loss and development fact loss.

## One-shot successor rule

The parent must pass the fresh whole declared suite before one-shot paired measurement against candidate-v3. The unchanged acceptance threshold is two distinct improved holdout topic groups with zero losses of baseline-reachable required facts. The paired report must retain relevance and loss counts, all necessary recall and lookup calls, response volume, call counts, latency, production-code complexity, and large-group reading. If this single frozen packet proves inadequate later, preserve its result and report failure; do not select replacement cases from observed ranks, screen repeated favorable packets, consume v3, or begin a multi-day collection cycle.
