# Historical recall-map relevance evidence

This is a deterministic, aggregate-only rendering of the frozen dataset manifest and feature analysis pinned at `2026-08-22T18:00:00Z`. The candidate holdout was neither inspected nor created.

**Result:** the organic-train-fitted model selected 395 of 686 observed-map evaluation items and consumed 54 of those 395, a rate of `0.136709`. The unchanged transfer floor of `0.233` therefore **did not pass**. The selected transfer cohort did pass every unchanged minimum: 294 events versus 20 required, 395 items versus 20 required, and 160 transport sessions versus 10 required. The overall transfer verdict is **FAIL**, with no evaluation refit.

This result is evidence about historical transfer, not a candidate deployment verdict. No feature is established for deployment by a model that failed the frozen transfer floor.

## Outcome, cohort construction, and exact denominators

The endpoint is non-anchor-stratum node-id consumption within 24 hours. Its unit is one `(delivering event, delivered item)` pair, and its rate is consumed items divided by items. The evaluator verified the sealed preregistration and effect-tool hashes before reusing the frozen protocol, consumer index, item construction, anchor stratification, and node-id scoring behavior. Organic items use the frozen window `[2026-08-01T00:00:00Z, 2026-08-18T11:00:00Z)`, an organic head cut of 3, and caps of 6 items for each arm.

The split is event-component based, not item based. Events are joined when they share a source-qualified cache identity, transport identity, or session identity. This produced 567 connected components. Every component touching any observed map delivery—including a delivery with no scorable primary item—was forced to evaluation; 117 components were forced this way. Each remaining organic component was assigned deterministically with the sealed seed and an 80% train fraction. The final assignment contains 369 train components and 198 evaluation components, with no component shared between train and evaluation. No identity value is published or used as a model feature.

The primary fitting cohort contains only pre-feature, opportunity-bearing organic items in the primary stratum. The organic evaluation cohort contains the remaining disjoint primary-stratum organic items. Every already-observed pre-candidate map item is transfer evaluation only; the model fit contains exactly zero map rows. Component counts within the two evaluation arms need not add to the 198 evaluation components because arms may occur in the same connected component and some bridge events have no scorable primary item.

| Cohort | Role | Components represented | Events | Items | Consumed | Not consumed | Rate | Transport sessions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Organic train | sole fitting population | 369 | 1,040 | 3,102 | 1,567 | 1,535 | 0.505158 | 737 |
| Organic evaluation | disjoint evaluation | 85 | 571 | 1,389 | 526 | 863 | 0.378690 | 294 |
| Observed-map evaluation | transfer evaluation only | 83 | 401 | 686 | 57 | 629 | 0.083090 | 189 |
| Secondary anchor | sensitivity diagnostic only | 502 | 1,782 | 4,061 | 1,559 | 2,502 | 0.383896 | 1,055 |

The secondary-anchor rows are outside the primary endpoint. Their events can overlap primary-stratum events, so they must not be added to the primary cohort denominators.

## Frozen sources and hashes

The single logical source is the redacted snapshot `$SNAPSHOT/local.sqlite3`. The database part is 529,993,728 bytes and has SHA-256 `b839891eb5a45f759dcc4fd89f77fdde7cc8125b28948b35fb256778578957ba`. Its canonical parts-receipt digest is `295cf59af63104d6ba17836086f405416acca4e1801f600782b190f0672a0035`; its schema digest is `b2b7965b3cac45692f96a21644a9646d3877ff718bfa917199c0cb8ded109039`. The pinned pre-candidate slice contains 56,013 event rows, from `2026-05-14T20:05:23Z` through `2026-08-22T17:50:12Z`, across 40 schema objects.

| Bound input or artifact | SHA-256 | Meaning |
|---|---|---|
| Historical evaluator | `dfa25689cfa71bf79de34e8943b7b6a8c68e11f5b148cf04c52e16a2e354cb41` | code bytes used to produce and verify the evidence |
| Frozen effect tool | `e92aceeafb27ae736378f1b1866886cf8446b1cf17c1269a7cd6ad61a039eeea` | reused 24-hour consumption implementation |
| Frozen preregistration | `9eb6a170459bb68138a972b4ba76b76abb044cbb00024cdb04e38e038a85b1b0` | sealed protocol input |
| Frozen baseline | `d1322b5d46664f81f8c19f3d0ee80f11942a33c001932987ee5cc1023c801905` | supplies the frozen organic baseline rate |
| Frozen plan binding | `d17b2e65a20df83968886d307f7f9792491d02cb0ebb406ddeb45b1392e7ad25` | protocol plan digest |
| Dataset manifest, canonical JSON | `34c38922a9b6245113a6ca07e20333eb8846a69d87b66f72ede5f52ac3f2afe2` | digest bound by the feature analysis |
| Dataset manifest, stored bytes | `67ecb4efabbc74c824c3542252b715b5304e4e1294daeffb5ea3a21836f1bee3` | includes the terminal newline |
| Feature analysis, stored bytes | `bd1fdbb072bc6b0512826e96e68ba30455f226815f954bab710bd7e1a29134c9` | machine-readable source for this report |
| Primary feature schema | `56322050f56778e07588cc5aacc069703020f673ab8f023fa9ad66cc7d809525` | ordered eight-feature model surface |

## Decision-time boundary and leakage matrix

Decision-time evidence is restricted to immutable node creation fields, the recorded delivery envelope, and earlier delivery outcomes whose complete 24-hour windows had already matured. Content, provenance, and context shape are current-snapshot diagnostics only. They were not fitted, because their historical values were not versioned. Unrecorded historical values were left unavailable rather than reconstructed from future state.

| Evidence surface | Organic train | Organic evaluation | Observed-map evaluation | Secondary anchor | Leakage disposition |
|---|---:|---:|---:|---:|---|
| Immutable node age | 3,102/3,102 available | 1,389/1,389 | 686/686 | 4,061/4,061 | decision-time; derived only from immutable `created_at` |
| Recorded or immutable level | 3,102 recorded | 1,389 recorded | 686 immutable | 3,723 recorded + 338 immutable | decision-time |
| Strictly matured past non-consumption | 3,102/3,102 | 1,389/1,389 | 686/686 | 4,061/4,061 | decision-time; only outcomes ending no later than the current delivery instant |
| Recorded delivery scores | 3,102/3,102 | 1,389/1,389 | 0/686 | 3,723/4,061 | analysis-only; missing for every map item, so not transferable |
| Content form and size | 3,102 snapshot-only | 1,389 snapshot-only | 686 snapshot-only | 4,061 snapshot-only | sensitivity only; not recorded at delivery |
| Historical context and provenance | 0 reconstructed; 3,102 snapshot-only | 0; 1,389 snapshot-only | 0; 686 snapshot-only | 0; 4,061 snapshot-only | not versioned; excluded from fitting |
| Context shape | 3,102 snapshot-only | 1,389 snapshot-only | 686 snapshot-only | 4,061 snapshot-only | sensitivity only; not recorded at delivery |
| Historical cascade stage | 0/3,102 | 0/1,389 | 0/686 | 0/4,061 | not recorded; not reconstructed |
| Current mutable node statistics | not read | not read | not read | not read | current `access_count`, `usefulness_score`, and `last_accessed`, plus `updated_at`-derived usage, are rejected |
| Identity and grouping fields | split construction only | split construction only | split construction only | split construction only | identifiers, task metadata, hosts, source identity, cache identity, and session identity are forbidden model features and are not published |
| Outcome label | target only | target only | target only | target only | never a predictor |
| Observed-map rows | 0 in fit | evaluation only | evaluation only | not in primary fit | no fit contamination and no evaluation refit |
| Candidate holdout | not accessed | not accessed | not accessed | not accessed | no holdout observations exist in this evidence |

The machine-readable leakage audit additionally records an empty mutable-column read set, `future_mutated_node_stats_read=false`, and the exact history cut `prior delivery outcome_end <= current delivery created_at`.

## Decision-time associations

Each association below is univariate: `C / N; Δ` means the feature mean among consumed items, the mean among non-consumed items, and `C − N`. A positive sign is association, not causation. For these eight features there is no missingness, so the exact consumed/non-consumed denominators are 1,567/1,535 for organic train, 526/863 for organic evaluation, 57/629 for observed-map evaluation, and 1,559/2,502 for the secondary-anchor sensitivity cohort.

| Decision-time feature | Organic train C / N; Δ | Organic eval C / N; Δ | Observed-map eval C / N; Δ | Secondary anchor C / N; Δ |
|---|---|---|---|---|
| `node_age_log_days` | 2.93767796 / 2.44468449; +0.49299348 | 3.17329942 / 2.67488855; +0.49841087 | 2.49567767 / 3.37643360; −0.88075593 | 3.54620425 / 3.75391968; −0.20771543 |
| `level_trace` | 0.61710274 / 0.81172638; −0.19462364 | 0.65209125 / 0.84241020; −0.19031894 | 0.85964912 / 0.82988871; +0.02976041 | 0.81654907 / 0.90087930; −0.08433023 |
| `level_concept` | 0.01595405 / 0.02866450; −0.01271044 | 0.06083650 / 0.04171495; +0.01912155 | 0.00000000 / 0.09220986; −0.09220986 | 0.00577293 / 0.00159872; +0.00417421 |
| `level_schema` | 0.36694320 / 0.15960912; +0.20733408 | 0.28707224 / 0.11587486; +0.17119739 | 0.14035088 / 0.07790143; +0.06244945 | 0.17767800 / 0.09752198; +0.08015602 |
| `prior_matured_log_count` | 3.45977349 / 2.24100998; +1.21876351 | 3.46141718 / 2.16405345; +1.29736373 | 2.33806851 / 1.43966354; +0.89840497 | 3.90317594 / 3.83785102; +0.06532492 |
| `prior_nonconsumed_log_count` | 3.38273171 / 2.18359482; +1.19913689 | 3.37831444 / 2.10114219; +1.27717225 | 2.14945110 / 1.37169247; +0.77775863 | 3.75670348 / 3.68829456; +0.06840891 |
| `prior_nonconsumption_streak_log` | 1.72859847 / 1.42826335; +0.30033512 | 2.03452456 / 1.38001537; +0.65450919 | 1.12589497 / 0.91825254; +0.20764243 | 2.25732199 / 2.11566555; +0.14165645 |
| `prior_consumption_rate` | 0.06956634 / 0.05144147; +0.01812487 | 0.07447021 / 0.05862764; +0.01584256 | 0.14837609 / 0.06247536; +0.08590073 | 0.12491475 / 0.12701497; −0.00210023 |

Transferability requires the same non-zero direction in organic train, organic evaluation, and observed-map evaluation.

| Feature | Train direction | Organic-eval direction | Map-eval direction | Same sign? | Interpretation |
|---|---|---|---|---|---|
| `node_age_log_days` | positive | positive | negative | No | freshness/age direction does not transfer |
| `level_trace` | negative | negative | positive | No | level effect reverses on map evaluation |
| `level_concept` | negative | positive | negative | No | direction already reverses within organic data |
| `level_schema` | positive | positive | positive | Yes | same-sign association only; not by itself deployment evidence |
| `prior_matured_log_count` | positive | positive | positive | Yes | same-sign association only |
| `prior_nonconsumed_log_count` | positive | positive | positive | Yes | does not support the hypothesized negative penalty |
| `prior_nonconsumption_streak_log` | positive | positive | positive | Yes | univariate direction does not support a negative penalty |
| `prior_consumption_rate` | positive | positive | positive | Yes | same-sign association only |

The age, trace-level, and concept-level signals must not be described as transferring. The four past-history count/rate associations and schema level have the same sign, but the frozen combined policy still fails the required transfer rate.

## Frozen primary model and transfer result

The primary model is deterministic L2 logistic regression, fitted once on all 3,102 organic-train items and their 1,567 positive outcomes. It ran 1,200 iterations with learning rate `0.05`, L2 value `0.01`, intercept `0.0370209161`, organic-train mean imputation, and no missingness indicator. Its probability threshold, `0.377120898`, was fitted only on train by maximum F1 subject to selecting at least 20% of train. It was then applied without refitting.

Coefficients are on standardized features and are adjusted model weights; they need not have the same sign as a feature's univariate association.

| Feature | Train mean | Train scale | Standardized coefficient |
|---|---:|---:|---:|
| `node_age_log_days` | 2.6937240679 | 1.7827266672 | −0.1038265282 |
| `level_trace` | 0.7134107028 | 0.4521679687 | −0.1305001282 |
| `level_concept` | 0.0222437137 | 0.1474751875 | −0.0702332736 |
| `level_schema` | 0.2643455835 | 0.4409841221 | +0.1572973710 |
| `prior_matured_log_count` | 2.8566780707 | 2.5772848150 | +0.2484349169 |
| `prior_nonconsumed_log_count` | 2.7893483661 | 2.5560016164 | +0.2552737496 |
| `prior_nonconsumption_streak_log` | 1.5799800265 | 1.7659676253 | −0.1573156163 |
| `prior_consumption_rate` | 0.0605973966 | 0.1167477646 | +0.0417787346 |

| Applied cohort | Total items | Selected items | Selection rate | Selected events | Selected transport sessions | Selected consumed | Selected rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Organic train | 3,102 | 2,921 | 0.941651 | 1,029 | 735 | 1,529 | 0.523451 |
| Organic evaluation | 1,389 | 1,242 | 0.894168 | 533 | 290 | 502 | 0.404187 |
| Observed-map evaluation | 686 | 395 | 0.575802 | 294 | 160 | 54 | 0.136709 |

The unchanged decision floor is `0.233`, derived from the frozen organic rate `0.4662` times the sealed relative floor `0.5`, with the preregistered rounded floor retained. For observed-map transfer:

| Gate | Observed | Required | Passed? |
|---|---:|---:|---|
| Consumption rate | 0.136709 | at least 0.233 | **No** |
| Events | 294 | at least 20 | Yes |
| Items | 395 | at least 20 | Yes |
| Transport sessions | 160 | at least 10 | Yes |

Thus the cohort minimums passed, the unchanged `0.233` threshold failed, and the combined verdict is **FAIL**. The unfiltered observed-map rate was 57/686 = `0.083090`; selection raised the descriptive rate but did not meet the sealed floor.

## Snapshot-only ballast diagnostic

The form classifier is the emitter-invariant `form/schema/provenance` classifier. Its real-corpus output is a current-snapshot prevalence diagnostic only and never enters fitting. The real corpus has no independent item-level form ground truth, so real-corpus false-positive and false-negative rates are unavailable; the counts below are classifier outputs, not a claim of perfect real-corpus accuracy.

| Cohort | Eligible items (consumed) | File-chunk items (consumed) | Strategy-stagnation items (consumed) | Supervision-journal items (consumed) | All classified ballast | Ballast prevalence | Ballast consumed rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Organic train | 3,065 (1,530) | 37 (37) | 0 (0) | 0 (0) | 37/3,102 | 1.1928% | 37/37 = 1.000000 |
| Organic evaluation | 1,381 (518) | 8 (8) | 0 (0) | 0 (0) | 8/1,389 | 0.5760% | 8/8 = 1.000000 |
| Observed-map evaluation | 652 (57) | 30 (0) | 4 (0) | 0 (0) | 34/686 | 4.9563% | 0/34 = 0.000000 |
| Secondary anchor | 3,983 (1,497) | 70 (61) | 8 (1) | 0 (0) | 78/4,061 | 1.9207% | 62/78 = 0.794872 |

The observed-map arm contains 34 classified ballast items and none was consumed. Organic file-chunk items, however, were all counted consumed under the frozen outcome construction. That arm-specific reversal is why the real-corpus form signal is reported as sensitivity evidence and is not smuggled into the decision-time model.

### Seeded randomized controls

The fixed seed is `20260823`. There are 32 independently randomized positive variants for each of the three ballast classes and 10 user-authored near-miss negative controls: 106 cases total, with 96 expected ballast and 10 expected eligible. Control text is not published.

| Expected class | Predicted eligible | Predicted file chunk | Predicted strategy stagnation | Predicted supervision journal | Row total |
|---|---:|---:|---:|---:|---:|
| Eligible near miss | 10 | 0 | 0 | 0 | 10 |
| File chunk | 0 | 32 | 0 | 0 | 32 |
| Strategy stagnation | 0 | 0 | 32 | 0 | 32 |
| Supervision journal | 0 | 0 | 0 | 32 | 32 |
| Column total | 10 | 32 | 32 | 32 | 106 |

The synthetic controls have 96 true positives, 10 true negatives, 0 false positives, and 0 false negatives. This validates the fixed randomized control set; it does not supply missing real-corpus ground truth.

## Snapshot-only content, provenance, and context associations

These features were computed from the pinned current snapshot. They are not decision-time evidence, were excluded from fitting, and must not be used to claim a historical causal or transferable effect. All values are present in all four cohorts, so their exact consumed/non-consumed denominators are the cohort denominators stated above. Notation remains `C / N; Δ`.

| Snapshot-only feature | Organic train C / N; Δ | Organic eval C / N; Δ | Observed-map eval C / N; Δ | Secondary anchor C / N; Δ |
|---|---|---|---|---|
| `form_file_chunk_envelope` | 0.02361200 / 0.00000000; +0.02361200 | 0.01520913 / 0.00000000; +0.01520913 | 0.00000000 / 0.04769475; −0.04769475 | 0.03912765 / 0.00359712; +0.03553052 |
| `form_strategy_stagnation` | 0 / 0; 0 | 0 / 0; 0 | 0.00000000 / 0.00635930; −0.00635930 | 0.00064144 / 0.00279776; −0.00215632 |
| `form_supervision_journal` | 0 / 0; 0 | 0 / 0; 0 | 0 / 0; 0 | 0 / 0; 0 |
| `form_machine_ballast` | 0.02361200 / 0.00000000; +0.02361200 | 0.01520913 / 0.00000000; +0.01520913 | 0.00000000 / 0.05405405; −0.05405405 | 0.03976908 / 0.00639488; +0.03337420 |
| `content_log_chars` | 7.27412610 / 6.89049594; +0.38363017 | 7.39901034 / 7.04349496; +0.35551537 | 6.91530311 / 7.03648097; −0.12117785 | 6.46432245 / 6.48498350; −0.02066105 |
| `content_log_lines` | 1.79193407 / 1.32369878; +0.46823529 | 1.80964802 / 1.43683890; +0.37280912 | 1.42599084 / 1.50943130; −0.08344045 | 1.05719735 / 0.89930548; +0.15789187 |
| `content_json_envelope` | 0.04530951 / 0.03583062; +0.00947889 | 0.10646388 / 0.08806489; +0.01839899 | 0.05263158 / 0.18918919; −0.13655761 | 0.08980115 / 0.03717026; +0.05263089 |
| `content_code_fence` | 0.02425016 / 0.00195440; +0.02229576 | 0.02471483 / 0.00463499; +0.02007983 | 0.00000000 / 0.05405405; −0.05405405 | 0.06221937 / 0.00879297; +0.05342641 |
| `provenance_log_key_count` | 1.18491951 / 0.84314975; +0.34176976 | 1.13824391 / 0.79837383; +0.33987008 | 0.59749089 / 0.82580318; −0.22831229 | 1.05377747 / 0.99139393; +0.06238354 |
| `provenance_log_source_trace_count` | 1.57454917 / 1.37800108; +0.19654809 | 1.70482135 / 1.39139720; +0.31342415 | 0.93042846 / 1.46875149; −0.53832303 | 1.64807995 / 1.68570771; −0.03762776 |
| `context_log_key_count` | 2.17407748 / 2.16166654; +0.01241094 | 2.08327642 / 2.06188690; +0.02138952 | 2.19749820 / 2.00147990; +0.19601830 | 2.10106969 / 2.08355765; +0.01751204 |
| `context_log_nested_count` | 0.00928915 / 0.01083748; −0.00154833 | 0.00790662 / 0.01044138; −0.00253476 | 0 / 0; 0 | 0.00400149 / 0.00729215; −0.00329066 |
| `context_log_sequence_count` | 0.43821020 / 0.38118732; +0.05702287 | 0.36671467 / 0.34267302; +0.02404165 | 0.31617240 / 0.26672292; +0.04944947 | 0.41609959 / 0.42108453; −0.00498495 |
| `context_log_scalar_count` | 2.45865808 / 2.37404659; +0.08461149 | 2.34164518 / 2.24267496; +0.09897022 | 2.37903357 / 2.13307235; +0.24596122 | 2.29293233 / 2.26471305; +0.02821928 |

## Recorded-score associations and unavailable cascade stage

Recorded delivery scores are historical for organic delivery results, but no residual score was recorded in any of the 686 observed-map payloads. They are therefore analysis-only and cannot establish transfer. For organic train the score denominators are 1,567 consumed and 1,535 non-consumed; for organic evaluation, 526 and 863. In the secondary-anchor diagnostic, scores are available for 3,723 items—1,510 consumed and 2,213 non-consumed—and missing for 338. Observed-map availability is 0 and missingness is 686.

| Recorded score | Organic train C / N; Δ | Organic eval C / N; Δ | Observed-map eval | Secondary anchor C / N; Δ |
|---|---|---|---|---|
| `score` | 1.37468760 / 0.90400123; +0.47068638 | 1.12355571 / 0.84388284; +0.27967287 | unavailable, 0/686 | 0.83723911 / 0.74971508; +0.08752404 |
| `bm25_score` | 0.04829810 / 0.09898715; −0.05068905 | 0.05400895 / 0.12857946; −0.07457051 | unavailable, 0/686 | 0.02978498 / 0.04973661; −0.01995163 |
| `vector_score` | 0.42067706 / 0.50210018; −0.08142312 | 0.41486141 / 0.50854578; −0.09368437 | unavailable, 0/686 | 0.47422133 / 0.49110656; −0.01688523 |
| `graph_score` | 0.06032358 / 0.03789395; +0.02242963 | 0.06228205 / 0.03600173; +0.02628032 | unavailable, 0/686 | 0.06934795 / 0.04853195; +0.02081600 |
| `trigger_score` | 0.35097320 / 0.09897394; +0.25199926 | 0.25000000 / 0.08256663; +0.16743337 | unavailable, 0/686 | 0.07509934 / 0.02829492; +0.04680441 |

`cascade_anchor` is unavailable for all 3,102 organic-train, 1,389 organic-evaluation, 686 observed-map-evaluation, and 4,061 secondary-anchor items. Historical cascade stage was not recorded at delivery and was not reconstructed.

## Complete omission and rejection ledger

The eight decision-time features listed in the model table are the entire frozen primary model surface. Every other measured feature is omitted from fitting as follows.

| Omitted feature | Exact reason |
|---|---|
| `form_file_chunk_envelope` | snapshot-only form classifier output; not recorded at delivery; ballast sensitivity diagnostic only |
| `form_strategy_stagnation` | snapshot-only form classifier output; not recorded at delivery; ballast sensitivity diagnostic only |
| `form_supervision_journal` | snapshot-only form classifier output; not recorded at delivery; ballast sensitivity diagnostic only |
| `form_machine_ballast` | snapshot-only union of form classifiers; not recorded at delivery; ballast sensitivity diagnostic only |
| `content_log_chars` | current-snapshot content size; historical value not versioned |
| `content_log_lines` | current-snapshot content size; historical value not versioned |
| `content_json_envelope` | current-snapshot content form; historical value not versioned |
| `content_code_fence` | current-snapshot content form; historical value not versioned |
| `provenance_log_key_count` | current-snapshot provenance shape; historical provenance not versioned |
| `provenance_log_source_trace_count` | current-snapshot provenance shape; historical provenance not versioned |
| `context_log_key_count` | current-snapshot context shape; historical context not versioned |
| `context_log_nested_count` | current-snapshot context shape; historical context not versioned |
| `context_log_sequence_count` | current-snapshot context shape; historical context not versioned |
| `context_log_scalar_count` | current-snapshot context shape; historical context not versioned |
| `cascade_anchor` | historical cascade stage not recorded at delivery and not reconstructed; unavailable in every arm |
| `score` | available for organic results but absent from every observed-map payload; no transfer cell |
| `bm25_score` | available for organic results but absent from every observed-map payload; no transfer cell |
| `vector_score` | available for organic results but absent from every observed-map payload; no transfer cell |
| `graph_score` | available for organic results but absent from every observed-map payload; no transfer cell |
| `trigger_score` | available for organic results but absent from every observed-map payload; no transfer cell |

Three fitted decision-time features are also omitted from any claim of transferable direction: `node_age_log_days`, `level_trace`, and `level_concept`, because their univariate association signs do not agree across train and both evaluation arms. The remaining five same-sign features are not declared deployable because the combined frozen model failed transfer.

The following values are rejected as predictors rather than treated as missing features:

| Rejected value | Reason |
|---|---|
| Node identifier | identity leakage and forbidden lookup behavior |
| Outcome label | target leakage |
| Task metadata | semantic identity leakage |
| Host | environment identity leakage |
| Source identity | source lookup leakage |
| Cache identity | grouping identity; used only to enforce split disjointness |
| Session identity | grouping identity; used only to enforce split disjointness |
| Current `access_count` | future-mutated after delivery and not reconstructible as of delivery |
| Current `usefulness_score` | future-mutated after delivery and not reconstructible as of delivery |
| Current `last_accessed` | future-mutated after delivery and not reconstructible as of delivery |
| `updated_at`-derived usage | future-mutated proxy and not reconstructible as of delivery |

## Aggregate-only publication audit

The published artifacts contain no item-level rows or corpus text. The privacy audit compared 110,313 raw strings against 660 published string cells and found zero exact raw-string matches. No actual identifier, query, task value, cache identity, session identity, raw content, or candidate-holdout observation is included here.
