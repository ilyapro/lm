# Frozen paired recall-cost verdict

**PASS on the frozen nine-case comparison.** Full-chain response characters fall from **194,277 to 96,040** (−98,237, 50.6%). The known large-carrier case falls from **67,007 to 12,745 characters**, with one recall and no lookup in either arm. The candidate preserves all required claims and does not add a mandatory call on any sufficient case.

## Reproduction and provenance

The accepted baseline is `cd4ad6c3984c41badc55199299532199ed4579b1`. The measured candidate is `0e6866cc8acb7f7f0b49e9aad87f3a9711192d39`, which contains delivery-chain-repair `f7dd10b`, report-verifier `34c2784`, carrier-sufficiency-repair `273f5a9`, and the production carrier-omission correction `0e6866c`. The latter changes `src/living_memory/delivery.py`; this is a fresh measurement of production bytes, not the previously rejected candidate.

The runner verified all frozen hashes, archived both code revisions separately, and used fresh writable copies of the same SQLite snapshot for each case. The original nine-case packet and independent oracle were unchanged. A separate single-case development diagnostic helped identify the omission and passed against this candidate; **the known carrier is development evidence, not independent holdout evidence**. The three holdout cases remain disjoint and are reported separately. Private queries, memory text, and retrieval IDs stay local.

| Frozen input | SHA-256 |
| --- | --- |
| Private case packet | `a54df4594910edb62d421fe7eec1add260022a6779809cdf710eef5619fd5d64` |
| Public synthetic fixtures | `972bd2e0e33baba2d2a03d2c0238deb46eae92bf303582943bbaf042f285b293` |
| SQLite snapshot | `2891a098b23c7ad0ef2a1c64a8bdfdd9dadccdccf4e2693afd31b09e55f09e69` |

```bash
python3 scripts/recall_needed_cost.py run \
  --packet /home/sfx/p/ae/artifacts/recall-needed-cost/private/cases.json \
  --packet-sha256 a54df4594910edb62d421fe7eec1add260022a6779809cdf710eef5619fd5d64 \
  --fixtures artifacts/recall-needed-cost/fixtures.json \
  --fixtures-sha256 972bd2e0e33baba2d2a03d2c0238deb46eae92bf303582943bbaf042f285b293 \
  --snapshot /home/sfx/p/ae/artifacts/recall-needed-cost/private/snapshot.sqlite3 \
  --snapshot-sha256 2891a098b23c7ad0ef2a1c64a8bdfdd9dadccdccf4e2693afd31b09e55f09e69 \
  --baseline-ref cd4ad6c3984c41badc55199299532199ed4579b1 \
  --candidate-ref 0e6866cc8acb7f7f0b49e9aad87f3a9711192d39 \
  --out /tmp/paired-private.json --aggregate-out /tmp/paired-aggregate.json
```

## Full-chain comparison

Each character count includes every recall response and necessary lookup response through the independent oracle's sufficiency decision. Calls are recall+lookup.

| Case | Split | Family | Base chars | Candidate chars | Base recall+lookup | Candidate recall+lookup | Base ms | Candidate ms | Oracle pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | development | known_large_carrier | 67,007 | 12,745 | 1+0 | 1+0 | 14,373.4 | 7,015.5 | true/true |
| 2 | development | ordinary_short_fact | 8,390 | 8,390 | 1+0 | 1+0 | 3,598.9 | 2,142.8 | true/true |
| 3 | development | ordinary_short_fact | 8,297 | 8,528 | 1+0 | 1+0 | 4,091.6 | 2,002.0 | true/true |
| 4 | development | buried_applicable_instruction | 28,101 | 12,522 | 1+0 | 1+0 | 2,082.4 | 1,940.3 | true/true |
| 5 | development | obsolete_rule_with_correction | 16,319 | 11,138 | 1+0 | 1+0 | 1,896.8 | 2,238.2 | true/true |
| 6 | development | missing_knowledge | 9,915 | 10,064 | 1+0 | 1+0 | 2,124.6 | 2,381.4 | true/true |
| 7 | holdout | ordinary_short_fact | 16,274 | 10,807 | 1+0 | 1+0 | 1,905.3 | 2,105.3 | true/true |
| 8 | holdout | buried_applicable_instruction | 29,931 | 11,563 | 1+1 | 1+0 | 1,933.5 | 1,698.2 | true/true |
| 9 | holdout | missing_knowledge | 10,043 | 10,283 | 1+0 | 1+0 | 1,684.6 | 1,745.7 | true/true |

| Split | Base chars | Candidate chars | Change | Base calls | Candidate calls | Base ms | Candidate ms | Oracle pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| development | 138,029 | 63,387 | -74,642 | 6 (6+0) | 6 (6+0) | 28,167.7 | 17,720.2 | 6/6 |
| holdout | 56,248 | 32,653 | -23,595 | 4 (3+1) | 3 (3+0) | 5,523.3 | 5,549.2 | 3/3 |
| all | 194,277 | 96,040 | -98,237 | 10 (9+1) | 9 (9+0) | 33,691.0 | 23,269.3 | 9/9 |

All **seven positive cases** reach sufficient knowledge; both missing-knowledge controls remain missing. Buried applicable instructions and the current correction survive. Ordered retrieval IDs are identical across all nine cases: **zero ranking changes**. Character savings span five cases in development and holdout; they do not rest on the diagnosed carrier alone. Total calls fall from 10 to 9, with recall calls 9→9 and necessary lookup calls 1→0. Latency is a single sequential replay per arm and is descriptive only.

## Independent checks and complexity

The frozen oracle uses source clauses independently of delivery code. Synthetic loss probes remove an applicable buried instruction and an explicit correction; both fail sufficiency. The combined evaluator, delivery, protocol, verifier, carrier-sufficiency and carrier-omission suite passed **195 tests** with no failures or timeouts. The executable `python3 scripts/recall_needed_cost.py verify --report artifacts/recall-needed-cost/report.json` validates the published report.

Against merge-base `cd4ad6c3984c41badc55199299532199ed4579b1`, `src/living_memory/delivery.py` and `src/living_memory/server.py` together grow from **2,816 to 2,918 production lines** (+102 net; 123 additions, 21 deletions), **111 to 114 functions** (+3), and **283 to 308 AST branch nodes** (+25). The branch proxy counts `If`, loops, `Try`, `ExceptHandler`, `With`, `IfExp`, `BoolOp`, and `Match` syntax; it is not a runtime complexity measure.

## Verdict and limits

The frozen comparison meets the measured P3 and P4 conditions: aggregate full-chain characters decrease across multiple cases, no sufficient case gains a mandatory call or costly forced lookup, all nine independent oracle checks pass, ranking is unchanged, and the combined targeted suite and report verifier pass. The diagnosed carrier is development evidence. The three independent holdout cases support savings in this packet, but the packet is too small to claim population-wide field benefit. No private memory content, query, or retrieval identifier enters these committed artifacts.
