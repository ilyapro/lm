# Recall applicability v2

P1: PASS; 3 improved independent groups; 0 new fact losses; 15 baseline misses.

| Holdout category | Baseline reached / required | Candidate reached / required | Irrelevant top four (baseline → candidate) | New losses | Baseline misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| absent-knowledge | 0 / 0 | 0 / 0 | 8 → 8 | 0 | 0 |
| compound | 2 / 5 | 2 / 5 | 7 → 7 | 0 | 3 |
| concrete | 2 / 4 | 4 / 4 | 7 → 6 | 0 | 2 |
| correction | 3 / 6 | 6 / 6 | 7 → 6 | 0 | 3 |
| cross-project | 5 / 5 | 5 / 5 | 6 → 6 | 0 | 0 |
| instruction | 5 / 5 | 5 / 5 | 6 → 5 | 0 | 0 |
| large-group | 0 / 7 | 0 / 7 | 8 → 8 | 0 | 7 |

## P1: independent applicability verdict

The new sealed holdout has 14 cases in 14 separate topic groups. Concrete,
correction, and instruction each lose one irrelevant top-four slot. The candidate
reaches 22 of 32 required clauses versus 17 of 32 for the baseline, with **zero
losses of any baseline-reachable clause**. Both instruction cases retain all five
required clauses, and both cross-project cases retain all five. The first paired
measurement and its immutable FAIL report remain historical facts; this PASS
describes only the independently sealed v2 packet.

The controls limit that conclusion. The compound cases stay at 2/5 clauses and
7 irrelevant slots in both arms. Absent-knowledge cases return 8 irrelevant
slots in both arms, and the two large-group cases reach 0/7 clauses in both
arms. Those large-group cases supply no proof of the accepted reading gain.

## P2: paired measurement and complete read chain

The private run receipt binds the original snapshot and v2 seal, exact
shape-only converted case input, raw output, runner hash, candidate receipt,
pre/post source tree and file hashes, actual command, UTC start/end times, and
the same effective recall and embedding settings for both arms. A consumption
record was written before launch. The runner used a separate writable copy of
the original frozen SQLite snapshot and fresh worker/session state per case and
arm, with alternating arm order and the same broad scope, depth 1, and top 4
limit. The ordinary trigger mode and installed embedding backend were used.

| Holdout total | Baseline | Candidate |
| --- | ---: | ---: |
| Recall calls / lookup calls / all calls | 14 / 48 / 62 | 14 / 46 / 60 |
| Complete response bytes | 426,486 | 280,868 |
| Necessary prefix calls in reached cases | 10 | 16 |
| Necessary prefix response bytes in reached cases | 62,709 | 87,863 |
| Exhausted calls in unreached positive cases | 24 | 13 |
| Exhausted response bytes in unreached positive cases | 181,811 | 56,822 |
| Summed warm latency, ms | 30,498.714 | 30,099.896 |
| Median case warm latency, ms | 2,160.657 | 2,035.327 |

Necessary-prefix totals are higher for the candidate because it reaches more
complete cases. The full per-category counts, every necessary recall/lookup
sequence, exhausted sequence, response-byte total, and warm latency are in the
machine-verified JSON report. Its bytes are JSON response bytes from the
runner, with no response content published. The helper's actual
`verify --require-success` command passed against the committed aggregate.

## P3: PASS by independent production diff review

I reviewed the two production files against accepted `b9769d8`. In
`retrieval.py`, the trigger still admits a schema when at least half its trigger
words appear, including a short applicable instruction inside a longer query.
The rank blend now scales that trigger by existing content vector evidence,
with the existing graph weight as a small floor, and removes the unconditional
1.8 multiplier. In `score_gate.py`, the ordinary channel score feeds that same
trigger contribution into the shared rank blend; the separate feedback-aware
trigger gate still permits sparse but applicable instructions. The name-mode
branch remains separate. The production delta is `retrieval.py` +18/-10 and
`score_gate.py` +10/-7 lines, including comments. Inspection found no default
scope restriction, history deletion, additional mandatory model call, or
provider/model/tier change.

## P4: targeted evidence only

The candidate's frozen receipt records 133 passing focused regressions across
applicability, gate, feedback, correction, delivery and carrier reading. In
this worktree the available focused files also passed, including executable
ordinary recall plus natural lookup cases for unrelated carriers, short
triggered instructions, cross-project facts, and long carrier delivery.
These tests are the evidence for the accepted large-carrier reading behavior;
the v2 field large-group cases were unreachable in both arms. The parent owns
the fresh declared-suite check and final P4 acceptance.
