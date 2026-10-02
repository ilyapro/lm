# Frozen v2 procedure-retention diagnosis

The hash-backend regression has **two positive read-chain losses** and one
negative mutant assertion whose expected loss is obsolete. This diagnosis does
not establish P4. The sealed v2 field PASS and the original field FAIL were
left untouched; neither private packet was rerun or inspected for tuning.

## Reproduction and provenance

The only state used here is each test's synthetic fixture. Each arm starts in a
new process and fixture database, uses the same query, scope, depth (default 1),
`max_results`, feedback and consolidation calls, then follows ordinary public
`memory_recall` results through their natural `content_ref` lookup. `src` from
actual commit `b9769d84e3188ee1e646627ebe1b4d454f69d6f9` was extracted
with `git archive`; an explicit import before pytest defeats this checkout's
root `living_memory` shim taking precedence over `PYTHONPATH`. The candidate
import was the current frozen `src` tree
`a054ce1b7e3e83a1ac543718c28d67ed442302e6`.

Run from this worktree (the three selectors are the three JSON case keys):

```bash
mkdir -p /tmp/lm-retention-baseline
git archive b9769d8 src | tar -x -C /tmp/lm-retention-baseline
selectors=(
  tests/test_procedure_case_separation.py::test_taught_recipe_arrives_whole_through_its_legacy_schema_turned_carrier
  tests/test_procedure_case_separation.py::test_crowding_oracle_detects_the_group_identity_mutant
  tests/test_procedure_case_separation.py::test_group_node_keeps_the_trigger_it_is_found_by
)
for backend in hash auto; do
  for source in /tmp/lm-retention-baseline/src "$PWD/src"; do
    LIVING_MEMORY_EMBEDDING_BACKEND="$backend" PYTHONPATH="$PWD/.cache/python-deps" \
      python3 -c 'import sys; sys.path.insert(0, sys.argv.pop(1)); import living_memory.retrieval as r; print("IMPORT", r.__file__); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))' \
      "$source" -q -o "pythonpath=$source" "${selectors[@]}"
  done
done
git rev-parse HEAD:src
sha256sum src/living_memory/{retrieval,score_gate}.py
sha256sum /tmp/lm-retention-baseline/src/living_memory/{retrieval,score_gate}.py
```

`hash` is the declared `scripts/test.sh` backend. The baseline imports printed
`/tmp/lm-retention-baseline/src/living_memory/retrieval.py` and passed `...`;
the candidate imports printed this worktree's `src/.../retrieval.py` and failed
`FFF`. `auto` loaded the installed
`paraphrase-multilingual-MiniLM-L12-v2` encoder (384 dimensions, not a hash
fallback): baseline `...`, candidate `.FF`. No provider, model or default was
changed. The candidate source hashes are `60c8a75222ba0384284a1048d59a27dd14878be147418a949443f92184ba26f1`
for retrieval.py and `ba12268ada15217bc43fc9f2a88c4ab8c4bd09c47960f0e7b953d50fa4a40f67`
for score_gate.py; the JSON also records the actual baseline file hashes.

For the stage trace, temporary process-local wrappers observed
`MemoryRecallService.rank_candidates`, `collapse_schema_duplicates`, and
`apply_score_gate` around the unchanged public `mcp.tools["memory_recall"]`
call. They did not replace scores or decisions. The returned `content_ref`
was looked up only when recall delivered it. Scores below are representative
fresh hash fixtures; transient node IDs, timestamps and vector scores vary
slightly. The rank and first-loss stage did not.

| Test fixture | Baseline rank and score | Candidate rank and score | First loss |
| --- | --- | --- | --- |
| Taught ledger correction | In-place carrier 1/11, 2.769 | Carrier 7/11, ~0.218; own correction trace ~10/11, ~0.128 | `max_results=5` truncates carrier after ranking; correction trace has gate score ~0.051 against 0.35 |
| Crowding with group-identity mutant | Ten carriers 1–10 at 2.16; recipe trace 11 at ~1.072 | Recipe trace 1 at ~1.072; ten carriers 2–11 at ≤~0.57 | Negative assertion stops being a valid loss detector at ranking |
| Preserved relay trigger | In-place carrier 1/18, 2.16 | Carrier 14/18, ~0.226 | `max_results=10` truncates carrier after ranking |

In both positive cases the carrier was **collected** with `trigger_score=1`,
retained its group identity, original trigger and full contents, and survived
deduplication. Its trigger-gate score exceeded the trigger threshold (about
1.35 for ledger and 1.05 for relay), but gate delivery cannot select it once
earlier results fill the requested slots. The public response therefore has no
carrier `content_ref` for a natural lookup. The ledger correction's own trace
does not rescue it: unrelated feedback and weak text match leave that direct
route below the score gate. This is loss of a current taught correction, not
an assertion about an obsolete version of the recipe. The relay case likewise
loses the intact old-label carrier, despite its trigger surviving both passes.

The candidate replaces the baseline's almost fixed trigger rank floor and
1.8 multiplier with content-supported trigger scoring in the shared blend.
That change explains the intended crowding improvement and both carrier rank
drops. It is **not** a collection, consolidation, deduplication or delivery
format defect. With the installed encoder, stronger ledger content similarity
raises that carrier to rank 2, so the first test passes; the relay carrier
remains at rank 14, and the positive loss persists. The mutant still fails:
breaking title dedup retains ten unrelated carriers, but the directly
applicable recipe ranks first on both backends. Its former assertion required
the wrong behavior—those carriers suppressing a useful direct trace. A new
mutant oracle should verify dedup's actual identity and slot-count effect
without requiring recipe suppression.

## Smallest cohesive repair recommendation

Keep the v2 content-sensitive ranking for generic trigger matches. In the
existing rank/admission path, restore a bounded route for a carrier reached by
its **complete saved trigger** when its full group evidence is otherwise
unreachable, using existing source relevance and feedback rather than the
old unconditional trigger floor. Exercise it against both current-correction
and preserved-label fixtures, plus an irrelevant same-title history control
and the compound questions already in the applicability regressions. Recast
the group-identity mutant assertion to require distinct carriers or excess
slots after dedup is broken; retain the positive direct-recipe assertion.
The exact ranking rule needs independent verification before adoption: merely
raising every carrier with a matching procedure name would recreate the root
applicability defect.

Any production repair changes at least `src/living_memory/retrieval.py` (and
possibly the shared gate in `score_gate.py`). It would invalidate the v2
source-bound `candidate-v2.json`/`.md` receipt, the `report-v2.json`/`.md`
candidate verdict and `--require-success` verification for a new source tree,
and the private v2 run receipt/raw measurement's candidate provenance. P1/P2
acceptance for that new tree and P4's focused plus declared-suite checks would
need new authorized evidence; the archived v2 PASS remains historical evidence
for its frozen tree. The parent must decide that scope and evidence path. This
node made no production or test edits.
