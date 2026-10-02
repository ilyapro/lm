SPEC:
- id: P1
  statement: Ordinary recall improves applicability for concrete and compound questions without requiring an internal
    ID or a shortened query. On a sealed holdout, at least two independent topic groups lose irrelevant top-four
    results, relevant facts do not become less reachable, and every baseline-reachable instruction, correction,
    and cross-project fact remains reachable through necessary recall and lookup calls.
  kind: capability
  check_protocol: measure
  train_set: The reported P5_RUSSIAN examples and synthetic development cases.
  eval_set: Frozen development questions from the local snapshot, disjoint from holdout answer groups.
  holdout_set: At least twelve sealed questions with disjoint queries and answer groups, covering concrete and compound
    questions, trigger-applicable instructions, corrections, cross-project transfer, absent knowledge, and large
    groups.
  generalization_threshold: Improvement in at least two independent holdout topic groups, with zero losses of baseline-reachable
    required facts.
  non_falsifiable: false
  met: true
- id: P2
  statement: Accepted baseline b9769d8 and candidate are compared on one frozen state with identical recall parameters
    and isolated prior-call effects; the report records relevance, losses, all necessary recall and lookup calls,
    response volume, latency, and production-code complexity.
  kind: verification
  check_protocol: measure
  non_falsifiable: false
  met: true
- id: P3
  statement: The repair uses existing applicability and ranking rules without enabling LM_RECALL_SCHEMA_TRIGGER=name,
    narrowing default scope, deleting history, adding a mandatory LLM call, or changing providers, models, or tiers.
  kind: refactor
  check_protocol: external_review
  non_falsifiable: false
  met: true
- id: P4
  statement: Focused regressions preserve used/irrelevant feedback, correction dominance, full lookup, and the accepted
    reading gain for large groups; the changed code passes targeted checks and the repository's declared suite.
  kind: fix
  check_protocol: exec
  non_falsifiable: false
  met: false
COVERAGE:
  forward:
    P1:
    - sealed-corpus
    - retrieval-repair
    - sealed-corpus-v2
    - retrieval-revision
    - paired-measurement-v2
    P2:
    - sealed-corpus
    - paired-runner
    - sealed-corpus-v2
    - paired-verdict
    - paired-measurement-v2
    P3:
    - retrieval-repair
    - retrieval-revision
    - paired-measurement-v2
    P4:
    - paired-runner
    - retrieval-repair
    - retrieval-revision
    - paired-verdict
    - procedure-retention-diagnosis
    - anchor-reference-compatibility
  reverse:
    sealed-corpus:
    - P1
    - P2
    paired-runner:
    - P2
    - P4
    retrieval-repair:
    - P1
    - P3
    - P4
    sealed-corpus-v2:
    - P1
    - P2
    retrieval-revision:
    - P1
    - P3
    - P4
    paired-verdict:
    - P2
    - P4
    paired-measurement-v2:
    - P1
    - P2
    - P3
    procedure-retention-diagnosis:
    - P4
    anchor-reference-compatibility:
    - P4
RESULT: DECOMPOSE
SUBGOALS:
- acceptance: []
  blocked_patterns: []
  depends_on: []
  description: 'Freeze a local recall snapshot and preregister an independent applicability corpus.

    Record baseline outcomes before seeing candidate results. Keep the snapshot, questions, facts, and node IDs
    in local artifacts only; commit a data-free protocol and receipt. Seal disjoint development and holdout answer
    groups. Include the reported failure as development evidence and independent concrete, compound, instruction,
    correction, cross-project, absent-knowledge, and large-group cases.'
  evidence_of_completion:
  - check: test -s artifacts/recall-applicability/prereg.md && test -s /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
      && test -s /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    kind: exec
  expected_mode: execute
  id: sealed-corpus
  interface_in:
    files_read:
    - src/living_memory/retrieval.py
    - scripts/recall_precision_replay.py
    - tests/test_recall_delivery_chain.py
  interface_out:
    files_owned:
    - scripts/recall_applicability_corpus.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
  invariants_preserved:
  - No sfx memory contents or project data enter tracked files, alt, or a remote repository.
  phase: design
  postcondition: A fixed local snapshot and sealed gold set exist, with a committed preregistration and snapshot
    hash.
  postcondition_targets:
  - P1
  - P2
  priority: 5
  required_artifacts:
  - scripts/recall_applicability_corpus.py
  - artifacts/recall-applicability/prereg.md
  - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
  - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
  verify_ui_url: ''
  rationale: Retain the completed first snapshot/corpus contract and all original required artifacts. Its broad
    private-directory grant is narrowed to the archived first-attempt files so the new packet and run have distinct
    owners. The consumed holdout remains historical failed evidence. Custody of its two archived aggregate reports
    is added here only to give each frozen path one owner; no completed work is relaunched or evidence edited.
- acceptance:
  - python -m pytest -q tests/test_recall_applicability_runner.py
  blocked_patterns: []
  depends_on: []
  description: 'Build a generic paired recall and necessary-lookup measurement runner.

    Accept a private JSON case file with case_id, query, scope, depth, max_results, category, required facts, and
    oracle-only acceptable source IDs. Run b9769d8 and candidate code in separate processes on separate copies of
    the same SQLite snapshot. Use identical environment and recall parameters, fresh sessions, controlled embedding
    warmup, and alternating arm order. Report fact reachability, relevant and irrelevant ranks, complete response
    bytes, call counts, and warm latency; test the runner on synthetic data.'
  evidence_of_completion:
  - check: python -m pytest -q tests/test_recall_applicability_runner.py
    kind: exec
  expected_mode: execute
  id: paired-runner
  interface_in:
    files_read:
    - scripts/recall_precision_replay.py
    - src/living_memory/server.py
    - src/living_memory/delivery.py
    - tests/test_recall_delivery_chain.py
  interface_out:
    files_owned:
    - scripts/recall_applicability_eval.py
    - tests/test_recall_applicability_runner.py
  phase: build
  postcondition: The runner produces and validates paired measurements from a synthetic fixture without relying
    on queried internal IDs.
  postcondition_targets:
  - P2
  - P4
  priority: 5
  required_artifacts:
  - scripts/recall_applicability_eval.py
  - tests/test_recall_applicability_runner.py
  verify_ui_url: ''
  rationale: Retain the completed synthetic paired-runner contract and implementation. The new verdict helper consumes
    its existing output; it does not replace or relaunch this successful node.
- acceptance:
  - python -m pytest -q tests/test_recall_applicability.py tests/test_recall_score_gate.py tests/test_retrieval_cross_scope_gate.py
    tests/test_schema_trigger_by_name.py
  blocked_patterns: []
  depends_on: []
  description: 'Repair ordinary recall so topic applicability governs trigger-bearing memory.

    Investigate the shared ranking, admission, and gate mechanism; unify existing rules where possible. Preserve
    genuinely applicable trigger instructions, cross-project search, corrections, feedback, full lookup, and large-group
    reading. Add focused synthetic regressions including an unrelated history carrier, a compound question, a trigger-only
    applicable instruction, a correction, and a cross-project fact. Keep the name valve disabled and avoid new providers,
    models, tiers, or mandatory LLM calls.'
  evidence_of_completion:
  - check: python -m pytest -q tests/test_recall_applicability.py tests/test_recall_score_gate.py tests/test_retrieval_cross_scope_gate.py
      tests/test_schema_trigger_by_name.py
    kind: exec
  expected_mode: execute
  id: retrieval-repair
  interface_in:
    files_read:
    - docs/recall-schema-trigger.md
    - tests/test_recall_delivery_chain.py
    - tests/test_schema_trigger_by_name.py
    - src/living_memory/retrieval.py
    - src/living_memory/score_gate.py
    - tests/test_recall_applicability.py
    - tests/test_recall_score_gate.py
    - tests/test_retrieval_cross_scope_gate.py
  interface_out:
    files_owned: []
  out_of_scope:
  - Deleting historical memory or restricting the default search scope.
  - Enabling LM_RECALL_SCHEMA_TRIGGER=name.
  phase: build
  postcondition: The focused retrieval tests pass and the production diff explains one general applicability mechanism.
  postcondition_targets:
  - P1
  - P3
  - P4
  priority: 5
  required_artifacts:
  - tests/test_recall_applicability.py
  verify_ui_url: ''
  rationale: Retain the completed first repair and its unchanged synthetic-test postcondition as historical evidence.
    Transfer all future production and regression-file writing to retrieval-revision; completion of these synthetic
    tests never establishes P1 generalization.
- id: sealed-corpus-v2
  phase: design
  expected_mode: execute
  depends_on: []
  description: 'Seal a new independent applicability holdout on the preserved snapshot.

    The first holdout is consumed and failed. Preserve df445b7 and every original sealed byte. Reuse the four original
    development cases unchanged, and author a bounded new holdout (prefer fourteen cases, at least twelve) on the
    exact original frozen snapshot. Its queries, answer sources, and semantic topic groups must be disjoint from
    every original development and holdout case. Exclude equivalent carrier/child-source and supersedes families,
    not merely renamed group labels. Cover concrete, compound, short-trigger applicable mandatory instructions,
    corrections, useful cross-project transfer, absent knowledge, and large groups. Validate every literal required
    clause against its source before sealing.

    Work independently of candidate development: do not inspect retrieval-revision''s changes, run any candidate
    on this packet, publish private case details to memory, or use candidate results to select cases. Use only accepted
    b9769d8 for the baseline capture, executing its actual archived src rather than relying on the corpus script''s
    hardcoded code_head field. Freeze all questions and predicates before baseline capture, and record baseline
    reachability for protected categories; a packet with no baseline-reachable instruction, correction, or cross-project
    item cannot establish retention and must be reported inadequate before candidate exposure. Do not collect a
    multi-day corpus or search repeatedly for favorable outcomes.

    Write private v2/goldset.json in the existing corpus schema, v2/baseline-outcomes.json, and v2/seal.json. The
    seal has schema_version 1, sealed_at_utc, full baseline_commit, baseline_src_tree, and files mapping snapshot,
    goldset, previous_goldset, baseline_outcomes to absolute path and sha256. The snapshot entry points to the original
    frozen database. Commit a data-free prereg-v2.md containing all four hashes, source-family exclusion method,
    baseline-code provenance, fixed recall/environment parameters, independent-authoring procedure, the unchanged
    two-group/zero-loss rule, and a one-shot measurement policy. Keep files local on sfx with restrictive permissions.
    Do not modify the old corpus validator or publish sfx questions, claims, node IDs, or raw responses.'
  postcondition: A newly sealed, independently authored and source-disjoint holdout and baseline receipt exist on
    the unchanged snapshot, with committed hashes and the original success threshold.
  postcondition_targets:
  - P1
  - P2
  interface_in:
    files_read:
    - scripts/recall_applicability_corpus.py
    - scripts/recall_applicability_eval.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
    files_immutable:
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
  interface_out:
    files_owned:
    - artifacts/recall-applicability/prereg-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/
  required_artifacts:
  - artifacts/recall-applicability/prereg-v2.md
  - /home/sfx/p/ae/artifacts/recall-applicability/v2/goldset.json
  - /home/sfx/p/ae/artifacts/recall-applicability/v2/baseline-outcomes.json
  - /home/sfx/p/ae/artifacts/recall-applicability/v2/seal.json
  evidence_of_completion:
  - kind: exec
    check: python scripts/recall_applicability_corpus.py validate --snapshot /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
      --corpus /home/sfx/p/ae/artifacts/recall-applicability/v2/goldset.json
  - kind: exec
    check: "python - <<'CHECK'\nimport hashlib, json\nfrom pathlib import Path\np=Path('/home/sfx/p/ae/artifacts/recall-applicability')\n\
      old=json.loads((p/'goldset.json').read_text())['cases']\nnew=json.loads((p/'v2/goldset.json').read_text())['cases']\n\
      hold=[c for c in new if c['split']=='holdout']\nassert len(hold)>=12\nassert [c for c in new if c['split']=='development']==[c\
      \ for c in old if c['split']=='development']\nfor key in ('query','topic_group'):\n    assert not {c[key]\
      \ for c in old} & {c[key] for c in hold}, key\nassert not {s for c in old for s in c['acceptable_source_ids']}\
      \ & {s for c in hold for s in c['acceptable_source_ids']}\nseal=json.loads((p/'v2/seal.json').read_text())\n\
      assert seal['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\nexpected={'snapshot':p/'sfx-frozen.sqlite3','goldset':p/'v2/goldset.json','previous_goldset':p/'goldset.json','baseline_outcomes':p/'v2/baseline-outcomes.json'}\n\
      for key,path in expected.items():\n    entry=seal['files'][key]\n    assert Path(entry['path']).resolve()==path.resolve(),\
      \ key\n    with path.open('rb') as f:\n        assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256'],\
      \ key\nassert seal['files']['snapshot']['sha256']=='3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef'\n\
      assert seal['files']['previous_goldset']['sha256']=='c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c'\n\
      prereg=Path('artifacts/recall-applicability/prereg-v2.md').read_text()\nassert all(seal['files'][k]['sha256']\
      \ in prereg for k in expected)\nprint('New seal hashes and disjoint holdout validated; no candidate was measured\
      \ by this check.')\nCHECK"
  acceptance:
  - python scripts/recall_applicability_corpus.py validate --snapshot /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    --corpus /home/sfx/p/ae/artifacts/recall-applicability/v2/goldset.json
  - "python - <<'CHECK'\nimport hashlib, json\nfrom pathlib import Path\np=Path('/home/sfx/p/ae/artifacts/recall-applicability')\n\
    old=json.loads((p/'goldset.json').read_text())['cases']\nnew=json.loads((p/'v2/goldset.json').read_text())['cases']\n\
    hold=[c for c in new if c['split']=='holdout']\nassert len(hold)>=12\nassert [c for c in new if c['split']=='development']==[c\
    \ for c in old if c['split']=='development']\nfor key in ('query','topic_group'):\n    assert not {c[key] for\
    \ c in old} & {c[key] for c in hold}, key\nassert not {s for c in old for s in c['acceptable_source_ids']} &\
    \ {s for c in hold for s in c['acceptable_source_ids']}\nseal=json.loads((p/'v2/seal.json').read_text())\nassert\
    \ seal['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\nexpected={'snapshot':p/'sfx-frozen.sqlite3','goldset':p/'v2/goldset.json','previous_goldset':p/'goldset.json','baseline_outcomes':p/'v2/baseline-outcomes.json'}\n\
    for key,path in expected.items():\n    entry=seal['files'][key]\n    assert Path(entry['path']).resolve()==path.resolve(),\
    \ key\n    with path.open('rb') as f:\n        assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256'],\
    \ key\nassert seal['files']['snapshot']['sha256']=='3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef'\n\
    assert seal['files']['previous_goldset']['sha256']=='c0b7efa1cb4337154152cee32b48e5be10c580d3aa80e065f18ad8b9af279f3c'\n\
    prereg=Path('artifacts/recall-applicability/prereg-v2.md').read_text()\nassert all(seal['files'][k]['sha256']\
    \ in prereg for k in expected)\nprint('New seal hashes and disjoint holdout validated; no candidate was measured\
    \ by this check.')\nCHECK"
  out_of_scope:
  - Candidate production changes or evaluation against candidate code.
  - Altering the original holdout, its failed report, or the two-group/zero-loss threshold.
  rationale: A consumed failed holdout cannot be repaired by rewriting a report. This new population satisfies the
    unchanged parent P1 contract for a sealed independent holdout while preserving the historical failure.
- id: retrieval-revision
  phase: build
  expected_mode: execute
  depends_on: []
  description: 'Repair shared recall applicability without losing short triggered instructions.

    Own one cohesive repair to collection, ranking, and admission/gating in retrieval.py and score_gate.py, with
    its focused regressions and mechanism explanation. The first repair only added matching-token-count/query-coverage
    admission; it kept the near-constant trigger floor and 1.8x multiplier. Its only trigger-only instruction test
    matched three words, so it could not catch a short valid trigger diluted by a compound question. The archived
    holdout lost two instruction clauses and improved only one topic group. Treat these aggregates as the counterexample
    to the approach, without inspecting or tuning on its private holdout cases.

    A probable path is to distinguish an applicability signal that keeps an instruction available from an unconditional
    rank advantage for any historical carrier, using the existing semantic/lexical/graph, authority/correction,
    feedback, and gate machinery. Investigate where the existing rank and gate use different evidence and consolidate
    that logic where justified. Do not simply replace the token-count cutoff with another keyword/count list, require
    three trigger tokens, blanket-suppress trigger-only instructions, or enable the name channel. This is solution
    guidance, not a requirement to prove one selected hypothesis.

    Use synthetic development cases and only the four original development cases extracted by split without exposing
    holdout records. Add contrasting short-trigger instructions and irrelevant historical carriers under both concrete
    and compound queries, with irrelevant clauses appended and topics renamed; include current corrections and cross-project
    answers. Exercise ordinary public memory_recall plus natural content_ref lookup, not only injected ranking scores.
    Preserve feedback and supersedes, full lookup bytes/authority, original memories, and the accepted large-carrier
    reading gain. Keep baseline-reachable development facts reachable. The previous tests'' observable safety promises
    must keep passing; replace implementation-coupled assertions only with stronger end-to-end assertions, documenting
    why.

    This node is the sole future owner of all first-repair production/regression paths plus tests/test_recall_delivery_chain.py
    and tests/test_recall_carrier_omission.py, resolving the unowned delivery-chain warning. Freeze the tested candidate
    before independent measurement. Commit candidate-v2.json with full baseline_commit, candidate_commit, src_tree
    (git rev-parse HEAD:src), and source_sha256 for retrieval.py and score_gate.py; candidate-v2.md explains the
    general mechanism, development outcomes, executed targeted checks, production delta and rejected alternatives.
    Commit source changes immediately and then commit the receipt. Never inspect the v2 packet or any private holdout
    measurements. The parent will run npm run check after integration; a targeted green suite is not P1 field success.'
  postcondition: A revised general applicability mechanism is frozen in a matching candidate receipt; short applicable
    instructions and irrelevant carriers are distinguished in executable ordinary-recall regressions while feedback,
    corrections, lookup and large-group reading checks pass.
  postcondition_targets:
  - P1
  - P3
  - P4
  interface_in:
    files_read:
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_corpus.py
    - tests/test_recall_feedback_loop.py
    - tests/test_explicit_feedback.py
    - tests/test_transport_feedback_closure_e2e.py
    - tests/test_retrieval_feedback_amplification.py
    - src/living_memory/server.py
    - src/living_memory/delivery.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    files_immutable:
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
  interface_out:
    files_owned:
    - src/living_memory/retrieval.py
    - src/living_memory/score_gate.py
    - docs/recall-schema-trigger.md
    - tests/test_recall_applicability.py
    - tests/test_recall_score_gate.py
    - tests/test_retrieval_cross_scope_gate.py
    - tests/test_schema_trigger_by_name.py
    - tests/test_recall_delivery_chain.py
    - tests/test_recall_carrier_omission.py
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/candidate-development/
  required_artifacts:
  - artifacts/recall-applicability/candidate-v2.json
  - artifacts/recall-applicability/candidate-v2.md
  - tests/test_recall_applicability.py
  evidence_of_completion:
  - kind: exec
    check: python -m pytest -q tests/test_recall_applicability.py tests/test_recall_score_gate.py tests/test_retrieval_cross_scope_gate.py
      tests/test_schema_trigger_by_name.py tests/test_recall_delivery_chain.py tests/test_recall_carrier_omission.py
      tests/test_recall_feedback_loop.py tests/test_explicit_feedback.py tests/test_transport_feedback_closure_e2e.py
      tests/test_retrieval_feedback_amplification.py
  - kind: exec
    check: "python - <<'CHECK'\nimport hashlib, json, subprocess\nfrom pathlib import Path\nr=json.loads(Path('artifacts/recall-applicability/candidate-v2.json').read_text())\n\
      assert r['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\nassert r['src_tree']==subprocess.check_output(['git','rev-parse','HEAD:src'],text=True).strip()\n\
      expected={'src/living_memory/retrieval.py','src/living_memory/score_gate.py'}\nassert set(r['source_sha256'])==expected\n\
      for name in expected:\n    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==r['source_sha256'][name]\n\
      assert Path('artifacts/recall-applicability/candidate-v2.md').stat().st_size>0\nprint('Candidate source freeze\
      \ matches current production tree.')\nCHECK"
  acceptance:
  - python -m pytest -q tests/test_recall_applicability.py tests/test_recall_score_gate.py tests/test_retrieval_cross_scope_gate.py
    tests/test_schema_trigger_by_name.py tests/test_recall_delivery_chain.py tests/test_recall_carrier_omission.py
    tests/test_recall_feedback_loop.py tests/test_explicit_feedback.py tests/test_transport_feedback_closure_e2e.py
    tests/test_retrieval_feedback_amplification.py
  - "python - <<'CHECK'\nimport hashlib, json, subprocess\nfrom pathlib import Path\nr=json.loads(Path('artifacts/recall-applicability/candidate-v2.json').read_text())\n\
    assert r['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\nassert r['src_tree']==subprocess.check_output(['git','rev-parse','HEAD:src'],text=True).strip()\n\
    expected={'src/living_memory/retrieval.py','src/living_memory/score_gate.py'}\nassert set(r['source_sha256'])==expected\n\
    for name in expected:\n    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==r['source_sha256'][name]\n\
    assert Path('artifacts/recall-applicability/candidate-v2.md').stat().st_size>0\nprint('Candidate source freeze\
    \ matches current production tree.')\nCHECK"
  invariants_preserved:
  - No name-channel activation, provider/model/tier change, extra mandatory LLM call, hidden scope restriction,
    memory deletion, or project/word exceptions.
  - No original or replacement holdout tuning; synthetic and declared development data alone may guide the implementation.
  out_of_scope:
  - Changing evaluation or success thresholds, the paired runner, corpus, verdict helper, or sealed measurements.
  rationale: The original synthetic repair contract still holds, but field evidence refutes its sufficiency. A separately
    owned revision supplies the missing capability prerequisite and executable P4 regression checks without pretending
    the failed report can fix production.
- id: paired-verdict
  phase: build
  expected_mode: execute
  depends_on: []
  description: 'Add an executable aggregate verdict contract for independently sealed recall measurements.

    Build scripts/recall_applicability_verdict.py and synthetic tests as a bounded consumer of the completed paired
    runner; do not replace that runner or alter production. The old report-only acceptance had no executable commands,
    and the runner has no verify subcommand. Supply build and verify CLI commands for v2 evidence. build accepts
    --seal, --raw, --run-receipt, --candidate-receipt, --output and --markdown; verify accepts --report and --require-success.
    Document this interface in the script. Work only from synthetic cases, existing schemas and the public aggregate
    report, not private field queries or candidate performance.

    Consume the existing raw runner schema, the corpus schema, the seal files mapping (snapshot, goldset, previous_goldset,
    baseline_outcomes), and candidate-v2.json (baseline_commit, candidate_commit, src_tree, source_sha256). Bind
    the aggregate report to their actual file hashes and code provenance. Recompute fact losses as baseline reached
    indexes minus candidate reached indexes for every case, check indexes/denominators, and count distinct holdout
    topic_group values with reduced irrelevant_top_four. Require at least two improved independent groups and zero
    baseline-reachable fact losses; do not count development cases, duplicate group names, smaller response bytes,
    or self-reported PASS flags as success. Verify holdout disjointness, category coverage, exact shape-only runner-input
    conversion and protected-category baseline reachability. Report baseline misses separately from new losses.

    Emit category aggregates and necessary-prefix/exhausted-read-chain metrics for relevance, facts, instruction/correction/cross-project
    losses, absent knowledge, calls, lookup sequence, total and necessary bytes, warm latency, and production complexity.
    Verify arithmetic against private raw input without leaking queries, facts or node IDs to tracked reports. Retain
    original failed-report provenance and never turn its P1 into PASS. The verifier must reject edited sealed inputs,
    mismatched current src tree or runner hash, inconsistent arm parameters or forbidden name mode, forged aggregates,
    and private content leakage. Verify the run receipt''s pre/post source hashes and single-consumption identity;
    a verification run reads frozen evidence and must not rerun the holdout.

    Use synthetic mutation tests: one improved group cannot pass; two groups plus an instruction or correction loss
    cannot pass; duplicate cases in one group do not count twice; baseline misses are distinct from losses; a necessary
    lookup can recover a fact; stale source/input hashes and a forged PASS are rejected; absent and large-group
    categories cannot silently disappear. A truthful failed report can be built and structurally verified, but verify
    --require-success must exit nonzero when P1 is unmet. This child proves the verifier with synthetic input, not
    the field capability. Root P4 full-suite success is established by the parent, never by a report met:true flag.'
  postcondition: A generic executable verifier with synthetic counterexamples checks v2 paired evidence and refuses
    to certify a loss, insufficient independent improvement, stale provenance or a forged PASS.
  postcondition_targets:
  - P2
  - P4
  interface_in:
    files_read:
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_corpus.py
    - tests/test_recall_applicability_runner.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    files_immutable:
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
  interface_out:
    files_owned:
    - scripts/recall_applicability_verdict.py
    - tests/test_recall_applicability_verdict.py
  required_artifacts:
  - scripts/recall_applicability_verdict.py
  - tests/test_recall_applicability_verdict.py
  evidence_of_completion:
  - kind: exec
    check: python -m pytest -q tests/test_recall_applicability_verdict.py
  acceptance:
  - python -m pytest -q tests/test_recall_applicability_verdict.py
  out_of_scope:
  - Changing production, the existing runner, any sealed case or raw measurement, or the parent SPEC thresholds.
  rationale: An independent executable report predicate is the missing infrastructure behind the failed report-only
    acceptance. It can be built in parallel with candidate repair and corpus sealing, avoiding another report node
    that can only assert met:true.
- id: paired-measurement-v2
  phase: integrate
  expected_mode: execute
  description: 'Measure the revised frozen candidate once and publish the independent applicability verdict.

    This is a fresh successor to the failed paired-measurement node, not a relaunch or an amendment of its consumed-holdout
    obligation. Its original required artifacts and journal remain archived; df445b7 and its immutable FAIL report
    are already in the parent branch. The unchanged parent P1 still requires two independent improved groups and
    zero losses. Never claim that a new result changes what happened in the first measurement.

    Consume the v2 seal, the frozen candidate receipt, the existing paired runner, and the independently tested
    verdict helper. First verify all original and new hashes, baseline-code identity b9769d8, candidate src tree,
    runner hash, and environment. Use the new packet once, on separate writable copies of the same original frozen
    snapshot with identical recall parameters, fresh worker/session state and alternating arm order. Keep production
    providers/models/tiers and the ordinary trigger mode unchanged. A shape-only private runner-input conversion
    must preserve every query, scope, fact and source. Keep a private run-receipt.json binding start/end timestamps,
    seal/input/raw hashes, pre/post candidate src tree, runner hash, actual command and non-secret effective recall/embedding
    settings for both arms. Create a consumption receipt before launch; an interrupted or partially exposed run
    must not silently become a fresh holdout on retry.

    Generate report-v2.json and report-v2.md from the paired raw output using the verdict helper, and run its actual
    --require-success acceptance command. Include every category''s reachability, losses and irrelevant slots, every
    necessary recall/lookup prefix and exhausted chain, complete/necessary response bytes, calls, warm latency and
    production-code delta. Independently inspect the production diff against P3 and explain the evidence in the
    report. Separate P1/P2/P3 verdicts from observed targeted P4 evidence; the parent owns fresh npm run check and
    final P4 acceptance. Do not use the previously unreachable large-group cases as proof of the accepted reading
    gain; reference the new candidate''s executable carrier/delivery regressions and report the new field controls
    honestly.

    Do not alter retrieval, the corpus, runner, verifier, root SPEC, gates or thresholds after seeing results. If
    the new one-shot result fails, commit the truthful aggregate failure and preserve the sealed raw evidence; leave
    P1 unmet. Do not relaunch the same holdout, recategorize losses, drop cases, or ask to make report completeness
    substitute for capability success. Reports and receipts must be committed immediately; private queries, claims,
    IDs, responses and database content stay local on sfx.'
  depends_on:
  - sealed-corpus-v2
  - retrieval-revision
  - paired-verdict
  - paired-runner
  dep_reasons:
    sealed-corpus-v2: New independent sealed packet, preregistration and baseline receipt.
    retrieval-revision: Frozen revised production candidate and targeted-regression receipt.
    paired-verdict: Executable aggregate builder/verifier with synthetic counterexamples.
    paired-runner: Accepted generic paired recall/lookup measurement executable.
  artifact_deps:
  - path: /home/sfx/p/ae/artifacts/recall-applicability/v2/seal.json
    producer: sealed-corpus-v2
  - path: artifacts/recall-applicability/candidate-v2.json
    producer: retrieval-revision
  - path: scripts/recall_applicability_verdict.py
    producer: paired-verdict
  postcondition: A new aggregate report is derived from one independent sealed paired measurement; executable verification
    establishes at least two improved holdout groups, zero baseline-reachable fact losses, correct measurement provenance
    and an unchanged historical FAIL report.
  postcondition_targets:
  - P1
  - P2
  - P3
  interface_in:
    files_read:
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_verdict.py
    - src/living_memory/retrieval.py
    - src/living_memory/score_gate.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
    files_immutable:
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/seal.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/baseline-outcomes.json
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
  interface_out:
    files_owned:
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/report-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/
  required_artifacts:
  - artifacts/recall-applicability/report-v2.json
  - artifacts/recall-applicability/report-v2.md
  - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/paired-raw.json
  - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/runner-input.json
  - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/run-receipt.json
  - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/consumption.json
  evidence_of_completion:
  - kind: exec
    check: python scripts/recall_applicability_verdict.py verify --report artifacts/recall-applicability/report-v2.json
      --require-success
  - kind: exec
    check: git diff --exit-code df445b7c44aa8b5d7511ab6f52b7a3cd02509ef0 -- artifacts/recall-applicability/prereg.md
      artifacts/recall-applicability/report.json artifacts/recall-applicability/report.md
  acceptance:
  - python scripts/recall_applicability_verdict.py verify --report artifacts/recall-applicability/report-v2.json
    --require-success
  - git diff --exit-code df445b7c44aa8b5d7511ab6f52b7a3cd02509ef0 -- artifacts/recall-applicability/prereg.md artifacts/recall-applicability/report.json
    artifacts/recall-applicability/report.md
  invariants_preserved:
  - Original report, corpus and measurement bytes remain unchanged and the historical P1 FAIL remains visible.
  - No private memory contents, questions, source IDs or secrets enter tracked reports, alt or a remote repository.
  out_of_scope:
  - Any production, runner, verifier, case, threshold or gate edits.
  - Claiming P4 completion from report assertions or replacing the parent whole-suite check.
  rationale: A fresh identity is required because the prior measurement published evidence under bound required_artifacts
    and has an irreparable historical capability failure. The successor consumes a revised candidate and newly sealed
    independent population, with real executable acceptance from the outset.
- id: procedure-retention-diagnosis
  phase: discover
  expected_mode: execute
  depends_on: []
  description: 'Diagnose the procedure-retention failures on the frozen applicability candidate.

    The parent verified report-v2 with --require-success but the declared suite exposed three failing tests in tests/test_procedure_case_separation.py:
    test_taught_recipe_arrives_whole_through_its_legacy_schema_turned_carrier, test_crowding_oracle_detects_the_group_identity_mutant,
    and test_group_node_keeps_the_trigger_it_is_found_by. All three pass with actual archived b9769d8 src and fail
    on the frozen v2 source under the same hash backend used by scripts/test.sh. The first and third are positive
    read-chain failures; the second is a mutant that now fails to suppress the recipe. Read artifacts/recall-applicability/parent-integration.md
    for the parent evidence and completed verifier fixes.

    Use the existing synthetic fixtures, ordinary public memory_recall and natural content_ref lookup to locate
    the exact failure stage: collection, ranking, truncation, deduplication, gating, or delivery. Compare identical
    fixture state and parameters with actual baseline and candidate imports, and capture relevant channel scores
    and the first point the instruction or carrier is lost. Check whether the declared hash-backend failure also
    occurs with the installed embedding backend, without changing production defaults or providers. Separate loss
    of an applicable current correction from an obsolete mutation expectation; do not assume that every failure
    has one cause or that a field holdout PASS cancels an executable loss.

    This is a bounded read-only diagnosis, not another production revision or a new corpus cycle. Keep production,
    existing tests, v2 evidence, private packets, and the verifier unchanged. Do not rerun either consumed field
    packet or read its private questions to tune a proposal. Write retention-diagnosis.json with schema_version
    1, the actual full baseline_commit, candidate src_tree and both source hashes, plus exactly three cases keyed
    by the failing test names with baseline_pass, candidate_pass, failure_stage, and diagnosis. Write retention-diagnosis.md
    with the concrete reproduction commands, import provenance, causal explanation, and the smallest cohesive repair
    recommendation. If the repair must change production, explicitly identify the v2 source-bound receipts and acceptance
    obligations that would become stale; leave that decision to the parent rather than silently changing frozen
    code or weakening a loss oracle. A completed diagnosis does not establish P4.'
  postcondition: open
  discovery_question: Which exact retrieval or delivery transition causes the two positive procedure losses, why
    does the crowding mutant no longer fail, and what bounded repair can restore the safety promises without reintroducing
    unconditional historical-carrier priority?
  postcondition_targets:
  - P4
  interface_in:
    files_read:
    - tests/test_procedure_case_separation.py
    - tests/test_transport_identity.py
    - src/living_memory/retrieval.py
    - src/living_memory/score_gate.py
    - src/living_memory/schema_dedup.py
    - src/living_memory/consolidation.py
    - artifacts/recall-applicability/candidate-v2.md
    - artifacts/recall-applicability/parent-integration.md
    files_immutable:
    - src/
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_verdict.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/report-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/
    - tests/
  interface_out:
    files_owned:
    - artifacts/recall-applicability/retention-diagnosis.json
    - artifacts/recall-applicability/retention-diagnosis.md
  required_artifacts:
  - artifacts/recall-applicability/retention-diagnosis.json
  - artifacts/recall-applicability/retention-diagnosis.md
  evidence_of_completion:
  - kind: external_review
    check: The parent reviews the source-bound synthetic comparisons and exact failure-stage traces; no report status
      is accepted as proof that the procedure losses are fixed.
  invariants_preserved:
  - All frozen production and sealed evidence bytes remain unchanged.
  - No field holdout is rerun or used to tune a proposal.
  - No weakening of current instruction, correction, or lookup retention oracles.
  out_of_scope:
  - Production and existing test edits.
  - New field corpus collection, holdout measurement, or publication of private queries, facts, or IDs.
  rationale: The full-suite failures include observable current-recipe loss as well as a mutant expectation; a narrow
    causal diagnosis is needed before changing a production tree already bound to one-shot evidence.
- id: anchor-reference-compatibility
  phase: build
  expected_mode: execute
  depends_on: []
  description: 'Reconcile the query-anchor regression oracle with intentional trigger ranking changes.

    The integrated full suite fails test_cold_start_ranking_is_byte_identical_to_pre_anchor_code and test_unresembled_query_class_ranks_identically_to_pre_anchor_code
    in tests/test_retrieval_query_anchors.py. Each divergence is the existing backup-snapshot trigger query; the
    historical reference ed20653 retains the unconditional trigger ranking now deliberately removed. Read artifacts/recall-applicability/parent-integration.md,
    the actual comparison code and historical reference before deciding how to repair the oracle.

    Own only this test file and a data-free anchor-reference-review.md. Keep production and every v2 receipt/report/input
    byte unchanged. Preserve every existing COLD_START_MATRIX query and mode, real-model jargon retrieval checks,
    anchor invisibility and scope isolation, the pre-fix demotion counterexample, and independent historical evidence
    for the anchor invariants. Replace only an equality claim invalidated by the intentional non-anchor ranking
    change, with stronger checks that isolate anchor effects under consistent trigger semantics. Retain an independent
    oracle; a current-code flag-on/flag-off comparison alone, copying the entire current ranker into the reference,
    ignoring a named divergence, deleting the trigger query, skips, or xfails do not establish the original safety
    promise. Include an executable counterexample or mutation control demonstrating that an anchor-induced cold-start/unresembled-query
    change would still fail the revised check.

    Run npm test -- -q tests/test_retrieval_query_anchors.py. Document exactly why the old assertion no longer isolates
    anchors, what new observable invariant replaces it, and the executed result. Do not claim this repairs the separate
    procedure-retention failures or establishes parent P4; the parent owns the fresh whole-suite run after integration.
    If correct anchor isolation requires changing production, report that scope requirement instead of altering
    frozen source.'
  postcondition: The full query-anchor test file passes with independent checks for cold-start and unresembled-query
    anchor neutrality, while preserving all queries, mutation controls, and frozen production/evidence bytes.
  postcondition_targets:
  - P4
  interface_in:
    files_read:
    - tests/test_retrieval_query_anchors.py
    - src/living_memory/retrieval.py
    - src/living_memory/query_anchors.py
    - artifacts/recall-applicability/candidate-v2.md
    - artifacts/recall-applicability/parent-integration.md
    files_immutable:
    - src/
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_verdict.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/report-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/
  interface_out:
    files_owned:
    - tests/test_retrieval_query_anchors.py
    - artifacts/recall-applicability/anchor-reference-review.md
  required_artifacts:
  - tests/test_retrieval_query_anchors.py
  - artifacts/recall-applicability/anchor-reference-review.md
  evidence_of_completion:
  - kind: exec
    check: npm test -- -q tests/test_retrieval_query_anchors.py
  - kind: external_review
    check: The parent verifies that the revised oracle retains every query and the historical/pre-fix counterexamples,
      and can detect an introduced anchor-induced divergence.
  acceptance:
  - npm test -- -q tests/test_retrieval_query_anchors.py
  invariants_preserved:
  - No production, provider, model, tier, scope, name valve, or sealed-evidence change.
  - All existing anchor evaluation queries and positive safety promises remain exercised.
  out_of_scope:
  - Procedure-retention tests and production changes.
  - Private field packet inspection or reruns.
  rationale: This test-only compatibility repair has disjoint ownership from the procedure diagnosis and can proceed
    independently without invalidating v2 evidence.
STEELMAN_AGAINST:
- Could acceptance pass through a hardcoded lookup, flag gate, or skill gate in the procedure probe or query-anchor
  reference, while the actual current recipe remains unreachable?
- Could the v2 three-group/zero-loss PASS hide the newly reproduced procedure-retention losses or be reused for
  a changed production tree?
- Could replacing the historical anchor comparison merely discard the trigger query or remove sensitivity to real
  anchor regressions?
- Could a diagnosis-only artifact be treated as P4 completion, or could an integration repair rewrite consumed evidence
  to make verification green?
MITIGATIONS:
- Use the existing public synthetic fixtures and ordinary recall/content_ref reader with actual archived b9769d8
  imports; prohibit case-specific retrieval rules, current-only flag comparisons as sole proof, and weakened loss
  oracles.
- Keep the frozen candidate source and all v2 evidence immutable in both new children. A production repair requires
  an explicit parent decision about stale code-bound evidence; the current positive losses remain open regardless
  of the field PASS.
- Keep the full query/mode matrix and historical/pre-fix controls, and require an executable mutation or counterexample
  that still exposes anchor-induced divergence under the revised test.
- The original SPEC and completed child contracts remain unchanged; only P4 is unmet. The parent has already repaired
  path relocation and synthetic preregistration isolation without changing report/input bytes or thresholds. Discovery
  stays open and never substitutes for the final declared-suite pass; no field rerun or new corpus is authorized
  here.
