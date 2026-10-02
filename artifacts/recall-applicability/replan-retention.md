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
  met: false
- id: P2
  statement: Accepted baseline b9769d8 and candidate are compared on one frozen state with identical recall parameters
    and isolated prior-call effects; the report records relevance, losses, all necessary recall and lookup calls,
    response volume, latency, and production-code complexity.
  kind: verification
  check_protocol: measure
  non_falsifiable: false
  met: false
- id: P3
  statement: The repair uses existing applicability and ranking rules without enabling LM_RECALL_SCHEMA_TRIGGER=name,
    narrowing default scope, deleting history, adding a mandatory LLM call, or changing providers, models, or tiers.
  kind: refactor
  check_protocol: external_review
  non_falsifiable: false
  met: false
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
    - procedure-retention-repair
    - sealed-corpus-v3
    P2:
    - sealed-corpus
    - paired-runner
    - sealed-corpus-v2
    - paired-verdict
    - sealed-corpus-v3
    - successor-verdict-contract
    P3:
    - retrieval-repair
    - procedure-retention-repair
    P4:
    - paired-runner
    - retrieval-repair
    - paired-verdict
    - procedure-retention-diagnosis
    - anchor-reference-compatibility
    - procedure-retention-repair
    - successor-verdict-contract
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
    paired-verdict:
    - P2
    - P4
    procedure-retention-diagnosis:
    - P4
    anchor-reference-compatibility:
    - P4
    procedure-retention-repair:
    - P1
    - P3
    - P4
    sealed-corpus-v3:
    - P1
    - P2
    successor-verdict-contract:
    - P2
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
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/report-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/
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
    unchanged parent P1 contract for a sealed independent holdout while preserving the historical failure. Custody
    of the archived v2 candidate and measurement artifacts transfers here solely to preserve their frozen bytes
    after the source-bound producer and measurement nodes are retired; this completed child is not relaunched and
    its bound required_artifacts and files_immutable remain unchanged.
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
    files_owned: []
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
  rationale: Retain the completed v2 synthetic verifier contract, required artifacts and immutable inputs; transfer
    all future writes of its script and tests to successor-verdict-contract. No historical report is amended.
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
- id: procedure-retention-repair
  phase: build
  expected_mode: execute
  depends_on: []
  description: 'Restore applicable procedure and correction recall without restoring historical-carrier priority.

    The remaining production gap is diagnosed in retention-diagnosis.json/.md and parent-retention-verification.md.
    On the frozen v2 tree, the two positive tests test_taught_recipe_arrives_whole_through_its_legacy_schema_turned_carrier
    and test_group_node_keeps_the_trigger_it_is_found_by lose intact carriers at ranking/top-N, despite collection
    and gate admission. The latter loss persists with the installed encoder. The crowding mutant is a different
    issue: the directly applicable recipe now survives, so its old required-suppression assertion is obsolete.


    Own one cohesive repair in retrieval.py and score_gate.py and the listed regressions. Use existing applicability,
    content/source evidence, authority/corrections and feedback to keep genuinely applicable carrier knowledge reachable
    at the original query and result limits. The diagnosis''s complete-saved-trigger route is a hypothesis, not
    a required algorithm. Do not restore an unconditional trigger rank floor or multiplier, special-case fixture/project
    words, add a token-count cutoff or blanket carrier promotion, change providers/models/tiers, enable the name
    valve, narrow scope, or add a mandatory LLM call. Prefer a shared evidence decision over separate rescue heuristics.


    Preserve the two positive tests'' queries, fixture state, result limits and whole current knowledge requirements.
    Restore their ordinary public memory_recall plus natural content_ref lookup outcome on the declared hash backend
    and installed embedding backend. Add renamed and compound-query contrasts with unrelated same-trigger historical
    carriers and short applicable instructions; retain correction/supersedes, cross-project, feedback, authority
    and full-byte lookup coverage. Update only the obsolete crowding mutant to prove the dedup identity/slot-count
    effect independently while positively requiring the direct recipe to remain reachable. Keep the existing positive
    dedup test and both identity/trigger corruption mutants sensitive; no skips, xfails, dropped records or ID lookup
    rescue. Explain any implementation-coupled assertion change.


    Use only synthetic development fixtures and the original four development cases extracted by split. Do not read
    any original, v2 or v3 holdout questions or field raw responses, run consumed packets, or inspect the new packet
    author''s work. Keep baseline-reachable development clauses reachable and record failures honestly. Preserve
    the accepted large-group reading gain through executable carrier/delivery regressions and comparable whole necessary
    read-chain cost, not through unreachable field cases.


    Commit source changes immediately. Run the complete owned focused suite and the two positive fixtures with the
    installed encoder, then freeze candidate-v3.json with full baseline_commit, candidate_commit, src_tree and SHA-256
    for retrieval.py and score_gate.py. candidate-v3.md records mechanism, measured synthetic/development retention
    and reading costs, rejected alternatives, targeted commands and production delta. The prior candidate-v2 receipt
    and reports remain immutable historical evidence and must never be updated to this source tree. The parent will
    run a fresh whole declared suite BEFORE spending the new one-shot holdout, then perform the paired measurement;
    targeted success is not P1 or parent P4 completion.'
  postcondition: The two positive procedure read chains and focused applicability, correction, feedback, dedup,
    lookup and large-group safety promises pass on a newly frozen candidate; generic history does not regain unconditional
    trigger priority.
  postcondition_targets:
  - P1
  - P3
  - P4
  interface_in:
    files_read:
    - src/living_memory/schema_dedup.py
    - src/living_memory/consolidation.py
    - src/living_memory/server.py
    - src/living_memory/delivery.py
    - scripts/recall_applicability_eval.py
    - artifacts/recall-applicability/retention-diagnosis.md
    - artifacts/recall-applicability/retention-diagnosis.json
    - artifacts/recall-applicability/parent-retention-verification.md
    - artifacts/recall-applicability/candidate-v2.md
    - artifacts/recall-applicability/anchor-reference-review.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    files_immutable:
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/report-v2.md
    - artifacts/recall-applicability/retention-diagnosis.json
    - artifacts/recall-applicability/retention-diagnosis.md
    - artifacts/recall-applicability/anchor-reference-review.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/
    - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/
    - /home/sfx/p/ae/artifacts/recall-applicability/candidate-development/
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_corpus.py
    - scripts/recall_applicability_verdict.py
    - tests/test_retrieval_query_anchors.py
  interface_out:
    files_owned:
    - src/living_memory/retrieval.py
    - src/living_memory/score_gate.py
    - tests/test_procedure_case_separation.py
    - tests/test_recall_applicability.py
    - tests/test_recall_score_gate.py
    - tests/test_retrieval_cross_scope_gate.py
    - tests/test_schema_trigger_by_name.py
    - tests/test_recall_delivery_chain.py
    - tests/test_recall_carrier_omission.py
    - docs/recall-schema-trigger.md
    - artifacts/recall-applicability/candidate-v3.json
    - artifacts/recall-applicability/candidate-v3.md
    - /home/sfx/p/ae/artifacts/recall-applicability/candidate-v3-development/
  required_artifacts:
  - artifacts/recall-applicability/candidate-v3.json
  - artifacts/recall-applicability/candidate-v3.md
  - tests/test_procedure_case_separation.py
  - tests/test_recall_applicability.py
  acceptance:
  - timeout 600 npm test -- -q tests/test_procedure_case_separation.py tests/test_recall_applicability.py tests/test_recall_score_gate.py
    tests/test_retrieval_cross_scope_gate.py tests/test_schema_trigger_by_name.py tests/test_recall_delivery_chain.py
    tests/test_recall_carrier_omission.py tests/test_recall_feedback_loop.py tests/test_explicit_feedback.py tests/test_transport_feedback_closure_e2e.py
    tests/test_retrieval_feedback_amplification.py
  - "python3 - <<'CHECK'\nimport hashlib, json, subprocess\nfrom pathlib import Path\nr=json.loads(Path('artifacts/recall-applicability/candidate-v3.json').read_text())\n\
    assert r['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\nassert len(r['candidate_commit'])==40\n\
    assert r['src_tree']==subprocess.check_output(['git','rev-parse','HEAD:src'],text=True).strip()\nassert r['src_tree']==subprocess.check_output(['git','rev-parse',r['candidate_commit']+':src'],text=True).strip()\n\
    assert set(r['source_sha256'])=={'src/living_memory/retrieval.py','src/living_memory/score_gate.py'}\nfor name,\
    \ expected in r['source_sha256'].items():\n    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected\n\
    print('New candidate freeze matches current production source.')\nCHECK"
  evidence_of_completion:
  - kind: exec
    check: timeout 600 npm test -- -q tests/test_procedure_case_separation.py tests/test_recall_applicability.py
      tests/test_recall_score_gate.py tests/test_retrieval_cross_scope_gate.py tests/test_schema_trigger_by_name.py
      tests/test_recall_delivery_chain.py tests/test_recall_carrier_omission.py tests/test_recall_feedback_loop.py
      tests/test_explicit_feedback.py tests/test_transport_feedback_closure_e2e.py tests/test_retrieval_feedback_amplification.py
  - kind: exec
    check: "python3 - <<'CHECK'\nimport hashlib, json, subprocess\nfrom pathlib import Path\nr=json.loads(Path('artifacts/recall-applicability/candidate-v3.json').read_text())\n\
      assert r['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\nassert len(r['candidate_commit'])==40\n\
      assert r['src_tree']==subprocess.check_output(['git','rev-parse','HEAD:src'],text=True).strip()\nassert r['src_tree']==subprocess.check_output(['git','rev-parse',r['candidate_commit']+':src'],text=True).strip()\n\
      assert set(r['source_sha256'])=={'src/living_memory/retrieval.py','src/living_memory/score_gate.py'}\nfor\
      \ name, expected in r['source_sha256'].items():\n    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected\n\
      print('New candidate freeze matches current production source.')\nCHECK"
  invariants_preserved:
  - Ordinary broad recall and natural content_ref lookup remain the consumer path; instructions, current corrections
    and full original memory stay available.
  - All original and v2 sealed evidence bytes remain unchanged; the new candidate needs its own source-bound measurement.
  out_of_scope:
  - Reading field holdouts or changing their predicates, the paired runner, verifier, embedding defaults, providers,
    models or tiers.
  - Weakening the two positive loss oracles or treating a passing targeted suite as a field verdict.
  rationale: This is the smallest cohesive production ownership for the diagnosed ranking/truncation losses, including
    the related obsolete mutant oracle. It succeeds the archived retrieval-revision node without changing that node's
    source-bound historical receipt.
- id: sealed-corpus-v3
  phase: design
  expected_mode: execute
  depends_on: []
  description: 'Seal a bounded independent confirmation packet for the procedure-retention repair.

    A production repair is required after the v2 field PASS because the declared suite exposes positive procedure/correction
    loss. Preserve the original FAIL, the v2 PASS and every old sealed byte. Author one bounded new holdout on the
    exact original frozen snapshot, independently of procedure-retention-repair and any candidate output. Prefer
    fourteen cases, at least twelve, plus the four original development cases unchanged. Exclude all original and
    v2 queries, sources and semantic answer families, including carrier/child evidence and supersedes families;
    renamed group labels alone do not establish disjointness. Cover concrete and compound questions, short-trigger
    mandatory instructions, corrections, useful cross-project transfer, absent knowledge and large groups. Include
    controls capable of exposing loss of applicable knowledge carried under a saved trigger, without copying the
    repair''s synthetic fixture vocabulary.


    Freeze every question and literal required clause before capturing baseline results. Validate each clause against
    its complete source. Execute the actual archived b9769d8 source with an asserted import origin, avoiding the
    checkout-root Python shim; keep broad scope, depth=1, max_results=4, ordinary trigger mode, installed embedding
    backend and identical non-secret settings. Capture baseline source ranks using the existing corpus schema, and
    require baseline-reachable instruction, correction and cross-project sources before any candidate exposure.
    Explain that source rank is a preflight control, while final fact retention requires natural recall/lookup scoring.
    If one bounded authored packet cannot provide these controls, report the inadequacy rather than repeatedly selecting
    favorable cases or collecting a multi-day corpus.


    Write local v3/goldset.json, v3/baseline-outcomes.json and v3/seal.json. The seal has schema_version=1, generation=3,
    sealed_at_utc, full baseline_commit, baseline_src_tree, and the existing files mapping snapshot/goldset/previous_goldset/baseline_outcomes
    to absolute path plus sha256. snapshot points to the original database and previous_goldset to the original
    goldset. Add prior_seals=[{path,sha256}] containing the immutable v2 seal, so the verifier can validate exclusion
    against both previous populations without rewriting old evidence. Commit prereg-v3.md containing all file and
    prior-seal hashes, actual baseline import provenance, fixed environment, source-family exclusions, bounded independent
    authoring, the unchanged two-independent-group/zero-loss criterion and one-shot policy.


    Keep private material local on sfx with restrictive permissions, never in git, Living Memory, alt or a remote
    repository. Do not inspect candidate implementation, development measurements or new verifier decisions to select
    cases; do not run any candidate or either consumed field packet. This task prepares evidence independently;
    the parent must pass the whole declared suite before consuming the new packet.'
  postcondition: An independently authored, source-family-disjoint holdout of at least twelve questions is sealed
    on the original snapshot with unchanged development cases, validated clauses, real-baseline provenance and protected
    baseline source controls.
  postcondition_targets:
  - P1
  - P2
  interface_in:
    files_read:
    - scripts/recall_applicability_corpus.py
    - scripts/recall_applicability_eval.py
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/prereg-v2.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/seal.json
    files_immutable:
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/report-v2.md
    - artifacts/recall-applicability/retention-diagnosis.json
    - artifacts/recall-applicability/retention-diagnosis.md
    - artifacts/recall-applicability/anchor-reference-review.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/
    - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/
    - /home/sfx/p/ae/artifacts/recall-applicability/candidate-development/
    - src/
    - tests/
    - scripts/recall_applicability_corpus.py
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_verdict.py
  interface_out:
    files_owned:
    - artifacts/recall-applicability/prereg-v3.md
    - /home/sfx/p/ae/artifacts/recall-applicability/v3/
  required_artifacts:
  - artifacts/recall-applicability/prereg-v3.md
  - /home/sfx/p/ae/artifacts/recall-applicability/v3/goldset.json
  - /home/sfx/p/ae/artifacts/recall-applicability/v3/baseline-outcomes.json
  - /home/sfx/p/ae/artifacts/recall-applicability/v3/seal.json
  acceptance:
  - python3 scripts/recall_applicability_corpus.py validate --snapshot /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    --corpus /home/sfx/p/ae/artifacts/recall-applicability/v3/goldset.json
  - "python3 - <<'CHECK'\nimport hashlib, json, subprocess\nfrom pathlib import Path\np=Path('/home/sfx/p/ae/artifacts/recall-applicability')\n\
    def read(x): return json.loads(x.read_text())\ndef digest(x):\n    with x.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()\n\
    s=read(p/'v3/seal.json')\nassert s['schema_version']==1 and s['generation']==3\nassert s['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\n\
    assert s['baseline_src_tree']==subprocess.check_output(['git','rev-parse',s['baseline_commit']+':src'],text=True).strip()\n\
    expected={'snapshot':p/'sfx-frozen.sqlite3','goldset':p/'v3/goldset.json','previous_goldset':p/'goldset.json','baseline_outcomes':p/'v3/baseline-outcomes.json'}\n\
    assert set(s['files'])==set(expected)\nfor k,x in expected.items():\n    assert Path(s['files'][k]['path']).resolve()==x.resolve()\n\
    \    assert digest(x)==s['files'][k]['sha256']\nassert s['files']['snapshot']['sha256']=='3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef'\n\
    assert len(s['prior_seals'])==1\nprior=s['prior_seals'][0]\nassert Path(prior['path']).resolve()==(p/'v2/seal.json').resolve()\n\
    assert prior['sha256']==digest(p/'v2/seal.json')\nfor entry in read(p/'v2/seal.json')['files'].values():\n \
    \   assert digest(Path(entry['path']))==entry['sha256']\nold=read(p/'goldset.json')['cases']; v2=read(p/'v2/goldset.json')['cases'];\
    \ new=read(p/'v3/goldset.json')['cases']\nhold=[c for c in new if c['split']=='holdout']\nassert len(hold)>=12\n\
    assert [c for c in new if c['split']=='development']==[c for c in old if c['split']=='development']\nfor key\
    \ in ('query','topic_group'):\n    assert {c[key] for c in hold}.isdisjoint(c[key] for c in old+v2)\nassert\
    \ {n for c in hold for n in c['acceptable_source_ids']}.isdisjoint(n for c in old+v2 for n in c['acceptable_source_ids'])\n\
    b=read(p/'v3/baseline-outcomes.json')\nassert b['code_head']==s['baseline_commit']\nassert b['inputs']['snapshot_sha256']==s['files']['snapshot']['sha256']\n\
    assert b['inputs']['corpus_sha256']==s['files']['goldset']['sha256']\nrows={x['case_id']:x for x in b['outcomes']}\n\
    assert set(rows)=={c['case_id'] for c in new}\nfor category in ('instruction','correction','cross-project'):\n\
    \    assert any(rows[c['case_id']]['source_ranks'] for c in hold if c['category']==category), category\ntext=Path('artifacts/recall-applicability/prereg-v3.md').read_text()\n\
    assert all(e['sha256'] in text for e in list(s['files'].values())+s['prior_seals'])\nprint('Successor seal hashes,\
    \ population, disjointness and protected baseline source controls verified.')\nCHECK"
  evidence_of_completion:
  - kind: exec
    check: "python3 - <<'CHECK'\nimport hashlib, json, subprocess\nfrom pathlib import Path\np=Path('/home/sfx/p/ae/artifacts/recall-applicability')\n\
      def read(x): return json.loads(x.read_text())\ndef digest(x):\n    with x.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()\n\
      s=read(p/'v3/seal.json')\nassert s['schema_version']==1 and s['generation']==3\nassert s['baseline_commit']=='b9769d84e3188ee1e646627ebe1b4d454f69d6f9'\n\
      assert s['baseline_src_tree']==subprocess.check_output(['git','rev-parse',s['baseline_commit']+':src'],text=True).strip()\n\
      expected={'snapshot':p/'sfx-frozen.sqlite3','goldset':p/'v3/goldset.json','previous_goldset':p/'goldset.json','baseline_outcomes':p/'v3/baseline-outcomes.json'}\n\
      assert set(s['files'])==set(expected)\nfor k,x in expected.items():\n    assert Path(s['files'][k]['path']).resolve()==x.resolve()\n\
      \    assert digest(x)==s['files'][k]['sha256']\nassert s['files']['snapshot']['sha256']=='3817f4c26f725f4a11f0237316f4a58656b646ea8402e160258143ddce241cef'\n\
      assert len(s['prior_seals'])==1\nprior=s['prior_seals'][0]\nassert Path(prior['path']).resolve()==(p/'v2/seal.json').resolve()\n\
      assert prior['sha256']==digest(p/'v2/seal.json')\nfor entry in read(p/'v2/seal.json')['files'].values():\n\
      \    assert digest(Path(entry['path']))==entry['sha256']\nold=read(p/'goldset.json')['cases']; v2=read(p/'v2/goldset.json')['cases'];\
      \ new=read(p/'v3/goldset.json')['cases']\nhold=[c for c in new if c['split']=='holdout']\nassert len(hold)>=12\n\
      assert [c for c in new if c['split']=='development']==[c for c in old if c['split']=='development']\nfor key\
      \ in ('query','topic_group'):\n    assert {c[key] for c in hold}.isdisjoint(c[key] for c in old+v2)\nassert\
      \ {n for c in hold for n in c['acceptable_source_ids']}.isdisjoint(n for c in old+v2 for n in c['acceptable_source_ids'])\n\
      b=read(p/'v3/baseline-outcomes.json')\nassert b['code_head']==s['baseline_commit']\nassert b['inputs']['snapshot_sha256']==s['files']['snapshot']['sha256']\n\
      assert b['inputs']['corpus_sha256']==s['files']['goldset']['sha256']\nrows={x['case_id']:x for x in b['outcomes']}\n\
      assert set(rows)=={c['case_id'] for c in new}\nfor category in ('instruction','correction','cross-project'):\n\
      \    assert any(rows[c['case_id']]['source_ranks'] for c in hold if c['category']==category), category\ntext=Path('artifacts/recall-applicability/prereg-v3.md').read_text()\n\
      assert all(e['sha256'] in text for e in list(s['files'].values())+s['prior_seals'])\nprint('Successor seal\
      \ hashes, population, disjointness and protected baseline source controls verified.')\nCHECK"
  invariants_preserved:
  - No changes to either consumed population, old raw outputs, consumption records or original snapshot.
  - No candidate-guided selection, weakened thresholds or publication of private case contents.
  out_of_scope:
  - Production, test, runner and verifier edits.
  - Candidate runs or tuning on either consumed holdout.
  rationale: Source repair makes the v2 PASS historical for its frozen tree. One bounded, independent successor
    packet is prepared in parallel with repair, not as a multi-day prerequisite or a repetition of a consumed holdout.
- id: successor-verdict-contract
  phase: build
  expected_mode: execute
  depends_on: []
  description: 'Extend the existing verdict verifier for source-bound successor evidence and non-vacuous retention
    controls.

    Own only scripts/recall_applicability_verdict.py, its synthetic test file and successor-verdict.md. Reuse the
    current runner and aggregation machinery; do not create another runner or change production. Work only with
    synthetic cases and schemas. Do not read private field cases, raw output or successor candidate performance.


    Support the v3 seal interface agreed with sealed-corpus-v3: schema_version=1, generation=3, the unchanged four-entry
    files map (snapshot, goldset, previous_goldset, baseline_outcomes), and prior_seals=[{path,sha256}] with the
    v2 seal. The original snapshot and original previous_goldset identities stay mandatory. Validate the prior seal
    and every bound file, disjoint queries/source IDs/topic groups against both older populations, the unchanged
    development cases and actual baseline-code provenance. The validate-seal preflight reuses the existing corpus
    validator to check literal clauses on the read-only frozen snapshot. Candidate-v3.json has the same full baseline_commit/candidate_commit/src_tree/source_sha256
    fields as v2. Preserve the existing build and verify CLI interface, choosing prereg-v3.md and the v3 candidate-receipt
    relocation path from the hash-bound generation; unversioned v2 evidence must retain its original arithmetic
    and JSON shape. Do not add a switch which bypasses source matching, history hashes, a protected control or an
    acceptance threshold. A v2 report on a different current src must still be rejected as current evidence.


    Fix the demonstrated retention-control gap: the current synthetic packet can lose all baseline instruction reachability
    and still pass verify --require-success. Require a baseline-reachable required fact in each protected holdout
    category instruction, correction and cross-project before claiming retention. Recompute fact losses from reached
    indexes for every case; report development losses explicitly and make success reject new baseline-reachable
    development losses as well as holdout losses. Keep at least two DISTINCT improved holdout groups and zero holdout
    baseline-reachable losses as the field threshold. Category presence or source rank alone is not delivered-fact
    retention. Baseline misses stay separate from losses.


    Retain generation-independent verification of exact shape-only runner input, isolated identical arm settings,
    ordinary trigger mode, runner hash, candidate commit and current src tree/file hashes, pre/post source receipts,
    timestamped single-consumption identity, all response/call/necessary-prefix/exhausted-chain metrics, complexity
    arithmetic and privacy. Preserve removed-worktree relocation safeguards. Bind and verify the historical original
    FAIL and v2 report hashes without treating them as proof for the new current source. Verification reads immutable
    evidence; it must never rerun a holdout.


    Add synthetic mutation tests for both prior populations'' overlap and edited seals, wrong generation/path/source/runner
    hashes, missing protected BASELINE reachability, instruction/correction/cross-project fact loss, development
    fact loss, insufficient distinct improvement, missing absent/large categories, forged PASS, and necessary-lookup
    recovery. Existing tests and v2 serialization must remain compatible; make synthetic tests independent of frozen
    real candidate receipts so they also run after a legitimate source revision. Verify a truthful failed report
    structurally but reject --require-success. Document commands and schemas, including a validate-seal --seal command
    for parent preflight. Commit edits immediately. The parent owns fresh npm run check before the one-shot run
    and final field/P4 acceptance.'
  postcondition: The reused verifier accepts correctly bound v3 synthetic evidence, rejects overlap and vacuous
    or lossy retention, preserves v2 evidence semantics, and exposes executable preflight/build/verify commands
    without touching field packets.
  postcondition_targets:
  - P2
  - P4
  interface_in:
    files_read:
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_corpus.py
    - tests/test_recall_applicability_runner.py
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/parent-retention-verification.md
    files_immutable:
    - artifacts/recall-applicability/prereg.md
    - artifacts/recall-applicability/report.json
    - artifacts/recall-applicability/report.md
    - artifacts/recall-applicability/prereg-v2.md
    - artifacts/recall-applicability/candidate-v2.json
    - artifacts/recall-applicability/candidate-v2.md
    - artifacts/recall-applicability/report-v2.json
    - artifacts/recall-applicability/report-v2.md
    - artifacts/recall-applicability/retention-diagnosis.json
    - artifacts/recall-applicability/retention-diagnosis.md
    - artifacts/recall-applicability/anchor-reference-review.md
    - /home/sfx/p/ae/artifacts/recall-applicability/sfx-frozen.sqlite3
    - /home/sfx/p/ae/artifacts/recall-applicability/goldset.json
    - /home/sfx/p/ae/artifacts/recall-applicability/baseline-outcomes.json
    - /home/sfx/p/ae/artifacts/recall-applicability/runner-input.json
    - /home/sfx/p/ae/artifacts/recall-applicability/paired-raw.json
    - /home/sfx/p/ae/artifacts/recall-applicability/v2/
    - /home/sfx/p/ae/artifacts/recall-applicability/v2-run/
    - /home/sfx/p/ae/artifacts/recall-applicability/candidate-development/
    - src/
    - scripts/recall_applicability_eval.py
    - scripts/recall_applicability_corpus.py
    - tests/test_recall_applicability_runner.py
  interface_out:
    files_owned:
    - scripts/recall_applicability_verdict.py
    - tests/test_recall_applicability_verdict.py
    - artifacts/recall-applicability/successor-verdict.md
  required_artifacts:
  - scripts/recall_applicability_verdict.py
  - tests/test_recall_applicability_verdict.py
  - artifacts/recall-applicability/successor-verdict.md
  acceptance:
  - python3 -m pytest -q tests/test_recall_applicability_verdict.py
  evidence_of_completion:
  - kind: exec
    check: python3 -m pytest -q tests/test_recall_applicability_verdict.py
  invariants_preserved:
  - No archived reports, receipts, private packet bytes or runner behavior changes.
  - No report can certify a different current production source, missing protected control, lost fact or a repeated
    consumption as an independent success.
  out_of_scope:
  - Production edits, candidate tuning, private holdout inspection and field measurement.
  - Changing the two-group/zero-loss criterion or claiming parent P4 from synthetic verifier success.
  rationale: This is a bounded extension of the existing evidence consumer plus its mutation tests, independent
    of candidate repair and corpus authoring. It also closes an executable vacuous-retention gap found during this
    parent verification.
STEELMAN_AGAINST:
- Could a hardcoded lookup, fixture-word exception, production flag gate or skill-gating rescue the named ledger/relay
  fixtures while unrelated historical carriers still win ordinary concrete or compound recall?
- Could the source repair borrow the v2 PASS, or a repeated consumed packet, even though candidate-v2.json binds
  another src tree?
- Could retention pass vacuously because every baseline instruction or correction is unreachable, or because tests
  weaken the two real positive losses along with the obsolete mutant?
- Could another full field packet be spent before the repository suite reveals the same class of preventable retention
  failure?
- Could renamed source groups or carrier/correction relatives leak old holdout knowledge into the supposedly independent
  new sample?
MITIGATIONS:
- Keep the positive queries, fixtures, limits and whole-knowledge requirements; add renamed and compound contrasts
  via public recall/lookup; review the shared production mechanism and retain broad scope and ordinary trigger mode.
- Archive retrieval-revision and paired-measurement-v2 identities and keep every required artifact byte unchanged.
  Their source-bound receipts remain historical; the new repair owns source/regression paths and freezes a distinct
  candidate-v3 receipt. No old acceptance predicate is relabelled or made to pass for the new tree.
- The verifier uses synthetic negative controls for missing baseline fact reachability and every protected loss.
  Only the obsolete mutant suppression expectation may change; its replacement measures dedup identity and slot-count
  corruption while retaining direct-recipe delivery.
- The three new leaves run independently. After they finish, the parent first runs fresh npm run check, validates
  the v3 seal and candidate freeze, and resolves any executable regression BEFORE creating v3-run/consumption.json.
  The parent then performs the existing paired runner once as bounded final verification, with separate snapshot
  copies, fresh sessions and alternating arms. It writes v3-run runner-input/raw/run-receipt/consumption and report-v3.json/.md,
  runs verify --require-success, independently reviews P3, commits aggregates immediately and marks P1/P2/P4 only
  from actual evidence. No final checker child or new runner is needed.
- The parent one-shot run retains complete per-category facts, baseline misses, new losses, ranks, necessary and
  exhausted lookup chains, bytes, calls, warm latency and production delta. A FAIL stays FAIL; neither consumed
  original/v2 packet is rerun. The new packet is bounded to one independent authoring/capture attempt rather than
  a multi-day collection loop.
- Corpus authoring excludes both prior populations using semantic topics, source lineage and supersedes families;
  the verifier binds both old populations and rejects direct overlap. Private source material remains local on sfx
  and never enters git, Living Memory, alt or remote repositories.
