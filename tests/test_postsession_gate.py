"""The anti-done-journal-dump gate, rule by rule.

Two things are pinned here. First, each rejection rule fires on the shape it
owns and stays quiet on the shape it does not -- with the *code* asserted, not
just the boolean, because the run report is keyed on codes and a rule that
quietly re-labels itself would silently rewrite the report. Second, the rule
order, so a text that trips two rules is always counted under the same one.

There is no accept-list and no expected-insight text anywhere in ``gate.py``;
these tests exercise properties of text and evidence, which is what lets the
same thresholds be reported on sessions they were never fitted to.
"""

from __future__ import annotations

import json

import pytest

from living_memory.postsession.gate import (
    DEFAULT_GATE,
    REJECTION_CODES,
    Gate,
    GateConfig,
    SessionContext,
    containment_of,
    rejection_breakdown,
)
from living_memory.postsession.mining import Candidate
from living_memory.postsession.session import (
    DeliveredNode,
    Evidence,
    FileMutation,
    RecallInteraction,
    SessionRecord,
    SourceSpan,
)

SPAN = SourceSpan("claude", "/tmp/t.jsonl", 42, "result")

GOOD = (
    "pytest reads `addopts = \"-q\"` from pyproject.toml, so passing `-q` again on "
    "the command line collapses the report to one line and hides which test failed."
)


def _candidate(quote: str = "", **overrides: object) -> Candidate:
    text = quote or (
        "pyproject.toml sets addopts = \"-q\"\n"
        "$ pytest -q\n"
        "1 failed\n"
        "src/living_memory/storage.py raised ValueError after 214ms at 0.95\n"
        "--strict-mcp-config /home/operator/p/lm/src/app.py 4821"
    )
    base = {
        "kind": "resolved_failure",
        "session_key": "claude:test",
        "ordinal": 0,
        "span": SPAN,
        "evidence": (Evidence(quote=text, locator=SPAN.locator()),),
        "identifiers": (),
        "signals": {},
    }
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


def _gate(**overrides: object) -> Gate:
    return Gate(GateConfig(**overrides) if overrides else DEFAULT_GATE)


def _check(fact: str, **overrides: object):
    return _gate(**overrides).check(
        fact, candidate=_candidate(), context=SessionContext()
    )


# --------------------------------------------------------------------------
# the rule this module exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fact",
    [
        "RESULT: DIRECT_EXECUTION Added the retry loop to src/app.py and all tests pass.",
        "OUTCOME pass: lm/insight-extraction-gate - built the extractor in src/x.py.",
        "СДЕЛАНО: перенёс конфиги в packages/rarity, проверки прошли на 9b1a78f.",
        "Completed lm/foo in worktree /repo/.worktrees/x with 12 checks green.",
        "feat(theme): статус-жёлтый обновлён на #eec80a в src/theme.ts, vitest 390 зелёные.",
    ],
)
def test_execution_journal_templates_are_rejected(fact: str) -> None:
    verdict = _check(fact)
    assert (verdict.accepted, verdict.code) == (False, "journal_phrasing")


@pytest.mark.parametrize(
    "fact",
    [
        "Added the React chats data layer: endpoint builders in src/api.ts and per-chat SSE.",
        "Implement scripts/ap_baseline.py so the report reproduces the frozen baseline.",
        "Перенесены TypeScript и ESLint конфиги rarity v47.0.1 в `packages/rarity`.",
        "Fixed the unsliced FTS document-frequency query in src/living_memory/storage.py.",
        "Убрать префилл из фидбек-флоу АЗС в src/modules/benz/index.tsx.",
    ],
)
def test_a_content_that_opens_with_a_work_verb_is_a_journal_entry(fact: str) -> None:
    """The merge-commit-subject shape, whatever identifiers follow it."""

    verdict = _check(fact)
    assert (verdict.accepted, verdict.code) == (False, "journal_phrasing")
    assert verdict.signals["tier"] == "opening"


def test_completion_claims_are_rejected_but_a_measurement_survives() -> None:
    """"All tests pass" is a journal entry; a measured count with its cause is not."""

    claim = _gate().check(
        "The goal is fully met at freeze commit 9b1a78f: all 88 tests pass and "
        "every manifest hash in artifacts/x.json matches.",
        candidate=_candidate(),
        context=SessionContext(),
    )
    assert (claim.accepted, claim.code) == (False, "journal_phrasing")

    measured = _gate().check(
        "`pytest --findRelatedTests src/living_memory/storage.py` reports 44 tests "
        "passing because pytest resolves related tests transitively, so a narrow "
        "run is not the narrow suite it looks like.",
        candidate=_candidate(
            quote="$ pytest --findRelatedTests src/living_memory/storage.py\n44 tests passing"
        ),
        context=SessionContext(),
    )
    assert measured.accepted, measured.detail


def test_a_real_lesson_that_mentions_a_fix_survives_on_its_mechanism() -> None:
    fact = (
        "immers.cloud hot-attached ports can stay status=DOWN forever because the "
        "guest never sees the NIC; a soft reboot does not help, only "
        "`openstack server reboot --hard` recreates the libvirt domain."
    )
    verdict = _gate().check(
        fact,
        candidate=_candidate(
            quote="status=DOWN\n$ openstack server reboot --hard vm1\nimmers.cloud"
        ),
        context=SessionContext(),
    )
    assert verdict.accepted, f"{verdict.code}: {verdict.detail}"


# --------------------------------------------------------------------------
# the other rules
# --------------------------------------------------------------------------


def test_a_serialized_context_object_in_content_is_the_measured_write_defect() -> None:
    leak = json.dumps(
        {"scope": "project:lm", "agent": "claude", "task": "t", "timestamp": "now"}
    )
    verdict = _check(f"Something happened and then {leak} was appended.")
    assert (verdict.accepted, verdict.code) == (False, "context_leak")


def test_one_quoted_json_key_is_not_a_context_leak() -> None:
    fact = (
        'The /api/status endpoint answers `{"error": "not found"}` with HTTP 404 when '
        "the dashboard is not running, so a 404 there is not an auth problem."
    )
    verdict = _gate().check(
        fact,
        candidate=_candidate(quote='$ curl /api/status\n{"error": "not found"}\n404'),
        context=SessionContext(),
    )
    assert verdict.accepted, f"{verdict.code}: {verdict.detail}"


def test_a_bare_json_object_is_never_a_trace() -> None:
    verdict = _check('{"content": "x", "scope": "project:lm", "note": "and more text here"}')
    assert (verdict.accepted, verdict.code) == (False, "context_leak")


def test_speculation_is_rejected_unless_it_was_resolved() -> None:
    hedged = _check(
        "The 214ms latency in src/app.py is probably caused by the cache, we should "
        "check that next and then decide about --strict-mcp-config."
    )
    assert (hedged.accepted, hedged.code) == (False, "speculation")

    resolved = _gate().check(
        "We suspected the cache, but measured it: src/app.py spends 214ms in the "
        "tokenizer, not the cache, so --strict-mcp-config changes nothing.",
        candidate=_candidate(),
        context=SessionContext(),
    )
    assert resolved.accepted, resolved.detail


def test_enumerated_facts_are_several_traces_wearing_one_coat() -> None:
    verdict = _check(
        "Three findings:\n- src/app.py raised ValueError\n- 214ms in the tokenizer\n"
        "- --strict-mcp-config is ignored\n- storage.py caches by mtime"
    )
    assert (verdict.accepted, verdict.code) == (False, "multi_fact")


def test_topic_disjoint_paragraphs_are_several_facts() -> None:
    verdict = _check(
        "src/living_memory/storage.py caches nodes by mtime, so a rewritten file is "
        "silently stale.\n\n"
        "Separately, --strict-mcp-config is ignored by the 214ms path in web/api.ts "
        "because the flag never reaches it."
    )
    assert (verdict.accepted, verdict.code) == (False, "multi_fact")


def test_a_single_fact_may_be_two_paragraphs_about_the_same_thing() -> None:
    verdict = _gate().check(
        "src/living_memory/storage.py caches nodes by mtime.\n\n"
        "So a rewritten src/living_memory/storage.py with an unchanged mtime is "
        "served stale, which is why the 214ms number never moved.",
        candidate=_candidate(),
        context=SessionContext(),
    )
    assert verdict.accepted, f"{verdict.code}: {verdict.detail}"


def test_a_fact_with_no_concrete_anchor_cannot_be_recalled_later() -> None:
    verdict = _check(
        "The system behaves differently under load and that is worth knowing for "
        "anyone who works on this area of the code in the future."
    )
    assert (verdict.accepted, verdict.code) == (False, "no_concrete_anchor")


def test_an_identifier_the_evidence_never_contained_was_invented() -> None:
    verdict = _check(
        "The failure comes from src/living_memory/retrieval.py raising KeyError "
        "whenever the scope is missing, which is why the run aborted."
    )
    assert (verdict.accepted, verdict.code) == (False, "ungrounded_identifier")
    assert "src/living_memory/retrieval.py" in verdict.signals["ungrounded"]


def test_grounding_accepts_a_home_path_the_judge_saw_redacted() -> None:
    """The judge is shown ``~/p/lm/...``; the transcript says ``/home/<user>/...``."""

    verdict = _gate().check(
        "The loader resolves ~/p/lm/src/app.py relative to the worktree, so a "
        "relative path fails there while an absolute one works.",
        candidate=_candidate(),
        context=SessionContext(),
    )
    assert verdict.accepted, f"{verdict.code}: {verdict.detail}"


def test_a_fact_the_diff_already_says_is_rederivable() -> None:
    record = SessionRecord(
        session_key="claude:t",
        source="claude",
        cli="claude",
        path="/tmp/t.jsonl",
        file_mutations=[
            FileMutation(
                ordinal=0,
                path="src/living_memory/storage.py",
                kind="update",
                span=SPAN,
                unified_diff=(
                    "--- a/src/living_memory/storage.py\n"
                    "+++ b/src/living_memory/storage.py\n@@\n"
                    "+    # pyproject.toml sets addopts -q so passing -q again\n"
                    "+    # collapses the pytest report to one line and hides\n"
                    "+    # which test failed\n"
                ),
            )
        ],
    )
    verdict = _gate().check(
        GOOD, candidate=_candidate(), context=SessionContext.from_record(record)
    )
    assert (verdict.accepted, verdict.code) == (False, "rederivable_from_diff")


def test_a_fact_the_session_was_already_shown_is_a_restatement() -> None:
    record = SessionRecord(
        session_key="claude:t",
        source="claude",
        cli="claude",
        path="/tmp/t.jsonl",
        recalls=[
            RecallInteraction(
                ordinal=0,
                query="pytest addopts",
                span=SPAN,
                delivered=[DeliveredNode(node_id="01ABC", rank=0, content=GOOD)],
            )
        ],
    )
    verdict = _gate().check(
        GOOD, candidate=_candidate(), context=SessionContext.from_record(record)
    )
    assert (verdict.accepted, verdict.code) == (False, "restates_recalled_node")
    assert verdict.signals["node_id"] == "01ABC"


@pytest.mark.parametrize(
    ("fact", "code"),
    [("", "empty_fact"), ("too short", "too_short"), ("x " * 900, "too_long")],
)
def test_shape_rules(fact: str, code: str) -> None:
    assert _check(fact).code == code


# --------------------------------------------------------------------------
# ordering, reporting, and the shared containment measure
# --------------------------------------------------------------------------


def test_rule_order_is_stable_for_a_text_that_trips_several_rules() -> None:
    """Journal wins over speculation and multi-fact, always."""

    fact = (
        "RESULT: DONE - I probably added tests to src/app.py.\n\n"
        "- also --strict-mcp-config\n- also 214ms\n- also storage.py"
    )
    assert _check(fact).code == "journal_phrasing"


def test_context_leak_outranks_every_quality_rule() -> None:
    fact = 'I probably added tests. {"scope": "project:lm", "agent": "a", "task": "t"}'
    assert _check(fact).code == "context_leak"


def test_rejection_breakdown_lists_every_known_code() -> None:
    breakdown = rejection_breakdown([_check("too short"), _check(GOOD)])
    assert set(REJECTION_CODES) <= set(breakdown)
    assert breakdown["too_short"] == 1


def test_containment_is_the_shared_grounding_measure() -> None:
    assert containment_of("src/app.py raised ValueError", "src/app.py raised ValueError") == 1.0
    assert containment_of("src/app.py", "") == 0.0
    assert containment_of("", "anything") == 0.0


def test_check_text_skips_the_session_rules_only() -> None:
    """The calibration path must apply every text rule and no session rule."""

    gate = Gate()
    journal = (
        "RESULT: DONE - added tests to src/app.py and src/living_memory/storage.py, "
        "and the whole suite is green at commit 9b1a78f."
    )
    assert gate.check_text(journal).code == "journal_phrasing"
    # ``check`` would reject GOOD for its ungrounded identifiers without a
    # candidate; ``check_text`` must not reach that rule at all.
    assert gate.check_text(GOOD).accepted
