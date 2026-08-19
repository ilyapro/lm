"""Contract tests for the contradiction -> ``memory_teach`` detector.

The detector's whole value is that it can overrule the judge. These tests are
therefore mostly *adversarial*: the judge is scripted to claim a contradiction
and each test asserts that the deterministic validator refuses it for the right
reason. A detector that only worked when the model behaved would be exactly the
thing the goal is trying to avoid -- a stale node keeps winning every recall,
and a wrong ``supersedes`` edge is worse than none.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import time
from typing import Any

import pytest

from living_memory import consolidation
from living_memory.postsession import corrections as C
from living_memory.postsession.judge import (
    FakeJudge,
    JudgeInvalidOutput,
    JudgeRefused,
    JudgeUnavailable,
    PromptConfig,
    default_redactor,
)
from living_memory.postsession.session import (
    DeliveredNode,
    FileMutation,
    ProposedOp,
    RecallInteraction,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
)

SOURCE = "claude"
PATH = "synthetic/session.jsonl"

NODE_ID = "01KTVGKXD2W9T9XRHX8X57WB0G"
OTHER_NODE_ID = "01KTSASZVQ7218KDDYE2TMP1XM"

NODE_CONTENT = (
    "scripts/postsession_corpus.py --build seals the manifest into "
    "artifacts/post-session/corpus.json and refuses to run when "
    "artifacts/post-session/corpus-index.jsonl already exists, so the operator "
    "has to delete the stale index by hand before every rebuild."
)

OTHER_NODE_CONTENT = (
    "living_memory.retrieval ranks bm25 above the vector channel whenever the "
    "query carries a quoted literal; see src/living_memory/retrieval.py:688 for "
    "the weighting that makes the causal graph channel empty today."
)

COMMAND = "python scripts/postsession_corpus.py --build"
OUTPUT = (
    "scanning roots...\n"
    "probed 3953 transcripts in 61.4s\n"
    "wrote artifacts/post-session/corpus-index.jsonl "
    "(overwrote the existing 2989141-byte index in place)\n"
    "wrote artifacts/post-session/corpus.json\n"
)

CLAIM = "refuses to run when artifacts/post-session/corpus-index.jsonl already exists"
QUOTE = (
    "wrote artifacts/post-session/corpus-index.jsonl "
    "(overwrote the existing 2989141-byte index in place)"
)
CORRECTION = (
    "scripts/postsession_corpus.py --build overwrites an existing "
    "artifacts/post-session/corpus-index.jsonl in place rather than refusing to "
    "run, so no manual delete is needed before a rebuild and the previous index "
    "is not preserved anywhere."
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _span(index: int, field: str = "") -> SourceSpan:
    return SourceSpan(SOURCE, PATH, index, field)


def make_record(
    *,
    nodes: tuple[tuple[str, str], ...] = ((NODE_ID, NODE_CONTENT),),
    recall_record_index: int = 1,
    with_assistant_claim: bool = False,
    with_diff: bool = False,
    teach_supersedes: str | None = None,
    extra_outputs: tuple[tuple[str, str, int], ...] = (),
) -> SessionRecord:
    """A structurally real one-recall session with one contradicting output."""

    record = SessionRecord(
        session_key="claude:test-session",
        source=SOURCE,
        cli="claude",
        path=PATH,
        cli_session_id="test-session",
        transcript_sha256="a" * 64,
        started_at="2026-08-19T00:00:00Z",
        ended_at="2026-08-19T01:00:00Z",
        record_count=40,
    )
    record.recalls.append(
        RecallInteraction(
            ordinal=0,
            query="how does the corpus builder handle an existing index",
            span=_span(recall_record_index, "mcpMeta.structuredContent"),
            recall_event_id="01RECALLEVENT0000000000001",
            delivered=[
                DeliveredNode(
                    node_id=node_id,
                    rank=rank,
                    content=content,
                    scope="project:lm",
                    level="trace",
                    agent="claude",
                    created_at="2026-07-02T00:00:00Z",
                    delivery="full",
                )
                for rank, (node_id, content) in enumerate(nodes)
            ],
            record_index=recall_record_index,
            position=0.05,
        )
    )
    record.tool_calls.append(
        ToolCall(
            ordinal=0,
            name="Bash",
            server=None,
            arguments={"command": COMMAND},
            span=_span(10, "message.content.tool_use"),
            call_id="call_10",
            ok=True,
            result_text=OUTPUT,
        )
    )
    # A memory tool call: never an observation, however loud its result.
    record.tool_calls.append(
        ToolCall(
            ordinal=1,
            name="memory_recall",
            server="living-memory",
            arguments={"query": "corpus index"},
            span=_span(11, "message.content.tool_use"),
            call_id="call_11",
            ok=True,
            result_text="artifacts/post-session/corpus-index.jsonl is never overwritten",
        )
    )
    for ordinal, (command, output, index) in enumerate(extra_outputs, start=2):
        record.tool_calls.append(
            ToolCall(
                ordinal=ordinal,
                name="Bash",
                server=None,
                arguments={"command": command},
                span=_span(index, "message.content.tool_use"),
                call_id=f"call_{index}",
                ok=True,
                result_text=output,
            )
        )
    record.turns.append(
        Turn(0, "user", "Rebuild the corpus and tell me what it did.", None, _span(0))
    )
    record.turns.append(
        Turn(
            1,
            "user",
            "<system-reminder>Tool use policy reminder</system-reminder>",
            None,
            _span(2),
        )
    )
    if with_assistant_claim:
        record.turns.append(
            Turn(
                2,
                "assistant",
                "The builder definitely refuses to overwrite "
                "artifacts/post-session/corpus-index.jsonl; memory was wrong.",
                None,
                _span(12),
            )
        )
    if with_diff:
        record.file_mutations.append(
            FileMutation(
                ordinal=0,
                path="scripts/postsession_corpus.py",
                kind="update",
                span=_span(13, "toolUseResult.structuredPatch"),
                unified_diff=(
                    "@@ -12,7 +12,7 @@\n"
                    "-    if index_path.exists():\n"
                    "-        raise SystemExit('index exists')\n"
                    "+    index_path.write_bytes(payload)\n"
                ),
                tool="Edit",
            )
        )
    if teach_supersedes:
        from living_memory.postsession.session import WriteInteraction

        record.writes.append(
            WriteInteraction(
                ordinal=0,
                kind="teach",
                span=_span(20, "mcpMeta.structuredContent"),
                node_id="01NEWCORRECTIVETRACE000001",
                supersedes=teach_supersedes,
            )
        )
    return record


def answer(**overrides: Any) -> dict[str, Any]:
    base = {
        "contradicted": True,
        "reason": "the build overwrote the index instead of refusing",
        "contradicted_claim": CLAIM,
        "observation_locator": f"{SOURCE}:{PATH}#10:tool_calls[0].result_text",
        "observation_quote": QUOTE,
        "correction": CORRECTION,
        "confidence": 0.85,
    }
    base.update(overrides)
    return base


def judged(record: SessionRecord, *responses: Any) -> tuple[C.DetectionResult, FakeJudge]:
    judge = FakeJudge(responses, prompt_config=PromptConfig(max_payload_bytes=24_000))
    result = C.detect_corrections(record, judge, split="train")
    return result, judge


def only_candidate(record: SessionRecord, node_id: str = NODE_ID) -> C.Candidate:
    for candidate in C.candidates(record):
        if candidate.node.node_id == node_id:
            return candidate
    raise AssertionError(f"no candidate for {node_id}")


def rejection(record: SessionRecord, **overrides: Any) -> C.Verdict:
    candidate = only_candidate(record, overrides.pop("_node_id", NODE_ID))
    _, excerpts = C.build_payload(record, candidate)
    already = overrides.pop("_already", ())
    return C.validate_verdict(
        record, candidate, answer(**overrides), excerpts, already_proposed=already
    )


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------


def test_anchors_pick_up_concrete_tokens_only():
    found = C.anchors(
        "the builder wrote src/living_memory/postsession/corpus.py with "
        "max_results=5 and --strict-mcp-config after 61.4s"
    )
    assert "src/living_memory/postsession/corpus.py" in found
    assert "max_results" in found
    assert "--strict-mcp-config" in found
    assert "61.4s" in found
    # prose and structural noise carry no aboutness
    assert "the" not in found
    assert "builder" not in found
    assert "5" not in found


def test_anchors_of_unrelated_texts_do_not_overlap():
    left = C.anchors(NODE_CONTENT)
    right = C.anchors(OTHER_NODE_CONTENT)
    assert left and right
    assert not left & right


def test_anchors_still_recognise_camel_case():
    """The camelCase pattern must keep matching after being made unambiguous."""

    for token in ("myVarName", "parseHTTPResponse", "getURL", "fooBar123"):
        assert token.lower() in C.anchors(f"call {token} now"), token


def test_anchors_do_not_backtrack_on_dense_alphanumeric_runs():
    """A long run of letters that never completes a token must stay linear.

    The camelCase pattern originally spelled its tail ``[A-Za-z0-9]*``, which
    overlaps the group's own ``[A-Z]`` start: every split of a letter run is a
    separate path, so a token that ultimately fails ``\\b`` backtracks
    exponentially. One real 3.2 KB command output in the eval split hung the
    detector for over four minutes. Anything superlinear here fails this test
    long before it can hang a run.
    """

    for size in (2_000, 4_000):
        text = "".join("aB" if index % 2 else "Cd" for index in range(size)) + "!"
        start = time.monotonic()
        C.anchors(text)
        assert time.monotonic() - start < 2.0, f"anchors() is superlinear at {size}"


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


def test_observations_cover_four_channels_and_exclude_agent_prose():
    record = make_record(with_assistant_claim=True, with_diff=True)
    found = C.observations(record)
    kinds = {item.kind for item in found}
    assert kinds == {"command_output", "file_diff", "operator_message"}
    texts = "\n".join(item.text for item in found)
    # the agent's own claim is not reality and must never be quotable
    assert "memory was wrong" not in texts
    # neither is a memory tool's own result
    assert "is never overwritten" not in texts
    # nor a host-injected pseudo-turn
    assert "system-reminder" not in texts
    assert found == sorted(found, key=lambda item: (item.record_index, item.kind, item.ordinal))


def test_observation_locators_are_unique_and_resolve_to_their_text():
    record = make_record(with_diff=True)
    found = C.observations(record)
    locators = [item.locator for item in found]
    assert len(locators) == len(set(locators))
    for item in found:
        span = SourceSpan.parse(item.locator)
        assert span.source == SOURCE
        assert span.path == PATH
        assert span.record_index == item.record_index
        assert span.field


def test_test_results_are_labelled_as_such():
    record = make_record(
        extra_outputs=(("python -m pytest tests/test_x.py -q", "3 passed in 0.2s\n", 14),)
    )
    kinds = {item.label: item.kind for item in C.observations(record)}
    assert kinds["Bash: python -m pytest tests/test_x.py -q"] == "test_result"


# --------------------------------------------------------------------------
# memory round trips are never observations
# --------------------------------------------------------------------------
#
# Transcribed from the shapes the real corpus uses: a large part of it reaches
# the memory server from *inside* a script tool, so the host records an ordinary
# `exec` call with `server=None` and the memory result lands in `result_text`.
# `ToolCall.is_memory_tool` cannot see those, and the normalizer stores the
# projected memory call separately -- without the result text.

EXEC_TEACH_INPUT = (
    'const r=await tools.mcp__living_memory__memory_teach({\n'
    f'  trace_id:"{NODE_ID}",\n'
    '  correction:"Correction: scripts/postsession_corpus.py --build overwrites '
    'artifacts/post-session/corpus-index.jsonl in place.",\n'
    '  confidence:1\n});'
)

EXEC_TEACH_RESULT = (
    '[{"type": "input_text", "text": "Script completed\\nWall time 0.1 seconds\\nOutput:\\n"}, '
    '{"type": "input_text", "text": "{\\"corrective_trace\\":{\\"id\\":\\"01KZRYNMKPFWKVFJ7Z1M4QNMZ2\\",'
    '\\"content\\":\\"Correction: scripts/postsession_corpus.py --build overwrites '
    'artifacts/post-session/corpus-index.jsonl in place.\\"}}"}]'
)

EXEC_RECALL_RESULT = (
    '[{"type": "input_text", "text": "{\\"query\\":\\"corpus index\\",'
    '\\"recall_event_id\\":\\"01KZV8X7GJTT7KMM7QVEY4S8B2\\",\\"count\\":1,\\"results\\":'
    '[{\\"node\\":{\\"id\\":\\"01KTSB8N0BN0GMK48N8DB5H03C\\",\\"content\\":'
    '\\"artifacts/post-session/corpus-index.jsonl is never overwritten\\"}}]}"}]'
)


def with_tool_call(
    record: SessionRecord, *, name: str, arguments: dict[str, Any], result: str, index: int
) -> SessionRecord:
    record.tool_calls.append(
        ToolCall(
            ordinal=len(record.tool_calls),
            name=name,
            server=None,
            arguments=arguments,
            span=_span(index, "message.content.tool_use"),
            call_id=f"call_{index}",
            ok=True,
            result_text=result,
        )
    )
    return record


def test_a_memory_call_through_a_script_tool_is_not_an_observation():
    record = with_tool_call(
        make_record(),
        name="exec",
        arguments={"input": EXEC_TEACH_INPUT},
        result=EXEC_TEACH_RESULT,
        index=14,
    )
    texts = "\n".join(item.text for item in C.observations(record))
    assert "corrective_trace" not in texts
    assert C.is_memory_roundtrip(record.tool_calls[-1])


def test_a_memory_response_envelope_is_not_an_observation():
    """Caught on the way back even when the call text gives nothing away."""

    record = with_tool_call(
        make_record(),
        name="exec",
        arguments={"input": "const r = await helper.fetchContext();"},
        result=EXEC_RECALL_RESULT,
        index=15,
    )
    texts = "\n".join(item.text for item in C.observations(record))
    assert "is never overwritten" not in texts
    assert C.is_memory_roundtrip(record.tool_calls[-1])


def test_grepping_the_memory_tool_names_is_still_an_observation():
    """The exclusion must not swallow real evidence about this repository.

    Much of the corpus is sessions working *on* Living Memory, where greping
    these very names is ordinary observed reality. Blocking those would trade a
    circular-evidence bug for a blind detector.
    """

    record = with_tool_call(
        make_record(),
        name="Bash",
        arguments={"command": 'grep -rn "memory_teach(" src/living_memory'},
        result=(
            "src/living_memory/consolidation.py:516:def memory_teach(store, trace_id, "
            "correction, confidence=None, context=None):\n"
        ),
        index=16,
    )
    call = record.tool_calls[-1]
    assert not C.is_memory_roundtrip(call)
    assert any("consolidation.py:516" in item.text for item in C.observations(record))


def test_the_sessions_own_teach_payload_cannot_become_the_contradicting_evidence():
    """The circular case, end to end.

    A session that teaches through a script tool gets its own correction echoed
    back in the tool result. That echo shares every anchor with the node it
    corrects, so it outranks real evidence and the judge happily "finds" the
    contradiction the agent had already written. The detector would then be
    scoring itself for rediscovering its own input.
    """

    record = with_tool_call(
        make_record(),
        name="exec",
        arguments={"input": EXEC_TEACH_INPUT},
        result=EXEC_TEACH_RESULT,
        index=14,
    )
    candidate = only_candidate(record)
    echo_locator = f"{SOURCE}:{PATH}#14:tool_calls[2].result_text"
    assert echo_locator not in {item.locator for item in candidate.observations}

    _, excerpts = C.build_payload(record, candidate)
    verdict = C.validate_verdict(
        record,
        candidate,
        answer(
            observation_locator=echo_locator,
            observation_quote=(
                "Correction: scripts/postsession_corpus.py --build overwrites "
                "artifacts/post-session/corpus-index.jsonl in place."
            ),
        ),
        excerpts,
    )
    assert not verdict.accepted
    assert verdict.reason == "unknown_locator"


# --------------------------------------------------------------------------
# excerpting
# --------------------------------------------------------------------------


def test_best_excerpt_returns_the_anchor_dense_window_contiguously():
    filler = "\n".join(f"line {index} of noise" for index in range(200))
    text = f"{filler}\nthe answer is max_results=5 in server.py\n{filler}"
    excerpt, offset = C.best_excerpt(text, {"max_results", "server.py"}, 200)
    assert "max_results=5" in excerpt
    assert len(excerpt) <= 200
    assert text[offset : offset + len(excerpt)] == excerpt


def test_best_excerpt_is_identity_below_the_limit():
    assert C.best_excerpt("short", {"short"}, 100) == ("short", 0)


def test_best_excerpt_reaches_anchors_inside_an_unbroken_blob():
    """A codex ``exec`` result is one JSON line, tens of kilobytes of it.

    Its newlines are ``\\n`` escapes, so line alignment sees a single line and
    used to fall back to ``text[:limit]`` -- the host's own "Script completed /
    Warning: truncated output" header, which can never mention the stored fact.
    """

    header = '[{"type": "input_text", "text": "Script completed\\nWall time 0.4 seconds\\nOutput:\\n"}, '
    noise = '{\\"unrelated\\":\\"padding value\\"},' * 400
    payload = '{\\"pipeline_id\\":3473927,\\"job\\":\\"jest-desktop-test/branches\\",\\"status\\":\\"failed\\"}'
    text = header + '{"type": "input_text", "text": "' + noise + payload + '"}]'
    assert text.count("\n") == 0 and len(text) > 5_000

    excerpt, offset = C.best_excerpt(text, {"3473927", "jest-desktop-test/branches"}, 1_200)
    assert "3473927" in excerpt
    assert "jest-desktop-test/branches" in excerpt
    assert len(excerpt) <= 1_200
    # contiguity is the whole basis of the verbatim check
    assert text[offset : offset + len(excerpt)] == excerpt


def test_best_excerpt_without_any_anchor_present_still_returns_a_window():
    text = "x" * 5_000
    excerpt, offset = C.best_excerpt(text, {"nothing-here"}, 100)
    assert len(excerpt) == 100
    assert text[offset : offset + len(excerpt)] == excerpt


# --------------------------------------------------------------------------
# candidate pairing
# --------------------------------------------------------------------------


def test_candidates_pair_the_node_with_later_related_observations():
    record = make_record()
    built = C.candidates(record)
    assert len(built) == 1
    candidate = built[0]
    assert candidate.node.node_id == NODE_ID
    assert all(item.record_index > candidate.recall.record_index for item in candidate.observations)
    assert candidate.score > 0


def test_candidates_ignore_observations_that_precede_the_recall():
    counters: dict[str, int] = {}
    record = make_record(recall_record_index=30)
    assert C.candidates(record, counters=counters) == []
    assert counters["no_related_observation"] == 1


def test_candidates_skip_a_node_the_session_already_taught():
    counters: dict[str, int] = {}
    record = make_record(teach_supersedes=NODE_ID)
    assert C.candidates(record, counters=counters) == []
    assert counters["already_taught_in_session"] == 1


def test_candidates_skip_unrelated_nodes_and_report_it():
    counters: dict[str, int] = {}
    record = make_record(nodes=((OTHER_NODE_ID, OTHER_NODE_CONTENT),))
    assert C.candidates(record, counters=counters) == []
    assert counters["no_related_observation"] == 1


def test_candidates_respect_the_per_session_cap():
    nodes = tuple(
        (f"01NODE{index:020d}", f"{NODE_CONTENT} variant {index}") for index in range(20)
    )
    counters: dict[str, int] = {}
    built = C.candidates(make_record(nodes=nodes), counters=counters)
    assert len(built) == C.DEFAULT_CONFIG.max_nodes_per_session
    assert counters["over_session_cap"] == 20 - C.DEFAULT_CONFIG.max_nodes_per_session


def test_payload_offers_only_the_ranked_observations_and_carries_the_node():
    record = make_record(with_diff=True)
    candidate = only_candidate(record)
    payload, excerpts = C.build_payload(record, candidate)
    assert payload["stored_fact"]["node_id"] == NODE_ID
    assert payload["stored_fact"]["content"] == NODE_CONTENT
    assert set(excerpts) == {item["locator"] for item in payload["observations"]}
    assert set(excerpts) == {item.locator for item in candidate.observations}
    for item in payload["observations"]:
        assert item["excerpt"] in excerpts[item["locator"]]


# --------------------------------------------------------------------------
# verbatim classification
# --------------------------------------------------------------------------


def test_quote_match_mode_distinguishes_exact_from_home_elision_from_secret():
    raw = "read /home/sfx/p/lm/corpus.json with LM_AUTH_TOKEN=sk-lm-0123456789abcdefghij"
    assert C.quote_match_mode(raw, "read") == "exact"
    assert C.quote_match_mode(raw, "/home/sfx/p/lm/corpus.json") == "exact"
    assert C.quote_match_mode(raw, "~/p/lm/corpus.json") == "home_path_normalized"
    assert C.quote_match_mode(raw, "LM_AUTH_TOKEN=<redacted>") == "redacted_secret"
    assert C.quote_match_mode(raw, "no such text") is None


def test_quote_match_mode_tolerates_rewrapped_whitespace_only():
    raw = "wrote the index\nand then the manifest"
    assert C.quote_match_mode(raw, "wrote the index and then the manifest") == "exact"
    assert C.quote_match_mode(raw, "wrote the manifest and then the index") is None


# --------------------------------------------------------------------------
# the accepted path
# --------------------------------------------------------------------------


def test_accepted_verdict_builds_a_teach_op_matching_memory_teach():
    record = make_record()
    verdict = rejection(record)
    assert verdict.accepted, verdict.reason
    op = verdict.op
    assert op is not None
    assert op.kind == "teach"

    # The payload is the memory_teach call, key for key.
    signature = inspect.signature(consolidation.memory_teach)
    assert set(op.payload) == {"trace_id", "correction", "confidence", "context"}
    assert set(op.payload) <= set(signature.parameters)
    assert op.payload["trace_id"] == NODE_ID
    assert op.payload["correction"] == CORRECTION
    assert 0.0 <= op.payload["confidence"] <= 1.0

    context = op.payload["context"]
    assert context["lesson_kind"] == C.LESSON_KIND == "correction"
    assert context["agent"] == C.EXTRACTOR_AGENT
    assert context["corrected_node_id"] == NODE_ID
    assert context["recall_event_id"] == "01RECALLEVENT0000000000001"
    assert context["source_session_key"] == record.session_key
    assert context["scope"] == "project:lm"

    assert op.provenance.session_key == record.session_key
    assert op.provenance.transcript_sha256 == record.transcript_sha256
    op.validate()


def test_accepted_evidence_occurs_verbatim_in_the_session_record():
    record = make_record()
    op = rejection(record).op
    assert op is not None
    haystacks = [item.text for item in C.observations(record)]
    haystacks += [node.content or "" for node in record.recalls[0].delivered]
    for item in op.evidence:
        assert any(C.quote_match_mode(text, item.quote) == "exact" for text in haystacks), item
        span = SourceSpan.parse(item.locator)
        assert span.path == record.path


def test_proposed_op_round_trips_through_the_wire():
    op = rejection(make_record()).op
    assert op is not None
    restored = ProposedOp.from_dict(json.loads(json.dumps(op.to_dict())))
    assert restored == op
    restored.validate()


def test_detect_corrections_emits_teach_and_never_remember():
    record = make_record()
    result, judge = judged(record, answer())
    assert [op.kind for op in result.ops] == ["teach"]
    assert result.judged == 1
    assert result.rejections == {}
    assert len(judge.calls) == 1
    # the judge saw the node content and the observation, not the agent's prose
    prompt = judge.calls[0]["prompt"]
    assert "corpus-index.jsonl" in prompt
    assert "memory was wrong" not in prompt


# --------------------------------------------------------------------------
# the vetoes -- each one on its own
# --------------------------------------------------------------------------


def test_veto_judge_says_no():
    verdict = rejection(make_record(), contradicted=False)
    assert (verdict.accepted, verdict.reason) == (False, "judge_says_no")
    assert verdict.judge_said_contradicted is False


@pytest.mark.parametrize(
    "field", ["contradicted_claim", "observation_locator", "observation_quote", "correction"]
)
def test_veto_missing_field(field: str):
    verdict = rejection(make_record(), **{field: ""})
    assert verdict.reason == "missing_field"
    assert field in verdict.detail


def test_locator_is_resolved_through_the_redaction_the_judge_was_shown():
    """The judge echoes what it saw: a home-elided path. That is not an invention."""

    record = make_record()
    record.path = "/home/sfx/.claude/projects/-home-sfx-p-lm/session.jsonl"
    record.source = SOURCE
    for call in record.tool_calls:
        call.span = SourceSpan(SOURCE, record.path, call.span.record_index, call.span.field)
    for recall in record.recalls:
        recall.span = SourceSpan(SOURCE, record.path, recall.span.record_index, recall.span.field)
    for turn in record.turns:
        turn.span = SourceSpan(SOURCE, record.path, turn.span.record_index, turn.span.field)

    candidate = only_candidate(record)
    _, excerpts = C.build_payload(record, candidate)
    canonical = candidate.observations[0].locator
    assert "/home/sfx" in canonical
    elided = default_redactor(canonical)
    assert elided != canonical

    assert C.resolve_locator(elided, excerpts) == canonical
    verdict = C.validate_verdict(
        record, candidate, answer(observation_locator=elided), excerpts
    )
    assert verdict.accepted, verdict.reason
    assert verdict.op is not None
    assert verdict.op.evidence[0].locator == canonical


def test_veto_unknown_locator():
    verdict = rejection(make_record(), observation_locator="claude:other.jsonl#99:turns[0].text")
    assert verdict.reason == "unknown_locator"


def test_veto_claim_not_in_node():
    verdict = rejection(
        make_record(),
        contradicted_claim="refuses to run when the sealed manifest already exists",
    )
    assert verdict.reason == "claim_not_in_node"


def test_veto_quote_not_verbatim():
    verdict = rejection(
        make_record(),
        observation_quote="overwrote artifacts/post-session/corpus-index.jsonl without asking",
    )
    assert verdict.reason == "quote_not_verbatim"


def test_veto_quote_too_thin():
    verdict = rejection(make_record(), observation_quote="wrote")
    assert verdict.reason == "quote_too_thin"


def test_veto_quote_touching_a_redacted_secret():
    secret_output = (
        "restored artifacts/post-session/corpus-index.jsonl using "
        "LM_AUTH_TOKEN=sk-lm-0123456789abcdefghij from the environment\n"
    )
    record = make_record(extra_outputs=(("./restore.sh", secret_output, 14),))
    verdict = rejection(
        record,
        observation_locator=f"{SOURCE}:{PATH}#14:tool_calls[2].result_text",
        observation_quote=(
            "restored artifacts/post-session/corpus-index.jsonl using "
            "LM_AUTH_TOKEN=<redacted> from the environment"
        ),
    )
    assert verdict.reason == "quote_touches_redacted_secret"


def test_veto_absence_of_evidence():
    empty_output = (
        "grep -c 'refuses to run' scripts/postsession_corpus.py\n"
        "no matches for artifacts/post-session/corpus-index.jsonl in the build log\n"
    )
    record = make_record(extra_outputs=(("grep -c refuse", empty_output, 15),))
    verdict = rejection(
        record,
        observation_locator=f"{SOURCE}:{PATH}#15:tool_calls[2].result_text",
        observation_quote=(
            "no matches for artifacts/post-session/corpus-index.jsonl in the build log"
        ),
    )
    assert verdict.reason == "absence_of_evidence"


def test_veto_claim_that_names_nothing_concrete():
    """A clause with no identifier in it cannot be pinned to an observation."""

    anchorless = "so the operator has to delete the stale index by hand"
    assert anchorless in NODE_CONTENT
    assert not C.anchors(anchorless)
    verdict = rejection(make_record(), contradicted_claim=anchorless)
    assert verdict.reason == "claim_has_no_anchor"


def test_veto_no_shared_anchor_when_the_quote_is_about_another_fact():
    """The 'session superseded a DIFFERENT fact' failure mode, at op level."""

    # One output, two subjects: the index rebuild (which the node is about) and
    # the retrieval weights (which another delivered node is about). The pair is
    # legitimately offered; the *quote* is about the wrong fact.
    mixed_output = (
        "artifacts/post-session/corpus-index.jsonl: 3953 rows\n"
        "vector 0.55 outranks bm25 0.30 even for quoted literals\n"
    )
    record = make_record(
        nodes=((NODE_ID, NODE_CONTENT), (OTHER_NODE_ID, OTHER_NODE_CONTENT)),
        extra_outputs=(("python scripts/postsession_corpus.py --stats", mixed_output, 16),),
    )
    verdict = rejection(
        record,
        observation_locator=f"{SOURCE}:{PATH}#16:tool_calls[2].result_text",
        observation_quote="vector 0.55 outranks bm25 0.30 even for quoted literals",
    )
    assert verdict.reason == "no_shared_anchor"


def test_veto_rephrasing():
    verdict = rejection(
        make_record(),
        correction=(
            "scripts/postsession_corpus.py --build refuses to run when "
            "artifacts/post-session/corpus-index.jsonl already exists on disk."
        ),
    )
    assert verdict.reason == "rephrasing"


@pytest.mark.parametrize(
    "correction",
    [
        "It overwrites the file instead, so the manual delete in "
        "artifacts/post-session/corpus-index.jsonl is unnecessary now.",
        "As noted above, scripts/postsession_corpus.py --build overwrites "
        "artifacts/post-session/corpus-index.jsonl in place every time it runs.",
        "The recalled fact is wrong about scripts/postsession_corpus.py --build "
        "and artifacts/post-session/corpus-index.jsonl being protected from rebuilds.",
        "Not true anymore.",
    ],
)
def test_veto_correction_that_cannot_be_read_alone(correction: str):
    verdict = rejection(make_record(), correction=correction)
    assert verdict.reason == "correction_not_self_contained"


def test_veto_correction_about_something_else_entirely():
    verdict = rejection(
        make_record(),
        correction=(
            "living_memory.retrieval weights the vector channel at 0.55 and bm25 at "
            "0.30, so a quoted literal no longer forces a lexical-first ranking in "
            "src/living_memory/retrieval.py."
        ),
    )
    assert verdict.reason == "correction_not_self_contained"
    assert "names nothing" in verdict.detail


def test_veto_low_confidence():
    verdict = rejection(make_record(), confidence=0.3)
    assert verdict.reason == "low_confidence"


def test_veto_duplicate_node():
    verdict = rejection(make_record(), _already=(NODE_ID,))
    assert verdict.reason == "duplicate_node"


def test_veto_unknown_node():
    record = make_record()
    candidate = only_candidate(record)
    _, excerpts = C.build_payload(record, candidate)
    ghost = C.Candidate(
        recall=candidate.recall,
        node=DeliveredNode(node_id="01GHOST0000000000000000001", rank=0, content=NODE_CONTENT),
        scored=candidate.scored,
    )
    verdict = C.validate_verdict(record, ghost, answer(), excerpts)
    assert verdict.reason == "unknown_node"


def test_veto_observation_that_precedes_the_recall():
    record = make_record()
    candidate = only_candidate(record)
    _, excerpts = C.build_payload(record, candidate)
    late = C.Candidate(
        recall=RecallInteraction(
            ordinal=0,
            query=candidate.recall.query,
            span=candidate.recall.span,
            recall_event_id=candidate.recall.recall_event_id,
            delivered=candidate.recall.delivered,
            record_index=999,
        ),
        node=candidate.node,
        scored=candidate.scored,
    )
    verdict = C.validate_verdict(record, late, answer(), excerpts)
    assert verdict.reason == "observation_not_after_recall"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (JudgeRefused("payload unusable"), "judge_refused"),
        (JudgeInvalidOutput("no schema-valid object"), "judge_invalid_output"),
        (JudgeUnavailable("cli not found"), "judge_unavailable"),
    ],
)
def test_judge_failures_never_produce_ops(error: Exception, expected: str):
    result, _ = judged(make_record(), error)
    assert result.ops == []
    assert result.rejections == {expected: 1}


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------


def test_every_negative_control_is_actually_judged():
    """A control that never reaches the judge would prove nothing."""

    for control in C.NEGATIVE_CONTROLS:
        built = C.candidates(control.record)
        assert built, f"{control.name} was filtered out before the judge"


def test_negative_controls_yield_nothing_when_the_judge_says_no():
    judge = FakeJudge(
        [{"contradicted": False, "reason": "nothing observed disagrees"}] * 12,
        prompt_config=PromptConfig(max_payload_bytes=24_000),
    )
    report = C.run_negative_controls(judge)
    assert [item["name"] for item in report] == [c.name for c in C.NEGATIVE_CONTROLS]
    assert all(item["passed"] and item["ops"] == 0 for item in report)
    assert all(item["judged"] >= 1 for item in report)


def _control(name: str) -> C.NegativeControl:
    return next(item for item in C.NEGATIVE_CONTROLS if item.name == name)


def test_control_matched_reality_survives_a_judge_that_says_yes():
    """The measurements confirm the fact; a claimed contradiction is vetoed."""

    control = _control("fact-matched-reality")
    node = control.record.recalls[0].delivered[0]
    candidate = C.candidates(control.record)[0]
    locator = candidate.observations[0].locator
    result, _ = judged(
        control.record,
        {
            "contradicted": True,
            "reason": "invented",
            "contradicted_claim": "Snapshotting living-memory's sqlite database with `cp` alone is",
            "observation_locator": locator,
            "observation_quote": "139696",
            "correction": (
                "Snapshotting living-memory's sqlite database with cp is safe because "
                "the -wal file is checkpointed automatically on every write."
            ),
            "confidence": 0.9,
        },
    )
    assert result.ops == []
    assert set(result.rejections) <= {
        "quote_too_thin",
        "quote_asserts_nothing",
        "no_shared_anchor",
        "judge_says_no",
    }
    assert node.node_id  # the control really does deliver a node


def test_control_different_fact_cannot_be_attached_to_the_recalled_node():
    """The session corrects max_results; the recalled node is about depth."""

    control = _control("different-fact-superseded")
    candidate = C.candidates(control.record)[0]
    operator = next(
        item for item in candidate.observations if item.kind == "operator_message"
    )
    assert "max_results" in operator.text
    result, _ = judged(
        control.record,
        {
            "contradicted": True,
            "reason": "the operator corrected something",
            "contradicted_claim": "depth='causal' to widen graph traversal beyond the numeric hop count",
            "observation_locator": operator.locator,
            "observation_quote": "the max_results default in src/living_memory/server.py is 5, not the 10 the docs claim",
            "correction": (
                "memory_recall's max_results parameter defaults to 5 in "
                "src/living_memory/server.py, not the 10 the documentation claims, so a "
                "caller that needs more results has to pass it explicitly."
            ),
            "confidence": 0.95,
        },
    )
    assert result.ops == []
    # Either aboutness veto is the right answer here: the claim is about the
    # depth parameter, the quote is about max_results, and the two never meet.
    assert set(result.rejections) <= {"claim_has_no_anchor", "no_shared_anchor"}
    assert sum(result.rejections.values()) == 1


def test_veto_correction_that_belongs_to_another_delivered_node():
    """The other fact was delivered too: the correction is about *it*."""

    mixed_output = (
        "artifacts/post-session/corpus-index.jsonl: 3953 rows\n"
        "vector 0.55 now outranks bm25 0.30 in src/living_memory/retrieval.py\n"
    )
    record = make_record(
        nodes=((NODE_ID, NODE_CONTENT), (OTHER_NODE_ID, OTHER_NODE_CONTENT)),
        extra_outputs=(("python scripts/postsession_corpus.py --stats", mixed_output, 16),),
    )
    verdict = rejection(
        record,
        observation_locator=f"{SOURCE}:{PATH}#16:tool_calls[2].result_text",
        observation_quote=(
            "artifacts/post-session/corpus-index.jsonl: 3953 rows\n"
            "vector 0.55 now outranks bm25 0.30 in src/living_memory/retrieval.py"
        ),
        correction=(
            "living_memory.retrieval weights the vector channel at 0.55 above bm25 at "
            "0.30 in src/living_memory/retrieval.py, so the ranking that produced the "
            "artifacts/post-session/corpus-index.jsonl row counts is no longer "
            "lexical-first for a quoted literal."
        ),
    )
    assert verdict.reason == "better_matched_by_another_node"
    assert OTHER_NODE_ID in verdict.detail


def test_control_untouched_fact_is_not_corrected_by_an_incidental_mention():
    control = _control("fact-untouched")
    candidate = C.candidates(control.record)[0]
    listing = next(
        item for item in candidate.observations if "rg -l" in item.label
    )
    result, _ = judged(
        control.record,
        {
            "contradicted": True,
            "reason": "the file list mentions f_theta.cpp",
            "contradicted_claim": "src/model/f_theta.cpp:475 is a precision-policy validation site",
            "observation_locator": listing.locator,
            "observation_quote": "src/model/f_theta.cpp",
            "correction": (
                "src/model/f_theta.cpp calls be->vector_pad_truncate at runtime, so "
                "src/model/stems/stems.cpp:213 is not the only model-layer call site."
            ),
            "confidence": 0.9,
        },
    )
    assert result.ops == []
    assert result.rejections == {"quote_asserts_nothing": 1}


# --------------------------------------------------------------------------
# ground-truth recovery
# --------------------------------------------------------------------------


def test_production_skips_a_node_the_session_taught_but_recovery_judges_it():
    """The corpus's own labelled positives: production ignores, recovery measures."""

    record = make_record(teach_supersedes=NODE_ID)
    assert C.taught_delivered_nodes(record) == (NODE_ID,)

    produced, _ = judged(record, answer())
    assert produced.judged == 0
    assert produced.ops == []
    assert produced.pair_stats["already_taught_in_session"] == 1

    judge = FakeJudge([answer()], prompt_config=PromptConfig(max_payload_bytes=24_000))
    recovered = C.recover_known_corrections(record, judge, split="train")
    assert recovered.judged == 1
    assert [op.payload["trace_id"] for op in recovered.ops] == [NODE_ID]


def test_recovery_is_empty_when_the_session_taught_nothing_it_was_shown():
    record = make_record(teach_supersedes="01SOMEOTHERNODE00000000001")
    assert C.taught_delivered_nodes(record) == ()
    judge = FakeJudge([], prompt_config=PromptConfig(max_payload_bytes=24_000))
    assert C.recover_known_corrections(record, judge).judged == 0


def test_recovery_section_is_marked_unexecutable():
    record = make_record(teach_supersedes=NODE_ID)
    judge = FakeJudge([answer()], prompt_config=PromptConfig(max_payload_bytes=24_000))
    recovered = C.recover_known_corrections(record, judge, split="eval")
    report = C.build_report(
        [],
        splits_read=["eval"],
        config=C.DEFAULT_CONFIG,
        judge_backend="fake",
        judge_model=None,
        judge_stats=judge.stats,
        recovery=[recovered],
    )
    section = report["ground_truth_recovery"]
    assert section["executable"] is False
    assert section["recovered"] == 1
    found = section["corrections"][0]
    assert found["op"]["kind"] == "teach"
    assert found["op"]["payload"]["trace_id"] == NODE_ID
    assert report["totals"]["ops"] == 0  # recovery never inflates the run's output


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


RESERVED_SPLIT = "hold" + "out"


def test_report_records_which_splits_were_read_and_names_no_reserved_split():
    record = make_record()
    result, judge = judged(record, answer())
    report = C.build_report(
        [result],
        splits_read=["eval"],
        config=C.DEFAULT_CONFIG,
        judge_backend="fake",
        judge_model=None,
        judge_stats=judge.stats,
        controls=[{"name": "x", "ops": 0, "passed": True}],
    )
    assert report["splits_read"] == ["eval"]
    assert report["totals"]["ops"] == 1
    assert report["ops"][0]["kind"] == "teach"
    assert RESERVED_SPLIT not in json.dumps(report)


def test_reserved_split_sessions_are_never_read(tmp_path: Path):
    """A group holding any reserved-bucket token is skipped whole."""

    from living_memory.postsession.corpus import (
        SPLITS,
        SessionEntry,
        assign_splits,
        bucket_for,
        split_for_bucket,
        write_index,
    )

    assert RESERVED_SPLIT in SPLITS
    reserved_key = next(
        f"claude:{index}"
        for index in range(10_000)
        if split_for_bucket(bucket_for(f"claude:{index}")) == RESERVED_SPLIT
    )
    train_key = next(
        f"claude:{index}"
        for index in range(10_000)
        if split_for_bucket(bucket_for(f"claude:{index}")) == "train"
    )
    entries = [
        SessionEntry(session_key=reserved_key, source="claude", cli="claude", path="a.jsonl"),
        SessionEntry(session_key=train_key, source="claude", cli="claude", path="b.jsonl"),
        # b and the reserved session are two recordings of one conversation:
        # the group is contaminated and neither may be read.
        SessionEntry(
            session_key="claude:linked",
            source="claude",
            cli="claude",
            path="c.jsonl",
            linked_session_ids=(train_key.split(":", 1)[1], reserved_key.split(":", 1)[1]),
        ),
    ]
    assign_splits(entries)
    index_path = tmp_path / "index.jsonl"
    write_index(entries, index_path)

    for split in ("train", "eval"):
        for entry in C._load_entries(index_path, split):
            assert entry.session_key != reserved_key
    keys = {entry.session_key for entry in C._load_entries(index_path, "train")}
    assert reserved_key not in keys
    assert "claude:linked" not in keys


def test_reserved_split_name_is_masked_out_of_a_tracked_report():
    """Real corrections quote sessions that discuss the sealed split by name."""

    assert C.reserved_splits() == (RESERVED_SPLIT,)
    quoted = f"the {RESERVED_SPLIT} packet was resealed at commit 78e02d9"
    masked = C.mask_reserved_splits(quoted)
    assert RESERVED_SPLIT not in masked
    assert "78e02d9" in masked and "resealed" in masked
    assert C.mask_reserved_splits("nothing to mask") == "nothing to mask"


def test_report_json_survives_redaction_and_stays_loadable():
    record = make_record()
    result, judge = judged(record, answer())
    report = C.build_report(
        [result],
        splits_read=["eval"],
        config=C.DEFAULT_CONFIG,
        judge_backend="fake",
        judge_model=None,
        judge_stats=judge.stats,
    )
    text = default_redactor(json.dumps(report, ensure_ascii=False))
    assert json.loads(text)["totals"]["ops"] == 1


def test_published_reason_vocabulary_is_complete():
    """Every reason the detector can emit is declared in REASONS."""

    record = make_record()
    emitted = {
        rejection(record, contradicted=False).reason,
        rejection(record, contradicted_claim="").reason,
        rejection(record, observation_locator="nope").reason,
        rejection(record, contradicted_claim="not in the node at all").reason,
        rejection(record, observation_quote="never written anywhere in this session").reason,
        rejection(record, observation_quote="wrote").reason,
        rejection(record, observation_quote="artifacts/post-session/corpus-index.jsonl").reason,
        rejection(record, confidence=0.1).reason,
        rejection(record, correction="Not true anymore.").reason,
        rejection(record, _already=(NODE_ID,)).reason,
        rejection(record, contradicted_claim="so the operator has to delete the stale index by hand").reason,
    }
    assert emitted <= set(C.REASONS)


# --------------------------------------------------------------------------
# the tracked eval artifact
# --------------------------------------------------------------------------

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "post-session" / "corrections-eval.json"


def _artifact() -> dict[str, Any]:
    assert ARTIFACT.exists(), f"{ARTIFACT} is a required artifact of this node"
    text = ARTIFACT.read_text(encoding="utf-8")
    assert text.strip(), "the eval report must not be empty"
    return json.loads(text)


def test_eval_artifact_read_only_the_eval_split():
    report = _artifact()
    assert report["splits_read"] == ["eval"]
    assert RESERVED_SPLIT not in ARTIFACT.read_text(encoding="utf-8")


def _artifact_ops(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Every teach op the eval run produced, from both of its passes.

    The production pass proposes; the ground-truth recovery pass re-derives
    corrections the sessions already made, and is marked unexecutable. Both are
    validated ops and both must satisfy the same structural contract.
    """

    ops = list(report["ops"])
    ops += [item["op"] for item in report["ground_truth_recovery"]["corrections"]]
    return ops


def test_eval_artifact_proposes_teach_ops_and_nothing_else():
    report = _artifact()
    assert report["ops_are_proposals"] is True
    assert report["living_memory_writes"] == 0
    assert report["ground_truth_recovery"]["executable"] is False
    ops = _artifact_ops(report)
    assert ops, "the eval run produced no teach ops at all"
    signature = inspect.signature(consolidation.memory_teach)
    for raw in ops:
        op = ProposedOp.from_dict(raw)
        op.validate()
        assert op.kind == "teach"
        assert set(op.payload) == {"trace_id", "correction", "confidence", "context"}
        assert set(op.payload) <= set(signature.parameters)
        assert op.payload["context"]["lesson_kind"] == C.LESSON_KIND
        assert op.payload["context"]["agent"] == C.EXTRACTOR_AGENT
        assert op.payload["trace_id"] == op.payload["context"]["corrected_node_id"]
        assert op.evidence and all(item.quote.strip() for item in op.evidence)
        assert op.provenance.session_key and op.provenance.transcript_sha256


def test_eval_artifact_ops_name_a_node_the_run_recorded_as_delivered():
    report = _artifact()
    accepted = {item["node_id"]: item for item in report["accepted"]}
    accepted.update(
        {
            item["verdict"]["node_id"]: item["verdict"]
            for item in report["ground_truth_recovery"]["corrections"]
        }
    )
    assert accepted
    for raw in _artifact_ops(report):
        node_id = raw["payload"]["trace_id"]
        assert node_id in accepted, f"{node_id} is not among the run's accepted verdicts"
        assert accepted[node_id]["quote_match"] in {"exact", "home_path_normalized"}
        assert accepted[node_id]["observation_kind"] in C.OBSERVATION_KINDS


def test_eval_artifact_evidence_is_never_a_memory_round_trip():
    """No op may be grounded in Living Memory's own traffic.

    This is the artifact-level form of :func:`C.is_memory_roundtrip`. The
    failure it pins actually happened: before that exclusion existed, the eval
    run's single op quoted the session's *own* ``memory_teach`` payload --
    echoed back in the tool result of the script call that made it -- as proof
    that the taught node was wrong. A detector rediscovering its own input
    scores itself for free, so this must be checked on the tracked evidence and
    not only on synthetic records.
    """

    report = _artifact()
    ops = _artifact_ops(report)
    assert ops
    for raw in ops:
        op = ProposedOp.from_dict(raw)
        for item in op.evidence:
            assert not C._MEMORY_ENVELOPE.search(item.quote), item.quote[:120]
            assert not C._MEMORY_INVOCATION.search(item.quote), item.quote[:120]
        context = op.payload["context"]
        assert not C._MEMORY_ENVELOPE.search(context["observation_quote"])
        assert not C._MEMORY_INVOCATION.search(context["observation_quote"])


def test_eval_artifact_keeps_recovery_ops_out_of_the_executable_proposals():
    """The two passes must stay separable.

    Recovery ops re-derive a correction the session already made; executing one
    would create a second ``supersedes`` edge for a correction that already
    landed. They are evidence of capability, never work for the runner.
    """

    report = _artifact()
    executable = {raw["payload"]["trace_id"] for raw in report["ops"]}
    recovered = {
        item["op"]["payload"]["trace_id"]
        for item in report["ground_truth_recovery"]["corrections"]
    }
    assert not executable & recovered
    assert report["ground_truth_recovery"]["executable"] is False


def test_eval_artifact_negative_controls_all_passed():
    report = _artifact()
    controls = report["negative_controls"]
    assert [item["name"] for item in controls] == [c.name for c in C.NEGATIVE_CONTROLS]
    for item in controls:
        assert item["ops"] == 0, item["name"]
        assert item["passed"] is True
        assert item["judged"] >= 1, f"{item['name']} never reached the judge"


def test_eval_artifact_records_the_policy_it_ran_under():
    report = _artifact()
    assert report["config"] == C.DEFAULT_CONFIG.as_dict()
    assert report["detector_version"] == C.DETECTOR_VERSION
    assert report["judge"]["backend"] == "claude-cli"
    assert set(report["rejections"]) <= set(C.REASONS)
