"""Temporal evidence pairs through the ordinary transcript reader and miner."""

from dataclasses import replace
import json

import pytest

from living_memory.postsession.corrections import observations
from living_memory.postsession.mining import MiningConfig, mine
from living_memory.postsession.session import SessionRecord, SourceSpan
from living_memory.postsession.transcripts import TranscriptRef, load_transcript


def transcript(tmp_path, command, claim, output, order):
    def message(role, block):
        return {"type": role, "message": {"role": role, "content": [block]}}

    rows = {
        "call": message("assistant", {"type": "tool_use", "id": "probe", "name": "Bash",
                                      "input": {"command": command}}),
        "claim": message("assistant", {"type": "text", "text": claim}),
        "result": message("user", {"type": "tool_result", "tool_use_id": "probe",
                                   "content": output}),
    }
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(rows[key]) + "\n" for key in order.split()))
    return load_transcript(TranscriptRef("claude", "claude", path, "synthetic"))


MEASURE = ("wc -l demo.txt", "I expect 374 rows in the table.", "374 demo.txt")
REFUTE = ("cat /demo/config.py", "I expect /demo/config.py to exist.",
          "Actually /demo/config.py is missing.")


@pytest.mark.parametrize(("example", "order", "kind", "indices"), [
    (MEASURE, "call claim result", None, []),
    (MEASURE, "call result claim", "measurement", [0, 1, 2]),
    (MEASURE, "claim call result", None, []),
    (REFUTE, "call claim result", "refuted_hypothesis", [1, 2]),
    (REFUTE, "call result claim", None, []),
    (REFUTE, "claim call result", "refuted_hypothesis", [0, 2]),
])
def test_transcript_permutations(tmp_path, example, order, kind, indices):
    record = transcript(tmp_path, *example, order)
    candidates = mine(record)
    assert [c.kind for c in candidates] == ([kind] if kind else [])
    if kind:
        evidence = candidates[0].evidence
        spans = [SourceSpan.parse(e.locator) for e in evidence]
        assert [s.record_index for s in spans] == indices
        result_index = order.split().index("result")
        assert spans[indices.index(result_index)] == record.span(result_index, "message.content.0.content")
        assert evidence[indices.index(result_index)].quote == example[2]


@pytest.mark.parametrize(("example", "order", "kind"), [
    (MEASURE, "call result claim", "measurement"),
    (REFUTE, "claim call result", "refuted_hypothesis"),
])
def test_window_measured_from_observation(tmp_path, example, order, kind):
    record = transcript(tmp_path, *example, order)
    candidates = mine(record, config=MiningConfig(lookahead_records=1, refutation_lookahead=1))
    assert [c.kind for c in candidates] == ([kind] if example == MEASURE else [])


@pytest.mark.parametrize(("example", "order"), [
    (MEASURE, "call result claim"), (REFUTE, "call claim result"),
])
@pytest.mark.parametrize("boundary", ["same record", "overlap", "other path", "other source"])
def test_pairs_need_ordered_spans_in_one_transcript(tmp_path, example, order, boundary):
    record = transcript(tmp_path, *example, order)
    turn, = record.turns
    result = record.tool_calls[0].result_span
    if boundary == "same record":
        turn.span = replace(turn.span, record_index=result.record_index)
    elif boundary == "overlap":
        if example == MEASURE:
            record.tool_calls[0].result_span = replace(result, end_record_index=turn.span.record_index)
        else:
            turn.span = replace(turn.span, end_record_index=result.record_index)
    elif boundary == "other path":
        turn.span = replace(turn.span, path="other.jsonl")
    else:
        turn.span = replace(turn.span, source="other")
    assert mine(record) == []


@pytest.mark.parametrize(("example", "order", "kind"), [
    (MEASURE, "call result claim", "measurement"),
    (REFUTE, "claim call result", "refuted_hypothesis"),
])
def test_legacy_record_without_result_locator(tmp_path, example, order, kind):
    data = transcript(tmp_path, *example, order).to_dict()
    del data["tool_calls"][0]["result_span"]
    candidate, = mine(SessionRecord.from_dict(data))
    assert candidate.kind == kind
    result = next(e for e in candidate.evidence if e.quote == example[2])
    assert SourceSpan.parse(result.locator).record_index == order.split().index("call")


def mixed_transcript(tmp_path, content, **metadata):
    def call(key, name):
        return {"type": "tool_use", "name": name, "id": key, "input": {}}

    rows = [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "The table uses the old layout."},
            call("probe", "Bash"), call("lookup", "mcp__living-memory__memory_recall"),
        ]}},
        {"type": "user", "message": {"role": "user", "content": content}, **metadata},
        {"type": "assistant", "message": {"content": [call("followup", "Edit")]}},
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return load_transcript(TranscriptRef("claude", "claude", path, "synthetic"))


RESULT = {"type": "tool_result", "tool_use_id": "probe", "content": "No, output from Bash."}
FIX = {"type": "text", "text": "No, use the new layout."}


def test_mixed_claude_keeps_result_and_human_coordinates(tmp_path):
    memory = {"type": "tool_result", "tool_use_id": "lookup", "content": [
        {"type": "text", "text": json.dumps({"recall_event_id": "demo", "results": []})},
    ]}
    reminder = {"type": "text", "text": "<system-reminder>No, host instruction.</system-reminder>"}
    extra = {"type": "text", "text": "Keep the heading."}
    record = mixed_transcript(tmp_path, [memory, RESULT, reminder, FIX, extra], origin={"kind": "human"})
    assert [(r.recall_event_id, r.turn_index) for r in record.recalls] == [("demo", 1)]
    expected = [
        ("command_output", RESULT["content"], "message.content.1.content"),
        ("operator_message", FIX["text"], "message.content.3.text"),
        ("operator_message", extra["text"], "message.content.4.text"),
    ]
    observed = observations(record)
    assert [(o.kind, o.text, o.locator) for o in observed] == [
        (kind, text, record.span(1, field).locator()) for kind, text, field in expected
    ]
    raw = json.loads((tmp_path / "session.jsonl").read_text().splitlines()[1])
    for item in observed:
        value = raw
        for part in SourceSpan.parse(item.locator).field.split("."):
            value = value[int(part) if part.isdecimal() else part]
        assert item.text == value
    candidate, = [c for c in mine(record) if c.kind == "operator_correction"]
    assert candidate.span.locator() == observed[1].locator


@pytest.mark.parametrize(("content", "metadata"), [
    ([RESULT, FIX], {}),
    ([RESULT, FIX], {"origin": {"kind": "peer"}}),
    ([RESULT, FIX], {"origin": {"kind": "human"}, "isMeta": True}),
    ([RESULT, FIX], {"origin": {"kind": "human"}, "sourceToolAssistantUUID": "tool-owner"}),
    ([FIX, RESULT, FIX], {"origin": {"kind": "human"}}),
    ([{**RESULT, "content": [FIX]}], {"origin": {"kind": "human"}}),
    ([RESULT, {"type": "text", "text": "Note: <system-reminder>No, change it.</system-reminder>"}],
     {"origin": {"kind": "human"}}),
    ([RESULT, {**FIX, "origin": {"kind": "peer"}}], {"origin": {"kind": "human"}}),
])
def test_mixed_claude_excludes_unattributed_text(tmp_path, content, metadata):
    record = mixed_transcript(tmp_path, content, **metadata)
    assert not any(o.kind == "operator_message" for o in observations(record))
    assert not any(c.kind == "operator_correction" for c in mine(record))
    assert record.tool_calls[0].ok


def test_mixed_permission_result_binds_only_the_recorded_feedback(tmp_path):
    denied = {**RESULT, "is_error": True}
    record = mixed_transcript(tmp_path, [denied, FIX, {"type": "text", "text": "No, host note."}],
                              sourceToolAssistantUUID="tool-owner", userFeedback=FIX["text"])
    assert not record.tool_calls[0].ok
    assert [(o.kind, o.text) for o in observations(record)] == [
        ("command_output", denied["content"]), ("operator_message", FIX["text"]),
    ]
