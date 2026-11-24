"""Temporal evidence pairs through the ordinary transcript reader and miner."""

from dataclasses import replace
import json

import pytest

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
