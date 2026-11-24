"""Whole-value evidence through the transcript reader, miner and insight extractor."""

import json

import pytest

from living_memory.postsession.insights import extract_session
from living_memory.postsession.judge import FakeJudge
from living_memory.postsession.mining import MiningConfig, mine
from living_memory.postsession.transcripts import TranscriptRef, load_transcript


def transcript(tmp_path, output, decisions, *, diff=False):
    def message(role, block, **fields):
        return {"type": role, "message": {"role": role, "content": [block]}, **fields}

    rows = [
        message("assistant", {"type": "tool_use", "id": "count", "name": "Bash",
                              "input": {"command": "wc -l demo.txt"}}),
        message("user", {"type": "tool_result", "tool_use_id": "count", "content": output}),
    ]
    if diff:
        rows.extend([
            message("assistant", {"type": "tool_use", "id": "edit", "name": "Edit",
                                  "input": {"file_path": "limit.py"}}),
            message("user", {"type": "tool_result", "tool_use_id": "edit", "content": "ok"},
                    toolUseResult={"filePath": "limit.py", "type": "update",
                                   "structuredPatch": [{"lines": ["-limit = 374", *decisions]}]}),
        ])
    else:
        rows.extend(message("assistant", {"type": "text", "text": text}) for text in decisions)
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return load_transcript(TranscriptRef("claude", "claude", path, "synthetic"))


@pytest.mark.parametrize("diff", [False, True])
@pytest.mark.parametrize("other", ["13740", "3740", "374.25", "0,374", "374 ms", "374%"])
def test_other_value_never_reaches_judge(tmp_path, diff, other):
    decision = f"limit = {other}"
    record = transcript(tmp_path, "374 demo.txt", ["+" + decision if diff else decision], diff=diff)
    assert mine(record) == []
    judge = FakeJudge()
    assert extract_session(record, judge=judge).candidates == 0
    assert not judge.calls

    candidate, = mine(record, config=MiningConfig(require_number_reuse=False))
    assert candidate.signals["reused_number"] == ""
    assert [item.quote for item in candidate.evidence] == ["$ wc -l demo.txt", "374 demo.txt"]


@pytest.mark.parametrize("value", ["374", "3.74", "3,74 ms", "374µs", "374 req/s", "374px", "374%", "0.25 MB"])
@pytest.mark.parametrize("diff", [False, True])
def test_first_whole_value_supplies_verbatim_evidence(tmp_path, value, diff):
    false = f"old = 1{value}"
    exact = f"limit = {value}."
    # A distant false occurrence in the same turn must not anchor the quote.
    decisions = ["+" + false, "+" + exact] if diff else [false + " " * 500 + exact]
    record = transcript(tmp_path, value + " demo.txt", decisions, diff=diff)
    candidate, = mine(record)
    assert candidate.signals["reused_number"] == value
    evidence = candidate.evidence[-1]
    assert evidence.quote == exact
    source = record.span(3, "toolUseResult") if diff else record.span(2, "text")
    assert evidence.locator == source.locator()
    judge = FakeJudge([{"verdict": "none", "reason": "synthetic"}])
    assert extract_session(record, judge=judge).judge_calls == 1
    assert judge.calls[0]["payload"]["evidence"][-1] == {"quote": exact, "locator": evidence.locator}


def test_output_order_then_first_eligible_decision(tmp_path):
    record = transcript(tmp_path, "374 rows, 812 bytes", ["812", "13740", "374", "374"])
    candidate, = mine(record)
    assert candidate.signals["reused_number"] == "374"
    assert candidate.evidence[-1].locator == record.span(4, "text").locator()
    candidate, = mine(record, config=MiningConfig(lookahead_records=2))
    assert candidate.signals["reused_number"] == "812"
    assert candidate.evidence[-1].locator == record.span(2, "text").locator()
