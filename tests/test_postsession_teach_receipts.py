"""Reconstruct synthetic Codex MCP calls and exercise the offline consumers."""

import json

import pytest

from living_memory.consolidation import TeachResult
from living_memory.models import Connection, Node
from living_memory.server import _teach_result_to_dict
from living_memory.postsession.corrections import candidates, taught_delivered_nodes
from living_memory.postsession.transcripts import TranscriptRef, load_transcript


ORIGINAL = "old-fact"
CORRECTION = "new-fact"
EDGE = {"type": "supersedes", "id": "edge-id"}
CLAIM = "scripts/build_cache.py --build refuses to overwrite the existing cache_index.json."
OBSERVED = "scripts/build_cache.py --build overwrote the existing cache_index.json in place."

RECEIPT = _teach_result_to_dict(TeachResult(
    corrective_trace=Node(CORRECTION, "trace", OBSERVED),
    supersedes=Connection(EDGE["id"], CORRECTION, ORIGINAL, "supersedes"),
    original=Node(ORIGINAL, "trace", CLAIM),
))


def mcp_end(tool, arguments, result):
    return {
        "type": "event_msg",
        "payload": {
            "type": "mcp_tool_call_end",
            "call_id": tool,
            "invocation": {"server": "living-memory", "tool": tool, "arguments": arguments},
            "result": result,
        },
    }


def ok(payload):
    return {"Ok": {"structuredContent": payload}}


@pytest.mark.parametrize(("result", "outcome"), [
    pytest.param(ok(RECEIPT), "confirmed", id="compact"),
    pytest.param({"Ok": {"content": [{"type": "text", "text": json.dumps({
        **RECEIPT,
        "corrective_trace": {"id": CORRECTION, "content": OBSERVED},
        "supersedes": {**EDGE, "source_id": CORRECTION, "target_id": ORIGINAL},
    })}]}}, "confirmed", id="verbose-text"),
    pytest.param(ok({"node": {"id": CORRECTION}, "supersedes": ORIGINAL}), "confirmed", id="legacy"),
    pytest.param(ok({"error": "write refused"}), "failed", id="server-error"),
    pytest.param(ok({**RECEIPT, "error": "write refused"}), "failed", id="error-with-ids"),
    pytest.param({"Ok": {"isError": True, "structuredContent": RECEIPT}}, "failed", id="mcp-error"),
    pytest.param({"Err": "transport failure"}, "failed", id="provider-error"),
    pytest.param(None, "unknown", id="missing-result"),
    pytest.param({"Ok": {"content": [{"type": "text", "text": "truncated"}]}}, "unknown", id="text"),
    pytest.param(ok({"node": {"id": CORRECTION}}), "unknown", id="remember-shaped"),
    pytest.param(ok({"node": {"id": CORRECTION}, "supersedes": "other"}), "unknown", id="legacy-mismatch"),
    pytest.param(ok({**RECEIPT, "original": {"id": "other"}}), "unknown", id="wrong-original"),
    pytest.param(ok({**RECEIPT, "corrective_trace": {"id": ORIGINAL}}), "unknown", id="self-edge"),
    pytest.param(ok({**RECEIPT, "corrective_trace": {}}), "unknown", id="missing-correction"),
    pytest.param(ok({**RECEIPT, "supersedes": {**EDGE, "source_id": ORIGINAL}}), "unknown", id="wrong-source"),
    pytest.param(ok({**RECEIPT, "supersedes": {**EDGE, "target_id": CORRECTION}}), "unknown", id="wrong-target"),
    pytest.param(ok({**RECEIPT, "supersedes": {**EDGE, "type": "related"}}), "unknown", id="wrong-relation"),
])
def test_teach_receipt_consumers(tmp_path, result, outcome):
    rows = [
        mcp_end("memory_recall", {"query": "build cache"}, ok({
            "results": [{"id": ORIGINAL, "content": CLAIM}],
        })),
        {"type": "event_msg", "payload": {"type": "user_message", "message": OBSERVED}},
        mcp_end("memory_teach", {"trace_id": ORIGINAL, "correction": OBSERVED}, result),
    ]
    path = tmp_path / "rollout-synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    record = load_transcript(TranscriptRef("codex", "codex", path, "codex:synthetic"))
    (attempt,) = record.writes
    assert attempt.teach_outcome == outcome
    assert attempt.content == OBSERVED
    assert record.tool_calls[-1].arguments["trace_id"] == ORIGINAL

    confirmed = outcome == "confirmed"
    assert attempt.supersedes == (ORIGINAL if confirmed else None)
    assert taught_delivered_nodes(record) == ((ORIGINAL,) if confirmed else ())
    assert record.written_node_ids == ((CORRECTION,) if confirmed else ())
    counters = {}
    assert [c.node.node_id for c in candidates(record, counters=counters)] == ([] if confirmed else [ORIGINAL])
    assert counters.get("already_taught_in_session", 0) == int(confirmed)


def test_claude_pairs_teach_receipts_by_tool_use_id(tmp_path):
    def call(call_id, trace_id):
        return {"type": "tool_use", "id": call_id, "name": "mcp__living-memory__memory_teach",
                "input": {"trace_id": trace_id, "correction": OBSERVED}}

    def result(call_id, *, error=False):
        return {"type": "tool_result", "tool_use_id": call_id, "is_error": error,
                "content": [{"type": "text", "text": json.dumps(RECEIPT)}]}

    rows = [
        {"type": "assistant", "message": {"content": [
            call("a", ORIGINAL), call("b", "another-fact"), call("c", ORIGINAL), call("d", ORIGINAL),
        ]}},
        {"type": "user", "message": {"content": [result("c", error=True), result("b"), result("a")]}},
    ]
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    record = load_transcript(TranscriptRef("claude", "claude", path, "claude:synthetic"))
    assert [call.call_id for call in record.tool_calls] == ["a", "b", "c", "d"]
    assert [w.teach_outcome for w in record.writes] == ["failed", "unknown", "confirmed"]
    assert [w.supersedes for w in record.writes] == [None, None, ORIGINAL]
    assert record.written_node_ids == (CORRECTION,)
