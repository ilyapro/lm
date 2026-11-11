"""Top-level transcript selection against an independent JSON oracle."""

from __future__ import annotations

import hashlib
from itertools import permutations
import json
from pathlib import Path
import tracemalloc

import pytest

from living_memory.postsession import corpus, transcripts
from living_memory.postsession.insights import scope_for, source_transport_id
from living_memory.postsession.session import SourceSpan


CHUNK_SIZES = (1, 2, 4, 7, 64, 65536)
METADATA = {
    "id": "real-session",
    "model": "real-model",
    "workspace": "/synthetic/real",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T01:00:00Z",
}
NESTED_METADATA = {
    "id": "nested-object", "model": "nested-model", "workspace": "/synthetic/nested",
}
TEXT = 'Quotes: \\"; braces: {}[]; keys: , "metadata": {}, "messages": []; \n☃'


def document():
    decoy = {"role": "user", "content": "nested message"}
    return {
        "metadata": {**METADATA, "messages": [decoy]},
        "messages": [
            None,
            {"role": "user", "content": TEXT},
            {"role": "assistant", "content": [{
                "type": "tool_use", "id": "real-call", "name": "inspect",
                "input": {"metadata": NESTED_METADATA, "messages": [decoy]},
            }]},
        ],
        "extra": [{"metadata": NESTED_METADATA, "messages": [decoy]},
                  [None, False, True, -123456789, 1.25e20, TEXT]],
        "description": TEXT,
    }


@pytest.fixture(params=CHUNK_SIZES, ids=lambda size: f"chunk-{size}")
def scanner_class(request, monkeypatch):
    class Scanner(transcripts._JsonMemberScanner):
        def __init__(self, path):
            super().__init__(path, chunk_size=request.param)

    monkeypatch.setattr(transcripts, "_JsonMemberScanner", Scanner)
    monkeypatch.setattr(corpus, "_JsonMemberScanner", Scanner)
    return Scanner


def consumers(path: Path):
    ref = next(transcripts.DeepseekAdapter().discover(
        transcripts.Roots(deepseek_sessions=path.parent),
    ))
    return transcripts.load_transcript(ref), corpus.probe(ref)


@pytest.mark.parametrize("order", list(permutations(document())))
@pytest.mark.parametrize("escaped_keys", [False, True])
def test_consumers_select_top_level_members(tmp_path, scanner_class, order, escaped_keys):
    original = document()
    body = json.dumps({key: original[key] for key in order}, ensure_ascii=False, indent=2)
    if escaped_keys:
        body = body.replace('"metadata":', '"meta\\u0064ata":')
        body = body.replace('"messages":', '"mess\\u0061ges":')
    oracle = json.loads(body)
    path = tmp_path / "file-session.json"
    path.write_text(body, encoding="utf-8")

    with scanner_class(path) as scanner:
        assert scanner.value("metadata") == oracle["metadata"]
    with scanner_class(path) as scanner:
        assert list(scanner.array("messages")) == oracle["messages"]
    record, entry = consumers(path)
    metadata = oracle["metadata"]
    assert record.cli_session_id == source_transport_id(record) == metadata["id"]
    assert record.model == entry.model == metadata["model"]
    assert record.cwd == entry.cwd == metadata["workspace"]
    assert record.started_at == entry.started_at == metadata["created_at"]
    assert record.ended_at == entry.ended_at == metadata["updated_at"]
    assert record.repo == entry.repo == metadata["workspace"]
    assert scope_for(record) == "project:real"
    assert record.session_key == entry.session_key == "deepseek:file-session"
    assert entry.cli_session_id == "file-session"
    assert record.linked_session_ids == entry.linked_session_ids == ("file-session",)
    assert record.transcript_sha256 == entry.sha256 == hashlib.sha256(body.encode()).hexdigest()
    assert record.record_count == len(oracle["messages"])
    assert len(record.turns) == len(record.tool_calls) == 1
    assert record.turns[0].text == oracle["messages"][1]["content"]
    assert record.turns[0].span == SourceSpan("deepseek", str(path), 1)
    call = record.tool_calls[0]
    expected = oracle["messages"][2]["content"][0]
    assert call.call_id == expected["id"]
    assert call.arguments == expected["input"]
    assert call.span == SourceSpan("deepseek", str(path), 2, "messages.content.tool_use")


@pytest.mark.parametrize(("body", "metadata", "messages"), [
    ("", None, []),
    (" \n\t", None, []),
    ("{}", None, []),
    ('{"metadata": null, "messages": null}', None, []),
    ('{"metadata": [], "messages": {}}', [], []),
    ('{"other": {"metadata": {"id": "nested"}, "messages": [{}]}}', None, []),
    ('[{"metadata": {"id": "nested"}, "messages": [{}]}]', None, []),
    (json.dumps('{"metadata": {}, "messages": [{}]}'), None, []),
    ('{} {"metadata": {}, "messages": [{}]}', None, []),
    ('{"metadata"', None, []),
    ('{"metadata" ', None, []),
    ('{"metadata" ! {}', None, []),
    ('{"metadata":', None, []),
    ('{"metadata": {"id": "unfinished', None, []),
    ('{"other": [{"metadata": {"id": "nested"}}', None, []),
    ('{"other": "unfinished \\"metadata\\": {}', None, []),
    ('{"metadata": {"id": "complete"}', {"id": "complete"}, []),
    ('{"messages": [', None, []),
    ('{"messages": [{"role": "user", "content": "complete"}, {"role":',
     None, [{"role": "user", "content": "complete"}]),
    ('{"metadata": {"id": "complete"}, "messages": [{"role": "user", "content": "ok"}',
     {"id": "complete"}, [{"role": "user", "content": "ok"}]),
])
def test_missing_and_partial_members(tmp_path, scanner_class, body, metadata, messages):
    path = tmp_path / "file-session.json"
    path.write_text(body, encoding="utf-8")
    with scanner_class(path) as scanner:
        assert scanner.value("metadata") == metadata
    with scanner_class(path) as scanner:
        assert list(scanner.array("messages")) == messages
    record, entry = consumers(path)
    assert record.cli_session_id == source_transport_id(record) == (
        metadata.get("id", "file-session") if isinstance(metadata, dict) else "file-session"
    )
    assert record.model is entry.model is None
    assert record.cwd is entry.cwd is None
    assert scope_for(record) == "global"
    assert record.record_count == len(messages)
    assert [turn.text for turn in record.turns] == [item["content"] for item in messages]


@pytest.mark.parametrize("member", ["metadata", "messages", "absent"])
def test_skipped_values_use_bounded_memory(tmp_path, member):
    path = tmp_path / "large.json"
    # The foreign string, array and key are each much larger than a read chunk.
    fragment = json.dumps(TEXT)[1:-1] * 1024
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{"other": [{"text": "')
        for _ in range(32):
            handle.write(fragment)
        handle.write('"}, [')
        for _ in range(32768):
            handle.write('{"metadata": {}, "messages": []},')
        handle.write('null]], "')
        for _ in range(32):
            handle.write("k" * 65536)
        handle.write('\": null, "metadata": {"id": "real"}, "messages": [{"role": "user"}]}')
    chunk_size = 65536
    tracemalloc.start()
    try:
        with transcripts._JsonMemberScanner(path, chunk_size=chunk_size) as scanner:
            if member == "messages":
                assert list(scanner.array(member)) == [{"role": "user"}]
            else:
                assert scanner.value(member) == ({"id": "real"} if member == "metadata" else None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Allow decoder and text I/O overhead, but never buffer a whole foreign value.
    assert peak < 16 * chunk_size
