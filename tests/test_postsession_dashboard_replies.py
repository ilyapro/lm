"""Synthetic AE writer envelopes through the normal transcript consumers."""

import json

import pytest

from living_memory.postsession.corrections import observations
from living_memory.postsession.corpus import assign_splits, probe
from living_memory.postsession.mining import mine
from living_memory.postsession.session import SourceSpan
from living_memory.postsession.transcripts import ADAPTERS, Roots, TranscriptRef, load_transcript


def dashboard(tmp_path, events):
    root = tmp_path / "ae"
    chat = root / "projects/demo/state/chats/chat"
    chat.mkdir(parents=True)
    path = chat / "transcript.jsonl"
    chat.joinpath("meta.json").write_text(json.dumps({"sessions_by_provider": {"codex": "native"}}))
    with path.open("w") as stream:
        for seq, event in enumerate(events, 1):
            if event is None:
                stream.write("{\n")
                continue
            envelope = {"v": 1, "chat_id": "chat", "seq": seq, "turn": 1, "ts": 1000 * seq}
            stream.write(json.dumps(envelope | event) + "\n")
    ref, = ADAPTERS["ae_chat"].discover(Roots.discover(home=tmp_path, ae_root=root))
    return ref


def test_dashboard_correction_and_native_twin(tmp_path):
    claim = "I will remove example.py."
    correction = "No, keep example.py."
    ref = dashboard(tmp_path, [
        {"type": "turn_start", "provider": "codex"},
        {"type": "text_delta", "text": "I will remove "},
        {"type": "text_delta", "text": "example.py."},
        {"type": "done", "subtype": "turn.completed"},
        {"type": "user_message", "turn": 2, "origin": "operator", "text": correction},
        {"type": "turn_start", "turn": 2, "provider": "claude"},
        {"type": "tool_use", "turn": 2, "name": "exec_command", "tool_id": "act",
         "input_summary": '{"cmd":"ls example.py"}'},
    ])
    native = TranscriptRef("codex", "codex", tmp_path / "native.jsonl", "codex:native",
                           cli_session_id="native")
    native.path.write_text("\n".join(json.dumps(row) for row in [
        {"type": "event_msg", "payload": {"type": "agent_message", "message": claim}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": correction}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                                               "call_id": "act", "arguments": '{"cmd":"ls example.py"}'}},
    ]))

    record = load_transcript(ref)
    assert [(t.index, t.role, t.cli, t.text) for t in record.turns] == [
        (0, "assistant", "codex", claim), (1, "user", "claude", correction),
    ]
    assert not record.turns[0].truncated
    for source in (record, load_transcript(native)):
        candidate, = [c for c in mine(source) if c.kind == "operator_correction"]
        assert [e.quote for e in candidate.evidence] == [claim, correction]
        observation, = observations(source)
        assert (observation.kind, observation.text) == ("operator_message", correction)

    rows = [json.loads(line) for line in ref.path.read_text().splitlines()]
    candidate, = mine(record)
    for evidence, bounds in zip(candidate.evidence, [(1, 2), (4, None)], strict=True):
        span = SourceSpan.parse(evidence.locator)
        assert span == SourceSpan("ae_chat", str(ref.path), bounds[0], "text", bounds[1])
        end = span.record_index if span.end_record_index is None else span.end_record_index
        assert evidence.quote == "".join(row[span.field] for row in rows[span.record_index:end + 1])
    assert observations(record)[0].record_index == 4
    assert record.linked_session_ids == ("native",)
    wrapper, provider = assign_splits([probe(ref), probe(native)])
    assert wrapper.split_key == provider.split_key
    assert wrapper.group_size == provider.group_size == 2


@pytest.mark.parametrize("boundary", [
    {"type": "done"}, {"type": "turn_start", "provider": "codex"},
    {"type": "tool_use", "name": "exec_command", "tool_id": "one", "input_summary": "tool input"},
    {"type": "tool_result", "tool_id": "one", "output_summary": "tool output"},
    {"type": "thinking", "text": "reasoning"}, {"type": "raw", "text": "provider event"},
    {"type": "camera_attachment", "text": "camera"}, {"type": "text_delta", "text": {}}, None,
])
def test_reply_boundaries_and_unfinished_tail(tmp_path, boundary):
    ref = dashboard(tmp_path, [
        {"type": "text_delta", "text": "first"}, boundary,
        {"type": "text_delta", "text": "second"},
    ])
    record = load_transcript(ref)
    assert [(t.role, t.text, t.span.record_index) for t in record.turns] == [
        ("assistant", "first", 0), ("assistant", "second", 2),
    ]
    assert record.turns[-1].truncated
    assert observations(record) == []
    assert mine(record) == []


def test_adjacent_events_from_different_messages_and_missing_events(tmp_path):
    ref = dashboard(tmp_path, [
        {"type": "text_delta", "message_id": "a", "text": "one"},
        {"type": "text_delta", "message_id": "b", "text": "two"},
        {"type": "text_delta", "message_id": "b", "turn": 2, "text": "three"},
        {"type": "text_delta", "message_id": "b", "turn": 2, "seq": 5, "text": "four"},
        {"type": "text_delta", "agent_id": "worker", "seq": 6, "text": "five"},
        {"type": "user_message", "origin": "api", "seq": 7, "text": "No, keep api.py."},
        {"type": "tool_use", "name": "exec_command", "seq": 8, "tool_id": "act"},
    ])
    record = load_transcript(ref)
    assert [(t.role, t.text) for t in record.turns] == [
        ("assistant", "one"), ("assistant", "two"), ("assistant", "three"),
        ("assistant", "four"), ("subagent", "five"), ("api", "No, keep api.py."),
    ]
    assert record.turns[2].truncated and record.turns[3].truncated
    assert mine(record) == observations(record) == []
