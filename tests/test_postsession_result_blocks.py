"""Native output locators select the producing block within the source record."""

import json

import pytest

from living_memory.postsession.corrections import observations
from living_memory.postsession.session import SourceSpan
from living_memory.postsession.transcripts import TranscriptRef, load_transcript


@pytest.mark.parametrize("source", ["claude", "gigacode", "deepseek"])
def test_sibling_results_in_source_order(tmp_path, source):
    calls = []
    results = []
    for key, name, text in [("one", "Write", "first invocation"), ("two", "Edit", "second invocation")]:
        if source == "gigacode":
            calls.append({"functionCall": {"name": name, "id": key, "args": {}}})
            results.append({"functionResponse": {"id": key, "response": {"output": text}}})
        else:
            tool = {"Write": "write_file", "Edit": "edit_file"}[name] if source == "deepseek" else name
            calls.append({"type": "tool_use", "id": key, "name": tool, "input": {}})
            results.append({"type": "tool_result", "tool_use_id": key, "content": text})

    blocks = [results[1], {}, results[0]]
    path = tmp_path / "transcript"
    if source == "deepseek":
        rows = [{"role": "assistant", "content": calls}, {"role": "user", "content": blocks}]
        path.write_text(json.dumps({"messages": rows}, indent=2))
    else:
        field = "parts" if source == "gigacode" else "content"
        rows = [{"type": role, "message": {"role": role, field: parts}}
                for role, parts in [("assistant", calls), ("user", blocks)]]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    record = load_transcript(TranscriptRef(source, source, path, "synthetic"))
    outputs = observations(record)
    assert [(obs.record_index, obs.ordinal) for obs in outputs] == [(1, 1), (1, 0)]
    for obs, index in zip(outputs, (0, 2), strict=True):
        field = {
            "claude": f"message.content.{index}.content",
            "deepseek": f"content.{index}.content",
            "gigacode": f"message.parts.{index}.functionResponse.response",
        }[source]
        assert SourceSpan.parse(obs.locator) == SourceSpan(source, str(path), 1, field)
        value = rows[1]
        for part in field.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
        assert obs.text == (json.dumps(value) if source == "gigacode" else value)
