"""Synthetic named-secret regressions through the offline extraction consumer."""

from copy import deepcopy
import json

import pytest

from living_memory.postsession.dedup import Deduper, StaticMemoryIndex
from living_memory.postsession.insights import extract_session
from living_memory.postsession.judge import (
    CassetteJudge,
    FakeJudge,
    PromptConfig,
    build_prompt,
    default_redactor,
    redact_value,
)
from living_memory.postsession.session import (
    FileMutation,
    SessionRecord,
    SourceSpan,
    ToolCall,
)


@pytest.mark.parametrize("prefix", ["", "APP_AUTH_"])
@pytest.mark.parametrize(
    "suffix", ["TOKEN", "SECRET", "PASSWORD", "PASSWD", "APIKEY", "API_KEY",
               "ACCESS_KEY", "PRIVATE_KEY"]
)
def test_named_secret_vocabulary(prefix, suffix):
    name = prefix + suffix
    for spelling in (name, name.lower()):
        assert default_redactor(spelling + "=dummy-value") == spelling + "=<redacted>"


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("value", ["dummy-head\ndummy-tail", "dummy-head\\\ndummy-tail"])
def test_quoted_assignments_allow_multiline_values(quote, value):
    text = "APP_TOKEN=" + quote + value + quote + "; status=ok"
    assert default_redactor(text) == "APP_TOKEN=<redacted>; status=ok"


@pytest.mark.parametrize(
    "text",
    [
        "TOKEN PASSWORD API_KEY are configuration names",
        "TOKEN_COUNT=12; PASSWORD_POLICY=strict; API_KEY_VERSION=2",
        '{"token_count": 12, "message": "ordinary quoted text"}',
        'NAME="dummy-ordinary-value"',
        r'{"message": "ordinary \"quoted\" text"}',
        "recall hit@5 rose to 0.41 on src/living_memory/retrieval.py",
    ],
)
def test_ordinary_text_keeps_prompt_bytes_and_hashes(text):
    payload = {"context": [{"text": text}], "count": 3}
    expected = build_prompt(text, {}, payload, config=PromptConfig(redactor=lambda s: s))
    assert default_redactor(text) == text
    assert build_prompt(text, {}, payload, config=PromptConfig()) == expected


def _record(text):
    def span(index):
        return SourceSpan("claude", "/tmp/synthetic-session.jsonl", index)

    return SessionRecord(
        session_key="claude:synthetic-redaction",
        source="claude",
        cli="claude",
        path="/tmp/synthetic-session.jsonl",
        repo="/tmp/synthetic",
        record_count=40,
        tool_calls=[
            ToolCall(0, "Bash", None, {"command": "python -m pytest -q"}, span(10),
                     ok=False, result_text=text),
            ToolCall(1, "Bash", None, {"command": "python -m pytest -q"}, span(30),
                     ok=True, result_text="12 passed in 3.20s"),
        ],
        file_mutations=[
            FileMutation(0, "src/app.py", "update", span(20), unified_diff="@@\n+ok\n")
        ],
    )


@pytest.mark.parametrize("name", ["TOKEN", "PASSWORD", "API_KEY", "LM_AUTH_TOKEN"])
@pytest.mark.parametrize(
    "assignment, cleaned",
    [
        ("NAME=dummy-value", "NAME=<redacted>"),
        ("NAME : dummy-value", "NAME : <redacted>"),
        ('NAME="dummy-quoted value"', "NAME=<redacted>"),
        ("NAME='dummy-quoted value'", "NAME=<redacted>"),
        ('{"NAME": "dummy-value", "count": 2}', '{"NAME": <redacted>, "count": 2}'),
        ("{'NAME': 'dummy-value', 'count': 2}", "{'NAME': <redacted>, 'count': 2}"),
        (r'{"NAME": "dummy-head\" dummy-tail", "count": 2}',
         '{"NAME": <redacted>, "count": 2}'),
        (r"NAME='dummy-head\' dummy-tail'", "NAME=<redacted>"),
        (r'{"NAME": "dummy-value\\", "count": 2}',
         '{"NAME": <redacted>, "count": 2}'),
        ('{"NAME": "", "count": 2}', '{"NAME": <redacted>, "count": 2}'),
    ],
)
def test_extraction_redacts_named_assignments_before_prompt_and_cassette(
    tmp_path, name, assignment, cleaned
):
    assignment = assignment.replace("NAME", name)
    cleaned = cleaned.replace("NAME", name)
    ordinary = ' TOKEN_COUNT=12; {"message": "ordinary text"}'
    raw = "ValueError: bad configuration in src/app.py: " + assignment + ordinary
    record = _record(raw)
    original = deepcopy(record)
    inner = FakeJudge([{"verdict": "none", "reason": "synthetic response"}])
    path = tmp_path / "cassette.json"
    recorder = CassetteJudge(path, inner=inner, record=True)
    result = extract_session(record, judge=recorder, deduper=Deduper(StaticMemoryIndex([])))

    assert result.judge_calls == len(inner.calls) == 1
    call = inner.calls[0]
    # A selected error line must actually reach the boundary for this to prove redaction.
    payload = call["payload"]
    expected_payload = redact_value(payload, lambda s: s.replace(assignment, cleaned))
    assert payload != expected_payload
    expected = build_prompt(
        call["task"], call["schema"], expected_payload,
        config=PromptConfig(redactor=lambda s: s),
    )
    rendered_payload = json.loads(
        call["prompt"].split("# Payload\n", 1)[1].split("\n\nReply", 1)[0]
    )
    assert any(ordinary in item["quote"] for item in rendered_payload["evidence"])
    assert call["prompt"] == expected.user_prompt
    assert call["prompt_sha256"] == expected.prompt_sha256
    assert "dummy-" not in call["prompt"]
    assert record == original

    body = path.read_text(encoding="utf-8")
    assert "dummy-" not in body
    entry = next(iter(json.loads(body)["entries"].values()))
    assert entry["prompt"] == expected.user_prompt
    assert entry["prompt_sha256"] == expected.prompt_sha256
    replay = CassetteJudge(path)
    replayed = extract_session(record, judge=replay, deduper=Deduper(StaticMemoryIndex([])))
    assert replayed.judge_calls == 1
    assert replay.stats[0].replayed


def test_named_assignment_redaction_precedes_bounding_without_mutating_input():
    secret = 'dummy-head\\"' + " dummy-tail" * 200
    raw = '{"API_KEY": "' + secret + '"}'
    payload = {"context": [raw, "ordinary " * 400]}
    original = deepcopy(payload)
    config = PromptConfig(max_payload_bytes=300, min_field_chars=20)
    expected = build_prompt(
        "inspect", {}, {"context": ['{"API_KEY": <redacted>}', payload["context"][1]]},
        config=PromptConfig(redactor=lambda s: s, max_payload_bytes=300, min_field_chars=20),
    )
    built = build_prompt("inspect", {}, payload, config=config)
    assert built == expected
    assert built.payload_truncated
    assert "dummy-" not in built.user_prompt
    assert payload == original


def test_custom_redactor_replaces_named_assignment_policy():
    text = '{"TOKEN": "dummy-value", "status": "old"}'
    payload = {"context": [text]}
    config = PromptConfig(redactor=lambda s: s.replace("old", "new"))
    expected = build_prompt(
        text.replace("old", "new"), {}, {"context": [text.replace("old", "new")]},
        config=PromptConfig(redactor=lambda s: s),
    )
    assert build_prompt(text, {}, payload, config=config) == expected
    assert payload == {"context": [text]}
