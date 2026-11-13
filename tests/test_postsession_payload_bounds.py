"""Offline contracts for deterministic JSON payload bounding and omission markers."""

from __future__ import annotations

import json

import pytest

from living_memory.postsession.judge import (
    JudgePayloadTooLarge,
    PromptConfig,
    bound_payload,
    build_prompt,
    canonical_json,
    sha256_text,
)

SCHEMA = {"type": "object"}


@pytest.fixture
def payload_renders(monkeypatch):
    rendered = []

    def counted(value, **kwargs):
        text = canonical_json(value, **kwargs)
        rendered.append(text)
        return text

    monkeypatch.setattr("living_memory.postsession.judge.canonical_json", counted)
    return rendered


def test_string_floor_allows_list_reduction(payload_renders):
    payload = {"summary": "x" * 201, "observations": ["a" * 100] * 40}
    before = canonical_json(payload)
    bounded, truncated = bound_payload(payload, max_bytes=2000)
    assert truncated
    assert bounded["summary"] == "x" * 200 + "… [truncated, 1 chars omitted]"
    retained = bounded["observations"][:-1]
    assert retained == payload["observations"][:len(retained)]
    assert 0 < len(retained) < 40
    assert bounded["observations"][-1] == (
        f"… [truncated, {40 - len(retained)} items omitted]"
    )
    assert len(canonical_json(bounded).encode("utf-8")) <= 2000
    # At most one character and 39 list items can be removed.
    assert len(payload_renders) <= 1 + 1 + 39
    assert canonical_json(payload) == before
    reordered = dict(reversed(list(payload.items())))
    assert bound_payload(reordered, max_bytes=2000) == (bounded, True)


def test_list_floor_allows_another_list_to_shrink(payload_renders):
    payload = {"a": ["a" * 200, "b" * 200], "z": ["x" * 200, "y" * 200]}
    bounded, truncated = bound_payload(payload, max_bytes=550)
    assert truncated
    assert bounded == {
        "a": ["a" * 200, "… [truncated, 1 items omitted]"],
        "z": ["x" * 200, "… [truncated, 1 items omitted]"],
    }
    assert len(canonical_json(bounded).encode("utf-8")) <= 550
    assert len(payload_renders) <= 1 + 2


@pytest.mark.parametrize("floor", [0, 1, 200])
def test_impossible_budget_exhausts_source_reductions(payload_renders, floor):
    payload = {"summary": "x" * (floor + 1), "observations": ["a", "b"]}
    before = canonical_json(payload)
    with pytest.raises(JudgePayloadTooLarge, match="cannot be shrunk"):
        bound_payload(payload, max_bytes=1, min_field_chars=floor)
    removable_chars = 1 + 2 * max(0, 1 - floor)
    removable_items = 1
    assert len(payload_renders) <= 1 + removable_chars + removable_items
    assert canonical_json(payload) == before


def test_repeated_string_cuts_count_source_characters_and_json_bytes():
    source = 'é"\\\n' * 500
    bounded, truncated = bound_payload(source, max_bytes=800, min_field_chars=20)
    assert truncated
    retained, marker = bounded.split("… [truncated, ")
    assert retained == source[:len(retained)]
    assert marker == f"{len(source) - len(retained)} chars omitted]"
    assert len(retained) < int(len(source) * 0.7)
    assert len(canonical_json(bounded).encode("utf-8")) <= 800
    exact_bytes = len(canonical_json(bounded).encode("utf-8"))
    assert bound_payload(bounded, max_bytes=exact_bytes) == (bounded, False)


def test_nested_list_markers_survive_parent_tail_reduction():
    payload = {"groups": [list(range(40)), list(range(40))]}
    bounded, truncated = bound_payload(payload, max_bytes=140)
    assert truncated
    assert bounded == {
        "groups": [[0, "… [truncated, 39 items omitted]"],
                   "… [truncated, 1 items omitted]"]
    }
    assert len(canonical_json(bounded).encode("utf-8")) <= 140
    assert payload == {"groups": [list(range(40)), list(range(40))]}


@pytest.mark.parametrize(
    "payload, budget, expected",
    [
        ({"z": "z" * 300, "a": "a" * 300}, 570,
         {"z": "z" * 210 + "… [truncated, 90 chars omitted]", "a": "a" * 300}),
        ({"z": ["z" * 100] * 10, "a": ["a" * 100] * 10}, 2000,
         {"z": ["z" * 100] * 7 + ["… [truncated, 3 items omitted]"],
          "a": ["a" * 100] * 10}),
    ],
)
def test_single_reduction_preserves_bytes_and_path_tiebreak(payload, budget, expected):
    bounded, truncated = bound_payload(payload, max_bytes=budget)
    assert truncated
    assert canonical_json(bounded) == canonical_json(expected)


def test_source_marker_text_is_ordinary_payload_data():
    source = "… [truncated, 999 chars omitted]" * 10
    bounded, truncated = bound_payload([source] * 2, max_bytes=100, min_field_chars=0)
    assert truncated
    for text in bounded:
        retained, marker = text.rsplit("… [truncated, ", 1)
        assert retained == source[:len(retained)]
        assert marker == f"{len(source) - len(retained)} chars omitted]"


def test_default_prompt_bounds_mixed_payload(payload_renders):
    payload = {
        "session": {"summary": "x" * 201},
        "observations": [{"source": "tool", "text": "a" * 100}] * 500,
    }
    before = canonical_json(payload)
    built = build_prompt("t", SCHEMA, payload, config=PromptConfig())
    payload_text = built.user_prompt.split("# Payload\n", 1)[1].split("\n\nReply", 1)[0]
    bounded = json.loads(payload_text)
    assert len(payload_text.encode("utf-8")) <= 60_000
    assert built.payload_truncated
    assert bounded["session"]["summary"] == "x" * 200 + "… [truncated, 1 chars omitted]"
    assert bounded["observations"] == (
        payload["observations"][:350] + ["… [truncated, 150 items omitted]"]
    )
    assert built.prompt_sha256 == sha256_text(built.user_prompt)
    assert built.prompt_bytes == len(built.user_prompt.encode("utf-8"))
    assert canonical_json(payload) == before
    # The prompt also serializes its schema and final payload.
    assert len(payload_renders) <= 1 + 1 + 499 + 2


def test_bounding_redacts_complete_values_before_cutting():
    dummy_secret = "dummy-value-" * 100
    payload = {"log": "API_TOKEN=" + dummy_secret, "notes": ["x" * 201] * 2}
    built = build_prompt(
        "t", SCHEMA, payload,
        config=PromptConfig(max_payload_bytes=250, min_field_chars=20),
    )
    assert built.payload_truncated
    assert "dummy-value" not in built.user_prompt
    assert "API_TOKEN=<redacted>" in built.user_prompt
