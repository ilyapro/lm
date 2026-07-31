"""Unit tests for the pure recall-delivery shaping module."""

import copy
from typing import Any

import pytest

from living_memory.delivery import (
    DEFAULT_SNIPPET_MAX_CHARS,
    DELIVERY_FULL,
    DELIVERY_SESSION_DUPLICATE,
    DELIVERY_SNIPPET,
    DELIVERY_TWIN_DUPLICATE,
    ELLIPSIS,
    PREVIEW_MAX_CHARS,
    SESSION_DEDUP_ENV,
    SNIPPET_CHARS_ENV,
    session_dedup_enabled_from_env,
    shape_recall_results,
    snippet_max_chars_from_env,
)
from living_memory.models import Node
from living_memory.resources import node_to_dict
from living_memory.retrieval import RecallResult

# Pinned copy of the per-result shape rendered by server._recall_result_to_dict.
BASELINE_RESULT_KEYS = {
    "node",
    "score",
    "bm25_score",
    "vector_score",
    "graph_score",
    "trigger_score",
    "scope_rank",
    "methods",
    "path",
    "recall_event_id",
}

PRIOR_RECALLS = [
    {"id": "evt-a", "query": "first query", "result_ids": ["n1", "n2"]},
    {"id": "evt-b", "query": "second query", "result_ids": ["n3"]},
    {"id": "evt-c", "query": "third query", "result_ids": []},
]


def make_node(node_id: str, content: str, **overrides: Any) -> Node:
    defaults: dict[str, Any] = {
        "id": node_id,
        "level": "trace",
        "content": content,
        "scope": "project:test",
        "agent": "tester",
        "task": "delivery-shaping",
        "timestamp": "2026-07-31T00:00:00Z",
        "context": {"scope": "project:test"},
        "created_at": "2026-07-31T00:00:00Z",
        "updated_at": "2026-07-31T00:00:00Z",
    }
    defaults.update(overrides)
    return Node(**defaults)


def make_result(node: Node, **overrides: Any) -> RecallResult:
    defaults: dict[str, Any] = {
        "node": node,
        "score": 1.0,
        "bm25_score": 0.25,
        "vector_score": 0.5,
        "graph_score": 0.0,
        "trigger_score": 0.0,
        "scope_rank": 1,
        "methods": ("bm25", "vector"),
        "path": (),
        "recall_event_id": "recall-evt-1",
    }
    defaults.update(overrides)
    return RecallResult(**defaults)


def shape(
    results: list[RecallResult],
    *,
    delivered: set[str] | None = None,
    snippet_max: int = DEFAULT_SNIPPET_MAX_CHARS,
    dedup: bool = True,
) -> list[dict[str, Any]]:
    return shape_recall_results(
        results,
        already_delivered_ids=delivered or set(),
        snippet_max_chars=snippet_max,
        session_dedup=dedup,
    )


def deliveries(shaped: list[dict[str, Any]]) -> list[str]:
    return [entry["delivery"] for entry in shaped]


# --- baseline shape -------------------------------------------------------


def test_full_delivery_keeps_node_dict_and_baseline_result_shape() -> None:
    node = make_node("n1", "short content", provenance={"prior_recalls": PRIOR_RECALLS})
    result = make_result(node, score=0.9, path=("n0", "n1"), recall_event_id="evt-9")

    shaped = shape([result])

    assert len(shaped) == 1
    entry = shaped[0]
    assert set(entry) == BASELINE_RESULT_KEYS | {"delivery"}
    assert entry["delivery"] == DELIVERY_FULL
    assert "content_ref" not in entry
    assert entry["node"] == node_to_dict(node)
    assert entry["score"] == 0.9
    assert entry["bm25_score"] == 0.25
    assert entry["vector_score"] == 0.5
    assert entry["graph_score"] == 0.0
    assert entry["trigger_score"] == 0.0
    assert entry["scope_rank"] == 1
    assert entry["methods"] == ["bm25", "vector"]
    assert entry["path"] == ["n0", "n1"]
    assert entry["recall_event_id"] == "evt-9"


def test_empty_results_shape_to_empty_list() -> None:
    assert shape([]) == []


def test_ranked_order_and_scoring_passthrough() -> None:
    results = [
        make_result(make_node("a", "content a"), score=0.9),
        make_result(make_node("b", "content b"), score=0.5, recall_event_id=None),
        make_result(make_node("c", "content c"), score=0.1, methods=("graph",), scope_rank=2),
    ]

    shaped = shape(results)

    assert [entry["node"]["id"] for entry in shaped] == ["a", "b", "c"]
    assert [entry["score"] for entry in shaped] == [0.9, 0.5, 0.1]
    assert shaped[1]["recall_event_id"] is None
    assert shaped[2]["methods"] == ["graph"]
    assert shaped[2]["scope_rank"] == 2
    assert deliveries(shaped) == [DELIVERY_FULL] * 3


# --- twin dedup -----------------------------------------------------------


def test_twin_dedup_marks_lower_ranked_copies() -> None:
    twin_content = "concept text copied verbatim from its source trace"
    results = [
        make_result(make_node("concept-1", twin_content, level="concept"), score=0.9),
        make_result(make_node("other", "unrelated content"), score=0.8),
        make_result(make_node("trace-1", twin_content), score=0.7),
    ]

    shaped = shape(results)

    assert deliveries(shaped) == [DELIVERY_FULL, DELIVERY_FULL, DELIVERY_TWIN_DUPLICATE]
    assert shaped[0]["node"]["content"] == twin_content
    stub = shaped[2]
    assert stub["node"]["content"] == twin_content  # preview: short line fits verbatim
    assert stub["content_ref"] == {
        "node_id": "trace-1",
        "fetch": 'memory_lookup(node_id="trace-1")',
        "full_content_chars": len(twin_content),
        "duplicate_of": "concept-1",
    }


def test_twin_chain_all_point_to_highest_ranked_bearer() -> None:
    content = "the same content repeated across four nodes"
    results = [make_result(make_node(f"n{i}", content), score=1.0 - i / 10) for i in range(4)]

    shaped = shape(results)

    assert deliveries(shaped) == [DELIVERY_FULL] + [DELIVERY_TWIN_DUPLICATE] * 3
    for stub in shaped[1:]:
        assert stub["content_ref"]["duplicate_of"] == "n0"


def test_twin_requires_byte_identical_content() -> None:
    results = [
        make_result(make_node("a", "same text")),
        make_result(make_node("b", "same text ")),
        make_result(make_node("c", "Same text")),
    ]

    assert deliveries(shape(results)) == [DELIVERY_FULL] * 3


def test_empty_content_does_not_twin() -> None:
    results = [make_result(make_node("a", "")), make_result(make_node("b", ""))]

    shaped = shape(results)

    assert deliveries(shaped) == [DELIVERY_FULL, DELIVERY_FULL]
    assert all("content_ref" not in entry for entry in shaped)


# --- session dedup --------------------------------------------------------


def test_session_dedup_stubs_previously_delivered_node() -> None:
    content = "first line of the delivered node\nsecond line with details"
    result = make_result(make_node("seen-1", content))

    shaped = shape([result], delivered={"seen-1"})

    entry = shaped[0]
    assert entry["delivery"] == DELIVERY_SESSION_DUPLICATE
    assert entry["node"]["content"] == "first line of the delivered node"
    assert entry["content_ref"] == {
        "node_id": "seen-1",
        "fetch": 'memory_lookup(node_id="seen-1")',
        "full_content_chars": len(content),
    }


def test_session_dedup_disabled_by_flag() -> None:
    result = make_result(make_node("seen-1", "some content"))

    shaped = shape([result], delivered={"seen-1"}, dedup=False)

    assert shaped[0]["delivery"] == DELIVERY_FULL
    assert shaped[0]["node"]["content"] == "some content"


def test_session_dedup_ignores_undelivered_ids() -> None:
    shaped = shape([make_result(make_node("fresh", "content"))], delivered={"other"})

    assert shaped[0]["delivery"] == DELIVERY_FULL


def test_session_dedup_beats_snippet_for_long_delivered_content() -> None:
    long_content = "lead sentence of a long node.\n" + "body " * 1000
    shaped = shape([make_result(make_node("seen-1", long_content))], delivered={"seen-1"}, snippet_max=100)

    entry = shaped[0]
    assert entry["delivery"] == DELIVERY_SESSION_DUPLICATE
    assert entry["node"]["content"] == "lead sentence of a long node."
    assert entry["content_ref"]["full_content_chars"] == len(long_content)


# --- rule interactions ----------------------------------------------------


def test_twin_beats_session_dedup_for_stub_class() -> None:
    content = "shared content delivered before"
    results = [
        make_result(make_node("bearer", content), score=0.9),
        make_result(make_node("twin", content), score=0.5),
    ]

    shaped = shape(results, delivered={"bearer", "twin"})

    assert deliveries(shaped) == [DELIVERY_SESSION_DUPLICATE, DELIVERY_TWIN_DUPLICATE]
    assert shaped[1]["content_ref"]["duplicate_of"] == "bearer"


def test_twin_of_session_duplicate_stays_stub() -> None:
    content = "shared content, bearer already delivered this session"
    results = [
        make_result(make_node("bearer", content), score=0.9),
        make_result(make_node("twin", content), score=0.5),
    ]

    shaped = shape(results, delivered={"bearer"})

    assert deliveries(shaped) == [DELIVERY_SESSION_DUPLICATE, DELIVERY_TWIN_DUPLICATE]
    assert shaped[1]["content_ref"]["duplicate_of"] == "bearer"


def test_twin_of_snippeted_bearer_stays_twin() -> None:
    content = "x" * 3000
    results = [
        make_result(make_node("bearer", content), score=0.9),
        make_result(make_node("twin", content), score=0.5),
    ]

    shaped = shape(results, snippet_max=1200)

    assert deliveries(shaped) == [DELIVERY_SNIPPET, DELIVERY_TWIN_DUPLICATE]
    assert len(shaped[1]["node"]["content"]) <= PREVIEW_MAX_CHARS
    assert shaped[1]["content_ref"]["duplicate_of"] == "bearer"


# --- snippeting -----------------------------------------------------------


def test_snippet_applied_to_long_full_delivery() -> None:
    content = "\n".join(f"line {i:04d} with some detail text" for i in range(100))
    result = make_result(make_node("long-1", content))

    shaped = shape([result], snippet_max=1200)

    entry = shaped[0]
    assert entry["delivery"] == DELIVERY_SNIPPET
    snippet = entry["node"]["content"]
    assert len(snippet) <= 1200
    assert snippet.endswith(ELLIPSIS)
    assert content.startswith(snippet[: -len(ELLIPSIS)])
    assert entry["content_ref"] == {
        "node_id": "long-1",
        "fetch": 'memory_lookup(node_id="long-1")',
        "full_content_chars": len(content),
    }


def test_snippet_cuts_at_paragraph_boundary() -> None:
    content = "a" * 900 + "\n\n" + "b" * 2000
    shaped = shape([make_result(make_node("n1", content))], snippet_max=1200)

    assert shaped[0]["node"]["content"] == "a" * 900 + ELLIPSIS


def test_snippet_cuts_at_sentence_boundary() -> None:
    content = "s" * 700 + ". " + "t" * 1000
    shaped = shape([make_result(make_node("n1", content))], snippet_max=1200)

    assert shaped[0]["node"]["content"] == "s" * 700 + ELLIPSIS


def test_snippet_cuts_at_word_boundary() -> None:
    content = "w" * 650 + " " + "v" * 1000
    shaped = shape([make_result(make_node("n1", content))], snippet_max=1200)

    assert shaped[0]["node"]["content"] == "w" * 650 + ELLIPSIS


def test_snippet_hard_cut_when_no_clean_boundary() -> None:
    shaped = shape([make_result(make_node("n1", "x" * 3000))], snippet_max=1200)

    snippet = shaped[0]["node"]["content"]
    assert snippet == "x" * 1199 + ELLIPSIS
    assert len(snippet) == 1200


def test_snippet_ignores_boundaries_before_half_budget() -> None:
    content = "short line\n" + "y" * 3000
    shaped = shape([make_result(make_node("n1", content))], snippet_max=1200)

    snippet = shaped[0]["node"]["content"]
    assert len(snippet) == 1200  # early newline skipped, hard cut used
    assert "\n" in snippet


def test_snippet_disabled_when_max_chars_zero() -> None:
    content = "x" * 5000
    shaped = shape([make_result(make_node("n1", content))], snippet_max=0)

    assert shaped[0]["delivery"] == DELIVERY_FULL
    assert shaped[0]["node"]["content"] == content


def test_content_exactly_at_limit_stays_full() -> None:
    content = "x" * 1200
    shaped = shape([make_result(make_node("n1", content))], snippet_max=1200)

    assert shaped[0]["delivery"] == DELIVERY_FULL
    assert shaped[0]["node"]["content"] == content


# --- preview extraction ---------------------------------------------------


def test_preview_short_first_line_verbatim() -> None:
    content = "First line summary.\nrest of the body\nmore lines"
    shaped = shape([make_result(make_node("n1", content))], delivered={"n1"})

    assert shaped[0]["node"]["content"] == "First line summary."


def test_preview_strips_leading_blank_lines_and_padding() -> None:
    content = "\n\n  Real first line  \nmore body"
    shaped = shape([make_result(make_node("n1", content))], delivered={"n1"})

    assert shaped[0]["node"]["content"] == "Real first line"


def test_preview_long_first_line_hard_capped() -> None:
    content = "p" * 400 + "\nrest"
    shaped = shape([make_result(make_node("n1", content))], delivered={"n1"})

    preview = shaped[0]["node"]["content"]
    assert preview == "p" * (PREVIEW_MAX_CHARS - len(ELLIPSIS)) + ELLIPSIS
    assert len(preview) == PREVIEW_MAX_CHARS


def test_preview_long_first_line_cut_at_sentence() -> None:
    content = "s" * 100 + ". " + "t" * 100 + "\nsecond line"
    shaped = shape([make_result(make_node("n1", content))], delivered={"n1"})

    assert shaped[0]["node"]["content"] == "s" * 100 + ELLIPSIS


def test_preview_never_contains_newline() -> None:
    content = "alpha beta\ngamma delta"
    shaped = shape([make_result(make_node("n1", content))], delivered={"n1"})

    assert "\n" not in shaped[0]["node"]["content"]


# --- backward-compat key invariants ---------------------------------------


def test_every_delivery_class_preserves_full_node_key_set() -> None:
    provenance = {"prior_recalls": PRIOR_RECALLS, "recalled_nodes": ["r1", "r2"]}
    shared = "content shared between bearer and twin"
    results = [
        make_result(make_node("long", "x" * 3000, provenance=dict(provenance)), score=0.9),
        make_result(make_node("seen", "short seen content", provenance=dict(provenance)), score=0.8),
        make_result(make_node("bearer", shared, provenance=dict(provenance)), score=0.7),
        make_result(make_node("twin", shared, provenance=dict(provenance)), score=0.6),
        make_result(make_node("fresh", "short fresh content"), score=0.5),
    ]

    shaped = shape(results, delivered={"seen"}, snippet_max=1200)

    assert deliveries(shaped) == [
        DELIVERY_SNIPPET,
        DELIVERY_SESSION_DUPLICATE,
        DELIVERY_FULL,
        DELIVERY_TWIN_DUPLICATE,
        DELIVERY_FULL,
    ]
    expected_node_keys = set(node_to_dict(make_node("ref", "ref content")))
    for entry, result in zip(shaped, results, strict=True):
        node_dict = entry["node"]
        assert set(node_dict) == expected_node_keys
        assert isinstance(node_dict["content"], str)
        assert node_dict["content"]
        assert isinstance(node_dict["provenance"], dict)
        assert set(node_dict["provenance"]) == set(node_to_dict(result.node)["provenance"])
        if entry["delivery"] == DELIVERY_FULL:
            assert set(entry) == BASELINE_RESULT_KEYS | {"delivery"}
        else:
            assert set(entry) == BASELINE_RESULT_KEYS | {"delivery", "content_ref"}
            assert entry["content_ref"]["node_id"] == node_dict["id"]


def test_prior_recalls_summarized_to_count_on_non_full_only() -> None:
    provenance = {"prior_recalls": PRIOR_RECALLS, "recalled_nodes": ["r1"]}
    shared = "identical content for provenance check"
    results = [
        make_result(
            make_node("bearer", shared, provenance=dict(provenance), source_traces=["t1"]),
            score=0.9,
        ),
        make_result(make_node("twin", shared, provenance=dict(provenance)), score=0.5),
    ]

    shaped = shape(results)

    full_prov = shaped[0]["node"]["provenance"]
    assert full_prov["prior_recalls"] == PRIOR_RECALLS
    assert full_prov["source_traces"] == ["t1"]

    stub_prov = shaped[1]["node"]["provenance"]
    assert stub_prov["prior_recalls"] == {"count": 3}
    assert stub_prov["recalled_nodes"] == ["r1"]
    assert stub_prov["source_traces"] == []
    assert stub_prov["corrections"] == []


def test_prior_recalls_absent_or_empty_left_untouched_on_stubs() -> None:
    results = [
        make_result(make_node("no-prov", "text without provenance", provenance={}), score=0.9),
        make_result(
            make_node("empty-prov", "text with empty prior recalls", provenance={"prior_recalls": []}),
            score=0.5,
        ),
    ]

    shaped = shape(results, delivered={"no-prov", "empty-prov"})

    assert deliveries(shaped) == [DELIVERY_SESSION_DUPLICATE] * 2
    assert "prior_recalls" not in shaped[0]["node"]["provenance"]
    assert shaped[1]["node"]["provenance"]["prior_recalls"] == []


# --- purity and determinism -----------------------------------------------


def test_inputs_not_mutated_and_output_deterministic() -> None:
    provenance = {"prior_recalls": copy.deepcopy(PRIOR_RECALLS)}
    long_node = make_node("long", "z" * 4000, provenance=provenance)
    twin_a = make_node("ta", "twin content", provenance=copy.deepcopy(provenance))
    twin_b = make_node("tb", "twin content")
    results = [
        make_result(long_node, score=0.9),
        make_result(twin_a, score=0.8),
        make_result(twin_b, score=0.7),
    ]
    delivered = {"ta", "unrelated"}
    delivered_before = set(delivered)
    provenance_before = copy.deepcopy(long_node.provenance)

    first = shape_recall_results(
        results, already_delivered_ids=delivered, snippet_max_chars=1200, session_dedup=True
    )
    second = shape_recall_results(
        results, already_delivered_ids=delivered, snippet_max_chars=1200, session_dedup=True
    )

    assert first == second
    assert delivered == delivered_before
    assert long_node.content == "z" * 4000
    assert long_node.provenance == provenance_before
    assert twin_a.provenance == provenance
    assert deliveries(first) == [DELIVERY_SNIPPET, DELIVERY_SESSION_DUPLICATE, DELIVERY_TWIN_DUPLICATE]


def test_shaped_node_dicts_are_independent_per_result() -> None:
    node = make_node("n1", "duplicated node object")
    shaped = shape([make_result(node, score=0.9), make_result(node, score=0.5)])

    assert shaped[0]["node"] is not shaped[1]["node"]
    shaped[0]["node"]["content"] = "mutated"
    assert node.content == "duplicated node object"
    assert shaped[1]["node"]["content"] != "mutated"


# --- env knobs ------------------------------------------------------------


def test_snippet_chars_env_default_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SNIPPET_CHARS_ENV, raising=False)
    assert snippet_max_chars_from_env() == DEFAULT_SNIPPET_MAX_CHARS == 1200

    monkeypatch.setenv(SNIPPET_CHARS_ENV, "500")
    assert snippet_max_chars_from_env() == 500

    monkeypatch.setenv(SNIPPET_CHARS_ENV, "  64  ")
    assert snippet_max_chars_from_env() == 64

    monkeypatch.setenv(SNIPPET_CHARS_ENV, "0")
    assert snippet_max_chars_from_env() == 0

    monkeypatch.setenv(SNIPPET_CHARS_ENV, "-3")
    assert snippet_max_chars_from_env() == 0

    monkeypatch.setenv(SNIPPET_CHARS_ENV, "not-a-number")
    assert snippet_max_chars_from_env() == DEFAULT_SNIPPET_MAX_CHARS


def test_session_dedup_env_default_and_rollback_valve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SESSION_DEDUP_ENV, raising=False)
    assert session_dedup_enabled_from_env() is True

    monkeypatch.setenv(SESSION_DEDUP_ENV, "0")
    assert session_dedup_enabled_from_env() is False

    monkeypatch.setenv(SESSION_DEDUP_ENV, "1")
    assert session_dedup_enabled_from_env() is True

    monkeypatch.setenv(SESSION_DEDUP_ENV, "off")
    assert session_dedup_enabled_from_env() is False

    monkeypatch.setenv(SESSION_DEDUP_ENV, "FALSE")
    assert session_dedup_enabled_from_env() is False

    monkeypatch.setenv(SESSION_DEDUP_ENV, "")
    assert session_dedup_enabled_from_env() is True
