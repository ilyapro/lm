"""Unit tests for the pure recall-delivery shaping module.

The ``shape`` helper below pins the LEGACY renderer (every new-lever valve
off), so the pre-existing tests double as the rollback-valve regression
suite: valves off must reproduce the v1 delivery behaviour exactly. The
``shape_default`` helper exercises the production defaults (snippet ladder,
full-node diet, provenance compaction, stats trim, sparse entries).
"""

import copy
import json
from typing import Any

import pytest

from living_memory.delivery import (
    CONTEXT_VALUE_CHARS_ENV,
    DEFAULT_CONTEXT_VALUE_MAX_CHARS,
    DEFAULT_PROVENANCE_VALUE_MAX_CHARS,
    DEFAULT_SNIPPET_LADDER,
    DEFAULT_SNIPPET_MAX_CHARS,
    DELIVERY_FULL,
    DELIVERY_SESSION_DUPLICATE,
    DELIVERY_SNIPPET,
    DELIVERY_TWIN_DUPLICATE,
    ELLIPSIS,
    FULL_NODE_DIET_ENV,
    LADDER_COMPLETE,
    PREVIEW_MAX_CHARS,
    PROVENANCE_VALUE_CHARS_ENV,
    SESSION_DEDUP_ENV,
    SNIPPET_CHARS_ENV,
    SNIPPET_LADDER_ENV,
    SPARSE_ENV,
    STATS_COMPACTION_ENV,
    context_value_max_chars_from_env,
    full_node_diet_enabled_from_env,
    provenance_value_max_chars_from_env,
    session_dedup_enabled_from_env,
    shape_recall_results,
    snippet_ladder_from_env,
    snippet_max_chars_from_env,
    sparse_entries_enabled_from_env,
    stats_compaction_enabled_from_env,
)
from living_memory.models import Node
from living_memory.resources import node_to_dict
from living_memory.retrieval import RecallResult

# The pre-ladder context budget: legacy-mode tests pin v1 behaviour with it.
LEGACY_CONTEXT_VALUE_MAX_CHARS = 240

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
    context_max: int = LEGACY_CONTEXT_VALUE_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Legacy renderer: every new-lever rollback valve is off."""

    return shape_recall_results(
        results,
        already_delivered_ids=delivered or set(),
        snippet_max_chars=snippet_max,
        context_value_max_chars=context_max,
        session_dedup=dedup,
        snippet_ladder=None,
        full_node_diet=False,
        provenance_value_max_chars=0,
        stats_compaction=False,
        sparse_entries=False,
    )


def shape_default(
    results: list[RecallResult],
    *,
    delivered: set[str] | None = None,
    **overrides: Any,
) -> list[dict[str, Any]]:
    """Production defaults: ladder, full-node diet, compaction, sparse."""

    kwargs: dict[str, Any] = {
        "already_delivered_ids": delivered or set(),
        "snippet_max_chars": DEFAULT_SNIPPET_MAX_CHARS,
        "context_value_max_chars": DEFAULT_CONTEXT_VALUE_MAX_CHARS,
        "session_dedup": True,
    }
    kwargs.update(overrides)
    return shape_recall_results(results, **kwargs)


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


# --- context shaping ------------------------------------------------------

PROCEDURE_STEPS = [
    "step one: " + "recalibrate the torque flange before every ledger sync. " * 20,
    "step two: " + "verify the manifold seal against the archived checklist. " * 20,
]


def procedural_context() -> dict[str, Any]:
    return {
        "scope": "project:test",
        "agent": "memory_consolidate",
        "procedure": list(PROCEDURE_STEPS),
        "procedure_key": "torque flange sync",
        "task_pattern": "torque-flange-sync",
        "trigger": "torque flange sync",
    }


def test_full_delivery_keeps_bulky_context_untouched() -> None:
    node = make_node("proc-full", "short procedural summary", context=procedural_context())

    shaped = shape([make_result(node)])

    assert shaped[0]["delivery"] == DELIVERY_FULL
    assert shaped[0]["node"]["context"] == procedural_context()
    assert shaped[0]["node"]["context"]["procedure"] == PROCEDURE_STEPS


def test_stub_compacts_oversized_procedure_list_to_counter() -> None:
    node = make_node("proc-seen", "Procedure: torque flange sync\nbody", context=procedural_context())

    shaped = shape([make_result(node)], delivered={"proc-seen"})

    entry = shaped[0]
    assert entry["delivery"] == DELIVERY_SESSION_DUPLICATE
    context = entry["node"]["context"]
    measured = len(json.dumps(PROCEDURE_STEPS, ensure_ascii=False))
    assert context["procedure"] == {"count": len(PROCEDURE_STEPS), "chars": measured}
    assert context["procedure_key"] == "torque flange sync"
    assert context["task_pattern"] == "torque-flange-sync"
    assert context["trigger"] == "torque flange sync"
    assert set(context) == set(procedural_context())  # key set preserved


def test_snippet_and_twin_compact_oversized_context_too() -> None:
    long_content = "torque flange procedure body\n" + "x" * 3000
    twin_content = "twin procedural content"
    results = [
        make_result(make_node("long", long_content, context=procedural_context()), score=0.9),
        make_result(make_node("bearer", twin_content), score=0.8),
        make_result(make_node("twin", twin_content, context=procedural_context()), score=0.7),
    ]

    shaped = shape(results, snippet_max=1200)

    assert deliveries(shaped) == [DELIVERY_SNIPPET, DELIVERY_FULL, DELIVERY_TWIN_DUPLICATE]
    for entry in (shaped[0], shaped[2]):
        assert entry["node"]["context"]["procedure"]["count"] == len(PROCEDURE_STEPS)


def test_oversized_string_context_value_truncated_at_boundary() -> None:
    notes = "n" * 200 + ". " + "m" * 400
    node = make_node("seen-str", "content", context={"scope": "project:test", "notes": notes})

    shaped = shape([make_result(node)], delivered={"seen-str"})

    shaped_notes = shaped[0]["node"]["context"]["notes"]
    assert shaped_notes == "n" * 200 + ELLIPSIS
    assert len(shaped_notes) <= LEGACY_CONTEXT_VALUE_MAX_CHARS


def test_small_context_values_kept_verbatim_on_stubs() -> None:
    context = {
        "scope": "project:test",
        "files": ["src/a.py", "src/b.py"],
        "attempt": 3,
        "flaky": False,
        "note": None,
        "meta": {"kind": "closure"},
    }
    node = make_node("seen-small", "content", context=dict(context))

    shaped = shape([make_result(node)], delivered={"seen-small"})

    assert shaped[0]["delivery"] == DELIVERY_SESSION_DUPLICATE
    assert shaped[0]["node"]["context"] == context


def test_oversized_dict_context_value_compacted_with_entry_count() -> None:
    blob = {f"key_{index}": "v" * 120 for index in range(4)}
    node = make_node("seen-dict", "content", context={"scope": "project:test", "blob": blob})

    shaped = shape([make_result(node)], delivered={"seen-dict"})

    measured = len(json.dumps(blob, ensure_ascii=False))
    assert shaped[0]["node"]["context"]["blob"] == {"count": 4, "chars": measured}


def test_context_value_exactly_at_limit_stays_verbatim() -> None:
    value = "v" * DEFAULT_CONTEXT_VALUE_MAX_CHARS
    node = make_node("seen-edge", "content", context={"scope": "project:test", "edge": value})

    shaped = shape([make_result(node)], delivered={"seen-edge"})

    assert shaped[0]["node"]["context"]["edge"] == value


def test_context_chars_measured_without_ascii_escapes() -> None:
    # ~110 chars as UTF-8 text, ~5x that if measured via \u-escapes.
    steps = ["шаг проверки данных"] * 5
    node = make_node("seen-cyr", "content", context={"scope": "project:test", "шаги": steps})

    shaped = shape([make_result(node)], delivered={"seen-cyr"})

    assert shaped[0]["node"]["context"]["шаги"] == steps


def test_context_shaping_disabled_when_zero() -> None:
    node = make_node("seen-off", "content", context=procedural_context())

    shaped = shape([make_result(node)], delivered={"seen-off"}, context_max=0)

    assert shaped[0]["delivery"] == DELIVERY_SESSION_DUPLICATE
    assert shaped[0]["node"]["context"]["procedure"] == PROCEDURE_STEPS


def test_context_not_mutated_and_shaping_deterministic() -> None:
    node = make_node("seen-pure", "content", context=procedural_context())
    result = make_result(node)
    before = copy.deepcopy(node.context)

    first = shape([result], delivered={"seen-pure"})
    second = shape([result], delivered={"seen-pure"})

    assert node.context == before
    assert first == second
    assert first[0]["node"]["context"]["procedure"]["count"] == len(PROCEDURE_STEPS)


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
        results,
        already_delivered_ids=delivered,
        snippet_max_chars=1200,
        context_value_max_chars=240,
        session_dedup=True,
    )
    second = shape_recall_results(
        results,
        already_delivered_ids=delivered,
        snippet_max_chars=1200,
        context_value_max_chars=240,
        session_dedup=True,
    )

    assert first == second
    assert delivered == delivered_before
    assert long_node.content == "z" * 4000
    assert long_node.provenance == provenance_before
    assert twin_a.provenance == provenance
    # Production defaults: the ladder ships the first bearer complete, and the
    # diet ran (summarized provenance in the output, inputs untouched above).
    assert deliveries(first) == [DELIVERY_FULL, DELIVERY_SESSION_DUPLICATE, DELIVERY_TWIN_DUPLICATE]
    assert first[0]["node"]["content"] == "z" * 4000
    assert first[0]["node"]["provenance"]["prior_recalls"] == {"count": len(PRIOR_RECALLS)}


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


def test_context_value_chars_env_default_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONTEXT_VALUE_CHARS_ENV, raising=False)
    assert context_value_max_chars_from_env() == DEFAULT_CONTEXT_VALUE_MAX_CHARS == 160

    monkeypatch.setenv(CONTEXT_VALUE_CHARS_ENV, "500")
    assert context_value_max_chars_from_env() == 500

    monkeypatch.setenv(CONTEXT_VALUE_CHARS_ENV, "  64  ")
    assert context_value_max_chars_from_env() == 64

    monkeypatch.setenv(CONTEXT_VALUE_CHARS_ENV, "0")
    assert context_value_max_chars_from_env() == 0

    monkeypatch.setenv(CONTEXT_VALUE_CHARS_ENV, "-3")
    assert context_value_max_chars_from_env() == 0

    monkeypatch.setenv(CONTEXT_VALUE_CHARS_ENV, "not-a-number")
    assert context_value_max_chars_from_env() == DEFAULT_CONTEXT_VALUE_MAX_CHARS


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


# --- snippet ladder (default mode) ----------------------------------------


def test_ladder_first_bearer_ships_complete_content_even_when_huge() -> None:
    huge = "top ranked dossier\n" + "x" * 20_000

    shaped = shape_default([make_result(make_node("top", huge))])

    assert deliveries(shaped) == [DELIVERY_FULL]
    assert shaped[0]["node"]["content"] == huge
    assert "content_ref" not in shaped[0]  # nothing was cut from this entry


def test_ladder_descending_budgets_by_bearer_position() -> None:
    results = [
        make_result(make_node(f"n{index}", f"bearer{index}" + "z" * 2000), score=1.0 - index / 10)
        for index in range(4)
    ]

    shaped = shape_default(results)

    assert deliveries(shaped) == [DELIVERY_FULL] + [DELIVERY_SNIPPET] * 3
    assert shaped[0]["node"]["content"] == "bearer0" + "z" * 2000
    for position, entry in enumerate(shaped[1:], start=1):
        assert len(entry["node"]["content"]) <= DEFAULT_SNIPPET_LADDER[position]
    lengths = [len(entry["node"]["content"]) for entry in shaped[1:]]
    assert lengths == sorted(lengths, reverse=True)
    assert lengths[0] > lengths[-1]  # genuinely descending, not a shared budget


def test_ladder_positions_past_end_reuse_last_budget() -> None:
    results = [
        make_result(make_node(f"n{index}", f"bearer{index}" + "z" * 2000), score=1.0 - index / 20)
        for index in range(7)
    ]

    shaped = shape_default(results)

    last_budget = DEFAULT_SNIPPET_LADDER[-1]
    tail_lengths = [len(entry["node"]["content"]) for entry in shaped[len(DEFAULT_SNIPPET_LADDER) - 1 :]]
    assert all(length == last_budget for length in tail_lengths)


def test_stubs_do_not_consume_ladder_slots() -> None:
    twin_content = "twinned charter body " + "y" * 1500
    results = [
        make_result(make_node("seen", "already delivered body"), score=0.9),
        make_result(make_node("bearer-a", twin_content), score=0.8),
        make_result(make_node("twin-a", twin_content), score=0.7),
        make_result(make_node("bearer-b", "second bearer" + "z" * 2000), score=0.6),
    ]

    shaped = shape_default(results, delivered={"seen"})

    assert deliveries(shaped) == [
        DELIVERY_SESSION_DUPLICATE,
        DELIVERY_FULL,
        DELIVERY_TWIN_DUPLICATE,
        DELIVERY_SNIPPET,
    ]
    # "seen" and the twin are stubs: bearer-a takes ladder position 0 (complete
    # despite exceeding every finite budget), bearer-b position 1.
    assert shaped[1]["node"]["content"] == twin_content
    assert len(shaped[3]["node"]["content"]) <= DEFAULT_SNIPPET_LADDER[1]


def test_ladder_none_restores_uniform_budget_including_top() -> None:
    huge = "uniform mode dossier\n" + "x" * 20_000

    shaped = shape_default([make_result(make_node("top", huge))], snippet_ladder=None)

    assert deliveries(shaped) == [DELIVERY_SNIPPET]
    assert len(shaped[0]["node"]["content"]) <= DEFAULT_SNIPPET_MAX_CHARS


# --- full-node diet ---------------------------------------------------------


def bulky_provenance_node(node_id: str) -> tuple[Node, list[str]]:
    recalled = [f"01H{index:022d}" for index in range(12)]
    node = make_node(
        node_id,
        "complete decision record body",
        context={"scope": "project:test", "procedure": ["step " + "s" * 120 for _ in range(4)]},
        provenance={
            "prior_recalls": copy.deepcopy(PRIOR_RECALLS),
            "recalled_nodes": list(recalled),
            "strategy": "procedural",
        },
    )
    return node, recalled


def test_full_delivery_dieted_provenance_context_and_minimal_ref() -> None:
    node, recalled = bulky_provenance_node("full-diet")

    shaped = shape_default([make_result(node)])

    entry = shaped[0]
    assert entry["delivery"] == DELIVERY_FULL
    assert entry["node"]["content"] == "complete decision record body"  # content stays whole
    provenance = entry["node"]["provenance"]
    assert provenance["prior_recalls"] == {"count": len(PRIOR_RECALLS)}
    measured = len(json.dumps(recalled, ensure_ascii=False))
    assert provenance["recalled_nodes"] == {"count": len(recalled), "chars": measured}
    assert provenance["strategy"] == "procedural"  # short values verbatim
    procedure = entry["node"]["context"]["procedure"]
    assert procedure["count"] == 4
    # The minimal hint: lookup returns strictly more than was delivered.
    assert entry["content_ref"] == {"node_id": "full-diet"}


def test_full_delivery_without_diet_losses_has_no_ref() -> None:
    shaped = shape_default([make_result(make_node("tidy", "tidy body"))])

    assert shaped[0]["delivery"] == DELIVERY_FULL
    assert "content_ref" not in shaped[0]


def test_full_node_diet_valve_off_restores_untouched_full_entries() -> None:
    node, _ = bulky_provenance_node("full-raw")

    shaped = shape_default([make_result(node)], full_node_diet=False)

    entry = shaped[0]
    assert entry["delivery"] == DELIVERY_FULL
    assert entry["node"]["provenance"] == node_to_dict(node)["provenance"]
    assert entry["node"]["context"] == node_to_dict(node)["context"]
    assert "content_ref" not in entry


# --- provenance value compaction -------------------------------------------


def test_provenance_values_compacted_per_key_on_stubs() -> None:
    node, recalled = bulky_provenance_node("seen-prov")

    shaped = shape_default([make_result(node)], delivered={"seen-prov"})

    provenance = shaped[0]["node"]["provenance"]
    assert provenance["prior_recalls"] == {"count": len(PRIOR_RECALLS)}
    assert provenance["recalled_nodes"]["count"] == len(recalled)
    assert provenance["strategy"] == "procedural"


def test_provenance_chars_zero_limits_shaping_to_legacy_prior_recalls() -> None:
    node, recalled = bulky_provenance_node("seen-legacy-prov")

    shaped = shape_default(
        [make_result(node)], delivered={"seen-legacy-prov"}, provenance_value_max_chars=0
    )

    provenance = shaped[0]["node"]["provenance"]
    assert provenance["prior_recalls"] == {"count": len(PRIOR_RECALLS)}  # legacy summary stays
    assert provenance["recalled_nodes"] == recalled  # everything else untouched


def test_corrections_survive_compaction_structured_with_truncated_text() -> None:
    long_text = "the flange torque threshold was recorded wrong and must read 42 Nm. " * 6
    corrections = [
        {
            "by": "reviewer",
            "timestamp": "2026-08-01T00:00:00Z",
            "text": long_text,
            "supersedes": "01OLDNODE",
        }
    ]
    node = make_node("seen-corr", "content", corrections=list(corrections))

    shaped = shape_default([make_result(node)], delivered={"seen-corr"})

    shaped_corrections = shaped[0]["node"]["provenance"]["corrections"]
    assert len(shaped_corrections) == 1
    entry = shaped_corrections[0]
    assert set(entry) == set(corrections[0])  # every correction key survives
    assert entry["by"] == "reviewer"
    assert entry["timestamp"] == "2026-08-01T00:00:00Z"
    assert entry["supersedes"] == "01OLDNODE"
    assert entry["text"].endswith(ELLIPSIS)
    assert len(entry["text"]) <= DEFAULT_PROVENANCE_VALUE_MAX_CHARS


# --- stats compaction -------------------------------------------------------


def test_default_valued_stats_compact_to_empty_dict() -> None:
    shaped = shape_default([make_result(make_node("st-default", "content"))])

    assert shaped[0]["node"]["stats"] == {}


def test_informative_stats_survive_with_rounded_usefulness() -> None:
    node = make_node(
        "st-info",
        "content",
        access_count=7,
        last_accessed="2026-08-01T00:00:00Z",
        usefulness_score=0.12345678,
        confidence=0.9,
        unique_agents=3,
        temporal_hint="2026-Q3",
    )

    shaped = shape_default([make_result(node)])

    assert shaped[0]["node"]["stats"] == {
        "access_count": 7,
        "usefulness_score": 0.123457,
        "confidence": 0.9,
        "unique_agents": 3,
        "temporal_hint": "2026-Q3",
    }


def test_stats_compaction_valve_off_restores_complete_stats() -> None:
    node = make_node("st-raw", "content")

    shaped = shape_default([make_result(node)], stats_compaction=False)

    assert shaped[0]["node"]["stats"] == node.stats


# --- sparse entries ---------------------------------------------------------


def test_sparse_drops_null_and_duplicate_fields_and_rounds_scores() -> None:
    node = make_node("sp", "content", agent=None, task=None)
    result = make_result(
        node,
        score=0.87654321,
        bm25_score=0.0,
        vector_score=0.4444449,
        graph_score=0.0,
        trigger_score=0.0,
        scope_rank=0,
        methods=("vector",),
        path=(),
    )

    shaped = shape_default([result])

    entry = shaped[0]
    assert set(entry) == (BASELINE_RESULT_KEYS | {"delivery"}) - {"path", "recall_event_id"}
    # Score fields stay present — zeros included — with 6-decimal precision:
    # wire consumers never need existence checks and keep 1e-5 tolerances.
    assert entry["score"] == 0.876543
    assert entry["bm25_score"] == 0.0
    assert entry["vector_score"] == 0.444445
    assert entry["graph_score"] == 0.0
    assert entry["trigger_score"] == 0.0
    assert entry["scope_rank"] == 0
    assert entry["methods"] == ["vector"]
    node_dict = entry["node"]
    for absent in ("agent", "task", "decay_reason", "decayed", "timestamp", "updated_at"):
        assert absent not in node_dict
    assert node_dict["created_at"] == "2026-07-31T00:00:00Z"


def test_sparse_keeps_divergent_node_fields_and_nonempty_path() -> None:
    node = make_node(
        "sp2",
        "content",
        timestamp="2026-08-01T12:00:00Z",
        updated_at="2026-08-02T00:00:00Z",
        decayed=True,
        decay_reason="superseded",
    )

    shaped = shape_default([make_result(node, path=("root", "sp2"))])

    entry = shaped[0]
    assert entry["bm25_score"] == 0.25
    assert entry["vector_score"] == 0.5
    assert entry["scope_rank"] == 1
    assert entry["methods"] == ["bm25", "vector"]
    assert entry["path"] == ["root", "sp2"]
    assert "recall_event_id" not in entry  # the envelope carries it once
    node_dict = entry["node"]
    assert node_dict["agent"] == "tester"
    assert node_dict["task"] == "delivery-shaping"
    assert node_dict["timestamp"] == "2026-08-01T12:00:00Z"
    assert node_dict["updated_at"] == "2026-08-02T00:00:00Z"
    assert node_dict["decayed"] is True
    assert node_dict["decay_reason"] == "superseded"


def test_sparse_content_ref_keeps_ids_and_drops_fetch_boilerplate() -> None:
    twin_content = "twinned content shared byte for byte"
    results = [
        make_result(make_node("bearer", twin_content), score=0.9),
        make_result(make_node("twin", twin_content), score=0.8),
    ]

    shaped = shape_default(results)

    assert shaped[1]["content_ref"] == {
        "node_id": "twin",
        "full_content_chars": len(twin_content),
        "duplicate_of": "bearer",
    }


def test_sparse_valve_off_restores_baseline_entry_shape_and_precision() -> None:
    result = make_result(
        make_node("sp-raw", "content", agent=None),
        score=0.87654321,
        graph_score=0.0,
        scope_rank=0,
        path=(),
    )

    shaped = shape_default([result], sparse_entries=False)

    entry = shaped[0]
    assert set(entry) == BASELINE_RESULT_KEYS | {"delivery"}
    assert entry["score"] == 0.87654321
    assert entry["node"]["agent"] is None
    assert entry["node"]["decayed"] is False


# --- new-lever env knobs ----------------------------------------------------


def test_snippet_ladder_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SNIPPET_LADDER_ENV, raising=False)
    monkeypatch.delenv(SNIPPET_CHARS_ENV, raising=False)
    assert snippet_ladder_from_env() == DEFAULT_SNIPPET_LADDER
    assert DEFAULT_SNIPPET_LADDER[0] == LADDER_COMPLETE  # the top-result guarantee

    # Explicit uniform budget with no ladder set: the legacy contract survives,
    # including "0 disables snippeting".
    monkeypatch.setenv(SNIPPET_CHARS_ENV, "900")
    assert snippet_ladder_from_env() is None

    # An explicit ladder wins over the uniform budget.
    monkeypatch.setenv(SNIPPET_LADDER_ENV, "full,800,300")
    assert snippet_ladder_from_env() == (LADDER_COMPLETE, 800, 300)
    monkeypatch.delenv(SNIPPET_CHARS_ENV, raising=False)

    monkeypatch.setenv(SNIPPET_LADDER_ENV, " Complete , 1200 ")
    assert snippet_ladder_from_env() == (LADDER_COMPLETE, 1200)

    for off_flag in ("off", "uniform", "none", "FALSE"):
        monkeypatch.setenv(SNIPPET_LADDER_ENV, off_flag)
        assert snippet_ladder_from_env() is None

    for malformed in ("full,eight", "-5", "800;300"):
        monkeypatch.setenv(SNIPPET_LADDER_ENV, malformed)
        assert snippet_ladder_from_env() == DEFAULT_SNIPPET_LADDER


def test_provenance_value_chars_env_default_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROVENANCE_VALUE_CHARS_ENV, raising=False)
    assert provenance_value_max_chars_from_env() == DEFAULT_PROVENANCE_VALUE_MAX_CHARS == 160

    monkeypatch.setenv(PROVENANCE_VALUE_CHARS_ENV, "500")
    assert provenance_value_max_chars_from_env() == 500

    monkeypatch.setenv(PROVENANCE_VALUE_CHARS_ENV, "0")
    assert provenance_value_max_chars_from_env() == 0

    monkeypatch.setenv(PROVENANCE_VALUE_CHARS_ENV, "-3")
    assert provenance_value_max_chars_from_env() == 0

    monkeypatch.setenv(PROVENANCE_VALUE_CHARS_ENV, "not-a-number")
    assert provenance_value_max_chars_from_env() == DEFAULT_PROVENANCE_VALUE_MAX_CHARS


def test_boolean_lever_envs_default_on_with_rollback_valves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levers = (
        (FULL_NODE_DIET_ENV, full_node_diet_enabled_from_env),
        (STATS_COMPACTION_ENV, stats_compaction_enabled_from_env),
        (SPARSE_ENV, sparse_entries_enabled_from_env),
    )
    for env_var, reader in levers:
        monkeypatch.delenv(env_var, raising=False)
        assert reader() is True, env_var

        monkeypatch.setenv(env_var, "0")
        assert reader() is False, env_var

        monkeypatch.setenv(env_var, "FALSE")
        assert reader() is False, env_var

        monkeypatch.setenv(env_var, "1")
        assert reader() is True, env_var

        monkeypatch.setenv(env_var, "")
        assert reader() is True, env_var
