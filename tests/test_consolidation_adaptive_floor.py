"""The cluster promotion floor is adaptive by default, on every path.

Each behavioural test carries its own discriminator: the same corpus is run
once under the shipped default and once under ``LM_CONSOLIDATE_MERGE_FLOOR=fixed``.
The ``fixed`` half is what proves the fixture — a corpus whose clusters
really are under 100 traces promotes nothing there — so an accidental
merge into one oversized cluster fails the test instead of passing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from living_memory.consolidation import (
    DEFAULT_MIN_CLUSTER_SIZE,
    MERGE_FLOOR_ENV,
    ConsolidationService,
    adaptive_merge_floor,
    memory_consolidate,
    resolve_min_cluster_size,
)
from living_memory.server import _auto_consolidate_if_due, create_mcp_server
from living_memory.storage import MemoryStore

SCOPE = "project:alpha"
TOPIC = "deploy rollback requires migration guard before release"
OTHER_TOPIC = "tokenizer vocabulary warmup blocks worker restart"


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)


@pytest.fixture(autouse=True)
def _no_ambient_valve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test states the valve it wants; none inherits the operator's."""

    monkeypatch.delenv(MERGE_FLOOR_ENV, raising=False)


def _seed(store: MemoryStore, scope: str, topic: str, count: int) -> list[str]:
    """Append `count` traces that all land in one token-Jaccard cluster.

    The trailing index keeps each write byte-distinct — the store collapses
    identical content in a scope, which would otherwise shrink the cluster.
    """

    return [
        store.append_trace(
            f"{topic} sample {index}",
            {"scope": scope, "agent": "agent-a" if index % 2 == 0 else "agent-b"},
            feedback={"confidence": 0.4, "usefulness_score": 0.2},
        ).id
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("trace_count", "expected"),
    [
        (0, 3),
        (3, 3),
        (24, 3),
        (25, 5),
        (99, 5),
        (100, 25),
        (999, 25),
        (1000, DEFAULT_MIN_CLUSTER_SIZE),
        (12_874, DEFAULT_MIN_CLUSTER_SIZE),
    ],
)
def test_adaptive_floor_ladder_steps_with_corpus_size(
    trace_count: int, expected: int
) -> None:
    assert adaptive_merge_floor(trace_count) == expected


def test_manual_consolidate_promotes_a_mid_sized_cluster_with_no_env_set(
    tmp_path: Path,
) -> None:
    """The headline change: twelve traces, no env var, a concept appears."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        seeded = _seed(store, SCOPE, TOPIC, 12)

        result = memory_consolidate(store, scope=SCOPE)

        assert len(result.concepts_created) == 1
        concept = result.concepts_created[0]
        assert concept.level == "concept"
        assert concept.scope == SCOPE
        assert concept.source_traces == sorted(seeded)
        # The traces stay: consolidation is a layer over the corpus, not a
        # compactor.
        assert len(store.list_nodes(level="trace", scope=SCOPE)) == 12


def test_fixed_valve_restores_the_pre_change_floor_on_the_manual_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MERGE_FLOOR_ENV, "fixed")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 12)

        result = memory_consolidate(store, scope=SCOPE)

        assert result.concepts_created == []
        assert result.concepts_updated == []
        assert store.list_nodes(level="concept", scope=SCOPE) == []


def test_mcp_memory_consolidate_tool_uses_the_adaptive_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manual MCP call is the path that never saw the floor before."""

    mcp = create_mcp_server(tmp_path / "adaptive.sqlite3", mcp_factory=FakeMCP)
    _seed(mcp.memory_store, SCOPE, TOPIC, 40)

    payload = mcp.tools["memory_consolidate"](scope=SCOPE)

    assert len(payload["concepts_created"]) == 1
    assert payload["concepts_created"][0]["level"] == "concept"

    monkeypatch.setenv(MERGE_FLOOR_ENV, "fixed")
    fixed = create_mcp_server(tmp_path / "fixed.sqlite3", mcp_factory=FakeMCP)
    _seed(fixed.memory_store, SCOPE, TOPIC, 40)

    fixed_payload = fixed.tools["memory_consolidate"](scope=SCOPE)

    assert fixed_payload["concepts_created"] == []
    assert fixed_payload["concepts_updated"] == []


def test_auto_path_promotes_sub_hundred_clusters_under_the_default_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default (fixed cadence) auto pass: fires at 100 traces, floor 25.

    Two 50-trace topics: the pre-change floor of 100 promoted neither, the
    adaptive floor of 25 promotes both.
    """

    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 50)
        _seed(store, SCOPE, OTHER_TOPIC, 50)
        service = ConsolidationService(store)

        summary = _auto_consolidate_if_due(store, service, SCOPE)

        assert summary is not None
        assert len(summary["concepts_created"]) == 2
        assert len(store.list_nodes(level="concept", scope=SCOPE)) == 2


def test_auto_path_fixed_valve_keeps_sub_hundred_clusters_unpromoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    monkeypatch.setenv(MERGE_FLOOR_ENV, "fixed")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 50)
        _seed(store, SCOPE, OTHER_TOPIC, 50)
        service = ConsolidationService(store)

        summary = _auto_consolidate_if_due(store, service, SCOPE)

        # The pass still runs on schedule; it just promotes nothing, exactly
        # as before this change.
        assert summary is not None
        assert summary["concepts_created"] == []
        assert store.list_nodes(level="concept", scope=SCOPE) == []


def test_adaptive_policy_branch_still_gets_the_adaptive_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression for the floor table server.py no longer owns."""

    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 10)  # trigger step 5, floor 3
        service = ConsolidationService(store)

        summary = _auto_consolidate_if_due(store, service, SCOPE)

        assert summary is not None
        assert len(summary["concepts_created"]) == 1


def test_explicit_min_cluster_size_beats_both_the_default_and_the_valve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "high.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 12)

        # Adaptive would allow 3; the caller asked for 50.
        assert memory_consolidate(store, scope=SCOPE, min_cluster_size=50).concepts == []

    monkeypatch.setenv(MERGE_FLOOR_ENV, "fixed")
    with MemoryStore(tmp_path / "low.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 12)

        # The valve would demand 100; the caller asked for 3.
        result = memory_consolidate(store, scope=SCOPE, min_cluster_size=3)

        assert len(result.concepts_created) == 1


def test_service_level_min_cluster_size_still_pins_the_floor(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 12)

        service = ConsolidationService(store, min_cluster_size=50)

        assert service.memory_consolidate(scope=SCOPE).concepts == []
        assert service.memory_consolidate(scope=SCOPE, min_cluster_size=3).concepts_created


@pytest.mark.parametrize("invalid", [0, -1])
def test_non_positive_min_cluster_size_is_still_rejected(
    tmp_path: Path, invalid: int
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        with pytest.raises(ValueError, match="min_cluster_size must be positive"):
            memory_consolidate(store, scope=SCOPE, min_cluster_size=invalid)
        with pytest.raises(ValueError, match="min_cluster_size must be positive"):
            ConsolidationService(store, min_cluster_size=invalid)
        with pytest.raises(ValueError, match="min_cluster_size must be positive"):
            ConsolidationService(store).memory_consolidate(
                scope=SCOPE, min_cluster_size=invalid
            )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 5),  # unset: the ladder decides
        ("", 5),
        ("adaptive", 5),
        ("ADAPTIVE", 5),
        ("fixed", DEFAULT_MIN_CLUSTER_SIZE),
        ("  Fixed  ", DEFAULT_MIN_CLUSTER_SIZE),
        ("40", 40),  # an operator-pinned literal floor
        ("0", 5),  # nonsense must not disable promotion entirely
        ("-7", 5),
        ("banana", 5),
    ],
)
def test_valve_values_resolve_without_breaking_a_pass(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int
) -> None:
    if value is None:
        monkeypatch.delenv(MERGE_FLOOR_ENV, raising=False)
    else:
        monkeypatch.setenv(MERGE_FLOOR_ENV, value)

    assert resolve_min_cluster_size(None, trace_count=60) == expected
    # An explicit argument is never touched by the valve.
    assert resolve_min_cluster_size(7, trace_count=60) == 7


def test_pinned_valve_integer_applies_to_a_real_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MERGE_FLOOR_ENV, "20")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store, SCOPE, TOPIC, 12)

        assert memory_consolidate(store, scope=SCOPE).concepts == []

        _seed(store, SCOPE, OTHER_TOPIC, 20)

        promoted = memory_consolidate(store, scope=SCOPE).concepts_created
        assert [concept.provenance["cluster_size"] for concept in promoted] == [20]
