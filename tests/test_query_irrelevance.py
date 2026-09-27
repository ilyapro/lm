"""Query-relative irrelevance demotion (goal explicit-recall-feedback, P3).

An accepted ``irrelevant`` mark under ``LM_EXPLICIT_FEEDBACK_POLICY=credit``
demotes the node only for queries that match the marked query's anchor:

* demoted for the marked query and for a near-paraphrase hitting the same
  anchor (weaker, by closeness), not for an unrelated query;
* global ``usefulness_score``, confidence and retrieval weights unchanged;
* ``LM_QUERY_IRRELEVANCE_FACTOR=1.0`` disables it;
* no effect under ``audit`` or ``off``;
* repeated marks accumulate with saturation, positive credit cancels;
* a database without anchor tables (or opened read-only) is tolerated.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sqlite3

import pytest

import living_memory.server as server_module
from living_memory.irrelevance import (
    DEFAULT_QUERY_IRRELEVANCE_FACTOR,
    IRRELEVANCE_MARK_WEIGHT,
    QUERY_IRRELEVANCE_TABLE,
    has_active_irrelevance,
    list_query_irrelevance,
    query_irrelevance_factor,
)
from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore

SCOPE = "project:query-irrelevance"
QUERY = "kappa ledger"
#: Hash-backend cosine to QUERY is ~0.79: over the anchor match floor (0.60),
#: under the dedup cut (0.95) -- a paraphrase that matches the same anchor.
PARAPHRASE = "kappa ledger please"
#: Shares no token with QUERY (cosine 0.0) but finds the marked node lexically.
UNRELATED = "redis eviction storm zset"
AMBIENT = {"agent": "agent-a", "task": "irrelevance-task", "session_id": "irr-session"}
TRANSPORT = "agent-connection"

BODIES = [
    "alembic migration checksum drift blocked the staging rollout entirely",
    "toolbar palette swatches moved into the ColorDock component last week",
    "cert-manager wildcard certificate renewal needs a dns01 solver token",
    "redis eviction storm traced to a runaway zset with unbounded members",
    "grafana dashboard panel queries broke after the datasource uid rename",
    "kafka consumer lag alert fires when the rebalance protocol thrashes",
    "terraform state lock stuck behind an abandoned dynamodb lease record",
    "webpack chunk splitting regressed the vendor bundle size by a third",
]
MARKED_INDEX = 3
#: Grounds BODIES[MARKED_INDEX] and nothing else.
GROUNDING_TRACE = f"{QUERY} follow-up: confirmed that {BODIES[MARKED_INDEX]} and closed it out"
#: Shares no content token with any body: grounds nothing.
UNRELATED_TRACE = "quartz sundial pigment observation unrelated to anything seeded"


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
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    for name in (
        "LM_RETRIEVAL_TUNING_POLICY",
        "LM_RECALL_CREDIT_POLICY",
        "LM_LOOKUP_CREDIT_POLICY",
        "LM_EXPLICIT_FEEDBACK_POLICY",
        "LM_EXPLICIT_CREDIT_WEIGHT",
        "LM_IMPLICIT_LINK_POLICY",
        "LM_QUERY_IRRELEVANCE_FACTOR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server_module, "_transport_session_id", lambda: TRANSPORT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(db_path: Path) -> tuple[Any, MemoryStore, list[str]]:
    mcp = create_mcp_server(db_path, mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    node_ids = [
        mcp.tools["memory_remember"](
            f"{QUERY} {body}",
            {"scope": SCOPE, "agent": "seeder", "session_id": "seed-session"},
        )["node"]["id"]
        for body in BODIES
    ]
    return mcp, store, node_ids


def _deliver(mcp: Any, query: str = QUERY) -> str:
    recalled = mcp.tools["memory_recall"](
        query,
        scope=SCOPE,
        max_results=len(BODIES),
        depth=0,
        ambient_context=dict(AMBIENT),
    )
    return recalled["recall_event_id"]


def _remember(mcp: Any, content: str, **kwargs: Any) -> dict[str, Any]:
    return mcp.tools["memory_remember"](content, {"scope": SCOPE, **AMBIENT}, **kwargs)


def _scores(store: MemoryStore, query: str) -> dict[str, float]:
    """Side-effect-free ranking: no access log, no event."""

    service = MemoryRecallService(store)
    return {
        result.node.id: result.score
        for result in service.memory_recall(
            query, scope=SCOPE, max_results=20, depth=1, log_access=False
        )
    }


def _undemoted(store: MemoryStore, query: str) -> dict[str, float]:
    """The same ranking on the same store state with demotion switched off.

    Deliveries and remembers move access counts and usefulness, so a baseline
    taken before the mark is not comparable; the counterfactual is.
    """

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "audit")
        return _scores(store, query)


def _globals(store: MemoryStore, node_ids: list[str]) -> tuple[Any, ...]:
    nodes = [store.get_node(node_id) for node_id in node_ids]
    weights = store.get_retrieval_weights(SCOPE)
    return (
        [node.usefulness_score for node in nodes],
        [node.confidence for node in nodes],
        (weights.bm25, weights.vector, weights.graph),
    )


def _mark_irrelevant(tmp_path: Path, name: str) -> tuple[Any, MemoryStore, list[str]]:
    """Seed, deliver QUERY, mark the MARKED node irrelevant on a closing remember."""

    mcp, store, node_ids = _seed(tmp_path / f"{name}.sqlite3")
    _deliver(mcp)
    _remember(mcp, UNRELATED_TRACE, irrelevant=[node_ids[MARKED_INDEX]])
    return mcp, store, node_ids


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_demoted_for_marked_query_and_paraphrase_not_for_unrelated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    _mcp, store, node_ids = _mark_irrelevant(tmp_path, "core")
    marked = node_ids[MARKED_INDEX]
    rows = list_query_irrelevance(store)
    assert [(row["node_id"], row["weight"], row["marks"]) for row in rows] == [
        (marked, IRRELEVANCE_MARK_WEIGHT, 1)
    ]
    base = {query: _undemoted(store, query) for query in (QUERY, PARAPHRASE, UNRELATED)}
    after = {query: _scores(store, query) for query in (QUERY, PARAPHRASE, UNRELATED)}
    others = [node_id for node_id in after[QUERY] if node_id != marked]

    # The marked query: exact anchor, closeness 1, one mark = half strength.
    expected = 1.0 - (1.0 - DEFAULT_QUERY_IRRELEVANCE_FACTOR) * IRRELEVANCE_MARK_WEIGHT
    assert after[QUERY][marked] == pytest.approx(base[QUERY][marked] * expected)
    assert all(after[QUERY][n] == pytest.approx(base[QUERY][n]) for n in others)

    # The paraphrase hits the same anchor: demoted, but less than the exact query.
    ratio = after[PARAPHRASE][marked] / base[PARAPHRASE][marked]
    assert expected < ratio < 1.0 - 1e-6
    assert all(
        after[PARAPHRASE][n] == pytest.approx(base[PARAPHRASE][n])
        for n in after[PARAPHRASE]
        if n != marked
    )

    # Unrelated query: the node still ranks exactly as it would without the mark.
    assert marked in after[UNRELATED]
    assert after[UNRELATED] == pytest.approx(base[UNRELATED])


def test_usefulness_confidence_and_weights_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids = _seed(tmp_path / "globals.sqlite3")
    _deliver(mcp)
    before = _globals(store, node_ids)
    _remember(mcp, UNRELATED_TRACE, irrelevant=node_ids[:4])
    assert len(list_query_irrelevance(store)) == 4
    assert _globals(store, node_ids) == before


def test_factor_one_disables_demotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    _mcp, store, node_ids = _mark_irrelevant(tmp_path, "factor-one")
    marked = node_ids[MARKED_INDEX]
    assert _scores(store, QUERY)[marked] < _undemoted(store, QUERY)[marked]
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", "1.0")
    # The association is still recorded; it just has no effect at 1.0.
    assert len(list_query_irrelevance(store)) == 1
    assert _scores(store, QUERY) == pytest.approx(_undemoted(store, QUERY))


@pytest.mark.parametrize("policy", ["audit", "off"])
def test_no_effect_under_audit_or_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
    _mcp, store, _node_ids = _mark_irrelevant(tmp_path, f"policy-{policy}")
    assert list_query_irrelevance(store) == []
    # Nothing to read back even if the valve is opened afterwards.
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    for query in (QUERY, PARAPHRASE, UNRELATED):
        assert _scores(store, query) == pytest.approx(_undemoted(store, query))


def test_valve_back_to_audit_restores_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    _mcp, store, node_ids = _mark_irrelevant(tmp_path, "rollback")
    marked = node_ids[MARKED_INDEX]
    demoted = _scores(store, QUERY)
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "audit")
    restored = _scores(store, QUERY)
    assert demoted[marked] < restored[marked]
    assert list_query_irrelevance(store)[0]["weight"] == IRRELEVANCE_MARK_WEIGHT


# ---------------------------------------------------------------------------
# Accumulation and cancellation
# ---------------------------------------------------------------------------


def test_repeated_marks_accumulate_with_saturation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids = _mark_irrelevant(tmp_path, "saturate")
    marked = node_ids[MARKED_INDEX]
    for _ in range(3):
        _deliver(mcp)
        _remember(mcp, UNRELATED_TRACE, irrelevant=[marked])
    (row,) = list_query_irrelevance(store)
    assert row["marks"] == 4 and row["weight"] == pytest.approx(1.0)
    # Saturated on the exact query: the score is scaled by exactly the factor.
    assert _scores(store, QUERY)[marked] == pytest.approx(
        _undemoted(store, QUERY)[marked] * DEFAULT_QUERY_IRRELEVANCE_FACTOR
    )


def test_grounded_credit_cancels_demotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, _node_ids = _mark_irrelevant(tmp_path, "grounded-cancel")
    _deliver(mcp)
    _remember(mcp, GROUNDING_TRACE)
    (row,) = list_query_irrelevance(store)
    assert row["weight"] == 0.0 and row["cancels"] == 1 and row["marks"] == 1
    assert not has_active_irrelevance(store)
    assert _scores(store, QUERY) == pytest.approx(_undemoted(store, QUERY))


def test_explicit_used_mark_cancels_demotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids = _mark_irrelevant(tmp_path, "used-cancel")
    _deliver(mcp)
    _remember(mcp, UNRELATED_TRACE, used=[node_ids[MARKED_INDEX]])
    (row,) = list_query_irrelevance(store)
    assert row["weight"] == 0.0 and row["cancels"] == 1
    assert _scores(store, QUERY) == pytest.approx(_undemoted(store, QUERY))


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------


def test_database_without_anchor_tables_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids = _seed(tmp_path / "no-anchors.sqlite3")
    with store.connection:
        store.connection.execute("DROP TABLE query_anchor_edges")
        store.connection.execute("DROP TABLE query_anchors")
    store._anchor_tables_present(refresh=True)
    _deliver(mcp)
    _remember(mcp, UNRELATED_TRACE, irrelevant=[node_ids[MARKED_INDEX]])
    assert list_query_irrelevance(store) == []
    assert _scores(store, QUERY) == pytest.approx(_undemoted(store, QUERY))


def test_read_only_snapshot_without_table_reports_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "ro.sqlite3"
    store = MemoryStore(db_path)
    store.connection.close()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        face = SimpleNamespace(connection=conn)
        assert has_active_irrelevance(face) is False
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert QUERY_IRRELEVANCE_TABLE not in tables
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", DEFAULT_QUERY_IRRELEVANCE_FACTOR),
        ("0.25", 0.25),
        ("1.0", 1.0),
        ("0", 0.0),
        ("1.5", DEFAULT_QUERY_IRRELEVANCE_FACTOR),
        ("-0.1", DEFAULT_QUERY_IRRELEVANCE_FACTOR),
        ("nan", DEFAULT_QUERY_IRRELEVANCE_FACTOR),
        ("junk", DEFAULT_QUERY_IRRELEVANCE_FACTOR),
    ],
)
def test_factor_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", raw)
    assert query_irrelevance_factor() == expected
