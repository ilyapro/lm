from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import living_memory.server as server_module
from living_memory.resources import node_to_dict
from living_memory.server import create_mcp_server
from living_memory.storage import RECALL_LOOKUP_EVENT_TABLE, MemoryStore


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

        if func is None:
            return decorate
        return decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)


def test_list_nodes_by_context_finds_exact_task_pattern_in_dense_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        target = store.append_trace(
            "target trace for deterministic lookup",
            {"scope": "project:lookup", "task_pattern": "pattern-target"},
        )
        for index in range(45):
            store.append_trace(
                f"decoy trace {index}",
                {
                    "scope": "project:lookup",
                    "task_pattern": f"pattern-decoy-{index}",
                },
            )
        store.append_trace(
            "other scope matching task pattern",
            {"scope": "project:other", "task_pattern": "pattern-target"},
        )

        matches = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "pattern-target"},
            limit=1000,
        )

        assert [node.id for node in matches] == [target.id]


def test_list_nodes_by_context_and_filters_and_excludes_decayed_by_default(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        target = store.append_trace(
            "and-filter target trace",
            {
                "scope": "project:lookup",
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-a",
                "lesson_kind": "fix",
            },
        )
        store.append_trace(
            "same task pattern different procedure",
            {
                "scope": "project:lookup",
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-b",
                "lesson_kind": "fix",
            },
        )
        store.append_trace(
            "same procedure different lesson kind",
            {
                "scope": "project:lookup",
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-a",
                "lesson_kind": "reference",
            },
        )

        matches = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-a",
                "lesson_kind": "fix",
            },
        )
        assert [node.id for node in matches] == [target.id]
        assert (
            store.list_nodes_by_context(
                scope="project:lookup",
                context_filters={"task_pattern": "does-not-exist"},
            )
            == []
        )

        decayed = store.append_trace(
            "decayed exact lookup trace",
            {"scope": "project:lookup", "task_pattern": "expired-pattern"},
        )
        store.soft_delete_node(decayed.id, "test decay")

        assert (
            store.list_nodes_by_context(
                scope="project:lookup",
                context_filters={"task_pattern": "expired-pattern"},
            )
            == []
        )
        with_decayed = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "expired-pattern"},
            include_decayed=True,
        )
        assert [node.id for node in with_decayed] == [decayed.id]

        with pytest.raises(ValueError, match="unsupported context lookup field"):
            store.list_nodes_by_context(
                scope="project:lookup",
                context_filters={"task_pattern[0]": "shared-pattern"},
            )


def test_list_nodes_by_context_returns_newest_first_and_honors_limit(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        older = store.append_trace(
            "older ordered lookup trace",
            {
                "scope": "project:lookup",
                "task_pattern": "ordered-pattern",
                "timestamp": "2026-01-01T00:00:00Z",
            },
        )
        newer = store.append_trace(
            "newer ordered lookup trace",
            {
                "scope": "project:lookup",
                "task_pattern": "ordered-pattern",
                "timestamp": "2026-01-02T00:00:00Z",
            },
        )

        matches = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "ordered-pattern"},
        )
        limited = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "ordered-pattern"},
            limit=1,
        )

        assert [node.id for node in matches] == [newer.id, older.id]
        assert [node.id for node in limited] == [newer.id]


def test_memory_lookup_tool_is_registered_and_deterministic(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    assert "memory_lookup" in mcp.tools

    target = mcp.tools["memory_remember"](
        "MCP lookup target trace",
        {"scope": "project:mcp-lookup", "task_pattern": "mcp-target-pattern"},
    )
    target_id = target["node"]["id"]
    for index in range(35):
        mcp.tools["memory_remember"](
            f"MCP lookup decoy trace {index}",
            {
                "scope": "project:mcp-lookup",
                "task_pattern": f"mcp-decoy-pattern-{index}",
            },
        )
    other_scope = mcp.tools["memory_remember"](
        "MCP lookup other scope trace",
        {"scope": "project:mcp-other", "task_pattern": "mcp-target-pattern"},
    )

    exact = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="mcp-target-pattern",
    )
    assert exact["scope"] == "project:mcp-lookup"
    assert exact["filters"] == {"task_pattern": "mcp-target-pattern"}
    assert exact["count"] == 1
    assert [result["id"] for result in exact["results"]] == [target_id]
    assert exact["results"][0]["context"]["task_pattern"] == "mcp-target-pattern"
    assert "score" not in exact["results"][0]
    assert other_scope["node"]["id"] not in {result["id"] for result in exact["results"]}

    and_target = mcp.tools["memory_remember"](
        "MCP lookup AND target trace",
        {
            "scope": "project:mcp-lookup",
            "task_pattern": "mcp-shared-pattern",
            "procedure_id": "proc-a",
            "lesson_kind": "fix",
        },
    )
    mcp.tools["memory_remember"](
        "MCP lookup AND nonmatch trace",
        {
            "scope": "project:mcp-lookup",
            "task_pattern": "mcp-shared-pattern",
            "procedure_id": "proc-b",
            "lesson_kind": "fix",
        },
    )
    and_lookup = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="mcp-shared-pattern",
        procedure_id="proc-a",
        lesson_kind="fix",
    )
    assert [result["id"] for result in and_lookup["results"]] == [and_target["node"]["id"]]

    missing = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="no-such-pattern",
    )
    assert missing["count"] == 0
    assert missing["results"] == []

    decayed = mcp.tools["memory_remember"](
        "MCP lookup decayed trace",
        {"scope": "project:mcp-lookup", "task_pattern": "expired-pattern"},
    )
    mcp.tools["memory_forget"](decayed["node"]["id"], "test decay")
    decayed_lookup = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="expired-pattern",
    )
    assert decayed_lookup["count"] == 0

    no_filters = mcp.tools["memory_lookup"](scope="project:mcp-lookup")
    assert no_filters["error"] == "at least one exact context filter is required"
    assert no_filters["results"] == []


# --- Fetch-by-id: the re-fetch path for snippet/duplicate recall stubs ------


def test_memory_lookup_by_id_round_trips_full_content(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    long_content = (
        "лунный трактор пересекает ρ-многообразие — unicode survives\n"
        "line two: exact bytes preserved, including trailing spaces  \n"
        + "z" * 2000
    )
    remembered = mcp.tools["memory_remember"](long_content, {"scope": "project:idfetch"})
    target_id = remembered["node"]["id"]

    fetched = mcp.tools["memory_lookup"](node_id=target_id)
    assert "error" not in fetched
    assert fetched["scope"] is None
    assert fetched["filters"] == {}
    assert fetched["count"] == 1
    assert fetched["missing"] == []
    node = fetched["results"][0]
    assert node["id"] == target_id
    assert node["content"] == long_content  # byte-identical, never snippeted
    assert set(node) == set(node_to_dict(mcp.memory_store.get_node(target_id)))

    # `level` defaults to "trace" but is an exact-context concern; id fetch
    # ignores it, so a mismatching level cannot hide the node.
    by_level = mcp.tools["memory_lookup"](node_id=target_id, level="concept")
    assert [entry["id"] for entry in by_level["results"]] == [target_id]


def test_memory_lookup_by_ids_preserves_order_and_lists_missing(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    first = mcp.tools["memory_remember"]("id-fetch first", {"scope": "project:idfetch"})
    second = mcp.tools["memory_remember"]("id-fetch second", {"scope": "project:idfetch"})
    first_id = first["node"]["id"]
    second_id = second["node"]["id"]

    fetched = mcp.tools["memory_lookup"](
        node_ids=[second_id, "01NOPEnotarealnodeid000000", first_id]
    )
    assert fetched["count"] == 2
    assert [entry["id"] for entry in fetched["results"]] == [second_id, first_id]
    assert fetched["missing"] == ["01NOPEnotarealnodeid000000"]

    # node_id combines with node_ids, deduplicated in request order.
    combined = mcp.tools["memory_lookup"](node_id=first_id, node_ids=[second_id, first_id])
    assert [entry["id"] for entry in combined["results"]] == [first_id, second_id]
    assert combined["missing"] == []


def test_memory_lookup_id_with_scope_verifies_instead_of_filtering(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    remembered = mcp.tools["memory_remember"](
        "scoped id-fetch trace", {"scope": "project:alpha"}
    )
    target_id = remembered["node"]["id"]

    matching = mcp.tools["memory_lookup"](scope="project:alpha", node_id=target_id)
    assert [entry["id"] for entry in matching["results"]] == [target_id]
    assert matching["scope_mismatches"] == []

    crossed = mcp.tools["memory_lookup"](scope="project:beta", node_id=target_id)
    assert [entry["id"] for entry in crossed["results"]] == [target_id]  # not filtered
    assert crossed["scope_mismatches"] == [{"id": target_id, "scope": "project:alpha"}]


def test_memory_lookup_by_id_returns_decayed_nodes(tmp_path: Path) -> None:
    """Exact refs must survive decay: a content_ref handed out before a sweep
    still resolves, with the decayed flag visible to the caller."""

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    remembered = mcp.tools["memory_remember"](
        "decay-surviving id-fetch trace", {"scope": "project:idfetch"}
    )
    target_id = remembered["node"]["id"]
    mcp.tools["memory_forget"](target_id, "test decay")

    fetched = mcp.tools["memory_lookup"](node_id=target_id)
    assert fetched["count"] == 1
    assert fetched["results"][0]["decayed"] is True
    assert fetched["results"][0]["content"] == "decay-surviving id-fetch trace"


def test_memory_lookup_ids_cannot_combine_with_context_filters(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    remembered = mcp.tools["memory_remember"](
        "combination probe trace",
        {"scope": "project:idfetch", "task_pattern": "combo-pattern"},
    )

    combined = mcp.tools["memory_lookup"](
        node_id=remembered["node"]["id"], task_pattern="combo-pattern"
    )
    assert combined["error"] == "node_id/node_ids cannot be combined with context filters"
    assert combined["count"] == 0
    assert combined["results"] == []


def test_memory_lookup_scope_optional_only_for_id_fetch(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    filters_without_scope = mcp.tools["memory_lookup"](task_pattern="some-pattern")
    assert filters_without_scope["error"] == "scope is required with context filters"
    assert filters_without_scope["results"] == []

    nothing = mcp.tools["memory_lookup"]()
    assert nothing["error"] == "at least one exact context filter is required"
    assert nothing["results"] == []


# --- The lookup as a follow signal -----------------------------------------
#
# Fetching a named ULID is the one act that proves somebody read a card the
# recall map offered, and until now the server threw it away. These pin both
# halves of the trade: the event is written, and the read stays a read.


def _lookup_events(store: MemoryStore) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in store.connection.execute(
            f"""
            SELECT lookup_event_id, node_id, occurred_at, transport_session_id
            FROM {RECALL_LOOKUP_EVENT_TABLE}
            ORDER BY occurred_at, node_id
            """
        )
    ]


def test_memory_lookup_id_fetch_records_the_requested_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server_module, "_transport_session_id", lambda: "agent-connection"
    )
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    remembered = mcp.tools["memory_remember"]("followed card", {"scope": "project:follow"})
    target_id = remembered["node"]["id"]

    fetched = mcp.tools["memory_lookup"](
        node_ids=[target_id, "01NOPEnotarealnodeid000000"]
    )
    assert [entry["id"] for entry in fetched["results"]] == [target_id]
    # The response contract is untouched: a missing id is still reported.
    assert fetched["missing"] == ["01NOPEnotarealnodeid000000"]

    events = _lookup_events(store)
    # Both ids are recorded, including the one that resolved to nothing: the
    # request is the signal, and asking for a node that is gone is still a
    # follow of whatever offered it.
    assert {event["node_id"] for event in events} == {
        target_id,
        "01NOPEnotarealnodeid000000",
    }
    assert len({event["lookup_event_id"] for event in events}) == 1
    assert {event["transport_session_id"] for event in events} == {"agent-connection"}
    assert all(event["occurred_at"] for event in events)

    # A second fetch is a second event, not an update of the first.
    mcp.tools["memory_lookup"](node_id=target_id)
    assert len({event["lookup_event_id"] for event in _lookup_events(store)}) == 2


def test_memory_lookup_id_fetch_leaves_the_node_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this whole signal exists to avoid feeding.

    ``access_count`` and ``last_accessed`` are what the map's own
    ``_was_followed`` probe and the usefulness score read. If a lookup bumped
    them, the signal would certify itself and the hub nodes it is meant to
    demote would be immortal by construction.

    Since the lookup-credit work a same-transport fetch of a node an earlier
    recall *delivered* does move ``usefulness_score`` (that is the usage
    signal; tests/test_lookup_credit.py owns it), so this test states the
    invariant honestly: with the transport identity stamped, a lookup with no
    prior delivery touches nothing under every policy; under
    ``LM_LOOKUP_CREDIT_POLICY=off`` even a delivered node stays untouched;
    and under the default the delivered node is credited while its access
    counters still never move.
    """

    monkeypatch.setattr(server_module, "_transport_session_id", lambda: "agent-connection")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    remembered = mcp.tools["memory_remember"]("untouched by reading", {"scope": "project:pure"})
    target_id = remembered["node"]["id"]

    def snapshot() -> dict[str, Any]:
        row = store.connection.execute(
            """
            SELECT access_count, last_accessed, usefulness_score, updated_at
            FROM nodes WHERE id = ?
            """,
            (target_id,),
        ).fetchone()
        return dict(row)

    # --- No prior delivery: untouched under the default policy ------------
    before = snapshot()
    before_events = len(store.list_recall_events())

    for _ in range(5):
        mcp.tools["memory_lookup"](node_id=target_id)

    assert snapshot() == before
    # No recall event, so no delivery, no pending feedback, no ledger window.
    assert len(store.list_recall_events()) == before_events
    assert store.connection.execute(
        "SELECT COUNT(*) FROM recall_delivery_history"
    ).fetchone()[0] == 0
    # ...and the lookups themselves were recorded all the same.
    assert len(_lookup_events(store)) == 5

    # --- Delivered on this transport, policy off: still untouched ----------
    recalled = mcp.tools["memory_recall"]("untouched reading", scope="project:pure")
    assert [result["node"]["id"] for result in recalled["results"]] == [target_id]
    delivered = snapshot()
    assert delivered["access_count"] == before["access_count"] + 1, "the recall, not a lookup"

    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "off")
    for _ in range(5):
        mcp.tools["memory_lookup"](node_id=target_id)
    assert snapshot() == delivered
    assert len(_lookup_events(store)) == 10

    # --- Delivered on this transport, default policy: credited once, and
    # only usefulness moves; the access counters stay a recall's alone.
    monkeypatch.delenv("LM_LOOKUP_CREDIT_POLICY")
    for _ in range(5):
        mcp.tools["memory_lookup"](node_id=target_id)
    credited = snapshot()
    assert credited["usefulness_score"] > delivered["usefulness_score"]
    assert credited["access_count"] == delivered["access_count"]
    assert credited["last_accessed"] == delivered["last_accessed"]
    assert len(store.list_recall_events()) == before_events + 1
    assert len(_lookup_events(store)) == 15


def test_memory_lookup_records_nothing_off_the_id_fetch_path(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    remembered = mcp.tools["memory_remember"](
        "context-filter probe",
        {"scope": "project:nolookup", "task_pattern": "no-event-pattern"},
    )

    # A context-filter query is a search, not a follow of a named id.
    by_filter = mcp.tools["memory_lookup"](
        scope="project:nolookup", task_pattern="no-event-pattern"
    )
    assert [entry["id"] for entry in by_filter["results"]] == [remembered["node"]["id"]]
    assert _lookup_events(store) == []

    # Rejected requests fetched nothing, so they followed nothing.
    for rejected in (
        lambda: mcp.tools["memory_lookup"](
            node_id=remembered["node"]["id"], task_pattern="no-event-pattern"
        ),
        lambda: mcp.tools["memory_lookup"](task_pattern="no-event-pattern"),
        lambda: mcp.tools["memory_lookup"](),
    ):
        assert "error" in rejected()
    assert _lookup_events(store) == []
