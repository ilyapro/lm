"""The recall map on the wire: additive, verbatim, recorded.

``memory_recall`` gains one top-level key and loses nothing. These tests pin
that trade from both sides:

* **Additive** — ``recall_map`` appears exactly when the ranked pool left a
  residual and the builder had something to say about it, and is absent when
  either half is missing (no residual, or a builder that returned ``None``).
* **Byte-identical** — with the map on and off, twin stores replaying the same
  recall produce the same ``results`` array and the same every-other-key
  envelope, byte for byte. The delivery diet is untouched; the map rides
  beside it.
* **Verbatim** — whatever the builder returns is what ships. A stubbed builder
  whose payload the map module would never produce arrives unedited, so a
  later change to the *builder* (the curtail work) needs no change here.
* **Recorded** — what was delivered lands in ``recall_events.recall_map`` and
  comes back through ``recent_recall_map_history`` for the downstream
  attribution consumers, filtered by scope, task and transport session.
* **Migratable** — the column is added to a database that lacks it, on open,
  as many times as it is opened, without disturbing what is already there.

Seeds are synthetic and carry explicit ``procedure_id`` keys, so the map's
label cascade stops at its first (structural) stage: these tests are about
delivery, and ``tests/test_recall_map.py`` owns what the labels say.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from living_memory import storage as storage_module
from living_memory.recall_map import RecallMapBuilder
from living_memory.server import create_mcp_server
from living_memory.storage import SCHEMA_VERSION, MemoryStore

from test_transport_identity import FakeMCP

MAP_SCOPE = "project:mapdelivery"
MAP_QUERY = "deployment failure database migration rollback"

#: Three procedures, unevenly filled, all matching ``MAP_QUERY`` on content.
#: Uneven on purpose: the map orders clusters largest-first, so equal piles
#: would make the ordering unobservable.
SEED_PROCEDURES: tuple[tuple[str, int], ...] = (
    ("deploy-rollback", 6),
    ("migration-repair", 4),
    ("failure-triage", 3),
)
SEED_LABELS = {"deploy rollback", "migration repair", "failure triage"}


@pytest.fixture(autouse=True)
def _default_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every scenario starts with the map valve and its neighbours unset."""

    for env in (
        "LM_RECALL_MAP",
        "LM_RECALL_REPEAT_GATING",
        "LM_DELIVERY_SESSION_DEDUP",
        "LM_DELIVERY_SNIPPET_CHARS",
        "LM_AUTO_CONSOLIDATE_POLICY",
        "LM_DECAY_SWEEP_INTERVAL_SEC",
    ):
        monkeypatch.delenv(env, raising=False)


def _seed(store: MemoryStore) -> None:
    for procedure, count in SEED_PROCEDURES:
        for index in range(count):
            store.append_trace(
                f"deployment failure case {procedure} {index}: database "
                f"migration rollback step {index}",
                {"scope": MAP_SCOPE, "procedure_id": procedure},
            )


def _server(db_path: Path) -> Any:
    return create_mcp_server(db_path, mcp_factory=FakeMCP)


def _recall(mcp: Any, **arguments: Any) -> dict[str, Any]:
    return mcp.tools["memory_recall"](MAP_QUERY, scope=MAP_SCOPE, **arguments)


# --- (a) The key is there when there is a residual to describe ---------------


def test_recall_map_rides_along_when_the_pool_leaves_a_residual(
    tmp_path: Path,
) -> None:
    mcp = _server(tmp_path / "memory.sqlite3")
    _seed(mcp.memory_store)

    response = _recall(mcp, max_results=3)

    assert response["count"] == 3
    assert "recall_map" in response
    payload = response["recall_map"]
    # The payload shape is the builder's; the server contributes no keys.
    assert set(payload) <= {"clusters", "pool", "covered", "more"}
    assert {"clusters", "pool", "covered"} <= set(payload)
    assert payload["pool"] == 13 - 3  # everything ranked, minus the delivered cut
    assert payload["clusters"]

    labels = [cluster["label"] for cluster in payload["clusters"]]
    counts = [cluster["count"] for cluster in payload["clusters"]]
    assert set(labels) <= SEED_LABELS
    assert len(set(labels)) == len(labels)  # no cluster is delivered twice
    assert counts == sorted(counts, reverse=True)
    assert sum(counts) == payload["covered"] <= payload["pool"]

    for cluster in payload["clusters"]:
        assert set(cluster) == {"label", "count", "medoid", "ask_hint", "plan_item"}
        assert cluster["medoid"]["node_id"]
        assert cluster["ask_hint"]
        # The line an agent can paste into its own todo plan.
        assert cluster["plan_item"].startswith(f"on touching {cluster['label']} -")

    # The map describes the residual and nothing else: no delivered node is in it.
    delivered = {entry["node"]["id"] for entry in response["results"]}
    mapped = {cluster["medoid"]["node_id"] for cluster in payload["clusters"]}
    assert delivered.isdisjoint(mapped)


# --- (b) ... and absent when there is not ------------------------------------


def test_no_residual_means_no_recall_map_key(tmp_path: Path) -> None:
    mcp = _server(tmp_path / "memory.sqlite3")
    _seed(mcp.memory_store)

    response = _recall(mcp, max_results=50)

    assert 0 < response["count"] < 50  # the whole pool fit inside the cut
    assert "recall_map" not in response
    assert mcp.memory_store.recent_recall_map_history() == []


def test_builder_returning_none_omits_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A residual the builder declines to describe is not an empty map."""

    mcp = _server(tmp_path / "memory.sqlite3")
    _seed(mcp.memory_store)
    seen: list[int] = []

    def decline(self: RecallMapBuilder, results: Any, **kwargs: Any) -> None:
        seen.append(len(results))
        return None

    monkeypatch.setattr(RecallMapBuilder, "build", decline)

    response = _recall(mcp, max_results=3)

    assert seen == [10]  # the builder did see the residual
    assert "recall_map" not in response
    assert mcp.memory_store.recent_recall_map_history() == []


# --- (c) Whatever the builder says is what ships -----------------------------


def test_the_builders_output_is_attached_and_recorded_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server adds no logic: a payload the module would never build lands
    on the wire, and in the column, exactly as returned."""

    sentinel = {
        "clusters": [
            {
                "label": "not a label this module would build",
                "count": 99,
                "medoid": {"node_id": "01SENTINEL", "example": "verbatim"},
                "ask_hint": "ask exactly this",
                "plan_item": "on touching X - recall 'X' (99)",
            }
        ],
        "pool": 1234,
        "covered": 99,
        "more": 7,
    }

    class _Stub:
        def to_dict(self) -> dict[str, Any]:
            return json.loads(json.dumps(sentinel))

    monkeypatch.setattr(
        RecallMapBuilder, "build", lambda self, results, **kwargs: _Stub()
    )

    mcp = _server(tmp_path / "memory.sqlite3")
    _seed(mcp.memory_store)

    response = _recall(mcp, max_results=3)

    assert response["recall_map"] == sentinel
    history = mcp.memory_store.recent_recall_map_history()
    assert [row["recall_map"] for row in history] == [sentinel]
    assert history[0]["id"] == response["recall_event_id"]


# --- (d) Everything else on the response is untouched ------------------------


def test_results_and_envelope_byte_identical_with_the_map_on_and_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Freeze every volatility source so twin stores replay identically:
    # timestamps, event ULIDs, and the opportunistic decay sweep.
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: "2026-01-01T00:00:00Z")
    state = {"n": 0}

    def fake_ulid() -> str:
        state["n"] += 1
        return f"01FAKEULID{state['n']:016d}"

    monkeypatch.setattr(storage_module, "new_ulid", fake_ulid)

    seed = tmp_path / "seed.sqlite3"
    with MemoryStore(seed) as store:
        _seed(store)
    twins = (tmp_path / "on.sqlite3", tmp_path / "off.sqlite3")
    for twin in twins:
        for suffix in ("", "-wal", "-shm"):
            source = Path(str(seed) + suffix)
            if source.exists():
                shutil.copy(source, str(twin) + suffix)

    def run(db_path: Path) -> dict[str, Any]:
        mcp = _server(db_path)
        state["n"] = 10_000  # identical event-id sequences across both runs
        return _recall(mcp, max_results=3)

    monkeypatch.setenv("LM_RECALL_MAP", "1")
    with_map = run(twins[0])
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    without_map = run(twins[1])

    assert "recall_map" in with_map
    assert "recall_map" not in without_map
    assert with_map["count"] == 3  # an empty-vs-empty tie would prove nothing

    def dumped(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    assert dumped(with_map["results"]) == dumped(without_map["results"])
    assert dumped({k: v for k, v in with_map.items() if k != "recall_map"}) == dumped(
        without_map
    )
    # The valve is a valve: off writes no map, on writes exactly one.
    with MemoryStore(twins[1]) as store:
        assert store.recent_recall_map_history() == []
    with MemoryStore(twins[0]) as store:
        assert len(store.recent_recall_map_history()) == 1


# --- (e) The delivered map is queryable history ------------------------------


def test_recent_recall_map_history_filters_and_orders(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        first = store.record_recall_event(
            query="first question",
            scope=MAP_SCOPE,
            ambient_context={"task": "recall-map", "transport_session_id": "ses-a"},
            recall_map={"clusters": [{"label": "alpha", "count": 2}], "pool": 9},
        )
        second = store.record_recall_event(
            query="second question",
            scope=MAP_SCOPE,
            ambient_context={"task": "other-task", "transport_session_id": "ses-b"},
            recall_map={"clusters": [{"label": "beta", "count": 3}], "pool": 4},
        )
        third = store.record_recall_event(
            query="third question",
            scope="project:elsewhere",
            ambient_context={"task": "recall-map", "transport_session_id": "ses-a"},
            recall_map={"clusters": [{"label": "gamma", "count": 1}], "pool": 2},
        )
        # A recall that carried no map is not history.
        store.record_recall_event(query="mapless", scope=MAP_SCOPE)
        # Neither is an empty one: nothing was shown.
        store.record_recall_event(query="empty map", scope=MAP_SCOPE, recall_map={})

        history = store.recent_recall_map_history()
        assert [row["id"] for row in history] == [third.id, second.id, first.id]
        assert history[0] == {
            "id": third.id,
            "created_at": third.created_at,
            "scope": "project:elsewhere",
            "task": "recall-map",
            "query": "third question",
            "transport_session_id": "ses-a",
            "recall_map": {"clusters": [{"label": "gamma", "count": 1}], "pool": 2},
        }

        assert [row["id"] for row in store.recent_recall_map_history(scope=MAP_SCOPE)] == [
            second.id,
            first.id,
        ]
        assert [
            row["id"] for row in store.recent_recall_map_history(task="recall-map")
        ] == [third.id, first.id]
        assert [
            row["id"]
            for row in store.recent_recall_map_history(transport_session_id="ses-a")
        ] == [third.id, first.id]
        assert [
            row["id"]
            for row in store.recent_recall_map_history(
                scope=MAP_SCOPE, task="recall-map", transport_session_id="ses-a"
            )
        ] == [first.id]
        assert [row["id"] for row in store.recent_recall_map_history(limit=1)] == [
            third.id
        ]
        assert store.recent_recall_map_history(limit=0) == []
        assert store.recent_recall_map_history(scope="project:nobody") == []


def test_attach_recall_map_overwrites_and_rejects_unknown_events(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        event = store.record_recall_event(query="q", scope=MAP_SCOPE)
        assert store.recent_recall_map_history() == []

        store.attach_recall_map(event.id, {"clusters": [], "pool": 3})
        assert store.recent_recall_map_history()[0]["recall_map"] == {
            "clusters": [],
            "pool": 3,
        }

        store.attach_recall_map(event.id, {"clusters": [], "pool": 4})
        history = store.recent_recall_map_history()
        assert len(history) == 1
        assert history[0]["recall_map"]["pool"] == 4

        store.attach_recall_map(event.id, None)
        assert store.recent_recall_map_history() == []

        with pytest.raises(KeyError):
            store.attach_recall_map("01NOSUCHEVENT", {"clusters": []})


# --- (f) The column arrives on databases that predate it ---------------------


def test_recall_map_column_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    payload = {"clusters": [{"label": "deploy rollback", "count": 3}], "pool": 8}

    with MemoryStore(db_path) as store:
        event_id = store.record_recall_event(query="q", scope=MAP_SCOPE).id

    # A database created before the column existed.
    connection = sqlite3.connect(db_path)
    connection.execute("ALTER TABLE recall_events DROP COLUMN recall_map")
    connection.commit()
    connection.close()

    def columns(store: MemoryStore) -> set[str]:
        return {
            row["name"]
            for row in store.connection.execute("PRAGMA table_info(recall_events)")
        }

    with MemoryStore(db_path) as store:
        assert "recall_map" in columns(store)
        assert store.recent_recall_map_history() == []  # legacy rows carry no map
        store.attach_recall_map(event_id, payload)

    # Re-opening reconciles nothing and destroys nothing.
    for _ in range(2):
        with MemoryStore(db_path) as store:
            assert "recall_map" in columns(store)
            history = store.recent_recall_map_history()
            assert [row["id"] for row in history] == [event_id]
            assert history[0]["recall_map"] == payload
            version = store.connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            assert version["value"] == str(SCHEMA_VERSION)
