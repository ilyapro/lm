"""A map nobody follows must go quiet — and say so.

The preprompt-push lesson is the reason this file exists: *delivered is not
used*, and a channel that keeps paying its budget after it stopped changing
behaviour is worse than no channel, because it looks like it is working. So
the builder measures whether its own past deliveries were followed, and these
tests pin every side of that measurement:

* **it collapses** — three deliveries under one key with nothing consumed and
  the fourth is a marker, not a map;
* **it takes evidence** — a later query echoing a cluster's phrasing, or a
  later access reaching a cluster's medoid, resets the streak, and each of the
  two probes is exercised alone so neither can hide behind the other;
* **it says what happened** — the collapsed payload carries ``curtailed`` and
  the streak, spends a fraction of the map's budget, and renders as nothing at
  all in the instructions channel;
* **it stays local** — the streak belongs to one ``(scope, task)`` key and no
  neighbouring key inherits it;
* **it comes back** — the window that holds the evidence also expires it, so a
  dark key retries rather than dying;
* **it is default-on and read-only** — no flag turns it on, and the probe adds
  no write to the recall path.

Every ambiguity in the module resolves against collapsing, and the tests that
matter most here are the ones proving that: an unreadable payload and a
foreign key both keep the map alive.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from living_memory.models import Node
from living_memory.recall_map import (
    CURTAIL_HISTORY_LIMIT,
    CURTAIL_QUERY_OVERLAP,
    CURTAIL_STREAK,
    MAX_RESPONSE_CHARS,
    RecallMap,
    RecallMapBuilder,
    _echoes,
)
from living_memory.grounding import token_set
from living_memory.retrieval import RecallResult
from living_memory.storage import MaturedRecallHistory, MemoryStore

SCOPE = "project:curtail"
TASK = "recall-map-curtail"

#: Deliveries whose query says nothing about any cluster on offer. The whole
#: file depends on this being true, so it is one constant rather than a phrase
#: retyped per test.
IDLE_QUERY = "unrelated housekeeping question"


#: Corpus the label gate measures rarity against. Every scenario here needs a
#: map to actually come back, and the gate withholds a cluster whose label is
#: house vocabulary — which, on a store holding five notes, is every label:
#: nothing is rare relative to nothing. Sized the same way
#: ``tests/test_recall_map.py`` sizes its fixture, and the pools below are
#: keyed on ``procedure_id`` values these documents never mention.
CORPUS_DOCUMENTS = 32


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with MemoryStore(tmp_path / "curtail.sqlite3") as opened:
        monkeypatch.setattr(
            opened,
            "matured_recall_history",
            lambda candidate_ids, _decision_at: {
                node_id: MaturedRecallHistory.known(100, 50, 50)
                for node_id in candidate_ids
            },
        )
        for index in range(CORPUS_DOCUMENTS):
            opened.create_node(
                level="trace",
                content=f"quarterly ledger reconciliation entry {index}",
                context={"scope": SCOPE},
            )
        yield opened


def make_node(store: MemoryStore, content: str, **context: Any) -> Node:
    payload: dict[str, Any] = {"scope": SCOPE}
    payload.update(context)
    return store.create_node(level="trace", content=content, context=payload)


def residual(store: MemoryStore) -> list[RecallResult]:
    """A pool that always produces a map: three filled structural keys."""

    nodes = [
        make_node(store, "anchor bench protocol note", procedure_id="anchor-bench"),
        make_node(store, "anchor bench arm rotation", procedure_id="anchor-bench"),
        make_node(store, "holdout seal recipe", procedure_id="holdout-seal"),
        make_node(store, "holdout seal amendment", procedure_id="holdout-seal"),
        make_node(store, "vocab index pitfall", procedure_id="vocab-index"),
    ]
    return [
        RecallResult(node=node, score=1.0 - index * 0.01)
        for index, node in enumerate(nodes)
    ]


def cluster(label: str, medoid_id: str, *, count: int = 2, ask_hint: str | None = None):
    """One cluster of a delivered payload, in ``MapCluster.to_dict`` shape."""

    hint = ask_hint if ask_hint is not None else label
    return {
        "label": label,
        "count": count,
        "medoid": {"node_id": medoid_id, "example": f"{label} example"},
        "ask_hint": hint,
        "plan_item": f"on touching {label} - recall '{hint}' ({count})",
    }


def deliver(
    store: MemoryStore,
    payload: dict[str, Any] | None,
    *,
    query: str = IDLE_QUERY,
    scope: str = SCOPE,
    task: str | None = TASK,
) -> str:
    """Record one recall event that carried ``payload`` as its map."""

    ambient: dict[str, Any] = {}
    if task is not None:
        ambient["task"] = task
    event = store.record_recall_event(
        query=query,
        scope=scope,
        ambient_context=ambient,
        results=[],
        recall_map=payload,
    )
    return event.id


def offered(*clusters: dict[str, Any]) -> dict[str, Any]:
    """A delivered map payload carrying real clusters."""

    return {
        "clusters": list(clusters),
        "pool": 40,
        "covered": sum(int(entry["count"]) for entry in clusters),
    }


def marker(streak: int) -> dict[str, Any]:
    """What a collapsed delivery persists."""

    return {
        "clusters": [],
        "pool": 40,
        "covered": 0,
        "curtailed": True,
        "streak": streak,
    }


def unread_deliveries(store: MemoryStore, count: int) -> list[Node]:
    """``count`` deliveries under one key, each offering an untouched medoid.

    Distinct medoids per delivery on purpose: a shared medoid would make one
    access reset every delivery at once, which is exactly the confound the
    per-delivery walk has to survive.
    """

    medoids: list[Node] = []
    for index in range(count):
        medoid = make_node(store, f"deployment rollback recipe {index}")
        medoids.append(medoid)
        deliver(store, offered(cluster(f"deploy rollback {index}", medoid.id)))
    return medoids


# ----------------------------------------------------------------------
# The streak
# ----------------------------------------------------------------------


def test_the_map_survives_one_short_of_the_streak(store: MemoryStore) -> None:
    """K-1 unread deliveries are not enough; the control for every collapse."""

    unread_deliveries(store, CURTAIL_STREAK - 1)

    built = RecallMapBuilder(store).build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    assert built.curtailed is False
    assert built.clusters, "a key still inside its allowance must map normally"
    assert "curtailed" not in built.to_dict()


def test_a_streak_of_unread_deliveries_collapses_the_map(store: MemoryStore) -> None:
    """At K, the builder stops describing the pool and reports the silence."""

    unread_deliveries(store, CURTAIL_STREAK)

    builder = RecallMapBuilder(store)
    built = builder.build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    assert built.curtailed is True
    assert built.streak == CURTAIL_STREAK
    assert built.clusters == ()
    assert builder.last_curtailment.offers == CURTAIL_STREAK


def test_the_collapsed_payload_carries_the_marker_and_the_streak(
    store: MemoryStore,
) -> None:
    """The signal is the payload: a marker, a number, and almost no budget."""

    unread_deliveries(store, CURTAIL_STREAK)

    built = RecallMapBuilder(store).build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    payload = built.to_dict()
    assert payload["curtailed"] is True
    assert payload["streak"] == CURTAIL_STREAK
    # Shape is preserved, so a consumer of persisted maps reaches the collapse
    # through the marker rather than through a KeyError.
    assert payload["clusters"] == []
    assert payload["covered"] == 0
    assert payload["pool"] == len(residual(store))
    # A collapsed delivery must be a rounding error next to the map it
    # replaces, or it has not given the channel back.
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    assert size < MAX_RESPONSE_CHARS // 5, size
    # And it occupies nothing at all in the instructions channel.
    assert built.render_compact() == ""
    assert built.plan_items() == []


def test_the_streak_grows_through_the_markers_it_produces(store: MemoryStore) -> None:
    """Collapsed deliveries extend the reported streak but never cause one.

    A marker offers nothing, so it cannot go unaccepted — ``offers`` stays at
    the deliveries that really carried clusters, which is what keeps the rule
    from bootstrapping itself into a collapse.
    """

    unread_deliveries(store, CURTAIL_STREAK)
    deliver(store, marker(CURTAIL_STREAK))
    deliver(store, marker(CURTAIL_STREAK + 1))

    builder = RecallMapBuilder(store)
    built = builder.build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    assert built.curtailed is True
    assert built.streak == CURTAIL_STREAK + 2
    assert builder.last_curtailment.offers == CURTAIL_STREAK


# ----------------------------------------------------------------------
# Consumption resets it — one test per probe
# ----------------------------------------------------------------------


def test_a_later_query_echoing_a_cluster_resets_the_streak(store: MemoryStore) -> None:
    """The plan item was followed: a later recall asked what the map said to ask."""

    medoids = unread_deliveries(store, CURTAIL_STREAK)
    # Paired against test_a_streak_of_unread_deliveries_collapses_the_map: the
    # only difference is what the newest delivery's query says.
    followed = make_node(store, "deployment rollback recipe followed")
    deliver(
        store,
        offered(cluster("vocab index", followed.id)),
        # Echoes the label of the delivery before it -- "deploy rollback 2".
        query="what does memory hold about the deploy rollback 2 recipe",
    )

    builder = RecallMapBuilder(store)
    built = builder.build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    assert built.curtailed is False
    assert built.clusters
    # The walk stopped at the delivery that was followed, so only the
    # deliveries newer than it remain in the streak.
    assert builder.last_curtailment.offers == 1
    assert medoids  # the medoid probe stayed silent; this was the query probe


def test_reaching_a_medoid_resets_the_streak(store: MemoryStore) -> None:
    """The example was reached: something delivered the node the map pointed at.

    ``record_access`` is exactly the write ``retrieval`` performs for every
    result it hands over, and the write a ``memory_lookup`` of that node id
    performs when it is recorded — so this is the medoid probe under both of
    its names, with no query anywhere near it.
    """

    medoids = unread_deliveries(store, CURTAIL_STREAK)
    # The middle delivery's example, not the newest: the walk must stop where
    # the evidence is, not wherever it happens to be looking first.
    store.record_access(medoids[1].id)

    builder = RecallMapBuilder(store)
    built = builder.build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    assert built.curtailed is False
    assert built.clusters
    assert builder.last_curtailment.offers == 1


def test_an_untouched_medoid_is_not_evidence(store: MemoryStore) -> None:
    """Control for the medoid probe: a node accessed *before* it was offered.

    Without the timestamp comparison the probe would read every previously
    delivered node as proof of its own success, which is the "delivered equals
    used" fallacy in miniature. The old access is written directly rather than
    slept for: at the second resolution the column stores, a real
    ``record_access`` here would land in the same second as the deliveries and
    the tie resolves — deliberately — the other way.
    """

    stale = make_node(store, "deployment rollback recipe stale")
    for index in range(CURTAIL_STREAK):
        deliver(store, offered(cluster(f"deploy rollback {index}", stale.id)))
    with store.connection:
        store.connection.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?",
            ("2020-01-01T00:00:00Z", stale.id),
        )

    built = RecallMapBuilder(store).build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None, "a stale access must not stop the collapse"
    assert built.curtailed is True


def test_the_query_probe_uses_the_shared_tokenizer(store: MemoryStore) -> None:
    """Overlap is measured on the grounding rail's tokens, not on raw words.

    That is what makes "how do we roll back a deployment" count as following a
    cluster labelled "deploy rollback": the shared tokenizer canonicalizes both
    sides, so the map's phrasing does not have to be echoed literally.
    """

    label = token_set("deploy rollback")
    assert _echoes(label, token_set("notes on deployment rollback steps"))
    assert not _echoes(label, token_set("notes on the anchor bench"))
    # Half of a two-token label is one token, and the threshold is inclusive.
    assert CURTAIL_QUERY_OVERLAP == 0.5
    assert _echoes(label, token_set("rollback"))
    # Nothing can be echoed by nothing.
    assert not _echoes(frozenset(), token_set("deploy rollback"))
    assert not _echoes(label, frozenset())


# ----------------------------------------------------------------------
# Isolation
# ----------------------------------------------------------------------


def test_the_streak_belongs_to_one_cache_key(store: MemoryStore) -> None:
    """A dark task must not silence its neighbour in the same scope."""

    unread_deliveries(store, CURTAIL_STREAK)

    builder = RecallMapBuilder(store)
    dark = builder.build(residual(store), scope=SCOPE, task=TASK)
    neighbour = builder.build(residual(store), scope=SCOPE, task="a different task")
    other_scope = builder.build(residual(store), scope="project:elsewhere", task=TASK)

    assert dark is not None and dark.curtailed is True
    assert neighbour is not None and neighbour.curtailed is False
    assert neighbour.clusters
    assert other_scope is not None and other_scope.curtailed is False
    assert other_scope.clusters


def test_a_taskless_key_does_not_inherit_a_tasks_streak(store: MemoryStore) -> None:
    """The storage filter cannot express "no task", so the builder re-checks it.

    Without that re-check a task-less recall would read every task in the scope
    as its own history and collapse on somebody else's silence.
    """

    unread_deliveries(store, CURTAIL_STREAK)

    built = RecallMapBuilder(store).build(residual(store), scope=SCOPE, task=None)

    assert built is not None
    assert built.curtailed is False
    assert built.clusters


def test_a_taskless_key_collapses_on_its_own_streak(store: MemoryStore) -> None:
    """...and still has a streak of its own, or the re-check would be a mute."""

    for index in range(CURTAIL_STREAK):
        medoid = make_node(store, f"taskless recipe {index}")
        deliver(store, offered(cluster(f"deploy rollback {index}", medoid.id)), task=None)

    built = RecallMapBuilder(store).build(residual(store), scope=SCOPE, task=None)

    assert built is not None
    assert built.curtailed is True
    assert built.streak == CURTAIL_STREAK


# ----------------------------------------------------------------------
# Fail-open: anything unreadable keeps the map alive
# ----------------------------------------------------------------------


def test_deliveries_this_code_cannot_parse_never_collapse_a_key(
    store: MemoryStore,
) -> None:
    """A payload from another version is not evidence that nobody read it."""

    for _ in range(CURTAIL_STREAK * 2):
        deliver(store, {"clusters": "not a list", "shape": "from the future"})

    builder = RecallMapBuilder(store)
    built = builder.build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    assert built.curtailed is False
    assert builder.last_curtailment.offers == 0
    assert builder.last_curtailment.streak == CURTAIL_STREAK * 2


def test_an_empty_history_is_not_a_streak(store: MemoryStore) -> None:
    """A key delivering for the first time has nothing to be judged on."""

    builder = RecallMapBuilder(store)
    built = builder.build(residual(store), scope=SCOPE, task=TASK)

    assert built is not None
    assert built.curtailed is False
    assert builder.last_curtailment == type(builder.last_curtailment)(streak=0, offers=0)


# ----------------------------------------------------------------------
# The gate in front of the expensive read
# ----------------------------------------------------------------------


def test_the_history_read_waits_until_it_could_matter(store: MemoryStore) -> None:
    """The window read is a partition scan, so it is asked, not paid.

    A builder that has made fewer offers since its last read than the streak
    requires cannot collapse anything whatever the history says, so it does
    not go and look.
    """

    pool = residual(store)
    builder = RecallMapBuilder(store)

    assert builder.build(pool, scope=SCOPE, task=TASK) is not None
    assert builder.last_curtail_read is True, "the first build has nothing cached"

    for _ in range(CURTAIL_STREAK - 1):
        assert builder.build(pool, scope=SCOPE, task=TASK) is not None
        assert builder.last_curtail_read is False

    # The offer that could complete a streak is the one that pays for a look.
    assert builder.build(pool, scope=SCOPE, task=TASK) is not None
    assert builder.last_curtail_read is True

    # A different key is a different count, and starts by looking.
    assert builder.build(pool, scope=SCOPE, task="another task") is not None
    assert builder.last_curtail_read is True


def test_the_gate_never_collapses_a_map_on_its_own_word(store: MemoryStore) -> None:
    """The decisive safety property: only a read may collapse a map.

    Here the builder makes a full streak of offers that are never delivered —
    the server dropped them, the process is a bench, whatever the reason — so
    its own count says "collapse" and the history says nothing happened. The
    history wins.
    """

    pool = residual(store)
    builder = RecallMapBuilder(store)
    for _ in range(CURTAIL_STREAK + 1):
        built = builder.build(pool, scope=SCOPE, task=TASK)
        assert built is not None
        assert built.curtailed is False, "no delivery was ever recorded"

    assert builder.last_curtail_read is True, "the gate asked; the read answered"
    assert builder.last_curtailment.offers == 0


# ----------------------------------------------------------------------
# The retry, and the cost
# ----------------------------------------------------------------------


def test_a_dark_key_offers_again_once_its_evidence_leaves_the_window(
    store: MemoryStore,
) -> None:
    """Collapsed is not dead: the window that holds the reason also expires it.

    The markers a collapsed key emits push the unread offers out of the window
    one by one, and the delivery on which the last of them falls below the
    streak is the delivery that offers a full map again. That count is
    arithmetic on the two constants, so it stays pinned if either moves.
    """

    unread_deliveries(store, CURTAIL_STREAK)
    pool = residual(store)
    builder = RecallMapBuilder(store)
    assert (built := builder.build(pool, scope=SCOPE, task=TASK)) is not None
    assert built.curtailed is True

    reopened_after = None
    for index in range(CURTAIL_HISTORY_LIMIT + 1):
        deliver(store, marker(CURTAIL_STREAK + index))
        again = builder.build(pool, scope=SCOPE, task=TASK)
        assert again is not None
        if not again.curtailed:
            reopened_after = index + 1
            assert again.clusters, "the retry must be a real map, not an empty one"
            break

    assert reopened_after == CURTAIL_HISTORY_LIMIT - CURTAIL_STREAK + 1


def test_the_curtail_probe_writes_nothing(store: MemoryStore) -> None:
    """Read-only by construction: the map may not pay for itself with writes.

    Measured with a SQLite authorizer rather than with ``total_changes``,
    because the two answer different questions. The label gate asks the FTS
    vocabulary index for document frequencies, and ``term_document_frequencies``
    folds its terms through a scratch ``temp.`` table — real INSERTs, counted by
    ``total_changes``, landing nowhere. What must stay untouched is the
    *database*, so that is what is asserted: not one insert, update or delete
    against ``main`` on either arm.

    Both arms measured on one pool, because building the pool is itself a
    write and would drown the thing under test.
    """

    unread_deliveries(store, CURTAIL_STREAK)
    pool = residual(store)
    builder = RecallMapBuilder(store)

    written: list[tuple[int, str]] = []
    mutations = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}

    def authorize(action: int, table: Any, column: Any, database: Any, trigger: Any) -> int:
        if action in mutations and database == "main":
            written.append((action, str(table)))
        return sqlite3.SQLITE_OK

    store.connection.set_authorizer(authorize)
    try:
        collapsed = builder.build(pool, scope=SCOPE, task=TASK)
        assert collapsed is not None and collapsed.curtailed is True
        assert written == []

        # The probe runs on the non-collapsing path too, where it is followed
        # by a full build -- label gate, term statistics and all -- also
        # read-only.
        mapped = builder.build(pool, scope=SCOPE, task="a different task")
        assert mapped is not None and mapped.clusters
        assert written == []
    finally:
        store.connection.set_authorizer(None)


def test_the_rule_needs_no_flag() -> None:
    """Default-on, and provably so: no env read in this module can switch it.

    The protocol channels ban machinery an agent cannot rely on being there,
    and a retreat that has to be enabled is exactly that. The check is on the
    module, because an env read is invisible from the outside until the day it
    fires.
    """

    import living_memory.recall_map as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "getenv" not in source
    assert "os.environ" not in source
    assert "LM_" not in source
    assert not hasattr(module, "os"), "the module has no business reading the env"


# ----------------------------------------------------------------------
# End to end, with no hand-written payload anywhere
# ----------------------------------------------------------------------


def test_real_maps_are_their_own_evidence(store: MemoryStore) -> None:
    """The loop closes on the builder's own output, byte for byte.

    Every other test in this file hands the probe a payload written by hand,
    which pins the contract but not the round trip. Here the maps are built,
    persisted exactly as the server persists them, and read back by the same
    builder — so a change to what :meth:`RecallMap.to_dict` writes cannot pass
    this file while quietly breaking the rule that reads it.
    """

    pool = residual(store)
    builder = RecallMapBuilder(store)
    for _ in range(CURTAIL_STREAK):
        built = builder.build(pool, scope=SCOPE, task=TASK)
        assert isinstance(built, RecallMap) and not built.curtailed
        deliver(store, built.to_dict())

    collapsed = builder.build(pool, scope=SCOPE, task=TASK)

    assert collapsed is not None
    assert collapsed.curtailed is True
    assert collapsed.streak == CURTAIL_STREAK

    # Collapsed stays collapsed while the evidence stands...
    assert (again := builder.build(pool, scope=SCOPE, task=TASK)) is not None
    assert again.curtailed is True

    # ...and reopens on real evidence: reach the example the last real map
    # pointed at, and the next build maps again.
    store.record_access(built.clusters[0].medoid.node_id)

    assert (revived := builder.build(pool, scope=SCOPE, task=TASK)) is not None
    assert revived.curtailed is False
    assert revived.clusters
