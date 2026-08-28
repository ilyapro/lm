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
* **it stays local** — the streak belongs to one ``(scope, task_pattern or
  task)`` key and no neighbouring key inherits it;
* **it is keyed like the map** — a client whose ``task`` changes every turn but
  whose ``task_pattern`` does not accumulates one streak, not N histories of
  one delivery each, which is the whole reason this rule could never fire in an
  AE chat;
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
import re
import sqlite3
from datetime import UTC, datetime, timedelta
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
from living_memory.storage import (
    RECALL_DELIVERY_HISTORY_HORIZON_HOURS,
    MaturedRecallHistory,
    MemoryStore,
)

SCOPE = "project:curtail"
TASK = "recall-map-curtail"

#: The stable half of the AE section-9 contract: one slug per tree-goal, sent
#: unchanged on every call of every child and every retry.
PATTERN = "lm/recall-map"


def turn(number: int) -> str:
    """The unstable half: what an AE chat puts in ``task`` on turn ``number``.

    Unique per turn by contract (``docs/recall-map.md`` section 9), which is
    exactly why keying the delivery history on it produced an empty window on
    every single turn.
    """

    return f"chat:5f3a91/turn-{number}"

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
    task_pattern: str | None = None,
) -> str:
    """Record one recall event that carried ``payload`` as its map.

    ``task_pattern`` defaults to absent, which is both the shape of every row
    in a database written before the column existed and the shape a client
    that sends no pattern still writes today.
    """

    ambient: dict[str, Any] = {}
    if task is not None:
        ambient["task"] = task
    if task_pattern is not None:
        ambient["task_pattern"] = task_pattern
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


def unread_deliveries(
    store: MemoryStore,
    count: int,
    *,
    per_turn_task: bool = False,
    task: str | None = TASK,
    task_pattern: str | None = None,
) -> list[Node]:
    """``count`` deliveries under one key, each offering an untouched medoid.

    Distinct medoids per delivery on purpose: a shared medoid would make one
    access reset every delivery at once, which is exactly the confound the
    per-delivery walk has to survive.

    ``per_turn_task`` replaces the fixed ``task`` with a fresh
    ``chat:<id>/turn-<N>`` per delivery — the AE shape, where the only thing
    holding the deliveries together is ``task_pattern``.
    """

    medoids: list[Node] = []
    for index in range(count):
        medoid = make_node(store, f"deployment rollback recipe {index}")
        medoids.append(medoid)
        deliver(
            store,
            offered(cluster(f"deploy rollback {index}", medoid.id)),
            task=turn(index) if per_turn_task else task,
            task_pattern=task_pattern,
        )
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


def test_a_medoid_lookup_inside_the_window_resets_the_streak(
    store: MemoryStore,
) -> None:
    """The one probe the ranker cannot satisfy on the agent's behalf.

    ``record_access`` — the probe above — is written by *any* recall that
    returns the node, so a hub the trigger boost mixes into every recall reads
    as "used" forever without a reader anywhere. An id-fetch of one exact ULID
    is different in kind: nothing produces it but a reader who was handed that
    id, which is what the map does.

    So this test deliberately proves the *new* probe and not the old one: the
    node's ``last_accessed`` is asserted untouched on both sides of the lookup,
    because ``memory_lookup`` is a pure read by construction. If the streak
    still resets, only the ledger can have reset it.
    """

    medoids = unread_deliveries(store, CURTAIL_STREAK, task_pattern=PATTERN)
    # The middle delivery's example again: the walk must stop where the
    # evidence is, not at whichever delivery it looks at first.
    before = store.get_node(medoids[1].id)

    store.record_lookup_event(
        [medoids[1].id],
        occurred_at=datetime.now(UTC) + timedelta(hours=1),
    )

    after = store.get_node(medoids[1].id)
    assert after.last_accessed == before.last_accessed
    assert after.access_count == before.access_count

    builder = RecallMapBuilder(store)
    built = builder.build(
        residual(store), scope=SCOPE, task=turn(99), task_pattern=PATTERN
    )

    assert built is not None
    assert built.curtailed is False
    assert built.clusters
    assert builder.last_curtailment.offers == 1


def test_a_lookup_after_the_window_closes_is_not_evidence(store: MemoryStore) -> None:
    """Control for the lookup probe, and the reason it borrows a window.

    The bound is not this module's to invent: ``recall_delivery_history``
    already correlates a lookup with the deliveries whose frozen
    ``(delivered_at, outcome_end]`` range contains it, and the curtail rule
    reads that verdict rather than recomputing one. Two consumers of "was this
    delivery followed" with two definitions of the delivery's window would
    disagree about the same delivery, which is worse than either answer.
    """

    medoids = unread_deliveries(store, CURTAIL_STREAK, task_pattern=PATTERN)

    store.record_lookup_event(
        [medoids[1].id],
        occurred_at=datetime.now(UTC)
        + timedelta(hours=RECALL_DELIVERY_HISTORY_HORIZON_HOURS + 1),
    )

    built = RecallMapBuilder(store).build(
        residual(store), scope=SCOPE, task=turn(99), task_pattern=PATTERN
    )

    assert built is not None, "a lookup outside every window must not stop the collapse"
    assert built.curtailed is True


def test_an_unobservable_window_is_not_a_follow(store: MemoryStore) -> None:
    """NULL is "we could not see", and it must not read as "somebody did".

    Every window that closed before this database recorded its first lookup is
    NULL in the ledger — honestly so. Counting NULL as evidence would hand a
    free pass to every historical delivery and switch the rule off for exactly
    the corpus it exists to describe, so the tri-state's unknown arm is
    silence: those deliveries keep being judged by the two probes that always
    judged them.
    """

    unread_deliveries(store, CURTAIL_STREAK, task_pattern=PATTERN)
    ledger = store.connection.execute(
        "SELECT lookup_consumed FROM recall_delivery_history"
    ).fetchall()
    assert ledger, "the deliveries reached the ledger"
    assert all(row["lookup_consumed"] is None for row in ledger)

    built = RecallMapBuilder(store).build(
        residual(store), scope=SCOPE, task=turn(99), task_pattern=PATTERN
    )

    assert built is not None
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
# The streak is keyed the way the map is keyed
# ----------------------------------------------------------------------


def test_a_per_turn_task_chat_accumulates_its_streak_under_one_pattern(
    store: MemoryStore,
) -> None:
    """The scenario this rule was silently useless for, now pinned.

    An AE chat sends ``task=chat:<id>/turn-<N>`` — a fresh string every turn,
    by contract — alongside a ``task_pattern`` that never changes. The map has
    always keyed its cache on the pattern; the delivery history could only be
    filtered by the task, so every turn read an empty window, every turn was
    the first delivery, and ``CURTAIL_STREAK`` could never be reached no matter
    how long nobody read the map.
    """

    unread_deliveries(
        store, CURTAIL_STREAK, per_turn_task=True, task_pattern=PATTERN
    )

    builder = RecallMapBuilder(store)
    built = builder.build(
        residual(store),
        scope=SCOPE,
        task=turn(CURTAIL_STREAK),  # a task none of those deliveries used
        task_pattern=PATTERN,
    )

    assert built is not None
    assert built.curtailed is True
    assert built.streak == CURTAIL_STREAK
    assert builder.last_curtailment.offers == CURTAIL_STREAK


def test_without_the_pattern_the_same_history_is_invisible(
    store: MemoryStore,
) -> None:
    """The control that makes the test above mean something.

    Identical rows, identical builder, one difference: the build carries no
    ``task_pattern``, so the read falls back to a per-turn ``task`` that
    matches nothing. This is the old behaviour reproduced exactly — and it is
    also the correct behaviour for a key that really has no pattern, since a
    key made of one turn's task genuinely has no history.
    """

    unread_deliveries(
        store, CURTAIL_STREAK, per_turn_task=True, task_pattern=PATTERN
    )

    builder = RecallMapBuilder(store)
    built = builder.build(residual(store), scope=SCOPE, task=turn(CURTAIL_STREAK))

    assert built is not None
    assert built.curtailed is False
    assert built.clusters
    assert builder.last_curtailment.streak == 0


def test_a_delivery_carrying_no_pattern_falls_back_to_its_task(
    store: MemoryStore,
) -> None:
    """Every row written before the column existed is this shape.

    A pattern-keyed build cannot ask those rows for a pattern they do not
    have, so it asks them for the only identity they carry. Without the
    fallback, deploying the column would reset every accrued streak in the
    database to zero.
    """

    unread_deliveries(store, CURTAIL_STREAK)  # no task_pattern anywhere
    assert all(
        row["task_pattern"] is None
        for row in store.connection.execute(
            "SELECT task_pattern FROM recall_events"
        ).fetchall()
    )

    builder = RecallMapBuilder(store)
    built = builder.build(
        residual(store), scope=SCOPE, task=TASK, task_pattern=PATTERN
    )

    assert built is not None
    assert built.curtailed is True
    assert built.streak == CURTAIL_STREAK


def test_the_fallback_still_needs_the_task_to_match(store: MemoryStore) -> None:
    """...and it is a fallback, not a wildcard.

    A pattern-less row says nothing about which pattern it belonged to, so the
    only honest thing to check is its task. A different task under the same
    pattern must not inherit it — that would be attributing one key's silence
    to another, the one direction this module never resolves ambiguity in.
    """

    unread_deliveries(store, CURTAIL_STREAK)  # task=TASK, no pattern

    built = RecallMapBuilder(store).build(
        residual(store), scope=SCOPE, task="some other node", task_pattern=PATTERN
    )

    assert built is not None
    assert built.curtailed is False
    assert built.clusters


def test_a_dark_pattern_does_not_silence_its_neighbour(store: MemoryStore) -> None:
    """Isolation again, at the level the key now actually uses.

    Two tree-goals in one scope: the first has been offered three maps nobody
    read, the second is on its first delivery. Sharing a scope must not make
    them share a verdict.
    """

    unread_deliveries(
        store, CURTAIL_STREAK, per_turn_task=True, task_pattern=PATTERN
    )

    builder = RecallMapBuilder(store)
    dark = builder.build(
        residual(store), scope=SCOPE, task=turn(9), task_pattern=PATTERN
    )
    neighbour = builder.build(
        residual(store), scope=SCOPE, task=turn(9), task_pattern="lm/decay-sweep"
    )

    assert dark is not None and dark.curtailed is True
    assert neighbour is not None and neighbour.curtailed is False
    assert neighbour.clusters


def test_the_memo_is_keyed_on_the_pattern_too(store: MemoryStore) -> None:
    """The gate must count for the key the read will ask about.

    ``_CurtailMemo`` exists to skip the window read while a collapse is
    arithmetically impossible. Keyed on the task while the map is keyed on the
    pattern, a per-turn client would seed a fresh memo every turn and pay the
    read it was built to avoid on every single delivery — the gate inverted
    into a tax.
    """

    pool = residual(store)
    builder = RecallMapBuilder(store)

    assert builder.build(pool, scope=SCOPE, task=turn(0), task_pattern=PATTERN)
    assert builder.last_curtail_read is True, "the first build has nothing cached"

    for index in range(1, CURTAIL_STREAK):
        assert builder.build(pool, scope=SCOPE, task=turn(index), task_pattern=PATTERN)
        assert builder.last_curtail_read is False, "same key, different turn"

    # And the offer that could complete a streak still pays for a look.
    assert builder.build(
        pool, scope=SCOPE, task=turn(CURTAIL_STREAK), task_pattern=PATTERN
    )
    assert builder.last_curtail_read is True


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
    """Default-on, and provably so: no env read reaches the curtail decision.

    The protocol channels ban machinery an agent cannot rely on being there,
    and a retreat that has to be enabled is exactly that. The check is on the
    source, because an env read is invisible from the outside until the day it
    fires.

    This originally asserted that the whole module never touched ``os.environ``
    — a coarse proxy that held while the module had no valves at all. It does
    now: the pool gates are operator-owned by contract, because their
    thresholds come off a field measurement rather than out of this codebase.
    So the check is narrowed to what it always meant, and made stricter in the
    part that matters. Every env name the module reads is enumerated, each one
    is a pool gate, and none of them appears anywhere in the curtail path —
    which is a claim about the code that decides the collapse rather than about
    the file that happens to contain it.
    """

    import inspect

    import living_memory.recall_map as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    read_names = set(re.findall(r'"(LM_[A-Z0-9_]+)"', source))
    assert read_names == {
        module.POOL_USEFULNESS_GATE_ENV,
        module.POOL_USEFULNESS_FLOOR_ENV,
        module.POOL_DEMOTION_GATE_ENV,
        module.POOL_DEMOTION_WINDOWS_ENV,
        module.POOL_COLD_QUOTA_GATE_ENV,
        module.POOL_COLD_SLOTS_ENV,
    }, "an env name appeared that is not a registered pool valve"

    builder = module.RecallMapBuilder
    curtail_path = "\n".join(
        inspect.getsource(member)
        for member in (
            builder._curtailment,
            builder._read_curtailment,
            builder._delivery_history,
            builder._medoid_access,
            builder._medoid_lookups,
            builder._was_followed,
            builder._note_delivery,
            module._delivered_items,
            module._echoes,
        )
    )
    for forbidden in ("getenv", "os.environ", "LM_"):
        assert forbidden not in curtail_path, (
            f"{forbidden!r} reached the curtail decision path"
        )


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
