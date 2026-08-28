"""The deterministic recall-map cascade, stage by stage.

Each stage is exercised on a pool that can only reach *it* — a structural-key
pool, a path-only pool, an anchor-covered pool, an embedding-only pool — so a
stage that silently stopped working cannot be covered up by the one below it.
A mixed pool then pins precedence: a node carrying both a structural key and a
file list is a structural cluster member, never a path one.

Three properties matter as much as the stages and get their own tests. The map
is *deterministic*: identical input builds a byte-identical map, whether the
second build recomputes or is served from cache. It does not *degenerate*: a
pool holding three distinct structural keys must come back as three clusters,
because a map that collapses everything into one generic label costs channel
budget and answers nothing. And it is *stable*: the same key returns the same
labels until the corpus moves, at which point the revision probe — not a timer
— invalidates it.

Every scenario runs against a *corpus*, not a handful of nodes — see
:func:`seed_corpus`. The delivery gate is a corpus statistic, so on a store of
four notes it has nothing to measure and degrades; a fixture that never gave it
a corpus would pin the degraded branch and call it the rule.

The two deliberate mirrors in ``recall_map`` are pinned against their originals
here rather than trusted: ``normalize_key`` against
``consolidation._normalize_trigger`` and ``_chunk_table_revision`` against
``retrieval._chunk_table_revision``.
"""

from __future__ import annotations

import json
import random
import sqlite3
from bisect import bisect_left, bisect_right
from math import nextafter
from pathlib import Path
from types import SimpleNamespace

import pytest

from living_memory import recall_map
from living_memory.chunking import TextChunk
from living_memory.grounding import token_set
from living_memory.models import Node
from living_memory.recall_map import (
    ASK_HINT_MAX_TOKENS,
    FILTER_JOURNAL_LABEL_CHARS,
    FILTER_JOURNAL_NAMES,
    LABEL_GATE_MIN_IC,
    MAX_CLUSTERS,
    MAX_INSTRUCTIONS_CHARS,
    MAX_LABEL_CHARS,
    MAX_RESPONSE_CHARS,
    MEDOID_EXAMPLE_CHARS,
    MIN_CLUSTERS,
    MIN_MEDOID_EXAMPLE_CHARS,
    POOL_DEMOTION_GATE_ENV,
    POOL_DEMOTION_WINDOWS_ENV,
    POOL_GATE_REASON_CODES,
    POOL_USEFULNESS_FLOOR_ENV,
    POOL_USEFULNESS_GATE_ENV,
    RELEVANCE_CALIBRATION_MAX_BREAKPOINTS,
    RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR,
    RELEVANCE_POLICY_CALIBRATION,
    RELEVANCE_POLICY_ID,
    SELECTION_REASON_CODES,
    STAGE_ANCHOR,
    STAGE_EMBEDDING,
    STAGE_PATH,
    STAGE_STRUCTURAL,
    RecallMapBuilder,
    _calibrate,
    _calibration_table,
    _echoes,
    _payload_size,
    cache_key,
    normalize_key,
    relevance_score,
)
from living_memory.retrieval import RecallResult
from living_memory.storage import MaturedRecallHistory, MemoryStore, recall_fingerprint

SCOPE = "project:lm"
DECISION_AT = "2026-08-23T12:00:00Z"

#: Documents the fixture store carries before a test writes anything.
#:
#: Derived, not picked. The gate scores a label's best terms by
#: ``log((N + 1) / (1 + df))`` against a floor of :data:`LABEL_GATE_MIN_IC`, so
#: the corpus size decides what the gate *can* say. Below ``2 * e**2 - 1 = 13.8``
#: documents even a two-word label of terms the corpus has never seen fails the
#: floor, and above ``e**4 - 1 = 53.6`` a single such word starts clearing it.
#: Anywhere between, and every test here sits between with room for the dozen
#: nodes a scenario adds, the fixture pins exactly the property under test:
#: **one generic word never passes, two distinguishing ones always do.**
CORPUS_DOCUMENTS = 32

#: Vocabulary the scenarios below never use, so seeding a corpus moves no
#: label's document frequency.
_CORPUS_FILLER = "quarterly ledger reconciliation entry"


def eligible_history(candidate_ids, _decision_at):
    """A frozen-feature history that clears the production threshold."""

    return {
        node_id: MaturedRecallHistory.known(100, 50, 50)
        for node_id in candidate_ids
    }


def seed_corpus(store: MemoryStore, count: int = CORPUS_DOCUMENTS) -> None:
    """Give the delivery gate a corpus to measure rarity against.

    These nodes never enter a pool. They exist only in the FTS index, which is
    where ``_deliverable`` reads document frequencies from, and they are what
    makes "generic" a statement about a corpus rather than about four notes.
    """

    for index in range(count):
        store.create_node(
            level="trace",
            content=f"{_CORPUS_FILLER} {index}",
            context={"scope": SCOPE},
        )


@pytest.fixture()
def bare_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A store with no corpus at all: the gate's degraded branch lives here."""

    with MemoryStore(tmp_path / "recall-map-bare.sqlite3") as opened:
        monkeypatch.setattr(opened, "matured_recall_history", eligible_history)
        yield opened


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with MemoryStore(tmp_path / "recall-map.sqlite3") as opened:
        monkeypatch.setattr(opened, "matured_recall_history", eligible_history)
        seed_corpus(opened)
        yield opened


def make_node(
    store: MemoryStore, content: str, context: dict | None = None, **extra
) -> Node:
    payload = {"scope": SCOPE}
    payload.update(context or {})
    payload.update(extra)
    return store.create_node(level="trace", content=content, context=payload)


def payload_chars(built) -> int:
    """The map's real size on the wire, journal block included."""

    return len(json.dumps(built.to_dict(), ensure_ascii=False, separators=(",", ":")))


def payload_bytes(built) -> bytes:
    return json.dumps(
        built.to_dict(), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def pool(nodes) -> list[RecallResult]:
    """Rank the nodes best-first, the shape ``last_residual`` hands over."""

    return [
        RecallResult(node=node, score=1.0 - index * 0.01)
        for index, node in enumerate(nodes)
    ]


def chunk_of(text: str) -> TextChunk:
    return TextChunk(
        text=text,
        chunk_index=0,
        token_start=0,
        token_end=1,
        char_start=0,
        char_end=len(text),
    )


# ----------------------------------------------------------------------
# Stage 1 — structural context keys
# ----------------------------------------------------------------------


def test_structural_keys_label_their_own_clusters(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This test owns cascade membership, not response fitting.
    monkeypatch.setattr(recall_map, "MAX_RESPONSE_CHARS", 10_000)
    nodes = [
        make_node(store, "ritual step one", {"procedure_id": "session-bootstrap"}),
        make_node(store, "ritual step two", {"procedure_id": "session-bootstrap"}),
        make_node(store, "the fix that cost a day", {"lesson_kind": "root_cause"}),
        make_node(store, "about latency budgets", {"topic": "latency_budget"}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="anything")

    assert built is not None
    assert {cluster.stage for cluster in built.clusters} == {STAGE_STRUCTURAL}
    labels = {cluster.label: cluster.count for cluster in built.clusters}
    assert labels == {
        "session bootstrap": 2,
        "root cause": 1,
        "latency budget": 1,
    }
    # The key value itself is the label, and it is also a way to ask.
    biggest = built.clusters[0]
    assert biggest.label == "session bootstrap"
    assert biggest.ask_hint == "session bootstrap"
    assert biggest.plan_item == "on touching session bootstrap - recall 'session bootstrap' (2)"
    assert biggest.medoid.node_id == nodes[0].id  # highest-ranked, no vectors here
    assert biggest.medoid.example == "ritual step one"


def test_field_precedence_is_fixed_within_the_structural_stage(store: MemoryStore) -> None:
    """``procedure_id`` outranks every other key on the same node."""

    node = make_node(
        store,
        "carries three keys at once",
        {"procedure_id": "recall-ritual", "lesson_kind": "root_cause", "topic": "recall"},
    )

    built = RecallMapBuilder(store).build(pool([node]), scope=SCOPE)

    assert built is not None
    assert [cluster.label for cluster in built.clusters] == ["recall ritual"]


def test_unreadable_task_pattern_takes_its_label_from_the_medoid(store: MemoryStore) -> None:
    """A hash groups perfectly and names nothing, so the medoid names it."""

    digest = "a3f19c8e77b204d5"
    nodes = [
        make_node(store, "postsession extraction writes attestations", {"task_pattern": digest}),
        make_node(store, "postsession extraction runs offline", {"task_pattern": digest}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    (cluster,) = built.clusters
    assert cluster.count == 2
    assert digest not in cluster.label
    assert cluster.label == "postsession extraction writes"


def test_readable_task_pattern_keeps_its_own_wording(store: MemoryStore) -> None:
    nodes = [
        make_node(store, "a", {"task_pattern": "per-role/recast-execution"}),
        make_node(store, "b", {"task_pattern": "per-role/recast-execution"}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert [cluster.label for cluster in built.clusters] == ["per role recast execution"]


# ----------------------------------------------------------------------
# Stage 2 — path collapse
# ----------------------------------------------------------------------


def test_paths_collapse_to_subsystems(store: MemoryStore) -> None:
    """The subsystem is the head of the label, never the whole of it.

    A directory is where the files happen to live, not what the cluster is
    about, and the field measured what a bare directory reads like on the
    wire: ``public (73)``. So the subsystem keeps its place and the bucket's
    own most distinguishing terms are appended to it.
    """

    nodes = [
        make_node(store, "extraction phase", {"files": ["src/living_memory/postsession/extract.py"]}),
        make_node(store, "attestation phase", {"files": ["src/living_memory/postsession/attest.py"]}),
        make_node(store, "report phase", {"files": ["src/living_memory/postsession/report.py"]}),
        make_node(store, "the field runner", {"files": ["scripts/postsession_field_run.py"]}),
        make_node(store, "a store change", {"files": ["src/living_memory/storage.py"]}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert {cluster.stage for cluster in built.clusters} == {STAGE_PATH}
    assert {cluster.label: cluster.count for cluster in built.clusters} == {
        "postsession phase attestation": 3,
        "scripts field runner": 1,
    }
    # The lowest-ranked row gives way so the delivered rows can carry a real
    # gist; size no longer lets it jump the higher residual hit.
    assert built.dropped == 1
    # Every delivered path label is multi-word and still starts at the
    # subsystem, which is what makes it findable by someone who knows the tree.
    for cluster in built.clusters:
        assert len(cluster.label.split()) >= 2
        assert len(cluster.label) <= MAX_LABEL_CHARS
    # The ask-hint does *not* follow the label to three tokens: it carries the
    # head plus at most one term, so a later query needs exactly the evidence
    # it needed before the labels got richer.
    for cluster in built.clusters:
        assert len(cluster.ask_hint.split()) <= max(
            ASK_HINT_MAX_TOKENS, len(cluster.label.split(" ", 1)[0].split())
        )
        assert cluster.label.startswith(cluster.ask_hint.split()[0])


def test_subsystem_collapse_handles_absolute_paths_and_line_refs(store: MemoryStore) -> None:
    nodes = [
        make_node(
            store,
            "the migration rewrites checkpoints",
            {"files": ["/home/u/p/lm/src/living_memory/postsession/a.py"]},
        ),
        make_node(
            store,
            "the migration replays checkpoints",
            {"files": ["src/living_memory/postsession/b.py:1426-1434"]},
        ),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert [c.count for c in built.clusters] == [2]
    # Both spellings of the path collapsed onto one subsystem, which is still
    # the head of the label the enrichment built on top of it.
    assert built.clusters[0].label.startswith("postsession ")


def test_summarized_file_context_is_not_mistaken_for_a_path(store: MemoryStore) -> None:
    """The delivery diet rewrites long lists as ``{"count": n, "chars": m}``."""

    node = make_node(store, "content with no usable path", {"files": {"count": 4, "chars": 196}})

    built = RecallMapBuilder(store).build(pool([node]), scope=SCOPE)

    assert built is None or all(cluster.stage != STAGE_PATH for cluster in built.clusters)


# ----------------------------------------------------------------------
# Stage 3 — query anchors
# ----------------------------------------------------------------------


def seed_anchor(store: MemoryStore, query: str, targets, *, weight: float = 0.5):
    anchor = store.insert_query_anchor(
        scope=SCOPE,
        query=query,
        fingerprint=recall_fingerprint(query, SCOPE),
        embedding=[1.0, 0.0, 0.0],
    )
    for target in targets:
        store.upsert_query_anchor_edge(anchor.id, target.id, weight=weight)
    return anchor


def test_anchor_query_becomes_both_label_and_ask_hint(store: MemoryStore) -> None:
    covered = [
        make_node(store, "the alt-bundle recipe"),
        make_node(store, "the alt-bundle pitfall"),
    ]
    other = [make_node(store, "how the merge gate picks targets")]
    seed_anchor(store, "как собрать alt-бандл", covered)
    seed_anchor(store, "how does the merge gate pick tests", other)

    built = RecallMapBuilder(store).build(pool(covered + other), scope=SCOPE)

    assert built is not None
    assert {cluster.stage for cluster in built.clusters} == {STAGE_ANCHOR}
    labels = {cluster.label: cluster.count for cluster in built.clusters}
    assert labels == {"как собрать alt-бандл": 2, "how does the merge gate pick tests": 1}
    # The hint is the past query verbatim: a proven way to ask, not a guess.
    for cluster in built.clusters:
        assert cluster.ask_hint == cluster.label


def test_heaviest_edge_wins_and_decayed_anchors_do_not_label(store: MemoryStore) -> None:
    node = make_node(store, "covered twice")
    seed_anchor(store, "the weak question", [node], weight=0.2)
    seed_anchor(store, "the strong question", [node], weight=0.9)
    dead = seed_anchor(store, "the retired question", [node], weight=1.0)
    store.decay_query_anchor(dead.id, "anchor_ttl_expired")

    built = RecallMapBuilder(store).build(pool([node]), scope=SCOPE)

    assert built is not None
    assert [cluster.label for cluster in built.clusters] == ["the strong question"]


# ----------------------------------------------------------------------
# Stage 4 — chunk-embedding fallback with c-TF-IDF labels
# ----------------------------------------------------------------------


def embed(store: MemoryStore, node: Node, vector: list[float]) -> None:
    store.replace_node_chunks(node.id, [(chunk_of(node.content), vector)])


def test_embedding_fallback_clusters_and_labels_by_ctfidf(bare_store: MemoryStore) -> None:
    # The corpus noise is the point of this test, so it is built here rather
    # than taken from the fixture: every document says "memory server", which
    # is what makes those two words house vocabulary the label must not use.
    store = bare_store
    for index in range(CORPUS_DOCUMENTS):
        make_node(store, f"the memory server records a recall event number {index}")

    deploy = [
        make_node(store, "the memory server rollout uses a canary namespace"),
        make_node(store, "the memory server rollout waits for the canary to settle"),
        make_node(store, "rollout of the memory server needs a canary check"),
    ]
    vacuum = [
        make_node(store, "the memory server sqlite file needs an occasional vacuum"),
        make_node(store, "vacuum the sqlite file when the memory server has churned"),
    ]
    for node in deploy:
        embed(store, node, [1.0, 0.05, 0.0])
    for node in vacuum:
        embed(store, node, [0.0, 0.05, 1.0])

    built = RecallMapBuilder(store).build(pool(deploy + vacuum), scope=SCOPE)

    assert built is not None
    assert {cluster.stage for cluster in built.clusters} == {STAGE_EMBEDDING}
    assert [cluster.count for cluster in built.clusters] == [3, 2]

    labels = [cluster.label for cluster in built.clusters]
    # Rare-in-corpus terms win; "memory" and "server" are in every document.
    assert "canary" in labels[0] and "rollout" in labels[0]
    assert "vacuum" in labels[1] and "sqlite" in labels[1]
    for label in labels:
        assert "memory" not in label and "server" not in label


def test_embedding_medoid_is_the_most_central_member(store: MemoryStore) -> None:
    """With vectors in play the medoid is maximal mean similarity, not rank."""

    outlier = make_node(store, "an outlying note about caching")
    centre = make_node(store, "a central note about caching")
    partner = make_node(store, "another note about caching")
    embed(store, outlier, [1.0, 0.9, 0.0])
    embed(store, centre, [1.0, 0.0, 0.0])
    embed(store, partner, [1.0, 0.05, 0.0])

    built = RecallMapBuilder(store).build(pool([outlier, centre, partner]), scope=SCOPE)

    assert built is not None
    (cluster,) = built.clusters
    assert cluster.count == 3
    assert cluster.medoid.node_id == partner.id
    assert cluster.medoid.node_id != outlier.id  # rank order would have said this


def test_unembeddable_nodes_are_left_out_rather_than_guessed(store: MemoryStore) -> None:
    node = make_node(store, "no structural key, no path, no anchor, no chunks")

    built = RecallMapBuilder(store).build(pool([node]), scope=SCOPE)

    assert built is not None
    assert built.clusters == ()
    assert built.pool_size == 1
    assert built.to_dict()["sel"] == {
        "v": "r1",
        "n": 1,
        "e": 1,
        "x": [0, 0, 0, 0, 0, 0, 0],
        "o": 0,
    }


# ----------------------------------------------------------------------
# Precedence across stages
# ----------------------------------------------------------------------


def test_structural_key_beats_a_path_on_the_same_node(store: MemoryStore) -> None:
    nodes = [
        make_node(
            store,
            "extraction writes the attestation",
            {"procedure_id": "postsession-extraction", "files": ["src/living_memory/x.py"]},
        ),
        make_node(store, "a delivery change", {"files": ["src/living_memory/delivery/shape.py"]}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    by_stage = {cluster.stage: cluster for cluster in built.clusters}
    assert set(by_stage) == {STAGE_STRUCTURAL, STAGE_PATH}
    assert by_stage[STAGE_STRUCTURAL].member_ids == (nodes[0].id,)
    assert by_stage[STAGE_PATH].member_ids == (nodes[1].id,)


def test_a_path_beats_an_anchor_on_the_same_node(store: MemoryStore) -> None:
    both = make_node(store, "a store change", {"files": ["src/living_memory/storage.py"]})
    anchored = make_node(store, "no path at all")
    seed_anchor(store, "how does the store migrate", [both, anchored])

    built = RecallMapBuilder(store).build(pool([both, anchored]), scope=SCOPE)

    assert built is not None
    by_stage = {cluster.stage: cluster for cluster in built.clusters}
    assert by_stage[STAGE_PATH].member_ids == (both.id,)
    assert by_stage[STAGE_ANCHOR].member_ids == (anchored.id,)


def test_an_anchor_beats_the_embedding_fallback_on_the_same_node(store: MemoryStore) -> None:
    both = make_node(store, "covered by an anchor and embedded too")
    loose = make_node(store, "only embedded")
    embed(store, both, [1.0, 0.0, 0.0])
    embed(store, loose, [1.0, 0.0, 0.0])
    seed_anchor(store, "which nodes carry chunks", [both])

    built = RecallMapBuilder(store).build(pool([both, loose]), scope=SCOPE)

    assert built is not None
    by_stage = {cluster.stage: cluster for cluster in built.clusters}
    assert by_stage[STAGE_ANCHOR].member_ids == (both.id,)
    assert by_stage[STAGE_EMBEDDING].member_ids == (loose.id,)


def test_mixed_pool_reaches_every_stage(monkeypatch: pytest.MonkeyPatch, store: MemoryStore) -> None:
    """All four stages fire on one pool; the response budget only trims them.

    The budget is lifted here on purpose. One serialized cluster costs 88
    characters of JSON keys before any content, so a four-stage map with real
    labels does not fit :data:`MAX_RESPONSE_CHARS` — which is the documented
    behaviour pinned by ``test_map_is_capped_in_clusters_and_in_characters``,
    and would otherwise hide whether the cascade ran at all.
    """

    structural = [
        make_node(
            store,
            "extraction writes the attestation",
            {"procedure_id": "postsession-extraction", "files": ["src/living_memory/x.py"]},
        ),
        make_node(store, "extraction runs offline", {"procedure_id": "postsession-extraction"}),
    ]
    paths = [
        make_node(store, "a delivery change", {"files": ["src/living_memory/delivery/shape.py"]}),
        make_node(store, "another delivery change", {"files": ["src/living_memory/delivery/diet.py"]}),
    ]
    anchored = [make_node(store, "the anchored leftover")]
    seed_anchor(store, "what did we decide about anchors", anchored)
    embedded = [
        make_node(store, "a leftover about pyproject packaging wheels"),
        make_node(store, "another leftover about pyproject packaging wheels"),
    ]
    for node in embedded:
        embed(store, node, [0.0, 1.0, 0.0])

    monkeypatch.setattr(recall_map, "MAX_RESPONSE_CHARS", 10_000)
    built = RecallMapBuilder(store).build(
        pool(structural + paths + anchored + embedded), scope=SCOPE, task="mixed"
    )

    assert built is not None
    by_stage = {cluster.stage: cluster for cluster in built.clusters}
    assert set(by_stage) == {STAGE_STRUCTURAL, STAGE_PATH, STAGE_ANCHOR, STAGE_EMBEDDING}
    # The node carrying both a procedure_id and a path is claimed by stage 1.
    assert by_stage[STAGE_STRUCTURAL].count == 2
    assert structural[0].id in by_stage[STAGE_STRUCTURAL].member_ids
    assert structural[0].id not in by_stage[STAGE_PATH].member_ids
    assert by_stage[STAGE_PATH].label.startswith("delivery ")
    assert by_stage[STAGE_ANCHOR].label == "what did we decide about anchors"
    assert built.covered == 7
    assert built.dropped == 0
    assert built.withheld == 0
    # Nothing was filtered, so the conditional cluster journal stays absent;
    # mandatory member-selection accounting is a separate additive sibling.
    assert "filtered" not in built.to_dict()


# ----------------------------------------------------------------------
# Determinism and the degenerate-map guard
# ----------------------------------------------------------------------


def test_two_builds_of_the_same_pool_are_identical(store: MemoryStore) -> None:
    nodes = [
        make_node(store, "alpha", {"procedure_id": "alpha-ritual"}),
        make_node(store, "beta", {"files": ["src/living_memory/postsession/b.py"]}),
        make_node(store, "gamma", {"procedure_id": "alpha-ritual"}),
        make_node(store, "delta", {"topic": "delta_topic"}),
    ]
    candidates = pool(nodes)

    # Same builder: the second build is served from the structure cache.
    builder = RecallMapBuilder(store)
    first = builder.build(candidates, scope=SCOPE, task="determinism")
    assert builder.last_cache_hit is False
    second = builder.build(candidates, scope=SCOPE, task="determinism")
    assert builder.last_cache_hit is True

    # Fresh builder: the second build recomputes the whole cascade.
    third = RecallMapBuilder(store).build(candidates, scope=SCOPE, task="determinism")

    assert first == second == third
    assert first is not None
    assert first.to_dict() == third.to_dict()
    assert first.render_compact() == third.render_compact()


def test_cluster_maximum_relevance_precedes_size_and_original_ordinal(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stronger selected member outranks size and retrieval position."""

    alpha = make_node(store, "alpha tail", {"procedure_id": "alpha-ritual"})
    beta = [
        make_node(store, "beta head", {"procedure_id": "beta-ritual"}),
        make_node(store, "beta middle", {"procedure_id": "beta-ritual"}),
    ]
    histories = {
        alpha.id: MaturedRecallHistory.known(200, 100, 100),
        beta[0].id: MaturedRecallHistory.known(100, 50, 50),
        beta[1].id: MaturedRecallHistory.known(100, 50, 50),
    }
    monkeypatch.setattr(
        store,
        "matured_recall_history",
        lambda ids, _at: {node_id: histories[node_id] for node_id in ids},
    )

    built = RecallMapBuilder(store).build(
        pool([*beta, alpha]), scope=SCOPE, decision_at=DECISION_AT
    )

    assert built is not None
    assert [cluster.label for cluster in built.clusters] == [
        "alpha ritual",
        "beta ritual",
    ]
    assert [cluster.count for cluster in built.clusters] == [1, 2]


def test_equal_cluster_maximum_uses_descending_fsum_mean(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpha = [
        make_node(store, "alpha high", {"procedure_id": "alpha-ritual"}),
        make_node(store, "alpha low", {"procedure_id": "alpha-ritual"}),
    ]
    beta = [
        make_node(store, "beta high", {"procedure_id": "beta-ritual"}),
        make_node(store, "beta middle", {"procedure_id": "beta-ritual"}),
    ]
    histories = {
        alpha[0].id: MaturedRecallHistory.known(200, 100, 100),
        alpha[1].id: MaturedRecallHistory.known(100, 50, 2),
        beta[0].id: MaturedRecallHistory.known(200, 100, 100),
        beta[1].id: MaturedRecallHistory.known(100, 50, 50),
    }
    monkeypatch.setattr(
        store,
        "matured_recall_history",
        lambda ids, _at: {node_id: histories[node_id] for node_id in ids},
    )

    built = RecallMapBuilder(store).build(
        pool([*alpha, *beta]), scope=SCOPE, decision_at=DECISION_AT
    )

    assert built is not None
    assert [cluster.label for cluster in built.clusters] == [
        "beta ritual",
        "alpha ritual",
    ]


def test_cluster_mean_uses_fsum_in_original_residual_order(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpha = [
        make_node(store, "alpha early", {"procedure_id": "alpha-ritual"}),
        make_node(store, "alpha late", {"procedure_id": "alpha-ritual"}),
    ]
    beta = make_node(store, "beta", {"procedure_id": "beta-ritual"})
    scores = {alpha[0].id: 5.0, alpha[1].id: 7.0, beta.id: 6.0}
    calls: list[list[float]] = []
    real_fsum = __import__("math").fsum
    monkeypatch.setattr(
        recall_map,
        "relevance_score",
        lambda node, _history, _result=None: scores[node.id],
    )

    def observed_fsum(values) -> float:
        materialized = list(values)
        calls.append(materialized)
        return real_fsum(materialized)

    monkeypatch.setattr(recall_map, "fsum", observed_fsum)
    built = RecallMapBuilder(store).build(
        pool([alpha[0], beta, alpha[1]]), scope=SCOPE, decision_at=DECISION_AT
    )

    assert built is not None
    assert [5.0, 7.0] in calls


def test_equal_cluster_scores_use_unique_best_original_ordinal(
    store: MemoryStore,
) -> None:
    beta = make_node(store, "beta", {"procedure_id": "beta-ritual"})
    alpha = make_node(store, "alpha", {"procedure_id": "alpha-ritual"})

    built = RecallMapBuilder(store).build(
        pool([beta, alpha]), scope=SCOPE, decision_at=DECISION_AT
    )

    assert built is not None
    assert [cluster.label for cluster in built.clusters] == [
        "beta ritual",
        "alpha ritual",
    ]


def test_structural_keys_that_normalize_alike_still_get_distinct_labels(
    store: MemoryStore,
) -> None:
    """``type: root-cause`` and ``lesson_kind: root_cause`` are two clusters."""

    nodes = [
        make_node(store, "a note about the failing deploy", {"type": "root-cause"}),
        make_node(store, "a note about the filling disk", {"lesson_kind": "root_cause"}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    labels = [cluster.label for cluster in built.clusters]
    assert labels == ["root cause deploy", "root cause"]
    assert [cluster.count for cluster in built.clusters] == [1, 1]
    # Not merged: each keeps its own member, and its own way to ask.
    assert [cluster.member_ids for cluster in built.clusters] == [
        (nodes[0].id,),
        (nodes[1].id,),
    ]
    # The *label* took the distinguishing term; the ask-hint did not follow it
    # there. A third token is where ``_echoes`` starts demanding two shared
    # tokens instead of one, so a hint that tracked the whole label would
    # tighten the curtail rule as a side effect of a naming fix.
    assert built.clusters[0].ask_hint == "root cause"
    assert len(built.clusters[0].ask_hint.split()) <= ASK_HINT_MAX_TOKENS


def test_vector_regions_sharing_their_top_terms_still_get_distinct_labels(
    store: MemoryStore,
) -> None:
    """Two regions, one vocabulary: the label must name the difference.

    Contents differ only by a trailing digit, which the term extractor drops,
    so both clusters rank exactly the same terms — the collision the c-TF-IDF
    stage can genuinely produce.
    """

    nodes = [make_node(store, f"recall path note caching {index}") for index in range(4)]
    embed(store, nodes[0], [1.0, 0.0, 0.0])
    embed(store, nodes[1], [1.0, 0.02, 0.0])
    embed(store, nodes[2], [0.0, 0.0, 1.0])
    embed(store, nodes[3], [0.0, 0.02, 1.0])

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    labels = [cluster.label for cluster in built.clusters]
    assert labels == ["caching note path", "caching note path recall"]
    assert [cluster.count for cluster in built.clusters] == [2, 2]
    # Extended with a real term, not renumbered into meaninglessness.
    assert "#" not in labels[1]
    assert built.clusters[1].member_ids == (nodes[2].id, nodes[3].id)


def test_anchor_queries_sharing_a_long_prefix_stay_distinguishable(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uniqueness is decided on the label the reader actually sees.

    Anchors work *because* one operator's questions share a house style, so two
    anchor queries agreeing for their first forty characters is the expected
    case, not a pathological one.
    """

    monkeypatch.setattr(recall_map, "MAX_RESPONSE_CHARS", 10_000)
    first = make_node(store, "a")
    second = make_node(store, "s")
    prefix = "what did retrieval weights decide about "
    seed_anchor(store, prefix + "august", [first])
    seed_anchor(store, prefix + "september", [second])

    built = RecallMapBuilder(store).build(pool([first, second]), scope=SCOPE)

    assert built is not None
    labels = [cluster.label for cluster in built.clusters]
    assert len(labels) == len(set(labels)) == 2
    assert all(len(label) <= MAX_LABEL_CHARS for label in labels)
    # The ask-hint is the anchor's own wording, untouched by disambiguation.
    hints = sorted(cluster.ask_hint for cluster in built.clusters)
    assert hints == [
        "what did retrieval weights decide about august",
        "what did retrieval weights decide about september",
    ]


def test_a_cached_build_agrees_with_a_cold_one_on_embedding_clusters(
    store: MemoryStore,
) -> None:
    """Warm stage 4 revalidates vectors and recomputes the same medoid."""

    outlier = make_node(store, "an outlying note about caching layers")
    centre = make_node(store, "a central note about caching layers")
    partner = make_node(store, "another note about caching layers")
    embed(store, outlier, [1.0, 0.9, 0.0])
    embed(store, centre, [1.0, 0.0, 0.0])
    embed(store, partner, [1.0, 0.05, 0.0])
    candidates = pool([outlier, centre, partner])

    builder = RecallMapBuilder(store)
    cold = builder.build(candidates, scope=SCOPE, task="medoid")
    warm = builder.build(candidates, scope=SCOPE, task="medoid")

    assert builder.last_cache_hit is True
    assert cold == warm
    assert cold is not None
    assert cold.clusters[0].medoid.node_id == partner.id  # not the top-ranked one


def test_relevance_reorder_invalidates_a_greedy_embedding_template(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = make_node(store, "left vector deployment evidence")
    right = make_node(store, "right vector rollback evidence")
    bridge = make_node(store, "bridge vector transition evidence")
    embed(store, left, [1.0, 0.0])
    embed(store, right, [0.0, 1.0])
    embed(store, bridge, [1.0, 1.0])
    histories = {
        left.id: MaturedRecallHistory.known(500, 250, 250),
        right.id: MaturedRecallHistory.known(200, 100, 100),
        bridge.id: MaturedRecallHistory.known(100, 50, 50),
    }
    monkeypatch.setattr(
        store,
        "matured_recall_history",
        lambda ids, _at: {node_id: histories[node_id] for node_id in ids},
    )
    candidates = pool([left, right, bridge])
    builder = RecallMapBuilder(store)
    first = builder.build(
        candidates, scope=SCOPE, task="embedding-order", decision_at=DECISION_AT
    )

    assert first is not None
    assert [cluster.count for cluster in first.clusters] == [2, 1]

    histories[bridge.id] = MaturedRecallHistory.known(1000, 500, 500)
    rebuilt = builder.build(
        candidates, scope=SCOPE, task="embedding-order", decision_at=DECISION_AT
    )
    cold = RecallMapBuilder(store).build(
        candidates, scope=SCOPE, task="embedding-order", decision_at=DECISION_AT
    )

    assert builder.last_cache_hit is False
    assert rebuilt is not None and cold is not None
    assert [cluster.count for cluster in rebuilt.clusters] == [3]
    assert payload_bytes(rebuilt) == payload_bytes(cold)


def test_distinct_structural_keys_do_not_collapse_into_one_cluster(store: MemoryStore) -> None:
    """The degenerate-map guard: three keys must stay three clusters."""

    nodes = [
        make_node(store, "one", {"procedure_id": "alpha-ritual"}),
        make_node(store, "two", {"procedure_id": "beta-ritual"}),
        make_node(store, "three", {"procedure_id": "gamma-ritual"}),
        make_node(store, "four", {"procedure_id": "gamma-ritual"}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert len(built.clusters) == 3
    assert {cluster.label for cluster in built.clusters} == {
        "alpha ritual",
        "beta ritual",
        "gamma ritual",
    }
    assert sum(cluster.count for cluster in built.clusters) == 4


# ----------------------------------------------------------------------
# Caps
# ----------------------------------------------------------------------


def test_map_is_capped_in_clusters_and_in_characters(store: MemoryStore) -> None:
    nodes = [
        make_node(
            store,
            f"a long note about subsystem {index}: " + "content that runs on and on " * 8,
            {"topic": f"subsystem-{index}-work"},
        )
        for index in range(12)
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert len(built.clusters) <= MAX_CLUSTERS
    assert payload_chars(built) <= MAX_RESPONSE_CHARS
    assert len(built.render_compact()) <= MAX_INSTRUCTIONS_CHARS
    assert built.render_compact().startswith("memory also holds: ")
    assert built.to_dict()["pool"] == 12
    assert list(built.to_dict()["clusters"][0]) == [
        "label",
        "count",
        "medoid",
        "ask_hint",
        "plan_item",
    ]
    # Twelve keys, at most six clusters, and the budget takes more: what was
    # dropped is reported rather than passed off as everything memory holds.
    assert built.dropped == 12 - len(built.clusters)
    assert built.to_dict()["more"] == built.dropped


def test_examples_are_preserved_before_a_cluster_is_dropped(store: MemoryStore) -> None:
    """Two affordable clusters retain informative bounded examples."""

    long_tail = "content that runs on and on " * 6
    nodes = [
        make_node(store, f"note {index}: {long_tail}", {"topic": f"area-{name}"})
        for index, name in enumerate(("alpha", "beta"))
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert len(built.clusters) == 2
    assert built.dropped == 0
    assert payload_chars(built) <= MAX_RESPONSE_CHARS
    # Every example survived, bounded rather than discarded.
    examples = [cluster.medoid.example for cluster in built.clusters]
    assert all(example.endswith("…") for example in examples)
    assert all(
        MIN_MEDOID_EXAMPLE_CHARS <= len(example) <= MEDOID_EXAMPLE_CHARS
        for example in examples
    )


def _rich_gist_pool(store: MemoryStore) -> list[Node]:
    """A deterministic pool whose labels and medoids make the 700-char cap bind."""

    topics = (
        "cobalt-handoff",
        "saffron-checkpoint",
        "indigo-attestation",
        "marble-retention",
        "quartz-snapshot",
        "willow-recovery",
        "zephyr-latency",
        "ember-consumption",
    )
    return [
        make_node(
            store,
            f"{topic} preserves a detailed operational account: "
            + "plain material for the medoid example " * 8,
            {"topic": topic},
        )
        for topic in topics
    ]


def test_every_delivered_cluster_carries_an_informative_example(
    store: MemoryStore,
) -> None:
    built = RecallMapBuilder(store).build(
        pool(_rich_gist_pool(store)), scope=SCOPE, task="gist-present"
    )

    assert built is not None
    assert len(built.clusters) == MIN_CLUSTERS
    assert all(
        len(cluster.medoid.example) >= MIN_MEDOID_EXAMPLE_CHARS
        for cluster in built.clusters
    )
    assert all(cluster.medoid.example.endswith("…") for cluster in built.clusters)


def test_example_budget_reflows_from_scratch_after_cluster_drops(
    store: MemoryStore,
) -> None:
    candidates = pool(_rich_gist_pool(store))

    reflowed = RecallMapBuilder(store).build(candidates, scope=SCOPE, task="reflow")
    direct = RecallMapBuilder(store, max_clusters=MIN_CLUSTERS).build(
        candidates, scope=SCOPE, task="reflow"
    )

    assert reflowed is not None and direct is not None
    assert reflowed.dropped == direct.dropped == len(candidates) - MIN_CLUSTERS
    assert reflowed.to_dict() == direct.to_dict()
    assert all(cluster.medoid.example for cluster in reflowed.clusters)
    journal = reflowed.to_dict()["filtered"]
    assert journal["dropped"] == reflowed.dropped
    assert journal.get("names_omitted", 0) + len(journal.get("names", [])) == (
        reflowed.dropped
    )


def test_rich_pool_uses_nearly_the_whole_response_budget(store: MemoryStore) -> None:
    built = RecallMapBuilder(store).build(
        pool(_rich_gist_pool(store)), scope=SCOPE, task="budget-utilization"
    )

    assert built is not None
    unused = MAX_RESPONSE_CHARS - payload_chars(built)
    assert 0 <= unused < len(built.clusters)


def test_selector_samples_yield_before_existing_cluster_content(
    store: MemoryStore,
) -> None:
    header = {
        "path": "src/generated.py",
        "kind": "source",
        "language": "python",
        "sha256": "a" * 64,
        "chunk": "1/1",
        "lines": "1-20",
    }
    ballast = make_node(store, "[file-chunk] " + json.dumps(header))
    built = RecallMapBuilder(store).build(
        pool([ballast, *_rich_gist_pool(store)]),
        scope=SCOPE,
        task="selector-budget",
        decision_at=DECISION_AT,
    )

    assert built is not None
    payload = built.to_dict()
    assert payload_chars(built) <= MAX_RESPONSE_CHARS
    assert len(payload["clusters"]) == MIN_CLUSTERS
    assert payload["sel"]["x"] == [0, 0, 1, 0, 0, 0, 0]
    assert payload["sel"]["o"] == 1
    assert "q" not in payload["sel"]


def test_gist_floor_fitting_is_deterministic_cold_and_warm(store: MemoryStore) -> None:
    candidates = pool(_rich_gist_pool(store))
    warm_builder = RecallMapBuilder(store)

    first = warm_builder.build(candidates, scope=SCOPE, task="gist-determinism")
    second = warm_builder.build(candidates, scope=SCOPE, task="gist-determinism")
    cold = RecallMapBuilder(store).build(
        candidates, scope=SCOPE, task="gist-determinism"
    )

    assert first is not None and second is not None and cold is not None
    assert warm_builder.last_cache_hit is True
    assert first.to_dict() == second.to_dict() == cold.to_dict()


def test_a_map_that_cannot_afford_the_gist_floor_is_journal_only(
    store: MemoryStore,
) -> None:
    """Long proven hints do not buy breadth by shipping sub-floor gists."""

    nodes = [
        make_node(
            store,
            f"medoid {index} carries enough explanatory material " + "detail " * 16,
        )
        for index in range(2)
    ]
    for index, node in enumerate(nodes):
        seed_anchor(
            store,
            " ".join(
                [
                    f"anchor{index}",
                    "production",
                    "equivalence",
                    "verification",
                    "boundary",
                    "evidence",
                    "handoff",
                    "protocol",
                ]
            ),
            [node],
        )

    builder = RecallMapBuilder(store)
    built = builder.build(pool(nodes), scope=SCOPE, task="unaffordable-floor")

    assert built is not None
    assert built.clusters == ()
    assert built.dropped == 2
    assert built.withheld == 0
    assert built.render_compact() == ""
    assert payload_chars(built) <= MAX_RESPONSE_CHARS
    assert built.to_dict()["filtered"] == {
        "withheld": 0,
        "dropped": 2,
        "names": [
            ["d", "anchor0 production equi…", 1],
            ["d", "anchor1 production equi…", 1],
        ],
    }
    assert builder.last_curtailment.offers == 0


def test_a_map_with_nothing_to_say_is_no_map_at_all(store: MemoryStore) -> None:
    assert RecallMapBuilder(store).build([], scope=SCOPE) is None


def test_plan_items_are_transferable_todo_lines(store: MemoryStore) -> None:
    nodes = [
        make_node(store, "one", {"topic": "deploy-recipe"}),
        make_node(store, "two", {"topic": "deploy-recipe"}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert built.plan_items() == ["on touching deploy recipe - recall 'deploy recipe' (2)"]


# ----------------------------------------------------------------------
# Cache: same key, same structure, until the corpus moves
# ----------------------------------------------------------------------


def test_cached_structure_survives_a_changed_pool_and_refreshes_counts(
    store: MemoryStore,
) -> None:
    nodes = [
        make_node(store, "alpha one", {"procedure_id": "alpha-ritual"}),
        make_node(store, "alpha two", {"procedure_id": "alpha-ritual"}),
        make_node(store, "alpha three", {"procedure_id": "alpha-ritual"}),
        make_node(store, "beta one", {"procedure_id": "beta-ritual"}),
    ]
    builder = RecallMapBuilder(store)

    first = builder.build(pool(nodes), scope=SCOPE, task="postsession extraction")
    assert first is not None
    assert builder.last_cache_hit is False

    # A later recall for the same task surfaces a different slice of the pool.
    second = builder.build(pool(nodes[1:]), scope=SCOPE, task="Postsession-Extraction")
    assert second is not None
    assert builder.last_cache_hit is True
    assert builder.cache_size == 1  # the two task spellings fold to one key

    # Same labels in the same order; counts follow the pool in hand.
    assert [c.label for c in first.clusters] == [c.label for c in second.clusters]
    assert [c.ask_hint for c in first.clusters] == [c.ask_hint for c in second.clusters]
    assert [c.count for c in first.clusters] == [3, 1]
    assert [c.count for c in second.clusters] == [2, 1]


def test_cached_structure_refreshes_order_from_the_current_residual(
    store: MemoryStore,
) -> None:
    alpha = make_node(store, "alpha head", {"procedure_id": "alpha-ritual"})
    beta = [
        make_node(store, "beta head", {"procedure_id": "beta-ritual"}),
        make_node(store, "beta tail", {"procedure_id": "beta-ritual"}),
    ]
    builder = RecallMapBuilder(store)

    first = builder.build(
        pool([alpha, *beta]), scope=SCOPE, task="rank-refresh"
    )
    assert first is not None
    assert builder.last_cache_hit is False

    second_candidates = pool([beta[0], alpha, beta[1]])
    second = builder.build(second_candidates, scope=SCOPE, task="rank-refresh")
    assert second is not None
    assert builder.last_cache_hit is True

    # The templates retain their labels, hints and medoid choices, but their
    # presentation order follows this call's ranks: beta moves from ranks 1/2
    # (mass 5/6) to ranks 0/2 (mass 4/3).
    assert [cluster.label for cluster in first.clusters] == [
        "alpha ritual",
        "beta ritual",
    ]
    assert [cluster.label for cluster in second.clusters] == [
        "beta ritual",
        "alpha ritual",
    ]
    first_structure = {
        cluster.label: (cluster.ask_hint, cluster.medoid.node_id)
        for cluster in first.clusters
    }
    second_structure = {
        cluster.label: (cluster.ask_hint, cluster.medoid.node_id)
        for cluster in second.clusters
    }
    assert first_structure == second_structure
    assert {cluster.label: cluster.count for cluster in second.clusters} == {
        "alpha ritual": 1,
        "beta ritual": 2,
    }

    cold = RecallMapBuilder(store).build(
        second_candidates, scope=SCOPE, task="rank-refresh"
    )
    assert cold is not None
    assert [cluster.label for cluster in cold.clusters] == [
        cluster.label for cluster in second.clusters
    ]


def test_cached_build_recomputes_history_and_selection_every_time(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpha = make_node(store, "alpha", {"procedure_id": "alpha-ritual"})
    beta = make_node(store, "beta", {"procedure_id": "beta-ritual"})
    histories = {
        alpha.id: MaturedRecallHistory.known(200, 100, 100),
        beta.id: MaturedRecallHistory.known(100, 50, 50),
    }
    reads: list[tuple[tuple[str, ...], str]] = []

    def read(candidate_ids, decision_at):
        ids = tuple(candidate_ids)
        reads.append((ids, str(decision_at)))
        return {node_id: histories[node_id] for node_id in ids}

    monkeypatch.setattr(store, "matured_recall_history", read)
    candidates = pool([alpha, beta])
    builder = RecallMapBuilder(store)
    first = builder.build(
        candidates, scope=SCOPE, task="fresh-history", decision_at=DECISION_AT
    )
    assert first is not None
    assert [cluster.label for cluster in first.clusters] == [
        "alpha ritual",
        "beta ritual",
    ]

    histories[alpha.id] = MaturedRecallHistory.known(0, 0, 0)
    warm = builder.build(
        candidates, scope=SCOPE, task="fresh-history", decision_at=DECISION_AT
    )
    cold = RecallMapBuilder(store).build(
        candidates, scope=SCOPE, task="fresh-history", decision_at=DECISION_AT
    )

    assert builder.last_cache_hit is True
    assert warm is not None and cold is not None
    assert payload_bytes(warm) == payload_bytes(cold)
    assert warm.pool_size == 1
    assert [cluster.label for cluster in warm.clusters] == ["beta ritual"]
    assert warm.to_dict()["sel"]["x"] == [0, 0, 0, 0, 0, 1, 0]
    assert len(reads) == 3
    assert all(instant == DECISION_AT for _ids, instant in reads)


def test_a_different_key_does_not_share_a_structure(store: MemoryStore) -> None:
    nodes = [make_node(store, "one", {"procedure_id": "alpha-ritual"})]
    builder = RecallMapBuilder(store)

    builder.build(pool(nodes), scope=SCOPE, task="first task")
    builder.build(pool(nodes), scope=SCOPE, task="second task")

    assert builder.last_cache_hit is False
    assert builder.cache_size == 2
    assert cache_key(SCOPE, task="first task") != cache_key(SCOPE, task="second task")


def test_corpus_movement_invalidates_the_cache_without_a_timer(store: MemoryStore) -> None:
    nodes = [
        make_node(store, "one about caching layers", {"procedure_id": "alpha-ritual"}),
        make_node(store, "two about caching layers", {"procedure_id": "alpha-ritual"}),
    ]
    builder = RecallMapBuilder(store)
    candidates = pool(nodes)

    builder.build(candidates, scope=SCOPE, task="caching")
    builder.build(candidates, scope=SCOPE, task="caching")
    assert builder.last_cache_hit is True

    # Writing a chunk moves the chunk-table revision, which is exactly what the
    # cached structure was validated against.
    embed(store, nodes[0], [1.0, 0.0, 0.0])

    builder.build(candidates, scope=SCOPE, task="caching")
    assert builder.last_cache_hit is False


def test_policy_digest_change_invalidates_even_an_unchanged_corpus(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = make_node(store, "one", {"procedure_id": "policy-bound-ritual"})
    candidates = pool([node])
    builder = RecallMapBuilder(store)

    first = builder.build(
        candidates, scope=SCOPE, task="policy-revision", decision_at=DECISION_AT
    )
    second = builder.build(
        candidates, scope=SCOPE, task="policy-revision", decision_at=DECISION_AT
    )
    assert first == second
    assert builder.last_cache_hit is True

    changed_digest = "0" * 64
    monkeypatch.setattr(recall_map, "RELEVANCE_POLICY_DIGEST", changed_digest)
    rebuilt = builder.build(
        candidates, scope=SCOPE, task="policy-revision", decision_at=DECISION_AT
    )

    assert rebuilt == first
    assert builder.last_cache_hit is False
    key = cache_key(SCOPE, task="policy-revision")
    assert builder._cache[key].revision[0] == changed_digest


def test_anchor_movement_invalidates_the_cache(store: MemoryStore) -> None:
    nodes = [make_node(store, "one", {"procedure_id": "alpha-ritual"})]
    builder = RecallMapBuilder(store)
    candidates = pool(nodes)

    builder.build(candidates, scope=SCOPE, task="anchors")
    builder.build(candidates, scope=SCOPE, task="anchors")
    assert builder.last_cache_hit is True

    seed_anchor(store, "a brand new question", nodes)

    builder.build(candidates, scope=SCOPE, task="anchors")
    assert builder.last_cache_hit is False


def test_cache_refuses_to_describe_a_pool_it_no_longer_covers(store: MemoryStore) -> None:
    """Stability is worth having only while it is stability about this pool."""

    first_pool = [
        make_node(store, "alpha one", {"procedure_id": "alpha-ritual"}),
        make_node(store, "alpha two", {"procedure_id": "alpha-ritual"}),
    ]
    builder = RecallMapBuilder(store)
    builder.build(pool(first_pool), scope=SCOPE, task="drift")

    second_pool = [
        make_node(store, "omega one", {"procedure_id": "omega-ritual"}),
        make_node(store, "omega two", {"procedure_id": "omega-ritual"}),
        make_node(store, "omega three", {"procedure_id": "omega-ritual"}),
    ]
    rebuilt = builder.build(pool(second_pool), scope=SCOPE, task="drift")

    assert builder.last_cache_hit is False
    assert rebuilt is not None
    assert [cluster.label for cluster in rebuilt.clusters] == ["omega ritual"]


def test_cached_structural_template_admits_a_node_it_never_saw(store: MemoryStore) -> None:
    """Stages 1 and 2 re-derive membership; they do not replay an id list."""

    seen = [
        make_node(store, "alpha one", {"procedure_id": "alpha-ritual"}),
        make_node(store, "alpha two", {"procedure_id": "alpha-ritual"}),
    ]
    builder = RecallMapBuilder(store)
    builder.build(pool(seen), scope=SCOPE, task="admit")

    newcomer = make_node(store, "alpha three", {"procedure_id": "alpha-ritual"})
    refitted = builder.build(pool([*seen, newcomer]), scope=SCOPE, task="admit")

    assert builder.last_cache_hit is True
    assert refitted is not None
    assert [cluster.count for cluster in refitted.clusters] == [3]
    assert newcomer.id in refitted.clusters[0].member_ids


def test_cached_path_template_cannot_steal_a_new_structural_cluster(
    store: MemoryStore,
) -> None:
    """Cache recount preserves the cascade's structural-before-path rule."""

    path_context = {"files": ["src/living_memory/recall_map.py"]}
    path_nodes = [
        make_node(store, f"path cache evidence {index}", path_context)
        for index in range(4)
    ]
    builder = RecallMapBuilder(store)
    first = builder.build(pool(path_nodes), scope=SCOPE, task="stage-precedence")
    assert first is not None
    assert builder.last_cache_hit is False
    assert {cluster.stage for cluster in first.clusters} == {STAGE_PATH}

    newcomer = make_node(
        store,
        "a structural lesson with the same file path",
        {**path_context, "lesson_kind": "production-equivalence-gap"},
    )
    rebuilt = builder.build(
        pool([*path_nodes, newcomer]), scope=SCOPE, task="stage-precedence"
    )

    # A path-only cached shape is not allowed to absorb the new stage-1
    # signature. Rebuilding recovers the name rather than silently omitting or
    # renaming the node after ``living_memory``.
    assert builder.last_cache_hit is False
    assert rebuilt is not None
    by_stage = {cluster.stage: cluster for cluster in rebuilt.clusters}
    assert newcomer.id in by_stage[STAGE_STRUCTURAL].member_ids
    assert newcomer.id not in by_stage[STAGE_PATH].member_ids


# ----------------------------------------------------------------------
# The delivery gate: a corpus statistic, and what it withholds
# ----------------------------------------------------------------------


def house_corpus(store: MemoryStore, word: str, count: int = CORPUS_DOCUMENTS) -> None:
    """A corpus whose every document carries ``word``: its house vocabulary."""

    for index in range(count):
        make_node(store, f"{word} routine {index} about scheduled maintenance")


def test_a_cluster_the_cascade_could_not_name_is_withheld(bare_store: MemoryStore) -> None:
    """Silence is cheaper than a row nobody can act on.

    ``widget`` is in every document of this corpus, so a cluster named after it
    says nothing that distinguishes it from the corpus — and a row like that
    costs a slot of a channel with a handful of them.
    """

    store = bare_store
    house_corpus(store, "widget")
    nodes = [
        make_node(store, f"widget upkeep {index}", {"topic": "widget"})
        for index in range(3)
    ]
    nodes.append(
        make_node(store, "the sealed holdout attestation", {"topic": "sealed-holdout"})
    )

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="gate")

    assert built is not None
    # The house word never reaches the wire, under any spelling and under no
    # catch-all label: it is removed, not renamed.
    assert [cluster.label for cluster in built.clusters] == ["sealed holdout"]
    assert all("widget" not in cluster.label for cluster in built.clusters)
    assert built.withheld == 1
    # ``covered`` still counts only what was delivered, and ``more`` still
    # means only what the budget dropped.
    assert built.covered == 1
    assert built.dropped == 0


def test_the_gate_is_a_corpus_statistic_not_a_word_list(
    bare_store: MemoryStore, tmp_path: Path
) -> None:
    """Two corpora, the same two labels, opposite verdicts.

    This is the property an enumerated blacklist cannot have, and it is the
    whole reason the gate is a statistic. Nothing in ``recall_map`` names
    ``widget`` or ``шестерёнка``. Each corpus withholds whichever of the two
    labels is built from *its own* house word and delivers the other, so the
    rule that was placed on one field corpus transfers to a corpus whose house
    vocabulary is different words — in a different script.
    """

    def verdicts(store: MemoryStore, house: str) -> set[str]:
        house_corpus(store, house)
        # Content that offers the enrichment nothing the key does not already
        # say, so each cluster reaches the gate on its key's own two words and
        # the verdict is the gate's alone.
        nodes = [
            make_node(store, f"widget cadence {index}", {"topic": "widget-cadence"})
            for index in range(2)
        ]
        nodes += [
            make_node(store, f"шестерёнка tempo {index}", {"topic": "шестерёнка-tempo"})
            for index in range(2)
        ]
        built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="transfer")
        return {cluster.label for cluster in built.clusters} if built else set()

    latin = verdicts(bare_store, "widget")
    with MemoryStore(tmp_path / "other-corpus.sqlite3") as other:
        other.matured_recall_history = eligible_history  # type: ignore[method-assign]
        cyrillic = verdicts(other, "шестерёнка")

    assert latin == {"шестерёнка tempo"}
    assert cyrillic == {"widget cadence"}


def test_no_observed_label_is_written_into_the_module() -> None:
    """The gate may not become a lookup table of what the field happened to show.

    Docstrings quote the field's degenerate labels because that is the evidence
    for the rule; *code* that compared against them would be a blacklist wearing
    a threshold, and would not transfer to the next corpus. So the check is on
    the module's real string constants, with every docstring removed first.
    """

    import ast

    source = (
        Path(__file__).resolve().parents[1] / "src" / "living_memory" / "recall_map.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is not None:
                node.body = node.body[1:]

    literals = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    observed = {"public", "test", "mmo", "server", "tools", "camera", "showcase"}
    # The one legitimate overlap is the path-container list stage 2 reads
    # directory names against, which predates this rule and decides where a
    # subsystem *starts*, not whether a label may be delivered.
    assert literals & observed <= recall_map._PATH_ROOT_SEGMENTS, sorted(
        (literals & observed) - recall_map._PATH_ROOT_SEGMENTS
    )


def test_a_pool_the_gate_empties_is_journaled_without_counting_an_offer(
    bare_store: MemoryStore,
) -> None:
    """A fully withheld pool persists why it is silent, without curtail cost."""

    store = bare_store
    house_corpus(store, "widget")
    nodes = [
        make_node(store, f"widget upkeep {index}", {"topic": "widget"})
        for index in range(3)
    ]

    builder = RecallMapBuilder(store)
    built = builder.build(pool(nodes), scope=SCOPE, task="silent")
    assert built is not None
    assert built.clusters == ()
    assert built.render_compact() == ""
    assert built.to_dict() == {
        "clusters": [],
        "pool": 3,
        "covered": 0,
        "filtered": {
            "withheld": 1,
            "dropped": 0,
            "names": [["w", "widget upkeep", 3]],
        },
        "sel": {
            "v": "r1",
            "n": 3,
            "e": 3,
            "x": [0, 0, 0, 0, 0, 0, 0],
            "o": 0,
        },
    }

    event = store.record_recall_event(
        query="what else is here",
        scope=SCOPE,
        ambient_context={"task": "silent"},
        recall_map=built.to_dict(),
    )
    (persisted,) = store.recent_recall_map_history(scope=SCOPE, task="silent")
    assert persisted["id"] == event.id
    assert persisted["recall_map"] == built.to_dict()

    # Journal-only payloads are observable but not offers: a key that keeps
    # going quiet never curtails itself into a collapse it cannot leave.
    for _ in range(4):
        again = builder.build(pool(nodes), scope=SCOPE, task="silent")
        assert again is not None and again.clusters == ()
    assert builder.last_curtailment.offers == 0


def test_without_corpus_statistics_the_gate_degrades_to_two_terms(
    bare_store: MemoryStore,
) -> None:
    """A store where "generic" is undefined admits a weak label rather than none.

    The empty store has no document frequencies to measure against, so the
    ``ic`` scale would call every label maximally rare or maximally common
    depending on which way the missing index is read. Withholding everything on
    a missing index is the worse failure, so what is left is the structural
    half of the predicate: two distinct content terms.
    """

    store = bare_store
    only = make_node(store, "the note", {"topic": "sealed-holdout"})

    builder = RecallMapBuilder(store)
    assert builder._document_count() <= 1
    assert builder._deliverable("sealed holdout") is True
    assert builder._deliverable("widget") is False
    assert builder._deliverable("") is False

    built = builder.build(pool([only]), scope=SCOPE, task="degraded")
    assert built is not None
    assert [cluster.label for cluster in built.clusters] == ["sealed holdout"]


def test_a_store_that_cannot_answer_the_index_still_builds_a_map(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old schema is not a reason to go quiet."""

    def unavailable(*_args, **_kwargs):
        raise sqlite3.OperationalError("no such table: nodes_fts_vocab")

    monkeypatch.setattr(type(store), "term_document_frequencies", unavailable)

    nodes = [
        make_node(store, "the first note", {"topic": "sealed-holdout"}),
        make_node(store, "the second note", {"topic": "sealed-holdout"}),
    ]
    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="old-store")

    assert built is not None
    assert [cluster.label for cluster in built.clusters] == ["sealed holdout"]


def test_a_house_word_structural_key_is_enriched_rather_than_lost(
    bare_store: MemoryStore,
) -> None:
    """Stage 1 gets the same rescue stage 2 gets, and only when it needs it.

    Six of the fifty-five ``server`` clusters in the field window were
    structural, not path: a stored key whose value is one house word is exactly
    as undeliverable as a bare directory segment. A key that already says
    something is left exactly alone.
    """

    store = bare_store
    house_corpus(store, "widget")
    generic = [
        make_node(store, "widget rotation drills the arm", {"procedure_id": "widget"}),
        make_node(store, "widget rotation seals the arm", {"procedure_id": "widget"}),
    ]
    specific = [
        make_node(store, "the ledger reconciles", {"procedure_id": "sealed-holdout-drill"}),
    ]

    built = RecallMapBuilder(store).build(
        pool(generic + specific), scope=SCOPE, task="structural-enrichment"
    )

    assert built is not None
    by_count = {cluster.count: cluster for cluster in built.clusters}
    enriched = by_count[2]
    assert enriched.label.startswith("widget ")
    assert len(enriched.label.split()) >= 2
    # Rescued rather than withheld: enrichment runs first, the gate is the net.
    assert built.withheld == 0
    # And the key that already said something keeps its own wording.
    assert by_count[1].label == "sealed holdout drill"
    assert by_count[1].ask_hint == "sealed holdout drill"


def test_enriched_ask_hints_still_clear_the_curtail_echo(bare_store: MemoryStore) -> None:
    """Richer labels must not tighten a pre-registered rule as a side effect.

    ``_echoes`` needs ``ceil(n/2)`` shared tokens for an ``n``-token phrasing,
    so one and two tokens ask for the same evidence and three asks for more.
    The enriched hint's tokens are a *superset* of the bare subsystem's, and it
    is still at most two tokens, so a query that cleared the old phrasing
    clears the new one by construction.
    """

    store = bare_store
    house_corpus(store, "widget")
    nodes = [
        make_node(
            store,
            f"the ledger reconciles checkpoint {index}",
            {"files": [f"src/living_memory/postsession/step{index}.py"]},
        )
        for index in range(3)
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="echo")

    assert built is not None
    (cluster,) = built.clusters
    bare = frozenset({"postsession"})
    enriched = token_set(cluster.ask_hint)
    assert len(cluster.ask_hint.split()) <= ASK_HINT_MAX_TOKENS
    assert bare <= enriched
    # Every query that used to clear the bare subsystem still clears the hint.
    for query in ("what did postsession decide", "postsession"):
        assert _echoes(bare, token_set(query))
        assert _echoes(enriched, token_set(query))


# ----------------------------------------------------------------------
# The filter journal: caps are allowed, being quiet about them is not
# ----------------------------------------------------------------------


def test_withheld_and_dropped_clusters_are_journaled_by_name_and_count(
    bare_store: MemoryStore,
) -> None:
    """Both filters report, and they report separately.

    "The memory had nothing to say" and "the channel was full" are opposite
    findings that both look like a short map from the outside, so the payload
    carries one number for each and never folds them together.
    """

    store = bare_store
    house_corpus(store, "widget")
    nodes: list[Node] = []
    for index in range(3):  # withheld: named after the corpus's house word
        nodes.append(make_node(store, f"widget upkeep {index}", {"topic": "widget"}))
    for index in range(8):  # deliverable, and more than the caps can carry
        nodes.append(
            make_node(
                store,
                f"sealed holdout attestation {index}: " + "material that runs on " * 8,
                {"topic": f"holdout-a{index}"},
            )
        )

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="journal")

    assert built is not None
    payload = built.to_dict()
    block = payload["filtered"]

    assert block["withheld"] == 1
    assert block["dropped"] == built.dropped > 0
    # ``dropped`` mirrors ``more`` so the block is self-contained; ``more``
    # itself keeps its exact old meaning and gains nothing from the gate.
    assert block["dropped"] == payload["more"]
    assert built.covered == sum(c["count"] for c in payload["clusters"])

    names = block["names"]
    assert 0 < len(names) <= FILTER_JOURNAL_NAMES
    # The gate's refusal leads, under the *enriched* name: stage 1 tried to
    # rescue the cluster and the terms it had to offer were house vocabulary
    # too, which is the case the gate exists for. Enrichment is the rescue, the
    # gate is the net, and the net reports what it caught.
    assert names[0] == ["w", "widget upkeep", 3]
    for tag, label, count in names:
        assert tag in ("w", "d")
        assert len(label) <= FILTER_JOURNAL_LABEL_CHARS
        assert count > 0
    # Names are convenience and give way to the budget; the count of what went
    # unnamed does not.
    assert block.get("names_omitted", 0) == block["withheld"] + block["dropped"] - len(names)
    assert block["names_omitted"] > 0


def test_a_filtered_cluster_never_appears_among_the_delivered_ones(
    bare_store: MemoryStore,
) -> None:
    """The hard rule the additive shape rests on.

    ``_delivered_items`` is paranoid about the *shape* of a cluster and not at
    all about extra ones, so a journal row inside ``clusters`` would be read as
    a delivered item by the curtail probe and as a novelty-consuming hint by
    the ae probe — on a cluster nobody was ever offered.
    """

    store = bare_store
    house_corpus(store, "widget")
    nodes = [make_node(store, f"widget upkeep {index}", {"topic": "widget"}) for index in range(2)]
    nodes.append(make_node(store, "the ledger reconciles", {"topic": "sealed-holdout"}))

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="never-inside")

    assert built is not None
    payload = built.to_dict()
    assert payload["filtered"]["withheld"] == 1
    assert len(payload["clusters"]) == 1
    for cluster in payload["clusters"]:
        assert set(cluster) == {"label", "count", "medoid", "ask_hint", "plan_item"}
        assert "widget" not in cluster["label"]
    assert built.render_compact() == "memory also holds: sealed holdout(1)"


def test_the_journal_block_is_absent_when_nothing_was_filtered(store: MemoryStore) -> None:
    """An unfiltered map adds only the mandatory selector accounting."""

    nodes = [
        make_node(store, "the first note", {"topic": "sealed-holdout"}),
        make_node(store, "the second note", {"topic": "anchor-bench-rotation"}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="unfiltered")

    assert built is not None
    assert built.withheld == 0 and built.dropped == 0
    assert list(built.to_dict()) == ["clusters", "pool", "covered", "sel"]


def test_the_journal_is_inside_the_response_budget(bare_store: MemoryStore) -> None:
    """The journal spends the same 700 characters the examples do."""

    store = bare_store
    house_corpus(store, "widget")
    nodes: list[Node] = [
        make_node(store, f"widget upkeep {index}", {"topic": f"widget-{'ab'[index % 2]}"})
        for index in range(4)
    ]
    for index in range(8):
        nodes.append(
            make_node(
                store,
                f"sealed holdout attestation {index}: " + "material that runs on " * 12,
                {"topic": f"sealed-holdout-chapter-{'ab'[index % 2]}{index}"},
            )
        )

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="budget")

    assert built is not None
    assert "filtered" in built.to_dict()
    assert payload_chars(built) <= MAX_RESPONSE_CHARS
    assert len(built.render_compact()) <= MAX_INSTRUCTIONS_CHARS


def test_the_journal_survives_persistence_and_read_back(bare_store: MemoryStore) -> None:
    """The additive fields are what the field measurement will actually read."""

    store = bare_store
    house_corpus(store, "widget")
    nodes = [
        make_node(store, "the ledger reconciles", {"topic": "sealed-holdout"}),
        make_node(store, "widget maintenance", {"topic": "widget"}),
    ]
    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE, task="persisted")
    assert built is not None
    assert built.withheld == 1

    event = store.record_recall_event(
        query="what else is here",
        scope=SCOPE,
        ambient_context={"task": "persisted"},
        recall_map=built.to_dict(),
    )
    (row,) = store.recent_recall_map_history(scope=SCOPE, task="persisted")
    assert row["id"] == event.id
    assert row["recall_map"] == built.to_dict()
    assert row["recall_map"]["filtered"] == {
        "withheld": 1,
        "dropped": 0,
        "names": [["w", "widget maintenance", 1]],
    }


def test_the_journal_changes_nothing_for_the_consumers_that_read_the_payload() -> None:
    """Additive means additive: every downstream reader is bit-identical.

    Checked against the two consumers this repository owns — the curtail
    probe's ``_delivered_items`` and the instructions channel's
    ``compose_map_section`` — because the third (the ae probe) reads the same
    ``clusters`` entries and reaching it would mean editing another repo.
    """

    from living_memory.instructions_map import _clusters_of, compose_map_section
    from living_memory.recall_map import _delivered_items

    plain = {
        "clusters": [
            {
                "label": "sealed holdout drill",
                "count": 4,
                "medoid": {"node_id": "01MEDOID", "example": "the note"},
                "ask_hint": "sealed holdout",
                "plan_item": "on touching sealed holdout drill - recall 'sealed holdout' (4)",
            }
        ],
        "pool": 40,
        "covered": 4,
        "more": 2,
    }
    journaled = {
        **plain,
        "filtered": {
            "withheld": 9,
            "dropped": 2,
            "names": [["w", "widget", 7], ["d", "anchor bench", 3]],
            "names_omitted": 5,
        },
    }

    assert _delivered_items(journaled) == _delivered_items(plain)
    assert _clusters_of(journaled) == _clusters_of(plain)
    rows = [{"recall_map": plain}], [{"recall_map": journaled}]
    assert compose_map_section(rows[1]) == compose_map_section(rows[0])
    assert compose_map_section(rows[0]) != ""


# ----------------------------------------------------------------------
# The mirrors, and the LLM-free contract
# ----------------------------------------------------------------------


def test_normalize_key_mirrors_consolidation_normalize_trigger() -> None:
    from living_memory.consolidation import _normalize_trigger

    for value in (
        "postsession-extraction",
        "per_role/recast-execution",
        "  Mixed_Case-Slug  ",
        "already words",
        "a3f19c8e77b204d5",
    ):
        assert normalize_key(value) == _normalize_trigger(value)


def test_chunk_revision_mirrors_the_retrieval_probe(store: MemoryStore) -> None:
    from living_memory.recall_map import _chunk_table_revision as mirrored
    from living_memory.retrieval import _chunk_table_revision as original

    node = make_node(store, "content that will gain a chunk")
    assert mirrored(store.connection) == original(store.connection)

    embed(store, node, [1.0, 0.0, 0.0])
    assert mirrored(store.connection) == original(store.connection)


def test_module_imports_nothing_that_calls_a_model_or_the_network() -> None:
    """The recall path may not grow an LLM or a socket by accident."""

    source = (
        Path(__file__).resolve().parents[1] / "src" / "living_memory" / "recall_map.py"
    ).read_text(encoding="utf-8")
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "living_memory" not in line
    ]
    assert imports  # the scan found something to check
    forbidden = (
        "anthropic",
        "openai",
        "httpx",
        "requests",
        "urllib",
        "socket",
        "http",
        "aiohttp",
        "sentence_transformers",
        "transformers",
        "torch",
    )
    for line in imports:
        assert not any(name in line for name in forbidden), line
    # The one local import that could pull a model in is not taken.
    assert "LocalEmbeddingModel" not in source


# ----------------------------------------------------------------------
# The frozen `sel` wire contract, guarded on the wire itself
# ----------------------------------------------------------------------
#
# `tests/test_recall_map_pool.py` guards the same contract one layer down, on
# the tuples and on `_SelectionLedger.freeze`. These guard what a *reader*
# actually receives: the `sel` block of `to_dict()`, which is what the server
# persists into `recall_events.recall_map` and what the field measurement read
# to conclude that no live server has either pool valve charged.
#
# A build is the only place the two halves meet -- a ledger that trims
# correctly still says nothing if the payload is assembled from somewhere else
# -- so these are end-to-end on purpose, and they cost one small store each.


@pytest.fixture(autouse=True)
def _valves_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file starts with no pool valve armed.

    Both live hosts now export the cold-quota pair into every login shell, so
    an inherited valve would make the seven-count guard below assert its own
    opposite.
    """

    for name in (
        POOL_USEFULNESS_GATE_ENV,
        POOL_USEFULNESS_FLOOR_ENV,
        POOL_DEMOTION_GATE_ENV,
        POOL_DEMOTION_WINDOWS_ENV,
        recall_map.POOL_COLD_QUOTA_GATE_ENV,
        recall_map.POOL_COLD_SLOTS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _rated_node(
    store: MemoryStore, content: str, key: str, usefulness: float
) -> Node:
    """A pool member carrying an explicit usefulness verdict.

    ``make_node`` folds its extras into ``context``; the usefulness gate reads
    ``stats``, so these rows are built directly.
    """

    return store.create_node(
        level="trace",
        content=content,
        context={"scope": SCOPE, "procedure_id": key},
        stats={"usefulness_score": usefulness},
    )


def _two_cluster_pool(store: MemoryStore) -> list[Node]:
    """Two structural clusters, enough to clear :data:`MIN_CLUSTERS`."""

    return [
        _rated_node(store, "alpha ritual step one", "alpha-ritual", 0.9),
        _rated_node(store, "alpha ritual step two", "alpha-ritual", 0.9),
        _rated_node(store, "beta ritual step one", "beta-ritual", 0.9),
        _rated_node(store, "beta ritual step two", "beta-ritual", 0.9),
    ]


def test_sel_contract_an_unarmed_build_puts_exactly_seven_counts_on_the_wire(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seven counts is the signature of a server with no gate armed.

    The field diagnosis rests on it: ``sel.x`` of width seven in the recorded
    events is *how* we know neither valve is charged in production. That
    inference holds only while an unarmed build cannot emit any other width,
    so the width is the assertion and the counts are incidental to it.
    """

    built = RecallMapBuilder(store).build(
        pool(_two_cluster_pool(store)),
        scope=SCOPE,
        task="sel-width-unarmed",
        decision_at=DECISION_AT,
    )

    assert built is not None
    sel = built.to_dict()["sel"]
    assert len(sel["x"]) == len(SELECTION_REASON_CODES) == 7
    assert sel["x"] == [0, 0, 0, 0, 0, 0, 0]
    assert sel["n"] == sel["e"] + sum(sel["x"]) == 4
    # The version tag travels with the vector; a width change without one
    # would be the unannounced break this guard exists to catch.
    assert sel["v"] == "r1"


def test_sel_contract_wire_counts_sit_at_their_frozen_positions(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sel.x` is read by index, so the index is the contract.

    The expected vector below is written out with its offsets hardcoded,
    exactly as a reader of a stored event hardcodes them, and then the offsets
    are checked against the tuple. Move ``fc`` or ``lr`` and the two halves
    disagree -- which is the failure, and it is not an arithmetic one.
    """

    header = json.dumps(
        {
            "path": "src/generated.py",
            "kind": "source",
            "language": "python",
            "sha256": "b" * 64,
            "chunk": "1/1",
            "lines": "1-20",
        }
    )
    ballast = make_node(store, "[file-chunk] " + header)
    cold = make_node(store, "a node no window has ever matured for")
    kept = _two_cluster_pool(store)

    def histories(candidate_ids, _decision_at):
        return {
            node_id: (
                MaturedRecallHistory.known(0, 0, 0)
                if node_id == cold.id
                else MaturedRecallHistory.known(100, 50, 50)
            )
            for node_id in candidate_ids
        }

    monkeypatch.setattr(store, "matured_recall_history", histories)
    built = RecallMapBuilder(store).build(
        pool([ballast, cold, *kept]),
        scope=SCOPE,
        task="sel-positions",
        decision_at=DECISION_AT,
    )

    assert built is not None
    counts = built.to_dict()["sel"]["x"]
    assert counts == [0, 0, 1, 0, 0, 1, 0]
    #        index:   0  1  2  3  4  5  6
    assert SELECTION_REASON_CODES[2] == "fc"  # the file-chunk ballast row
    assert SELECTION_REASON_CODES[5] == "lr"  # the row the frozen score refused
    assert sum(counts) == 2


def test_sel_contract_only_an_armed_gate_widens_the_wire_vector(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same store, same residual, same history -- only a valve moves.

    Width seven means "unarmed" only if arming is what produces width eight,
    and the frozen prefix has to survive the extension unchanged: that is the
    whole content of calling the ``sel`` contract additive.
    """

    hub = _rated_node(
        store, "alpha ritual the map keeps re-offering", "alpha-ritual", 0.078
    )
    candidates = pool([hub, *_two_cluster_pool(store)])

    unarmed = RecallMapBuilder(store).build(
        candidates, scope=SCOPE, task="sel-arming", decision_at=DECISION_AT
    )
    assert unarmed is not None
    unarmed_sel = unarmed.to_dict()["sel"]
    assert len(unarmed_sel["x"]) == 7
    assert unarmed_sel["x"] == [0, 0, 0, 0, 0, 0, 0]
    assert unarmed_sel["e"] == 5

    monkeypatch.setenv(POOL_USEFULNESS_GATE_ENV, "on")
    monkeypatch.setenv(POOL_USEFULNESS_FLOOR_ENV, "0.25")
    armed = RecallMapBuilder(store).build(
        candidates, scope=SCOPE, task="sel-arming", decision_at=DECISION_AT
    )
    assert armed is not None
    armed_sel = armed.to_dict()["sel"]

    assert len(armed_sel["x"]) == 8
    assert armed_sel["x"] == [0, 0, 0, 0, 0, 0, 0, 1]
    assert armed_sel["x"][:7] == unarmed_sel["x"]
    assert POOL_GATE_REASON_CODES[0] == "uf"
    assert armed_sel["e"] == 4
    assert armed_sel["n"] == unarmed_sel["n"] == 5
    # The vector got wider and nothing else about how it is read changed.
    assert armed_sel["v"] == unarmed_sel["v"] == "r1"
    assert armed_sel["n"] == armed_sel["e"] + sum(armed_sel["x"])


def test_coldstart_contract_build_hands_the_scorer_each_candidates_own_result(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold node owns its query-time scores and nothing else.

    A build is the whole path from residual to map, and this pins that the
    ``RecallResult`` survives all of it: the scorer is called with the entry
    the candidate actually arrived in -- that object, for that node -- and
    under the frozen policy the third argument changes no float it returns.
    Both halves matter. The plumbing is what a cold-start revision will spend,
    and the identity is why spending it is still ahead rather than behind us.
    """

    nodes = _two_cluster_pool(store)
    candidates = pool(nodes)
    by_node = {result.node.id: result for result in candidates}
    frozen_scorer = relevance_score
    seen: list[tuple[str, object]] = []

    def spy(node, history, result=None):
        seen.append((node.id, result))
        # Identical, not merely equal: `float.hex` separates 0.0 from -0.0.
        threaded = frozen_scorer(node, history, result)
        assert threaded.hex() == frozen_scorer(node, history).hex()
        return threaded

    monkeypatch.setattr(recall_map, "relevance_score", spy)
    built = RecallMapBuilder(store).build(
        candidates, scope=SCOPE, task="coldstart-plumbing", decision_at=DECISION_AT
    )

    assert built is not None
    assert [node_id for node_id, _result in seen] == [node.id for node in nodes]
    for node_id, result in seen:
        assert result is by_node[node_id]
    assert built.pool_size == len(nodes)


# ----------------------------------------------------------------------
# The dormant rank-calibration surface
# ----------------------------------------------------------------------
#
# These hold the evaluator to `coldstart-prereg-v2.json#scorer_shape`
# `.calibration.out_of_sample_and_live_application`, not to itself. The
# fixture builds a real fitting split, computes `u` from the registered
# midrank formula, publishes a table off that, and then asks the evaluator to
# reproduce the same `u` by lookup. A wrong bracket, a nearest-knot shortcut
# or a dropped midrank each fails against the definition rather than against a
# transcription of the implementation.
#
# Nothing in `recall_map` calls any of this yet, and the last two tests are
# what say so out loud.

_CALIBRATION_QUANTITY = "candidate_time_family_arm"

#: Values carrying enough of the fitting split to be published exactly. The
#: registered form calls a value an atom at 0.5% multiplicity; these are 20%
#: and 13.3% of 4500 rows, which is also why no grid can resolve the jump
#: across them and why the form publishes their ``u`` instead of interpolating.
_CALIBRATION_ATOMS: tuple[float, ...] = (700.0, 1900.0)

#: The continuum sits on half-integers, so a fitting row either *is* an atom
#: or is nowhere near one -- there is no float here that is almost 700.0.
_CALIBRATION_CONTINUUM: tuple[float, ...] = tuple(index + 0.5 for index in range(3000))


def _calibration_sample() -> list[float]:
    """``coldstart_train`` stood in for: a continuum and two mass points."""

    return sorted([*_CALIBRATION_CONTINUUM, *([700.0] * 900), *([1900.0] * 600)])


def _registered_u(ordered: list[float], value: float) -> float:
    """``u(v) = (A(v) + M(v)/2) / n - 0.5``, midrank ties.

    Transcribed from ``scorer_shape.calibration.rank_uniform_centered`` and
    computed against the fitting sample, never read off the table -- it is the
    yardstick the table and the evaluator are both measured with.
    """

    below = bisect_left(ordered, value)
    equal = bisect_right(ordered, value) - below
    return (below + equal / 2) / len(ordered) - 0.5


def _calibration_payload(stride: int = 6) -> dict[str, list[list[float]]]:
    """A published table: the atoms exactly, plus a strided quantile grid.

    ``stride`` is the grid's coarseness in fitting rows. At six the grid steps
    ``u`` by ``6/4500``, inside the registered ``2**-9``; at nine it steps by
    ``9/4500``, outside it. That is how the resolution rejection below is
    falsified rather than merely asserted.

    The grid is struck blind to the atoms -- it is a quantile grid over "the
    non-atom part" and nothing more, so no knot lands adjacent to an atom by
    arrangement. That matters: bracket an atom symmetrically and its midrank
    lands exactly on the interpolant between its neighbours, which would hide
    the difference between merging atoms into the breakpoints and holding them
    apart as bare equality overrides. Struck blind, the two disagree, and the
    monotonicity test can tell them apart.
    """

    ordered = _calibration_sample()
    knot_values = sorted(
        {_CALIBRATION_CONTINUUM[index] for index in range(0, 3000, stride)}
        | {_CALIBRATION_CONTINUUM[-1]}
    )
    return {
        "atoms": [[atom, _registered_u(ordered, atom)] for atom in _CALIBRATION_ATOMS],
        "knots": [[value, _registered_u(ordered, value)] for value in knot_values],
    }


def _calibration_policy(payload: object | None = None) -> dict[str, object]:
    """The table, wrapped the way a policy's calibration block carries it."""

    return {
        _CALIBRATION_QUANTITY: _calibration_payload() if payload is None else payload
    }


def _read_calibration() -> object:
    """The table a well-formed policy publishes, read through the reader."""

    return _calibration_table(_calibration_policy(), _CALIBRATION_QUANTITY)


def test_calibration_recovers_every_published_breakpoint_exactly() -> None:
    """Equality first, before any arithmetic can land an ulp away.

    The sealed plan's feasibility argument rests on the available-but-empty
    history composite being a tie block that reproduces *exactly*, so "close
    enough" is the one answer this may not give. Held to ``float.hex`` rather
    than ``==`` for that reason.

    Knots get the same treatment as atoms: a probe that ties with a published
    breakpoint is a tie with the fitting sample, and midrank says its answer is
    that block's published ``u``, not an interpolation approaching it.
    """

    table = _read_calibration()
    ordered = _calibration_sample()

    assert [value for value, _u in table.atoms] == list(_CALIBRATION_ATOMS)
    for value, u in table.atoms:
        assert u == _registered_u(ordered, value)
        assert _calibrate(table, value).hex() == u.hex()
    for value, u in table.knots:
        assert u == _registered_u(ordered, value)
        assert _calibrate(table, value).hex() == u.hex()

    # Atoms and knots are one merged breakpoint array, inside the registered
    # ceiling, and the atoms really are jumps a grid could not have carried.
    assert len(table.values) == len(table.atoms) + len(table.knots)
    assert len(table.values) <= RELEVANCE_CALIBRATION_MAX_BREAKPOINTS
    for atom in _CALIBRATION_ATOMS:
        above = _registered_u(ordered, atom + 0.5)
        below = _registered_u(ordered, atom - 0.5)
        assert above - below > RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR


def test_calibration_is_monotone_over_a_shuffled_probe_grid() -> None:
    """Order in, order out -- and evaluation order changes nothing.

    Probes are evaluated shuffled and compared sorted, so a lookup that
    remembered anything between calls, or that leaned on being asked in
    ascending order, cannot pass. The grid deliberately crowds the atoms from
    both sides, an ulp away and a quarter-unit away, because that neighbourhood
    is where an atom held apart as a pure equality override -- rather than
    merged into the breakpoints -- would invert the ordering.
    """

    table = _read_calibration()
    rng = random.Random(20260824)
    probes = [
        -1e6,
        -500.0,
        -0.25,
        0.0,
        3500.0,
        1e6,
        *_CALIBRATION_ATOMS,
        *(
            atom + delta
            for atom in _CALIBRATION_ATOMS
            for delta in (-0.25, -1e-9, 1e-9, 0.25)
        ),
        *(value for value, _u in table.knots),
        *(rng.uniform(-50.0, 3050.0) for _ in range(2000)),
    ]
    rng.shuffle(probes)

    readings = {probe: _calibrate(table, probe) for probe in probes}
    ascending = sorted(readings)
    for lower, higher in zip(ascending, ascending[1:]):
        assert readings[lower] <= readings[higher], (lower, higher)

    # Monotone and not merely constant: a lookup that answered one number
    # everywhere would satisfy every inequality above.
    assert readings[ascending[0]] < readings[ascending[-1]]
    assert readings[ascending[0]] == table.us[0]
    assert readings[ascending[-1]] == table.us[-1]


def test_calibration_interpolation_stays_inside_the_registered_bound() -> None:
    """``max_interpolation_error_on_u`` is 2**-9, where the form claims it.

    It claims it *between consecutive knots*: the grid is published over "the
    non-atom part". A knot pair straddling an atom is exempt by construction --
    the atom's block is at least 0.5% of the fitting rows, a jump in ``u``
    wider than 2**-9 that no grid of any density could resolve, which is the
    whole reason the form publishes that value exactly instead.

    So the exemption is derived from the table rather than declared, held to
    the one knot pair each atom is entitled to, and the bound is enforced on
    every other probe. An exemption that quietly grew to cover the grid would
    show up as a collapsed ``checked`` count, not as a silently passing
    assertion.
    """

    table = _read_calibration()
    ordered = _calibration_sample()
    exempt = tuple(
        (low, high)
        for (low, _low_u), (high, _high_u) in zip(table.knots, table.knots[1:])
        if any(low < atom < high for atom in _CALIBRATION_ATOMS)
    )
    assert len(exempt) == len(_CALIBRATION_ATOMS)
    # One stride-wide window per atom, out of a probe range 2999 wide.
    assert sum(high - low for low, high in exempt) == 12.0

    rng = random.Random(915)
    probes = [rng.uniform(0.5, 2999.5) for _ in range(4000)]
    checked = 0
    worst = 0.0
    for probe in probes:
        if any(low < probe < high for low, high in exempt):
            continue
        error = abs(_calibrate(table, probe) - _registered_u(ordered, probe))
        worst = max(worst, error)
        checked += 1
    assert checked > 3900
    assert worst <= RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR
    # The grid is genuinely interpolating, not sitting on the answer: a probe
    # off a breakpoint has a real error, just a bounded one.
    assert worst > 0.0

    # And the interpolation is *linear*, which the bound alone cannot pin.
    # The reader has already forced every non-exempt knot pair to be narrower
    # in u than the bound, so every answer inside such a bracket satisfies it
    # -- nearest-knot included, and nearest-knot is not what the form says.
    low_value, low_u = table.knots[100]
    high_value, high_u = table.knots[101]
    assert not any(low_value < atom < high_value for atom in _CALIBRATION_ATOMS)
    for fraction in (0.125, 0.25, 0.5, 0.75, 0.875):
        probe = low_value + fraction * (high_value - low_value)
        straight = low_u + fraction * (high_u - low_u)
        assert _calibrate(table, probe) == pytest.approx(straight, abs=1e-15)


def test_calibration_clamps_outside_the_outermost_knots() -> None:
    """Past the evidence, the answer is the extreme rank, not an extrapolation.

    The fitting sample never went there. Continuing the last segment's slope
    would invent ranks below -0.5 and above +0.5 -- positions no row can hold
    -- so the registered form clamps, and clamps to the published value rather
    than to a recomputed one.
    """

    table = _read_calibration()
    lowest_value, lowest_u = table.knots[0]
    highest_value, highest_u = table.knots[-1]
    assert table.values[0] == lowest_value
    assert table.values[-1] == highest_value

    for probe in (lowest_value - 1e-9, 0.0, -1.0, -1e6, -1e300):
        assert _calibrate(table, probe).hex() == lowest_u.hex()
    for probe in (highest_value + 1e-9, 3000.0, 1e6, 1e300):
        assert _calibrate(table, probe).hex() == highest_u.hex()


#: A table built for floating-point awkwardness rather than for realism: its
#: two outer ``u`` values straddle zero, and a knot pair straddling zero is
#: exactly what sits either side of a median atom on a centred rank scale.
#: The knots are far apart in ``u`` and allowed to be, because the atom
#: between them is what carries the jump.
_AWKWARD_CALIBRATION: dict[str, list[list[float]]] = {
    "atoms": [[2.0, -0.17]],
    "knots": [[0.0, -0.5], [4.0, -0.02]],
}


def test_calibration_answers_a_published_value_before_it_does_arithmetic() -> None:
    """Equality first is not decoration: interpolation does not round-trip.

    ``low + (high - low)`` gives ``high`` back for most pairs and stops doing
    so once the two straddle zero. On this table the atom's own value, reached
    by arithmetic instead of by equality, comes back an ulp low --
    ``-0.17000000000000004`` where the policy published ``-0.17``. That is the
    difference between a tie block that reproduces and one that merely nearly
    reproduces, and the sealed plan's feasibility argument needs the first.
    """

    table = _calibration_table(
        {_CALIBRATION_QUANTITY: _AWKWARD_CALIBRATION}, _CALIBRATION_QUANTITY
    )
    ((atom_value, atom_u),) = table.atoms
    (low_value, low_u), (high_value, high_u) = table.knots

    # The arithmetic the lookup must not have reached, shown missing its mark.
    assert low_u + (atom_u - low_u) != atom_u
    for value, u in ((low_value, low_u), (atom_value, atom_u), (high_value, high_u)):
        assert _calibrate(table, value).hex() == u.hex()

    # Still monotone across the awkward pair, and still clamped outside it.
    walk = (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 9.0)
    readings = [_calibrate(table, probe) for probe in walk]
    assert readings == sorted(readings)
    assert readings[0].hex() == low_u.hex()
    assert readings[-1].hex() == high_u.hex()


#: A table whose values span far enough that ``(probe - low) / (high - low)``
#: rounds to exactly ``1.0`` for a probe strictly below the top breakpoint.
#: The reader accepts it: it constrains a span to be *representable*, not to
#: be narrow.
_WIDE_SPAN_CALIBRATION: dict[str, list[list[float]]] = {
    "atoms": [[-9e299, -0.3]],
    "knots": [[-1e300, -0.5], [1e299, 0.1]],
}


def test_calibration_stays_monotone_where_the_arithmetic_overshoots() -> None:
    """Clamping the interpolation into its own bracket is not idle.

    On this table the fraction rounds to exactly ``1.0`` a step below the top
    breakpoint, and ``-0.3 + 0.4`` lands on ``0.10000000000000003`` rather
    than ``0.1``. Unclamped, the segment would end *above* the breakpoint the
    next one starts at: a table the reader certified as monotone evaluating
    non-monotonically for purely floating-point reasons.

    Exotic values, ordinary consequence. The reader does not bound how far
    apart two published values may sit, so "monotone" has to mean monotone on
    every table it accepts, not only on the well-scaled ones.
    """

    table = _calibration_table(
        {_CALIBRATION_QUANTITY: _WIDE_SPAN_CALIBRATION}, _CALIBRATION_QUANTITY
    )
    top_value, top_u = table.knots[-1]
    low_value, low_u = table.values[1], table.us[1]
    below = nextafter(top_value, low_value)
    assert below < top_value

    # The arithmetic really does overshoot the top of the bracket here.
    assert (below - low_value) / (top_value - low_value) == 1.0
    assert low_u + (top_u - low_u) > top_u

    assert _calibrate(table, below) <= _calibrate(table, top_value)
    assert _calibrate(table, below).hex() == top_u.hex()
    assert _calibrate(table, top_value).hex() == top_u.hex()


def _calibration_rejections() -> list[tuple[str, object, str]]:
    """Every shape the reader must refuse, each a mutation of one it accepts.

    Built by mutation on purpose. A check that quietly stopped checking is
    caught here; a check so strict that it also refuses the honest table is
    caught by every other test in this section still reading one.
    """

    good = _calibration_payload()
    knots: list[list[float]] = [list(pair) for pair in good["knots"]]
    atoms: list[list[float]] = [list(pair) for pair in good["atoms"]]

    def mutated(**changes: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "atoms": [list(pair) for pair in atoms],
            "knots": [list(pair) for pair in knots],
        }
        payload.update(changes)
        return payload

    def with_knots(edit) -> dict[str, object]:
        copy = [list(pair) for pair in knots]
        edit(copy)
        return mutated(knots=copy)

    def swap_values(rows: list[list[float]]) -> None:
        rows[10], rows[11] = rows[11], rows[10]

    def duplicate_value(rows: list[list[float]]) -> None:
        rows[11][0] = rows[10][0]

    def step_down(rows: list[list[float]]) -> None:
        rows[10][1], rows[11][1] = rows[11][1], rows[10][1]

    return [
        (
            "no_calibration_block",
            {},
            "publishes no calibration tables",
        ),
        (
            "calibration_block_is_not_a_mapping",
            [[_CALIBRATION_QUANTITY, good]],
            "publishes no calibration tables",
        ),
        (
            "no_table_for_this_quantity",
            {"some_other_arm": good},
            "no calibration table for",
        ),
        (
            "table_is_not_a_mapping",
            _calibration_policy(payload=knots),
            "is not a mapping",
        ),
        (
            "no_knots_at_all",
            _calibration_policy(payload={"atoms": atoms}),
            "is not a sequence of value/u pairs",
        ),
        (
            "a_knot_is_not_a_pair",
            _calibration_policy(
                payload=with_knots(lambda rows: rows.__setitem__(5, [0.5]))
            ),
            "is not a value/u pair",
        ),
        (
            "a_single_knot_interpolates_nothing",
            _calibration_policy(payload=mutated(atoms=[], knots=knots[:1])),
            "fewer than two knots",
        ),
        (
            "a_knot_value_is_not_a_real_number",
            _calibration_policy(
                payload=with_knots(lambda rows: rows[5].__setitem__(0, "0.5"))
            ),
            "is not a real number",
        ),
        (
            "a_knot_u_is_a_bool",
            _calibration_policy(
                payload=with_knots(lambda rows: rows[5].__setitem__(1, True))
            ),
            "is not a real number",
        ),
        (
            "a_knot_u_is_not_finite",
            _calibration_policy(
                payload=with_knots(lambda rows: rows[5].__setitem__(1, float("nan")))
            ),
            "is not finite",
        ),
        (
            "a_knot_value_is_not_finite",
            _calibration_policy(
                payload=with_knots(lambda rows: rows[5].__setitem__(0, float("inf")))
            ),
            "is not finite",
        ),
        (
            "knots_are_unsorted",
            _calibration_policy(payload=with_knots(swap_values)),
            "are not sorted by strictly increasing value",
        ),
        (
            "two_knots_share_a_value",
            _calibration_policy(payload=with_knots(duplicate_value)),
            "are not sorted by strictly increasing value",
        ),
        (
            "atoms_are_unsorted",
            _calibration_policy(payload=mutated(atoms=list(reversed(atoms)))),
            "are not sorted by strictly increasing value",
        ),
        (
            "an_atom_sits_on_a_knot",
            _calibration_policy(payload=mutated(atoms=[list(knots[3]), *atoms])),
            "both an atom and a knot",
        ),
        (
            "u_steps_down",
            _calibration_policy(payload=with_knots(step_down)),
            "is not monotone",
        ),
        (
            "more_breakpoints_than_the_registered_ceiling",
            _calibration_policy(payload=_calibration_payload(stride=2)),
            "breakpoints, over the registered",
        ),
        (
            "the_grid_is_too_coarse_for_the_published_bound",
            _calibration_policy(payload=_calibration_payload(stride=9)),
            "steps u by",
        ),
        (
            "the_value_span_is_not_representable",
            _calibration_policy(
                payload={"atoms": [], "knots": [[-1.5e308, -0.5], [1.5e308, -0.4999]]}
            ),
            "is not representable",
        ),
    ]


@pytest.mark.parametrize(
    ("calibration", "message"),
    [
        pytest.param(calibration, message, id=name)
        for name, calibration, message in _calibration_rejections()
    ],
)
def test_calibration_reader_refuses_a_table_it_cannot_trust(
    calibration: object, message: str
) -> None:
    """Fail closed, because there is no honest neutral answer to fall back on.

    ``rank_uniform_centered`` is centred on the fitting median, so its neutral
    value is ``0.0`` -- a number that *means* "this row sits where half the
    population sits". Returning it for a table this could not read would not be
    a degraded answer, it would be a fabricated measurement, and the arm it
    feeds would go on being summed as though it had been calibrated.
    """

    with pytest.raises(ValueError, match=message):
        _calibration_table(calibration, _CALIBRATION_QUANTITY)


@pytest.mark.parametrize(
    "probe",
    [float("nan"), float("inf"), float("-inf"), "0.5", None, True, object()],
    ids=["nan", "inf", "minus_inf", "str", "none", "bool", "object"],
)
def test_calibration_lookup_refuses_a_probe_it_cannot_read(probe: object) -> None:
    """A row whose own value cannot be read has no rank, not a middling one."""

    table = _read_calibration()
    with pytest.raises(ValueError):
        _calibrate(table, probe)


def test_the_frozen_policy_publishes_no_calibration_table() -> None:
    """Dormant means the surface exists and has nothing to read.

    ``directional-zsum-r1`` z-scores five features and calibrates none of them,
    so the reader's answer for it is a refusal. That refusal is the interlock:
    a half-wired revision that spent a calibrated arm without publishing its
    table would raise here rather than quietly score every row at the median.
    """

    assert RELEVANCE_POLICY_ID == "directional-zsum-r1"
    assert dict(RELEVANCE_POLICY_CALIBRATION) == {}
    for quantity in ("history_arm", "level_arm", _CALIBRATION_QUANTITY):
        with pytest.raises(ValueError, match="publishes no calibration tables"):
            _calibration_table(RELEVANCE_POLICY_CALIBRATION, quantity)


def test_only_the_cold_lane_reads_the_calibration_surface() -> None:
    """The registered consumer, and no other -- the frozen scorer stays blind.

    The surface landed empty-handed and unread; the cold exploration lane
    (cold-quota-prereg.json, plan cold-quota-r1) is its first and only
    licensed reader. A source scan rather than a behavioural one,
    deliberately: the claim is about every call site in the module at once,
    and a test that scored a handful of nodes would pass just as happily if
    the surface were wired into a branch those nodes happen to miss. The
    frozen warm arithmetic is pinned bit-for-bit by the test below this one.
    """

    import ast

    source = (
        Path(__file__).resolve().parents[1] / "src" / "living_memory" / "recall_map.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    callers: set[str] = set()
    for definition in ast.walk(module):
        if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(definition):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"_calibrate", "_calibration_table"}
            ):
                callers.add(definition.name)
    assert callers == {"_cold_ranking_tables", "_cold_composite"}


def test_the_cold_start_scores_this_commit_had_to_leave_alone() -> None:
    """The two floats the whole cold-start defect is stated in, pinned.

    A node with no delivery history scores on its level term alone: schema
    ``+1.6682``, trace and concept ``-0.5995``, against a threshold of
    ``3.8708``. Those are the numbers a revision exists to move, and the number
    this commit must not. Pinned by ``float.hex`` so an arithmetic change too
    small to print is still a failure.
    """

    assert (
        relevance_score(SimpleNamespace(level="schema"), None).hex()
        == "0x1.ab0fdab1508aap+0"
    )
    for level in ("trace", "concept"):
        assert (
            relevance_score(SimpleNamespace(level=level), None).hex()
            == "-0x1.32ea699041f5bp-1"
        )
