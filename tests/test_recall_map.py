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

The two deliberate mirrors in ``recall_map`` are pinned against their originals
here rather than trusted: ``normalize_key`` against
``consolidation._normalize_trigger`` and ``_chunk_table_revision`` against
``retrieval._chunk_table_revision``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from living_memory import recall_map
from living_memory.chunking import TextChunk
from living_memory.models import Node
from living_memory.recall_map import (
    MAX_CLUSTERS,
    MAX_INSTRUCTIONS_CHARS,
    MAX_LABEL_CHARS,
    MAX_RESPONSE_CHARS,
    MEDOID_EXAMPLE_CHARS,
    STAGE_ANCHOR,
    STAGE_EMBEDDING,
    STAGE_PATH,
    STAGE_STRUCTURAL,
    RecallMapBuilder,
    _payload_size,
    cache_key,
    normalize_key,
)
from living_memory.retrieval import RecallResult
from living_memory.storage import MemoryStore, recall_fingerprint

SCOPE = "project:lm"


@pytest.fixture()
def store(tmp_path: Path):
    with MemoryStore(tmp_path / "recall-map.sqlite3") as opened:
        yield opened


def make_node(
    store: MemoryStore, content: str, context: dict | None = None, **extra
) -> Node:
    payload = {"scope": SCOPE}
    payload.update(context or {})
    payload.update(extra)
    return store.create_node(level="trace", content=content, context=payload)


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


def test_structural_keys_label_their_own_clusters(store: MemoryStore) -> None:
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
        "postsession": 3,
        "scripts": 1,
        "living memory": 1,
    }


def test_subsystem_collapse_handles_absolute_paths_and_line_refs(store: MemoryStore) -> None:
    nodes = [
        make_node(store, "one", {"files": ["/home/u/p/lm/src/living_memory/postsession/a.py"]}),
        make_node(store, "two", {"files": ["src/living_memory/postsession/b.py:1426-1434"]}),
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert [(c.label, c.count) for c in built.clusters] == [("postsession", 2)]


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


def test_embedding_fallback_clusters_and_labels_by_ctfidf(store: MemoryStore) -> None:
    # Corpus noise, so document frequency has something to discount against.
    for index in range(6):
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

    assert RecallMapBuilder(store).build(pool([node]), scope=SCOPE) is None


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
    assert by_stage[STAGE_PATH].label == "delivery"
    assert by_stage[STAGE_ANCHOR].label == "what did we decide about anchors"
    assert built.covered == 7
    assert built.dropped == 0


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
    assert labels == ["root cause", "root cause deploy"]
    assert [cluster.count for cluster in built.clusters] == [1, 1]
    # Not merged: each keeps its own member, and its own way to ask.
    assert [cluster.member_ids for cluster in built.clusters] == [
        (nodes[1].id,),
        (nodes[0].id,),
    ]
    assert built.clusters[1].ask_hint == "root cause deploy"


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
    store: MemoryStore,
) -> None:
    """Uniqueness is decided on the label the reader actually sees.

    Anchors work *because* one operator's questions share a house style, so two
    anchor queries agreeing for their first forty characters is the expected
    case, not a pathological one.
    """

    first = make_node(store, "the august retrieval weights")
    second = make_node(store, "the september retrieval weights")
    seed_anchor(store, "what did we decide about the retrieval weights in august", [first])
    seed_anchor(store, "what did we decide about the retrieval weights in september", [second])

    built = RecallMapBuilder(store).build(pool([first, second]), scope=SCOPE)

    assert built is not None
    labels = [cluster.label for cluster in built.clusters]
    assert len(labels) == len(set(labels)) == 2
    assert all(len(label) <= MAX_LABEL_CHARS for label in labels)
    # The ask-hint is the anchor's own wording, untouched by disambiguation.
    hints = sorted(cluster.ask_hint for cluster in built.clusters)
    assert hints == [
        "what did we decide about the retrieval weights in august",
        "what did we decide about the retrieval weights in september",
    ]


def test_a_cached_build_agrees_with_a_cold_one_on_embedding_clusters(
    store: MemoryStore,
) -> None:
    """The medoid is structure, so the cache must not re-decide it.

    Stage 4 picks its medoid by mean intra-cluster similarity, using vectors
    the cache path deliberately does not re-read. Without carrying the choice
    across, the same pool would report the most central member cold and the
    highest-ranked one warm.
    """

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
    assert _payload_size(built.clusters, built.pool_size, built.dropped) <= MAX_RESPONSE_CHARS
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


def test_examples_are_shaved_before_a_cluster_is_dropped(store: MemoryStore) -> None:
    """One character over budget costs one character, not every example."""

    long_tail = "content that runs on and on " * 6
    nodes = [
        make_node(store, f"note {index}: {long_tail}", {"topic": f"area-{index}"})
        for index in range(3)
    ]

    built = RecallMapBuilder(store).build(pool(nodes), scope=SCOPE)

    assert built is not None
    assert len(built.clusters) == 3
    assert built.dropped == 0
    assert _payload_size(built.clusters, built.pool_size, built.dropped) <= MAX_RESPONSE_CHARS
    # Every example survived, shortened rather than discarded.
    examples = [cluster.medoid.example for cluster in built.clusters]
    assert all(example.endswith("…") for example in examples)
    assert all(0 < len(example) < MEDOID_EXAMPLE_CHARS for example in examples)


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
