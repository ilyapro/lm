"""Query-anchor storage, dedup, weighted edges, and edge migration (schema v7).

Three contracts are under test here.

*The migration is additive.* ``nodes.level`` and ``connections.type`` carry
baked-in CHECK constraints on a 500 MB live database, so anchors may not be a
new node level nor anchor edges a new connection type — either would need a
table rebuild. ``_seed_v6_database`` writes the **verbatim** ``nodes`` and
``connections`` DDL read out of the live file (including the ALTER TABLE trail
that previous additive migrations left behind), and the tests assert those two
strings come back byte-identical after the v7 migration has run.

*Anchors deduplicate and accumulate.* Cosine dedup is asserted against
``ANCHOR_DEDUP_COSINE_THRESHOLD`` rather than against a literal, because the
empirical value is due from ``artifacts/anchors/estimate.json`` and retuning it
must not require editing a test. Edge weight accumulates, and one test pins the
downgrade that ``MemoryStore.create_connection`` exhibits, as the control that
says why anchor edges do not reuse it.

*Anchor edges follow their target.* Every write path that supersedes a node —
teach, duplicate-content dedup, derived R1c corrections — plus consolidation's
era displacement, plus the batch sweep.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from living_memory import query_anchors as qa
from living_memory.consolidation import memory_teach
from living_memory.edge_derivation import derive_edges_for_new_trace
from living_memory.query_anchors import (
    ANCHOR_DEDUP_COSINE_THRESHOLD,
    ANCHOR_EDGE_WEIGHT,
    ANCHOR_TTL_DAYS,
    ANCHOR_TTL_DECAY_REASON,
)
from living_memory.storage import (
    QUERY_ANCHOR_EDGE_TABLE,
    QUERY_ANCHOR_TABLE,
    SCHEMA_VERSION,
    MemoryStore,
    pack_chunk_embedding,
    recall_fingerprint,
    unpack_chunk_embedding,
)

SCOPE = "project:demo"

# The live database's own DDL for the two tables the v7 migration must not
# touch, copied verbatim from `SELECT sql FROM sqlite_master` on a
# sqlite-backup-API snapshot of ~/.local/share/living-memory/global.sqlite3
# (schema_version 6, 16,717 nodes, 143,209 connections). The trailing
# `, content_fingerprint TEXT` on `nodes` is the v3 ALTER TABLE's trail and is
# part of what byte-identical means here.
LIVE_NODES_DDL = """CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL CHECK (level IN ('trace', 'concept', 'schema')),
                    content TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'global',
                    agent TEXT,
                    task TEXT,
                    context TEXT NOT NULL DEFAULT '{}',
                    timestamp TEXT NOT NULL,
                    decayed INTEGER NOT NULL DEFAULT 0 CHECK (decayed IN (0, 1)),
                    decay_reason TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,
                    usefulness_score REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    unique_agents INTEGER NOT NULL DEFAULT 0,
                    temporal_hint TEXT,
                    source_traces TEXT NOT NULL DEFAULT '[]',
                    corrections TEXT NOT NULL DEFAULT '[]',
                    provenance TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                , content_fingerprint TEXT)"""

LIVE_CONNECTIONS_DDL = """CREATE TABLE connections (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES nodes(id),
                    target_id TEXT NOT NULL REFERENCES nodes(id),
                    type TEXT NOT NULL CHECK (
                        type IN ('related', 'caused', 'contradicts', 'supersedes', 'requires')
                    ),
                    weight REAL NOT NULL DEFAULT 1.0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_id, target_id, type)
                )"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path, name: str = "memory.sqlite3") -> MemoryStore:
    return MemoryStore(tmp_path / name)


def _trace(store: MemoryStore, content: str, scope: str = SCOPE):
    return store.append_trace(content, {"scope": scope})


def _unit_at_cosine(target: float) -> list[float]:
    """A unit 2-vector whose cosine against ``[1.0, 0.0]`` is ``target``."""

    clamped = max(-1.0, min(1.0, float(target)))
    return [clamped, math.sqrt(max(0.0, 1.0 - clamped * clamped))]


def _table_sql(store: MemoryStore, name: str) -> str:
    row = store.connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return "" if row is None else str(row["sql"])


def _seed_v6_database(path: Path) -> list[str]:
    """A schema-v6 file carrying the live database's exact nodes/connections DDL."""

    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('schema_version', '6')"
        )
        connection.execute(LIVE_NODES_DDL)
        connection.execute(LIVE_CONNECTIONS_DDL)
        seeded: list[str] = []
        for index in range(3):
            node_id = f"LEGACY{index:020d}"
            connection.execute(
                """
                INSERT INTO nodes (id, level, content, scope, timestamp,
                                   created_at, updated_at, content_fingerprint)
                VALUES (?, 'trace', ?, 'project:legacy', '2026-01-01T00:00:00Z',
                        '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', ?)
                """,
                (node_id, f"legacy node {index}", f"fp{index}"),
            )
            seeded.append(node_id)
        connection.execute(
            """
            INSERT INTO connections (id, source_id, target_id, type, weight,
                                     created_at, updated_at)
            VALUES ('LEGACYEDGE', ?, ?, 'related', 0.5,
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """,
            (seeded[0], seeded[1]),
        )
        connection.commit()
        return seeded
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------


def test_fresh_database_carries_v7_anchor_tables_and_indexes(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        version = store.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert version["value"] == str(SCHEMA_VERSION) == "8"

        objects = {
            str(row["name"])
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master"
            ).fetchall()
        }
        assert QUERY_ANCHOR_TABLE in objects
        assert QUERY_ANCHOR_EDGE_TABLE in objects
        assert "idx_query_anchors_scope_active" in objects
        assert "idx_query_anchor_edges_target" in objects


def test_v6_database_migrates_leaving_nodes_and_connections_ddl_identical(
    tmp_path: Path,
) -> None:
    db = tmp_path / "legacy.sqlite3"
    seeded = _seed_v6_database(db)

    with MemoryStore(db) as store:
        assert _table_sql(store, "nodes") == LIVE_NODES_DDL
        assert _table_sql(store, "connections") == LIVE_CONNECTIONS_DDL

        assert store.count_query_anchors() == 0
        assert store.count_query_anchor_edges() == 0
        version = store.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert version["value"] == "8"

        # Additive: every legacy row survives the migration untouched.
        for node_id in seeded:
            assert store.get_node(node_id) is not None
        assert len(store.list_connections(source_id=seeded[0])) == 1

    # Idempotent: a second open changes nothing.
    with MemoryStore(db) as store:
        assert _table_sql(store, "nodes") == LIVE_NODES_DDL
        assert _table_sql(store, "connections") == LIVE_CONNECTIONS_DDL
        anchor_tables = store.connection.execute(
            "SELECT COUNT(*) AS count FROM sqlite_master WHERE name LIKE 'query_anchor%'"
        ).fetchone()
        assert int(anchor_tables["count"]) == 2


def test_partial_anchor_rollout_gains_missing_columns(tmp_path: Path) -> None:
    """A half-created anchor table is ALTERed, not silently left short."""

    db = tmp_path / "partial.sqlite3"
    _seed_v6_database(db)
    connection = sqlite3.connect(str(db))
    try:
        connection.executescript(
            """
            CREATE TABLE query_anchors (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                query TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                first_seen TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(scope, fingerprint)
            );
            CREATE TABLE query_anchor_edges (
                anchor_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (anchor_id, target_id)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO query_anchors (id, scope, query, fingerprint, dimensions,
                                       embedding, first_seen, created_at, updated_at)
            VALUES ('OLDANCHOR', 'project:demo', 'q', 'fp', 2, ?,
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                    '2026-01-01T00:00:00Z')
            """,
            (pack_chunk_embedding([1.0, 0.0]),),
        )
        connection.commit()
    finally:
        connection.close()

    with MemoryStore(db) as store:
        columns = {
            str(row["name"])
            for row in store.connection.execute(
                f"PRAGMA table_info({QUERY_ANCHOR_TABLE})"
            ).fetchall()
        }
        assert {
            "reinforcement_count",
            "usefulness_score",
            "decayed",
            "decay_reason",
            "last_matched_at",
        } <= columns
        edge_columns = {
            str(row["name"])
            for row in store.connection.execute(
                f"PRAGMA table_info({QUERY_ANCHOR_EDGE_TABLE})"
            ).fetchall()
        }
        assert {"weight", "hits"} <= edge_columns

        # The pre-existing row survives and reads back through the accessor.
        anchor = store.get_query_anchor("OLDANCHOR")
        assert anchor is not None
        assert anchor.decayed is False
        assert anchor.reinforcement_count == 0


def test_anchor_reads_degrade_to_empty_without_the_v7_tables(tmp_path: Path) -> None:
    """A snapshot opened without the anchor tables reports nothing, never raises."""

    with _store(tmp_path) as store:
        store.connection.executescript(
            f"DROP TABLE {QUERY_ANCHOR_EDGE_TABLE}; DROP TABLE {QUERY_ANCHOR_TABLE};"
        )
        store._invalidate_schema_shape_cache()

        assert store.list_query_anchors() == []
        assert store.count_query_anchors() == 0
        assert store.count_query_anchor_edges() == 0
        assert store.list_query_anchor_edges(anchor_id="x") == []
        assert store.list_anchor_edge_targets() == []
        assert list(store.iter_query_anchor_vectors()) == []
        assert store.find_query_anchor(SCOPE, "fp") is None
        assert store.repoint_query_anchor_edges("a", "b") == 0
        assert qa.match_anchors(store, [1.0, 0.0], SCOPE) == []

        # A supersedes write still succeeds on a database with no anchor tables.
        old = _trace(store, "pre-v7 content")
        new = _trace(store, "pre-v7 replacement")
        store.create_connection(new.id, old.id, "supersedes")


def test_anchor_embedding_round_trips_as_float32_blob(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        vector = [0.5, -0.25, 0.125]
        anchor = store.insert_query_anchor(
            scope=SCOPE, query="q", fingerprint="fp", embedding=vector
        )
        assert anchor.dimensions == 3
        assert list(anchor.embedding) == vector

        row = store.connection.execute(
            f"SELECT dimensions, embedding FROM {QUERY_ANCHOR_TABLE} WHERE id = ?",
            (anchor.id,),
        ).fetchone()
        assert len(bytes(row["embedding"])) == 3 * 4
        # Width comes from the row, never from the byte length.
        assert unpack_chunk_embedding(row["embedding"], int(row["dimensions"])) == vector
        with pytest.raises(ValueError):
            unpack_chunk_embedding(row["embedding"], 4)


# ---------------------------------------------------------------------------
# upsert and dedup
# ---------------------------------------------------------------------------


def test_upsert_creates_anchor_with_weighted_edges(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        used = _trace(store, "the ticket review runbook")
        other = _trace(store, "a second consulted node")

        result = qa.upsert_anchor(
            store, "поревьювь EZ-13871", SCOPE, [1.0, 0.0], [used.id, other.id]
        )

        assert result.created is True
        assert result.reinforced is False
        assert result.matched_by is None
        assert result.anchor.scope == SCOPE
        assert result.anchor.query == "поревьювь EZ-13871"
        assert result.anchor.fingerprint == recall_fingerprint("поревьювь EZ-13871", SCOPE)
        assert {edge.target_id for edge in result.edges} == {used.id, other.id}
        assert all(edge.weight == ANCHOR_EDGE_WEIGHT for edge in result.edges)
        assert all(edge.hits == 1 for edge in result.edges)


def test_exact_repeat_reinforces_the_same_anchor_by_fingerprint(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        used = _trace(store, "runbook")
        first = qa.upsert_anchor(store, "как деплоить", SCOPE, [1.0, 0.0], [used.id])
        # Whitespace differences collapse: same fingerprint, same anchor.
        second = qa.upsert_anchor(
            store, "  как   деплоить  ", SCOPE, [0.0, 1.0], [used.id]
        )

        assert second.created is False
        assert second.matched_by == "fingerprint"
        assert second.anchor.id == first.anchor.id
        assert second.anchor.reinforcement_count == 1
        assert store.count_query_anchors() == 1
        # The matched anchor keeps its own text and vector; a reinforcement is
        # not an opportunity for the anchor to drift onto the newest phrasing.
        assert second.anchor.query == "как деплоить"
        assert list(second.anchor.embedding) == [1.0, 0.0]


def test_near_identical_query_deduplicates_at_the_cosine_threshold(
    tmp_path: Path,
) -> None:
    """Behaviour is pinned to the constant, not to its current numeric value."""

    with _store(tmp_path) as store:
        used = _trace(store, "runbook")
        base = qa.upsert_anchor(store, "первый запрос", SCOPE, [1.0, 0.0], [used.id])

        # Comfortably above the threshold (float32 storage costs a little
        # precision, so do not sit exactly on it).
        above = qa.upsert_anchor(
            store,
            "первый запрос, чуть иначе",
            SCOPE,
            _unit_at_cosine(min(0.999, ANCHOR_DEDUP_COSINE_THRESHOLD + 0.01)),
            [used.id],
        )
        assert above.created is False
        assert above.matched_by == "cosine"
        assert above.anchor.id == base.anchor.id
        assert above.similarity >= ANCHOR_DEDUP_COSINE_THRESHOLD

        # Clearly below it: a distinct situation earns its own anchor.
        below = qa.upsert_anchor(
            store,
            "совсем другой запрос",
            SCOPE,
            _unit_at_cosine(ANCHOR_DEDUP_COSINE_THRESHOLD - 0.05),
            [used.id],
        )
        assert below.created is True
        assert below.anchor.id != base.anchor.id
        assert store.count_query_anchors() == 2


def test_dedup_threshold_keeps_same_jargon_different_object_apart(
    tmp_path: Path,
) -> None:
    """0.931 is the measured «поревьювь EZ-13871» vs «поревьювь EZ-12826» pair.

    Two different tickets are two different situations with two different sets
    of useful nodes; merging them would leave one anchor answering neither.
    """

    assert ANCHOR_DEDUP_COSINE_THRESHOLD > 0.931

    with _store(tmp_path) as store:
        first_node = _trace(store, "EZ-13871 review notes")
        second_node = _trace(store, "EZ-12826 review notes")
        first = qa.upsert_anchor(
            store, "поревьювь EZ-13871", SCOPE, [1.0, 0.0], [first_node.id]
        )
        second = qa.upsert_anchor(
            store, "поревьювь EZ-12826", SCOPE, _unit_at_cosine(0.931), [second_node.id]
        )

        assert second.anchor.id != first.anchor.id
        assert {
            edge.target_id
            for edge in store.list_query_anchor_edges(anchor_id=first.anchor.id)
        } == {first_node.id}
        assert {
            edge.target_id
            for edge in store.list_query_anchor_edges(anchor_id=second.anchor.id)
        } == {second_node.id}


def test_anchor_never_crosses_scope(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        here = _trace(store, "scoped node", scope=SCOPE)
        there = _trace(store, "other node", scope="project:other")

        mine = qa.upsert_anchor(store, "один и тот же запрос", SCOPE, [1.0, 0.0], [here.id])
        theirs = qa.upsert_anchor(
            store, "один и тот же запрос", "project:other", [1.0, 0.0], [there.id]
        )

        # Identical text and identical vector, yet two anchors: the fingerprint
        # carries the scope and the cosine scan is scoped.
        assert theirs.created is True
        assert theirs.anchor.id != mine.anchor.id
        assert store.count_query_anchors() == 2

        matched = qa.match_anchors(store, [1.0, 0.0], SCOPE)
        assert [m.anchor.id for m in matched] == [mine.anchor.id]


def test_upsert_skips_targets_that_have_no_live_node(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        gone = _trace(store, "forgotten node")
        store.soft_delete_node(gone.id, "manual forget")
        alive = _trace(store, "live node")

        result = qa.upsert_anchor(
            store, "запрос", SCOPE, [1.0, 0.0], [gone.id, alive.id, "NOSUCHNODE"]
        )

        assert {edge.target_id for edge in result.edges} == {alive.id}
        assert set(result.skipped_targets) == {gone.id, "NOSUCHNODE"}


def test_upsert_points_at_the_replacement_of_an_already_superseded_target(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        old = _trace(store, "one restart is enough")
        new = _trace(store, "two restarts are required")
        store.create_connection(new.id, old.id, "supersedes")

        result = qa.upsert_anchor(store, "деплой", SCOPE, [1.0, 0.0], [old.id])

        assert [edge.target_id for edge in result.edges] == [new.id]


def test_empty_embedding_is_rejected(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        with pytest.raises(ValueError):
            qa.upsert_anchor(store, "q", SCOPE, [], [])


# ---------------------------------------------------------------------------
# edge weight accumulation
# ---------------------------------------------------------------------------


def test_edge_weight_accumulates_across_reinforcements(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        used = _trace(store, "repeatedly useful node")
        for _ in range(3):
            qa.upsert_anchor(store, "повторяющийся запрос", SCOPE, [1.0, 0.0], [used.id])

        edges = store.list_query_anchor_edges(target_id=used.id)
        assert len(edges) == 1
        assert edges[0].hits == 3
        assert edges[0].weight == pytest.approx(3 * ANCHOR_EDGE_WEIGHT)


def test_edge_weight_saturates_at_one_while_hits_keep_counting(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        anchor = store.insert_query_anchor(
            scope=SCOPE, query="q", fingerprint="fp", embedding=[1.0, 0.0]
        )
        used = _trace(store, "node")
        for _ in range(10):
            edge = store.upsert_query_anchor_edge(anchor.id, used.id, weight=0.4)

        assert edge.weight == pytest.approx(1.0)
        assert edge.hits == 10


def test_anchor_edge_upsert_never_downgrades_unlike_create_connection(
    tmp_path: Path,
) -> None:
    """The control for why anchor edges do not reuse ``create_connection``.

    ``create_connection``'s ``ON CONFLICT ... DO UPDATE SET weight =
    excluded.weight`` clobbers, so a later weaker write silently discards what
    repeated use had built up. The anchor edge upsert must not.
    """

    with _store(tmp_path) as store:
        source = _trace(store, "source node")
        target = _trace(store, "target node")
        store.create_connection(source.id, target.id, "related", weight=0.9)
        store.create_connection(source.id, target.id, "related", weight=0.1)
        clobbered = store.list_connections(
            source_id=source.id, target_id=target.id, relation_type="related"
        )
        assert clobbered[0].weight == pytest.approx(0.1)  # the defect, pinned

        anchor = store.insert_query_anchor(
            scope=SCOPE, query="q", fingerprint="fp", embedding=[1.0, 0.0]
        )
        store.upsert_query_anchor_edge(anchor.id, target.id, weight=0.9)
        edge = store.upsert_query_anchor_edge(anchor.id, target.id, weight=0.1)
        assert edge.weight == pytest.approx(1.0)
        assert edge.weight >= 0.9


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def test_match_anchors_ranks_by_similarity_and_honours_the_floor(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        near_node = _trace(store, "near node")
        far_node = _trace(store, "far node")
        near = qa.upsert_anchor(store, "близкий", SCOPE, [1.0, 0.0], [near_node.id])
        far = qa.upsert_anchor(
            store, "далёкий", SCOPE, _unit_at_cosine(0.5), [far_node.id]
        )

        matches = qa.match_anchors(store, [1.0, 0.0], SCOPE, min_similarity=0.0)
        assert [m.anchor.id for m in matches] == [near.anchor.id, far.anchor.id]
        assert matches[0].similarity == pytest.approx(1.0, abs=1e-6)
        assert matches[1].similarity == pytest.approx(0.5, abs=1e-6)
        assert matches[0].targets == ((near_node.id, ANCHOR_EDGE_WEIGHT),)

        # NamedTuple: the documented (anchor, similarity, targets) shape.
        anchor, similarity, targets = matches[0]
        assert anchor.id == near.anchor.id
        assert similarity > 0.9
        assert targets == ((near_node.id, ANCHOR_EDGE_WEIGHT),)

        assert [m.anchor.id for m in qa.match_anchors(store, [1.0, 0.0], SCOPE)] == [
            near.anchor.id
        ]
        assert qa.match_anchors(store, [1.0, 0.0], SCOPE, min_similarity=1.5) == []
        assert (
            len(qa.match_anchors(store, [1.0, 0.0], SCOPE, limit=1, min_similarity=0.0))
            == 1
        )
        assert qa.match_anchors(store, [1.0, 0.0], SCOPE, limit=0) == []
        assert qa.match_anchors(store, [], SCOPE) == []
        assert qa.match_anchors(store, [0.0, 0.0], SCOPE, min_similarity=0.0) == []


def test_match_anchors_accepts_a_scope_plan(tmp_path: Path) -> None:
    from living_memory.scope import ScopePlan

    with _store(tmp_path) as store:
        local = _trace(store, "local node", scope=SCOPE)
        shared = _trace(store, "global node", scope="global")
        qa.upsert_anchor(store, "локальный", SCOPE, [1.0, 0.0], [local.id])
        qa.upsert_anchor(store, "общий", "global", [1.0, 0.0], [shared.id])

        plan = ScopePlan(requested_scope=SCOPE, scopes=(SCOPE, "global"))
        matched = qa.match_anchors(store, [1.0, 0.0], plan, min_similarity=0.0)
        assert {m.anchor.scope for m in matched} == {SCOPE, "global"}

        narrow = qa.match_anchors(store, [1.0, 0.0], [SCOPE], min_similarity=0.0)
        assert {m.anchor.scope for m in narrow} == {SCOPE}

        everywhere = qa.match_anchors(store, [1.0, 0.0], None, min_similarity=0.0)
        assert len(everywhere) == 2

        assert qa.match_anchors(store, [1.0, 0.0], [], min_similarity=0.0) == []


def test_match_anchors_ignores_anchors_of_a_different_width(tmp_path: Path) -> None:
    """Different vector widths are different spaces, not a truncation problem."""

    with _store(tmp_path) as store:
        node = _trace(store, "node")
        qa.upsert_anchor(store, "двумерный", SCOPE, [1.0, 0.0], [node.id])
        qa.upsert_anchor(store, "трёхмерный", SCOPE, [1.0, 0.0, 0.0], [node.id])

        matches = qa.match_anchors(store, [1.0, 0.0], SCOPE, min_similarity=0.0)
        assert [m.anchor.query for m in matches] == ["двумерный"]

        matches = qa.match_anchors(store, [1.0, 0.0, 0.0], SCOPE, min_similarity=0.0)
        assert [m.anchor.query for m in matches] == ["трёхмерный"]


def test_match_anchors_returns_only_live_targets(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        alive = _trace(store, "still useful")
        doomed = _trace(store, "about to decay")
        anchor = qa.upsert_anchor(
            store, "запрос", SCOPE, [1.0, 0.0], [alive.id, doomed.id]
        ).anchor
        store.soft_delete_node(doomed.id, "ttl expired")

        matches = qa.match_anchors(store, [1.0, 0.0], SCOPE, min_similarity=0.0)
        assert [target for target, _ in matches[0].targets] == [alive.id]
        # The edge row survives the soft delete; only the read filters it.
        assert len(store.list_query_anchor_edges(anchor_id=anchor.id)) == 2


def test_decayed_anchors_do_not_match(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        node = _trace(store, "node")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [node.id]).anchor
        store.decay_query_anchor(anchor.id, "manual")

        assert qa.match_anchors(store, [1.0, 0.0], SCOPE, min_similarity=0.0) == []
        assert store.count_query_anchors(include_decayed=False) == 0
        assert store.count_query_anchors() == 1


# ---------------------------------------------------------------------------
# decay
# ---------------------------------------------------------------------------


def test_stale_anchors_decay_and_a_new_match_revives_them(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        node = _trace(store, "node")
        fresh = qa.upsert_anchor(
            store, "свежий", SCOPE, [1.0, 0.0], [node.id], "2026-08-18T00:00:00Z"
        )
        stale = qa.upsert_anchor(
            store,
            "устаревший",
            SCOPE,
            _unit_at_cosine(0.1),
            [node.id],
            "2026-01-01T00:00:00Z",
        )

        decayed = qa.decay_stale_anchors(store, now="2026-08-18T00:00:00Z")
        assert [a.id for a in decayed] == [stale.anchor.id]
        assert decayed[0].decay_reason == ANCHOR_TTL_DECAY_REASON
        refetched = store.get_query_anchor(fresh.anchor.id)
        assert refetched is not None and refetched.decayed is False

        # Soft: the edges outlive the decay, and asking the question again
        # brings the anchor back with everything it had learned.
        assert len(store.list_query_anchor_edges(anchor_id=stale.anchor.id)) == 1
        revived = qa.upsert_anchor(
            store, "устаревший", SCOPE, _unit_at_cosine(0.1), [node.id]
        )
        assert revived.anchor.id == stale.anchor.id
        assert revived.anchor.decayed is False
        assert revived.anchor.decay_reason is None
        assert store.list_query_anchor_edges(anchor_id=stale.anchor.id)[0].hits == 2


def test_decay_ttl_boundary_and_validation(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        node = _trace(store, "node")
        anchor = qa.upsert_anchor(
            store, "граница", SCOPE, [1.0, 0.0], [node.id], "2026-01-01T00:00:00Z"
        ).anchor
        assert anchor.last_matched_at == "2026-01-01T00:00:00Z"

        # One day short of the TTL: still live. One day past it: decayed.
        inside = f"2026-01-0{1}T00:00:00Z"
        assert qa.decay_stale_anchors(store, now=inside, ttl_days=1) == []
        assert qa.decay_stale_anchors(store, now="2026-01-01T23:00:00Z", ttl_days=1) == []
        decayed = qa.decay_stale_anchors(store, now="2026-01-02T01:00:00Z", ttl_days=1)
        assert [a.id for a in decayed] == [anchor.id]

        # ttl_days=0 decays anything not matched at this very instant, and a
        # negative TTL is a caller error rather than a silent no-op.
        with pytest.raises(ValueError):
            qa.decay_stale_anchors(store, ttl_days=-1)
        assert ANCHOR_TTL_DAYS > 0


# ---------------------------------------------------------------------------
# edge migration: supersedes write paths
# ---------------------------------------------------------------------------


def test_teach_supersede_moves_the_anchor_edge_to_the_replacement(
    tmp_path: Path,
) -> None:
    """The goal's mandatory test: supersede a node an anchor points at."""

    with _store(tmp_path) as store:
        stale = _trace(store, "the deploy needs one server restart")
        anchor = qa.upsert_anchor(
            store, "как деплоить", SCOPE, [1.0, 0.0], [stale.id]
        ).anchor
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            stale.id
        ]

        result = memory_teach(store, stale.id, "the deploy needs TWO server restarts")

        edges = store.list_query_anchor_edges(anchor_id=anchor.id)
        assert [e.target_id for e in edges] == [result.corrective_trace.id]
        assert edges[0].weight == pytest.approx(ANCHOR_EDGE_WEIGHT)
        assert edges[0].hits == 1
        # And the anchor now answers with the correction, not the stale node.
        match = qa.match_anchors(store, [1.0, 0.0], SCOPE, min_similarity=0.0)[0]
        assert [target for target, _ in match.targets] == [result.corrective_trace.id]


def test_duplicate_content_dedup_moves_the_anchor_edge(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        first = _trace(store, "identical remembered content")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [first.id]).anchor

        second = _trace(store, "identical remembered content")

        assert store.get_node(first.id).decayed is True
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            second.id
        ]


def test_derived_r1c_correction_moves_the_anchor_edge(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        stale = _trace(store, "The runner lock file lives in /tmp/runner.lock")
        anchor = qa.upsert_anchor(
            store, "где лок-файл раннера", SCOPE, [1.0, 0.0], [stale.id]
        ).anchor
        correction = _trace(
            store,
            f"CORRECTION to trace {stale.id}: the lock file lives in /var/lock, NOT /tmp",
        )

        derive_edges_for_new_trace(store, correction)

        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            correction.id
        ]


def test_supersedes_chain_leaves_edges_on_the_latest_node(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        first = _trace(store, "generation one")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [first.id]).anchor
        second = _trace(store, "generation two")
        store.create_connection(second.id, first.id, "supersedes")
        third = _trace(store, "generation three")
        store.create_connection(third.id, second.id, "supersedes")

        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            third.id
        ]
        assert qa.resolve_replacement(store, first.id) == third.id


def test_migration_merges_weights_when_the_anchor_already_knows_the_replacement(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        old = _trace(store, "older answer")
        new = _trace(store, "newer answer")
        anchor = qa.upsert_anchor(
            store, "запрос", SCOPE, [1.0, 0.0], [old.id, new.id]
        ).anchor
        # Reinforce the old edge so the merge has something to lose.
        store.upsert_query_anchor_edge(anchor.id, old.id, weight=ANCHOR_EDGE_WEIGHT)

        store.create_connection(new.id, old.id, "supersedes")

        edges = store.list_query_anchor_edges(anchor_id=anchor.id)
        assert len(edges) == 1
        assert edges[0].target_id == new.id
        # 0.25 (new) + 0.50 (old, twice reinforced) — merged, not overwritten.
        assert edges[0].weight == pytest.approx(3 * ANCHOR_EDGE_WEIGHT)
        assert edges[0].hits == 3


def test_migration_refuses_self_loops_cycles_and_dead_destinations(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        left = _trace(store, "left node")
        right = _trace(store, "right node")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [left.id]).anchor

        assert qa.migrate_anchor_edges(store, left.id, left.id) == 0
        assert qa.migrate_anchor_edges(store, left.id, "NOSUCHNODE") == 0

        gone = _trace(store, "decayed destination")
        store.soft_delete_node(gone.id, "ttl expired")
        assert qa.migrate_anchor_edges(store, left.id, gone.id) == 0

        # Build a supersedes cycle. Each write-path hook moves the edge exactly
        # once, onto the node the edge it just saw declares to be the
        # replacement — deterministic and terminating, never a loop.
        store.create_connection(right.id, left.id, "supersedes")
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            right.id
        ]
        store.create_connection(left.id, right.id, "supersedes")
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            left.id
        ]

        # The explicit entry point is the one that must refuse: with the cycle
        # in place neither side is "later", so re-pointing would bounce the
        # edges back and forth forever.
        assert qa.migrate_anchor_edges(store, left.id, right.id) == 0
        assert qa.migrate_anchor_edges(store, right.id, left.id) == 0
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            left.id
        ]
        # And the walk terminates rather than looping.
        assert qa.resolve_replacement(store, left.id) == right.id
        assert qa.resolve_replacement(store, right.id) == left.id
        # So does the sweep, which must not move anything into a cycle.
        assert qa.sweep_anchor_edge_migration(store).edges_moved == 0


def test_self_superseding_edge_does_not_move_anything(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        node = _trace(store, "node")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [node.id]).anchor
        store.connection.execute(
            """
            INSERT INTO connections (id, source_id, target_id, type, weight,
                                     metadata, created_at, updated_at)
            VALUES ('SELFEDGE', ?, ?, 'supersedes', 1.0, '{}',
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """,
            (node.id, node.id),
        )
        store.connection.commit()

        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            node.id
        ]
        assert qa.resolve_replacement(store, node.id) is None


# ---------------------------------------------------------------------------
# edge migration: era displacement and sweep
# ---------------------------------------------------------------------------


def test_era_exclusions_move_edges_for_the_reasons_that_name_a_replacement(
    tmp_path: Path,
) -> None:
    """Consolidation writes no edge for era displacement, so it is passed in."""

    with _store(tmp_path) as store:
        displaced = _trace(store, "member folded into a concept")
        concept = store.create_node(
            level="concept", content="the consolidated concept", context={"scope": SCOPE}
        )
        replacement = _trace(store, "the node that replaced the concept")
        anchor = qa.upsert_anchor(
            store, "запрос", SCOPE, [1.0, 0.0], [displaced.id]
        ).anchor
        store.create_connection(replacement.id, concept.id, "supersedes")

        moved = qa.migrate_anchor_edges_for_exclusions(
            store, {displaced.id: f"displaced-via:{concept.id}"}
        )

        assert moved == 1
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            replacement.id
        ]


def test_era_exclusions_without_a_named_replacement_leave_edges_alone(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        member = _trace(store, "corrected member")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [member.id]).anchor

        moved = qa.migrate_anchor_edges_for_exclusions(
            store, {member.id: "corrected", "OTHER": "obsolete-era"}
        )

        assert moved == 0
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            member.id
        ]


def test_era_exclusion_superseded_by_reason_moves_edges(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        member = _trace(store, "superseded member")
        replacement = _trace(store, "replacement")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [member.id]).anchor

        moved = qa.migrate_anchor_edges_for_exclusions(
            store, {member.id: f"superseded-by:{replacement.id}"}
        )

        assert moved == 1
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            replacement.id
        ]


def test_sweep_repairs_edges_written_before_the_hook_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """The backfill's form: edges land on targets superseded long ago."""

    with _store(tmp_path) as store:
        stale = _trace(store, "generation one")
        newer = _trace(store, "generation two")
        store.create_connection(newer.id, stale.id, "supersedes")

        # Write the edge behind the hook's back, exactly as a pre-v7 backfill
        # or a database written before this change would have left it.
        anchor = store.insert_query_anchor(
            scope=SCOPE, query="запрос", fingerprint="fp", embedding=[1.0, 0.0]
        )
        store.upsert_query_anchor_edge(anchor.id, stale.id, weight=ANCHOR_EDGE_WEIGHT)

        first = qa.sweep_anchor_edge_migration(store)
        assert first.targets_examined == 1
        assert first.targets_migrated == 1
        assert first.edges_moved == 1
        assert first.migrations == ((stale.id, newer.id),)
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            newer.id
        ]

        second = qa.sweep_anchor_edge_migration(store)
        assert second.targets_migrated == 0
        assert second.edges_moved == 0


def test_sweep_leaves_a_healthy_graph_untouched(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        node = _trace(store, "current node")
        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [node.id]).anchor

        sweep = qa.sweep_anchor_edge_migration(store)

        assert sweep.targets_examined == 1
        assert sweep.targets_migrated == 0
        assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [
            node.id
        ]


def test_query_anchor_revision_moves_exactly_when_the_matrix_changes(
    tmp_path: Path,
) -> None:
    """The cache key a caller may rely on to skip re-scanning the vectors."""

    with _store(tmp_path) as store:
        node = _trace(store, "node")
        empty = store.query_anchor_revision()

        anchor = qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [node.id]).anchor
        after_insert = store.query_anchor_revision()
        assert after_insert != empty

        # A reinforcement rewrites no vector, so the matrix is still current.
        qa.upsert_anchor(store, "запрос", SCOPE, [1.0, 0.0], [node.id])
        assert store.query_anchor_revision() == after_insert

        # Decay removes a row from the live set, revival puts it back.
        store.decay_query_anchor(anchor.id, "manual")
        after_decay = store.query_anchor_revision()
        assert after_decay != after_insert
        store.reinforce_query_anchor(anchor.id)
        assert store.query_anchor_revision() == after_insert

        # A second anchor moves it again.
        qa.upsert_anchor(store, "другой запрос", SCOPE, [0.0, 1.0], [node.id])
        assert store.query_anchor_revision() != after_insert

        store.connection.executescript(
            f"DROP TABLE {QUERY_ANCHOR_EDGE_TABLE}; DROP TABLE {QUERY_ANCHOR_TABLE};"
        )
        store._invalidate_schema_shape_cache()
        assert store.query_anchor_revision() == ()


def test_anchor_edge_rows_are_unique_per_anchor_target_pair(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        anchor = store.insert_query_anchor(
            scope=SCOPE, query="q", fingerprint="fp", embedding=[1.0, 0.0]
        )
        node = _trace(store, "node")
        store.upsert_query_anchor_edge(anchor.id, node.id, weight=0.1)
        store.upsert_query_anchor_edge(anchor.id, node.id, weight=0.1)

        assert store.count_query_anchor_edges() == 1

        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                f"""
                INSERT INTO {QUERY_ANCHOR_EDGE_TABLE}
                    (anchor_id, target_id, weight, hits, created_at, updated_at)
                VALUES (?, ?, 0.1, 1, 'x', 'x')
                """,
                (anchor.id, node.id),
            )


def test_anchor_scope_is_normalized_on_write_and_on_read(tmp_path: Path) -> None:
    """A raw scope string must not produce an anchor no scoped read can see."""

    with _store(tmp_path) as store:
        anchor = store.insert_query_anchor(
            scope="demo", query="q", fingerprint="fp", embedding=[1.0, 0.0]
        )
        assert anchor.scope == "project:demo"
        assert store.find_query_anchor("demo", "fp") is not None
        assert store.find_query_anchor("project:demo", "fp") is not None
        assert [a.id for a in store.list_query_anchors(scope="demo")] == [anchor.id]
        matched = qa.match_anchors(store, [1.0, 0.0], "demo", min_similarity=0.0)
        assert [m.anchor.id for m in matched] == [anchor.id]


def test_anchor_fingerprint_is_unique_per_scope(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        store.insert_query_anchor(
            scope=SCOPE, query="q", fingerprint="fp", embedding=[1.0, 0.0]
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_query_anchor(
                scope=SCOPE, query="q again", fingerprint="fp", embedding=[0.0, 1.0]
            )
        # A different scope is a different anchor, not a conflict.
        store.insert_query_anchor(
            scope="global", query="q", fingerprint="fp", embedding=[1.0, 0.0]
        )
        assert store.count_query_anchors() == 2
