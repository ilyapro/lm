"""Synthetic pre-ledger schema and precise compatibility assertions.

Ledger compatibility requires byte identity for every unrelated object.
For the six rebuilt FTS objects, accept only the known old-to-new definitions
(or formatting of the new definition on subsequent opens).
This is deliberately not a blanket exemption for ``nodes_fts*``.
"""

from pathlib import Path
import sqlite3
from typing import Any


PENDING_RECALL_INDEXES = {
    "idx_recall_pending_session",
    "idx_recall_pending_task_session",
    "idx_recall_pending_transport",
    "idx_recall_pending_transport_task_session",
    "idx_recall_pending_transport_session",
    "idx_recall_events_requested_scope_pending_created",
}

_ADDITIVE_TABLES = {
    "transcript_grounding_verdicts",
    "recall_credit_ledger",
    "recall_explicit_credit",
    "recall_feedback_marks",
    "consolidation_embeddings",
}
ADDITIVE_OBJECTS = _ADDITIVE_TABLES | PENDING_RECALL_INDEXES | {
    "idx_transcript_grounding_node_graded",
    "idx_recall_feedback_marks_event",
}


_LEGACY_FTS = {
    "nodes_fts": """CREATE VIRTUAL TABLE nodes_fts USING fts5(
        node_id UNINDEXED, content, level UNINDEXED,
        scope UNINDEXED, tokenize = 'unicode61' )""",
    "nodes_fts_content": (
        "CREATE TABLE 'nodes_fts_content'(id INTEGER PRIMARY KEY, c0, c1, c2, c3)"
    ),
    "nodes_fts_insert": """CREATE TRIGGER nodes_fts_insert
        AFTER INSERT ON nodes BEGIN
        INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
        VALUES (new.rowid, new.id, new.content, new.level, new.scope); END""",
    "nodes_fts_delete": """CREATE TRIGGER nodes_fts_delete
        AFTER DELETE ON nodes BEGIN
        DELETE FROM nodes_fts WHERE rowid = old.rowid; END""",
    "nodes_fts_update": """CREATE TRIGGER nodes_fts_update
        AFTER UPDATE OF content, level, scope ON nodes BEGIN
        DELETE FROM nodes_fts WHERE rowid = old.rowid;
        INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
        VALUES (new.rowid, new.id, new.content, new.level, new.scope); END""",
    "nodes_fts_vocab": (
        "CREATE VIRTUAL TABLE nodes_fts_vocab USING fts5vocab('nodes_fts', 'row')"
    ),
}


def _normalized(sql: str) -> str:
    return " ".join(sql.split())


_TASK_FTS = {
    name: _normalized(sql)
    .replace("content, level", "content, task, level")
    .replace("new.content, new.level", "new.content, new.task, new.level")
    .replace("c2, c3)", "c2, c3, c4)")
    .replace("'row')", "'col')")
    for name, sql in _LEGACY_FTS.items()
}


def schema_objects(db_path: Path) -> dict[str, str]:
    """Named SQLite objects with their verbatim DDL."""

    with sqlite3.connect(db_path) as conn:
        return dict(conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"
        ).fetchall())


def schema_version(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]


def memory_rows(conn: sqlite3.Connection) -> dict[str, list[tuple[Any, ...]]]:
    """Exact source rows, including SQLite row identities."""

    return {
        table: [tuple(row) for row in conn.execute(f"SELECT rowid, * FROM {table} ORDER BY id")]
        for table in ("nodes", "recall_events")
    }


def prepare_pre_ledger_schema(db_path: Path) -> dict[str, str]:
    """Give seeded current rows a literal pre-task FTS and no additive ledgers.

    This covers the specified missing objects and FTS shape, not every schema
    that an older runtime might have produced. No MemoryStore opens this input
    until the caller exercises the migration.
    """

    with sqlite3.connect(db_path) as conn:
        rows = memory_rows(conn)
        for name in sorted(_ADDITIVE_TABLES):
            conn.execute(f"DROP TABLE {name}")
        for name in sorted(PENDING_RECALL_INDEXES):
            conn.execute(f"DROP INDEX {name}")
        conn.execute("""CREATE INDEX idx_recall_pending_agent
            ON recall_events(COALESCE(agent, ''), created_at DESC)
            WHERE feedback_applied = 0""")
        for name in ("nodes_fts_insert", "nodes_fts_delete", "nodes_fts_update"):
            conn.execute(f"DROP TRIGGER {name}")
        conn.execute("DROP TABLE nodes_fts_vocab")
        conn.execute("DROP TABLE nodes_fts")
        for name, sql in _LEGACY_FTS.items():
            if name != "nodes_fts_content":  # FTS5 creates its own shadow table.
                conn.execute(sql)
        conn.execute("""INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
            SELECT rowid, id, content, level, scope FROM nodes""")
        assert memory_rows(conn) == rows

    before = schema_objects(db_path)
    assert ADDITIVE_OBJECTS.isdisjoint(before)
    assert "idx_recall_pending_agent" in before
    for name, sql in _LEGACY_FTS.items():
        assert _normalized(before[name]) == _normalized(sql)
    assert schema_version(db_path) == "8"
    return before


def assert_schema_upgrade(before: dict[str, str], after: dict[str, str]) -> None:
    """Opening must create every missing object and fully rebuild task FTS."""

    assert_preserved_schema(before, after)
    assert set(after) - set(before) == ADDITIVE_OBJECTS
    for name, sql in _TASK_FTS.items():
        assert _normalized(after[name]) == sql


def assert_preserved_schema(before: dict[str, str], after: dict[str, str]) -> None:
    """Preserve DDL except task FTS and the unused agent-only pending index."""

    for name, sql in before.items():
        if name == "idx_recall_pending_agent":
            assert name not in after
            continue
        assert name in after, f"{name} disappeared"
        if after[name] == sql:
            continue
        assert name in _TASK_FTS, f"{name} DDL changed:\n{sql!r}\n->\n{after[name]!r}"
        assert _normalized(sql) in {_normalized(_LEGACY_FTS[name]), _TASK_FTS[name]}, (
            f"{name} has an unrecognized pre-migration definition: {sql!r}"
        )
        assert _normalized(after[name]) == _TASK_FTS[name], (
            f"{name} differs from the task-index migration: {after[name]!r}"
        )
