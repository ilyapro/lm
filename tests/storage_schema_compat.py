"""Narrow DDL allowance for the derived task-search index migration.

Ledger compatibility still requires byte identity for every unrelated object.
For the six rebuilt FTS objects, accept only the known old-to-new definitions
(or formatting of the new definition after master has received the migration).
This is deliberately not a blanket exemption for ``nodes_fts*``.
"""

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


def assert_preserved_schema(before: dict[str, str], after: dict[str, str]) -> None:
    """Preserve all existing DDL except the exact task-FTS migration delta."""

    for name, sql in before.items():
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
