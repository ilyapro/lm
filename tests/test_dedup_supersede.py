"""Contract tests for content-fingerprint dedup of identical traces.

Covers the write-path dedup behavior introduced for schema v3: identical
``content`` in the same scope/level collapses to one active trace plus an
append-only chain of decayed older copies linked by ``supersedes`` edges.
History remains queryable via ``get_node`` and the supersedes edges.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore


def _fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_same_scope_identical_content_supersedes_older_trace(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        first = store.append_trace(
            "bootstrap chunk hello",
            {"scope": "project:demo", "agent": "a"},
        )
        second = store.append_trace(
            "bootstrap chunk hello",
            {"scope": "project:demo", "agent": "b"},
        )

        refetched_first = store.get_node(first.id)
        refetched_second = store.get_node(second.id)
        assert refetched_first is not None
        assert refetched_second is not None
        assert refetched_first.decayed is True
        assert refetched_first.decay_reason == "duplicate_content"
        assert refetched_second.decayed is False

        active = store.list_nodes(level="trace", scope="project:demo")
        assert [n.id for n in active] == [second.id]


def test_supersedes_connection_has_dedup_metadata(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        first = store.append_trace(
            "supersede edge content",
            {"scope": "project:demo"},
        )
        second = store.append_trace(
            "supersede edge content",
            {"scope": "project:demo"},
        )

        edges = store.list_connections(source_id=second.id, relation_type="supersedes")
        assert len(edges) == 1
        edge = edges[0]
        assert edge.target_id == first.id
        assert edge.metadata.get("kind") == "duplicate_content"
        assert edge.metadata.get("fingerprint") == _fingerprint(
            "supersede edge content"
        )


def test_cross_scope_identical_content_is_not_deduped(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        a = store.append_trace(
            "shared content across scopes",
            {"scope": "project:alpha"},
        )
        b = store.append_trace(
            "shared content across scopes",
            {"scope": "project:beta"},
        )

        assert store.get_node(a.id).decayed is False
        assert store.get_node(b.id).decayed is False
        assert store.list_connections(source_id=b.id, relation_type="supersedes") == []


def test_cross_level_identical_content_is_not_deduped(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "level-overlap content",
            {"scope": "project:demo"},
        )
        concept = store.create_node(
            level="concept",
            content="level-overlap content",
            context={"scope": "project:demo"},
            stats={"confidence": 0.9, "unique_agents": 2},
        )

        assert store.get_node(trace.id).decayed is False
        assert store.get_node(concept.id).decayed is False
        # No supersedes edge should be created across levels.
        edges = store.list_connections(source_id=concept.id, relation_type="supersedes")
        assert edges == []


def test_dedup_handles_file_chunk_shaped_content(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        header = json.dumps(
            {
                "path": "src/foo.py",
                "kind": "code",
                "language": "python",
                "sha256": "deadbeef" * 8,
                "chunk": "1/2",
                "lines": "1-40",
            },
            sort_keys=True,
        )
        body = "```python\ndef foo():\n    return 42\n```"
        content = f"[file-chunk] {header}\n{body}"

        first = store.append_trace(content, {"scope": "project:ae"})
        second = store.append_trace(content, {"scope": "project:ae"})

        assert store.get_node(first.id).decayed is True
        assert store.get_node(second.id).decayed is False
        edges = store.list_connections(source_id=second.id, relation_type="supersedes")
        assert [e.target_id for e in edges] == [first.id]


def test_dedup_handles_file_summary_shaped_content(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        header = json.dumps(
            {
                "path": "src/bar.py",
                "language": "python",
                "sha256": "feedface" * 8,
                "summary": "module entry",
            },
            sort_keys=True,
        )
        content = f"[file-summary] {header}"

        first = store.append_trace(content, {"scope": "project:ae"})
        second = store.append_trace(content, {"scope": "project:ae"})

        assert store.get_node(first.id).decayed is True
        assert store.get_node(second.id).decayed is False


def test_dedup_handles_project_overview_shaped_content(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        payload = json.dumps(
            {"files": 12, "languages": {"python": 8, "markdown": 4}},
            sort_keys=True,
        )
        content = f"[project-overview] {payload}"

        first = store.append_trace(content, {"scope": "project:ae"})
        second = store.append_trace(content, {"scope": "project:ae"})

        active = store.list_nodes(level="trace", scope="project:ae")
        assert [n.id for n in active] == [second.id]
        assert store.get_node(first.id).decayed is True


def test_n_repeated_inserts_yield_one_active_n_minus_one_decayed(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = [
            store.append_trace(
                "repeated bootstrap content",
                {"scope": "project:demo", "agent": f"agent-{i}"},
            )
            for i in range(10)
        ]

        active = store.list_nodes(
            level="trace", scope="project:demo", include_decayed=False
        )
        assert [n.id for n in active] == [nodes[-1].id]

        for older in nodes[:-1]:
            refetched = store.get_node(older.id)
            assert refetched.decayed is True
            assert refetched.decay_reason == "duplicate_content"

        # 9 supersedes edges total: one from each new insert to every still-
        # active older trace at the moment of insertion.
        edge_count = store.connection.execute(
            "SELECT COUNT(*) AS c FROM connections WHERE type = 'supersedes'"
        ).fetchone()["c"]
        assert int(edge_count) == 9


def test_dedup_preserves_original_content_in_store(tmp_path: Path) -> None:
    """History is preserved: superseded rows remain queryable with original content."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        original = store.append_trace(
            "history-preservation payload",
            {"scope": "project:demo", "agent": "a"},
        )
        replacement = store.append_trace(
            "history-preservation payload",
            {"scope": "project:demo", "agent": "b"},
        )

        # Raw id-based fetch still returns the row with its original content.
        old = store.get_node(original.id)
        assert old is not None
        assert old.content == "history-preservation payload"
        assert old.decayed is True

        # include_decayed=True surfaces both rows (full history).
        history = store.list_nodes(
            level="trace", scope="project:demo", include_decayed=True
        )
        assert {n.id for n in history} == {original.id, replacement.id}

        # The supersedes edge traversal recovers the chain.
        edges = store.list_connections(
            source_id=replacement.id, relation_type="supersedes"
        )
        assert [e.target_id for e in edges] == [original.id]


def test_dedup_excludes_already_decayed_traces(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        original = store.append_trace(
            "decayed-by-hand content",
            {"scope": "project:demo"},
        )
        store.soft_delete_node(original.id, "manual")

        replacement = store.append_trace(
            "decayed-by-hand content",
            {"scope": "project:demo"},
        )

        # The manually-decayed trace stays decayed with its original reason,
        # and the new trace must NOT create a supersedes edge to it.
        old = store.get_node(original.id)
        assert old.decayed is True
        assert old.decay_reason == "manual"
        edges = store.list_connections(
            source_id=replacement.id, relation_type="supersedes"
        )
        assert edges == []
        assert store.get_node(replacement.id).decayed is False


def test_memory_recall_returns_only_active_traces_after_dedup(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for _ in range(5):
            store.append_trace(
                "lighthouse bootstrap fixture",
                {"scope": "project:demo", "agent": "agent-a"},
            )

        results = memory_recall(
            store,
            "lighthouse bootstrap fixture",
            scope="project:demo",
            max_results=10,
        )
        # Recall surfaces the active trace only; the four superseded copies are
        # decayed and excluded from active retrieval. The dedup chain's active
        # head is the superseding end of its edges, never a supersedes target,
        # so it must not carry the superseded flag.
        assert len(results) == 1
        assert results[0].superseded is False
