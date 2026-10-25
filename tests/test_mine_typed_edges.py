"""Offline edge proposals use the same correction semantics as runtime reads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_typed_edges.py"
_SPEC = importlib.util.spec_from_file_location("mine_typed_edges", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
miner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = miner
_SPEC.loader.exec_module(miner)


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize(
    "metadata",
    [
        {"kind": "drain_near_duplicate"},
        {"kind": "duplicate_content", "basis": "procedure_key"},
        {"by": "lm_collapse_near_dups.py", "reason": "near-duplicate collapse"},
    ],
)
def test_technical_edge_cannot_support_transitive_correction(position, metadata) -> None:
    edges = [("new", "middle", "supersedes", {}), ("middle", "old", "supersedes", {})]
    src, dst, relation, _ = edges[position]
    edges[position] = src, dst, relation, metadata

    assert miner.rule_supersedes_transitive({}, edges) == []


def test_explicit_and_unknown_corrections_remain_eligible() -> None:
    edges = [("new", "middle", "supersedes", {"kind": "teach"}),
             ("middle", "old", "supersedes", {"kind": "unknown"})]

    candidates = miner.rule_supersedes_transitive({}, edges)

    assert [(item.src, item.dst, item.type) for item in candidates] == [("new", "old", "supersedes")]


def test_edge_inventory_excludes_historical_heuristic_corrections() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE connections (source_id TEXT, target_id TEXT, type TEXT, metadata TEXT)")
        connection.executemany(
            "INSERT INTO connections VALUES (?, ?, ?, ?)",
            [("a", "b", "supersedes", json.dumps({"kind": "drain_near_duplicate"})),
             ("b", "c", "supersedes", json.dumps({"kind": "teach"})),
             ("c", "d", "related", json.dumps({"kind": "content_reference"}))],
        )

        edges = miner.load_edges(connection)

    assert [(src, dst, relation) for src, dst, relation, _ in edges] == [
        ("b", "c", "supersedes"), ("c", "d", "related")
    ]
