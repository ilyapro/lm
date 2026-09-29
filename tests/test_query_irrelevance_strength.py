"""Demotion-strength valves and the strength census (goal recall-precision, P3).

* ``LM_QUERY_IRRELEVANCE_FULL_COSINE`` / ``LM_QUERY_IRRELEVANCE_MARK_WEIGHT``
  unset (or invalid): ``query_demotions`` returns exactly what the original
  formula ``1 - (1-F) * min(1, w) * (cos - 0.60) / 0.40`` gives.
* Set: the curve steepens as documented; stored rows are never rewritten.
* ``scripts/query_demotion_strength_census.py`` counts only (event x node)
  cases with a row created before the event on a matched in-scope anchor,
  and opens the store read-only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import living_memory.irrelevance as irrelevance
from living_memory.irrelevance import (
    QUERY_IRRELEVANCE_SCHEMA_SQL,
    demotion_closeness,
    effective_row_weight,
    query_demotions,
    query_irrelevance_full_cosine,
    query_irrelevance_mark_weight,
)

VALVES = (
    "LM_QUERY_IRRELEVANCE_FACTOR",
    "LM_QUERY_IRRELEVANCE_FULL_COSINE",
    "LM_QUERY_IRRELEVANCE_MARK_WEIGHT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VALVES:
        monkeypatch.delenv(name, raising=False)


def _store(rows: list[tuple[str, str, float]]) -> Any:
    conn = sqlite3.connect(":memory:")
    conn.execute(QUERY_IRRELEVANCE_SCHEMA_SQL)
    conn.executemany(
        "INSERT INTO query_irrelevance (anchor_id, node_id, weight, marks, created_at, updated_at)"
        " VALUES (?, ?, ?, 1, 't', 't')",
        rows,
    )
    return SimpleNamespace(connection=conn)


def _patch_matches(monkeypatch: pytest.MonkeyPatch, sims: dict[str, float]) -> None:
    matches = [
        SimpleNamespace(anchor=SimpleNamespace(id=anchor_id), similarity=sim)
        for anchor_id, sim in sims.items()
    ]
    monkeypatch.setattr(irrelevance, "match_anchors", lambda *a, **k: list(matches))


def _old_formula(rows, sims, factor=0.5):
    # The pre-valve implementation, verbatim arithmetic.
    span = max(1e-9, 1.0 - 0.60)
    closeness = {a: min(1.0, max(0.0, (s - 0.60) / span)) for a, s in sims.items()}
    strongest: dict[str, float] = {}
    for anchor_id, node_id, weight in rows:
        if weight <= 0.0 or anchor_id not in closeness:
            continue
        strength = min(1.0, max(0.0, float(weight))) * closeness[anchor_id]
        if strength > strongest.get(node_id, 0.0):
            strongest[node_id] = strength
    return {n: 1.0 - (1.0 - factor) * s for n, s in strongest.items() if s > 0.0}


def _random_case(seed: int):
    rng = random.Random(seed)
    sims = {f"a{i}": rng.uniform(0.6, 1.0) for i in range(rng.randint(1, 5))}
    rows = [
        (rng.choice(list(sims)), f"n{rng.randint(0, 6)}", rng.choice([0.0, 0.5, 1.0, 0.25]))
        for _ in range(rng.randint(1, 12))
    ]
    rows = list({(a, n): (a, n, w) for a, n, w in rows}.values())
    return rows, sims


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize(
    "env",
    [
        {},
        {"LM_QUERY_IRRELEVANCE_FULL_COSINE": ""},
        {"LM_QUERY_IRRELEVANCE_FULL_COSINE": "0.55"},  # at/below floor: ignored
        {"LM_QUERY_IRRELEVANCE_FULL_COSINE": "1.5"},
        {"LM_QUERY_IRRELEVANCE_FULL_COSINE": "nan"},
        {"LM_QUERY_IRRELEVANCE_MARK_WEIGHT": "0"},
        {"LM_QUERY_IRRELEVANCE_MARK_WEIGHT": "abc"},
        {"LM_QUERY_IRRELEVANCE_MARK_WEIGHT": "2"},
    ],
)
def test_unset_or_invalid_valves_keep_output_identical(
    monkeypatch: pytest.MonkeyPatch, seed: int, env: dict[str, str]
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    rows, sims = _random_case(seed)
    _patch_matches(monkeypatch, sims)
    assert query_demotions(_store(rows), [1.0], None) == _old_formula(rows, sims)


def test_valve_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    assert query_irrelevance_full_cosine() is None
    assert query_irrelevance_mark_weight() is None
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FULL_COSINE", " 0.75 ")
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_MARK_WEIGHT", "1.0")
    assert query_irrelevance_full_cosine() == 0.75
    assert query_irrelevance_mark_weight() == 1.0
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FULL_COSINE", "0.60")
    assert query_irrelevance_full_cosine() is None


def test_full_cosine_steepens_closeness(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [("a", "n", 0.5)]
    _patch_matches(monkeypatch, {"a": 0.675})
    store = _store(rows)
    base = query_demotions(store, [1.0], None)["n"]
    assert base == pytest.approx(1 - 0.5 * 0.5 * (0.075 / 0.40))
    assert base > 0.9
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FULL_COSINE", "0.75")
    steep = query_demotions(store, [1.0], None)["n"]
    assert steep == pytest.approx(1 - 0.5 * 0.5 * 0.5)
    # Past the full cosine closeness saturates; the floor F still bounds it.
    _patch_matches(monkeypatch, {"a": 0.9})
    assert query_demotions(store, [1.0], None)["n"] == pytest.approx(0.75)


def test_mark_weight_reads_one_mark_as_full(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [("a", "one", 0.5), ("a", "two", 1.0)]
    _patch_matches(monkeypatch, {"a": 1.0})
    store = _store(rows)
    assert query_demotions(store, [1.0], None) == {"one": 0.75, "two": 0.5}
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_MARK_WEIGHT", "1.0")
    assert query_demotions(store, [1.0], None) == {"one": 0.5, "two": 0.5}
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_MARK_WEIGHT", "0.25")
    got = query_demotions(store, [1.0], None)
    assert got["one"] == pytest.approx(1 - 0.5 * 0.25) and got["two"] == pytest.approx(0.75)
    # Read side only: stored rows untouched.
    stored = store.connection.execute(
        "SELECT node_id, weight FROM query_irrelevance ORDER BY node_id"
    ).fetchall()
    assert stored == [("one", 0.5), ("two", 1.0)]


def test_bounded_below_by_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FULL_COSINE", "0.61")
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_MARK_WEIGHT", "1.0")
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", "0.4")
    _patch_matches(monkeypatch, {"a": 0.99})
    assert query_demotions(_store([("a", "n", 1.0)]), [1.0], None) == {"n": pytest.approx(0.4)}
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", "1.0")
    assert query_demotions(_store([("a", "n", 1.0)]), [1.0], None) == {}


def test_helpers_defaults() -> None:
    assert demotion_closeness(0.60) == 0.0
    assert demotion_closeness(1.0) == 1.0
    assert demotion_closeness(0.70, full_cosine=0.80) == pytest.approx(0.5)
    assert effective_row_weight(0.7) == 0.7
    assert effective_row_weight(1.3) == 1.0
    assert effective_row_weight(0.5, 0.75) == 0.75


# ---------------------------------------------------------------------------
# Census script
# ---------------------------------------------------------------------------


def _load_census():
    path = Path(__file__).resolve().parents[1] / "scripts" / "query_demotion_strength_census.py"
    spec = importlib.util.spec_from_file_location("query_demotion_strength_census", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unit(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    return list(array / np.linalg.norm(array))


def _vec_at(cos: float) -> list[float]:
    """Unit vector with cosine ``cos`` to e1."""
    return [cos, (1 - cos * cos) ** 0.5, 0.0]


def _fixture_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE query_anchors (id TEXT PRIMARY KEY, scope TEXT, query TEXT,
            fingerprint TEXT, dimensions INTEGER, embedding BLOB, decayed INTEGER DEFAULT 0,
            created_at TEXT);
        CREATE TABLE recall_events (id TEXT PRIMARY KEY, query TEXT, scope TEXT,
            resolved_scopes TEXT, results TEXT, created_at TEXT);
        """
    )
    conn.execute(QUERY_IRRELEVANCE_SCHEMA_SQL)
    anchors = [
        ("A1", "project:x", [1.0, 0.0, 0.0], "2026-09-27T10:00:00Z"),
        ("A2", "project:other", [1.0, 0.0, 0.0], "2026-09-27T10:00:00Z"),  # out of scope
        ("A3", "project:x", [0.0, 0.0, 1.0], "2026-09-27T10:00:00Z"),  # orthogonal
    ]
    for anchor_id, scope, vec, created in anchors:
        conn.execute(
            "INSERT INTO query_anchors VALUES (?, ?, 'q', 'f', 3, ?, 0, ?)",
            (anchor_id, scope, np.asarray(vec, dtype="<f4").tobytes(), created),
        )
    rows = [
        # anchor, node, weight, marks, cancels, created, updated
        ("A1", "n1", 0.5, 1, 0, "2026-09-27T11:00:00Z", "2026-09-27T11:00:00Z"),
        ("A1", "n2", 1.0, 2, 0, "2026-09-27T11:00:00Z", "2026-09-27T13:00:00Z"),
        ("A1", "late", 0.5, 1, 0, "2026-09-27T20:00:00Z", "2026-09-27T20:00:00Z"),
        ("A1", "gone", 0.0, 1, 1, "2026-09-27T11:00:00Z", "2026-09-27T11:30:00Z"),
        ("A2", "n3", 1.0, 2, 0, "2026-09-27T11:00:00Z", "2026-09-27T11:00:00Z"),
        ("A3", "n4", 1.0, 2, 0, "2026-09-27T11:00:00Z", "2026-09-27T11:00:00Z"),
    ]
    for row in rows:
        conn.execute(
            "INSERT INTO query_irrelevance (anchor_id, node_id, weight, marks, cancels,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    events = [
        ("E1", "q-close", '["project:x"]', "n1", "2026-09-27T12:00:00Z"),
        ("E2", "q-far", '["project:x","global"]', "", "2026-09-27T14:00:00Z"),
        ("E3", "q-below", '["project:x"]', "", "2026-09-27T14:00:00Z"),
    ]
    for event_id, query, scopes, delivered, created in events:
        results = json.dumps([{"node_id": delivered}] if delivered else [])
        conn.execute(
            "INSERT INTO recall_events VALUES (?, ?, 'project:x', ?, ?, ?)",
            (event_id, query, scopes, results, created),
        )
    conn.commit()
    conn.close()


EMBED = {"q-close": _vec_at(0.70), "q-far": _vec_at(0.95), "q-below": _vec_at(0.55)}


def test_census_counts_only_applicable_cases(tmp_path: Path) -> None:
    census = _load_census()
    db = tmp_path / "store.sqlite3"
    _fixture_db(db)
    settings = [census.parse_setting("default:"), census.parse_setting("steep:full=0.70,mw=1.0")]
    conn = census._connect_ro(db)
    result = census.census(conn, settings, lambda texts: [EMBED[t] for t in texts])
    # E1 (12:00, cos .70): n1 (w .5), n2 (still one mark: second at 13:00),
    #   gone (cancelled at 11:30 -> not live). E2 (14:00, cos .95): n1, n2 (w 1).
    # E3 below the floor; A2 out of scope; A3 orthogonal; "late" row after both.
    assert result["applicable"] == 4
    default = result["settings"]["default"]
    one_close = 1 - 0.5 * 0.5 * 0.25
    assert default["all"]["cases"] == 4
    assert default["delivered"]["cases"] == 1
    assert default["all"]["share_gt_0_9"] == pytest.approx(0.5)  # both E1 cases ~0.94
    steep = result["settings"]["steep"]
    assert steep["all"]["share_gt_0_9"] == 0.0
    assert steep["all"]["share_le_0_75"] == 1.0
    assert one_close > 0.9
    # Read-only: the census connection cannot write.
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM query_irrelevance")
    conn.close()


def test_census_without_table_reports_nothing(tmp_path: Path) -> None:
    census = _load_census()
    db = tmp_path / "empty.sqlite3"
    sqlite3.connect(db).close()
    conn = census._connect_ro(db)
    result = census.census(conn, [census.parse_setting("default:")], lambda t: [])
    assert result["applicable"] == 0


def test_parse_setting_rejects_bad_values() -> None:
    census = _load_census()
    assert census.parse_setting("default:") == census.Setting("default")
    with pytest.raises(ValueError):
        census.parse_setting("x:full=0.5")
    with pytest.raises(ValueError):
        census.parse_setting("x:mw=0")
    with pytest.raises(ValueError):
        census.parse_setting("x:bogus=1")
