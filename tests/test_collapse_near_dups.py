"""Tests for ``scripts/lm_collapse_near_dups.py``.

Every fixture here is a hand-built sqlite database, never a ``MemoryStore``:
the script's contract is the schema, so the tests have to speak the schema too.
Nothing in this file makes the script importable through ``living_memory``, and
one test runs it in an isolated subprocess to prove it never needs to be.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lm_collapse_near_dups.py"


def _load_script_module():
    """Import the script by path, the way a foreign installation would run it."""

    spec = importlib.util.spec_from_file_location("lm_collapse_near_dups", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collapse = _load_script_module()


# --------------------------------------------------------------------------
# fixture construction
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    content TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    decayed INTEGER NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    usefulness_score REAL NOT NULL DEFAULT 0.0,
    source_traces TEXT NOT NULL DEFAULT '[]',
    provenance TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z'
);
CREATE TABLE node_chunk_embeddings (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
    UNIQUE(node_id, chunk_index)
);
CREATE TABLE connections (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
    updated_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
    UNIQUE(source_id, target_id, type)
);
"""


def _pack(values):
    return array("f", values).tobytes()


def _unit(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


class Fixture(object):
    """A minimal Living Memory database built straight from the schema."""

    def __init__(self, path, schema=SCHEMA):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(schema)

    def add_node(
        self,
        node_id,
        vectors=None,
        content="fact",
        level="trace",
        scope="global",
        access_count=0,
        usefulness=0.0,
        decayed=0,
        source_traces=(),
        created_at="2026-01-01T00:00:00Z",
        dimensions=None,
    ):
        self.connection.execute(
            "INSERT INTO nodes (id, level, content, scope, decayed, access_count,"
            " usefulness_score, source_traces, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node_id,
                level,
                content,
                scope,
                decayed,
                access_count,
                usefulness,
                json.dumps(list(source_traces)),
                created_at,
            ),
        )
        for index, vector in enumerate(vectors or []):
            self.connection.execute(
                "INSERT INTO node_chunk_embeddings"
                " (id, node_id, chunk_index, dimensions, embedding)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    "%s-c%d" % (node_id, index),
                    node_id,
                    index,
                    len(vector) if dimensions is None else dimensions,
                    _pack(vector),
                ),
            )
        self.connection.commit()
        return node_id

    def add_edge(self, source, target, relation="supersedes"):
        self.connection.execute(
            "INSERT INTO connections (id, source_id, target_id, type)"
            " VALUES (?, ?, ?, ?)",
            ("%s-%s-%s" % (source, target, relation), source, target, relation),
        )
        self.connection.commit()

    def close(self):
        self.connection.close()


@pytest.fixture
def fixture(tmp_path):
    made = Fixture(tmp_path / "fixture.sqlite3")
    yield made
    made.close()


def run_script(*argv):
    """Run the script in-process and capture (exit code, stdout, stderr)."""

    import io
    import contextlib

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = collapse.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def collapses_in(stdout):
    """[(superseded, bearer)] parsed out of the report's COLLAPSE lines."""

    pairs = []
    for line in stdout.splitlines():
        if line.startswith("COLLAPSE "):
            parts = line.split()
            pairs.append((parts[parts.index("supersede") + 1], parts[parts.index("keep") + 1]))
    return pairs


def skip_reasons(stdout):
    """{reason: count} parsed out of the report's skip breakdown."""

    reasons = {}
    inside = False
    for line in stdout.splitlines():
        if line.startswith("## skip breakdown"):
            inside = True
            continue
        if inside:
            if line.startswith("## "):
                break
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                reasons[parts[0]] = int(parts[1])
    return reasons


# --------------------------------------------------------------------------
# portability contract
# --------------------------------------------------------------------------


def test_script_never_imports_living_memory():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    code_lines = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) or " import " in line
    ]
    offenders = [line for line in code_lines if "living_memory" in line]
    assert offenders == [], offenders


def test_runs_under_an_isolated_interpreter(fixture, tmp_path):
    """No PYTHONPATH, no user site, no repo on sys.path: it still runs.

    This is the portability claim made executable -- the script has to work
    under the host interpreter of a server whose checkout is a different
    version of living_memory, or none at all.
    """

    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="alpha", access_count=5)
    fixture.add_node("B", [_unit([0.999, 0.03, 0.0, 0.0])], content="alpha")

    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT_PATH), fixture.path],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert "## max-cosine distribution" in completed.stdout
    assert collapses_in(completed.stdout) == [("B", "A")]


def test_pure_python_matches_numpy(fixture):
    """The numpy-free path is a fallback, not a different answer."""

    pytest.importorskip("numpy")
    vectors = [
        _unit([1.0, 0.02, 0.0, 0.0]),
        _unit([1.0, 0.05, 0.01, 0.0]),
        _unit([0.0, 1.0, 0.0, 0.0]),
        _unit([0.01, 1.0, 0.02, 0.0]),
        _unit([0.0, 0.0, 1.0, 0.0]),
    ]
    for index, vector in enumerate(vectors):
        fixture.add_node("N%d" % index, [vector], content="fact %d" % index, access_count=index)

    _, with_numpy, _ = run_script(fixture.path)
    _, without_numpy, _ = run_script(fixture.path, "--no-numpy")

    assert collapses_in(with_numpy) == collapses_in(without_numpy)
    assert skip_reasons(with_numpy) == skip_reasons(without_numpy)

    def distribution(stdout):
        return [line for line in stdout.splitlines() if line.strip().startswith(("p5", "p1", "p9", ">= "))]

    assert distribution(with_numpy) == distribution(without_numpy)


def test_pure_python_path_runs_with_numpy_unimportable(fixture, monkeypatch):
    """With numpy blocked at import time the script still produces a report."""

    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="alpha", access_count=5)
    fixture.add_node("B", [_unit([0.999, 0.03, 0.0, 0.0])], content="alpha")

    monkeypatch.setitem(sys.modules, "numpy", None)  # `import numpy` -> ImportError
    assert collapse.load_numpy() is None

    code, stdout, _ = run_script(fixture.path)
    assert code == 0
    assert "numpy is absent" in stdout
    assert collapses_in(stdout) == [("B", "A")]


# --------------------------------------------------------------------------
# preflight refusals
# --------------------------------------------------------------------------


def test_refuses_database_without_chunk_table(tmp_path):
    path = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(str(path))
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT NOT NULL,"
        " decayed INTEGER NOT NULL DEFAULT 0, source_traces TEXT DEFAULT '[]');"
    )
    connection.close()

    code, _, stderr = run_script(str(path))
    assert code == collapse.EXIT_REFUSED
    assert "node_chunk_embeddings" in stderr
    assert "predates chunk embeddings" in stderr


def test_refuses_mixed_vector_widths(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])])
    fixture.add_node("B", [_unit([1.0, 0.0, 0.0])])

    code, _, stderr = run_script(fixture.path)
    assert code == collapse.EXIT_REFUSED
    assert "mixed vector widths" in stderr
    assert "different width are points in different spaces" in stderr
    assert "4 dims" in stderr and "3 dims" in stderr


def test_refuses_when_chunk_coverage_is_too_thin(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])])
    for index in range(4):
        fixture.add_node("N%d" % index, [])

    code, _, stderr = run_script(fixture.path)
    assert code == collapse.EXIT_REFUSED
    assert "1 of 5 active nodes (20.0%)" in stderr
    assert "--min-coverage" in stderr

    code, stdout, _ = run_script(fixture.path, "--min-coverage", "0.1")
    assert code == collapse.EXIT_OK
    assert "chunk coverage          : 1 of 5 active nodes (20.0%)" in stdout


def test_refuses_missing_provenance_columns(tmp_path):
    path = tmp_path / "no-provenance.sqlite3"
    connection = sqlite3.connect(str(path))
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT NOT NULL,"
        " decayed INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE node_chunk_embeddings (id TEXT PRIMARY KEY, node_id TEXT,"
        " chunk_index INTEGER, dimensions INTEGER, embedding BLOB);"
    )
    connection.close()

    code, _, stderr = run_script(str(path))
    assert code == collapse.EXIT_REFUSED
    assert "provenance guard" in stderr


def test_refuses_mid_band_threshold_without_opt_in(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])])
    fixture.add_node("B", [_unit([1.0, 0.4, 0.0, 0.0])])

    code, _, stderr = run_script(fixture.path, "--threshold", "0.90")
    assert code == collapse.EXIT_REFUSED
    assert "0.85-0.95" in stderr
    assert "--allow-mid-band" in stderr

    code, stdout, _ = run_script(fixture.path, "--threshold", "0.90", "--allow-mid-band")
    assert code == collapse.EXIT_OK
    assert "MID-BAND OVERRIDE ACTIVE" in stdout


# --------------------------------------------------------------------------
# report order and content
# --------------------------------------------------------------------------


def test_distribution_is_printed_before_the_threshold_is_applied(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], access_count=3)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])])
    fixture.add_node("C", [_unit([0.0, 1.0, 0.0, 0.0])])

    _, stdout, _ = run_script(fixture.path)
    distribution_at = stdout.index("## max-cosine distribution for THIS database")
    threshold_at = stdout.index("## threshold")
    collapse_at = stdout.index("## would collapse")
    assert distribution_at < threshold_at < collapse_at
    assert "p50" in stdout and ">= 0.950" in stdout
    assert "  max    1.0000" not in stdout  # the max is a real measurement


def test_dry_run_is_the_default_and_writes_nothing(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], access_count=5)
    fixture.add_node("B", [_unit([0.999, 0.03, 0.0, 0.0])])

    code, stdout, _ = run_script(fixture.path)
    assert code == collapse.EXIT_OK
    assert "## dry run" in stdout
    assert "opened during a dry run" not in stdout
    assert "read-only (dry run)" in stdout
    assert collapses_in(stdout) == [("B", "A")]

    remaining = fixture.connection.execute(
        "SELECT COUNT(*) FROM connections"
    ).fetchone()[0]
    assert remaining == 0


def test_dry_run_prints_the_exact_apply_command(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], access_count=5)
    fixture.add_node("B", [_unit([0.999, 0.03, 0.0, 0.0])])

    _, stdout, _ = run_script(fixture.path)
    command = [line.strip() for line in stdout.splitlines() if "--apply" in line]
    assert command
    assert os.path.abspath(fixture.path) in command[0]
    assert "--threshold 0.95" in command[0]


# --------------------------------------------------------------------------
# skip rules
# --------------------------------------------------------------------------


def test_skips_a_source_trace_of_a_live_concept(fixture):
    """The provenance band is never duplication, whatever the cosine says."""

    fixture.add_node("TRACE-OLD", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("TRACE-DUP", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node(
        "CONCEPT",
        [_unit([0.0, 0.0, 1.0, 0.0])],
        content="digest",
        level="concept",
        source_traces=["TRACE-DUP"],
    )

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"provenance_source_trace": 1}
    assert "that band is provenance, never duplication" in stdout


def test_a_decayed_concept_no_longer_protects_its_sources(fixture):
    fixture.add_node("TRACE-OLD", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("TRACE-DUP", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node(
        "CONCEPT",
        [_unit([0.0, 0.0, 1.0, 0.0])],
        content="digest",
        level="concept",
        decayed=1,
        source_traces=["TRACE-DUP"],
    )

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == [("TRACE-DUP", "TRACE-OLD")]


def test_provenance_can_come_from_the_provenance_json_column(fixture):
    """Installations differ in where source_traces is spelled; both are read."""

    fixture.add_node("TRACE-OLD", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("TRACE-DUP", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node("CONCEPT", [_unit([0.0, 0.0, 1.0, 0.0])], content="digest", level="concept")
    fixture.connection.execute(
        "UPDATE nodes SET provenance = ? WHERE id = 'CONCEPT'",
        (json.dumps({"source_traces": ["TRACE-DUP"]}),),
    )
    fixture.connection.commit()

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"provenance_source_trace": 1}


def test_skips_a_candidate_materially_longer_than_its_bearer(fixture):
    fixture.add_node(
        "SHORT", [_unit([1.0, 0.0, 0.0, 0.0])], content="the fact", access_count=9
    )
    fixture.add_node(
        "LONG",
        [_unit([0.999, 0.02, 0.0, 0.0])],
        content="the fact, plus the detail that only this copy carries",
    )

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"longer_than_bearer": 1}
    assert "the detail has to reach the agent" in stdout


def test_a_slightly_longer_candidate_inside_the_margin_still_collapses(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="x" * 100, access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="x" * 105)

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == [("B", "A")]


def test_skips_across_levels(fixture):
    fixture.add_node("TRACE", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=1)
    fixture.add_node("CONCEPT", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact", level="concept")

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"level_mismatch": 1}


def test_skips_a_candidate_that_already_corrects_something(fixture):
    fixture.add_node("KEEP", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("FIX", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node("OLD", [_unit([0.0, 0.0, 1.0, 0.0])], content="stale")
    fixture.add_edge("FIX", "OLD")

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"candidate_is_correction": 1}


def test_skips_when_the_would_be_bearer_is_itself_superseded(fixture):
    """Nothing is hung off a node the system already demotes."""

    fixture.add_node("STALE", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("DUP", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node("FIX", [_unit([0.0, 0.0, 1.0, 0.0])], content="corrected fact")
    fixture.add_edge("FIX", "STALE")

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"bearer_already_superseded": 1}


def test_a_node_without_chunk_vectors_is_never_collapsed(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node("NOVEC", [], content="fact")

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == [("B", "A")]
    assert "nodes without chunks    : 1" in stdout


def test_scope_is_respected_by_default_and_crossable_on_request(fixture):
    fixture.add_node(
        "A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", scope="project:one", access_count=9
    )
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact", scope="project:two")

    _, same_scope, _ = run_script(fixture.path)
    assert collapses_in(same_scope) == []
    assert "nodes alone in their scope: 2" in same_scope

    _, any_scope, _ = run_script(fixture.path, "--scope-mode", "any")
    assert collapses_in(any_scope) == [("B", "A")]


def test_decayed_nodes_take_no_part(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact", decayed=1)
    fixture.add_node("C", [_unit([0.0, 1.0, 0.0, 0.0])], content="other")

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert "nodes compared          : 2" in stdout


# --------------------------------------------------------------------------
# mean pooling
# --------------------------------------------------------------------------


def test_chunks_are_mean_pooled_before_comparison(fixture):
    """Two chunks average to the midpoint; a single-chunk node at that midpoint
    is therefore an exact match, which max-pooling would not produce."""

    fixture.add_node(
        "TWOCHUNK",
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        content="fact",
        access_count=9,
    )
    fixture.add_node("MEAN", [_unit([1.0, 1.0, 0.0, 0.0])], content="fact")
    fixture.add_node("MAXPOOL", [_unit([1.0, 1.0, 0.0, 0.0])], content="fact", decayed=1)

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == [("MEAN", "TWOCHUNK")]
    assert "  max    1.0000" in stdout


def test_a_node_whose_chunks_disagree_on_width_is_excluded(fixture):
    """Mixed width inside one node is refused at preflight; when the widths are
    only inconsistent against the recorded dimensions the node drops out."""

    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node("BROKEN", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact")
    fixture.connection.execute(
        "UPDATE node_chunk_embeddings SET embedding = ? WHERE node_id = 'BROKEN'",
        (_pack([1.0, 0.0, 0.0]),),
    )
    fixture.connection.commit()

    code, _, stderr = run_script(fixture.path)
    assert code == collapse.EXIT_REFUSED
    assert "does not match their recorded dimensions" in stderr


# --------------------------------------------------------------------------
# chaining
# --------------------------------------------------------------------------


def test_a_cluster_of_three_chains_onto_one_bearer(fixture):
    """Everything in a cluster points at the node that survives, not at a
    node that is itself superseded."""

    fixture.add_node("TOP", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=50)
    fixture.add_node("MID", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact", access_count=10)
    fixture.add_node("LOW", [_unit([0.998, 0.03, 0.0, 0.0])], content="fact", access_count=1)

    _, stdout, _ = run_script(fixture.path)
    pairs = collapses_in(stdout)
    assert sorted(pairs) == [("LOW", "TOP"), ("MID", "TOP")]
    assert "rechained from MID" in stdout


def test_the_length_guard_is_re_checked_against_the_surviving_node(fixture):
    """A chain must not smuggle a long candidate onto a short survivor.

    LOW -> MID passes the guard and MID -> TOP passes it, but LOW is 21%
    longer than TOP, so once the chain is flattened LOW must be refused
    instead of being pointed at TOP.
    """

    fixture.add_node(
        "TOP", [_unit([1.0, 0.0, 0.0, 0.0])], content="x" * 100, access_count=50
    )
    fixture.add_node(
        "MID", [_unit([0.999, 0.02, 0.0, 0.0])], content="x" * 110, access_count=10
    )
    fixture.add_node(
        "LOW", [_unit([0.998, 0.03, 0.0, 0.0])], content="x" * 121, access_count=1
    )

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == [("MID", "TOP")]
    assert "at the surviving node (chained through MID)" in stdout
    # The TOP/LOW pair was already noted as skipped when LOW still looked
    # collapsed; dropping the chain leaves that earlier note standing, which
    # under-collapses rather than over-collapses.
    assert skip_reasons(stdout) == {
        "longer_than_bearer": 1,
        "candidate_already_collapsed": 1,
    }


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


def test_apply_writes_supersedes_edges_and_nothing_else(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    before = fixture.connection.execute(
        "SELECT COUNT(*), SUM(decayed) FROM nodes"
    ).fetchone()

    code, stdout, _ = run_script(fixture.path, "--apply")
    assert code == collapse.EXIT_OK
    assert "supersedes edges written : 1" in stdout

    rows = fixture.connection.execute(
        "SELECT source_id, target_id, type, metadata FROM connections"
    ).fetchall()
    assert len(rows) == 1
    source, target, relation, metadata = rows[0]
    assert (source, target, relation) == ("A", "B", "supersedes")
    assert json.loads(metadata)["by"] == "lm_collapse_near_dups.py"
    assert json.loads(metadata)["cosine"] >= 0.95

    after = fixture.connection.execute(
        "SELECT COUNT(*), SUM(decayed) FROM nodes"
    ).fetchone()
    assert after == before  # no row deleted, nothing marked decayed


def test_apply_prints_the_stopped_server_note(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")

    _, stdout, _ = run_script(fixture.path, "--apply")
    assert "STOPPED" in stdout
    assert "restart that server" in stdout
    assert "chunk matrix cached" in stdout


def test_apply_is_idempotent(fixture):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")

    run_script(fixture.path, "--apply")
    _, stdout, _ = run_script(fixture.path, "--apply")
    assert "supersedes edges written : 0" in stdout
    assert skip_reasons(stdout) == {"candidate_already_superseded": 1}
    assert (
        fixture.connection.execute("SELECT COUNT(*) FROM connections").fetchone()[0] == 1
    )


def test_json_summary_records_the_distribution_and_the_decisions(fixture, tmp_path):
    fixture.add_node("A", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("B", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")
    fixture.add_node("C", [_unit([0.0, 1.0, 0.0, 0.0])], content="other")

    target = tmp_path / "summary.json"
    run_script(fixture.path, "--json", str(target))
    payload = json.loads(target.read_text())

    assert payload["threshold"] == collapse.SAFE_THRESHOLD
    assert payload["applied"] is False
    assert payload["nodes_compared"] == 3
    assert payload["percentiles"]["p50"] > 0.0
    assert payload["counts_at_threshold"]["0.950"]["pairs"] == 1
    assert payload["collapses"] == [
        {
            "bearer_chars": 4,
            "candidate_chars": 4,
            "cosine": pytest.approx(payload["collapses"][0]["cosine"]),
            "keep": "A",
            "level": "trace",
            "scope": "global",
            "supersede": "B",
        }
    ]
