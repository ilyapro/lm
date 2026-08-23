"""Tests for ``scripts/lm_collapse_near_dups.py``.

Every fixture here is a hand-built sqlite database, never a ``MemoryStore``:
the script's contract is the schema, so the tests have to speak the schema too.
Nothing in this file makes the script importable through ``living_memory``, and
one test runs it in an isolated subprocess to prove it never needs to be.

This file does import ``living_memory.near_dup``, and that asymmetry is the
point. The script duplicates the identifier veto instead of importing it,
because importing it would cost the portability the script exists for; the
duplication is then held to account from here, where both implementations are
in scope at once (see "the mirror" at the bottom). A test may import the
package. The script may not.
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

from living_memory import near_dup as reference

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


def skip_details(stdout):
    """{superseded id: [(reason, detail), ...]} from the per-reason sections.

    A list, not one entry: one node can be skipped by several pairs, and a
    dict keyed by node would silently keep whichever reason printed last.
    """

    details = {}
    reason = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            reason = stripped[4:].rsplit(" (", 1)[0]
            continue
        if reason and stripped.startswith("SKIP "):
            parts = stripped.split()
            candidate = parts[parts.index("would-supersede") + 1]
            detail = ""
            if "(" in stripped:
                detail = stripped.split("(", 1)[1].rsplit(")", 1)[0]
            details.setdefault(candidate, []).append((reason, detail))
    return details


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
    assert payload["identifier_veto"] is True
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


# --------------------------------------------------------------------------
# the identifier veto
# --------------------------------------------------------------------------


def test_skips_a_candidate_naming_an_identifier_the_bearer_lacks(fixture):
    """The motivating pair, at the cosine that actually collapsed it.

    «узел layer-fauna СДЕЛАН» and «узел layer-actors СДЕЛАН» scored 0.9547 on
    the alt corpus -- above this script's default threshold -- and they are two
    different facts. The template dominates the vector, so no threshold
    separates the pair from an honest repeat and the refusal has to be a veto.
    """

    fixture.add_node(
        "KEEP",
        [_unit([1.0, 0.0, 0.0, 0.0])],
        content="узел layer-actors СДЕЛАН и записан",
        access_count=9,
    )
    fixture.add_node(
        "DROP",
        [_unit([0.999, 0.02, 0.0, 0.0])],
        content="узел layer-fauna СДЕЛАН и записан",
    )

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"identifier_veto": 1}
    assert skip_details(stdout)["DROP"] == [
        ("identifier_veto", "candidate names layer-fauna; the bearer's text does not")
    ]
    assert "two different facts" in stdout


def test_the_bearers_own_extra_identifier_does_not_block_the_collapse(fixture):
    """One-directional, like the length guard, and for the same reason.

    The bearer keeps its whole text, so an identifier only IT names still
    reaches the agent. It is the candidate's that would vanish behind the edge.
    """

    fixture.add_node(
        "KEEP",
        [_unit([1.0, 0.0, 0.0, 0.0])],
        content="the survey report near-dup-identifier-veto was filed",
        access_count=9,
    )
    fixture.add_node(
        "DROP",
        [_unit([0.999, 0.02, 0.0, 0.0])],
        content="the survey report was filed and noted",
    )

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == [("DROP", "KEEP")]


def test_prose_only_near_duplicates_still_collapse(fixture):
    """The veto's negative control: without it the guard could be "never".

    A rule that vetoes everything passes every veto test and destroys the
    script, so one pair of identifier-free paraphrases has to keep collapsing.
    """

    fixture.add_node(
        "KEEP",
        [_unit([1.0, 0.0, 0.0, 0.0])],
        content="the ballast survey runs at slack water",
        access_count=9,
    )
    fixture.add_node(
        "DROP",
        [_unit([0.999, 0.02, 0.0, 0.0])],
        content="at slack water the ballast survey runs",
    )

    _, stdout, _ = run_script(fixture.path)
    assert collapses_in(stdout) == [("DROP", "KEEP")]
    assert skip_reasons(stdout) == {}


def test_the_env_valve_off_restores_the_previous_behaviour(fixture, monkeypatch):
    """``LM_NEAR_DUP_IDENTIFIER_VETO=0`` is the rollback, not a second mode."""

    fixture.add_node(
        "KEEP",
        [_unit([1.0, 0.0, 0.0, 0.0])],
        content="узел layer-actors СДЕЛАН и записан",
        access_count=9,
    )
    fixture.add_node(
        "DROP",
        [_unit([0.999, 0.02, 0.0, 0.0])],
        content="узел layer-fauna СДЕЛАН и записан",
    )

    monkeypatch.setenv(collapse.IDENTIFIER_VETO_ENV, "0")
    _, stdout, _ = run_script(fixture.path)

    assert collapses_in(stdout) == [("DROP", "KEEP")]
    assert "identifier veto         : OFF" in stdout
    # The apply command carries the valve, because a pass run without the veto
    # is not the pass the default command would apply.
    apply_line = [line for line in stdout.splitlines() if "--apply" in line][0]
    assert apply_line.strip().startswith("LM_NEAR_DUP_IDENTIFIER_VETO=0 python3")


def test_the_chain_leak_is_closed_at_its_first_link(fixture):
    """The substring defect in miniature, and why token equality composes.

    ``MID``'s text spells a 28-character run, ``…ABCDEF``. That run is not an
    identifier of anything: two characters too long for a ULID, no digit, no
    case transition. ``LOW``'s 26-character ULID sits inside it.

    Under substring containment ``LOW -> MID`` PASSED -- the token "occurred
    in" the bearer's text -- and ``MID -> TOP`` passed too, MID naming no
    identifier of its own. The chain then carried LOW's ULID to a root whose
    text does not contain it, and only ``_flatten``'s re-check at the root
    caught it. That is the same non-transitivity the suffixed goal-tree node
    names showed at cosine 0.99350: "token inside the bearer's TEXT" does not
    compose, because the next link is decided between EXTRACTED tokens and this
    one was not.

    Token equality composes -- token sets, and set membership is transitive --
    so the leak is refused where it starts. LOW names a ULID that is not one of
    MID's tokens, and the first link never forms. ``_flatten`` still re-checks
    the veto at the root, now as insurance rather than as the thing that
    catches this: the length guard beside it is genuinely not transitive.
    """

    ulid = "ABCDEFGHJKMNPQRSTVWXYZABCD"  # 26 Crockford characters, no digit
    fixture.add_node(
        "TOP",
        [_unit([1.0, 0.0, 0.0, 0.0])],
        content="the identifier was set down and then noted in the record",
        access_count=50,
    )
    fixture.add_node(
        "MID",
        [_unit([0.999, 0.02, 0.0, 0.0])],
        content="the identifier %sEF was noted in the record" % ulid,
        access_count=10,
    )
    fixture.add_node(
        "LOW",
        [_unit([0.998, 0.03, 0.0, 0.0])],
        content="the identifier %s was noted in the record" % ulid,
        access_count=1,
    )
    # The premise, stated rather than assumed: MID's longer run is not a name,
    # so MID contributes no token at all -- and the ULID is not among its
    # tokens even though it is inside its text.
    assert collapse.extract_identifiers("%sEF" % ulid) == ()
    assert ulid in "the identifier %sEF was noted in the record" % ulid

    # The length margin is out of the way on purpose: this test is about which
    # tokens the identifiers are compared against, not about how long they are.
    _, stdout, _ = run_script(fixture.path, "--length-margin", "1.0")

    assert collapses_in(stdout) == [("MID", "TOP")]
    # LOW is refused against both of them, directly, rather than collapsing
    # into MID and being unwound at the root afterwards.
    assert skip_reasons(stdout) == {"identifier_veto": 2}
    assert set(dict(skip_details(stdout)["LOW"]).items()) == {
        ("identifier_veto", "candidate names %s; the bearer's text does not" % ulid)
    }


def test_a_pair_whose_text_cannot_be_read_is_not_collapsed(fixture, monkeypatch):
    """An unenforceable veto refuses; it does not quietly pass everything."""

    fixture.add_node("KEEP", [_unit([1.0, 0.0, 0.0, 0.0])], content="fact", access_count=9)
    fixture.add_node("DROP", [_unit([0.999, 0.02, 0.0, 0.0])], content="fact")

    monkeypatch.setattr(collapse, "load_contents", lambda connection, ids: {})
    _, stdout, _ = run_script(fixture.path)

    assert collapses_in(stdout) == []
    assert skip_reasons(stdout) == {"identifier_veto": 1}
    assert "could not be read" in stdout


# --------------------------------------------------------------------------
# the mirror: this script's veto against living_memory.near_dup's
# --------------------------------------------------------------------------
#
# The script may not import the package, so the two implementations of the
# identifier veto are two copies of the same rule. That duplication is only
# safe while it is proven, and this is where it is proven: one table of pairs,
# both implementations asked, a hand-written verdict for each. A drift in
# either direction fails here -- the script gaining a class the map lacks is
# as much a bug as the reverse, because the drain, the delivery layer and this
# script are supposed to refuse the same corpus.

#: (label, bearer text, candidate text, collapses?). Every identifier class the
#: veto is required to cover, each with a negative, plus the pairs that must
#: still collapse: prose paraphrases, shared identifiers, and identifiers only
#: the bearer names. The suffix/prefix and numeric-prefix classes are the ones
#: substring containment could not see -- one token sitting inside a longer one
#: is not the same token, and both copies have to agree about that too.
MIRROR_PAIRS = (
    (
        "ulid",
        "the recall returned the same node twice",
        "the recall returned 01M0QJV1BBRXNHF7D5NS23FD35 twice",
        False,
    ),
    (
        "crockford ulid carrying no digit at all",
        "the id was written into the trace",
        "the id ABCDEFGHJKMNPQRSTVWXYZABCD was written into the trace",
        False,
    ),
    (
        "repo path",
        "the veto lives in the shared leaf module",
        "the veto lives in src/living_memory/near_dup.py",
        False,
    ),
    (
        "home path",
        "the backup was taken before the hygiene pass",
        "the backup ~/.local/share/living-memory/global.sqlite3 was taken",
        False,
    ),
    (
        "absolute path",
        "the checkout was made this morning",
        "the checkout /home/sfx/p/lm was made this morning",
        False,
    ),
    (
        "goal-node name",
        "the branch was merged this morning",
        "the branch near-dup-identifier-veto was merged this morning",
        False,
    ),
    (
        "tree node slug -- the motivating pair",
        "узел layer-actors СДЕЛАН и записан",
        "узел layer-fauna СДЕЛАН и записан",
        False,
    ),
    (
        "underscore symbol",
        "the map is built by one function",
        "the map is built by build_duplicate_map",
        False,
    ),
    (
        "dotted module path",
        "the leaf module is imported by both",
        "living_memory.near_dup is imported by both",
        False,
    ),
    (
        "hex digest",
        "the frozen selection was measured again",
        "the frozen selection digest 19877a4383b2e2d2 was measured",
        False,
    ),
    (
        "short commit id",
        "the fix landed at that commit exactly",
        "the fix landed at commit ce93bae exactly",
        False,
    ),
    (
        "ticket id",
        "the ticket was closed on friday",
        "ticket LM-123 was closed on friday",
        False,
    ),
    (
        "scoped name",
        "written into the project scope here",
        "written with scope project:lm here",
        False,
    ),
    (
        "year",
        "the note was written by an agent",
        "the note was written in 2026 by an agent",
        False,
    ),
    (
        "measured cosine",
        "the pair scores high on the alt corpus",
        "the pair scores cos 0.9547 on the alt corpus",
        False,
    ),
    (
        "version",
        "the host received the shipped build",
        "the host received v1.2 of the build",
        False,
    ),
    (
        "camel case symbol",
        "the store satisfies the protocol",
        "MemoryStore satisfies the protocol",
        False,
    ),
    (
        "url",
        "the page was fetched from there",
        "https://example.test/x was fetched from there",
        False,
    ),
    (
        "env var, where case is meaning",
        "set lm_near_dup_identifier_veto to zero",
        "set LM_NEAR_DUP_IDENTIFIER_VETO to zero",
        False,
    ),
    (
        "a fuller path against a bare filename",
        "edit near_dup.py today",
        "edit src/living_memory/near_dup.py today",
        False,
    ),
    (
        # Was the one collapse in this table that substring containment bought:
        # the bearer's fuller path "contained" the candidate's bare filename.
        # Token equality refuses it, and the three classes below are why -- no
        # rule can pass this and still block a node name inside its own
        # suffixed twin.
        "a bare filename against a fuller path",
        "edit src/living_memory/near_dup.py today",
        "edit near_dup.py today",
        False,
    ),
    (
        # The pair the drain simulation caught escaping at cosine 0.99350:
        # goal-tree names are built by suffixing, so the shorter node name is
        # literally a substring of the longer one and containment saw nothing
        # missing. Two distinct nodes, two distinct recorded failures, one fact.
        "a node name the bearer's suffixed name contains",
        "OUTCOME fail: x/checkpoint-selected-profile-v2-repaired - measured",
        "OUTCOME fail: x/checkpoint-selected-profile-v2 - measured",
        False,
    ),
    (
        "the same suffixed pair the other way round, which already vetoed",
        "OUTCOME fail: x/checkpoint-selected-profile-v2 - measured",
        "OUTCOME fail: x/checkpoint-selected-profile-v2-repaired - measured",
        False,
    ),
    (
        # The second escaper, at 0.99064. No mirror direction existed in any
        # arm of the simulation, so the veto never saw this pair from the side
        # that would have blocked it -- it was simply inert.
        "a node name the bearer extends with -reintegrate",
        "OUTCOME pass: p/stock-contract-preservation-audit-reintegrate - r",
        "OUTCOME pass: p/stock-contract-preservation-audit - r",
        False,
    ),
    (
        # Numbers suffix too: 0.99 sits inside 0.99350, so containment read the
        # coarser measurement as already carried by the finer one. Same bug,
        # one level down in the grammar.
        "a measurement that is a numeric prefix of the bearer's",
        "the band was measured at 0.99350 exactly",
        "the band was measured at 0.99 exactly",
        False,
    ),
    (
        "only the bearer carries the identifier",
        "the survey report near-dup-identifier-veto was filed",
        "the survey report was filed and noted",
        True,
    ),
    (
        "both name the same identifiers",
        "layer-fauna was checked on 2026-08-23",
        "on 2026-08-23 layer-fauna was checked",
        True,
    ),
    (
        "english prose only",
        "the ballast survey runs at slack water",
        "at slack water the ballast survey runs",
        True,
    ),
    (
        "russian prose only",
        "узел проверен и записан в память",
        "узел записан и проверен в память",
        True,
    ),
    (
        "emphasis in caps is not a name",
        "this is IMPORTANT for the recall path",
        "this is СДЕЛАН for the recall path",
        True,
    ),
    (
        "abbreviations are not names",
        "the fact holds here e.g. in this case",
        "the fact holds here i.e. in this case",
        True,
    ),
    (
        "small numbers are arithmetic",
        "alpha drift measured at 3 units in all",
        "alpha drift measured at 12 units in all",
        True,
    ),
    (
        "a short decimal is not a version",
        "a ratio of 1.5 between the two runs",
        "between the two runs a ratio of 1.5",
        True,
    ),
    (
        "long bare words are not names",
        "understanding notwithstanding the drift",
        "notwithstanding the drift understanding",
        True,
    ),
)

MIRROR_TEXTS = tuple(
    dict.fromkeys(
        text for _, bearer, candidate, _ in MIRROR_PAIRS for text in (bearer, candidate)
    )
)

#: Length is taken out of the mirror on purpose: both implementations are asked
#: with a margin no pair in the table can reach, so the only guard that can
#: decide a verdict is the identifier veto. A divergence therefore cannot hide
#: behind "one of them refused for length".
MIRROR_LENGTH_MARGIN = 4.0


@pytest.mark.parametrize("text", MIRROR_TEXTS)
def test_mirror_extracts_the_same_identifiers(text):
    assert collapse.extract_identifiers(text) == reference.extract_identifiers(text)


@pytest.mark.parametrize(
    ("label", "bearer", "candidate"),
    [(label, bearer, candidate) for label, bearer, candidate, _ in MIRROR_PAIRS],
)
def test_mirror_agrees_on_what_a_collapse_would_hide(label, bearer, candidate):
    """Both directions: the veto's asymmetry has to be the same asymmetry."""

    assert collapse.identifiers_absent_from(
        candidate, bearer
    ) == reference.identifiers_absent_from(candidate, bearer), label
    assert collapse.identifiers_absent_from(
        bearer, candidate
    ) == reference.identifiers_absent_from(bearer, candidate), label


#: Tokens the two grammars have to agree about, including the ones that sit on
#: a boundary: trailing punctuation, a lone separator, mixed scripts, a token
#: one character short of a ULID.
FUZZ_VOCABULARY = (
    "the fact узел СДЕЛАН layer-fauna layer-actors near_dup.py project:lm "
    "src/living_memory/near_dup.py ~/.local/share/x /home/sfx/p/lm 19877a4383b2e2d2 "
    "01M0QJV1BBRXNHF7D5NS23FD35 ABCDEFGHJKMNPQRSTVWXYZABCD ce93bae LM-123 2026-08-23 "
    "v1.2 0.9547 MemoryStore https://example.test/x e.g. i.e. т.е. 3 12 1.5 2026 "
    "understanding IMPORTANT что-то well-known build_duplicate_map living_memory.near_dup "
    "tail. -lead :colon mixedCaseWord ALLCAPS 0x1f ver.2.0 a-b _x_ / ~ .. :: -- 0.0.0.0 "
    "192.168.1.1 f00d deadbeef Ünicode-Ślug (quoted) «угловые» 🙂"
).split()


def test_mirror_agrees_on_generated_text():
    """The table names the classes; this covers the space between them.

    A hand-written table only proves agreement on the cases someone thought of,
    and the two grammars are two copies of a dozen interacting rules. Seeded,
    so a failure here is reproducible rather than a Heisenbug.
    """

    import random

    rng = random.Random(20260823)
    for _ in range(2000):
        text = " ".join(rng.choice(FUZZ_VOCABULARY) for _ in range(rng.randint(1, 14)))
        other = " ".join(rng.choice(FUZZ_VOCABULARY) for _ in range(rng.randint(1, 14)))
        assert collapse.extract_identifiers(text) == reference.extract_identifiers(
            text
        ), text
        assert collapse.identifiers_absent_from(
            text, other
        ) == reference.identifiers_absent_from(text, other), (text, other)


def test_mirror_reads_the_same_valve_name():
    assert collapse.IDENTIFIER_VETO_ENV == reference.IDENTIFIER_VETO_ENV


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("1", True),
        ("on", True),
        ("true", True),
        ("yes", True),
        ("maybe", True),  # unrecognized must not silently disable the guard
        ("0", False),
        ("off", False),
        ("false", False),
        ("no", False),
        ("OFF", False),
        ("  0  ", False),
    ],
)
def test_mirror_reads_the_valve_the_same_way(monkeypatch, value, expected):
    """One env line, one meaning, in the map and in this script alike."""

    if value is None:
        monkeypatch.delenv(collapse.IDENTIFIER_VETO_ENV, raising=False)
    else:
        monkeypatch.setenv(collapse.IDENTIFIER_VETO_ENV, value)

    assert collapse.identifier_veto_enabled() is expected
    assert reference.identifier_veto_enabled() is expected


def _map_collapses(bearer_text, candidate_text):
    """``build_duplicate_map``'s verdict for one pair, at cosine 1.0.

    Rank order is the map's bearer rule, so the bearer is offered first -- the
    same node the script's usage ranking picks in the fixture below.
    """

    vector = [1.0, 0.0, 0.0, 0.0]
    duplicates = reference.build_duplicate_map(
        [
            reference.DuplicateCandidate("KEEP", len(bearer_text), bearer_text),
            reference.DuplicateCandidate("DROP", len(candidate_text), candidate_text),
        ],
        {"KEEP": vector, "DROP": vector},
        cosine_threshold=collapse.SAFE_THRESHOLD,
        min_length_ratio=MIRROR_LENGTH_MARGIN,
        identifier_veto=True,
    )
    assert duplicates in ({}, {"DROP": "KEEP"})
    return bool(duplicates)


def test_mirror_verdicts_agree_pair_for_pair(tmp_path):
    """The whole table, once through the script and once through the map.

    One database, one pair per scope, one run: the script pairs nodes within a
    scope, so each pair is decided in isolation while still exercising the real
    report path -- preflight, scan, ``decide`` and the printed skip reason.
    """

    made = Fixture(tmp_path / "mirror.sqlite3")
    vector = _unit([1.0, 0.0, 0.0, 0.0])
    for index, (_, bearer, candidate, _) in enumerate(MIRROR_PAIRS):
        made.add_node(
            "P%02d-KEEP" % index,
            [vector],
            content=bearer,
            scope="pair-%02d" % index,
            access_count=9,
        )
        made.add_node(
            "P%02d-DROP" % index,
            [vector],
            content=candidate,
            scope="pair-%02d" % index,
        )

    try:
        code, stdout, stderr = run_script(
            made.path,
            "--length-margin",
            "%.2f" % MIRROR_LENGTH_MARGIN,
            "--report-limit",
            "0",
        )
        assert code == collapse.EXIT_OK, stderr
        collapsed = set(collapses_in(stdout))
        skipped = skip_details(stdout)
    finally:
        made.close()

    divergences = []
    for index, (label, bearer, candidate, expected) in enumerate(MIRROR_PAIRS):
        keep, drop = "P%02d-KEEP" % index, "P%02d-DROP" % index
        script_verdict = (drop, keep) in collapsed
        map_verdict = _map_collapses(bearer, candidate)
        if script_verdict != map_verdict:
            divergences.append(
                "%s: script %s, near_dup %s"
                % (
                    label,
                    "collapsed" if script_verdict else "vetoed",
                    "collapsed" if map_verdict else "vetoed",
                )
            )
            continue
        if script_verdict != expected:
            divergences.append(
                "%s: both %s, expected %s"
                % (
                    label,
                    "collapsed" if script_verdict else "vetoed",
                    "collapse" if expected else "veto",
                )
            )
            continue
        if not expected:
            # Agreement is only worth something if both refused for THIS
            # reason: two implementations refusing a pair for unrelated causes
            # would look identical from the outside.
            assert "identifier_veto" in [
                reason for reason, _ in skipped.get(drop, [])
            ], label

    assert divergences == []
    # Half the table has to survive the veto, or "refuse everything" would pass.
    assert sum(1 for _, _, _, expected in MIRROR_PAIRS if expected) >= 8
