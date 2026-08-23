#!/usr/bin/env python3
"""Collapse already-accumulated near-duplicate nodes through ``supersedes`` edges.

Portability is the contract of this file, and the contract is the DATABASE
SCHEMA, not the ``living_memory`` package version:

* It imports nothing from ``living_memory`` and nothing outside the standard
  library.  Several Living Memory installations exist, each with its own
  database and its own host interpreter (system python here, a venv there), and
  their code versions have drifted.  The script therefore has to run under the
  interpreter of whichever server owns the database it is pointed at, which
  rules out both the package and third-party dependencies.
* numpy is used when it imports and is not required.  Without it the cosine
  scan runs in pure Python: corpora here are 2.7k-13k nodes, so the pure-Python
  cost is minutes, which is the right trade against not running at all.
* Every schema expectation is probed and refused explicitly instead of blowing
  up in the middle of a scan.  A drifted database gets a sentence saying what is
  missing, not a traceback.

Report order is load-bearing.  The max-cosine distribution *of this database* is
printed BEFORE the threshold is applied, because a calibration taken on one
corpus does not transfer blind to another: the operator has to see the local
distribution before choosing where to cut.

``--dry-run`` is the default and opens the database strictly read-only
(``file:...?mode=ro``), so a dry run cannot write even by accident.  ``--apply``
writes ``supersedes`` edges and nothing else: no row is ever deleted, no
``decayed`` flag is set, no content is rewritten.

Bands, and why the default refuses to go below 0.95:

    >= 0.95   the same fact in different words -- collapsible
    0.85-0.95 DIFFERENT facts, and also the band where a concept sits next to
              its own source traces (that similarity is provenance, never
              duplication) -- refused unless --allow-mid-band is passed

The identifier veto is a copy, and the copy is the point.  A pair that differs
in an identifier -- a ULID, a path, a node name, a slug -- is two different
facts at ANY cosine, so it is refused rather than re-ranked (skip reason
``identifier_veto``).  The same rule lives in ``living_memory.near_dup``, and
this file duplicates it instead of importing it because importing it would cost
exactly the portability stated above.  The duplication is not taken on trust:
``tests/test_collapse_near_dups.py`` imports both implementations -- the test
may, this script may not -- and fails on any divergence in either direction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.parse
from array import array
from operator import mul

PROGRAM = "lm_collapse_near_dups.py"

#: Collapsing below this cosine needs an explicit opt-in: 0.85-0.95 is the band
#: where different facts -- and concept/source-trace provenance pairs -- live.
SAFE_THRESHOLD = 0.95
MID_BAND_FLOOR = 0.85

#: A candidate longer than ``bearer_chars * (1 + margin)`` is "the same fact
#: plus a new detail"; the detail has to survive, so the pair is never
#: collapsed.
DEFAULT_LENGTH_MARGIN = 0.10

#: Node levels whose ``source_traces`` mark the provenance band by default.
#: Both concepts and schemas are digests written OVER a cluster of sources, so
#: their similarity to those sources is provenance in exactly the same way.
DEFAULT_PROVENANCE_LEVELS = ("concept", "schema")

DEFAULT_MIN_COVERAGE = 0.50

#: Reported in the distribution section, ascending.  0.80 is also the floor
#: below which pair-level counting stops, which bounds the counting work.
REPORT_THRESHOLDS = (0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.99, 0.999)
REPORT_PERCENTILES = (10, 25, 50, 75, 90, 95, 99)

#: Loud, non-silent cap on how many above-threshold pairs are held in memory.
MAX_PAIRS = 500_000

BLOCK_ROWS = 512

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_REFUSED = 2

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class Refusal(Exception):
    """A preflight/precondition refusal: printed as one clear sentence."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def emit(line: str = "") -> None:
    sys.stdout.write(line + "\n")


def _ulid() -> str:
    """A ULID-shaped 26-char id, matching the ids already in the database.

    The schema only asks for TEXT, but staying in the house format keeps the
    rows sortable next to everything else that lives in ``connections``.
    """

    value = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _short(node_id: str) -> str:
    return node_id if len(node_id) <= 26 else node_id[:26]


def _preview(text, width=110):
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1] + "…"


def _percentile(sorted_values, percentile):
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (percentile / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def load_numpy(enabled=True):
    """Return the numpy module, or None when it is absent or switched off."""

    if not enabled:
        return None
    try:
        import numpy  # noqa: F401  (probe import; kept local on purpose)
    except Exception:  # pragma: no cover - exercised via LM_COLLAPSE_NUMPY=0
        return None
    return numpy


# --------------------------------------------------------------------------
# database access
# --------------------------------------------------------------------------


def connect(path, read_only):
    if not os.path.exists(path):
        raise Refusal("database file does not exist: %s" % path)
    if read_only:
        uri = "file:%s?mode=ro" % urllib.parse.quote(os.path.abspath(path))
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    else:
        connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    return connection


def table_names(connection):
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row["name"]) for row in rows}


def column_names(connection, table):
    rows = connection.execute("PRAGMA table_info(%s)" % table).fetchall()
    return {str(row["name"]) for row in rows}


class Preflight(object):
    """What the probe found, so the report can state it before doing anything."""

    def __init__(self):
        self.node_columns = set()
        self.chunk_columns = set()
        self.active_nodes = 0
        self.nodes_with_chunks = 0
        self.chunk_rows = 0
        self.widths = []  # [(dimensions, rows), ...]
        self.coverage = 0.0
        self.has_connections = False
        self.notes = []

    @property
    def width(self):
        return self.widths[0][0] if self.widths else 0


def preflight(connection, min_coverage):
    """Probe the schema and refuse clearly rather than crash later."""

    report = Preflight()
    tables = table_names(connection)

    if "nodes" not in tables:
        raise Refusal(
            "table 'nodes' is absent: this is not a Living Memory database "
            "(tables found: %s)" % (", ".join(sorted(tables)) or "none")
        )
    if "node_chunk_embeddings" not in tables:
        raise Refusal(
            "table 'node_chunk_embeddings' is absent: this database predates "
            "chunk embeddings, so there are no vectors to compare. Migrate it "
            "with the owning server (which creates the table on open) and "
            "backfill vectors before running hygiene."
        )

    report.has_connections = "connections" in tables
    report.node_columns = column_names(connection, "nodes")
    report.chunk_columns = column_names(connection, "node_chunk_embeddings")

    missing = {"id", "content", "decayed"} - report.node_columns
    if missing:
        raise Refusal(
            "table 'nodes' is missing required column(s): %s"
            % ", ".join(sorted(missing))
        )
    missing = {"node_id", "embedding"} - report.chunk_columns
    if missing:
        raise Refusal(
            "table 'node_chunk_embeddings' is missing required column(s): %s"
            % ", ".join(sorted(missing))
        )
    if not ({"source_traces", "provenance"} & report.node_columns):
        raise Refusal(
            "table 'nodes' has neither a 'source_traces' column nor a "
            "'provenance' column, so the provenance guard (never supersede a "
            "trace a live concept was built from) cannot be enforced. Refusing "
            "to collapse anything on this schema."
        )
    for optional in ("level", "scope", "access_count", "usefulness_score", "created_at"):
        if optional not in report.node_columns:
            report.notes.append(
                "column nodes.%s is absent; its guard/ranking input degrades to "
                "a constant" % optional
            )

    report.active_nodes = int(
        connection.execute("SELECT COUNT(*) FROM nodes WHERE decayed = 0").fetchone()[0]
    )
    if not report.active_nodes:
        raise Refusal("no active (decayed = 0) nodes in this database")

    if "dimensions" in report.chunk_columns:
        rows = connection.execute(
            """
            SELECT e.dimensions AS dimensions, COUNT(*) AS rows_count
            FROM node_chunk_embeddings e
            JOIN nodes n ON n.id = e.node_id
            WHERE n.decayed = 0
            GROUP BY e.dimensions
            ORDER BY rows_count DESC
            """
        ).fetchall()
        report.widths = [(int(r["dimensions"]), int(r["rows_count"])) for r in rows]
    else:
        report.notes.append(
            "column node_chunk_embeddings.dimensions is absent; vector width is "
            "derived from BLOB length / 4"
        )
        rows = connection.execute(
            """
            SELECT LENGTH(e.embedding) / 4 AS dimensions, COUNT(*) AS rows_count
            FROM node_chunk_embeddings e
            JOIN nodes n ON n.id = e.node_id
            WHERE n.decayed = 0
            GROUP BY 1
            ORDER BY rows_count DESC
            """
        ).fetchall()
        report.widths = [(int(r["dimensions"]), int(r["rows_count"])) for r in rows]

    report.chunk_rows = sum(count for _, count in report.widths)
    if not report.chunk_rows:
        raise Refusal(
            "table 'node_chunk_embeddings' holds no rows for active nodes: "
            "nothing to compare. Backfill chunk embeddings first."
        )
    if len(report.widths) > 1:
        detail = ", ".join(
            "%d dims x %d rows" % (width, count) for width, count in report.widths
        )
        raise Refusal(
            "mixed vector widths in node_chunk_embeddings (%s). Vectors of "
            "different width are points in different spaces and must not be "
            "compared; re-embed the corpus with one model before running "
            "hygiene." % detail
        )

    width = report.width
    if "dimensions" in report.chunk_columns:
        bad = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM node_chunk_embeddings e
                JOIN nodes n ON n.id = e.node_id
                WHERE n.decayed = 0 AND LENGTH(e.embedding) <> e.dimensions * 4
                """
            ).fetchone()[0]
        )
        if bad:
            raise Refusal(
                "%d chunk row(s) of active nodes have a BLOB whose length does "
                "not match their recorded dimensions (expected %d bytes for %d "
                "float32 values). The vector store is inconsistent; refusing to "
                "compare." % (bad, width * 4, width)
            )

    report.nodes_with_chunks = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT e.node_id)
            FROM node_chunk_embeddings e
            JOIN nodes n ON n.id = e.node_id
            WHERE n.decayed = 0
            """
        ).fetchone()[0]
    )
    report.coverage = report.nodes_with_chunks / float(report.active_nodes)
    if report.coverage < min_coverage:
        raise Refusal(
            "only %d of %d active nodes (%.1f%%) have chunk rows, below the "
            "--min-coverage floor of %.1f%%. A scan over that slice would call "
            "the corpus deduplicated while most of it was never looked at. "
            "Backfill chunk embeddings, or lower --min-coverage deliberately."
            % (
                report.nodes_with_chunks,
                report.active_nodes,
                100.0 * report.coverage,
                100.0 * min_coverage,
            )
        )
    return report


class NodeRow(object):
    __slots__ = (
        "id",
        "level",
        "scope",
        "chars",
        "preview",
        "access_count",
        "usefulness",
        "created_at",
    )

    def __init__(self, row, columns):
        self.id = str(row["id"])
        self.level = str(row["level"]) if "level" in columns else ""
        self.scope = str(row["scope"]) if "scope" in columns else ""
        self.chars = int(row["chars"] or 0)
        self.preview = str(row["preview"] or "")
        self.access_count = (
            int(row["access_count"] or 0) if "access_count" in columns else 0
        )
        self.usefulness = (
            float(row["usefulness_score"] or 0.0)
            if "usefulness_score" in columns
            else 0.0
        )
        self.created_at = (
            str(row["created_at"] or "") if "created_at" in columns else ""
        )


def load_nodes(connection, columns):
    """Active node metadata, without pulling whole contents into memory."""

    selected = ["id", "LENGTH(content) AS chars", "SUBSTR(content, 1, 160) AS preview"]
    for optional in ("level", "scope", "access_count", "usefulness_score", "created_at"):
        if optional in columns:
            selected.append(optional)
    sql = "SELECT %s FROM nodes WHERE decayed = 0" % ", ".join(selected)
    return {
        str(row["id"]): NodeRow(row, columns)
        for row in connection.execute(sql).fetchall()
    }


def _iter_source_trace_ids(raw):
    """Yield node ids out of a ``source_traces`` payload, tolerating drift."""

    if not raw:
        return
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (ValueError, TypeError):
        return
    if isinstance(value, dict):
        value = value.get("source_traces") or value.get("ids") or []
    if not isinstance(value, (list, tuple)):
        return
    for item in value:
        if isinstance(item, str) and item:
            yield item
        elif isinstance(item, dict):
            for key in ("id", "node_id", "trace_id"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate:
                    yield candidate
                    break


def load_provenance_sources(connection, columns, levels):
    """Ids listed in ``source_traces`` of live nodes at ``levels``.

    Read from the ``source_traces`` column and from a ``source_traces`` key
    inside the ``provenance`` JSON, because installations differ in which of
    the two carries it (the model layer merges the column into the provenance
    dict on the way out, so both spellings are in the wild).
    """

    selected = []
    if "source_traces" in columns:
        selected.append("source_traces")
    if "provenance" in columns:
        selected.append("provenance")
    if not selected:
        return set()

    where = ["decayed = 0"]
    params = []
    if "level" in columns and levels:
        where.append("level IN (%s)" % ", ".join("?" for _ in levels))
        params.extend(levels)
    sql = "SELECT %s FROM nodes WHERE %s" % (", ".join(selected), " AND ".join(where))

    protected = set()
    for row in connection.execute(sql, params).fetchall():
        if "source_traces" in selected:
            for node_id in _iter_source_trace_ids(row["source_traces"]):
                protected.add(node_id)
        if "provenance" in selected:
            raw = row["provenance"]
            if raw and "source_traces" in str(raw):
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    payload = None
                if isinstance(payload, dict):
                    for node_id in _iter_source_trace_ids(
                        payload.get("source_traces")
                    ):
                        protected.add(node_id)
    return protected


#: Ids per ``IN (...)`` query when contents are fetched. Well under sqlite's
#: variable limit on every version this script may meet.
CONTENT_BATCH = 400


def load_contents(connection, node_ids):
    """Full text of the named nodes -- the identifier veto's only input.

    Fetched for the nodes that reached the threshold and no others.  The rest
    of this script works from ``LENGTH(content)`` and a 160-character preview
    on purpose, and a corpus is orders of magnitude larger than the handful of
    pairs a 0.95 threshold admits.  With the veto off this is never called at
    all, which is what makes the valve a true rollback: no text is even read.
    """

    contents = {}
    ids = sorted(node_ids)
    for start in range(0, len(ids), CONTENT_BATCH):
        batch = ids[start : start + CONTENT_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        rows = connection.execute(
            "SELECT id, content FROM nodes WHERE id IN (%s)" % placeholders, batch
        ).fetchall()
        for row in rows:
            content = row["content"]
            contents[str(row["id"])] = "" if content is None else str(content)
    return contents


def load_supersedes(connection, has_connections):
    """(already superseded targets, superseding sources, existing pairs)."""

    if not has_connections:
        return set(), set(), set()
    rows = connection.execute(
        "SELECT source_id, target_id FROM connections WHERE type = 'supersedes'"
    ).fetchall()
    superseded = set()
    superseding = set()
    pairs = set()
    for row in rows:
        source = str(row["source_id"])
        target = str(row["target_id"])
        superseding.add(source)
        superseded.add(target)
        pairs.add((source, target))
    return superseded, superseding, pairs


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------


class VectorLoad(object):
    def __init__(self):
        self.ids = []
        self.vectors = []  # list[list[float]] -- unit-normalised mean pool
        self.chunk_counts = {}
        self.skipped_mixed_width = []
        self.skipped_zero_norm = []
        self.nodes_without_chunks = 0


def load_mean_pooled_vectors(connection, node_ids, width, has_dimensions):
    """Mean-pool each node's chunk vectors, then L2-normalise the mean.

    Mean-pool rather than max-pool because whole nodes are being compared with
    whole nodes here, not a query with a node.  Normalising after the mean is
    what makes the dot product a cosine downstream.
    """

    load = VectorLoad()
    dimension_expression = "e.dimensions" if has_dimensions else "LENGTH(e.embedding)/4"
    sql = """
        SELECT e.node_id AS node_id, %s AS dimensions, e.embedding AS embedding
        FROM node_chunk_embeddings e
        JOIN nodes n ON n.id = e.node_id
        WHERE n.decayed = 0
        ORDER BY e.node_id
    """ % dimension_expression

    current_id = None
    accumulator = None
    count = 0
    mixed = False

    def flush():
        if current_id is None:
            return
        load.chunk_counts[current_id] = count
        if mixed:
            load.skipped_mixed_width.append(current_id)
            return
        mean = [value / count for value in accumulator]
        norm = math.sqrt(sum(value * value for value in mean))
        if norm <= 0.0:
            load.skipped_zero_norm.append(current_id)
            return
        load.ids.append(current_id)
        load.vectors.append([value / norm for value in mean])

    for row in connection.execute(sql):
        node_id = str(row["node_id"])
        if node_id not in node_ids:
            continue
        if node_id != current_id:
            flush()
            current_id = node_id
            accumulator = [0.0] * width
            count = 0
            mixed = False
        blob = bytes(row["embedding"])
        dimensions = int(row["dimensions"] or 0)
        if dimensions != width or len(blob) != width * 4:
            mixed = True
            count += 1
            continue
        buffer = array("f")
        buffer.frombytes(blob)
        if sys.byteorder != "little":  # pragma: no cover - little-endian hosts
            buffer.byteswap()
        for index, value in enumerate(buffer):
            accumulator[index] += value
        count += 1
    flush()

    load.nodes_without_chunks = len(node_ids) - len(load.chunk_counts)
    return load


# --------------------------------------------------------------------------
# the cosine scan
# --------------------------------------------------------------------------


class ScanResult(object):
    def __init__(self):
        self.max_cosine = {}  # node_id -> best cosine against another node
        self.best_neighbour = {}  # node_id -> that neighbour's id
        self.pairs = []  # [(cosine, id_a, id_b)] at or above the threshold
        self.pair_counts = dict((t, 0) for t in REPORT_THRESHOLDS)
        self.compared_pairs = 0
        self.alone_in_group = []  # nodes whose group has no second member
        self.pairs_truncated = False


def _group_indices(ids, nodes, scope_mode):
    if scope_mode == "any":
        return {"*": list(range(len(ids)))}
    groups = {}
    for index, node_id in enumerate(ids):
        node = nodes.get(node_id)
        scope = node.scope if node is not None else ""
        groups.setdefault(scope, []).append(index)
    return groups


def scan(ids, vectors, nodes, threshold, scope_mode, numpy_module):
    """Max cosine per node plus every pair at or above ``threshold``.

    One pass produces both the distribution (which must be reported first) and
    the candidate pairs, so the threshold never re-reads the corpus.
    """

    result = ScanResult()
    groups = _group_indices(ids, nodes, scope_mode)
    count_floor = min(REPORT_THRESHOLDS)
    collect_floor = min(threshold, count_floor)

    for _, members in sorted(groups.items()):
        size = len(members)
        if size < 2:
            for index in members:
                result.alone_in_group.append(ids[index])
            continue
        result.compared_pairs += size * (size - 1) // 2
        if numpy_module is not None:
            _scan_group_numpy(result, ids, vectors, members, threshold, numpy_module)
        else:
            _scan_group_python(result, ids, vectors, members, threshold, collect_floor)

    # Rank pairs on a rounded cosine, not the raw one. numpy accumulates the
    # dot product in float32 and the pure-Python path in float64, so the same
    # pair differs by ~1e-7 between them; on an exactly-tied cluster (byte-equal
    # nodes, cosine 1.0) that noise decides the processing order and therefore
    # which member of the cluster becomes the bearer. Rounding puts the tie
    # break back on the ids, so both backends chain onto the same node.
    result.pairs.sort(key=lambda item: (-round(item[0], 5), item[1], item[2]))
    return result


def _record_pair(result, cosine, id_a, id_b, threshold, collect_floor):
    # collect_floor is only an optimisation: it is never above the lowest
    # reported threshold, so a pair below it contributes to no count anyway.
    # The numpy path counts straight off the block matrix and needs no floor.
    if cosine >= collect_floor:
        for reported in REPORT_THRESHOLDS:
            if cosine >= reported:
                result.pair_counts[reported] += 1
    if cosine >= threshold:
        if len(result.pairs) < MAX_PAIRS:
            result.pairs.append((cosine, id_a, id_b))
        else:
            result.pairs_truncated = True


def _bump_max(result, node_id, cosine, neighbour_id):
    if cosine > result.max_cosine.get(node_id, -2.0):
        result.max_cosine[node_id] = cosine
        result.best_neighbour[node_id] = neighbour_id


def _scan_group_python(result, ids, vectors, members, threshold, collect_floor):
    size = len(members)
    local = [vectors[index] for index in members]
    local_ids = [ids[index] for index in members]
    for i in range(size):
        left = local[i]
        left_id = local_ids[i]
        for j in range(i + 1, size):
            cosine = sum(map(mul, left, local[j]))
            if cosine > 1.0:
                cosine = 1.0
            right_id = local_ids[j]
            _bump_max(result, left_id, cosine, right_id)
            _bump_max(result, right_id, cosine, left_id)
            _record_pair(result, cosine, left_id, right_id, threshold, collect_floor)


def _scan_group_numpy(result, ids, vectors, members, threshold, numpy):
    local_ids = [ids[index] for index in members]
    matrix = numpy.asarray([vectors[index] for index in members], dtype=numpy.float32)
    size = matrix.shape[0]
    transposed = numpy.ascontiguousarray(matrix.T)
    columns = numpy.arange(size)
    for start in range(0, size, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, size)
        block = matrix[start:stop]
        sims = block.dot(transposed)
        numpy.clip(sims, -1.0, 1.0, out=sims)
        rows = numpy.arange(start, stop)
        sims[numpy.arange(stop - start), rows] = -2.0  # never a node with itself
        row_max = sims.max(axis=1)
        row_arg = sims.argmax(axis=1)
        for offset in range(stop - start):
            _bump_max(
                result,
                local_ids[start + offset],
                float(row_max[offset]),
                local_ids[int(row_arg[offset])],
            )
        # Count and collect on the upper triangle only, so a symmetric pair is
        # seen exactly once.
        upper = numpy.where(columns[None, :] > rows[:, None], sims, -2.0)
        for reported in REPORT_THRESHOLDS:
            result.pair_counts[reported] += int((upper >= reported).sum())
        for row_index, column_index in numpy.argwhere(upper >= threshold):
            cosine = float(upper[row_index, column_index])
            if len(result.pairs) < MAX_PAIRS:
                result.pairs.append(
                    (
                        cosine,
                        local_ids[start + int(row_index)],
                        local_ids[int(column_index)],
                    )
                )
            else:
                result.pairs_truncated = True


# --------------------------------------------------------------------------
# the identifier veto -- a deliberate copy of living_memory.near_dup
# --------------------------------------------------------------------------
#
# Everything down to ``IdentifierVeto`` is duplicated from
# ``living_memory.near_dup``, not imported from it.  Importing would buy one
# definition and cost the portability contract at the top of this file: this
# script runs under the interpreter of whichever server owns the database it is
# pointed at, and that server's checkout is a different version of the package,
# or is not a checkout at all.  The duplication is instead PROVEN harmless:
# ``tests/test_collapse_near_dups.py`` imports both implementations (a test may;
# this script may not) and asserts verdict-for-verdict agreement over a table
# covering every identifier class and its negatives, so a drift in either
# direction fails the suite rather than quietly changing what is collapsed here.
#
# Why the rule is a veto and not a threshold: «узел layer-fauna СДЕЛАН» and
# «узел layer-actors СДЕЛАН» score 0.9547 on the alt corpus -- above the 0.95
# this script collapses at -- and they are different facts.  The template
# dominates the vector and the one token that carries the fact is a rounding
# error in it, so no cut separates that pair from an honest repeat.  The rule is
# deliberately trigger-happy for the same reason: a false veto costs one
# uncollapsed near-duplicate, a false pass hides a fact.

#: The one valve, spelled exactly as in ``living_memory.near_dup``, so a single
#: env line means the same thing to the map, the drain and this script.
IDENTIFIER_VETO_ENV = "LM_NEAR_DUP_IDENTIFIER_VETO"

#: Everything else -- unset, ``1``, ``on``, or a typo -- leaves the veto on. An
#: unreadable value must not silently disable a guard whose whole purpose is to
#: fire when the evidence is unclear.
_VETO_OFF_FLAGS = frozenset({"0", "false", "no", "off"})

#: A maximal run of identifier-shaped characters. ``\w`` is Unicode, so
#: Cyrillic slugs are tokens too; ``/`` and ``~`` may also *start* one, which is
#: what keeps ``/home/sfx`` and ``~/.local/share`` whole.
_IDENTIFIER_TOKEN_RE = re.compile(r"[\w~/][\w./:~-]*", re.UNICODE)
#: Trailing punctuation ends a sentence, not a name: ``near_dup.py.`` and
#: ``project:`` lose their tail. Leading ``/`` and ``~`` are kept -- they are
#: the path -- while a leading ``.``/``:``/``-`` is trimmed like the tail.
_TOKEN_TRAILING_TRIM = "./:-~"
_TOKEN_LEADING_TRIM = ".:-"
#: One character is never a name; two can be (``v2``, ``id``-shaped slugs).
_MIN_IDENTIFIER_CHARS = 2
#: The length below which a token needs actual structure rather than merely a
#: digit or a case transition. Four is where a bare number stops reading as
#: prose: ``2026`` is a year, ``3`` in "measured at 3 units" is arithmetic.
_MIN_WEAK_IDENTIFIER_CHARS = 4
#: Any one of these makes a token an identifier outright.
_SLUG_SEPARATORS = ("-", "_", "/", ":")
#: 26 characters of Crockford base32 (no I, L, O or U): a ULID.
_ULID_RE = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
#: Seven hex characters is a short git id; digests are longer. The digit
#: requirement lives at the call site, so hex-shaped words never match.
_HEX_DIGEST_RE = re.compile(r"[0-9a-fA-F]{7,}")

#: Stands in for a text the veto was asked about and could not read. It is a
#: lost identifier as far as the decision goes, which is what makes an
#: unenforceable check refuse instead of pass.
TEXT_UNAVAILABLE = "<text not loaded>"


def identifier_veto_enabled():
    """Read ``LM_NEAR_DUP_IDENTIFIER_VETO``; on unless explicitly turned off.

    Only ``0``/``false``/``no``/``off`` (any case, surrounding blank ignored)
    turn it off; unset and unrecognized both leave it ON, because the guard
    exists for the cases where the evidence is unclear and a mistyped valve is
    one of them.
    """

    return os.environ.get(IDENTIFIER_VETO_ENV, "").strip().lower() not in _VETO_OFF_FLAGS


def extract_identifiers(text):
    """Identifier-shaped tokens in ``text``, de-duplicated, in first-seen order.

    A token is the maximal run of identifier-shaped characters (word characters
    plus ``. / : - ~``), trimmed of the punctuation that ends a sentence rather
    than a name. It counts as an identifier when any of these holds:

    * it contains ``-``, ``_``, ``/`` or ``:`` -- hyphen and underscore slugs
      (``layer-fauna``, ``build_duplicate_map``), paths
      (``src/living_memory/near_dup.py``, ``~/.local/share``), scoped names
      (``project:lm``) and URLs. Hyphen slugs are why the rule cannot be
      narrower: ``layer-fauna`` is two lowercase words joined by a hyphen, so
      Russian ``что-то`` and English ``well-known`` are read as identifiers
      too. That is the trade -- a false veto costs one uncollapsed pair;
    * it is dotted with at least one part of two characters or more --
      ``living_memory.near_dup``, ``near_dup.py``, ``0.9547``. The
      two-character part is what keeps ``e.g``, ``i.e`` and ``т.е`` out;
    * it is a 26-character Crockford id (a ULID);
    * it is seven or more hex characters including a digit -- commit ids and
      digests. The digit is what keeps hex-shaped English words out;
    * it is four characters or more and carries a digit -- years, ticket ids,
      versions, PIDs;
    * it is four characters or more with a lowercase-to-uppercase transition.

    Deliberately NOT an identifier: a bare word however long, an all-caps word
    (``СДЕЛАН``, ``IMPORTANT`` -- emphasis is not a name), and a number of
    three characters or fewer.
    """

    seen = {}
    for match in _IDENTIFIER_TOKEN_RE.finditer(text):
        token = match.group().rstrip(_TOKEN_TRAILING_TRIM).lstrip(_TOKEN_LEADING_TRIM)
        if token and _is_identifier(token):
            seen.setdefault(token, None)
    return tuple(seen)


def identifiers_absent_from(text, bearer_text):
    """``text``'s identifiers that are not *tokens* of ``bearer_text``.

    Token equality, not substring containment, and case-sensitive. Both sides
    go through :func:`extract_identifiers` and the comparison is string
    equality between extracted tokens: the bearer has to name the token as a
    token of its own, not merely contain it inside a longer one.

    Containment was the rule until it was measured. Node names in a goal tree
    are built by suffixing, so ``…/checkpoint-selected-profile-v2`` is literally
    a substring of ``…/checkpoint-selected-profile-v2-repaired``; with the
    longer-named node bearing, the candidate's whole path "occurred in" the
    bearer's text, the diff came back empty, and two distinct tree nodes with
    two distinct recorded failures collapsed as one fact at cosine 0.99350
    (project:x, ``01KTRR6WHFXFW16M691Q1N98E1`` ->
    ``01KTRJCB5FQ7ZT2WKD02TQ20B3``). The same shape put
    ``…/stock-contract-preservation-audit`` under ``…-audit-reintegrate`` at
    0.99064. The token rule also closes a numeric class: ``0.99`` is a
    substring of ``0.99350`` but not a token of it.

    What it costs is the case containment was written for -- a bearer spelling
    ``src/living_memory/near_dup.py`` no longer covers a candidate spelling the
    bare ``near_dup.py`` -- and that is the trade this veto has made
    throughout: a false veto costs one uncollapsed pair, a false pass hides a
    fact. Not softened into ``/``-aligned containment either; ``a/x.py`` and
    ``b/x.py`` break that the same way.

    Case-sensitive because these are paths, env vars and ids, where case is
    meaning, and because the ambiguous direction is the one that vetoes.
    """

    bearer_identifiers = frozenset(extract_identifiers(bearer_text))
    return tuple(
        token
        for token in extract_identifiers(text)
        if token not in bearer_identifiers
    )


def _is_identifier(token):
    """One trimmed token against the classes documented on ``extract_identifiers``."""

    if len(token) < _MIN_IDENTIFIER_CHARS:
        return False
    if any(separator in token for separator in _SLUG_SEPARATORS):
        return True
    if "." in token and _is_dotted_path(token):
        return True
    if _ULID_RE.fullmatch(token):
        return True
    has_digit = any(character.isdigit() for character in token)
    if has_digit and _HEX_DIGEST_RE.fullmatch(token):
        return True
    if len(token) < _MIN_WEAK_IDENTIFIER_CHARS:
        return False
    return has_digit or _has_case_transition(token)


def _is_dotted_path(token):
    """A dotted token with two or more parts, one of them two characters or more.

    ``0.95`` and ``near_dup.py`` qualify; ``e.g``, ``i.e``, ``т.е`` and ``1.5``
    do not. Abbreviations are dots between single characters; names are not.
    """

    parts = [part for part in token.split(".") if part]
    return len(parts) >= 2 and any(len(part) >= 2 for part in parts)


def _has_case_transition(token):
    """Whether a lowercase character is immediately followed by an uppercase one."""

    return any(
        left.islower() and right.isupper() for left, right in zip(token, token[1:])
    )


class IdentifierVeto(object):
    """What a collapse would hide, pair by pair.

    Holds the full text of the nodes that reached the threshold, and only
    those. Switched off it answers "nothing" without reading anything, so the
    valve is a rollback rather than a second code path.
    """

    def __init__(self, contents, enabled=True):
        self.enabled = bool(enabled)
        self.contents = contents

    def lost(self, candidate_id, bearer_id):
        """Identifiers the candidate names that the bearer's text does not.

        Empty is the only answer that lets a collapse through. A side whose
        text could not be read yields the unreadable marker rather than an
        empty tuple: a veto that cannot see the text cannot be enforced, and an
        unenforceable guard has to fail closed.

        One-directional on purpose. The bearer's own extra identifiers still
        reach the agent -- the bearer keeps its whole text -- and it is the
        candidate's that would disappear behind a supersedes edge.

        The comparison goes through :func:`identifiers_absent_from` rather than
        being spelled out here, and re-extracts BOTH texts for every pair it is
        asked about instead of caching either. Both are deliberate: the
        mirrored function is then the one the decision actually runs on, so the
        agreement test pins the decision and not merely a lookalike beside it.
        (``near_dup.build_duplicate_map`` does cache the bearer's token set --
        its inner scan walks every pair of arrivals, not just the pairs a
        threshold already admitted. Same rule, different shape of loop.) Here
        a threshold admits a few thousand pairs on a corpus of tens of
        thousands of nodes, so two extractions per pair are noise next to the
        scan that produced them.
        """

        if not self.enabled:
            return ()
        candidate_text = self.contents.get(candidate_id)
        bearer_text = self.contents.get(bearer_id)
        if candidate_text is None or bearer_text is None:
            return (TEXT_UNAVAILABLE,)
        return identifiers_absent_from(candidate_text, bearer_text)


def _veto_detail(lost):
    """The skip line's parenthetical: what this pair would have hidden."""

    if lost == (TEXT_UNAVAILABLE,):
        return "one side's text could not be read; refusing to collapse unchecked"
    shown = ", ".join(lost[:4])
    if len(lost) > 4:
        shown += ", and %d more" % (len(lost) - 4)
    return "candidate names %s; the bearer's text does not" % shown


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------

SKIP_REASONS = (
    ("level_mismatch", "the two nodes are different levels (trace/concept/schema)"),
    (
        "provenance_source_trace",
        "the node that would be superseded is listed in source_traces of a live "
        "concept/schema: that band is provenance, never duplication",
    ),
    (
        "candidate_is_correction",
        "the node that would be superseded is itself the source of a supersedes "
        "edge (it corrects something); collapsing it would bury the correction",
    ),
    (
        "candidate_already_superseded",
        "the node that would be superseded already has an incoming supersedes edge",
    ),
    (
        "bearer_already_superseded",
        "the node that would bear already has an incoming supersedes edge: pointing "
        "anything at it would hang the pair off a node the system already demotes",
    ),
    ("edge_already_present", "this supersedes edge already exists in the database"),
    (
        "longer_than_bearer",
        "the candidate is materially longer than its bearer: same fact plus a new "
        "detail, and the detail has to reach the agent",
    ),
    (
        "identifier_veto",
        "the candidate names an identifier -- a ULID, a path, a node name, a slug "
        "-- that the bearer's text does not: whatever the cosine says, those are "
        "two different facts",
    ),
    ("candidate_already_collapsed", "the candidate was already collapsed in this run"),
    ("cycle", "collapsing this pair would point a node at something it already bears"),
)


class Collapse(object):
    def __init__(self, cosine, bearer, candidate, rechained_from=None):
        self.cosine = cosine
        self.bearer = bearer
        self.candidate = candidate
        self.rechained_from = rechained_from


class Skip(object):
    def __init__(self, cosine, reason, bearer, candidate, detail=""):
        self.cosine = cosine
        self.reason = reason
        self.bearer = bearer
        self.candidate = candidate
        self.detail = detail


def _better_bearer(left, right):
    """True when ``left`` should survive and ``right`` should be superseded.

    Usage first (a node agents actually read is the one worth keeping), then
    usefulness, then age (older wins: it is the one other things already point
    at), then id for determinism.  Provenance deliberately does NOT enter this
    ranking -- it stays a visible skip reason instead of being silently routed
    around.
    """

    left_key = (left.access_count, left.usefulness)
    right_key = (right.access_count, right.usefulness)
    if left_key != right_key:
        return left_key > right_key
    if left.created_at != right.created_at:
        return left.created_at < right.created_at
    return left.id < right.id


def decide(
    pairs,
    nodes,
    protected,
    superseded,
    superseding,
    existing_pairs,
    margin,
    veto=None,
):
    """Turn ranked pairs into collapses plus a reason for every refusal.

    ``veto`` is an :class:`IdentifierVeto`; omitted (or switched off) it lets
    every pair through untouched, which is the ``LM_NEAR_DUP_IDENTIFIER_VETO=0``
    path and the pre-veto behaviour of this function.
    """

    if veto is None:
        veto = IdentifierVeto({}, enabled=False)

    collapses = []
    skips = []
    bearer_of = {}  # candidate id -> bearer id, for this run

    def root(node_id):
        seen = set()
        while node_id in bearer_of and node_id not in seen:
            seen.add(node_id)
            node_id = bearer_of[node_id]
        return node_id

    for cosine, id_a, id_b in pairs:
        node_a = nodes.get(id_a)
        node_b = nodes.get(id_b)
        if node_a is None or node_b is None:
            continue
        if node_a.level != node_b.level:
            skips.append(
                Skip(cosine, "level_mismatch", node_a, node_b, "%s vs %s" % (node_a.level, node_b.level))
            )
            continue

        if _better_bearer(node_a, node_b):
            bearer, candidate = node_a, node_b
        else:
            bearer, candidate = node_b, node_a

        if candidate.id in bearer_of:
            skips.append(
                Skip(
                    cosine,
                    "candidate_already_collapsed",
                    bearer,
                    candidate,
                    "already superseded by %s in this run" % _short(bearer_of[candidate.id]),
                )
            )
            continue

        rechained_from = None
        bearer_root = root(bearer.id)
        if bearer_root != bearer.id:
            rechained_from = bearer.id
            replacement = nodes.get(bearer_root)
            if replacement is None:
                skips.append(Skip(cosine, "cycle", bearer, candidate, "bearer chain left the corpus"))
                continue
            bearer = replacement
        if bearer.id == candidate.id:
            skips.append(Skip(cosine, "cycle", bearer, candidate, "bearer chain returns to the candidate"))
            continue

        if bearer.id in superseded:
            skips.append(
                Skip(
                    cosine,
                    "bearer_already_superseded",
                    bearer,
                    candidate,
                    "the node that would bear is itself superseded already",
                )
            )
            continue
        if candidate.id in protected:
            skips.append(
                Skip(
                    cosine,
                    "provenance_source_trace",
                    bearer,
                    candidate,
                    "bearer-side collapse also blocked"
                    if bearer.id in protected
                    else "collapsing the other way round would be possible",
                )
            )
            continue
        if candidate.id in superseding:
            skips.append(Skip(cosine, "candidate_is_correction", bearer, candidate))
            continue
        if candidate.id in superseded:
            skips.append(Skip(cosine, "candidate_already_superseded", bearer, candidate))
            continue
        if (bearer.id, candidate.id) in existing_pairs:
            skips.append(Skip(cosine, "edge_already_present", bearer, candidate))
            continue
        if candidate.chars > bearer.chars * (1.0 + margin):
            skips.append(
                Skip(
                    cosine,
                    "longer_than_bearer",
                    bearer,
                    candidate,
                    "%d chars vs %d (+%.0f%% allowed)"
                    % (candidate.chars, bearer.chars, 100.0 * margin),
                )
            )
            continue
        # Last, and in the same position the map checks it: the length guard
        # fires on a missing detail, this one on a missing name.
        lost = veto.lost(candidate.id, bearer.id)
        if lost:
            skips.append(
                Skip(cosine, "identifier_veto", bearer, candidate, _veto_detail(lost))
            )
            continue

        bearer_of[candidate.id] = bearer.id
        collapses.append(Collapse(cosine, bearer, candidate, rechained_from))

    return _flatten(collapses, skips, margin, veto), skips


def _flatten(collapses, skips, margin, veto):
    """Re-point every collapse at the node that actually survives.

    The main pass can record LOW -> MID on a high-cosine pair and then, on a
    lower-cosine pair, MID -> TOP. Left as a chain the report would name MID as
    "kept" while MID is itself superseded, and -- worse -- the length guard
    would have been checked against a node that no longer bears anything: LOW
    could be materially longer than the node that ends up surviving. So each
    edge is re-pointed at the root of its chain and the guard is re-checked
    there, and a collapse whose surviving root is too short is dropped rather
    than quietly chained.

    The identifier veto is re-checked at the root for exactly the same reason:
    what the agent is left holding is the root's text, so that is the text the
    candidate's identifiers have to be in.
    """

    alive = list(collapses)
    while True:
        links = dict((item.candidate.id, item.bearer) for item in alive)
        roots = {}
        dropped = []
        for item in alive:
            root = item.bearer
            seen = set([item.candidate.id])
            while root.id in links and root.id not in seen:
                seen.add(root.id)
                root = links[root.id]
            rechained = root.id != item.bearer.id
            root_lost = veto.lost(item.candidate.id, root.id) if rechained else ()
            if root.id == item.candidate.id:
                dropped.append((item, "cycle", "chain returns to the candidate"))
            elif rechained and item.candidate.chars > root.chars * (1.0 + margin):
                dropped.append(
                    (
                        item,
                        "longer_than_bearer",
                        "%d chars vs %d at the surviving node (chained through %s)"
                        % (item.candidate.chars, root.chars, _short(item.bearer.id)),
                    )
                )
            elif root_lost:
                dropped.append(
                    (
                        item,
                        "identifier_veto",
                        "%s at the surviving node (chained through %s)"
                        % (_veto_detail(root_lost), _short(item.bearer.id)),
                    )
                )
            else:
                roots[id(item)] = root
        if not dropped:
            for item in alive:
                root = roots[id(item)]
                if root.id != item.bearer.id:
                    item.rechained_from = item.bearer.id
                    item.bearer = root
            return alive
        # Dropping a link changes where the chains below it end up, so resolve
        # again rather than trusting the roots computed against the old map.
        for item, reason, detail in dropped:
            skips.append(Skip(item.cosine, reason, item.bearer, item.candidate, detail))
        gone = set(id(item) for item, _, _ in dropped)
        alive = [item for item in alive if id(item) not in gone]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_preflight(report, args, numpy_module):
    emit("## preflight")
    emit("database                : %s" % os.path.abspath(args.database))
    emit("opened                  : %s" % ("read-only (dry run)" if not args.apply else "read-write (apply)"))
    emit("numpy                   : %s" % (numpy_module.__version__ if numpy_module else "absent - pure-Python scan"))
    emit("node_chunk_embeddings   : present, %d rows for active nodes" % report.chunk_rows)
    emit("vector width            : %d (uniform)" % report.width)
    emit(
        "chunk coverage          : %d of %d active nodes (%.1f%%), floor %.1f%%"
        % (
            report.nodes_with_chunks,
            report.active_nodes,
            100.0 * report.coverage,
            100.0 * args.min_coverage,
        )
    )
    emit("connections table       : %s" % ("present" if report.has_connections else "ABSENT (apply impossible)"))
    for note in report.notes:
        emit("note                    : %s" % note)
    emit()


def print_distribution(scan_result, load, args):
    """The local distribution, printed BEFORE any threshold is applied."""

    emit("## max-cosine distribution for THIS database")
    emit(
        "Calibration does not transfer between corpora. These are the numbers of"
    )
    emit("the database named above, measured before any threshold was applied.")
    emit()
    values = sorted(scan_result.max_cosine.values())
    emit("nodes compared          : %d" % len(values))
    emit(
        "pairing                 : %s"
        % (
            "within the same scope"
            if args.scope_mode == "same"
            else "across all scopes"
        )
    )
    emit("pairs compared          : %d" % scan_result.compared_pairs)
    if load.nodes_without_chunks:
        emit(
            "nodes without chunks    : %d (excluded: nothing to compare)"
            % load.nodes_without_chunks
        )
    if load.skipped_mixed_width:
        emit(
            "nodes with mixed widths : %d (excluded)" % len(load.skipped_mixed_width)
        )
    if load.skipped_zero_norm:
        emit("nodes with a zero vector: %d (excluded)" % len(load.skipped_zero_norm))
    if scan_result.alone_in_group:
        emit(
            "nodes alone in their scope: %d (no in-scope neighbour to compare with)"
            % len(scan_result.alone_in_group)
        )
    emit()
    if values:
        emit("max-cosine percentiles (per node, against its nearest other node):")
        emit("  mean   %.4f" % (sum(values) / len(values)))
        emit("  min    %.4f" % values[0])
        for percentile in REPORT_PERCENTILES:
            emit("  p%-5d %.4f" % (percentile, _percentile(values, percentile)))
        emit("  max    %.4f" % values[-1])
        emit()
        emit("counts at meaningful thresholds:")
        emit("  %-9s %10s %14s %10s" % ("cosine", "nodes", "share of nodes", "pairs"))
        total = len(values)
        for threshold in REPORT_THRESHOLDS:
            node_count = sum(1 for value in values if value >= threshold)
            emit(
                "  >= %-6.3f %10d %13.2f%% %10d"
                % (threshold, node_count, 100.0 * node_count / total, scan_result.pair_counts[threshold])
            )
    emit()


def print_band_note(args, identifier_veto):
    emit("## threshold")
    emit("threshold applied       : %.4f" % args.threshold)
    emit(
        "band policy             : >= %.2f collapsible; %.2f-%.2f left alone by "
        "default (different facts, and concept/source-trace provenance)"
        % (SAFE_THRESHOLD, MID_BAND_FLOOR, SAFE_THRESHOLD)
    )
    if args.threshold < SAFE_THRESHOLD:
        emit("MID-BAND OVERRIDE ACTIVE: --allow-mid-band was passed.")
    emit("length guard            : candidate may not exceed bearer by more than %.0f%%" % (100.0 * args.length_margin))
    emit(
        "provenance guard        : never supersede a node listed in source_traces "
        "of a live %s node" % "/".join(args.provenance_levels)
    )
    if identifier_veto:
        emit(
            "identifier veto         : ON (%s); a candidate naming an identifier "
            "its bearer's text does not is never collapsed" % IDENTIFIER_VETO_ENV
        )
    else:
        emit(
            "identifier veto         : OFF (%s is set off); identifier-losing "
            "pairs collapse like any other" % IDENTIFIER_VETO_ENV
        )
    emit()


def print_decisions(collapses, skips, scan_result, args):
    emit("## pairs at or above the threshold")
    emit("pairs                   : %d" % len(scan_result.pairs))
    if scan_result.pairs_truncated:
        emit(
            "TRUNCATED               : more than %d pairs matched; only the first "
            "%d were kept. Raise the threshold." % (MAX_PAIRS, MAX_PAIRS)
        )
    emit("would collapse          : %d" % len(collapses))
    emit("skipped                 : %d" % len(skips))
    emit()

    emit("## would collapse")
    if not collapses:
        emit("(nothing)")
    limit = args.report_limit
    for index, item in enumerate(collapses):
        if limit and index >= limit:
            emit("... and %d more (raise --report-limit to see them)" % (len(collapses) - limit))
            break
        emit(
            "COLLAPSE cos=%.4f  supersede %s  ->  keep %s"
            % (item.cosine, _short(item.candidate.id), _short(item.bearer.id))
        )
        emit(
            "         level=%s scope=%s  chars %d -> %d  access %d -> %d  created %s -> %s%s"
            % (
                item.candidate.level or "?",
                item.candidate.scope or "?",
                item.candidate.chars,
                item.bearer.chars,
                item.candidate.access_count,
                item.bearer.access_count,
                item.candidate.created_at[:10] or "?",
                item.bearer.created_at[:10] or "?",
                "  [rechained from %s]" % _short(item.rechained_from) if item.rechained_from else "",
            )
        )
        emit("         drop: %s" % _preview(item.candidate.preview))
        emit("         keep: %s" % _preview(item.bearer.preview))
    emit()

    emit("## skipped, and why")
    if not skips:
        emit("(nothing)")
    by_reason = {}
    for item in skips:
        by_reason.setdefault(item.reason, []).append(item)
    for reason, description in SKIP_REASONS:
        items = by_reason.get(reason)
        if not items:
            continue
        emit("### %s (%d)" % (reason, len(items)))
        emit("    %s" % description)
        for index, item in enumerate(items):
            if limit and index >= limit:
                emit("    ... and %d more" % (len(items) - limit))
                break
            emit(
                "    SKIP cos=%.4f  keep %s  would-supersede %s%s"
                % (
                    item.cosine,
                    _short(item.bearer.id),
                    _short(item.candidate.id),
                    "  (%s)" % item.detail if item.detail else "",
                )
            )
        emit()

    emit("## skip breakdown")
    if not skips:
        emit("(no skips)")
    for reason, _ in SKIP_REASONS:
        if reason in by_reason:
            emit("  %-28s %d" % (reason, len(by_reason[reason])))
    unknown = set(by_reason) - set(reason for reason, _ in SKIP_REASONS)
    for reason in sorted(unknown):
        emit("  %-28s %d" % (reason, len(by_reason[reason])))
    emit()


OPERATIONAL_NOTE = (
    "## operational note\n"
    "Run this against a database whose server is STOPPED, or restart that server\n"
    "immediately afterwards. A running server keeps the scope chunk matrix cached\n"
    "in memory; collapsing nodes underneath it leaves that cache stale, so recall\n"
    "keeps serving the nodes this pass just superseded until the process restarts.\n"
    "The edges written here also bypass the server's own supersedes hook, so query\n"
    "anchor edges still point at superseded nodes until the next anchor maintenance\n"
    "pass; recall demotes superseded nodes regardless, so this is a ranking nicety,\n"
    "not a correctness hole."
)


def apply_collapses(connection, collapses, threshold):
    """Write ``supersedes`` edges. Nothing else is written, ever.

    One IMMEDIATE transaction: either every edge of this pass lands or none
    does, and a database another process is writing to fails here loudly
    instead of half-way through.
    """

    now = _utc_now()
    written = 0
    skipped_existing = 0
    connection.isolation_level = None
    connection.execute("BEGIN IMMEDIATE")
    try:
        for item in collapses:
            metadata = json.dumps(
                {
                    "by": PROGRAM,
                    "reason": "near-duplicate collapse",
                    "cosine": round(item.cosine, 6),
                    "threshold": threshold,
                    "bearer_chars": item.bearer.chars,
                    "candidate_chars": item.candidate.chars,
                },
                sort_keys=True,
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO connections
                    (id, source_id, target_id, type, weight, metadata, created_at, updated_at)
                VALUES (?, ?, ?, 'supersedes', 1.0, ?, ?, ?)
                """,
                (_ulid(), item.bearer.id, item.candidate.id, metadata, now, now),
            )
            if cursor.rowcount:
                written += 1
            else:
                skipped_existing += 1
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
    return written, skipped_existing


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Collapse accumulated near-duplicate Living Memory nodes through "
            "supersedes edges. Standard library only; never imports "
            "living_memory; runs with or without numpy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The max-cosine distribution of the given database is always printed\n"
            "before the threshold is applied. --dry-run (the default) opens the\n"
            "database read-only. --apply writes supersedes edges and nothing else.\n"
        ),
    )
    parser.add_argument("database", help="path to the Living Memory sqlite database")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SAFE_THRESHOLD,
        help="cosine at or above which a pair counts as a near-duplicate "
        "(default %.2f)" % SAFE_THRESHOLD,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        default=False,
        help="report only, opening the database read-only (default)",
    )
    mode.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="write the supersedes edges the dry run listed",
    )
    parser.add_argument(
        "--allow-mid-band",
        action="store_true",
        help="permit a threshold below %.2f, i.e. inside the %.2f-%.2f band where "
        "different facts live" % (SAFE_THRESHOLD, MID_BAND_FLOOR, SAFE_THRESHOLD),
    )
    parser.add_argument(
        "--scope-mode",
        choices=("same", "any"),
        default="same",
        help="compare nodes only within one scope (default) or across all scopes",
    )
    parser.add_argument(
        "--length-margin",
        type=float,
        default=DEFAULT_LENGTH_MARGIN,
        help="a candidate longer than bearer*(1+margin) is never collapsed "
        "(default %.2f)" % DEFAULT_LENGTH_MARGIN,
    )
    parser.add_argument(
        "--provenance-levels",
        default=",".join(DEFAULT_PROVENANCE_LEVELS),
        help="levels whose source_traces mark protected provenance "
        "(default %s)" % ",".join(DEFAULT_PROVENANCE_LEVELS),
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="refuse if fewer than this fraction of active nodes have chunk rows "
        "(default %.2f)" % DEFAULT_MIN_COVERAGE,
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=60,
        help="max lines per report section, 0 for all (default 60)",
    )
    parser.add_argument("--json", dest="json_path", help="also write a JSON summary here")
    parser.add_argument(
        "--no-numpy",
        action="store_true",
        help="force the pure-Python scan even when numpy imports",
    )
    args = parser.parse_args(argv)
    args.provenance_levels = tuple(
        part.strip() for part in args.provenance_levels.split(",") if part.strip()
    )
    return args


def validate(args):
    if not (0.0 < args.threshold <= 1.0):
        raise Refusal("--threshold must be in (0, 1]; got %r" % args.threshold)
    if args.length_margin < 0.0:
        raise Refusal("--length-margin must not be negative")
    if args.threshold < SAFE_THRESHOLD and not args.allow_mid_band:
        raise Refusal(
            "refusing a threshold of %.4f: %.2f-%.2f is the band where DIFFERENT "
            "facts live, and where a concept sits next to its own source traces "
            "(that similarity is provenance, not duplication). Pass "
            "--allow-mid-band if you have read this database's distribution and "
            "mean it." % (args.threshold, MID_BAND_FLOOR, SAFE_THRESHOLD)
        )


def build_command_line(args, database, identifier_veto=True):
    """The apply command for exactly this pass, env valve included.

    The valve rides in front of the command rather than being left implicit:
    a pass run with the veto off collapses pairs a default pass refuses, so an
    operator retyping the command without it would apply a different pass.
    """

    parts = []
    if not identifier_veto:
        parts.append("%s=0" % IDENTIFIER_VETO_ENV)
    parts += [
        "python3",
        "scripts/%s" % PROGRAM,
        database,
        "--threshold",
        "%.2f" % args.threshold,
    ]
    if args.scope_mode != "same":
        parts.extend(["--scope-mode", args.scope_mode])
    if args.length_margin != DEFAULT_LENGTH_MARGIN:
        parts.extend(["--length-margin", "%.2f" % args.length_margin])
    if args.provenance_levels != DEFAULT_PROVENANCE_LEVELS:
        parts.extend(["--provenance-levels", ",".join(args.provenance_levels)])
    if args.allow_mid_band:
        parts.append("--allow-mid-band")
    return " ".join(parts)


def run(argv):
    args = parse_args(argv)
    validate(args)

    numpy_enabled = not args.no_numpy and os.environ.get("LM_COLLAPSE_NUMPY", "1") != "0"
    numpy_module = load_numpy(numpy_enabled)

    connection = connect(args.database, read_only=not args.apply)
    try:
        report = preflight(connection, args.min_coverage)
        if args.apply and not report.has_connections:
            raise Refusal(
                "table 'connections' is absent, so supersedes edges cannot be "
                "written to this database"
            )
        print_preflight(report, args, numpy_module)

        nodes = load_nodes(connection, report.node_columns)
        protected = load_provenance_sources(
            connection, report.node_columns, args.provenance_levels
        )
        superseded, superseding, existing_pairs = load_supersedes(
            connection, report.has_connections
        )

        started = time.time()
        load = load_mean_pooled_vectors(
            connection,
            set(nodes),
            report.width,
            "dimensions" in report.chunk_columns,
        )
        if numpy_module is None:
            pairs_ahead = 0
            for members in _group_indices(load.ids, nodes, args.scope_mode).values():
                pairs_ahead += len(members) * (len(members) - 1) // 2
            emit(
                "numpy is absent: scanning %d pairs in pure Python "
                "(~%.0f s at 4.6 us/pair). This is slow, not broken."
                % (pairs_ahead, pairs_ahead * 4.6e-6)
            )
            emit()
        scan_result = scan(
            load.ids, load.vectors, nodes, args.threshold, args.scope_mode, numpy_module
        )
        elapsed = time.time() - started

        # Order is the contract: distribution first, threshold after it.
        print_distribution(scan_result, load, args)
        emit("scan took %.1f s (%s)" % (elapsed, "numpy" if numpy_module else "pure Python"))
        emit()

        veto_on = identifier_veto_enabled()
        pair_members = set()
        if veto_on:
            for _, id_a, id_b in scan_result.pairs:
                pair_members.add(id_a)
                pair_members.add(id_b)
        veto = IdentifierVeto(
            load_contents(connection, pair_members) if veto_on else {},
            enabled=veto_on,
        )
        print_band_note(args, veto_on)

        collapses, skips = decide(
            scan_result.pairs,
            nodes,
            protected,
            superseded,
            superseding,
            existing_pairs,
            args.length_margin,
            veto,
        )
        emit(
            "provenance-protected nodes: %d of %d active (source_traces of live %s)"
            % (
                len(protected & set(nodes)),
                len(nodes),
                "/".join(args.provenance_levels),
            )
        )
        emit("existing supersedes edges : %d" % len(existing_pairs))
        emit()
        print_decisions(collapses, skips, scan_result, args)

        if args.json_path:
            write_json(args, report, load, scan_result, collapses, skips, veto_on)

        if args.apply:
            emit(OPERATIONAL_NOTE)
            emit()
            written, already = apply_collapses(connection, collapses, args.threshold)
            emit("## applied")
            emit("supersedes edges written : %d" % written)
            emit("already present          : %d" % already)
            emit("rows deleted             : 0 (this script never deletes)")
            emit("nodes marked decayed     : 0 (this script only writes edges)")
            emit()
            emit(OPERATIONAL_NOTE)
        else:
            emit("## dry run")
            emit("Nothing was written; the database was opened read-only.")
            emit("To apply exactly this pass:")
            emit(
                "    %s --apply"
                % build_command_line(args, os.path.abspath(args.database), veto_on)
            )
            emit()
            emit(OPERATIONAL_NOTE)
    finally:
        connection.close()
    return EXIT_OK


def write_json(args, report, load, scan_result, collapses, skips, identifier_veto=True):
    values = sorted(scan_result.max_cosine.values())
    by_reason = {}
    for item in skips:
        by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
    payload = {
        "database": os.path.abspath(args.database),
        "threshold": args.threshold,
        "scope_mode": args.scope_mode,
        "length_margin": args.length_margin,
        "identifier_veto": bool(identifier_veto),
        "provenance_levels": list(args.provenance_levels),
        "active_nodes": report.active_nodes,
        "nodes_with_chunks": report.nodes_with_chunks,
        "vector_width": report.width,
        "coverage": report.coverage,
        "nodes_compared": len(values),
        "pairs_compared": scan_result.compared_pairs,
        "percentiles": dict(
            ("p%d" % percentile, _percentile(values, percentile))
            for percentile in REPORT_PERCENTILES
        ),
        "counts_at_threshold": dict(
            (
                "%.3f" % threshold,
                {
                    "nodes": sum(1 for value in values if value >= threshold),
                    "pairs": scan_result.pair_counts[threshold],
                },
            )
            for threshold in REPORT_THRESHOLDS
        ),
        "pairs_at_threshold": len(scan_result.pairs),
        "collapses": [
            {
                "cosine": item.cosine,
                "keep": item.bearer.id,
                "supersede": item.candidate.id,
                "scope": item.candidate.scope,
                "level": item.candidate.level,
                "bearer_chars": item.bearer.chars,
                "candidate_chars": item.candidate.chars,
            }
            for item in collapses
        ],
        "skips": by_reason,
        "applied": bool(args.apply),
    }
    with open(args.json_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return run(argv)
    except Refusal as refusal:
        sys.stderr.write("%s: refusing: %s\n" % (PROGRAM, refusal))
        return EXIT_REFUSED
    except sqlite3.OperationalError as error:
        sys.stderr.write("%s: sqlite error: %s\n" % (PROGRAM, error))
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
