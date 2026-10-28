"""Provenance-derived typed-edge rules applied at write time.

Derives shared-commit/file, schema-source and content reference relations.
Text, context labels and chronology never establish causality or supersession;
these require explicit typed connections or memory_teach corrections.

Rule functions are pure: they take the new node plus explicitly passed
candidate rows and return :class:`DerivedEdge` specs, so the write path and
the corpus backfill CLI share one implementation with identical caps and
exclusions. :func:`derive_edges_for_new_trace` performs the bounded candidate
lookups (primary-key fetches for content ULIDs, an indexed commit-prefix
lookup and per-file citation lookups over the files-bearing subset) and persists results with the same upsert semantics
as ``consolidation._upsert_weighted_connection``: weight max-merge, metadata
merge. Edges to decayed nodes are never created; both-rejected-alternative
pairs are skipped for the undirected rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from living_memory.models import Connection, ConnectionType, Node
from living_memory.relations import CONTENT_REFERENCE_WEIGHT, shared_files_weight
from living_memory.storage import _node_from_row

if TYPE_CHECKING:  # pragma: no cover - typing only
    from living_memory.storage import MemoryStore

# ULID node ids as they appear verbatim inside trace content (Crockford base32).
ULID_RE = re.compile(r"\b01[0-9A-HJKMNP-TV-Z]{24}\b")

MAX_COMMIT_GROUP = 10
MAX_FILE_FANOUT = 20
COMMIT_PREFIX_LENGTH = 7

# Edge metadata kinds/bases emitted by the rules. retrieval._traversal keys
# its kind-specific factors on these and consolidation reuses the R5a
# annotation, so they live here as the single source of truth.
DERIVED_FROM_KIND = "derived_from"
CONTENT_REFERENCE_KIND = "content_reference"
SAME_COMMIT_BASIS = "same_commit"
SHARED_FILES_BASIS = "shared_files"

# R5a is metadata-only: existing schema→source-trace edges gain this
# annotation (no new rows); concept edges are excluded (typed-edge-rules.md
# §R5a: concept clusters scored 0/16 as derived-from).
DERIVED_FROM_ANNOTATION: dict[str, str] = {"kind": DERIVED_FROM_KIND, "rule": "R5a"}


@dataclass(frozen=True, slots=True)
class DerivedEdge:
    """One provenance-derived edge emission, before upsert."""

    source_id: str
    target_id: str
    type: ConnectionType
    weight: float
    metadata: dict[str, Any]
    rule: str


# --- node predicates and extractors -------------------------------------------


def is_rejected_alternative(node: Node) -> bool:
    return bool(node.context.get("is_rejected_alternative"))


def commit_prefix(node: Node) -> str | None:
    """Normalized 7-char commit prefix, or None when too short/absent."""

    commit = str(node.context.get("commit") or "").strip().lower()
    if len(commit) < COMMIT_PREFIX_LENGTH:
        return None
    return commit[:COMMIT_PREFIX_LENGTH]


def node_files(node: Node) -> list[str]:
    """Distinct ``context.files`` entries in order; [] unless a real list."""

    files = node.context.get("files")
    if not isinstance(files, list):
        return []
    return list(dict.fromkeys(str(item) for item in files))


# --- pure rule functions -------------------------------------------------------


def rule_r6_content_reference(
    node: Node, candidates: Mapping[str, Node]
) -> list[DerivedEdge]:
    """R6: content ULID reference ⇒ directed ``related``.

    Referrer → referee stored as source → target, independent of wording.
    """

    if node.decayed or node.level != "trace":
        return []
    edges: list[DerivedEdge] = []
    for ulid in dict.fromkeys(ULID_RE.findall(node.content or "")):
        if ulid == node.id:
            continue
        target = candidates.get(ulid)
        if target is None or target.decayed:
            continue
        edges.append(
            DerivedEdge(
                source_id=node.id,
                target_id=target.id,
                type="related",
                weight=CONTENT_REFERENCE_WEIGHT,
                metadata={"rule": "R6", "kind": CONTENT_REFERENCE_KIND},
                rule="R6",
            )
        )
    return edges


def rule_r4a_same_commit(
    node: Node,
    others: Sequence[Node],
    *,
    max_group: int = MAX_COMMIT_GROUP,
) -> list[DerivedEdge]:
    """R4a: same 7-char commit prefix ⇒ undirected ``related``.

    ``others`` are the other active nodes sharing the prefix; groups larger
    than ``max_group`` (including ``node``) emit nothing, both-RA pairs are
    skipped, and pairs are stored in sorted-id canonical direction.
    """

    if node.decayed:
        return []
    prefix = commit_prefix(node)
    if prefix is None:
        return []
    members: list[Node] = []
    seen_ids: set[str] = set()
    for other in others:
        if other.id == node.id or other.id in seen_ids:
            continue
        seen_ids.add(other.id)
        if not other.decayed and commit_prefix(other) == prefix:
            members.append(other)
    if not members or len(members) + 1 > max_group:
        return []
    node_is_ra = is_rejected_alternative(node)
    edges: list[DerivedEdge] = []
    for other in members:
        if node_is_ra and is_rejected_alternative(other):
            continue
        source_id, target_id = sorted((node.id, other.id))
        edges.append(
            DerivedEdge(
                source_id=source_id,
                target_id=target_id,
                type="related",
                weight=CONTENT_REFERENCE_WEIGHT,
                metadata={
                    "rule": "R4a",
                    "basis": SAME_COMMIT_BASIS,
                    "commit": prefix,
                },
                rule="R4a",
            )
        )
    return edges


def rule_r4b_shared_files(
    node: Node,
    citing_by_file: Mapping[str, Sequence[Node]],
    *,
    max_fanout: int = MAX_FILE_FANOUT,
) -> list[DerivedEdge]:
    """R4b: shared ``context.files`` entry ⇒ undirected ``related``.

    ``citing_by_file`` maps each of the node's file entries to every citing
    occurrence row (one entry per occurrence, the new node's own occurrences
    included — the fanout cap counts occurrences exactly like the miner).
    Files cited more than ``max_fanout`` times are topic-dilute and skipped
    entirely. Weight grows with the number of shared files:
    ``min(0.95, 0.5 + 0.15 × n_shared)``.
    """

    if node.decayed:
        return []
    shared_by_other: dict[str, list[str]] = {}
    node_is_ra = is_rejected_alternative(node)
    for fname in node_files(node):
        rows = citing_by_file.get(fname) or ()
        if not 2 <= len(rows) <= max_fanout:
            continue
        seen_ids: set[str] = set()
        for other in rows:
            if other.id == node.id or other.id in seen_ids:
                continue
            seen_ids.add(other.id)
            if other.decayed:
                continue
            if node_is_ra and is_rejected_alternative(other):
                continue
            shared_by_other.setdefault(other.id, []).append(fname)

    edges: list[DerivedEdge] = []
    for other_id in sorted(shared_by_other):
        shared = shared_by_other[other_id]
        source_id, target_id = sorted((node.id, other_id))
        edges.append(
            DerivedEdge(
                source_id=source_id,
                target_id=target_id,
                type="related",
                weight=shared_files_weight(shared),
                metadata={
                    "rule": "R4b",
                    "basis": SHARED_FILES_BASIS,
                    "files": list(shared),
                },
                rule="R4b",
            )
        )
    return edges


def rule_r5a_derived_from_annotations(
    schema: Node, connections: Iterable[Connection]
) -> list[Connection]:
    """R5a: select existing schema→source-trace edges to annotate.

    Metadata-only — callers merge :data:`DERIVED_FROM_ANNOTATION` into the
    returned connections' metadata; no new rows are ever created. Concepts
    are excluded by requiring ``schema.level == "schema"``; already-annotated
    edges are filtered out so re-runs are no-ops.
    """

    if schema.decayed or schema.level != "schema":
        return []
    sources = set(schema.source_traces)
    eligible: list[Connection] = []
    for connection in connections:
        if (
            connection.source_id != schema.id
            or connection.type != "related"
            or connection.target_id not in sources
        ):
            continue
        if all(
            connection.metadata.get(key) == value
            for key, value in DERIVED_FROM_ANNOTATION.items()
        ):
            continue
        eligible.append(connection)
    return eligible


# --- persistence ---------------------------------------------------------------


def upsert_derived_edge(store: "MemoryStore", edge: DerivedEdge) -> Connection:
    """Apply this rule's contribution through the graph writer."""

    return store.create_connection(
        edge.source_id,
        edge.target_id,
        edge.type,
        weight=edge.weight,
        metadata=edge.metadata,
        merge=True,
    )


# --- orchestrator: bounded candidate lookups + rules --------------------------


def derive_edges_for_new_trace(store: "MemoryStore", node: Node) -> list[Connection]:
    """Run every write-path go-rule for one newly written trace.

    All lookups are bounded: primary-key fetches for content ULIDs, an
    expression-indexed commit-prefix probe capped at the group limit,
    per-file citation probes capped at the fanout limit over the partial
    files-present index. The recall query path is untouched.
    """

    if node is None or node.decayed or node.level != "trace":
        return []

    edges: list[DerivedEdge] = []

    ulids = [ulid for ulid in dict.fromkeys(ULID_RE.findall(node.content or "")) if ulid != node.id]
    if ulids:
        candidates = store.get_nodes(ulids)
        edges.extend(rule_r6_content_reference(node, candidates))

    prefix = commit_prefix(node)
    if prefix is not None:
        others = _commit_group_candidates(store, node, prefix)
        if others:
            edges.extend(rule_r4a_same_commit(node, others))

    files = node_files(node)
    if files:
        citing = _file_citation_candidates(store, files)
        edges.extend(rule_r4b_shared_files(node, citing))

    return [upsert_derived_edge(store, edge) for edge in edges]


def _commit_group_candidates(
    store: "MemoryStore", node: Node, prefix: str
) -> list[Node]:
    """Other active nodes sharing the commit prefix, capped at the group limit.

    The substr/lower/trim expression matches ``idx_nodes_commit_prefix``
    verbatim so the probe is index-served. Fetching ``MAX_COMMIT_GROUP``
    others is enough to detect an oversized group (node itself included makes
    it ``MAX_COMMIT_GROUP + 1``), which the rule then rejects.
    """

    rows = store.connection.execute(
        """
        SELECT * FROM nodes
        WHERE substr(lower(trim(json_extract(context, '$.commit'))), 1, 7) = ?
          AND json_extract(context, '$.commit') IS NOT NULL
          AND decayed = 0
          AND id != ?
        LIMIT ?
        """,
        (prefix, node.id, MAX_COMMIT_GROUP),
    ).fetchall()
    return [_node_from_row(row) for row in rows]


def _file_citation_candidates(
    store: "MemoryStore", files: Sequence[str]
) -> dict[str, list[Node]]:
    """Citing occurrence rows per file entry (the new node's own included).

    ``json_each`` yields one row per array element, so a node listing a file
    twice counts twice — identical to the miner's occurrence-based fanout.
    Fetching ``MAX_FILE_FANOUT + 1`` rows is enough for the rule to detect
    and skip an over-fanout file. String entries only, like the miner's
    ``isinstance(files, list)`` + ``str(f)`` postings.
    """

    citing: dict[str, list[Node]] = {}
    for fname in files:
        rows = store.connection.execute(
            """
            SELECT n.* FROM nodes AS n, json_each(n.context, '$.files') AS entry
            WHERE json_extract(n.context, '$.files') IS NOT NULL
              AND n.decayed = 0
              AND json_type(n.context, '$.files') = 'array'
              AND entry.value = ?
            LIMIT ?
            """,
            (fname, MAX_FILE_FANOUT + 1),
        ).fetchall()
        citing[fname] = [_node_from_row(row) for row in rows]
    return citing
