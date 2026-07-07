"""Provenance-derived typed-edge rules applied at write time.

Implements exactly the go-rules from artifacts/discovery/typed-edge-rules.md
section 5 — R1c (content-correction supersedes), R2a (root-cause caused
resolution), R4a (same commit), R4b (shared files), R5a (derived-from
annotation of existing schema edges), and R6 (content ULID reference) — with
the final parameters, weights, metadata, and direction semantics measured by
scripts/mine_typed_edges.py on the live corpus. The rejected rules (R1a, R1b,
R2b, R3, R5b) are deliberately not implemented.

Rule functions are pure: they take the new node plus explicitly passed
candidate rows and return :class:`DerivedEdge` specs, so the write path and
the corpus backfill CLI share one implementation with identical caps and
exclusions. :func:`derive_edges_for_new_trace` performs the bounded candidate
lookups (primary-key fetches for content ULIDs, an indexed commit-prefix
lookup, per-file citation lookups over the files-bearing subset, one recent
same-scope window scan) and persists results with the same upsert semantics
as ``consolidation._upsert_weighted_connection``: weight max-merge, metadata
merge. Edges to decayed nodes are never created; both-rejected-alternative
pairs are skipped for the undirected rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from living_memory.models import Connection, ConnectionType, Node
from living_memory.storage import _node_from_row

if TYPE_CHECKING:  # pragma: no cover - typing only
    from living_memory.storage import MemoryStore

# ULID node ids as they appear verbatim inside trace content (Crockford base32).
ULID_RE = re.compile(r"\b01[0-9A-HJKMNP-TV-Z]{24}\b")

# Adjacent correction/update phrasing shortly before a named node ULID
# (typed-edge-rules.md §R1c, sampled precision 9/9). A looser 80-char
# any-correction-word window admitted pairs whose trigger word referred to a
# third node; the adjacent form below is the measured final rule.
CORRECTION_NEAR_ULID_RE = re.compile(
    r"(?i)(?:\b(?:correction|update)s?(?:/\w+)?[-\s]+(?:to|of|for)\b|\b(?:corrects|supersedes)\b)"
    r"[^\n]{0,60}?\b(01[0-9A-HJKMNP-TV-Z]{24})\b"
)

# Context-label sets grounded in observed values on the live corpus
# (scripts/mine_typed_edges.py; do not extend without re-mining).
CORRECTION_CONTEXT_TYPES = frozenset(
    {"correction", "memory_correction", "decomposition_correction"}
)
CORRECTION_CONTENT_PREFIXES = ("Correction:", "CORRECTION:", "Correction ")
SOLUTION_TYPES = frozenset(
    {
        "failure_resolution",
        "working_fix",
        "fix",
        "gate_failure_resolution",
        "supervision_repair",
    }
)
SOLUTION_LESSON_KINDS = frozenset(
    {
        "tree_decomposition_repair",
        "decomposition_repair",
        "decomposition-repair",
        "parent-decomposition-repair",
        "working_fix",
    }
)
ROOT_CAUSE_TYPES = frozenset({"root_cause", "root_cause_and_debugging_insights"})
TIGHT_PROBLEM_OUTCOMES = frozenset({"structural_fail", "reopened"})

RESOLUTION_WINDOW_SECONDS = 3600
MAX_COMMIT_GROUP = 10
MAX_FILE_FANOUT = 20
COMMIT_PREFIX_LENGTH = 7
# Safety bound for the R2a same-scope recent-window fetch; one scope rarely
# accrues more than a handful of traces per hour.
R2A_WINDOW_FETCH_LIMIT = 500

# Edge metadata kinds/bases emitted by the rules. retrieval._traversal keys
# its kind-specific factors on these and consolidation reuses the R5a
# annotation, so they live here as the single source of truth.
CONTENT_CORRECTION_KIND = "content_correction"
FAILURE_RESOLUTION_KIND = "failure_resolution"
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


def is_correction_marked(node: Node) -> bool:
    ctx_type = str(node.context.get("type") or "")
    lesson_kind = str(node.context.get("lesson_kind") or "")
    return (
        ctx_type in CORRECTION_CONTEXT_TYPES
        or lesson_kind == "correction"
        or (node.content or "").startswith(CORRECTION_CONTENT_PREFIXES)
    )


def is_solution_labeled(node: Node) -> bool:
    ctx = node.context
    return (
        str(ctx.get("type") or "") in SOLUTION_TYPES
        or str(ctx.get("lesson_kind") or "") in SOLUTION_LESSON_KINDS
    )


def is_tight_problem(node: Node) -> bool:
    ctx = node.context
    return (
        str(ctx.get("type") or "") in ROOT_CAUSE_TYPES
        or str(ctx.get("outcome") or "") in TIGHT_PROBLEM_OUTCOMES
    )


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


def correction_ulid_matches(node: Node) -> list[tuple[str, str]]:
    """R1c targets as ``(ulid, matched_phrase)`` pairs, deduped, self excluded.

    Primary form: the adjacent correction-phrase regex. Fallback (only when
    the regex matches nothing): a correction-marked node claims every ULID in
    its content, with the marker recorded as the matched phrase.
    """

    content = node.content or ""
    matches = [
        (match.group(1), match.group(0))
        for match in CORRECTION_NEAR_ULID_RE.finditer(content)
    ]
    if not matches and is_correction_marked(node):
        marker = _correction_marker(node)
        matches = [(ulid, marker) for ulid in ULID_RE.findall(content)]
    seen: dict[str, str] = {}
    for ulid, phrase in matches:
        if ulid != node.id and ulid not in seen:
            seen[ulid] = phrase
    return list(seen.items())


def _correction_marker(node: Node) -> str:
    ctx_type = str(node.context.get("type") or "")
    if ctx_type in CORRECTION_CONTEXT_TYPES:
        return f"context.type={ctx_type}"
    if str(node.context.get("lesson_kind") or "") == "correction":
        return "context.lesson_kind=correction"
    return "content-prefix:Correction"


def node_time(node: Node) -> datetime | None:
    """Timestamp as an aware UTC datetime, tolerant of client formats."""

    value = str(node.timestamp or node.context.get("timestamp") or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# --- pure rule functions -------------------------------------------------------


def rule_r1c_content_correction(
    node: Node, candidates: Mapping[str, Node]
) -> list[DerivedEdge]:
    """R1c: adjacent correction phrase naming a node ULID ⇒ ``supersedes`` it.

    Direction correction → named node (matches memory_teach). ``candidates``
    maps content ULIDs to their fetched nodes; missing or decayed targets are
    skipped.
    """

    if node.decayed or node.level != "trace":
        return []
    edges: list[DerivedEdge] = []
    for ulid, phrase in correction_ulid_matches(node):
        target = candidates.get(ulid)
        if target is None or target.decayed or target.id == node.id:
            continue
        edges.append(
            DerivedEdge(
                source_id=node.id,
                target_id=target.id,
                type="supersedes",
                weight=0.9,
                metadata={
                    "rule": "R1c",
                    "kind": CONTENT_CORRECTION_KIND,
                    "matched": phrase,
                },
                rule="R1c",
            )
        )
    return edges


def rule_r6_content_reference(
    node: Node, candidates: Mapping[str, Node]
) -> list[DerivedEdge]:
    """R6: non-correction content ULID reference ⇒ directed ``related``.

    Referrer → referee stored as source → target; ULIDs claimed by R1c for
    this node are excluded.
    """

    if node.decayed or node.level != "trace":
        return []
    claimed = {ulid for ulid, _ in correction_ulid_matches(node)}
    edges: list[DerivedEdge] = []
    for ulid in dict.fromkeys(ULID_RE.findall(node.content or "")):
        if ulid == node.id or ulid in claimed:
            continue
        target = candidates.get(ulid)
        if target is None or target.decayed:
            continue
        edges.append(
            DerivedEdge(
                source_id=node.id,
                target_id=target.id,
                type="related",
                weight=0.9,
                metadata={"rule": "R6", "kind": CONTENT_REFERENCE_KIND},
                rule="R6",
            )
        )
    return edges


def rule_r2a_failure_resolution(
    node: Node,
    candidates_by_group: Mapping[str, Sequence[Node]],
    *,
    window_seconds: int = RESOLUTION_WINDOW_SECONDS,
) -> list[DerivedEdge]:
    """R2a: nearest root-cause trace ``caused`` the solution written after it.

    ``node`` is the candidate solution; ``candidates_by_group`` maps a group
    key (same scope + ``task``, then same scope + ``session_id``) to traces in
    that group. Per group the nearest problem 0..window_seconds earlier is
    linked; the same problem reached via several groups keeps the first
    smallest gap. Direction root_cause → fix.
    """

    if (
        node.decayed
        or node.level != "trace"
        or is_rejected_alternative(node)
        or not is_solution_labeled(node)
    ):
        return []
    sol_ts = node_time(node)
    if sol_ts is None:
        return []

    best: dict[str, tuple[float, str]] = {}
    for group_key, members in candidates_by_group.items():
        scored: list[tuple[float, Node]] = []
        for problem in members:
            if (
                problem.id == node.id
                or problem.decayed
                or problem.level != "trace"
                or is_rejected_alternative(problem)
                or is_solution_labeled(problem)
                or not is_tight_problem(problem)
            ):
                continue
            prob_ts = node_time(problem)
            if prob_ts is None:
                continue
            delta = (sol_ts - prob_ts).total_seconds()
            if 0 <= delta <= window_seconds:
                scored.append((delta, problem))
        if not scored:
            continue
        delta, problem = min(scored, key=lambda item: (item[0], item[1].id))
        current = best.get(problem.id)
        if current is None or delta < current[0]:
            best[problem.id] = (delta, group_key)

    return [
        DerivedEdge(
            source_id=problem_id,
            target_id=node.id,
            type="caused",
            weight=0.9,
            metadata={
                "rule": "R2a",
                "kind": FAILURE_RESOLUTION_KIND,
                "gap_seconds": int(round(delta)),
                "group_key": group_key,
            },
            rule="R2a",
        )
        for problem_id, (delta, group_key) in best.items()
    ]


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
                weight=0.9,
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
                weight=min(0.95, 0.5 + 0.15 * len(shared)),
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
    """Persist one derived edge with ``_upsert_weighted_connection`` semantics.

    Mirrors consolidation.py: same-direction lookup, weight max-merge,
    metadata merge, clamp to [0, 1] rounded to 6 decimals. Kept local because
    consolidation imports this module for the R5a annotation.
    """

    existing = store.list_connections(
        source_id=edge.source_id,
        target_id=edge.target_id,
        relation_type=edge.type,
    )
    weight = float(edge.weight)
    metadata = dict(edge.metadata)
    if existing:
        weight = max(weight, existing[0].weight)
        metadata = {**existing[0].metadata, **metadata}
    return store.create_connection(
        edge.source_id,
        edge.target_id,
        edge.type,
        weight=round(min(1.0, max(0.0, weight)), 6),
        metadata=metadata,
    )


# --- orchestrator: bounded candidate lookups + rules --------------------------


def derive_edges_for_new_trace(store: "MemoryStore", node: Node) -> list[Connection]:
    """Run every write-path go-rule for one newly written trace.

    All lookups are bounded: primary-key fetches for content ULIDs, an
    expression-indexed commit-prefix probe capped at the group limit,
    per-file citation probes capped at the fanout limit over the partial
    files-present index, and one recent same-scope window scan served by
    ``idx_nodes_trace_scope_time``. The recall query path is untouched.
    """

    if node is None or node.decayed or node.level != "trace":
        return []

    edges: list[DerivedEdge] = []

    ulids = [ulid for ulid in dict.fromkeys(ULID_RE.findall(node.content or "")) if ulid != node.id]
    if ulids:
        candidates = store.get_nodes(ulids)
        edges.extend(rule_r1c_content_correction(node, candidates))
        edges.extend(rule_r6_content_reference(node, candidates))

    if is_solution_labeled(node) and not is_rejected_alternative(node):
        groups = _recent_group_candidates(store, node)
        if groups:
            edges.extend(rule_r2a_failure_resolution(node, groups))

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


def _recent_group_candidates(store: "MemoryStore", node: Node) -> dict[str, list[Node]]:
    """Active same-scope traces in the R2a window, split into task/session groups."""

    sol_ts = node_time(node)
    if sol_ts is None:
        return {}
    task = str(node.context.get("task") or "")
    session = str(node.context.get("session_id") or "")
    if not task and not session:
        return {}

    lower = _iso_z(sol_ts - timedelta(seconds=RESOLUTION_WINDOW_SECONDS))
    upper = str(node.timestamp or _iso_z(sol_ts))
    rows = store.connection.execute(
        """
        SELECT * FROM nodes
        WHERE level = 'trace' AND decayed = 0 AND scope = ?
          AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (node.scope, lower, upper, R2A_WINDOW_FETCH_LIMIT),
    ).fetchall()
    candidates = [_node_from_row(row) for row in rows]

    groups: dict[str, list[Node]] = {}
    if task:
        members = [c for c in candidates if str(c.context.get("task") or "") == task]
        if members:
            groups[f"task={task}"] = members
    if session:
        members = [
            c for c in candidates if str(c.context.get("session_id") or "") == session
        ]
        if members:
            groups[f"session_id={session}"] = members
    return groups


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


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
