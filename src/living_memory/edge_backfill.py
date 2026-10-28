"""Idempotent, audited typed-edge backfill for the corpus.

The write-path engine (:mod:`living_memory.edge_derivation`) types edges for
*new* traces as they are written. This module applies the very same go-rules
(R4a / R4b / R6, and optionally the R5a ``derived_from``
annotation) to the *existing* corpus on a snapshot, so historical nodes gain
the same typed, directional edges.

Design guarantees (see ``derive-edges`` CLI below):

* **Existing semantics preserved** — a pair already connected with the same
  semantic type is skipped, keeping its weight/metadata/timestamps unchanged.
  A semantic contribution can replace a known technical row; these updates
  are reported separately from new rows and use the shared storage writer.
  They do not carry the new-row rollback marker. The
  optional R5a pass merges the ``{kind: derived_from, rule: R5a}`` annotation
  onto existing schema→trace ``related`` edges — additive metadata keys only,
  weight and type unchanged — and is off by default.
* **Idempotent** — every inserted row carries ``metadata.basis =
  "provenance_derivation"``; a second run finds all its pairs already
  connected and inserts zero. R5a re-runs are no-ops because
  :func:`edge_derivation.rule_r5a_derived_from_annotations` filters
  already-annotated edges.
* **Rollback-precise** — the marker ``basis = "provenance_derivation"`` is a
  namespace no rule or hand edge uses, so ``DELETE FROM connections WHERE
  json_extract(metadata,'$.basis')='provenance_derivation'`` removes exactly
  what a row-additive backfill added and nothing else. (The rules' own bases
  ``same_commit``/``shared_files`` are preserved under ``rule_basis``; the
  ``kind`` that ``retrieval._traversal`` reads is never touched.)

The engine reuses :mod:`edge_derivation`'s pure rule functions and serves the
volume rules (R4a same commit, R4b shared files) from two corpus-wide index
scans that reproduce the
write path's candidate WHERE-clauses (active nodes, any level, no fanout/commit
cap applied before the rule) — so the derived edge set is identical to running
the write path over every node, only faster.

CLI::

    python3 -m living_memory.edge_backfill derive-edges --db PATH \
        [--dry-run | --apply] [--report OUT.json] [--md OUT.md] \
        [--annotate-derived-from] [--sample-size N]

Dry-run is the default: it prints/writes the full audit and touches nothing.
Open the CLI against a *copy* of the live database — applying to the live
``global.sqlite3`` is an operator action, out of scope for automated runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from living_memory.config import MemoryConfig
from living_memory.edge_derivation import (
    DERIVED_FROM_ANNOTATION,
    ULID_RE,
    DerivedEdge,
    commit_prefix,
    node_files,
    rule_r4a_same_commit,
    rule_r4b_shared_files,
    rule_r5a_derived_from_annotations,
    rule_r6_content_reference,
)
from living_memory.models import Connection, Node
from living_memory.relations import is_semantic_relation, semantic_relation_sql
from living_memory.storage import (
    MemoryStore,
    _json_dumps,
    _node_from_row,
    _utc_now,
    new_ulid,
)

# Metadata marker stamped on every backfill-inserted row. Chosen so the exact
# rollback is one DELETE and never touches write-path or hand-authored edges
# (verified empty on the live corpus). retrieval._traversal ignores `basis`.
BACKFILL_BASIS = "provenance_derivation"

STRICT_TYPES: tuple[str, ...] = ("caused", "contradicts", "supersedes", "requires")
ALL_TYPES: tuple[str, ...] = ("related", *STRICT_TYPES)

# Provenance-derived membership (typed-edge-rules.md §4): strictly-typed edges
# plus related edges carrying a provenance kind/basis.
_PROVENANCE_KINDS: tuple[str, ...] = (
    "derived_from",
    "content_reference",
)
_PROVENANCE_BASES: tuple[str, ...] = ("same_commit", "shared_files", BACKFILL_BASIS)
_PROVENANCE_RULE_BASES: tuple[str, ...] = ("same_commit", "shared_files")

# Merge order == write-path application order (edge_derivation.derive_edges_for
# _new_trace): later rules win metadata conflicts, mirroring the upsert. Only
# R4a/R4b set `basis`; only R6 sets `kind`, so `kind` never collides.
RULE_ORDER: dict[str, int] = {"R6": 0, "R4a": 1, "R4b": 2}

DEFAULT_SAMPLE_SIZE = 5


@dataclass(frozen=True, slots=True)
class BackfillEdge:
    """One merged, ready-to-record derived relation."""

    source_id: str
    target_id: str
    type: str
    weight: float
    metadata: dict[str, Any]
    rules: tuple[str, ...]


@dataclass(slots=True)
class RuleAudit:
    """Per-rule participation counts and samples.

    ``net_new_pairs`` is the subset of ``new`` rows whose two nodes had *no*
    prior semantic edge of any type in either direction — the non-redundant
    connectivity gain the mining artifact §4 projects. ``new`` (exact
    ``(source, target, type)`` novelty, what actually gets inserted) is larger
    because same-type edges are also added on pairs already connected in the
    reverse direction or by another type, exactly as the write path does.
    Semantic relations replacing technical rows count as ``semantic_updates``.
    """

    type: str
    derived: int = 0
    new: int = 0
    net_new_pairs: int = 0
    existing_skipped: int = 0
    semantic_updates: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "derived": self.derived,
            "new": self.new,
            "net_new_pairs": self.net_new_pairs,
            "existing_skipped": self.existing_skipped,
            "semantic_updates": self.semantic_updates,
            "samples": self.samples,
        }


# --- corpus loading and candidate indexes -------------------------------------


def _load_active_nodes(store: MemoryStore) -> dict[str, Node]:
    """All active nodes (any level) keyed by id — the candidate universe."""

    rows = store.connection.execute("SELECT * FROM nodes WHERE decayed = 0").fetchall()
    return {str(row["id"]): _node_from_row(row) for row in rows}


def _commit_prefix_groups(nodes: Iterable[Node]) -> dict[str, list[Node]]:
    """prefix -> active nodes sharing it (mirrors _commit_group_candidates pool)."""

    groups: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        prefix = commit_prefix(node)
        if prefix is not None:
            groups[prefix].append(node)
    return groups


def _file_postings(
    store: MemoryStore, active_nodes: Mapping[str, Node]
) -> dict[str, list[Node]]:
    """file entry -> citing active nodes, one entry per occurrence.

    Single ``json_each`` scan reproducing ``_file_citation_candidates``'s
    per-occurrence fanout (a node listing a file twice is posted twice), over
    the same active files-present subset the write path probes.
    """

    postings: dict[str, list[Node]] = defaultdict(list)
    rows = store.connection.execute(
        """
        SELECT n.id AS nid, entry.value AS fname
        FROM nodes AS n, json_each(n.context, '$.files') AS entry
        WHERE json_extract(n.context, '$.files') IS NOT NULL
          AND n.decayed = 0
          AND json_type(n.context, '$.files') = 'array'
        """
    ).fetchall()
    for row in rows:
        node = active_nodes.get(str(row["nid"]))
        if node is not None:
            postings[str(row["fname"])].append(node)
    return postings


# --- derivation (write-path parity, persistence stripped) ---------------------


def _derived_edges_for_trace(
    store: MemoryStore,
    node: Node,
    prefix_groups: Mapping[str, list[Node]],
    file_postings: Mapping[str, list[Node]],
) -> list[DerivedEdge]:
    """Every go-rule for one active trace, identical to the write path.

    Mirrors ``edge_derivation.derive_edges_for_new_trace`` exactly, except R4a
    and R4b are served from the corpus-wide indexes instead of per-node probes.
    """

    edges: list[DerivedEdge] = []

    ulids = [
        ulid
        for ulid in dict.fromkeys(ULID_RE.findall(node.content or ""))
        if ulid != node.id
    ]
    if ulids:
        candidates = store.get_nodes(ulids)
        edges.extend(rule_r6_content_reference(node, candidates))

    prefix = commit_prefix(node)
    if prefix is not None:
        others = [other for other in prefix_groups.get(prefix, ()) if other.id != node.id]
        if others:
            edges.extend(rule_r4a_same_commit(node, others))

    files = node_files(node)
    if files:
        citing = {fname: file_postings.get(fname, []) for fname in files}
        edges.extend(rule_r4b_shared_files(node, citing))

    return edges


def _merge_derived(derived: Iterable[DerivedEdge]) -> dict[tuple[str, str, str], BackfillEdge]:
    """Collapse duplicate (source, target, type) emissions into one edge.

    Undirected rules emit each pair from both endpoints; a pair can also be
    derived by several rules. Metadata is merged in write-path order (later
    rules win — matching ``_upsert_weighted_connection``); weight is the max;
    the backfill marker is stamped last.
    """

    groups: dict[tuple[str, str, str], list[DerivedEdge]] = defaultdict(list)
    for edge in derived:
        groups[(edge.source_id, edge.target_id, edge.type)].append(edge)

    merged: dict[tuple[str, str, str], BackfillEdge] = {}
    for key, items in groups.items():
        items.sort(key=lambda edge: RULE_ORDER.get(edge.rule, 99))
        metadata: dict[str, Any] = {}
        weight = 0.0
        rules: list[str] = []
        for edge in items:
            metadata.update(edge.metadata)
            weight = max(weight, edge.weight)
            if edge.rule not in rules:
                rules.append(edge.rule)
        merged[key] = BackfillEdge(
            source_id=key[0],
            target_id=key[1],
            type=key[2],
            weight=round(min(1.0, max(0.0, weight)), 6),
            metadata=_stamp_backfill_metadata(metadata),
            rules=tuple(rules),
        )
    return merged


def _stamp_backfill_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Add the rollback marker, preserving any rule-supplied basis."""

    stamped = dict(metadata)
    existing_basis = stamped.get("basis")
    if existing_basis is not None and existing_basis != BACKFILL_BASIS:
        stamped["rule_basis"] = existing_basis
    stamped["basis"] = BACKFILL_BASIS
    return stamped


# --- existing edges, R5a, persistence -----------------------------------------


def _existing_edges(
    store: MemoryStore,
) -> tuple[set[tuple[str, str, str]], set[frozenset[str]], set[tuple[str, str, str]]]:
    """Return stored keys, semantic undirected pairs and technical keys.

    The exact keys drive additive/idempotent insertion; the undirected pairs
    (any semantic type, either direction) drive ``net_new_pairs`` connectivity.
    """

    rows = store.connection.execute(
        "SELECT source_id, target_id, type, metadata FROM connections"
    ).fetchall()
    keys: set[tuple[str, str, str]] = set()
    pairs: set[frozenset[str]] = set()
    technical: set[tuple[str, str, str]] = set()
    for row in rows:
        source, target = str(row["source_id"]), str(row["target_id"])
        relation_type = str(row["type"])
        key = (source, target, relation_type)
        keys.add(key)
        if is_semantic_relation(relation_type, json.loads(row["metadata"])):
            pairs.add(frozenset((source, target)))
        else:
            technical.add(key)
    return keys, pairs, technical


def _r5a_eligible(
    store: MemoryStore, active_nodes: Mapping[str, Node]
) -> list[Connection]:
    """Existing schema→source-trace ``related`` edges lacking the annotation."""

    eligible: list[Connection] = []
    for node in active_nodes.values():
        if node.level != "schema":
            continue
        connections = store.list_connections(source_id=node.id)
        eligible.extend(rule_r5a_derived_from_annotations(node, connections))
    return eligible


def _insert_edge(store: MemoryStore, edge: BackfillEdge) -> None:
    """Insert one new edge row; ON CONFLICT DO NOTHING never reweights."""

    now = _utc_now()
    store.connection.execute(
        """
        INSERT INTO connections (
            id, source_id, target_id, type, weight, metadata, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, target_id, type) DO NOTHING
        """,
        (
            new_ulid(),
            edge.source_id,
            edge.target_id,
            edge.type,
            float(edge.weight),
            _json_dumps(edge.metadata),
            now,
            now,
        ),
    )


# --- share / count helpers ----------------------------------------------------


def _type_counts(store: MemoryStore) -> dict[str, int]:
    counts = {
        edge_type: int(
            store.connection.execute(
                f"SELECT COUNT(*) FROM connections WHERE type = ? AND {semantic_relation_sql()}",
                (edge_type,)
            ).fetchone()[0]
        )
        for edge_type in ALL_TYPES
    }
    counts["total"] = sum(counts.values())
    return counts


def _share_counts(store: MemoryStore) -> dict[str, Any]:
    total = int(store.connection.execute(
        f"SELECT COUNT(*) FROM connections WHERE {semantic_relation_sql()}"
    ).fetchone()[0])
    strict = int(
        store.connection.execute(
            f"SELECT COUNT(*) FROM connections WHERE type IN ({_placeholders(STRICT_TYPES)}) "
            f"AND {semantic_relation_sql()}",
            STRICT_TYPES,
        ).fetchone()[0]
    )
    provenance = int(
        store.connection.execute(
            f"""
            SELECT COUNT(*) FROM connections
            WHERE {semantic_relation_sql()} AND (type IN ({_placeholders(STRICT_TYPES)})
               OR json_extract(metadata, '$.kind') IN ({_placeholders(_PROVENANCE_KINDS)})
               OR json_extract(metadata, '$.basis') IN ({_placeholders(_PROVENANCE_BASES)})
               OR json_extract(metadata, '$.rule_basis') IN ({_placeholders(_PROVENANCE_RULE_BASES)}))
            """,
            (*STRICT_TYPES, *_PROVENANCE_KINDS, *_PROVENANCE_BASES, *_PROVENANCE_RULE_BASES),
        ).fetchone()[0]
    )
    return _share_block(total, strict, provenance)


def _share_block(total: int, strict: int, provenance: int) -> dict[str, Any]:
    return {
        "total_edges": total,
        "strict_typed": strict,
        "strict_typed_share": _ratio(strict, total),
        "provenance_derived": provenance,
        "provenance_derived_share": _ratio(provenance, total),
    }


def _placeholders(values: Sequence[Any]) -> str:
    return ",".join("?" for _ in values)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


# --- orchestration ------------------------------------------------------------


def backfill_edges(
    store: MemoryStore,
    *,
    apply: bool = False,
    annotate_derived_from: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Plan (and optionally apply) the corpus backfill, returning the audit.

    Read-only until ``apply`` is set; writes new edges, semantic
    contributions and optional R5a annotations inside one transaction.
    """

    before_types = _type_counts(store)
    before_share = _share_counts(store)

    active_nodes = _load_active_nodes(store)
    prefix_groups = _commit_prefix_groups(active_nodes.values())
    file_postings = _file_postings(store, active_nodes)

    active_traces = sorted(
        (node for node in active_nodes.values() if node.level == "trace"),
        key=lambda node: node.id,
    )
    derived: list[DerivedEdge] = []
    for node in active_traces:
        derived.extend(_derived_edges_for_trace(store, node, prefix_groups, file_postings))

    merged = _merge_derived(derived)
    existing_keys, existing_pairs, technical_keys = _existing_edges(store)

    rule_audits: dict[str, RuleAudit] = {}
    to_insert: list[BackfillEdge] = []
    to_replace: list[BackfillEdge] = []
    net_new_rows = 0
    for key in sorted(merged):
        edge = merged[key]
        is_new = key not in existing_keys
        semantic_update = key in technical_keys and is_semantic_relation(edge.type, edge.metadata)
        if semantic_update:
            metadata = {key: value for key, value in edge.metadata.items()
                        if key not in ("basis", "rule_basis")}
            if "rule_basis" in edge.metadata:
                metadata["basis"] = edge.metadata["rule_basis"]
            edge = replace(edge, metadata=metadata)
        net_new = is_new and frozenset((edge.source_id, edge.target_id)) not in existing_pairs
        if net_new:
            net_new_rows += 1
        for rule in edge.rules:
            audit = rule_audits.setdefault(rule, RuleAudit(type=edge.type))
            audit.derived += 1
            if is_new:
                audit.new += 1
                if net_new:
                    audit.net_new_pairs += 1
                if len(audit.samples) < sample_size:
                    audit.samples.append(_edge_sample(edge))
            elif semantic_update:
                audit.semantic_updates += 1
                if len(audit.samples) < sample_size:
                    audit.samples.append(_edge_sample(edge))
            else:
                audit.existing_skipped += 1
        if is_new:
            to_insert.append(edge)
        elif semantic_update:
            to_replace.append(edge)

    r5a_eligible = _r5a_eligible(store, active_nodes) if annotate_derived_from else []

    if apply:
        with store.connection:
            for edge in to_insert:
                _insert_edge(store, edge)
            for edge in to_replace:
                store._insert_connection(
                    edge.source_id, edge.target_id, edge.type,
                    weight=edge.weight, metadata=edge.metadata, merge=True,
                )
            for connection in r5a_eligible:
                store.update_connection(
                    connection.id,
                    metadata={**connection.metadata, **DERIVED_FROM_ANNOTATION},
                )
        after_types = _type_counts(store)
        after_share = _share_counts(store)
    else:
        after_types = _project_types(before_types, [*to_insert, *to_replace])
        after_share = _project_share(before_share, [*to_insert, *to_replace], len(r5a_eligible))

    return _build_report(
        apply=apply,
        annotate_derived_from=annotate_derived_from,
        sample_size=sample_size,
        db_path=str(store.db_path),
        before_types=before_types,
        after_types=after_types,
        before_share=before_share,
        after_share=after_share,
        to_insert=to_insert,
        to_replace=to_replace,
        net_new_rows=net_new_rows,
        rule_audits=rule_audits,
        r5a_eligible=r5a_eligible,
    )


def _edge_sample(edge: BackfillEdge) -> dict[str, Any]:
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "type": edge.type,
        "weight": edge.weight,
        "rules": list(edge.rules),
        "metadata": edge.metadata,
    }


def _project_types(before: Mapping[str, int], to_insert: Sequence[BackfillEdge]) -> dict[str, int]:
    projected = dict(before)
    for edge in to_insert:
        projected[edge.type] = projected.get(edge.type, 0) + 1
    projected["total"] = sum(projected[t] for t in ALL_TYPES)
    return projected


def _project_share(
    before: Mapping[str, Any], to_insert: Sequence[BackfillEdge], r5a_count: int
) -> dict[str, Any]:
    strict_new = sum(1 for edge in to_insert if edge.type in STRICT_TYPES)
    # Every inserted row carries a provenance marker; each R5a annotation flips
    # an existing procedural edge (no prior provenance kind/basis) into one.
    total = before["total_edges"] + len(to_insert)
    strict = before["strict_typed"] + strict_new
    provenance = before["provenance_derived"] + len(to_insert) + r5a_count
    block = _share_block(total, strict, provenance)
    block["projected"] = True
    return block


def _build_report(
    *,
    apply: bool,
    annotate_derived_from: bool,
    sample_size: int,
    db_path: str,
    before_types: Mapping[str, int],
    after_types: Mapping[str, int],
    before_share: Mapping[str, Any],
    after_share: Mapping[str, Any],
    to_insert: Sequence[BackfillEdge],
    to_replace: Sequence[BackfillEdge],
    net_new_rows: int,
    rule_audits: Mapping[str, RuleAudit],
    r5a_eligible: Sequence[Connection],
) -> dict[str, Any]:
    new_by_type: dict[str, int] = defaultdict(int)
    multi_rule_rows = 0
    for edge in to_insert:
        new_by_type[edge.type] += 1
        if len(edge.rules) > 1:
            multi_rule_rows += 1

    rules_report = {rule: rule_audits[rule].to_dict() for rule in sorted(rule_audits)}
    if annotate_derived_from:
        rules_report["R5a"] = {
            "type": "related/derived_from (annotation, no new rows)",
            "eligible": len(r5a_eligible),
            "applied": bool(apply),
            "samples": [
                {
                    "connection_id": connection.id,
                    "source_id": connection.source_id,
                    "target_id": connection.target_id,
                    "annotation": dict(DERIVED_FROM_ANNOTATION),
                }
                for connection in r5a_eligible[:sample_size]
            ],
        }

    return {
        "command": "derive-edges",
        "mode": "apply" if apply else "dry-run",
        "applied": bool(apply),
        "db_path": db_path,
        "generated_at": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "annotate_derived_from": bool(annotate_derived_from),
        "sample_size": sample_size,
        "rollback": (
            "DELETE FROM connections "
            f"WHERE json_extract(metadata,'$.basis')='{BACKFILL_BASIS}';"
        ),
        "guarantees": {
            "existing_semantics_preserved": "semantic evidence replaces only known technical relations",
            "idempotent": "second run inserts 0 new rows (pairs already connected)",
            "marker": {"metadata.basis": BACKFILL_BASIS},
            "rollback_scope": "new rows only; semantic updates and optional R5a annotations are retained",
        },
        "totals": {
            "new_rows": len(to_insert),
            "semantic_updates": len(to_replace),
            "new_rows_by_type": {t: new_by_type[t] for t in ALL_TYPES if new_by_type[t]},
            "net_new_connectivity": net_new_rows,
            "multi_rule_rows": multi_rule_rows,
            "r5a_annotations": len(r5a_eligible),
        },
        "edges_by_type": {"before": dict(before_types), "after": dict(after_types)},
        "typed_share": {
            "before": dict(before_share),
            "after": dict(after_share),
        },
        "rules": rules_report,
    }


# --- markdown -----------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Typed-edge backfill audit")
    lines.append("")
    lines.append(
        f"Generated: {report['generated_at']} | mode: `{report['mode']}` | "
        f"DB: `{report['db_path']}`"
    )
    lines.append("")
    lines.append("## Guarantees")
    lines.append("")
    lines.append("- **Existing semantic edges preserved**: only known technical relations can be replaced with semantic evidence.")
    lines.append(
        "- **Idempotent**: a second run inserts 0 new rows (all pairs already connected)."
    )
    lines.append(
        f"- **Rollback**: `{report['rollback']}` (marker `metadata.basis = "
        f"\"{BACKFILL_BASIS}\"`, a namespace no other edge uses)."
    )
    lines.append("  This removes new rows only; semantic updates and optional R5a annotations remain.")
    lines.append("")
    totals = report["totals"]
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- New edge rows inserted: **{totals['new_rows']}**")
    lines.append(f"- Semantic evidence replacing technical relations: **{totals['semantic_updates']}**")
    by_type = ", ".join(f"{t}={n}" for t, n in totals["new_rows_by_type"].items()) or "none"
    lines.append(f"- New rows by type: {by_type}")
    lines.append(
        f"- Net-new connectivity (pairs with no prior semantic edge, any type/direction): "
        f"**{totals['net_new_connectivity']}** — the non-redundant graph gain (§4)."
    )
    lines.append(f"- Rows derived by more than one rule (collapsed): {totals['multi_rule_rows']}")
    lines.append(
        f"- R5a `derived_from` annotations on existing schema→trace edges: "
        f"{totals['r5a_annotations']}"
        + ("" if report["annotate_derived_from"] else " (disabled; --annotate-derived-from)")
    )
    lines.append("")
    lines.append("## Semantic edge type counts")
    lines.append("")
    lines.append("| type | before | after |")
    lines.append("|---|---|---|")
    before_types = report["edges_by_type"]["before"]
    after_types = report["edges_by_type"]["after"]
    for edge_type in (*ALL_TYPES, "total"):
        lines.append(
            f"| {edge_type} | {before_types.get(edge_type, 0)} | {after_types.get(edge_type, 0)} |"
        )
    lines.append("")
    share = report["typed_share"]
    before_share = share["before"]
    after_share = share["after"]
    lines.append("## Semantic typed / provenance share")
    lines.append("")
    lines.append("| metric | before | after |")
    lines.append("|---|---|---|")
    lines.append(
        f"| strict-typed | {before_share['strict_typed']} "
        f"({before_share['strict_typed_share']:.4f}) | {after_share['strict_typed']} "
        f"({after_share['strict_typed_share']:.4f}) |"
    )
    lines.append(
        f"| provenance-derived | {before_share['provenance_derived']} "
        f"({before_share['provenance_derived_share']:.4f}) | "
        f"{after_share['provenance_derived']} "
        f"({after_share['provenance_derived_share']:.4f}) |"
    )
    lines.append("")
    lines.append("## Per-rule counts")
    lines.append("")
    lines.append("| rule | type | derived | new rows | semantic updates | net-new pairs | existing skipped |")
    lines.append("|---|---|---|---|---|---|---|")
    for rule, data in report["rules"].items():
        if rule == "R5a":
            lines.append(
                f"| R5a | {data['type']} | {data['eligible']} | "
                f"{data['eligible'] if data['applied'] else 0} | - | - | - |"
            )
            continue
        lines.append(
            f"| {rule} | {data['type']} | {data['derived']} | {data['new']} "
            f"| {data['semantic_updates']} | {data['net_new_pairs']} | {data['existing_skipped']} |"
        )
    lines.append("")
    lines.append(
        "> Per-rule `derived` counts a pair under each contributing rule; the "
        "physical row total is `totals.new_rows` (collapses in "
        "`totals.multi_rule_rows`). `net-new pairs` (pairs with no prior edge "
        "of any type in either direction) is the §4 non-redundancy projection: "
        "`new rows` is larger because exact `(source,target,type)` novelty also "
        "types pairs already linked in reverse/another type, as the write path does."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------


def run_backfill(
    db_path: str | Path,
    *,
    apply: bool = False,
    annotate_derived_from: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Open the store at ``db_path`` and run the backfill (default dry-run)."""

    config = MemoryConfig(db_path=Path(db_path))
    with MemoryStore(config) as store:
        return backfill_edges(
            store,
            apply=apply,
            annotate_derived_from=annotate_derived_from,
            sample_size=sample_size,
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "derive-edges":
        parser.error("missing command")
        return 2

    report = run_backfill(
        args.db or args.sqlite_file,
        apply=args.apply,
        annotate_derived_from=args.annotate_derived_from,
        sample_size=args.sample_size,
    )

    payload = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
        md_path = Path(args.md) if args.md else report_path.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")
    else:
        print(payload)

    print(
        f"[{report['mode']}] {report['totals']['new_rows']} new edges "
        f"({report['totals']['semantic_updates']} semantic updates, "
        f"{report['totals']['r5a_annotations']} R5a annotations) on {report['db_path']}",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python3 -m living_memory.edge_backfill",
        description="Additive-only, idempotent, audited typed-edge corpus backfill.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    derive = subcommands.add_parser(
        "derive-edges",
        help="Derive typed edges for the existing corpus on a snapshot copy.",
    )
    derive.add_argument(
        "sqlite_file",
        nargs="?",
        help="SQLite database file (a snapshot COPY, not the live DB).",
    )
    derive.add_argument("--db", dest="db", help="SQLite database file. Overrides the positional path.")
    derive.add_argument("--report", help="Write the JSON audit to this path.")
    derive.add_argument("--md", help="Write the markdown audit here (default: report with .md).")
    derive.add_argument(
        "--annotate-derived-from",
        action="store_true",
        help="Also merge the R5a derived_from annotation onto existing schema→trace edges.",
    )
    derive.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Per-rule sample edges recorded in the audit (default: 5).",
    )
    mode = derive.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit without writing (the default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Insert the derived edges into the (copy) database.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main(sys.argv[1:]))
