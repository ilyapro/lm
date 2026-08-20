#!/usr/bin/env python3
"""Field extraction for the recall-map polish: journals + persisted payloads.

What this reads, and why each source is needed
----------------------------------------------

Three sources, joined into one case per delivered map:

* **The ae inject journals** (``stream_inject/*.jsonl``) — what the *agent*
  actually saw. They carry the rendered inject line, the labels, the medoid
  node ids, and, crucially, the consumption evidence (``kind: consume``) that
  no server-side table holds.
* **The persisted ``recall_events.recall_map`` payloads** — what the *server*
  actually built. The journal renders a map; only the payload proves what the
  builder put in it, and the payload is the only place the missing medoid
  ``example`` can be confirmed rather than inferred from a renderer.
* **The corpus itself** — the medoid nodes, the query anchors and the FTS
  document-frequency index. Stage attribution is not guessable from a label:
  it is recomputed here with the *same* functions ``recall_map`` uses, so an
  attribution error would be a bug in this script rather than a judgement call.

Everything is read-only. The live databases are never opened by this script;
it takes frozen snapshots (``retrieval_harness.create_snapshot``) and opens a
scratch working copy of each, because ``MemoryStore`` migrates what it opens.

Stage attribution
-----------------

The cascade is precedence-ordered and per-node: a node with a structural key is
a structural member, never a path one (recall_map.py:917-931). The medoid is by
construction a member of its own cluster, so attributing the medoid attributes
the cluster. Each attribution also recomputes the label that stage *would* have
produced, so the result is falsifiable: a cluster whose observed label does not
match its attributed stage's construction shows up as ``label_match:
"mismatch"`` rather than being quietly filed.

Gist counterfactual
-------------------

For every delivered map the script rebuilds the exact payload from its own
fields and asks the only question that separates "the budget had no room" from
"the fit pass threw the room away": what is the largest per-cluster example
length that still fits :data:`MAX_RESPONSE_CHARS`? ``_payload_size`` is the
builder's own serializer, so the answer is the builder's own arithmetic.

Usage::

    python3 scripts/recall_map_field_extract.py \\
        --streams /tmp/fieldstream \\
        --field-snapshot ~/.cache/living-memory-field/recall-map-field-alt-20260820.sqlite3 \\
        --e2e-snapshot ~/.cache/living-memory-field/recall-map-e2e-local-20260820.sqlite3 \\
        --e2e-journals '/tmp/ae-stream-live-e2e.*/journal.jsonl' \\
        --out-manifest artifacts/recall-map/field/manifest.json \\
        --out-analysis artifacts/recall-map/field/analysis.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import glob as globlib
import json
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from living_memory.config import MemoryConfig  # noqa: E402
from living_memory.recall_map import (  # noqa: E402
    MAX_CLUSTERS,
    MAX_LABEL_CHARS,
    MAX_RESPONSE_CHARS,
    MEDOID_EXAMPLE_CHARS,
    STAGE_ANCHOR,
    STAGE_EMBEDDING,
    STAGE_PATH,
    STAGE_STRUCTURAL,
    MapCluster,
    MapMedoid,
    _collapse,
    _looks_unreadable,
    _node_subsystem,
    _payload_size,
    _phrase_from_content,
    _shorten,
    _structural_key,
    _terms,
    cache_key,
    normalize_key,
)
from living_memory.recall_map import RecallMapBuilder  # noqa: E402
from living_memory.retrieval import MemoryRecallService  # noqa: E402
from living_memory.retrieval_harness import load_manifest, working_copy  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402


# ----------------------------------------------------------------------
# Splits. Disjoint by cache key, because the builder caches cluster
# *structure* per (scope, normalized task): two cases under one key are two
# looks at one cached structure, and straddling a key across splits would
# leak the very thing the splits exist to hold out.
# ----------------------------------------------------------------------

SPLIT_BY_KEY: dict[str, str] = {
    "project:game|quality clips coverage registry refresh": "train",
    "project:game|quality clips coverage clips restore": "train",
    "project:game|quality meter honesty": "train",
    "project:game|quality footslip rest proc footlock": "train",
    "project:game|quality clips coverage idle rest clips": "eval",
    "project:game|quality clips coverage rest clips land": "eval",
    "project:game|quality footslip rest gait reach": "eval",
    "project:game|quality footslip rest": "holdout",
    "project:game|quality clips land": "holdout",
    "project:game|feeling control": "holdout",
}

#: The contrastive live-e2e key. Quoted in diagnosis.md, so it is train by
#: construction: a case the diagnosis reads cannot also be held out.
E2E_TASK_PREFIX = "stream-agents-thinking-map-live-e2e"


# ----------------------------------------------------------------------
# Journals
# ----------------------------------------------------------------------


def read_journals(paths: Iterable[Path], root: Path | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(paths):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_file"] = str(path.relative_to(root)) if root else str(path)
            record["_line"] = number
            events.append(record)
    return events


def journal_index(events: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Journal events grouped by session, in file order."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event.get("session_id", "")), []).append(event)
    return grouped


def inject_for_probe(
    session_events: Sequence[dict[str, Any]], probe: dict[str, Any]
) -> dict[str, Any] | None:
    """The inject a probe produced: the next inject in the same session.

    The channel is strictly serial per session — one probe in flight at a time,
    the inject written when its answer lands — so "next inject after this probe,
    before the next probe" identifies it without a shared correlation id.
    """

    seen = False
    for event in session_events:
        if event is probe:
            seen = True
            continue
        if not seen:
            continue
        if event["kind"] == "probe":
            return None
        if event["kind"] == "inject":
            return event
    return None


def consumes_after(
    session_events: Sequence[dict[str, Any]], inject: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        event
        for event in session_events
        if event["kind"] == "consume" and event.get("inject_seq") == inject.get("seq")
    ]


# ----------------------------------------------------------------------
# Stage attribution
# ----------------------------------------------------------------------


def attribute_stage(node: Any, store: MemoryStore) -> dict[str, Any]:
    """Which cascade stage owns this node, and the label that stage builds.

    Precedence is the cascade's own (recall_map.py:917-931). ``expected_label``
    is recomputed with the stage's own construction so the attribution can be
    checked against the label the field actually delivered.
    """

    key = _structural_key(node)
    if key is not None:
        field, raw = key
        unreadable = _looks_unreadable(raw)
        readable = _phrase_from_content(node.content) if unreadable else normalize_key(raw)
        return {
            "stage": STAGE_STRUCTURAL,
            "expected_label": _shorten(readable, MAX_LABEL_CHARS),
            "evidence": {"field": field, "raw": raw, "opaque_key": unreadable},
        }

    subsystem = _node_subsystem(node)
    if subsystem is not None:
        return {
            "stage": STAGE_PATH,
            "expected_label": _shorten(normalize_key(subsystem), MAX_LABEL_CHARS),
            "evidence": {"subsystem": subsystem},
        }

    best: tuple[float, str, Any] | None = None
    for edge in store.list_query_anchor_edges(target_id=node.id):
        anchor = store.get_query_anchor(edge.anchor_id)
        if anchor is None or anchor.decayed or not str(anchor.query).strip():
            continue
        weight = float(edge.weight)
        if best is None or (-weight, edge.anchor_id) < (-best[0], best[1]):
            best = (weight, edge.anchor_id, anchor)
    if best is not None:
        return {
            "stage": STAGE_ANCHOR,
            "expected_label": _shorten(_collapse(best[2].query), MAX_LABEL_CHARS),
            "evidence": {"anchor_id": best[1], "edge_weight": best[0]},
        }

    return {
        "stage": STAGE_EMBEDDING,
        "expected_label": None,
        "evidence": {"reason": "no structural key, no context path, no live anchor edge"},
    }


def classify_label_match(observed: str, expected: str | None) -> str:
    """How the delivered label relates to the one its stage constructs.

    ``_disambiguate`` may extend a colliding label with one extra term
    (recall_map.py:973-992), so an extended label is a match, not a mismatch.
    """

    if expected is None:
        return "unconstrained"
    if observed == expected:
        return "exact"
    base = observed.rsplit(" ", 1)[0].rstrip("… ")
    if base and (expected == base or expected.startswith(base)):
        return "disambiguated"
    if observed.endswith(("#2", "#3", "#4", "#5")):
        return "disambiguated"
    return "mismatch"


# ----------------------------------------------------------------------
# Label statistics: corpus-derived, never an enumerated blacklist
# ----------------------------------------------------------------------


def label_statistics(labels: Sequence[str], store: MemoryStore) -> dict[str, dict[str, Any]]:
    total = store.fts_document_count()
    terms: set[str] = set()
    for label in labels:
        terms.update(_terms(label))
    frequencies = store.term_document_frequencies(sorted(terms)) if terms else {}
    stats: dict[str, dict[str, Any]] = {}
    for label in set(labels):
        content_terms = _terms(label)
        shares = [
            (term, frequencies.get(term, 0), (frequencies.get(term, 0) / total) if total else 0.0)
            for term in content_terms
        ]
        stats[label] = {
            "tokens": len(label.split()),
            "content_terms": content_terms,
            "term_document_frequency": {term: df for term, df, _ in shares},
            "term_document_share": {term: round(share, 6) for term, _, share in shares},
            "max_document_share": round(max((s for _, _, s in shares), default=0.0), 6),
            "fts_document_count": total,
        }
    return stats


# ----------------------------------------------------------------------
# The gist counterfactual
# ----------------------------------------------------------------------


def rebuild_clusters(payload: dict[str, Any], contents: dict[str, str], chars: int) -> list[MapCluster]:
    """The payload's own clusters, re-cut with an example of ``chars``.

    Every field is taken from the persisted payload verbatim; only the example
    varies. So ``_payload_size`` over the result is the size the builder itself
    would have measured for that example budget.
    """

    clusters: list[MapCluster] = []
    for entry in payload.get("clusters", []):
        node_id = entry["medoid"]["node_id"]
        clusters.append(
            MapCluster(
                label=entry["label"],
                count=int(entry["count"]),
                medoid=MapMedoid(
                    node_id=node_id,
                    example=_shorten(contents.get(node_id, ""), chars),
                ),
                ask_hint=entry["ask_hint"],
                plan_item=entry["plan_item"],
                stage="",
                member_ids=(),
            )
        )
    return clusters


def _fit_example(size_at: Any) -> int:
    """Largest example length that still fits, by bisection.

    Linear from the cap downward would be 120 serializations; the size is
    monotonic in the example length, so a bisection over ``[0, 120]`` gets the
    same answer in seven.
    """

    low, high = 0, MEDOID_EXAMPLE_CHARS
    if size_at(high) <= MAX_RESPONSE_CHARS:
        return high
    if size_at(1) > MAX_RESPONSE_CHARS:
        return 0
    while low + 1 < high:
        middle = (low + high) // 2
        if size_at(middle) <= MAX_RESPONSE_CHARS:
            low = middle
        else:
            high = middle
    return low


def largest_fitting_example(payload: dict[str, Any], contents: dict[str, str]) -> dict[str, Any]:
    """What room the delivered map actually had for a gist, and at what breadth.

    Two questions, not one. First: with the clusters the field delivered, how
    long an example fits? Second — the one that decides whether the fix is a
    reflow or a breadth trade — how long an example fits if the map keeps only
    its first ``k`` clusters? ``_payload_size`` is the builder's own serializer,
    so both answers are the builder's own arithmetic rather than an estimate.
    """

    pool = int(payload.get("pool", 0))
    dropped = int(payload.get("more", 0))
    entries = payload.get("clusters", [])
    observed_size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def size_at(chars: int, keep: int | None = None) -> int:
        clusters = rebuild_clusters(payload, contents, chars)
        if keep is not None:
            extra = len(clusters) - keep
            clusters = clusters[:keep]
        else:
            extra = 0
        return _payload_size(clusters, pool, dropped + extra)

    by_breadth = {
        str(keep): _fit_example(lambda chars, keep=keep: size_at(chars, keep))
        for keep in range(1, len(entries) + 1)
    }
    best = _fit_example(size_at)
    return {
        "observed_payload_chars": observed_size,
        "budget_chars": MAX_RESPONSE_CHARS,
        "headroom_chars": MAX_RESPONSE_CHARS - observed_size,
        "largest_fitting_example_chars": best,
        "payload_chars_at_best": size_at(best),
        "payload_chars_at_cap": size_at(MEDOID_EXAMPLE_CHARS),
        "example_cap_chars": MEDOID_EXAMPLE_CHARS,
        "largest_fitting_example_by_cluster_count": by_breadth,
    }


# ----------------------------------------------------------------------
# Case assembly
# ----------------------------------------------------------------------


def load_events(connection: sqlite3.Connection, since: str) -> dict[str, list[sqlite3.Row]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM recall_events WHERE created_at >= ? ORDER BY created_at", (since,)
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["query"], []).append(row)
    return grouped


def head_ids(row: sqlite3.Row) -> list[str]:
    try:
        results = json.loads(row["results"])
    except (TypeError, ValueError):
        return []
    return [str(item.get("node_id")) for item in results if isinstance(item, dict)]


def build_case(
    row: sqlite3.Row,
    payload: dict[str, Any] | None,
    store: MemoryStore,
    *,
    source: str,
    probe: dict[str, Any] | None = None,
    inject: dict[str, Any] | None = None,
    consumes: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    key = cache_key(row["scope"], task=row["task"])
    clusters: list[dict[str, Any]] = []
    contents: dict[str, str] = {}
    if payload:
        for entry in payload.get("clusters", []):
            node_id = entry["medoid"]["node_id"]
            node = store.get_node(node_id)
            contents[node_id] = node.content if node is not None else ""
            attribution = (
                attribute_stage(node, store)
                if node is not None
                else {"stage": "unknown", "expected_label": None, "evidence": {"reason": "node absent from snapshot"}}
            )
            clusters.append(
                {
                    "label": entry["label"],
                    "count": int(entry["count"]),
                    "ask_hint": entry["ask_hint"],
                    "delivered_at": row["created_at"],
                    "medoid_node_id": node_id,
                    "medoid_present_in_snapshot": node is not None,
                    "medoid_content_chars": len(contents[node_id]),
                    "medoid_updated_at": getattr(node, "updated_at", None) if node is not None else None,
                    "example_chars": len(entry["medoid"].get("example", "")),
                    "stage": attribution["stage"],
                    "stage_expected_label": attribution["expected_label"],
                    "stage_evidence": attribution["evidence"],
                    "label_match": classify_label_match(entry["label"], attribution["expected_label"]),
                }
            )

    delivered = bool(payload and payload.get("clusters"))
    case: dict[str, Any] = {
        "case_id": row["id"],
        "source": source,
        "split": None,
        "cache_key": key,
        "scope": row["scope"],
        "requested_scope": row["requested_scope"],
        "task": row["task"],
        "task_pattern": (json.loads(row["ambient_context"] or "{}") or {}).get("task_pattern"),
        "agent": row["agent"],
        "query": row["query"],
        "depth": row["depth"],
        "max_results": int(row["max_results"]),
        "ambient_context": json.loads(row["ambient_context"] or "{}"),
        "created_at": row["created_at"],
        "persisted_payload": payload,
        "delivered": delivered,
        "curtailed": bool(payload and payload.get("curtailed")),
        "cluster_count": len(clusters),
        "clusters": clusters,
        "pool_size": int(payload.get("pool", 0)) if payload else 0,
        "covered": int(payload.get("covered", 0)) if payload else 0,
        "dropped": int(payload.get("more", 0)) if payload else 0,
        "head_node_ids": head_ids(row),
        "medoid_node_ids": [cluster["medoid_node_id"] for cluster in clusters],
        "budget": largest_fitting_example(payload, contents) if delivered else None,
        "journal": None,
    }
    if probe is not None:
        case["journal"] = {
            "file": probe["_file"],
            "session_id": probe.get("session_id"),
            "goal_path": probe.get("goal_path"),
            "probe_ts": probe.get("ts"),
            "probe_outcome": probe.get("outcome"),
            "probe_latency_ms": probe.get("latency_ms"),
            "inject_seq": inject.get("seq") if inject else None,
            "inject_content": inject.get("content") if inject else None,
            "inject_labels": inject.get("labels") if inject else None,
            "inject_node_ids": inject.get("node_ids") if inject else None,
            "consumed": [
                {
                    "ref": event.get("ref"),
                    "tool": event.get("tool"),
                    "distance": event.get("distance"),
                    "ts": event.get("ts"),
                }
                for event in consumes
            ],
        }
    return case


# ----------------------------------------------------------------------
# Replay: ground truth for stage attribution
# ----------------------------------------------------------------------


def _depth(raw: Any) -> Any:
    if raw is None:
        return 1
    try:
        return int(str(raw))
    except ValueError:
        return str(raw)


def describe_map(built: Any) -> dict[str, Any] | None:
    if built is None:
        return None
    return {
        "pool": built.pool_size,
        "covered": built.covered,
        "more": built.dropped,
        "curtailed": built.curtailed,
        "clusters": [
            {
                "label": cluster.label,
                "count": cluster.count,
                "stage": cluster.stage,
                "medoid_node_id": cluster.medoid.node_id,
                "example_chars": len(cluster.medoid.example),
            }
            for cluster in built.clusters
        ],
    }


def attach_replay(cases: list[dict[str, Any]], snapshot: Path) -> dict[str, Any]:
    """Re-run every case through the real recall and the real builder.

    Two builder arms over one recall, because the field maps were not all built
    the same way. ``cold`` is a fresh :class:`RecallMapBuilder` per case, i.e.
    the full four-stage cascade — that is the arm whose ``stage`` field is
    ground truth. ``warm`` shares one builder per cache key, in the order the
    field ran them, which is the arm that can reproduce a map served from the
    per-key *structure cache*. Whether a field map matches cold, warm or
    neither is itself a measurement, and it is recorded rather than assumed.
    """

    ordered = sorted(cases, key=lambda case: case["created_at"])
    summary = Counter()
    with working_copy(snapshot) as working:
        store = MemoryStore(MemoryConfig(db_path=str(working)))
        service = MemoryRecallService(store)
        warm_builders: dict[str, RecallMapBuilder] = {}
        for case in ordered:
            results = service.memory_recall(
                query=case["query"],
                scope=case["requested_scope"],
                depth=_depth(case["depth"]),
                max_results=case["max_results"],
                ambient_context=case["ambient_context"] or None,
                log_access=False,
                log_event=False,
            )
            residual = list(service.last_residual)
            head = [result.node.id for result in results]
            cold = RecallMapBuilder(store).build(
                residual,
                scope=case["scope"],
                task=case["task"],
                task_pattern=case["task_pattern"],
            )
            warm_builder = warm_builders.setdefault(case["cache_key"], RecallMapBuilder(store))
            warm = warm_builder.build(
                residual,
                scope=case["scope"],
                task=case["task"],
                task_pattern=case["task_pattern"],
            )
            # The residual is the whole ranked tail; the *pool* is what
            # RecallMapBuilder._pool makes of it — deduplicated and capped at
            # MAX_POOL_NODES. Only the pool is what a map was ever built from.
            members = RecallMapBuilder(store)._pool(residual)
            composition = Counter(
                attribute_stage(member.node, store)["stage"] for member in members
            )
            case["replay"] = {
                "head_node_ids": head,
                "head_reproduced": head == case["head_node_ids"],
                "pool_node_ids": [member.node.id for member in members],
                "pool_size": len(members),
                "residual_size": len(residual),
                "pool_stage_composition": dict(composition),
                "cold": describe_map(cold),
                "warm": describe_map(warm),
                "warm_cache_hit": warm_builder.last_cache_hit,
            }
            for stage, count in composition.items():
                summary[f"pool_{stage}"] += count
            summary["cases"] += 1
            summary["head_reproduced"] += int(head == case["head_node_ids"])
            observed = [cluster["label"] for cluster in case["clusters"]]
            for arm in ("cold", "warm"):
                built = case["replay"][arm]
                labels = [cluster["label"] for cluster in built["clusters"]] if built else []
                if observed and labels == observed:
                    summary[f"{arm}_labels_identical"] += 1
                if observed and set(labels) & set(observed):
                    summary[f"{arm}_labels_overlap"] += 1
        store.close()
    return dict(summary)


def recount_drift_probe(cases: Sequence[dict[str, Any]], snapshot: Path) -> dict[str, Any]:
    """Does a cache hit put a structurally-keyed node into a *path* cluster?

    ``_matches`` (recall_map.py:1502-1511) re-derives path membership from the
    subsystem alone: it never asks whether stage 1 would have claimed the node
    first, because in a fresh cascade stage 1 already had its chance. On a cache
    hit that precondition is gone — the cached template set only holds the
    structural keys the *cached* pool happened to show — so a node whose key is
    new to the key falls through to a path template and is renamed after a
    directory.

    This runs the mechanism directly on real field pools: build cold on one
    case's residual to seed the cache, then ``_recount`` the next case's
    residual through those templates and count the members that land in a path
    group while carrying a structural key of their own.
    """

    ordered = sorted(cases, key=lambda case: case["created_at"])
    by_key: dict[str, list[dict[str, Any]]] = {}
    for case in ordered:
        by_key.setdefault(case["cache_key"], []).append(case)

    pairs = 0
    drifted = 0
    drifted_pairs = 0
    examples: list[dict[str, Any]] = []
    with working_copy(snapshot) as working:
        store = MemoryStore(MemoryConfig(db_path=str(working)))
        service = MemoryRecallService(store)

        def residual_of(case: dict[str, Any]) -> list[Any]:
            service.memory_recall(
                query=case["query"],
                scope=case["requested_scope"],
                depth=_depth(case["depth"]),
                max_results=case["max_results"],
                ambient_context=case["ambient_context"] or None,
                log_access=False,
                log_event=False,
            )
            return list(service.last_residual)

        for key, group in by_key.items():
            if len(group) < 2:
                continue
            first, second = group[0], group[1]
            builder = RecallMapBuilder(store)
            if builder.build(
                residual_of(first),
                scope=first["scope"],
                task=first["task"],
                task_pattern=first["task_pattern"],
            ) is None:
                continue
            cached = builder._cache.get(
                cache_key(first["scope"], task=first["task"], task_pattern=first["task_pattern"])
            )
            if cached is None:
                continue
            members = builder._pool(residual_of(second))
            groups = builder._recount(cached.templates, members)
            if groups is None:
                continue
            pairs += 1
            local = 0
            for built in groups:
                if built.stage != STAGE_PATH:
                    continue
                for member in built.members:
                    if _structural_key(member.node) is None:
                        continue
                    local += 1
                    if len(examples) < 12:
                        field, raw = _structural_key(member.node)
                        examples.append(
                            {
                                "cache_key": key,
                                "node_id": member.node.id,
                                "path_label": built.label,
                                "structural_field": field,
                                "structural_key": raw,
                                "label_stage_1_would_have_built": _shorten(
                                    normalize_key(raw), MAX_LABEL_CHARS
                                ),
                            }
                        )
            drifted += local
            drifted_pairs += int(local > 0)
        store.close()
    return {
        "pairs_probed": pairs,
        "pairs_with_drift": drifted_pairs,
        "members_renamed_after_a_directory": drifted,
        "examples": examples,
    }


def reconcile_attribution(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the builder's own stage over the medoid-precedence inference.

    A replayed cluster carries the stage the cascade actually assigned it. When
    a replay arm reproduces a delivered label, that stage is the answer and the
    medoid inference is demoted to a cross-check — and the rate at which the two
    disagree is exactly the interesting number, because a disagreement means the
    delivered cluster held a node the fresh cascade would have claimed earlier.
    """

    counters = Counter()
    for case in cases:
        replay = case.get("replay") or {}
        by_label: dict[str, tuple[str, str]] = {}
        for arm in ("cold", "warm"):
            built = replay.get(arm)
            if not built:
                continue
            for cluster in built["clusters"]:
                by_label.setdefault(cluster["label"], (cluster["stage"], arm))
        for cluster in case["clusters"]:
            match = by_label.get(cluster["label"])
            if match is None:
                cluster["stage_source"] = "medoid_precedence"
                counters["medoid_precedence"] += 1
                continue
            stage, arm = match
            cluster["stage_source"] = f"replay_{arm}"
            counters[f"replay_{arm}"] += 1
            if stage != cluster["stage"]:
                counters["disagree"] += 1
                cluster["stage_medoid_inference"] = cluster["stage"]
                cluster["stage"] = stage
            else:
                counters["agree"] += 1
    return dict(counters)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def field_cases(streams: Path, snapshot: Path, since: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = read_journals(streams.glob("**/*.jsonl"), root=streams)
    sessions = journal_index(events)
    with working_copy(snapshot) as working:
        store = MemoryStore(MemoryConfig(db_path=str(working)))
        connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
        by_query = load_events(connection, since)
        cases: list[dict[str, Any]] = []
        unmatched: list[str] = []
        for event in events:
            if event["kind"] != "probe" or event.get("outcome") == "dedup":
                continue
            rows = by_query.get(event["query"])
            if not rows:
                unmatched.append(event["query"])
                continue
            row = min(rows, key=lambda candidate: abs_delta(candidate["created_at"], event["ts"]))
            payload = json.loads(row["recall_map"]) if row["recall_map"] else None
            session_events = sessions.get(str(event.get("session_id", "")), [])
            inject = inject_for_probe(session_events, event)
            cases.append(
                build_case(
                    row,
                    payload,
                    store,
                    source="field_alt",
                    probe=event,
                    inject=inject,
                    consumes=consumes_after(session_events, inject) if inject else (),
                )
            )
        journal_summary = {
            "files": sorted({event["_file"] for event in events}),
            "sessions": len(sessions),
            "events_by_kind": dict(Counter(event["kind"] for event in events)),
            "probe_outcomes": dict(
                Counter(event["outcome"] for event in events if event["kind"] == "probe")
            ),
            "ts_min": min(event["ts"] for event in events),
            "ts_max": max(event["ts"] for event in events),
            "probes_unmatched_in_store": len(unmatched),
            "consume_events": [
                {
                    "file": event["_file"],
                    "ts": event["ts"],
                    "goal_path": event.get("goal_path"),
                    "ref": event.get("ref"),
                    "tool": event.get("tool"),
                    "distance": event.get("distance"),
                    "inject_seq": event.get("inject_seq"),
                }
                for event in events
                if event["kind"] == "consume"
            ],
        }
        store.close()
        connection.close()
    return cases, journal_summary


def abs_delta(left: str, right: str) -> float:
    from datetime import datetime

    def parse(value: str) -> float:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

    return abs(parse(left) - parse(right))


def e2e_cases(snapshot: Path, journals: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = read_journals(journals)
    sessions = journal_index(events)
    by_inject_nodes: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for event in events:
        if event["kind"] == "inject":
            by_inject_nodes.setdefault(tuple(event.get("node_ids") or ()), []).append(event)

    with working_copy(snapshot) as working:
        store = MemoryStore(MemoryConfig(db_path=str(working)))
        connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM recall_events WHERE task LIKE ? AND recall_map IS NOT NULL ORDER BY created_at",
            (f"{E2E_TASK_PREFIX}%",),
        ).fetchall()
        cases = []
        for row in rows:
            payload = json.loads(row["recall_map"])
            node_ids = tuple(
                entry["medoid"]["node_id"] for entry in payload.get("clusters", [])
            )
            matches = by_inject_nodes.get(node_ids) or []
            inject = matches[0] if matches else None
            session_events = sessions.get(str(inject.get("session_id", "")), []) if inject else []
            case = build_case(row, payload, store, source="e2e_local")
            if inject is not None:
                case["journal"] = {
                    "file": inject["_file"],
                    "session_id": inject.get("session_id"),
                    "goal_path": inject.get("goal_path"),
                    "probe_ts": None,
                    "probe_outcome": None,
                    "probe_latency_ms": None,
                    "inject_seq": inject.get("seq"),
                    "inject_content": inject.get("content"),
                    "inject_labels": inject.get("labels"),
                    "inject_node_ids": inject.get("node_ids"),
                    "consumed": [
                        {
                            "ref": event.get("ref"),
                            "tool": event.get("tool"),
                            "distance": event.get("distance"),
                            "ts": event.get("ts"),
                        }
                        for event in consumes_after(session_events, inject)
                    ],
                }
            cases.append(case)
        summary = {
            "journals": sorted(str(path) for path in journals),
            "events_by_kind": dict(Counter(event["kind"] for event in events)),
            "consume_events": [
                {
                    "file": event["_file"],
                    "ts": event["ts"],
                    "ref": event.get("ref"),
                    "tool": event.get("tool"),
                    "distance": event.get("distance"),
                }
                for event in events
                if event["kind"] == "consume"
            ],
        }
        store.close()
        connection.close()
    return cases, summary


def assign_splits(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        if case["source"] == "e2e_local":
            case["split"] = "train"
            case["split_role"] = "contrastive_control"
            continue
        split = SPLIT_BY_KEY.get(case["cache_key"])
        if split is None:
            raise SystemExit(f"no split pinned for cache key {case['cache_key']!r}")
        case["split"] = split
        case["split_role"] = "field"


def analyse(cases: Sequence[dict[str, Any]], snapshot: Path) -> dict[str, Any]:
    delivered = [case for case in cases if case["delivered"]]
    clusters = [cluster for case in delivered for cluster in case["clusters"]]
    with working_copy(snapshot) as working:
        store = MemoryStore(MemoryConfig(db_path=str(working)))
        stats = label_statistics([cluster["label"] for cluster in clusters], store)
        store.close()
    drift = [
        cluster
        for cluster in clusters
        if cluster["stage"] == STAGE_PATH
        and cluster.get("stage_medoid_inference") == STAGE_STRUCTURAL
    ]
    return {
        "cases": len(cases),
        "delivered": len(delivered),
        "curtailed": sum(1 for case in cases if case["curtailed"]),
        "no_map": sum(1 for case in cases if case["persisted_payload"] is None),
        "clusters": len(clusters),
        "clusters_with_example": sum(1 for cluster in clusters if cluster["example_chars"] > 0),
        "stage_histogram": dict(Counter(cluster["stage"] for cluster in clusters)),
        "stage_source_histogram": dict(
            Counter(cluster.get("stage_source", "medoid_precedence") for cluster in clusters)
        ),
        "path_clusters_whose_medoid_carries_a_structural_key": len(drift),
        "drift_medoids_written_before_delivery": sum(
            1
            for cluster in drift
            if cluster["medoid_updated_at"] and cluster["medoid_updated_at"] <= cluster["delivered_at"]
        ),
        "cluster_count_histogram": dict(
            Counter(case["cluster_count"] for case in delivered)
        ),
        "budget_dropped_histogram": dict(
            Counter(MAX_CLUSTERS - case["cluster_count"] for case in delivered)
        ),
        "label_match_histogram": dict(Counter(cluster["label_match"] for cluster in clusters)),
        "label_frequency": dict(Counter(cluster["label"] for cluster in clusters).most_common()),
        "label_stage": {
            label: sorted({c["stage"] for c in clusters if c["label"] == label})
            for label in {cluster["label"] for cluster in clusters}
        },
        "label_statistics": stats,
        "payload_chars": sorted(case["budget"]["observed_payload_chars"] for case in delivered),
        "headroom_chars": sorted(case["budget"]["headroom_chars"] for case in delivered),
        "largest_fitting_example_chars": sorted(
            case["budget"]["largest_fitting_example_chars"] for case in delivered
        ),
        "maps_with_room_for_a_gist": sum(
            1 for case in delivered if case["budget"]["largest_fitting_example_chars"] > 0
        ),
        "maps_that_dropped_clusters": sum(1 for case in delivered if case["dropped"] > 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", required=True)
    parser.add_argument("--field-snapshot", required=True)
    parser.add_argument("--e2e-snapshot", required=True)
    parser.add_argument("--e2e-journals", required=True)
    parser.add_argument("--since", default="2026-08-20T04:00")
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-analysis", required=True)
    args = parser.parse_args(argv)

    field_snapshot = Path(args.field_snapshot).expanduser()
    e2e_snapshot = Path(args.e2e_snapshot).expanduser()

    cases, journal_summary = field_cases(Path(args.streams), field_snapshot, args.since)
    controls, e2e_summary = e2e_cases(
        e2e_snapshot, [Path(path) for path in sorted(globlib.glob(args.e2e_journals))]
    )
    everything = cases + controls
    assign_splits(everything)

    replay_field = attach_replay(cases, field_snapshot)
    replay_e2e = attach_replay(controls, e2e_snapshot)
    reconciliation = {
        "field_alt": reconcile_attribution(cases),
        "e2e_local": reconcile_attribution(controls),
    }

    per_split = {
        split: analyse(
            [case for case in everything if case["split"] == split and case["source"] == "field_alt"],
            field_snapshot,
        )
        for split in ("train", "eval", "holdout")
    }
    analysis = {
        "generated_from": {
            "field_snapshot": str(field_snapshot),
            "field_snapshot_manifest": load_manifest(field_snapshot),
            "e2e_snapshot": str(e2e_snapshot),
            "e2e_snapshot_manifest": load_manifest(e2e_snapshot),
            "streams": str(args.streams),
        },
        "journals": journal_summary,
        "e2e_journals": e2e_summary,
        "replay": {"field_alt": replay_field, "e2e_local": replay_e2e},
        "attribution_reconciliation": reconciliation,
        "recount_drift_probe": recount_drift_probe(
            [case for case in cases if case["split"] in ("train", "eval")], field_snapshot
        ),
        "all_field": analyse([case for case in everything if case["source"] == "field_alt"], field_snapshot),
        "train_eval_field": analyse(
            [
                case
                for case in everything
                if case["source"] == "field_alt" and case["split"] in ("train", "eval")
            ],
            field_snapshot,
        ),
        "per_split": per_split,
        "controls": analyse(controls, e2e_snapshot),
    }

    manifest = {
        "version": 1,
        "purpose": (
            "Replayable field material for the recall-map polish: every case is one "
            "persisted recall_events.recall_map payload with the recall that produced it, "
            "so RecallMapBuilder can be re-run offline against the pinned snapshot."
        ),
        "generated_at": journal_summary["ts_max"],
        "snapshots": {
            "field_alt": {
                "path": str(field_snapshot),
                "manifest": load_manifest(field_snapshot),
                "provenance": (
                    "sqlite backup-API snapshot of a read-only stream of host alt's live "
                    "living-memory store (/home/user/.local/share/living-memory/global.sqlite3 "
                    "plus its WAL). Nothing was written on alt."
                ),
            },
            "e2e_local": {
                "path": str(e2e_snapshot),
                "manifest": load_manifest(e2e_snapshot),
                "provenance": (
                    "sqlite backup-API snapshot of this workstation's living-memory store, "
                    "which is where the contrastive live-e2e run persisted its maps."
                ),
            },
        },
        "replay_contract": {
            "how": (
                "Open a working_copy() of the case's snapshot, build a MemoryRecallService, "
                "call memory_recall(query, scope=requested_scope, depth=depth, "
                "max_results=max_results, ambient_context=ambient_context, log_access=False, "
                "log_event=False), then feed service.last_residual to "
                "RecallMapBuilder.build(scope=scope, task=task, task_pattern=task_pattern)."
            ),
            "pool_recoverability": (
                "The residual pool itself is not persisted anywhere: retrieval.py:614 keeps it "
                "in memory only. What IS recoverable per case is head_node_ids (the ranked head "
                "the recall delivered, which is exactly the complement of the pool) and "
                "medoid_node_ids (pool members the map named). Replay fidelity should be "
                "asserted on head_node_ids: a replay that reproduces the head reproduced the "
                "ranking, and therefore the residual behind it."
            ),
            "caveat": (
                "The snapshot is later than the field run (recall_events up to "
                "2026-08-20T07:31Z), so a replay reconstructs the pool a recall would get "
                "against this corpus, not a byte-identical rerun of the historical one. "
                "Cases whose head_node_ids do not reproduce should be reported, not silently "
                "dropped."
            ),
        },
        "splits": {
            "unit": "cache_key = cache_key(scope, task=task) -- the builder's structure-cache key",
            "rationale": (
                "RecallMapBuilder caches cluster structure per (scope, normalized task) "
                "(recall_map.py:836-895). Two cases under one key are two looks at one cached "
                "structure, so keys are never split across train/eval/holdout."
            ),
            "shared_by_construction": (
                "All field keys live in scope project:game over one corpus; the splits are "
                "disjoint in cases and in cache keys, not in corpus. A rule fitted on train "
                "still meets unseen tasks, unseen pools and unseen label sets on holdout."
            ),
            "assignment": SPLIT_BY_KEY,
            "controls": (
                "The live-e2e cases (source e2e_local, split_role contrastive_control) are "
                "train by construction: diagnosis.md quotes them, and a case the diagnosis "
                "reads cannot also be held out. They are a different corpus and a different "
                "snapshot, so they are counted separately below and must not be pooled with "
                "the field cases in any metric."
            ),
            "counts": {
                split: sum(1 for case in everything if case["split"] == split)
                for split in ("train", "eval", "holdout")
            },
            "counts_by_source": {
                source: {
                    split: sum(
                        1
                        for case in everything
                        if case["split"] == split and case["source"] == source
                    )
                    for split in ("train", "eval", "holdout")
                }
                for source in ("field_alt", "e2e_local")
            },
            "holdout_discipline": (
                "Holdout cases are listed here in full so they can be replayed, and are "
                "neither analysed nor quoted in diagnosis.md. Every aggregate in diagnosis.md "
                "is computed over train+eval field cases only."
            ),
        },
        "cases": everything,
    }

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    Path(args.out_analysis).write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in analysis.items() if k != "generated_from"}, ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
