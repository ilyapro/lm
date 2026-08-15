#!/usr/bin/env python3
"""Build the privacy-safe de-identified replay corpus with pre-declared splits.

Reads the private staging dataset (step 01, never tracked) and writes the
tracked corpus:

- artifacts/animal-planet/corpus/{dev,eval,holdout}.jsonl — two record types
  per file: {"type":"node",...} (deduped within file) and
  {"type":"event",...}; each file is self-contained (embeds every node its
  events reference via results[].node_id or feedback_trace_id).
- artifacts/animal-planet/corpus/splits.json — the pre-declared split rule,
  hash bucket constants, per-split/per-scope/per-class counts, window bounds.

Also writes a PRIVATE id/originals map into staging (deid_map/) for the
failures-curation child; nothing private is ever printed to stdout — only
aggregate counts.

Split rule (declared in advance, by rule, not by inspection):

- source=local (all local-DB non-animal-planet workloads, >=2 project
  scopes) -> holdout only.
- source=alt: bucket = int.from_bytes(sha256(event_id)[:8], big) % 100;
  bucket < 15 -> holdout (the ~15% holdout-AP slice); 15 <= bucket < 66 ->
  dev; else eval (60/40 of the remainder: 51/85 and 34/85 buckets).
- No stratification by outcome.  The only allowed check: correction/
  supersedes-bearing events must land in both dev and eval; if the hash
  split left the family empty the DEV bound would be adjusted once and
  documented in splits.json (see family_check there).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deid import (  # noqa: E402
    HEX32_RE,
    TEMPORAL_HINT_RE,
    SurrogatePool,
    classify_class,
    classify_template,
    context_chars_map,
    provenance_shape_of,
    transform_value,
)

HERE = Path(__file__).resolve().parent
DEFAULT_STAGING = Path(
    os.environ.get("AP_STAGING", "/home/sfx/.cache/ap-audit/staging")
)
DEFAULT_OUT = HERE.parent.parent / "corpus"

# Pre-declared split constants (recorded verbatim in splits.json).
HASH_SPEC = "int.from_bytes(sha256(event_id.encode('utf-8')).digest()[:8], 'big') % 100"
MOD = 100
HOLDOUT_BOUND = 15  # alt buckets [0,15) -> holdout-AP slice (~15%)
DEV_BOUND = 66  # alt buckets [15,66) -> dev (51/85=60%), [66,100) -> eval (34/85=40%)

LOCAL_SLUGS = ("project_octopus", "project_online", "project_x")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def bucket_of(event_id: str) -> int:
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % MOD


def split_of(event_id: str, source: str, dev_bound: int) -> str:
    if source == "local":
        return "holdout"
    bucket = bucket_of(event_id)
    if bucket < HOLDOUT_BOUND:
        return "holdout"
    return "dev" if bucket < dev_bound else "eval"


def parse_json_col(raw, default):
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def surr_pair(value, pool, known_scopes):
    """(surrogate, chars) for a nullable private text field."""

    if value is None:
        return None, None
    return pool.surrogate(value), len(value)


def build_event_record(
    row: dict,
    source: str,
    pool: SurrogatePool,
    known_scopes: frozenset[str],
    matched_chars: dict[str, int],
    supersedes_endpoints: set[str],
    stats: Counter,
) -> dict:
    results = parse_json_col(row["results"], [])
    ambient = parse_json_col(row["ambient_context"], {})
    if not isinstance(ambient, dict):
        ambient = {"_raw": ambient}
        stats["ambient_context_nondict"] += 1
    resolved = parse_json_col(row["resolved_scopes"], [])

    query_surr, query_chars = surr_pair(row["query"], pool, known_scopes)
    agent_surr, agent_chars = surr_pair(row["agent"], pool, known_scopes)
    task_surr, task_chars = surr_pair(row["task"], pool, known_scopes)
    session_surr, session_chars = surr_pair(row["session_id"], pool, known_scopes)

    transport = row["transport_session_id"]
    if transport is not None and not HEX32_RE.match(transport):
        transport = pool.surrogate(transport)
        stats["transport_session_id_surrogated"] += 1

    node_ids = {r["node_id"] for r in results if r.get("node_id")}
    bearing = bool(node_ids & supersedes_endpoints)

    matched = row["id"] in matched_chars
    return {
        "type": "event",
        "id": row["id"],
        "source": source,
        "created_at": row["created_at"],
        "scope": row["scope"],
        "requested_scope": row["requested_scope"],
        "resolved_scopes": resolved,
        "depth": row["depth"],
        "max_results": row["max_results"],
        "class": classify_class(row["agent"]),
        "template_id": classify_template(row["query"]),
        "query_surrogate": query_surr,
        "query_chars": query_chars,
        "agent_surrogate": agent_surr,
        "agent_chars": agent_chars,
        "task_surrogate": task_surr,
        "task_chars": task_chars,
        "session_id_surrogate": session_surr,
        "session_id_chars": session_chars,
        "transport_session_id": transport,
        "ambient_context_surrogate": transform_value(ambient, pool, known_scopes),
        "ambient_context_chars": context_chars_map(ambient),
        "results": results,
        "feedback_applied": row["feedback_applied"],
        "feedback_trace_id": row["feedback_trace_id"],
        "feedback_applied_at": row["feedback_applied_at"],
        "supersedes_bearing": bearing,
        "transcript_matched": matched,
        "transcript_serialized_chars": matched_chars.get(row["id"]),
    }


def build_node_base(
    row: dict, pool: SurrogatePool, known_scopes: frozenset[str], stats: Counter
) -> dict:
    context = parse_json_col(row["context"], {})
    if not isinstance(context, dict):
        context = {"_raw": context}
        stats["node_context_nondict"] += 1
    source_traces = parse_json_col(row["source_traces"], [])
    corrections = parse_json_col(row["corrections"], [])
    provenance = parse_json_col(row["provenance"], {})
    if not isinstance(provenance, dict):
        provenance = {"_raw": provenance}
        stats["node_provenance_nondict"] += 1

    agent_surr, agent_chars = surr_pair(row["agent"], pool, known_scopes)
    task_surr, task_chars = surr_pair(row["task"], pool, known_scopes)
    decay_surr, decay_chars = surr_pair(row["decay_reason"], pool, known_scopes)

    temporal_hint = row["temporal_hint"]
    if temporal_hint is not None and not TEMPORAL_HINT_RE.match(temporal_hint):
        temporal_hint = pool.surrogate(temporal_hint)
        stats["temporal_hint_surrogated"] += 1

    corr_records = []
    corr_chars = []
    for corr in corrections:
        if not isinstance(corr, dict):
            stats["corrections_nondict"] += 1
            continue
        entry = dict(corr)
        chars = {}
        for key in ("old", "new"):
            if isinstance(entry.get(key), str):
                chars[key] = len(entry[key])
                entry[key] = pool.surrogate(entry[key])
        for key, value in list(entry.items()):
            if key not in ("old", "new"):
                entry[key] = transform_value(value, pool, known_scopes)
        corr_records.append(entry)
        corr_chars.append(chars)

    # node_to_dict ships provenance as {**provenance, source_traces, corrections};
    # the corpus keeps only its shape/sizes (prior_recalls carry raw queries).
    prov_shape = provenance_shape_of(provenance, source_traces, corrections)

    return {
        "type": "node",
        "id": row["id"],
        "level": row["level"],
        "scope": row["scope"],
        "included_via": [],  # filled per split file
        "agent_surrogate": agent_surr,
        "agent_chars": agent_chars,
        "task_surrogate": task_surr,
        "task_chars": task_chars,
        "timestamp": row["timestamp"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "decayed": row["decayed"],
        "decay_reason_surrogate": decay_surr,
        "decay_reason_chars": decay_chars,
        "stats": {
            "access_count": row["access_count"],
            "usefulness_score": row["usefulness_score"],
            "confidence": row["confidence"],
            "unique_agents": row["unique_agents"],
            "last_accessed": row["last_accessed"],
            "temporal_hint": temporal_hint,
        },
        "content_surrogate": pool.surrogate(row["content"]),
        "content_chars": len(row["content"]),
        "context_surrogate": transform_value(context, pool, known_scopes),
        "context_chars": context_chars_map(context),
        "source_traces": source_traces,
        "corrections": corr_records,
        "corrections_chars": corr_chars,
        "provenance_shape": prov_shape,
        "relations": [],  # filled per split file
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    staging: Path = args.staging
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((staging / "METADATA.json").read_text(encoding="utf-8"))
    stats: Counter = Counter()

    # ---- load staging ----------------------------------------------------
    alt_events = read_jsonl(staging / "alt-db" / "recall_events.jsonl")
    local_events: list[tuple[str, dict]] = []
    for slug in LOCAL_SLUGS:
        for row in read_jsonl(
            staging / "local-db" / f"holdout_{slug}.recall_events.jsonl"
        ):
            local_events.append((slug, row))

    alt_pool_files = [
        "nodes.jsonl",
        "nodes_typed_edges.jsonl",
        "game_schema_nodes.jsonl",
        "game_schema_source_nodes.jsonl",
    ]
    alt_nodes: dict[str, dict] = {}
    for name in alt_pool_files:
        for row in read_jsonl(staging / "alt-db" / name):
            prior = alt_nodes.get(row["id"])
            if prior is not None and prior != row:
                raise SystemExit(f"inconsistent duplicate node row: {row['id']}")
            alt_nodes[row["id"]] = row
    local_nodes: dict[str, dict] = {}
    for slug in LOCAL_SLUGS:
        for row in read_jsonl(staging / "local-db" / f"holdout_{slug}.nodes.jsonl"):
            prior = local_nodes.get(row["id"])
            if prior is not None and prior != row:
                raise SystemExit(f"inconsistent duplicate node row: {row['id']}")
            local_nodes[row["id"]] = row
    id_overlap = set(alt_nodes) & set(local_nodes)
    if id_overlap:
        raise SystemExit(f"alt/local node id collision: {sorted(id_overlap)[:5]}")

    typed_edges = read_jsonl(staging / "alt-db" / "connections_typed.jsonl")
    supersedes_endpoints = set()
    for edge in typed_edges:
        if edge["type"] == "supersedes":
            supersedes_endpoints.add(edge["source_id"])
            supersedes_endpoints.add(edge["target_id"])

    matched_chars = {
        row["db_id"]: row["len_json_content"]
        for row in read_jsonl(staging / "transcripts" / "matched.jsonl")
    }

    # ---- kept-real scope vocabulary --------------------------------------
    known_scopes = {"global"}
    for row in alt_events + [r for _, r in local_events]:
        known_scopes.add(row["scope"])
        known_scopes.add(row["requested_scope"])
        known_scopes.update(parse_json_col(row["resolved_scopes"], []))
    for row in list(alt_nodes.values()) + list(local_nodes.values()):
        known_scopes.add(row["scope"])
    known_scopes = frozenset(s for s in known_scopes if s)

    # ---- ephemeral salt: generated here, never recorded anywhere ---------
    pool = SurrogatePool(os.urandom(32))

    # ---- split assignment (pre-declared rule) ----------------------------
    assignments: dict[str, list[tuple[str, dict]]] = {
        "dev": [],
        "eval": [],
        "holdout": [],
    }
    for row in alt_events:
        assignments[split_of(row["id"], "alt", DEV_BOUND)].append(("alt", row))
    for slug, row in local_events:
        assignments["holdout"].append(("local", row))

    # Documented family check (the only allowed stratification check): the
    # hash split must leave correction/supersedes-bearing events in BOTH dev
    # and eval.  If violated, DEV_BOUND would be adjusted once; record the
    # outcome either way.
    def family_count(split: str) -> int:
        count = 0
        for _, row in assignments[split]:
            node_ids = {
                r.get("node_id") for r in parse_json_col(row["results"], [])
            }
            if node_ids & supersedes_endpoints:
                count += 1
        return count

    family = {"dev": family_count("dev"), "eval": family_count("eval")}
    family_check = {
        "predicate": (
            "event is correction/supersedes-bearing iff any results[].node_id is an"
            " endpoint (source or target) of a supersedes connection in"
            " alt-db/connections_typed.jsonl"
        ),
        "dev_bearing_events": family["dev"],
        "eval_bearing_events": family["eval"],
        "dev_bound_adjusted": False,
        "dev_bound": DEV_BOUND,
    }
    if not (family["dev"] and family["eval"]):
        raise SystemExit(
            "family check failed with DEV_BOUND=66; adjust the constant once and"
            f" document it (dev={family['dev']}, eval={family['eval']})"
        )

    # ---- build event records ---------------------------------------------
    split_events: dict[str, list[dict]] = {}
    for split, pairs in assignments.items():
        records = []
        for source, row in pairs:
            records.append(
                build_event_record(
                    row,
                    source,
                    pool,
                    known_scopes,
                    matched_chars,
                    supersedes_endpoints,
                    stats,
                )
            )
        records.sort(key=lambda r: (r["created_at"], r["id"]))
        split_events[split] = records

    # ---- per-split node sets ----------------------------------------------
    def node_row(node_id: str) -> dict | None:
        return alt_nodes.get(node_id) or local_nodes.get(node_id)

    evidence_ids = set()
    for name in ("game_schema_nodes.jsonl", "game_schema_source_nodes.jsonl"):
        for row in read_jsonl(staging / "alt-db" / name):
            evidence_ids.add(row["id"])

    split_node_reasons: dict[str, dict[str, set]] = {}
    for split, records in split_events.items():
        reasons: dict[str, set] = {}
        for record in records:
            for result in record["results"]:
                nid = result.get("node_id")
                if nid:
                    reasons.setdefault(nid, set()).add("results")
            if record["feedback_trace_id"]:
                reasons.setdefault(record["feedback_trace_id"], set()).add(
                    "feedback"
                )
        if split in ("dev", "eval"):
            # mixed-era consolidation evidence: project:game schemas + sources
            for nid in evidence_ids:
                reasons.setdefault(nid, set()).add("consolidation_evidence")
        # one-hop typed-edge partners (alt connections only; the local DB's
        # connections were not exported by the extract stage)
        base = set(reasons)
        for edge in typed_edges:
            for here, there in (
                (edge["source_id"], edge["target_id"]),
                (edge["target_id"], edge["source_id"]),
            ):
                if here in base and there not in base and node_row(there):
                    reasons.setdefault(there, set()).add("typed_edge_partner")
        missing = [nid for nid in reasons if not node_row(nid)]
        if missing:
            raise SystemExit(
                f"{split}: {len(missing)} referenced nodes missing from staging"
                f" pools, e.g. {sorted(missing)[:3]}"
            )
        split_node_reasons[split] = reasons

    # ---- transform nodes once, attach per-split relations ------------------
    node_base: dict[str, dict] = {}
    all_needed = sorted(set().union(*[set(r) for r in split_node_reasons.values()]))
    for nid in all_needed:
        node_base[nid] = build_node_base(node_row(nid), pool, known_scopes, stats)

    def relations_for(node_id: str, present: set[str]) -> list[dict]:
        rels = []
        for edge in typed_edges:
            if edge["source_id"] == node_id and edge["target_id"] in present:
                rels.append(
                    {
                        "type": edge["type"],
                        "direction": "out",
                        "other_id": edge["target_id"],
                        "weight": edge["weight"],
                        "created_at": edge["created_at"],
                    }
                )
            elif edge["target_id"] == node_id and edge["source_id"] in present:
                rels.append(
                    {
                        "type": edge["type"],
                        "direction": "in",
                        "other_id": edge["source_id"],
                        "weight": edge["weight"],
                        "created_at": edge["created_at"],
                    }
                )
        rels.sort(key=lambda r: (r["type"], r["direction"], r["other_id"], r["created_at"]))
        return rels

    # ---- write split files -------------------------------------------------
    file_stats = {}
    for split in ("dev", "eval", "holdout"):
        reasons = split_node_reasons[split]
        present = set(reasons)
        path = out / f"{split}.jsonl"
        node_count = 0
        with path.open("w", encoding="utf-8") as fh:
            for nid in sorted(present):
                record = dict(node_base[nid])
                record["included_via"] = sorted(reasons[nid])
                record["relations"] = relations_for(nid, present)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                node_count += 1
            for record in split_events[split]:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        events = split_events[split]
        file_stats[split] = {
            "events": len(events),
            "nodes": node_count,
            "by_source": dict(Counter(r["source"] for r in events)),
            "by_scope": dict(
                sorted(
                    Counter(r["scope"] for r in events).items(),
                    key=lambda kv: -kv[1],
                )
            ),
            "by_class": dict(Counter(r["class"] for r in events)),
            "by_template": dict(
                Counter(str(r["template_id"]) for r in events)
            ),
            "supersedes_bearing_events": sum(
                1 for r in events if r["supersedes_bearing"]
            ),
            "transcript_matched_events": sum(
                1 for r in events if r["transcript_matched"]
            ),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    # ---- splits.json --------------------------------------------------------
    all_ids = [r["id"] for records in split_events.values() for r in records]
    assert len(all_ids) == len(set(all_ids)), "split ids must be disjoint"
    splits = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "artifacts/animal-planet/recipe/transform/build_corpus.py",
        "staging_metadata_sha256": hashlib.sha256(
            (staging / "METADATA.json").read_bytes()
        ).hexdigest(),
        "rule": {
            "text": (
                "Declared in advance, by rule, not by inspection. source=local"
                " (all local-DB non-animal-planet workloads) -> holdout only."
                " source=alt (every recall_event exported from the alt DB in the"
                f" export window) -> bucket = {HASH_SPEC}; bucket <"
                f" {HOLDOUT_BOUND} -> holdout (~15% holdout-AP slice);"
                f" {HOLDOUT_BOUND} <= bucket < {DEV_BOUND} -> dev (51/85 = 60% of"
                f" the remainder); {DEV_BOUND} <= bucket < {MOD} -> eval (34/85 ="
                " 40%). No stratification by outcome."
            ),
            "hash": HASH_SPEC,
            "mod": MOD,
            "holdout_bound": HOLDOUT_BOUND,
            "dev_bound": DEV_BOUND,
            "local_sources": [f"local-db/holdout_{slug}" for slug in LOCAL_SLUGS],
        },
        "family_check": family_check,
        # bounds only — the METADATA *_source annotations quote private node
        # text and must not reach a tracked file
        "windows": {
            name: {
                key: value
                for key, value in window.items()
                if key in ("start", "end")
            }
            for name, window in metadata["windows"].items()
        },
        "event_class_predicate": metadata["predicates"]["organic_vs_automatic"][
            "pinned_automatic"
        ]
        + " -> automatic, else organic",
        "template_predicate": metadata["predicates"]["deterministic_prerecalls"][
            "pinned"
        ],
        "counts": file_stats,
        "totals": {
            "events": len(all_ids),
            "alt_events": len(alt_events),
            "local_events": len(local_events),
        },
    }
    (out / "splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ---- private id/originals map for the failures child (staging only) ----
    map_dir = staging / "deid_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    mapping = pool.mapping()
    with (map_dir / "surrogates.jsonl").open("w", encoding="utf-8") as fh:
        for original in sorted(mapping):
            fh.write(
                json.dumps(
                    {"surrogate": mapping[original], "original": original},
                    ensure_ascii=False,
                )
                + "\n"
            )
    (map_dir / "README.md").write_text(
        "# deid_map (PRIVATE, staging-only, never tracked)\n\n"
        "`surrogates.jsonl`: one `{surrogate, original}` pair per unique\n"
        "de-identified string of the tracked corpus build (global equality\n"
        "classes: the same surrogate always denotes the same original).\n\n"
        "To resolve a corpus record back to raw text: look its `id` up in the\n"
        "staging exports (alt-db/*.jsonl, local-db/*.jsonl) — corpus ids are\n"
        "real — or look any `*_surrogate` string up here.  The HMAC salt was\n"
        "ephemeral and is NOT recorded; this file is the only bridge.\n",
        encoding="utf-8",
    )
    (map_dir / "build_info.json").write_text(
        json.dumps(
            {
                "built_at": splits["generated_at"],
                "corpus_sha256": {s: file_stats[s]["sha256"] for s in file_stats},
                "unique_strings": len(mapping),
                "collision_retries": pool.collision_retries,
                "salt": "ephemeral, not recorded (non-invertibility)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ---- aggregate report (no raw content) ---------------------------------
    print(
        json.dumps(
            {
                "out": str(out),
                "splits": {
                    s: {
                        k: v
                        for k, v in file_stats[s].items()
                        if k not in ("sha256",)
                    }
                    for s in file_stats
                },
                "family_check": family_check,
                "unique_surrogated_strings": len(mapping),
                "collision_retries": pool.collision_retries,
                "anomalies": dict(stats),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
