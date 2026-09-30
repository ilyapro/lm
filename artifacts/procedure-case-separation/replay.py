"""Replay the procedural stage on a private read-only copy; print aggregates only.

Usage: python3 replay.py <copy.sqlite3> <work-dir>

The copy is never modified: ``before`` is measured on it read as-is (the state
the replaced mechanism produced), ``after`` on a second copy in <work-dir>
after the procedural stage of an ordinary pass ran twice per scope with the
current code. Evaluation sets are drawn with a fixed seed from the copy
itself; nothing here names a project, node or text, only counts.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from living_memory import consolidation as C  # noqa: E402
from living_memory.server import create_mcp_server  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402
from test_transport_identity import FakeMCP  # noqa: E402

SEED = 20260930
SAMPLE = 40
STEP_KEYS = ("step", "step_order", "step_description", "step_content")
# Evaluation-only selector of whole prose recipes; production reads no text.
RECIPE_HEAD = re.compile(r"^\s*(procedure|процедура|recipe|рецепт)\b", re.IGNORECASE)


def read(result: dict, lookup) -> tuple[str, int]:
    """What a reader takes from one delivered result, and the lookup chars that
    forces. The policy sees only the query and what recall delivered: a result
    delivered cut (any delivery but ``full``) is re-fetched whole by its id;
    nothing else is looked up (no target id, no ids a carrier lists, nothing
    undelivered). Fixed before the full-text carrier replay; the same reader
    measures original and candidate."""

    node = result["node"]
    text, forced = node.get("content") or "", 0
    if result.get("delivery", "full") != "full":
        ref = (result.get("content_ref") or {}).get("node_id") or node["id"]
        fetched = lookup(node_id=ref)
        forced += len(json.dumps(fetched, ensure_ascii=False))
        text = fetched["results"][0]["content"] if fetched["results"] else text
    return text, forced


def load_operator_env() -> None:
    env = Path.home() / ".config/living-memory/env"
    skip = {"LM_AUTH_TOKEN", "LM_DB_PATH", "LM_AUTO_CONSOLIDATE_POLICY"}
    for line in env.read_text().splitlines():
        key, _, value = line.partition("=")
        if key and not key.startswith("#") and key not in skip:
            os.environ[key.strip()] = value.strip().strip('"')
    os.environ.pop("LM_AUTO_CONSOLIDATE_POLICY", None)


def readable(key: str) -> bool:
    return bool(re.search(r"[^\W\d_]{3,}", key)) and not re.fullmatch(r"[0-9a-f]{8,}", key)


def census(store: MemoryStore) -> dict:
    schemas = [
        node
        for node in store.list_nodes(level="schema", include_decayed=False, limit=1_000_000)
        if node.provenance.get("strategy") == "procedural"
    ]
    visited = visited_groups(store)
    per_schema: dict[str, int] = {}
    in_visited: dict[str, int] = {}
    groups: dict[tuple[str, str], int] = {}
    triggers: dict[tuple[str, str], set[str]] = {}
    for schema in schemas:
        sources = [store.get_node(node_id) for node_id in schema.source_traces]
        declared = {
            C._step_description(node) for node in sources if node is not None and C._declares_step(node)
        }
        steps = list(schema.context.get("procedure") or [])
        per_schema[schema.id] = sum(1 for step in steps if step not in declared)
        group = (schema.scope, C._schema_group_id(schema))
        if group in visited:
            in_visited[schema.id] = per_schema[schema.id]
        groups[group] = groups.get(group, 0) + 1
        triggers.setdefault((schema.scope, str(schema.context.get("trigger"))), set()).add(group[1])
    steps_total = sum(len(schema.context.get("procedure") or []) for schema in schemas)
    carriers = [
        node
        for node in store.list_nodes(level="concept", include_decayed=False, limit=1_000_000)
        if node.provenance.get("strategy") == "procedural"
    ]
    for carrier in carriers:
        groups[(carrier.scope, C._schema_group_id(carrier))] = groups.get((carrier.scope, C._schema_group_id(carrier)), 0) + 1
    return {
        "active_schemas": len(schemas),
        "active_group_carriers": len(carriers),
        "carrier_chars": sum(len(node.content) for node in carriers),
        "max_carrier_chars": max((len(node.content) for node in carriers), default=0),
        "schema_chars": sum(len(schema.content) for schema in schemas),
        "max_schema_chars": max((len(schema.content) for schema in schemas), default=0),
        "steps": steps_total,
        "unconfirmed_steps": sum(per_schema.values()),
        "unconfirmed_steps_in_visited_groups": sum(in_visited.values()),
        "unconfirmed_share": round(sum(per_schema.values()) / steps_total, 4) if steps_total else 0.0,
        "schemas_with_unconfirmed_steps": sum(1 for count in per_schema.values() if count),
        "max_active_nodes_per_group": max(groups.values(), default=0),
        "triggers_shared_by_groups": sum(1 for keys in triggers.values() if len(keys) > 1),
        "_unconfirmed": per_schema,
        "_visited": in_visited,
    }


def procedural_scopes(store: MemoryStore) -> list[str]:
    return sorted(
        {
            row[0]
            for row in store.connection.execute(
                "SELECT DISTINCT scope FROM nodes WHERE level='trace' AND decayed=0 AND "
                "(json_extract(context,'$.procedure_id') IS NOT NULL OR "
                "json_extract(context,'$.task_pattern') IS NOT NULL)"
            )
        }
    )


def visited_groups(store: MemoryStore) -> set[tuple[str, str]]:
    """Groups the ordinary pass examines: enough active keyed traces in its listing."""

    sizes: dict[tuple[str, str], int] = {}
    for scope in procedural_scopes(store):
        for trace in store.list_nodes(
            level="trace", scope=scope, include_decayed=False, limit=C.DEFAULT_RECENT_LIMIT
        ):
            if key := C._procedure_key(trace):
                sizes[(trace.scope, key.group_id)] = sizes.get((trace.scope, key.group_id), 0) + 1
    return {group for group, size in sizes.items() if size >= C.PROCEDURAL_MIN_CLUSTER_SIZE}


def procedural_pass(store: MemoryStore) -> dict:
    scopes = procedural_scopes(store)
    before = {node.id for node in store.list_nodes(level="schema", limit=1_000_000)}
    created = updated = 0
    started = time.perf_counter()
    for scope in scopes:
        traces = store.list_nodes(
            level="trace", scope=scope, include_decayed=False, limit=C.DEFAULT_RECENT_LIMIT
        )
        for _schema, fresh in C._materialize_procedural_schemas(
            store, traces, min_cluster_size=C.PROCEDURAL_MIN_CLUSTER_SIZE
        ):
            created += fresh
            updated += not fresh
    elapsed = time.perf_counter() - started
    after = {node.id for node in store.list_nodes(level="schema", limit=1_000_000)}
    return {
        "scopes": len(scopes),
        "seconds": round(elapsed, 2),
        "created": created,
        "updated": updated,
        "retired": len(before - after),
    }


def eval_sets(store: MemoryStore) -> dict[str, list[tuple[str, str, str | None]]]:
    """(scope, query, target recipe id) — drawn once, before any after-state exists."""

    rng = random.Random(SEED)
    schema_queries = []
    for schema in store.list_nodes(level="schema", include_decayed=False, limit=1_000_000):
        trigger = str(schema.context.get("trigger") or "")
        if schema.provenance.get("strategy") == "procedural" and readable(trigger):
            schema_queries.append((schema.scope, f"how to {trigger}", None))
    recipes = []
    for node in store.list_nodes(level="trace", include_decayed=False, limit=1_000_000):
        key = str(node.context.get("procedure_id") or node.context.get("task_pattern") or "")
        if not key or any(k in node.context for k in STEP_KEYS) or not RECIPE_HEAD.match(node.content):
            continue
        if readable(key):
            recipes.append((node.scope, f"how to {C._normalize_trigger(key)}", node.id))
    rng.shuffle(schema_queries)
    rng.shuffle(recipes)
    return {"schema_topics": schema_queries[:SAMPLE], "prose_recipes": recipes[:SAMPLE]}


def measure(db: Path, queries, unconfirmed: dict[str, int], visited: dict[str, int]) -> dict:
    mcp = create_mcp_server(db, mcp_factory=FakeMCP)
    lookup, recall = mcp.tools["memory_lookup"], mcp.tools["memory_recall"]
    store = mcp.memory_store
    rows = []
    for scope, query, target in queries:
        started = time.perf_counter()
        response = recall(query, scope=scope, max_results=10)
        latency = time.perf_counter() - started
        delivered = len(json.dumps(response, ensure_ascii=False))
        forced = 0
        schemas = unconfirmed_delivered = unconfirmed_visited = 0
        obtained = own = False
        target_text = store.get_node(target).content if target else None
        for result in response["results"]:
            node = result["node"]
            if node.get("level") == "schema":
                schemas += 1
                unconfirmed_delivered += unconfirmed.get(node["id"], 0)
                unconfirmed_visited += visited.get(node["id"], 0)
            text, cost = read(result, lookup)
            forced += cost
            if target_text is not None and target_text in text:
                obtained = True
                own = own or node["id"] == target
        by_name = False
        if target:
            context = store.get_node(target).context
            field = "procedure_id" if context.get("procedure_id") else "task_pattern"
            found = lookup(scope=scope, **{field: context[field]})["results"]
            by_name = target in {node["id"] for node in found}
        rows.append((delivered, forced, latency, schemas, unconfirmed_delivered, obtained, by_name, own, unconfirmed_visited))
    store.close()
    total = [row[0] + row[1] for row in rows]
    return {
        "queries": len(rows),
        "delivered_chars_total": sum(row[0] for row in rows),
        "forced_lookup_chars_total": sum(row[1] for row in rows),
        "delivered_plus_lookup_median": median(total) if total else 0,
        "delivered_plus_lookup_total": sum(total),
        "recall_ms_median": round(median(row[2] for row in rows) * 1000, 1) if rows else 0,
        "schemas_delivered": sum(row[3] for row in rows),
        "unconfirmed_steps_delivered": sum(row[4] for row in rows),
        "unconfirmed_steps_delivered_from_visited_groups": sum(row[8] for row in rows),
        "recipe_obtained_whole": sum(row[5] for row in rows),
        "recipe_in_lookup_by_procedure_name": sum(row[6] for row in rows),
        "recipe_obtained_from_own_trace": sum(row[7] for row in rows),
        "_obtained": [row[5] for row in rows],
        "_own": [row[7] for row in rows],
    }


# Live ranking valves that act on a single trace; toggled only in this
# diagnostic process, never in production or on the operator's server.
VALVES = ("LM_HUB_SUPPRESSION_FACTOR", "LM_RECALL_MIN_SCORE")


def attribute_losses(db: Path, queries, before: dict, after: dict) -> dict:
    """For each recipe obtained before but not after: which carrier it lost, and
    which live valve keeps its own trace out of recall (the same in both states)."""

    lost = [
        index
        for index, (was, now) in enumerate(zip(before["_obtained"], after["_obtained"]))
        if was and not now
    ]
    carriers = {"own_trace": 0, "schema_only": 0}
    released_by: dict[str, int] = {}
    for index in lost:
        carriers["own_trace" if before["_own"][index] else "schema_only"] += 1
        for valve in VALVES:
            saved = os.environ.pop(valve, None)
            try:
                if measure(db, [queries[index]], {}, {})["recipe_obtained_from_own_trace"]:
                    released_by[valve] = released_by.get(valve, 0) + 1
            finally:
                if saved is not None:
                    os.environ[valve] = saved
    return {"lost": len(lost), "lost_carrier": carriers, "own_trace_released_with_valve_off": released_by}


def main() -> None:
    source, work = Path(sys.argv[1]), Path(sys.argv[2])
    load_operator_env()
    work.mkdir(parents=True, exist_ok=True)
    before_db, after_db = work / "before.sqlite3", work / "after.sqlite3"
    shutil.copyfile(source, before_db)
    shutil.copyfile(source, after_db)

    with MemoryStore(before_db) as store:
        sets = eval_sets(store)
        before = census(store)
    with MemoryStore(after_db) as store:
        first = procedural_pass(store)
        second = procedural_pass(store)
        after = census(store)

    report = {"census_before": before, "pass_1": first, "pass_2": second, "census_after": after}
    for name, queries in sets.items():
        was = measure(before_db, queries, before["_unconfirmed"], before["_visited"])
        now = measure(after_db, queries, after["_unconfirmed"], after["_visited"])
        if name == "prose_recipes":
            report["prose_recipes_losses"] = attribute_losses(after_db, queries, was, now)
        for block in (was, now):
            block.pop("_obtained")
            block.pop("_own")
        report[f"{name}_before"], report[f"{name}_after"] = was, now
    for block in (before, after):
        block.pop("_unconfirmed")
        block.pop("_visited")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
