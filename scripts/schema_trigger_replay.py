#!/usr/bin/env python3
"""Live-path replay of ``LM_RECALL_SCHEMA_TRIGGER`` (goal schema-ranks-by-meaning).

Two arms on the same counterfactual store, one host at a time, never merged:

- ``legacy``: the host's field env (``FIELD_ENV``), trigger channel as it is;
- ``name``: the same env plus ``LM_RECALL_SCHEMA_TRIGGER=name``.

Reuses ``scripts/recall_precision_replay.py`` for everything that is not
about schemas: snapshots (SQLite backup API, sfx live DB ``mode=ro``, alt
streamed to sfx over ssh), the marks-era event window, the counterfactual copy
that strips every mark/credit/anchor/edge learned at or after the cutoff, the
future-candidate horizon and the latest accepted mark per ``(event, node)``.

Segments (``segment_bounds``): ``dev`` is every window event before the
recall-precision holdout start committed in ``artifacts/recall-precision/split.json``
(that goal's train+eval; the design was looked at here, counterfactual cutoff
at the split's eval start), ``holdout`` is every window event from that
holdout start up to :data:`HOLDOUT_END` (cutoff at the holdout start). The
fresh snapshot adds the events recorded after that split was made; the
holdout end stops before this goal's own recall traffic began.

Metrics per arm, on accepted marks of ``level='schema'`` nodes (``summarize``):
irrelevant/used marked schemas that the arm still delivers in full, the
irrelevant share among delivered marked schemas, the share of ``used`` kept
against ``legacy``; the same restricted to schemas the field found by trigger
(recorded ``methods`` contain ``trigger``); schema slots and schema text
delivered (content chars after ``delivery.shape_recall_results``, and full
node chars); rank-1 changes; recall latency.

Usage::

    python3 scripts/schema_trigger_replay.py run --host sfx --segment holdout \\
        --snapshot /path/sfx.sqlite3 --out artifacts/schema-trigger/holdout-sfx.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import recall_precision_replay as rpr  # noqa: E402

#: Exclusive end of the holdout: this goal's own recalls start after it.
HOLDOUT_END = "2026-09-29T07:00:00Z"
#: The valve env lines each host runs with (``~/.config/living-memory/env``,
#: 2026-09-29), plus the explicit-feedback policy demotion is read under.
FIELD_ENV: dict[str, dict[str, str]] = {
    "sfx": {
        "LM_EXPLICIT_FEEDBACK_POLICY": "credit",
        "LM_RECALL_NEAR_DUP_COSINE": "0.97",
        "LM_QUERY_IRRELEVANCE_FULL_COSINE": "0.80",
        "LM_QUERY_IRRELEVANCE_MARK_WEIGHT": "1.0",
        "LM_RECALL_SCHEMA_DEDUP": "1",
        "LM_RECALL_MIN_SCORE": "0.35",
        "LM_RECALL_GATE_FORM": "drop",
        "LM_HUB_SUPPRESSION_FACTOR": "0.1",
    },
    "alt": {
        "LM_EXPLICIT_FEEDBACK_POLICY": "credit",
        "LM_RECALL_NEAR_DUP_COSINE": "0.97",
        "LM_QUERY_IRRELEVANCE_FULL_COSINE": "0.80",
        "LM_QUERY_IRRELEVANCE_MARK_WEIGHT": "1.0",
        "LM_RECALL_SCHEMA_DEDUP": "1",
        "LM_RECALL_MIN_SCORE": "0.30",
        "LM_RECALL_GATE_FORM": "drop",
        "LM_HUB_SUPPRESSION_FACTOR": "0.1",
    },
}
VALVE = {"LM_RECALL_SCHEMA_TRIGGER": "name"}


def arms_for(host: str) -> dict[str, dict[str, str]]:
    return {"legacy": dict(FIELD_ENV[host]), "name": {**FIELD_ENV[host], **VALVE}}


def segment_bounds(host_split: Mapping[str, Any], segment: str) -> tuple[tuple[str, str] | None, tuple[str, str] | None, str]:
    """``(start, end, cutoff)`` with ``(created_at, id)`` keys; ``None`` is open."""

    hold = (host_split["holdout_start"]["created_at"], host_split["holdout_start"]["event_id"])
    if segment == "dev":
        return None, hold, host_split["eval_start"]["created_at"]
    if segment == "holdout":
        return hold, (HOLDOUT_END, ""), host_split["holdout_start"]["created_at"]
    raise ValueError(segment)


def recorded_trigger_nodes(snapshot: str | Path, event_ids: Sequence[str]) -> dict[str, set[str]]:
    """Per event, the node ids the field delivered with ``trigger`` in ``methods``."""

    connection = rpr.open_readonly(snapshot)
    try:
        out: dict[str, set[str]] = {}
        for event_id in event_ids:
            row = connection.execute("SELECT results FROM recall_events WHERE id = ?", (event_id,)).fetchone()
            items = rpr.efa_json(row[0] if row else None, [])
            out[event_id] = {
                str(item["node_id"])
                for item in items
                if isinstance(item, dict) and item.get("node_id") and "trigger" in (item.get("methods") or ())
            }
        return out
    finally:
        connection.close()


def node_levels(snapshot: str | Path) -> dict[str, str]:
    connection = rpr.open_readonly(snapshot)
    try:
        return {str(row[0]): str(row[1]) for row in connection.execute("SELECT id, level FROM nodes")}
    finally:
        connection.close()


def replay_arms(
    db_path: str | Path, arms: Mapping[str, Mapping[str, str]], events: Sequence[rpr.ReplayEvent]
) -> tuple[dict[str, list[list[dict[str, Any]]]], dict[str, list[float]]]:
    """Each event through ``memory_recall`` per arm; delivered entries with shaped sizes."""

    from living_memory.config import MemoryConfig
    from living_memory.delivery import shape_recall_results
    from living_memory.retrieval import MemoryRecallService
    from living_memory.storage import MemoryStore

    class HorizonRecallService(MemoryRecallService):
        horizon: str | None = None

        def rank_candidates(self, candidates, plan, **kwargs):  # type: ignore[no-untyped-def]
            if self.horizon is not None:
                candidates = {
                    key: candidate
                    for key, candidate in candidates.items()
                    if rpr.efa.normalize_ts(str(candidate.node.created_at)) <= self.horizon
                }
            return super().rank_candidates(candidates, plan, **kwargs)

    names = list(arms)
    stores = {name: MemoryStore(MemoryConfig(db_path=Path(db_path))) for name in names}
    out: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in names}
    latency: dict[str, list[float]] = {name: [] for name in names}
    try:
        services = {name: HorizonRecallService(stores[name]) for name in names}
        for index, event in enumerate(events):
            for name in names if index % 2 == 0 else list(reversed(names)):
                services[name].horizon = event.created_at
                with rpr.applied_env(arms[name]):
                    started = time.perf_counter()
                    results = services[name].memory_recall(
                        event.query,
                        scope=event.scope,
                        ambient_context=dict(event.ambient_context),
                        depth=event.depth,
                        max_results=event.max_results,
                        log_access=False,
                        log_event=False,
                    )
                    latency[name].append(time.perf_counter() - started)
                    shaped = shape_recall_results(
                        results,
                        already_delivered_ids=set(),
                        snippet_max_chars=1000,
                        context_value_max_chars=300,
                        session_dedup=False,
                    )
                out[name].append(
                    [
                        {
                            "node_id": result.node.id,
                            "level": str(result.node.level),
                            "full": not result.withheld,
                            "created_at": rpr.efa.normalize_ts(str(result.node.created_at)),
                            "chars": len(str(entry.get("content") or entry.get("node", {}).get("content") or "")),
                            "full_chars": len(result.node.content or ""),
                            "trigger": result.trigger_score > 0.0,
                        }
                        for result, entry in zip(results, shaped, strict=True)
                    ]
                )
    finally:
        for store in stores.values():
            store.close()
    return out, latency


def score_arm(
    events: Sequence[rpr.ReplayEvent],
    delivered: Sequence[Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Mapping[str, rpr.Label]],
    levels: Mapping[str, str],
    trigger_nodes: Mapping[str, set[str]],
    node_created: Mapping[str, str],
    cutoff: str,
) -> dict[str, Any]:
    stamp = rpr.stamp_prefix(cutoff)
    total: Counter[str] = Counter()
    rank1: list[str | None] = []
    for event, items in zip(events, delivered, strict=True):
        kept = [item for item in items if not (item["created_at"] and item["created_at"] > event.created_at)]
        full = [item for item in kept if item["full"]]
        full_ids = {item["node_id"] for item in full}
        rank1.append(full[0]["node_id"] if full else None)
        schemas = [item for item in full if item["level"] == "schema"]
        total["events"] += 1
        total["slots"] += len(full)
        total["chars"] += sum(item["chars"] for item in full)
        total["schema_slots"] += len(schemas)
        total["schema_chars"] += sum(item["chars"] for item in schemas)
        total["schema_full_chars"] += sum(item["full_chars"] for item in schemas)
        total["schema_trigger_slots"] += sum(1 for item in schemas if item["trigger"])
        total["schema_rank1"] += bool(full) and full[0]["level"] == "schema"
        for node_id, label in labels.get(event.event_id, {}).items():
            if levels.get(node_id) != "schema":
                continue
            if rpr.stamp_prefix(node_created.get(node_id, "") or "0000") >= stamp:
                total["labels_created_after_cutoff"] += 1
                continue
            mark = "irr" if label.mark == "irrelevant" else "used" if label.mark == "used" else None
            if mark is None:
                continue
            for prefix, applies in (("", True), ("trig_", node_id in trigger_nodes.get(event.event_id, ()))):
                if not applies:
                    continue
                total[f"{prefix}{mark}_marked"] += 1
                total[f"{prefix}{mark}_full"] += node_id in full_ids
    summary: dict[str, Any] = {key: int(value) for key, value in sorted(total.items())}
    for prefix in ("", "trig_"):
        irr, used = total[f"{prefix}irr_full"], total[f"{prefix}used_full"]
        summary[f"{prefix}irrelevant_share"] = round(irr / (irr + used), 4) if irr + used else None
    return {"summary": summary, "rank1": rank1}


def run(host: str, segment: str, snapshot: Path, split_path: Path, workdir: Path, sample: int | None) -> dict[str, Any]:
    started = time.perf_counter()
    host_split = json.loads(split_path.read_text(encoding="utf-8"))["hosts"][host]
    start, end, cutoff = segment_bounds(host_split, segment)
    store = rpr.load_host_store(snapshot)
    chosen = [
        event.id
        for event in rpr.window_events(store)
        if (start is None or (event.created_at, event.id) >= start)
        and (end is None or (event.created_at, event.id) < end)
    ]
    if sample is not None and sample < len(chosen):
        import random

        chosen = sorted(random.Random(7).sample(chosen, sample), key=lambda eid: (store.events[eid].created_at, eid))
    workdir.mkdir(parents=True, exist_ok=True)
    counterfactual = rpr.build_counterfactual(snapshot, workdir / f"{host}-{segment}.sqlite3", cutoff)
    events = rpr.load_replay_events(snapshot, chosen)
    labels = rpr.event_labels(store)
    levels = node_levels(snapshot)
    trigger_nodes = recorded_trigger_nodes(snapshot, [event.event_id for event in events])
    arms = arms_for(host)
    delivered, latency = replay_arms(counterfactual["path"], arms, events)
    scored = {
        name: score_arm(events, delivered[name], labels, levels, trigger_nodes, store.node_created, cutoff)
        for name in arms
    }
    summaries: dict[str, Any] = {}
    for name in arms:
        summary = dict(scored[name]["summary"])
        values = sorted(latency[name])
        summary["latency_s"] = {
            "mean": round(statistics.fmean(values), 4) if values else None,
            "p50": round(values[len(values) // 2], 4) if values else None,
            "p90": round(values[int(0.9 * (len(values) - 1))], 4) if values else None,
        }
        summaries[name] = summary
    base, new = summaries["legacy"], summaries["name"]
    for prefix in ("", "trig_"):
        kept = new.get(f"{prefix}used_full", 0)
        was = base.get(f"{prefix}used_full", 0)
        new[f"{prefix}used_kept_vs_legacy"] = round(kept / was, 4) if was else None
    new["rank1_changed"] = sum(
        1 for a, b in zip(scored["legacy"]["rank1"], scored["name"]["rank1"], strict=True) if a != b
    )
    return {
        "host": host,
        "segment": segment,
        "snapshot": str(snapshot),
        "snapshot_sha256": rpr.sha256_file(snapshot),
        "window": {"start": start, "end": end, "cutoff": cutoff},
        "events": len(events),
        "sample": sample,
        "arms": arms,
        "counterfactual": counterfactual,
        "summary": summaries,
        "runtime_s": round(time.perf_counter() - started, 1),
        "generated_at": rpr.utc_now(),
    }


def names(host: str, snapshot: Path, workdir: Path, sample: int) -> dict[str, Any]:
    """L5: each sampled schema queried by its own trigger, in its own scope, per arm.

    Runs on a backup copy (recall may backfill embeddings). A same-title
    schema counts as the hit (``LM_RECALL_SCHEMA_DEDUP`` keeps one of them).
    ``agree`` is meaning agreement as ``legacy`` saw it: the schema's
    vector >= 0.55 or bm25 >= 0.2.
    """

    import random

    from living_memory.config import MemoryConfig
    from living_memory.retrieval import MemoryRecallService
    from living_memory.score_gate import gate_score
    from living_memory.storage import MemoryStore

    workdir.mkdir(parents=True, exist_ok=True)
    copy = workdir / f"{host}-names.sqlite3"
    rpr.backup_database(snapshot, copy)
    arms = arms_for(host)
    counts = {name: Counter() for name in arms}
    latency: dict[str, list[float]] = {name: [] for name in arms}
    misses: dict[str, list[str]] = {name: [] for name in arms}
    rows: list[dict[str, Any]] = []
    with MemoryStore(MemoryConfig(db_path=copy)) as store:
        service = MemoryRecallService(store)
        schemas = [
            node
            for node in store.list_nodes(level="schema", include_decayed=False, limit=100_000)
            if node.context.get("trigger")
        ]
        schemas.sort(key=lambda node: node.id)
        random.Random(7).shuffle(schemas)
        for index, schema in enumerate(schemas[:sample]):
            query = str(schema.context["trigger"])
            title = rpr.schema_title(schema.content)
            ranks: dict[str, int | None] = {}
            agree = False
            seen: dict[str, Any] = {"query": query, "schema": schema.id}
            for name in list(arms) if index % 2 == 0 else list(reversed(arms)):
                with rpr.applied_env(arms[name]):
                    started = time.perf_counter()
                    results = service.memory_recall(
                        query, scope=schema.scope, max_results=5, log_access=False, log_event=False
                    )
                    latency[name].append(time.perf_counter() - started)
                hits = [
                    position
                    for position, result in enumerate(results)
                    if result.node.id == schema.id
                    or (result.node.level == "schema" and rpr.schema_title(result.node.content) == title)
                ]
                ranks[name] = hits[0] if hits else None
                seen[f"{name}_rank"] = ranks[name]
                if name == "legacy" and hits:
                    hit = results[hits[0]]
                    agree = hit.vector_score >= 0.55 or hit.bm25_score >= 0.2
                    seen["vector"] = round(hit.vector_score, 4)
                    seen["bm25"] = round(hit.bm25_score, 4)
                if name == "name" and not hits:
                    ranked = [
                        result
                        for result in service.last_residual
                        if result.node.id == schema.id or rpr.schema_title(result.node.content) == title
                    ]
                    if ranked:
                        with rpr.applied_env(arms[name]):
                            seen["name_gate_score"] = round(gate_score(ranked[0]), 4)
            seen["agree"] = agree
            rows.append(seen)
            for name, rank in ranks.items():
                counts[name]["queries"] += 1
                counts[name]["rank1"] += rank == 0
                counts[name]["missing"] += rank is None
                counts[name]["agree"] += agree
                counts[name]["agree_rank1"] += agree and rank == 0
                if rank != 0:
                    misses[name].append(query)
    return {
        "host": host,
        "snapshot_sha256": rpr.sha256_file(snapshot),
        "sample": sample,
        "arms": arms,
        "summary": {
            name: {
                **{key: int(value) for key, value in sorted(counts[name].items())},
                "latency_mean_s": round(statistics.fmean(latency[name]), 4) if latency[name] else None,
                "not_rank1_queries": misses[name],
            }
            for name in arms
        },
        "queries": rows,
        "generated_at": rpr.utc_now(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("run")
    cmd.add_argument("--host", choices=rpr.HOSTS, required=True)
    cmd.add_argument("--segment", choices=("dev", "holdout"), required=True)
    cmd.add_argument("--snapshot", type=Path, required=True)
    cmd.add_argument("--split", type=Path, default=rpr.SPLIT_PATH)
    cmd.add_argument("--workdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "schema-trigger")
    cmd.add_argument("--sample", type=int)
    cmd.add_argument("--out", type=Path, required=True)
    cmd_names = sub.add_parser("names", help="L5: exact procedure-name queries")
    cmd_names.add_argument("--host", choices=rpr.HOSTS, required=True)
    cmd_names.add_argument("--snapshot", type=Path, required=True)
    cmd_names.add_argument("--workdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "schema-trigger")
    cmd_names.add_argument("--sample", type=int, default=150)
    cmd_names.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "names":
        result = names(args.host, args.snapshot, args.workdir, args.sample)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({name: {k: v for k, v in s.items() if k != "not_rank1_queries"}
                          for name, s in result["summary"].items()}, indent=1))
        return 0
    result = run(args.host, args.segment, args.snapshot, args.split, args.workdir, args.sample)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"host": result["host"], "segment": result["segment"], "events": result["events"],
                      "summary": result["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
