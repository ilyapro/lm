#!/usr/bin/env python3
"""Driver for the pre-registered recall precision measurement (prereg.md).

Reuses ``scripts/recall_precision_replay.py`` (snapshot, split, counterfactual
store, labels, train hubs, ``score_event``/``summarize``) and adds the
prereg metrics the harness does not compute:

* context chars per event of the shaped answer
  (``delivery.shape_recall_results``, fresh session, env defaults);
* ``rank1_removed``: the baseline's rank-1 node not delivered in full at rank 1;
* entrant quality: grounding-minus-twin over the baseline's full slots and
  over the arm's full slots not in the baseline list for the same event;
* ``irrelevant_full_cut`` / ``used_full_lost`` vs baseline per recorded-rank bucket.

Usage::

    python3 artifacts/recall-precision/runs/driver.py --host sfx --segment eval \\
        --arm gate_drop_040:LM_RECALL_MIN_SCORE=0.4,LM_RECALL_GATE_FORM=drop \\
        --out artifacts/recall-precision/runs/eval-sfx.json

The process env is expected to carry the production LM_* env (prereg.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import recall_precision_replay as rpr  # noqa: E402

PRODUCTION_ENV = {
    "LM_EXPLICIT_FEEDBACK_POLICY": "credit",
    "LM_EXPLICIT_FEEDBACK_PROMPT": "mandatory",
    "LM_IMPLICIT_LINK_POLICY": "credited",
    "LM_AUTO_CONSOLIDATE_POLICY": "adaptive",
    "LM_RETRIEVAL_TUNING_POLICY": "adaptive",
    "LM_RECALL_NEAR_DUP_COSINE": "0.97",
    "LM_DRAIN_NEAR_DUP_SUPERSEDES": "1",
    "LM_MAP_POOL_COLD_QUOTA_GATE": "1",
    "LM_MAP_POOL_COLD_SLOTS": "2",
    "LM_MAP_CURTAIL_DECAY": "1",
    "LM_DEFAULT_SCOPE": "global",
}
EXTRA_VALVES = ("LM_QUERY_IRRELEVANCE_FULL_COSINE", "LM_QUERY_IRRELEVANCE_MARK_WEIGHT")


def replay_with_chars(db_path, arms, events):  # type: ignore[no-untyped-def]
    """``rpr.replay`` plus the JSON chars of the shaped answer per event."""

    from living_memory.config import MemoryConfig
    from living_memory.delivery import (
        context_value_max_chars_from_env,
        full_node_diet_enabled_from_env,
        provenance_value_max_chars_from_env,
        shape_recall_results,
        snippet_ladder_from_env,
        snippet_max_chars_from_env,
        sparse_entries_enabled_from_env,
        stats_compaction_enabled_from_env,
    )
    from living_memory.retrieval import MemoryRecallService
    from living_memory.storage import MemoryStore

    class HorizonRecallService(MemoryRecallService):
        horizon: str | None = None
        hidden = 0

        def rank_candidates(self, candidates, plan, **kwargs):  # type: ignore[no-untyped-def]
            if self.horizon is not None:
                kept = {
                    key: candidate
                    for key, candidate in candidates.items()
                    if rpr.efa.normalize_ts(str(candidate.node.created_at)) <= self.horizon
                }
                self.hidden += len(candidates) - len(kept)
                candidates = kept
            return super().rank_candidates(candidates, plan, **kwargs)

    names = list(arms)
    stores = {name: MemoryStore(MemoryConfig(db_path=Path(db_path))) for name in names}
    lists: dict[str, list[list[rpr.Delivered]]] = {name: [] for name in names}
    chars: dict[str, list[int]] = {name: [] for name in names}
    latency: dict[str, list[float]] = {name: [] for name in names}
    try:
        services = {name: HorizonRecallService(stores[name]) for name in names}
        for index, event in enumerate(events):
            if index % 10 == 0:
                print(f"[driver] event {index}/{len(events)} {rpr.utc_now()}", file=sys.stderr, flush=True)
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
                        snippet_max_chars=snippet_max_chars_from_env(),
                        context_value_max_chars=context_value_max_chars_from_env(),
                        session_dedup=False,
                        snippet_ladder=snippet_ladder_from_env(),
                        full_node_diet=full_node_diet_enabled_from_env(),
                        provenance_value_max_chars=provenance_value_max_chars_from_env(),
                        stats_compaction=stats_compaction_enabled_from_env(),
                        sparse_entries=sparse_entries_enabled_from_env(),
                    )
                chars[name].append(len(json.dumps(shaped, ensure_ascii=False, default=str)))
                lists[name].append(
                    [
                        rpr.Delivered(
                            node_id=result.node_id,
                            full=not getattr(result, "withheld", None),
                            level=str(result.node.level),
                            title=rpr.schema_title(result.node.content) if str(result.node.level) == "schema" else "",
                            created_at=rpr.efa.normalize_ts(str(result.node.created_at)),
                        )
                        for result in results
                    ]
                )
        hidden = {name: services[name].hidden for name in names}
    finally:
        for store in stores.values():
            store.close()
    return lists, chars, latency, hidden


def excess(grades: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [g for g in grades if g.get("graded")]
    with_twin = [g for g in graded if g["has_twin"]]
    return {
        "graded": len(graded),
        "grounded_rate": rpr._ratio(sum(g["grounded"] for g in graded), len(graded)),
        "twin_grounded_rate": rpr._ratio(sum(g["twin_grounded"] for g in with_twin), len(with_twin)),
        "excess": rpr._ratio(sum(g["grounded"] - (g["twin_grounded"] if g["has_twin"] else 0) for g in graded), len(graded)),
    }


def bucket_vs_base(rows, base_rows) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    arm: dict[str, Counter[str]] = defaultdict(Counter)
    base: dict[str, Counter[str]] = defaultdict(Counter)
    for target, source in ((arm, rows), (base, base_rows)):
        for row in source:
            for bucket, counter in row["by_rank"].items():
                target[bucket].update(counter)
    out = {}
    for _, _, name in rpr.RANK_BUCKETS:
        if name not in base:
            continue
        b, a = base[name], arm[name]
        out[name] = {
            "base_irr_full": b["irr_full"],
            "irr_full_cut": rpr._ratio(b["irr_full"] - a["irr_full"], b["irr_full"]),
            "base_used_full": b["used_full"],
            "used_full_lost": rpr._ratio(b["used_full"] - a["used_full"], b["used_full"]),
        }
    return out


def run(
    host: str, segment: str, arms: dict[str, dict[str, str]], workdir: Path, grade: bool, limit: int | None = None
) -> dict[str, Any]:
    started = time.perf_counter()
    split = json.loads(rpr.SPLIT_PATH.read_text())["hosts"][host]
    snapshot = rpr.SNAPSHOT_DIR / f"{host}.sqlite3"
    if rpr.sha256_file(snapshot) != split["snapshot_sha256"]:
        raise SystemExit(f"{host}: snapshot sha256 differs from split.json")
    store = rpr.load_host_store(snapshot)
    events_all = [e for e in rpr.window_events(store) if rpr.segment_of(e, split) == segment]
    if limit:
        events_all = events_all[:limit]
    cutoff = rpr.cutoff_for(split, segment)
    workdir.mkdir(parents=True, exist_ok=True)
    counterfactual = rpr.build_counterfactual(snapshot, workdir / f"{host}-{segment}.sqlite3", cutoff)
    labels = rpr.event_labels(store)
    hubs = rpr.train_hubs(store, split)
    events = rpr.load_replay_events(snapshot, [e.id for e in events_all])
    lists, chars, latency, hidden = replay_with_chars(counterfactual["path"], arms, events)
    replayed = time.perf_counter()
    rows = {
        name: [
            rpr.score_event(event, delivered, labels.get(event.event_id, {}), hubs, cutoff, store.node_created)
            for event, delivered in zip(events, lists[name], strict=True)
        ]
        for name in arms
    }
    grader = rpr.EntrantGrader(snapshot, store.node_created) if grade else None
    base_rows = rows["baseline"]
    summaries: dict[str, Any] = {}
    try:
        base_grades: list[dict[str, Any]] = []
        if grader is not None:
            for event, row in zip(events, base_rows, strict=True):
                base_grades += grader.grade(event, row["full_ids"], row["full_ids"])
        for name in arms:
            summary = rpr.summarize(rows[name], None if name == "baseline" else base_rows)
            summary["chars_per_event"] = rpr._ratio(sum(chars[name]), len(events))
            summary["latency_s_total"] = round(sum(latency[name]), 2)
            summary["future_candidates_hidden"] = hidden[name]
            if name == "baseline":
                summary["delivered_set_quality"] = excess(base_grades)
            else:
                summary["rank1_removed"] = sum(
                    1
                    for row, ref, delivered in zip(rows[name], base_rows, lists[name], strict=True)
                    if ref["rank1"]
                    and not (delivered and delivered[0].node_id == ref["rank1"] and delivered[0].full)
                )
                summary["chars_saved_per_event"] = rpr._ratio(sum(chars["baseline"]) - sum(chars[name]), len(events))
                summary["chars_saved_share"] = rpr._ratio(sum(chars["baseline"]) - sum(chars[name]), sum(chars["baseline"]))
                summary["vs_baseline_by_rank"] = bucket_vs_base(rows[name], base_rows)
                new_grades: list[dict[str, Any]] = []
                if grader is not None:
                    for event, row, ref in zip(events, rows[name], base_rows, strict=True):
                        new = [n for n in row["full_ids"] if n not in set(ref["full_ids"])]
                        new_grades += grader.grade(event, row["full_ids"], new)
                summary["new_entrant_quality"] = excess(new_grades)
            summaries[name] = summary
    finally:
        if grader is not None:
            grader.close()
    return {
        "host": host,
        "segment": segment,
        "snapshot_sha256": split["snapshot_sha256"],
        "cutoff": cutoff,
        "events": len(events),
        "arms": arms,
        "hubs_train_defined": len(hubs),
        "counterfactual_stripped": counterfactual.get("stripped"),
        "summary": summaries,
        "runtime_s": round(time.perf_counter() - started, 1),
        "replay_s": round(replayed - started, 1),
        "generated_at": rpr.utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=rpr.HOSTS, required=True)
    parser.add_argument("--segment", choices=("eval", "holdout"), required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--no-grade", action="store_true")
    parser.add_argument("--limit", type=int, help="first N events only (driver smoke)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import os

    os.environ.update(PRODUCTION_ENV)
    for key in EXTRA_VALVES:
        os.environ.pop(key, None)
    arms: dict[str, dict[str, str]] = {"baseline": {}}
    for spec in args.arm:
        name, env = rpr.parse_arm(spec)
        arms[name] = env
    # Demotion strength valves are not in the harness's KNOWN_VALVES; make every
    # arm that does not set them run with them unset.
    for env in arms.values():
        for key in EXTRA_VALVES:
            env.setdefault(key, "")
    result = run(args.host, args.segment, arms, args.workdir, not args.no_grade, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({n: {k: s.get(k) for k in ("vs_baseline", "chars_per_event", "rank1_removed")} for n, s in result["summary"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
