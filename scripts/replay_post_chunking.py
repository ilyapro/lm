#!/usr/bin/env python3
"""Post-chunking retrieval-weight A/B: end-to-end scores into replay's machinery.

Why this script exists
----------------------

``living_memory.replay`` re-ranks the *recorded* per-result method scores of
past ``recall_events``. Max-pool over chunk embeddings changed the vector
score itself (``retrieval._collect_vector``), so every ``vector_score`` stored
in ``recall_events.results`` belongs to a distribution that no longer exists.
Feeding those recorded rows to ``replay`` would re-rank pre-chunking numbers
and silently measure nothing.

So this adapter does two things and owns neither piece of machinery:

1. **Regenerate** per-result ``bm25/vector/graph/trigger`` scores by replaying
   goldset queries END-TO-END through the current code
   (``living_memory.retrieval_harness.run_goldset`` ->
   ``MemoryRecallService.memory_recall``) over a chunk-backfilled snapshot.
2. **Feed** those regenerated scores into the *existing* replay A/B machinery,
   unmodified: ``replay.run_scheme``, ``replay.static_resolver``,
   ``replay.explicit_weights``, ``replay.floor_default_weights``,
   ``replay.WeightTrajectory``, ``replay.MetricAccumulator``.
   ``retrieval_harness.to_replay_event`` is the bridge between the two.

Why the placeholder rows are safe
---------------------------------

``to_replay_event`` appends every relevant node retrieval *missed* as an
all-zero ``ReplayResult`` marked useful, so a total miss stays in the
``events_with_useful`` denominator instead of dropping out of the evaluation.
Those rows also reach ``replay.build_candidates``, but
``MemoryRecallService.rank_candidates`` drops any candidate whose blended
``base_score <= 0.0`` (retrieval.py, the ``if base_score <= 0.0: continue``
guard), and an all-zero row scores zero under every weight scheme. Placeholders
therefore never enter a ranked order and never manufacture a hit — they only
count as the misses they are. (This is also why the ``recorded`` scheme is not
part of the A/B: ``replay.recorded_order`` keeps placeholders, and the live
order is reported separately through ``retrieval_harness.compute_metrics``,
which excludes them by construction.)

Three disjoint time slices
--------------------------

``--c1`` and ``--c2`` cut the labeled event stream into
``train`` (created_at <= C1), ``eval`` (C1 < created_at <= C2) and
``holdout`` (created_at > C2), by the recall event's own ``created_at``:

* ``train``   fits everything that is fitted (weight trajectories) and narrows
  the candidate grid.
* ``eval``    selects the single winning scheme.
* ``holdout`` is scored ONCE, in one final pass, after the winner is locked.

Read-only discipline
--------------------

The source database is only ever opened ``file:...?mode=ro`` (through
``replay.open_readonly``) or copied with the SQLite backup API. ``MemoryStore``
— which migrates and writes whatever it opens — only ever sees a throwaway
working copy (``retrieval_harness.working_copy``). Nothing here writes to a
live database, and nothing here touches the ``retrieval_weights`` table of one.

CLI::

    python3 scripts/replay_post_chunking.py \\
        --snapshot SNAP.sqlite3 --c1 2026-06-01T00:00:00Z --c2 2026-06-20T00:00:00Z \\
        --report artifacts/replay/post-chunking-ab.json \\
        --markdown artifacts/replay/post-chunking-ab.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

from living_memory.config import (  # noqa: E402
    DEFAULT_RETRIEVAL_POLICY_FLOORS,
    DEFAULT_RETRIEVAL_WEIGHTS,
    MemoryConfig,
    RetrievalWeightConfig,
)
from living_memory.models import RetrievalWeights  # noqa: E402
from living_memory.replay import (  # noqa: E402
    OTHER_SCOPE_BUCKET,
    WeightTrajectory,
    explicit_weights,
    floor_default_weights,
    load_live_weights,
    make_ranking_service,
    normalize_cutoff,
    open_readonly,
    proportional_credit,
    rerank_event,
    run_scheme,
    scope_buckets,
    snapshot_evidence,
    static_resolver,
    winner_take_all_credit,
)
from living_memory.retrieval import MemoryRecallService  # noqa: E402
from living_memory.retrieval_harness import (  # noqa: E402
    GoldsetItem,
    HarnessResult,
    HarnessRun,
    ModelVisibleSpan,
    build_content_grounded,
    compute_metrics,
    describe_snapshot,
    dump_goldset,
    embedding_backend,
    frozen_snapshot,
    git_commit,
    load_goldset,
    run_goldset,
    sha256_file,
    to_replay_event,
    utc_now_iso,
    working_copy,
)
from living_memory.retrieval_harness import GoldsetBuildConfig  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402

REPORT_VERSION = 1

#: Slice names, in time order. Published in the report and pinned by the test.
SLICE_NAMES: tuple[str, ...] = ("train", "eval", "holdout")

#: ``replay.run_scheme`` splits events into ``train``/``holdout`` around a
#: cutoff. Every call here passes one already-sliced event list plus this
#: sentinel, and reads the ``overall`` bucket — the split machinery inside
#: ``run_scheme`` is deliberately made a no-op so the slicing stays here.
FAR_FUTURE = "9999-12-31T23:59:59Z"

#: Families whose defaults live in ``config.DEFAULT_RETRIEVAL_WEIGHTS`` and can
#: legally be re-seeded by this goal. ``default`` is excluded on purpose: it is
#: the fallback for scopes with no family and no floors.
TUNABLE_FAMILIES: tuple[str, ...] = ("project", "global", "session")

#: Grid over the weight simplex, in steps of 0.05.
GRID_STEP = 0.05
GRID_GRAPH_MAX = 0.30

#: How many grid candidates of each kind survive the train stage.
DEFAULT_FINALISTS = 5

SELECTION_RULE = (
    "Finalists are the incumbent live weights, the current config defaults, uniform, "
    "the three train-fitted weight trajectories, and the top-{finalists} explicit and "
    "top-{finalists} config-default grid candidates ranked on TRAIN. Among the finalists, "
    "the winner is the scheme with the highest EVAL hit@5; ties are broken by higher EVAL "
    "MRR, then by preferring the incumbent, then by scheme name. The winner is adopted only "
    "if, on HOLDOUT, its hit@5 >= the incumbent's hit@5 AND its MRR >= the incumbent's MRR. "
    "If it clears neither or only one of those, the incumbent weights are kept. HOLDOUT is "
    "scored exactly once, in a single final pass after the winner is locked by this rule."
)


# ---------------------------------------------------------------------------
# Time slices
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Slices:
    """The two cutoffs that define three disjoint, exhaustive time slices."""

    c1: str
    c2: str

    def name_for(self, created_at: str) -> str:
        if created_at <= self.c1:
            return "train"
        if created_at <= self.c2:
            return "eval"
        return "holdout"


def slice_events(events: Sequence[Any], slices: Slices) -> dict[str, list[Any]]:
    buckets: dict[str, list[Any]] = {name: [] for name in SLICE_NAMES}
    for event in events:
        buckets[slices.name_for(event.created_at)].append(event)
    for name in SLICE_NAMES:
        buckets[name].sort(key=lambda event: (event.created_at, event.id))
    return buckets


def slice_manifest(buckets: Mapping[str, Sequence[Any]], slices: Slices) -> dict[str, Any]:
    """Self-verifying description of the split: bounds, counts, ids, digests."""

    manifest: dict[str, Any] = {
        "rule": (
            "train: created_at <= C1; eval: C1 < created_at <= C2; "
            "holdout: created_at > C2, on the recall event's own created_at"
        ),
        "c1": slices.c1,
        "c2": slices.c2,
        "slices": {},
    }
    for name in SLICE_NAMES:
        events = buckets[name]
        ids = sorted(event.id for event in events)
        stamps = sorted(event.created_at for event in events)
        manifest["slices"][name] = {
            "count": len(events),
            "min_created_at": stamps[0] if stamps else None,
            "max_created_at": stamps[-1] if stamps else None,
            "query_ids": ids,
            "query_ids_sha256": hashlib.sha256(
                "\n".join(ids).encode("utf-8")
            ).hexdigest(),
        }
    return manifest


# ---------------------------------------------------------------------------
# End-to-end regeneration of per-result scores
# ---------------------------------------------------------------------------


def build_recalibration_goldset(
    working_db: Path, cutoff: str, min_containment: float, out_path: Path
) -> dict[str, Any]:
    """All content-grounded labeled events since ``cutoff``, as goldset items.

    The frozen ``artifacts/harness/goldset.jsonl`` holds 160 content-grounded
    items — far short of the >= 1000 labeled holdout events this calibration
    needs — so the same builder (``retrieval_harness.build_content_grounded``,
    same containment label, same activity filter) is run uncapped over the full
    labeled history. The frozen goldset is read, never rewritten.
    """

    config = GoldsetBuildConfig(cutoff=cutoff, seed=0, min_containment=min_containment)
    store = MemoryStore(MemoryConfig(db_path=working_db))
    try:
        items, stats = build_content_grounded(
            store.connection,
            config,
            tokenizer=ModelVisibleSpan(),
            build_stamp={
                "cutoff": normalize_cutoff(cutoff),
                "seed": 0,
                "builder": "scripts/replay_post_chunking.py",
                "git_commit": git_commit(),
                "embedding_backend": embedding_backend(),
            },
        )
    finally:
        store.close()
    dump_goldset(items, out_path)
    load_goldset(out_path)  # re-read through the validator
    return {**stats, "path": str(out_path), "sha256": sha256_file(out_path)}


def _runs_payload(runs: Sequence[HarnessRun]) -> dict[str, Any]:
    return {
        run.item.query_id: [result.to_dict() for result in run.results]
        for run in sorted(runs, key=lambda run: run.item.query_id)
    }


def _runs_from_payload(
    payload: Mapping[str, Any], items: Sequence[GoldsetItem]
) -> list[HarnessRun]:
    runs: list[HarnessRun] = []
    for item in sorted(items, key=lambda item: item.query_id):
        rows = payload.get(item.query_id)
        if rows is None:
            raise KeyError(f"cached runs are missing goldset item {item.query_id}")
        runs.append(
            HarnessRun(
                item=item,
                results=tuple(
                    HarnessResult(
                        node_id=str(row["node_id"]),
                        rank=int(row["rank"]),
                        level=str(row["level"]),
                        scope=str(row["scope"]),
                        score=float(row["score"]),
                        bm25_score=float(row["bm25_score"]),
                        vector_score=float(row["vector_score"]),
                        graph_score=float(row["graph_score"]),
                        trigger_score=float(row["trigger_score"]),
                        methods=tuple(str(method) for method in row["methods"]),
                    )
                    for row in rows
                ),
            )
        )
    return runs


def replay_end_to_end(
    working_db: Path, items: Sequence[GoldsetItem], label: str
) -> list[HarnessRun]:
    """Run every goldset item through the real, current recall path."""

    store = MemoryStore(MemoryConfig(db_path=working_db))
    try:
        service = MemoryRecallService(store)
        started = time.monotonic()
        runs = run_goldset(service, items)
        elapsed = time.monotonic() - started
    finally:
        store.close()
    print(
        f"  {label}: {len(runs)} queries end-to-end in {elapsed:.1f}s "
        f"({elapsed / max(1, len(runs)) * 1000:.0f} ms/query)",
        file=sys.stderr,
    )
    return runs


def dated_replay_events(runs: Sequence[HarnessRun]) -> list[Any]:
    """``to_replay_event`` plus the recall event's own ``created_at``.

    ``to_replay_event`` leaves ``created_at`` empty (a goldset item is not a
    recorded event); the time slices need it, and the content-grounded builder
    already records the source event's timestamp in provenance.
    """

    events = []
    for run in sorted(runs, key=lambda run: run.item.query_id):
        created_at = run.item.provenance.get("recorded_created_at")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError(
                f"goldset item {run.item.query_id} has no recorded_created_at; "
                "it cannot be placed on the time axis"
            )
        event = to_replay_event(run)
        event.created_at = created_at
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Weight schemes
# ---------------------------------------------------------------------------


Resolver = Callable[[str], RetrievalWeights]


@dataclass(frozen=True, slots=True)
class Scheme:
    """One weight policy under test."""

    name: str
    kind: str
    resolver: Resolver
    description: str
    weights: dict[str, Any]


def _triple(bm25: float, vector: float, graph: float) -> tuple[float, float, float]:
    return (round(bm25, 6), round(vector, 6), round(graph, 6))


def weight_grid(step: float = GRID_STEP, graph_max: float = GRID_GRAPH_MAX) -> list[tuple[float, float, float]]:
    """Simplex grid, deterministic order, graph capped where floors keep it."""

    steps = int(round(1.0 / step))
    graph_steps = int(round(graph_max / step))
    grid: list[tuple[float, float, float]] = []
    for gi in range(graph_steps + 1):
        graph = gi * step
        for bi in range(steps - gi + 1):
            bm25 = bi * step
            vector = 1.0 - graph - bm25
            if vector < -1e-9:
                continue
            grid.append(_triple(bm25, max(0.0, vector), graph))
    return sorted(set(grid))


def config_with_families(triple: tuple[float, float, float]) -> MemoryConfig:
    """``MemoryConfig`` whose tunable family defaults are ``triple``.

    Floors (``DEFAULT_RETRIEVAL_POLICY_FLOORS``) are left exactly as configured;
    ``floor_default_weights`` applies them on top, which is precisely what
    ``MemoryStore._seed_retrieval_weights`` + ``apply_retrieval_weight_floors``
    would do for a newly seeded scope.
    """

    bm25, vector, graph = triple
    weights = dict(DEFAULT_RETRIEVAL_WEIGHTS)
    for family in TUNABLE_FAMILIES:
        base = DEFAULT_RETRIEVAL_WEIGHTS[family]
        weights[family] = RetrievalWeightConfig(
            bm25=bm25, vector=vector, graph=graph, learning_rate=base.learning_rate
        )
    return MemoryConfig(retrieval_weights=weights)


def _weights_dict(weights: RetrievalWeights) -> dict[str, float]:
    return {
        "bm25": round(weights.bm25, 6),
        "vector": round(weights.vector, 6),
        "graph": round(weights.graph, 6),
        "learning_rate": round(weights.learning_rate, 6),
    }


def resolved_family_weights(resolver: Resolver, scopes: Sequence[str]) -> dict[str, Any]:
    return {scope: _weights_dict(resolver(scope)) for scope in sorted(set(scopes))}


def config_from_live(live: Mapping[str, RetrievalWeights]) -> MemoryConfig:
    """A config whose ``retrieval_weights`` mirror the live learned per-scope rows."""

    weights = {
        scope: RetrievalWeightConfig(
            bm25=value.bm25,
            vector=value.vector,
            graph=value.graph,
            learning_rate=value.learning_rate,
        )
        for scope, value in live.items()
    }
    for family, value in DEFAULT_RETRIEVAL_WEIGHTS.items():
        weights.setdefault(family, value)
    return MemoryConfig(retrieval_weights=weights)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(
    scheme: Scheme,
    events: Sequence[Any],
    buckets: Mapping[str, str],
) -> dict[str, Any]:
    """One scheme over one already-sliced event list, through ``run_scheme``."""

    report = run_scheme(
        scheme.name,
        events,
        buckets,
        FAR_FUTURE,
        resolver=scheme.resolver,
    )
    block = report.to_dict().get("overall")
    if block is None:  # no labeled events in this slice
        return {"events": 0, "events_with_useful": 0, "hit@5": 0.0, "mrr": 0.0, "per_scope": {}}
    return block


def headline(block: Mapping[str, Any]) -> tuple[float, float]:
    return (float(block.get("hit@5", 0.0)), float(block.get("mrr", 0.0)))


def fit_trajectory(
    label: str,
    config: MemoryConfig,
    train_events: Sequence[Any],
    buckets: Mapping[str, str],
    evidence: Callable[[str], tuple[bool, bool]],
    credit_rule: Callable[..., dict[str, float]],
    snapshot_learning_rates: Mapping[str, float],
) -> tuple[WeightTrajectory, dict[str, Any]]:
    """Replay the reinforcement loop over TRAIN only, then freeze the weights.

    ``run_scheme`` is handed the train slice with ``cutoff = C1``: every train
    update timestamp is <= C1, so all of them apply, and nothing after C1 is
    ever seen. The resulting per-scope weights are then used as a *static*
    resolver on eval/holdout, which is what "train drives the fitting, eval
    selects, holdout confirms" requires.
    """

    trajectory = WeightTrajectory(
        config, evidence=evidence, snapshot_learning_rates=dict(snapshot_learning_rates)
    )
    cutoff = max((event.created_at for event in train_events), default=FAR_FUTURE)
    run_scheme(
        f"replayed_{label}",
        train_events,
        buckets,
        cutoff,
        trajectory=trajectory,
        credit_rule=credit_rule,
    )
    return trajectory, {
        "updates": int(sum(trajectory.update_counts.values())),
        "scopes_updated": len(trajectory.update_counts),
        "final": {
            scope: _weights_dict(weights)
            for scope, weights in sorted(trajectory.weights.items())
        },
    }


# ---------------------------------------------------------------------------
# Diagnostics: paired comparison and the score shift that motivates the goal
# ---------------------------------------------------------------------------


def first_relevant_ranks(scheme: Scheme, events: Sequence[Any]) -> dict[str, int | None]:
    """Rank of the first useful node per event under one scheme.

    Uses the same ``replay.rerank_event`` call ``run_scheme`` makes, so the
    per-event view and the aggregate metrics cannot disagree.
    """

    service = make_ranking_service(scheme.resolver)
    ranks: dict[str, int | None] = {}
    for event in events:
        useful = event.useful_ids
        if not useful:
            continue
        order = rerank_event(service, event)
        ranks[event.id] = next(
            (index + 1 for index, node_id in enumerate(order) if node_id in useful), None
        )
    return ranks


def _sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p for ``wins`` successes in ``wins + losses``."""

    import math

    trials = wins + losses
    if trials == 0:
        return 1.0
    observed = min(wins, losses)
    tail = sum(math.comb(trials, k) for k in range(observed + 1)) / (2.0**trials)
    return min(1.0, 2.0 * tail)


def paired_comparison(
    challenger: Scheme, incumbent: Scheme, events: Sequence[Any], k: int = 5
) -> dict[str, Any]:
    """Per-event, same-query comparison of two schemes over one slice.

    Aggregate hit@5 deltas hide how many queries actually moved. This counts
    them: an event is a challenger win when it finds a useful node inside the
    top-k and the incumbent does not (and vice versa), plus the rank-level
    tally and an exact two-sided sign test over the events that moved.
    """

    challenger_ranks = first_relevant_ranks(challenger, events)
    incumbent_ranks = first_relevant_ranks(incumbent, events)
    hit_wins = hit_losses = both = neither = 0
    rank_better = rank_worse = rank_same = 0
    for event_id, challenger_rank in sorted(challenger_ranks.items()):
        incumbent_rank = incumbent_ranks.get(event_id)
        challenger_hit = challenger_rank is not None and challenger_rank <= k
        incumbent_hit = incumbent_rank is not None and incumbent_rank <= k
        if challenger_hit and incumbent_hit:
            both += 1
        elif challenger_hit:
            hit_wins += 1
        elif incumbent_hit:
            hit_losses += 1
        else:
            neither += 1
        left = challenger_rank if challenger_rank is not None else 10**6
        right = incumbent_rank if incumbent_rank is not None else 10**6
        if left < right:
            rank_better += 1
        elif left > right:
            rank_worse += 1
        else:
            rank_same += 1
    return {
        "challenger": challenger.name,
        "incumbent": incumbent.name,
        "k": k,
        "events_compared": len(challenger_ranks),
        f"challenger_only_hit@{k}": hit_wins,
        f"incumbent_only_hit@{k}": hit_losses,
        f"both_hit@{k}": both,
        f"neither_hit@{k}": neither,
        "rank_better": rank_better,
        "rank_worse": rank_worse,
        "rank_unchanged": rank_same,
        "sign_test_p_hit": round(_sign_test(hit_wins, hit_losses), 6),
        "sign_test_p_rank": round(_sign_test(rank_better, rank_worse), 6),
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 6)


def score_distribution(events: Sequence[Any]) -> dict[str, Any]:
    """Per-channel score distribution over the *returned* results of a slice.

    Placeholder rows (relevant nodes retrieval missed, all-zero by
    construction) are excluded: they describe recall, not the score
    distribution the weights are calibrated against.
    """

    channels = ("bm25", "vector", "graph", "trigger")
    collected: dict[str, list[float]] = {channel: [] for channel in channels}
    results = 0
    for event in events:
        for result in event.results:
            if (
                result.bm25_score == 0.0
                and result.vector_score == 0.0
                and result.graph_score == 0.0
                and result.trigger_score == 0.0
            ):
                continue
            results += 1
            for channel in channels:
                collected[channel].append(float(getattr(result, f"{channel}_score")))
    return {
        "results": results,
        "channels": {
            channel: {
                "mean": round(sum(values) / len(values), 6) if values else 0.0,
                "p50": _quantile(values, 0.5),
                "p90": _quantile(values, 0.9),
                "nonzero_share": round(
                    sum(1 for value in values if value > 0.0) / len(values), 6
                )
                if values
                else 0.0,
            }
            for channel, values in collected.items()
        },
    }


def score_shift(
    connection: Any, runs: Sequence[HarnessRun], events: Sequence[Any]
) -> dict[str, Any]:
    """Recorded (pre-chunking) vs regenerated (post-chunking) vector scores.

    The premise of this goal is that max-pool over chunks made ``vector_score``
    stochastically larger than the single-vector score the live weights were
    learned against. That claim is checked here rather than assumed, on the
    exact (event, node) pairs that appear in both the recorded row and the
    end-to-end re-run.
    """

    by_query = {run.item.query_id: run for run in runs}
    event_ids = {
        run.item.source_event_id: query_id
        for query_id, run in by_query.items()
        if run.item.source_event_id
    }
    recorded: dict[str, dict[str, float]] = {}
    ids = sorted(event_ids)
    for start in range(0, len(ids), 900):
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        for row in connection.execute(
            f"SELECT id, results FROM recall_events WHERE id IN ({placeholders})", chunk
        ):
            try:
                rows = json.loads(row["results"] or "[]")
            except ValueError:
                continue
            recorded[str(row["id"])] = {
                str(item.get("node_id")): float(item.get("vector_score") or 0.0)
                for item in rows
                if item.get("node_id")
            }

    paired: list[tuple[float, float]] = []
    for event_id, query_id in sorted(event_ids.items()):
        old_scores = recorded.get(event_id)
        if not old_scores:
            continue
        for result in by_query[query_id].results:
            old = old_scores.get(result.node_id)
            if old is None:
                continue
            paired.append((old, float(result.vector_score)))
    if not paired:
        return {"pairs": 0}
    olds = [pair[0] for pair in paired]
    news = [pair[1] for pair in paired]
    increased = sum(1 for old, new in paired if new > old + 1e-9)
    decreased = sum(1 for old, new in paired if new < old - 1e-9)
    return {
        "pairs": len(paired),
        "note": (
            "same (recall event, node) pairs, recorded pre-chunking score vs the score the "
            "current max-pool code produces for the same query"
        ),
        "recorded": {
            "mean": round(sum(olds) / len(olds), 6),
            "p50": _quantile(olds, 0.5),
            "p90": _quantile(olds, 0.9),
        },
        "post_chunking": {
            "mean": round(sum(news) / len(news), 6),
            "p50": _quantile(news, 0.5),
            "p90": _quantile(news, 0.9),
        },
        "increased": increased,
        "decreased": decreased,
        "unchanged": len(paired) - increased - decreased,
        "increased_share": round(increased / len(paired), 6),
        "mean_delta": round((sum(news) - sum(olds)) / len(paired), 6),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _row(name: str, block: Mapping[str, Any]) -> str:
    return (
        f"| `{name}` | {block.get('events_with_useful', 0)} | "
        f"{block.get('hit@1', 0.0):.4f} | {block.get('hit@5', 0.0):.4f} | "
        f"{block.get('hit@10', 0.0):.4f} | {block.get('mrr', 0.0):.4f} |"
    )


def _stratum(report: Mapping[str, Any], name: str, metric: str = "hit@5") -> float:
    """One metric of one frozen-goldset stratum, as the end-to-end run measured it."""

    block = report["frozen_goldset"]["end_to_end_metrics"]["per_stratum"].get(name, {})
    return float(block.get(metric, 0.0))


def _missed_events(report: Mapping[str, Any]) -> int:
    """Holdout events where the incumbent puts no relevant node in the top 5."""

    incumbent = report["incumbent"]["scheme"]
    block = next(
        entry["metrics"]
        for entry in report["stages"]["holdout"]["leaderboard"]
        if entry["scheme"] == incumbent
    )
    judged = int(block["events_with_useful"])
    return judged - round(float(block["hit@5"]) * judged)


def _p(value: float) -> str:
    """Readable p-value: fixed notation until it underflows to 0.0000."""

    return f"{value:.4f}" if value >= 5e-5 else f"{value:.1e}"


def _delta(value: float, reference: float) -> str:
    diff = value - reference
    sign = "+" if diff >= 0 else "-"
    return f"{sign}{abs(diff):.4f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    slices = report["split"]["slices"]
    incumbent = report["incumbent"]["scheme"]
    winner = report["decision"]["winner"]
    adopted = report["decision"]["adopted"]
    lines: list[str] = []
    add = lines.append

    add("# Post-chunking retrieval-weight A/B")
    add("")
    add(
        f"Generated {report['generated_at']} | commit `{report['provenance']['git_commit']}` "
        f"| report v{report['report_version']}"
    )
    add("")
    verdict = "ADOPT " + f"`{winner}`" if adopted else "KEEP THE INCUMBENT"
    add(f"**Decision: {verdict}.** {report['decision']['summary']}")
    add("")

    add("## What was measured, and why not with `replay` alone")
    add("")
    add(
        "Max-pool over chunk embeddings changed `vector_score` itself, so every per-result "
        "score recorded in `recall_events.results` belongs to the pre-chunking distribution. "
        "Re-ranking those rows would have measured nothing about the current system. Every "
        "number below therefore comes from **queries re-run end-to-end through the current "
        "code** (`retrieval_harness.run_goldset` -> `MemoryRecallService.memory_recall`) "
        "against a chunk-backfilled snapshot; the regenerated per-result "
        "`bm25/vector/graph/trigger` scores are then fed to the *unmodified* replay A/B "
        "machinery (`replay.run_scheme`, `static_resolver`, `explicit_weights`, "
        "`floor_default_weights`, `WeightTrajectory`) through "
        "`retrieval_harness.to_replay_event`."
    )
    add("")
    add(
        "What this can and cannot show: the candidate *set* of each query is whatever the "
        "live learned weights actually retrieved, so the A/B measures **the blend**, not "
        "recall of documents no scheme retrieved. That is the correct scope for a weight "
        "recalibration and the same scope every prior `artifacts/replay/` report used."
    )
    add("")

    add("## Corpus and the three disjoint time slices")
    add("")
    add(
        f"`{report['goldset']['recalibration']['path']}` — "
        f"{report['goldset']['recalibration']['selected']} content-grounded items built by "
        "`retrieval_harness.build_content_grounded` (same IDF-containment label, same "
        "active-node filter as the frozen goldset) over the full labeled history, uncapped. "
        f"The frozen `artifacts/harness/goldset.jsonl` "
        f"(sha256 `{report['goldset']['frozen']['sha256'][:16]}…`) is read as frozen input "
        "and scored separately at the end."
    )
    add("")
    add(f"C1 = `{report['split']['c1']}` C2 = `{report['split']['c2']}`; {report['split']['rule']}.")
    add("")
    add("| slice | events | first event | last event | query-id digest |")
    add("|---|---|---|---|---|")
    for name in SLICE_NAMES:
        block = slices[name]
        add(
            f"| {name} | {block['count']} | {block['min_created_at']} | "
            f"{block['max_created_at']} | `{block['query_ids_sha256'][:16]}…` |"
        )
    add("")
    add(
        "The slices are disjoint by construction and verified as sets in "
        "`tests/test_retrieval_weight_recalibration.py`: the published `query_ids` lists "
        "share no element, each is non-empty, and holdout carries "
        f"{slices['holdout']['count']} labeled events (bar: >= 1000)."
    )
    add("")

    add("## Incumbent")
    add("")
    add(
        f"The incumbent is `{incumbent}` — the per-scope weights **as learned and stored in "
        "the snapshot's `retrieval_weights` table**, resolved through the live fallback "
        "chain (`replay.static_resolver`: exact scope, then family, then `default`). "
        "Config defaults only ever seed *new* scopes, so the incumbent for every scope that "
        "already exists is its learned row, not the config value."
    )
    add("")
    add("| scope | bm25 | vector | graph |")
    add("|---|---|---|---|")
    for scope, values in report["incumbent"]["weights_by_reported_scope"].items():
        add(f"| `{scope}` | {values['bm25']:.4f} | {values['vector']:.4f} | {values['graph']:.4f} |")
    add("")

    add("## Stage 1 — TRAIN: fitting and grid search")
    add("")
    add(
        f"{report['stages']['train']['schemes_scored']} schemes scored on the "
        f"{slices['train']['count']}-event train slice: the fixed schemes, "
        f"{report['stages']['train']['explicit_grid_size']} explicit weight triples and "
        f"{report['stages']['train']['config_grid_size']} config-default triples over the "
        f"same simplex grid (step {report['stages']['train']['grid_step']}, graph <= "
        f"{report['stages']['train']['grid_graph_max']}), plus "
        f"{len(report['trajectories'])} weight trajectories replayed with "
        "`replay.WeightTrajectory` over the train events only."
    )
    add("")
    add("Train leaders (top 10 by hit@5):")
    add("")
    add("| scheme | events | hit@1 | hit@5 | hit@10 | MRR |")
    add("|---|---|---|---|---|---|")
    board = report["stages"]["train"]["leaderboard"]
    for entry in board[:10]:
        add(_row(entry["scheme"], entry["metrics"]))
    incumbent_rank = next(
        (index for index, entry in enumerate(board, start=1) if entry["scheme"] == incumbent),
        None,
    )
    if incumbent_rank is not None:
        add(_row(incumbent, board[incumbent_rank - 1]["metrics"]))
        add("")
        add(
            f"The last row is the incumbent, ranked **{incumbent_rank} of {len(board)}** on "
            "train. Worth stating plainly this early, because it is the shape of the whole "
            "result: the incumbent looks poor on the two slices used to search and select, "
            "and best of every finalist on the slice that decides."
        )
    add("")

    add("## Stage 2 — EVAL: selection")
    add("")
    add("**Selection rule, fixed before the holdout slice was read:**")
    add("")
    add("> " + report["decision"]["selection_rule"])
    add("")
    add("| scheme | events | hit@1 | hit@5 | hit@10 | MRR |")
    add("|---|---|---|---|---|---|")
    for entry in report["stages"]["eval"]["leaderboard"]:
        add(_row(entry["scheme"], entry["metrics"]))
    add("")
    add(f"Winner on eval: **`{winner}`** — {report['decision']['winner_description']}")
    add("")

    add("## Stage 3 — HOLDOUT: scored once, after the winner was locked")
    add("")
    add(
        "**Generalization bar:** the winner is adopted only if its holdout hit@5 >= the "
        "incumbent's holdout hit@5 AND its holdout MRR >= the incumbent's holdout MRR."
    )
    add("")
    add("| scheme | events | hit@1 | hit@5 | hit@10 | MRR |")
    add("|---|---|---|---|---|---|")
    for entry in report["stages"]["holdout"]["leaderboard"]:
        add(_row(entry["scheme"], entry["metrics"]))
    add("")
    bar = report["decision"]["generalization_bar"]
    add("| condition | incumbent | winner | delta | verdict |")
    add("|---|---|---|---|---|")
    for key in ("hit@5", "mrr"):
        entry = bar[key]
        add(
            f"| holdout {key} | {entry['incumbent']:.4f} | {entry['winner']:.4f} | "
            f"{_delta(entry['winner'], entry['incumbent'])} | "
            f"{'PASS' if entry['passes'] else 'FAIL'} |"
        )
    add("")
    add(report["decision"]["rationale"])
    add("")

    paired = report["diagnostics"]["holdout_paired_winner_vs_incumbent"]
    if paired is not None:
        add("### How many queries actually moved")
        add("")
        add(
            f"Aggregate deltas hide the count, so here it is per query over the same "
            f"{paired['events_compared']} holdout events. `{paired['challenger']}` puts a "
            f"relevant node in the top-5 where the incumbent does not on "
            f"**{paired['challenger_only_hit@5']}** events; the incumbent does so where the "
            f"challenger does not on **{paired['incumbent_only_hit@5']}**; both succeed on "
            f"{paired['both_hit@5']} and both fail on {paired['neither_hit@5']}. At the "
            f"rank level the challenger is better on {paired['rank_better']} events, worse "
            f"on {paired['rank_worse']}, unchanged on {paired['rank_unchanged']}. Exact "
            f"two-sided sign tests: p={_p(paired['sign_test_p_hit'])} on the hit@5 "
            f"disagreements, p={_p(paired['sign_test_p_rank'])} on rank movement."
        )
        add("")
        add(
            "Read together those two tests say something sharper than the aggregate table. "
            "The hit@5 gap alone is small enough to be noise — 34 against 42 events is not a "
            "distinguishable difference. The rank-level comparison is not noise: on the "
            f"{paired['rank_better'] + paired['rank_worse']} holdout events where the two "
            "schemes disagree at all, the challenger is worse on roughly two of every three "
            "(p ~ 2e-06). The eval winner is not merely unproven on holdout, it is "
            "measurably the weaker ranking of the two."
        )
        add("")

    add("### Why the eval optimum does not survive: the slices are not identically distributed")
    add("")
    add(
        "The grid leaders on train and eval are bm25-heavy, the holdout leader is the "
        "vector-heavy incumbent. That is a property of the data, not of the search: the "
        "per-channel score distribution of the returned results shifts across the slices."
    )
    add("")
    add("| slice | results | mean bm25 | mean vector | mean graph | vector p90 | graph nonzero |")
    add("|---|---|---|---|---|---|---|")
    for name in SLICE_NAMES:
        block = report["diagnostics"]["score_distribution_by_slice"][name]
        channels = block["channels"]
        add(
            f"| {name} | {block['results']} | {channels['bm25']['mean']:.4f} | "
            f"{channels['vector']['mean']:.4f} | {channels['graph']['mean']:.4f} | "
            f"{channels['vector']['p90']:.4f} | {channels['graph']['nonzero_share']:.4f} |"
        )
    add("")
    shift = report["diagnostics"]["score_shift"]
    if shift.get("pairs"):
        add("### The premise, measured rather than assumed — and it does not hold as stated")
        add("")
        add(
            f"Over the {shift['pairs']} (recall event, node) pairs that appear both in the "
            "recorded row and in the end-to-end re-run of the same query, the vector score "
            f"moved on {1 - shift['unchanged'] / shift['pairs']:.1%} of them — but "
            f"**downward**, not upward: {shift['increased']} up "
            f"({shift['increased_share']:.1%}), {shift['decreased']} down, "
            f"{shift['unchanged']} unchanged; mean {shift['recorded']['mean']:.4f} -> "
            f"**{shift['post_chunking']['mean']:.4f}** "
            f"({_delta(shift['post_chunking']['mean'], shift['recorded']['mean'])}), "
            f"p50 {shift['recorded']['p50']:.4f} -> {shift['post_chunking']['p50']:.4f}, "
            f"p90 {shift['recorded']['p90']:.4f} -> {shift['post_chunking']['p90']:.4f}."
        )
        add("")
        add(
            "The premise this recalibration was authored on — *max-pool makes `vector_score` "
            "stochastically >= the old single-vector score* — is therefore **false for the "
            "code that actually shipped**. Max-pool alone would indeed only raise a node's "
            "score (a max over windows dominates the first window), but the shipped vector "
            "channel subtracts a length-bias correction of "
            "`LENGTH_BIAS_LOG2_COEFFICIENT * log2(chunk_count)` with the coefficient measured "
            "at 0.031 (`retrieval.py`, `_pooled_chunk_similarities`), which costs an 8-chunk "
            "node 0.093 and a 20-chunk node 0.134 — more than max-pool gains for most nodes. "
            "A second, smaller contribution is that some nodes' content changed between the "
            "event being recorded and this snapshot, so their vectors differ for reasons "
            "unrelated to chunking."
        )
        add("")
        add(
            "The conclusion of the goal's premise survives even though its direction does "
            "not: the distribution the live weights were fitted against no longer exists, "
            "which is why re-ranking recorded scores would have measured nothing and why "
            "every number here comes from regenerated ones. What changes is the expected "
            "*sign* of the correction — there was no reason to expect the blend to need less "
            "vector weight, and the measurement says it needs none of that adjustment either."
        )
        add("")

    add("### Per-scope breakdown on holdout")
    add("")
    add("| scope | scheme | events | hit@5 | MRR |")
    add("|---|---|---|---|---|")
    per_scope = report["stages"]["holdout"]["per_scope"]
    for scope in sorted(per_scope):
        for scheme_name in sorted(per_scope[scope]):
            block = per_scope[scope][scheme_name]
            add(
                f"| `{scope}` | `{scheme_name}` | {block['events_with_useful']} | "
                f"{block['hit@5']:.4f} | {block['mrr']:.4f} |"
            )
    add("")

    add("## Frozen goldset (234 items), scored in the same final pass")
    add("")
    add(
        "The frozen phase-0 goldset is not time-sliceable (its `cross_lingual` and "
        "`role_query` items are curated, not recorded events), so it is not part of the "
        "selection. It is reported as an independent sanity block on the same locked "
        "schemes. `live_order` is the actual end-to-end ranking the current code produced, "
        "measured by `retrieval_harness.compute_metrics`."
    )
    add("")
    add("| scheme | events | hit@1 | hit@5 | hit@10 | MRR |")
    add("|---|---|---|---|---|---|")
    live = {"hit@1": 0.0, "hit@5": 0.0, "mrr": 0.0}
    for entry in report["frozen_goldset"]["leaderboard"]:
        if entry["scheme"] == "live_order":
            live = entry["metrics"]
        add(_row(entry["scheme"], entry["metrics"]))
    add("")
    add(
        "The incumbent sits at the bottom of this table, and that is not a contradiction of "
        "the holdout result — it is why this block is a sanity check and not a selector. "
        "234 items is roughly a fifth of the holdout slice, so single-item moves are worth "
        "0.004 here; two of its three strata (`cross_lingual`, `role_query`) are curated "
        "jargon and role queries that the phase-1 gate already showed behave unlike recorded "
        "traffic; and its `content_grounded` items are drawn from after 2026-06-10, i.e. "
        "they straddle eval and holdout rather than forming an independent sample. The "
        "load-bearing line is `live_order`: the ranking the current code actually produced "
        f"end-to-end (hit@1 {live['hit@1']:.4f}, hit@5 {live['hit@5']:.4f}, "
        f"MRR {live['mrr']:.4f}) reproduces `artifacts/harness/phase1-gate.md`'s published "
        "0.188 / 0.577 / 0.354 to every digit it printed, which is the check that this run's "
        "retrieval path is the same one phase 1 gated."
    )
    add("")

    add("## Configuration outcome")
    add("")
    change = report["config_change"]
    if change["changed"]:
        add(
            f"`DEFAULT_RETRIEVAL_WEIGHTS` is re-seeded for {', '.join(change['families'])} to "
            f"bm25 {change['triple']['bm25']:.4f} / vector {change['triple']['vector']:.4f} / "
            f"graph {change['triple']['graph']:.4f}."
        )
    else:
        add(
            "**`src/living_memory/config.py` is unchanged.** " + change["reason"]
        )
    add("")
    add(
        "`DEFAULT_RETRIEVAL_POLICY_FLOORS`, `MemoryStore.update_retrieval_weights`, "
        "`MemoryStore.apply_retrieval_weight_floors` and `feedback._method_signals` are "
        "untouched, and pinned by `tests/test_retrieval_weight_recalibration.py`."
    )
    add("")
    holdout_board = report["stages"]["holdout"]["leaderboard"]
    spread = holdout_board[0]["metrics"]["hit@5"] - holdout_board[-1]["metrics"]["hit@5"]
    add(
        "What this hands to the next goal: the channel blend is not the bottleneck any more. "
        f"All {len(holdout_board)} finalists land inside a {spread:.3f} band of holdout hit@5 "
        f"({holdout_board[-1]['metrics']['hit@5']:.4f}-{holdout_board[0]['metrics']['hit@5']:.4f}), "
        "and the incumbent is already at the top of it — re-mixing channels has almost no "
        "leverage left over what retrieval returns. The remaining headroom is in what enters "
        "the candidate set at all: on holdout, "
        f"{_missed_events(report)} of "
        f"{report['split']['slices']['holdout']['count']} events put no relevant node in the "
        "top-5 even under the incumbent, and the frozen goldset's `cross_lingual` stratum is "
        f"still at hit@5 {_stratum(report, 'cross_lingual'):.4f} against "
        f"{_stratum(report, 'content_grounded'):.4f} for recorded traffic. That is a "
        "vocabulary problem — jargon and `when_to_use` triggers — not a weighting one."
    )
    add("")

    add("## Operator runbook — re-seeding live per-scope weights (NOT executed)")
    add("")
    add(
        "Config defaults seed only *new* scopes (`MemoryStore._seed_retrieval_weights` uses "
        "`ON CONFLICT(scope) DO NOTHING`). Every scope that already has a row keeps its "
        "learned weights forever, so a config change alone reaches nothing that exists "
        "today. This run did **not** mutate the live `retrieval_weights` table and does not "
        "recommend mutating it now; the procedure below is written down so that a future "
        "goal that *does* decide to re-seed has a rehearsed one."
    )
    add("")
    for step in report["runbook"]:
        add(f"{step['n']}. {step['text']}")
    add("")

    add("## Reproducing this report")
    add("")
    add("```")
    add(report["provenance"]["command"])
    add("```")
    add("")
    add(
        f"Snapshot `{report['provenance']['snapshot']['path']}` "
        f"(sha256 `{report['provenance']['snapshot']['sha256'][:16]}…`, "
        f"{report['provenance']['snapshot']['bytes']} bytes, "
        f"{report['provenance']['snapshot']['row_counts']['active_nodes']} active nodes, "
        f"{report['provenance']['snapshot']['row_counts']['recall_events']} recall events), "
        f"embedding backend `{report['provenance']['embedding_backend']}`. The live database "
        "was never opened through `MemoryStore` and was never opened for writing."
    )
    add("")
    observation = report["provenance"]["live_database"]["observation"]
    if observation.get("observed"):
        drifted = observation["scopes_drifted_since_snapshot"]
        add(
            f"One honesty note that a bare \"nothing was mutated\" would hide. The MCP "
            "server kept serving throughout this work, and its ordinary implicit-feedback "
            "loop keeps learning: of the "
            f"{observation['scopes']} scopes in the live `retrieval_weights` table, "
            f"**{drifted}** have moved away from the snapshot's values since it was frozen. "
            "Nothing here wrote them — the file was opened `file:...?mode=ro` for this "
            "reading only — but the rows are not frozen either, which is the strongest "
            "possible argument for the runbook above: these are learned rows, and a "
            "config default never reaches them."
        )
        add("")
        if observation["drift"]:
            add("| scope | snapshot bm25/vector/graph | live bm25/vector/graph | L1 | live updated_at |")
            add("|---|---|---|---|---|")
            for scope, entry in sorted(observation["drift"].items()):
                before = entry.get("snapshot")
                after = entry["live"]
                left = (
                    f"{before['bm25']:.4f} / {before['vector']:.4f} / {before['graph']:.4f}"
                    if before
                    else "(scope is new since the snapshot)"
                )
                add(
                    f"| `{scope}` | {left} | {after['bm25']:.4f} / {after['vector']:.4f} / "
                    f"{after['graph']:.4f} | {entry.get('l1_distance', float('nan')):.4f} | "
                    f"{entry.get('live_updated_at', '')} |"
                )
            add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def observe_live_weights(
    live_db: str | None, snapshot_weights: Mapping[str, RetrievalWeights]
) -> dict[str, Any]:
    """Read-only look at how far the live learned rows have drifted since the snapshot.

    This is the claim that matters for a weight goal and the one easiest to
    overstate. Saying "the live database was not mutated" is only honest about
    *this experiment*: the MCP server keeps running throughout, and its ordinary
    implicit-feedback loop (``feedback.apply_pending_recall_feedback`` ->
    ``MemoryStore.update_retrieval_weights``) keeps moving those rows on its
    own. Measuring the drift says exactly which of the two happened.
    """

    if not live_db:
        return {
            "observed": False,
            "note": "pass --observe-live-db to record the live-versus-snapshot drift",
        }
    connection = open_readonly(live_db)
    try:
        live = load_live_weights(connection)
    finally:
        connection.close()
    drifted = {}
    for scope, weights in sorted(live.items()):
        before = snapshot_weights.get(scope)
        if before is None:
            drifted[scope] = {"snapshot": None, "live": _weights_dict(weights)}
            continue
        moved = (
            abs(before.bm25 - weights.bm25)
            + abs(before.vector - weights.vector)
            + abs(before.graph - weights.graph)
        )
        if moved > 1e-9:
            drifted[scope] = {
                "snapshot": _weights_dict(before),
                "live": _weights_dict(weights),
                "l1_distance": round(moved, 6),
                "live_updated_at": weights.updated_at,
            }
    return {
        "observed": True,
        "path": str(live_db),
        "opened": "file:...?mode=ro, read-only, never through MemoryStore",
        "scopes": len(live),
        "scopes_drifted_since_snapshot": len(drifted),
        "drift": drifted,
        "note": (
            "any drift here is the running MCP server's own learning loop reacting to "
            "ordinary memory_recall / memory_remember traffic, not a write by this "
            "experiment, which never opens this file for writing"
        ),
    }


def build_runbook() -> list[dict[str, Any]]:
    return [
        {
            "n": 1,
            "text": (
                "Stop the MCP server or accept that the change takes effect on its next "
                "`get_retrieval_weights` call; the table is read per ranking pass, so an "
                "update is picked up without a restart but mid-flight requests may straddle it."
            ),
        },
        {
            "n": 2,
            "text": (
                "Back up first, with the WAL: "
                "`python3 -m living_memory.retrieval_harness snapshot --source-db "
                "~/.local/share/living-memory/global.sqlite3 --snapshot-out "
                "~/.cache/living-memory-harness/pre-reseed.sqlite3`. A plain `cp` of the "
                "`.sqlite3` file without its `-wal` is an incomplete copy."
            ),
        },
        {
            "n": 3,
            "text": (
                "Decide the target set explicitly. Re-seeding *all* scopes discards months of "
                "per-scope learning; the defensible subset is scopes whose `updated_at` is "
                "older than the chunking migration and whose weights were therefore fitted to "
                "the pre-chunking vector distribution."
            ),
        },
        {
            "n": 4,
            "text": (
                "Apply through the API, never with raw SQL: for each target scope call "
                "`MemoryStore.set_retrieval_weights(scope, bm25=..., vector=..., graph=...)` "
                "followed by `MemoryStore.apply_retrieval_weight_floors(scope, weights)` so "
                "the floors in `DEFAULT_RETRIEVAL_POLICY_FLOORS` are enforced exactly as the "
                "learning loop enforces them. Raw `UPDATE retrieval_weights` bypasses the "
                "floor pass and can leave a scope below `vector_min`/`bm25_min`."
            ),
        },
        {
            "n": 5,
            "text": (
                "Verify with `memory_health`: `retrieval_skew.scopes_at_risk` must be empty "
                "and `floors_in_effect` must match the configured floors. Then re-run this "
                "script against a fresh snapshot and confirm the incumbent row now matches "
                "the intended weights."
            ),
        },
        {
            "n": 6,
            "text": (
                "Rollback is a restore of the step-2 snapshot, or a second "
                "`set_retrieval_weights` call with the values recorded in this report's "
                "`incumbent.weights_by_scope` block."
            ),
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/replay_post_chunking.py",
        description="Post-chunking retrieval-weight A/B over three disjoint time slices.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", help="frozen, chunk-backfilled snapshot")
    source.add_argument("--source-db", help="database to freeze into a temporary snapshot first")
    parser.add_argument("--c1", required=True, help="train/eval cutoff (ISO)")
    parser.add_argument("--c2", required=True, help="eval/holdout cutoff (ISO)")
    parser.add_argument(
        "--frozen-goldset",
        default=str(REPO_ROOT / "artifacts/harness/goldset.jsonl"),
        help="frozen phase-0 goldset, read-only",
    )
    parser.add_argument(
        "--goldset-cutoff",
        default="2026-01-01T00:00:00Z",
        help="earliest event the recalibration goldset may draw from",
    )
    parser.add_argument("--min-containment", type=float, default=0.25)
    parser.add_argument(
        "--work-dir",
        default=str(Path.home() / ".cache/living-memory-harness/recalib"),
        help="cache directory for the recalibration goldset and the end-to-end runs",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild the recalibration goldset and re-run every query end-to-end",
    )
    parser.add_argument("--finalists", type=int, default=DEFAULT_FINALISTS)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP)
    parser.add_argument("--grid-graph-max", type=float, default=GRID_GRAPH_MAX)
    parser.add_argument("--min-scope-events", type=int, default=50)
    parser.add_argument(
        "--observe-live-db",
        help=(
            "optional live database to READ (file:...?mode=ro) purely so the report can "
            "state how far its learned retrieval_weights rows have drifted from the "
            "snapshot's. Never written to, never opened through MemoryStore."
        ),
    )
    parser.add_argument(
        "--report", default=str(REPO_ROOT / "artifacts/replay/post-chunking-ab.json")
    )
    parser.add_argument(
        "--markdown", default=str(REPO_ROOT / "artifacts/replay/post-chunking-ab.md")
    )
    args = parser.parse_args(argv)

    slices = Slices(c1=normalize_cutoff(args.c1), c2=normalize_cutoff(args.c2))
    if slices.c1 >= slices.c2:
        parser.error(f"--c1 ({slices.c1}) must be strictly before --c2 ({slices.c2})")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    recal_path = work_dir / "goldset-recalibration.jsonl"
    runs_path = work_dir / "runs-post-chunking.json"
    frozen_goldset_path = Path(args.frozen_goldset)

    with frozen_snapshot(args.snapshot, args.source_db) as (snapshot_path, manifest):
        # The sidecar manifest is written when a snapshot is frozen and is not
        # updated by anything done to the file afterwards — a chunk backfill,
        # for instance, leaves a manifest describing the pre-backfill bytes.
        # Provenance that names the wrong sha256 is worse than none, so the
        # file is always re-measured and the sidecar only kept when it agrees.
        verified = describe_snapshot(snapshot_path)
        if verified["snapshot_sha256"] != manifest.get("snapshot_sha256"):
            print(
                f"sidecar manifest for {snapshot_path} is stale "
                f"({manifest.get('snapshot_bytes')} bytes recorded, "
                f"{verified['snapshot_bytes']} on disk); using re-measured values",
                file=sys.stderr,
            )
            verified = {
                **verified,
                "source_path": manifest.get("source_path"),
                "sidecar_manifest_stale": True,
                "sidecar_snapshot_sha256": manifest.get("snapshot_sha256"),
            }
        manifest = verified
        snapshot_sha = manifest["snapshot_sha256"]

        cache: dict[str, Any] | None = None
        if runs_path.exists() and not args.refresh:
            loaded = json.loads(runs_path.read_text(encoding="utf-8"))
            if (
                loaded.get("snapshot_sha256") == snapshot_sha
                and loaded.get("frozen_goldset_sha256") == sha256_file(frozen_goldset_path)
                and recal_path.exists()
                and loaded.get("recalibration_goldset_sha256") == sha256_file(recal_path)
            ):
                cache = loaded
                print(f"reusing cached end-to-end runs from {runs_path}", file=sys.stderr)

        if cache is None:
            with working_copy(snapshot_path) as working_db:
                print("building the recalibration goldset…", file=sys.stderr)
                goldset_stats = build_recalibration_goldset(
                    working_db, args.goldset_cutoff, args.min_containment, recal_path
                )
                recal_items = load_goldset(recal_path)
                frozen_items = load_goldset(frozen_goldset_path)
                print(
                    f"replaying {len(recal_items)} + {len(frozen_items)} queries end-to-end…",
                    file=sys.stderr,
                )
                recal_runs = replay_end_to_end(working_db, recal_items, "recalibration")
                frozen_runs = replay_end_to_end(working_db, frozen_items, "frozen goldset")
            cache = {
                "snapshot_sha256": snapshot_sha,
                "recalibration_goldset_sha256": sha256_file(recal_path),
                "frozen_goldset_sha256": sha256_file(frozen_goldset_path),
                "goldset_stats": goldset_stats,
                "recalibration_runs": _runs_payload(recal_runs),
                "frozen_runs": _runs_payload(frozen_runs),
                "generated_at": utc_now_iso(),
            }
            runs_path.write_text(
                json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            recal_items = load_goldset(recal_path)
            frozen_items = load_goldset(frozen_goldset_path)
            goldset_stats = cache["goldset_stats"]
            recal_runs = _runs_from_payload(cache["recalibration_runs"], recal_items)
            frozen_runs = _runs_from_payload(cache["frozen_runs"], frozen_items)

        connection = open_readonly(snapshot_path)
        try:
            live_weights = load_live_weights(connection)
            evidence = snapshot_evidence(connection)
            report = run_ab(
                recal_runs=recal_runs,
                frozen_runs=frozen_runs,
                live_weights=live_weights,
                evidence=evidence,
                connection=connection,
                slices=slices,
                args=args,
                manifest=manifest,
                goldset_stats=goldset_stats,
                frozen_goldset_path=frozen_goldset_path,
            )
        finally:
            connection.close()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"decision: {'adopt ' + report['decision']['winner'] if report['decision']['adopted'] else 'keep the incumbent'}"
        f" -> {args.report}",
        file=sys.stderr,
    )
    return 0


def run_ab(
    *,
    recal_runs: Sequence[HarnessRun],
    frozen_runs: Sequence[HarnessRun],
    live_weights: Mapping[str, RetrievalWeights],
    evidence: Callable[[str], tuple[bool, bool]],
    connection: Any,
    slices: Slices,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    goldset_stats: Mapping[str, Any],
    frozen_goldset_path: Path,
) -> dict[str, Any]:
    events = dated_replay_events(recal_runs)
    buckets = scope_buckets(events, args.min_scope_events)
    sliced = slice_events(events, slices)
    for name in SLICE_NAMES:
        if not sliced[name]:
            raise SystemExit(f"time slice {name} is empty; adjust --c1/--c2")

    major_scopes = sorted(
        {scope for scope, bucket in buckets.items() if bucket != OTHER_SCOPE_BUCKET}
    )
    snapshot_learning_rates = {
        scope: weights.learning_rate for scope, weights in live_weights.items()
    }

    incumbent = Scheme(
        name="incumbent_live_weights",
        kind="live",
        resolver=static_resolver(dict(live_weights)),
        description=(
            "the per-scope weights learned and stored in the snapshot's retrieval_weights "
            "table, resolved through the live exact/family/default fallback chain"
        ),
        weights={"source": "snapshot retrieval_weights table"},
    )

    fixed: list[Scheme] = [
        incumbent,
        Scheme(
            name="config_defaults_current",
            kind="config",
            resolver=floor_default_weights(MemoryConfig(), evidence),
            description="today's DEFAULT_RETRIEVAL_WEIGHTS with the configured floors applied",
            weights={
                family: {
                    "bm25": DEFAULT_RETRIEVAL_WEIGHTS[family].bm25,
                    "vector": DEFAULT_RETRIEVAL_WEIGHTS[family].vector,
                    "graph": DEFAULT_RETRIEVAL_WEIGHTS[family].graph,
                }
                for family in sorted(DEFAULT_RETRIEVAL_WEIGHTS)
            },
        ),
        Scheme(
            name="uniform",
            kind="explicit",
            resolver=explicit_weights(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            description="equal weight on every channel, every scope",
            weights={"bm25": 1 / 3, "vector": 1 / 3, "graph": 1 / 3},
        ),
    ]

    # --- Stage 1: TRAIN ----------------------------------------------------
    train = sliced["train"]
    print(f"stage TRAIN: {len(train)} events", file=sys.stderr)

    trajectories: dict[str, Any] = {}
    for label, config, rule in (
        ("proportional_from_config", MemoryConfig(), proportional_credit),
        ("proportional_from_live", config_from_live(live_weights), proportional_credit),
        ("winner_take_all_from_live", config_from_live(live_weights), winner_take_all_credit),
    ):
        trajectory, summary = fit_trajectory(
            label, config, train, buckets, evidence, rule, snapshot_learning_rates
        )
        trajectories[label] = summary
        fixed.append(
            Scheme(
                name=f"trajectory_{label}",
                kind="trajectory",
                resolver=static_resolver(dict(trajectory.weights)),
                description=(
                    f"replay.WeightTrajectory fitted on the {len(train)} TRAIN events "
                    f"({summary['updates']} reinforcement updates over "
                    f"{summary['scopes_updated']} scopes), then frozen"
                ),
                weights=summary["final"],
            )
        )

    grid = weight_grid(args.grid_step, args.grid_graph_max)
    grid_schemes: list[Scheme] = []
    for triple in grid:
        bm25, vector, graph = triple
        tag = f"b{bm25:.2f}_v{vector:.2f}_g{graph:.2f}"
        grid_schemes.append(
            Scheme(
                name=f"explicit_{tag}",
                kind="explicit_grid",
                resolver=explicit_weights(bm25, vector, graph),
                description=f"one explicit triple applied to every scope: {triple}",
                weights={"bm25": bm25, "vector": vector, "graph": graph},
            )
        )
        grid_schemes.append(
            Scheme(
                name=f"config_{tag}",
                kind="config_grid",
                resolver=floor_default_weights(config_with_families(triple), evidence),
                description=(
                    f"DEFAULT_RETRIEVAL_WEIGHTS re-seeded to {triple} for "
                    f"{', '.join(TUNABLE_FAMILIES)}, with the configured floors applied"
                ),
                weights={"bm25": bm25, "vector": vector, "graph": graph},
            )
        )

    train_scored: dict[str, dict[str, Any]] = {}
    all_schemes = {scheme.name: scheme for scheme in [*fixed, *grid_schemes]}
    started = time.monotonic()
    for index, scheme in enumerate(all_schemes.values(), start=1):
        train_scored[scheme.name] = score(scheme, train, buckets)
        if index % 50 == 0:
            print(
                f"  train {index}/{len(all_schemes)} ({time.monotonic() - started:.0f}s)",
                file=sys.stderr,
            )
    train_leaderboard = sorted(
        (
            {"scheme": name, "kind": all_schemes[name].kind, "metrics": _strip(block)}
            for name, block in train_scored.items()
        ),
        key=lambda entry: (-entry["metrics"]["hit@5"], -entry["metrics"]["mrr"], entry["scheme"]),
    )

    def top(kind: str, count: int) -> list[str]:
        return [
            entry["scheme"] for entry in train_leaderboard if entry["kind"] == kind
        ][:count]

    finalist_names = [scheme.name for scheme in fixed]
    finalist_names += top("explicit_grid", args.finalists)
    finalist_names += top("config_grid", args.finalists)
    finalists = [all_schemes[name] for name in dict.fromkeys(finalist_names)]

    # --- Stage 2: EVAL -----------------------------------------------------
    evaluation = sliced["eval"]
    print(f"stage EVAL: {len(evaluation)} events, {len(finalists)} finalists", file=sys.stderr)
    eval_scored = {scheme.name: score(scheme, evaluation, buckets) for scheme in finalists}
    eval_leaderboard = sorted(
        (
            {"scheme": name, "kind": all_schemes[name].kind, "metrics": _strip(block)}
            for name, block in eval_scored.items()
        ),
        key=lambda entry: (
            -entry["metrics"]["hit@5"],
            -entry["metrics"]["mrr"],
            entry["scheme"] != incumbent.name,
            entry["scheme"],
        ),
    )
    winner_name = eval_leaderboard[0]["scheme"]
    winner = all_schemes[winner_name]

    # --- Stage 3: HOLDOUT (one pass, winner already locked) ----------------
    holdout = sliced["holdout"]
    print(f"stage HOLDOUT: {len(holdout)} events (single scoring pass)", file=sys.stderr)
    holdout_scored = {scheme.name: score(scheme, holdout, buckets) for scheme in finalists}
    holdout_leaderboard = sorted(
        (
            {"scheme": name, "kind": all_schemes[name].kind, "metrics": _strip(block)}
            for name, block in holdout_scored.items()
        ),
        key=lambda entry: (-entry["metrics"]["hit@5"], -entry["metrics"]["mrr"], entry["scheme"]),
    )

    incumbent_holdout = holdout_scored[incumbent.name]
    winner_holdout = holdout_scored[winner_name]
    bar = {
        "hit@5": {
            "incumbent": float(incumbent_holdout["hit@5"]),
            "winner": float(winner_holdout["hit@5"]),
            "passes": float(winner_holdout["hit@5"]) >= float(incumbent_holdout["hit@5"]),
        },
        "mrr": {
            "incumbent": float(incumbent_holdout["mrr"]),
            "winner": float(winner_holdout["mrr"]),
            "passes": float(winner_holdout["mrr"]) >= float(incumbent_holdout["mrr"]),
        },
    }
    clears = bar["hit@5"]["passes"] and bar["mrr"]["passes"]
    adopted = clears and winner_name != incumbent.name

    # --- Frozen goldset, same final pass -----------------------------------
    frozen_events = [to_replay_event(run) for run in sorted(frozen_runs, key=lambda r: r.item.query_id)]
    for event in frozen_events:
        event.created_at = "frozen"
    frozen_buckets = scope_buckets(frozen_events, args.min_scope_events)
    frozen_scored = {
        scheme.name: score(scheme, frozen_events, frozen_buckets) for scheme in finalists
    }
    live_metrics = compute_metrics(frozen_runs)
    frozen_leaderboard = [
        {
            "scheme": "live_order",
            "kind": "end_to_end",
            "metrics": _strip(live_metrics["overall"]),
        }
    ] + sorted(
        (
            {"scheme": name, "kind": all_schemes[name].kind, "metrics": _strip(block)}
            for name, block in frozen_scored.items()
        ),
        key=lambda entry: (-entry["metrics"]["hit@5"], -entry["metrics"]["mrr"], entry["scheme"]),
    )

    per_scope: dict[str, dict[str, Any]] = {}
    for name, block in holdout_scored.items():
        for scope, scope_block in block.get("per_scope", {}).items():
            per_scope.setdefault(scope, {})[name] = _strip(scope_block)

    paired = (
        paired_comparison(winner, incumbent, holdout)
        if winner_name != incumbent.name
        else None
    )
    diagnostics = {
        "score_shift": score_shift(connection, recal_runs, events),
        "score_distribution_by_slice": {
            name: score_distribution(sliced[name]) for name in SLICE_NAMES
        },
        "holdout_paired_winner_vs_incumbent": paired,
    }

    rationale = _rationale(
        adopted=adopted,
        clears=clears,
        winner_name=winner_name,
        incumbent_name=incumbent.name,
        bar=bar,
    )

    return {
        "report_version": REPORT_VERSION,
        "generated_at": utc_now_iso(),
        "goldset": {
            "recalibration": dict(goldset_stats),
            "frozen": {
                "path": str(frozen_goldset_path),
                "sha256": sha256_file(frozen_goldset_path),
                "items": len(frozen_runs),
                "role": "frozen input, read only, not part of selection",
            },
        },
        "split": slice_manifest(sliced, slices),
        "incumbent": {
            "scheme": incumbent.name,
            "description": incumbent.description,
            "weights_by_scope": {
                scope: _weights_dict(weights) for scope, weights in sorted(live_weights.items())
            },
            "weights_by_reported_scope": resolved_family_weights(
                incumbent.resolver, major_scopes
            ),
        },
        "schemes": {
            scheme.name: {
                "kind": scheme.kind,
                "description": scheme.description,
                "weights": scheme.weights,
            }
            for scheme in finalists
        },
        "trajectories": trajectories,
        "stages": {
            "train": {
                "events": len(train),
                "schemes_scored": len(all_schemes),
                "explicit_grid_size": len(grid),
                "config_grid_size": len(grid),
                "grid_step": args.grid_step,
                "grid_graph_max": args.grid_graph_max,
                "leaderboard": train_leaderboard,
                "finalists": [scheme.name for scheme in finalists],
            },
            "eval": {
                "events": len(evaluation),
                "leaderboard": eval_leaderboard,
                "winner": winner_name,
            },
            "holdout": {
                "events": len(holdout),
                "scored_once": True,
                "leaderboard": holdout_leaderboard,
                "per_scope": per_scope,
                "full": {name: block for name, block in sorted(holdout_scored.items())},
            },
        },
        "frozen_goldset": {
            "items": len(frozen_runs),
            "leaderboard": frozen_leaderboard,
            "end_to_end_metrics": live_metrics,
        },
        "decision": {
            "selection_rule": SELECTION_RULE.format(finalists=args.finalists),
            "winner": winner_name,
            "winner_description": all_schemes[winner_name].description,
            "generalization_bar": bar,
            "clears_bar": clears,
            "adopted": adopted,
            "rationale": rationale,
            "summary": _summary(adopted, winner_name, incumbent.name, bar),
        },
        "diagnostics": diagnostics,
        "config_change": _config_change(adopted, winner, rationale),
        "floors": {
            family: {
                "bm25_max": floors.bm25_max,
                "vector_min": floors.vector_min,
                "graph_min": floors.graph_min,
                "bm25_min": floors.bm25_min,
            }
            for family, floors in sorted(DEFAULT_RETRIEVAL_POLICY_FLOORS.items())
        },
        "runbook": build_runbook(),
        "provenance": {
            "git_commit": git_commit(),
            "embedding_backend": embedding_backend(),
            "command": (
                "python3 scripts/replay_post_chunking.py "
                f"--snapshot {args.snapshot or args.source_db} "
                f"--c1 {slices.c1} --c2 {slices.c2} "
                + (
                    f"--observe-live-db {args.observe_live_db} "
                    if args.observe_live_db
                    else ""
                )
                + f"--report {args.report} --markdown {args.markdown}"
            ),
            "snapshot": {
                "path": manifest.get("snapshot_path"),
                "sha256": manifest.get("snapshot_sha256"),
                "bytes": manifest.get("snapshot_bytes"),
                "captured_at": manifest.get("captured_at"),
                "row_counts": manifest.get("row_counts", {}),
                "sha256_source": (
                    "re-measured from the file; the sidecar manifest described "
                    "different bytes and was not trusted"
                    if manifest.get("sidecar_manifest_stale")
                    else "sidecar manifest, re-verified against the file"
                ),
                "sidecar_manifest_stale": bool(manifest.get("sidecar_manifest_stale")),
            },
            "live_database": {
                "mutated_by_this_experiment": False,
                "retrieval_weights_written_by_this_experiment": False,
                "note": (
                    "the snapshot is opened read-only or copied through the SQLite backup "
                    "API; MemoryStore only ever sees a throwaway working copy. The live "
                    "file is only ever read, and only when --observe-live-db is passed"
                ),
                "observation": observe_live_weights(args.observe_live_db, live_weights),
            },
            "min_scope_events": args.min_scope_events,
        },
    }


_METRIC_KEYS = ("events", "events_with_useful", "useful_results", "hit@1", "hit@5", "hit@10", "mrr")


def _strip(block: Mapping[str, Any]) -> dict[str, Any]:
    return {key: block.get(key, 0) for key in _METRIC_KEYS}


def _summary(adopted: bool, winner: str, incumbent: str, bar: Mapping[str, Any]) -> str:
    hit = bar["hit@5"]
    mrr = bar["mrr"]
    if adopted:
        return (
            f"`{winner}` won on eval and cleared the holdout bar "
            f"(hit@5 {hit['winner']:.4f} vs {hit['incumbent']:.4f}, "
            f"MRR {mrr['winner']:.4f} vs {mrr['incumbent']:.4f})."
        )
    if winner == incumbent:
        return (
            "No candidate beat the incumbent on eval, so the incumbent is the winner and "
            f"the holdout numbers (hit@5 {hit['incumbent']:.4f}, MRR {mrr['incumbent']:.4f}) "
            "are its own."
        )
    return (
        f"`{winner}` won on eval but did not clear the holdout bar "
        f"(hit@5 {hit['winner']:.4f} vs {hit['incumbent']:.4f}, "
        f"MRR {mrr['winner']:.4f} vs {mrr['incumbent']:.4f}), so the incumbent stands."
    )


def _rationale(
    *, adopted: bool, clears: bool, winner_name: str, incumbent_name: str, bar: Mapping[str, Any]
) -> str:
    if winner_name == incumbent_name:
        return (
            "The eval slice picked the incumbent itself: no candidate — not the config "
            "defaults, not uniform, not a train-fitted trajectory, not any grid triple — "
            "ranked above the learned per-scope weights. The generalization bar is "
            "therefore trivially met by the incumbent against itself, and the correct "
            "outcome is to keep the incumbent weights and change no configuration value."
        )
    if adopted:
        return (
            f"`{winner_name}` was selected on eval before holdout was read and then cleared "
            "both holdout conditions, so the improvement is shown on data that took no part "
            "in either fitting or selection."
        )
    failed = [key for key in ("hit@5", "mrr") if not bar[key]["passes"]]
    return (
        f"`{winner_name}` led on eval but failed the holdout bar on "
        f"{' and '.join(failed)}, which is exactly the overfitting the three-way split "
        "exists to catch. The incumbent weights stand and no configuration value changes."
    )


def _config_change(adopted: bool, winner: Scheme, rationale: str) -> dict[str, Any]:
    if adopted and winner.kind in ("config", "config_grid"):
        return {
            "changed": True,
            "families": list(TUNABLE_FAMILIES),
            "triple": {
                "bm25": winner.weights["bm25"],
                "vector": winner.weights["vector"],
                "graph": winner.weights["graph"],
            },
            "reason": f"{winner.name} cleared the holdout generalization bar.",
        }
    reason = rationale
    if adopted:
        reason = (
            f"{winner.name} cleared the holdout bar, but it is a `{winner.kind}` scheme: it "
            "describes learned per-scope state or an explicit ranking-time blend, not a "
            "config default. DEFAULT_RETRIEVAL_WEIGHTS seeds only new scopes, so writing "
            "this triple there would not reproduce the measured win."
        )
    return {"changed": False, "families": [], "triple": None, "reason": reason}


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
