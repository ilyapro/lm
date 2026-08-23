"""Residual ranked pool exposure: ``MemoryRecallService.last_residual``.

The recall map is built from the candidates ranked past the ``max_results``
cut. These tests pin the exposure contract: the residual is the tail of the
same ranking that produced the delivered slice, it is overwritten on every
call (to ``[]`` when nothing remains), and it stays invisible to delivery,
access logging, and the recorded recall event.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from living_memory import recall_map as recall_map_module
from living_memory.chunking import TextChunk
from living_memory.models import Node
from living_memory.recall_map import (
    MAX_POOL_NODES,
    POOL_DEMOTION_GATE_ENV,
    POOL_DEMOTION_WINDOWS_ENV,
    POOL_GATE_REASON_CODES,
    POOL_USEFULNESS_FLOOR_ENV,
    POOL_USEFULNESS_GATE_ENV,
    RELEVANCE_FEATURE_MEANS,
    RELEVANCE_FEATURE_SCALES,
    RELEVANCE_THRESHOLD,
    SELECTION_LEDGER_REASON_CODES,
    SELECTION_REASON_CODES,
    SELECTION_SAMPLE_GIST_CHARS,
    RecallMapBuilder,
    _ballast_reason,
    _directional_zsum,
    _relevance_features,
    relevance_score,
)
from living_memory.retrieval import MemoryRecallService
from living_memory.retrieval import RecallResult
from living_memory.storage import (
    MAX_RECALL_HISTORY_CANDIDATES,
    MaturedRecallHistory,
    MemoryStore,
)

POOL_QUERY = "deployment failure database migration rollback"
POOL_SCOPE = "project:alpha"

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "scripts" / "recall_map_relevance_eval.py"


def _load_evaluator():
    """Load the unchanged evaluator without importing production helpers."""

    name = "recall_map_relevance_eval_pool_differential"
    present = sys.modules.get(name)
    if present is not None:
        return present
    spec = importlib.util.spec_from_file_location(name, EVALUATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _form_name(reason: str | None) -> str:
    return {
        None: "eligible",
        "fc": "file_chunk_envelope",
        "ss": "strategy_stagnation",
        "sj": "supervision_journal",
    }[reason]


def _valid_file_chunk(rng: random.Random, index: int) -> dict[str, str]:
    part = rng.randint(1, 30)
    total = rng.randint(part, part + 30)
    first = rng.randint(1, 20_000)
    return {
        "language": rng.choice(("python", "rust", "typescript", "markdown", "")),
        "path": f"pkg/random-{rng.randrange(1_000_000):06d}/file-{index}.py",
        "lines": f"{first}-{first + rng.randint(0, 400)}",
        "chunk": f"{part}/{total}",
        "kind": rng.choice(("source", "test", "docs", "configuration")),
        "sha256": hashlib.sha256(f"independent-{index}".encode()).hexdigest(),
    }


def _independent_form_controls(
    *, seed: int = 0xA11CE823, variants: int = 64
) -> list[tuple[str | None, str, dict[str, object], dict[str, object]]]:
    """Fixed-seed controls generated independently of both classifiers."""

    rng = random.Random(seed)
    cases: list[tuple[str | None, str, dict[str, object], dict[str, object]]] = []
    detail_keys = ("attempts", "strategy", "window", "reason")
    journal_kinds = (
        "execution_journal",
        "goal_tree_journal",
        "monitoring_journal",
        "supervision_journal",
    )
    metadata_keys = ("kind", "type", "lesson_kind", "record_kind")
    for index in range(variants):
        header = _valid_file_chunk(rng, index)
        encoded = json.dumps(
            header, separators=(",", ":"), sort_keys=bool(index % 2)
        )
        cases.append(
            (
                "fc",
                f"[file-chunk]{rng.choice((' ', '  ', chr(9)))}{encoded}\r\nbody",
                {},
                {},
            )
        )

        details = rng.sample(detail_keys, rng.randrange(len(detail_keys) + 1))
        stagnation = f"Strategy stagnation detected on branch-{rng.randrange(1_000_000)}"
        stagnation += "".join(
            f"\n{key}:{rng.choice((' unchanged', ' retrying', ' 7', ' exhausted'))}"
            for key in details
        )
        cases.append(("ss", stagnation, {}, {}))

        kind = rng.choice(journal_kinds)
        if rng.randrange(2):
            mapping = {
                rng.choice(metadata_keys): kind.replace(
                    "_", rng.choice(("_", "-"))
                )
            }
            context, provenance = (mapping, {}) if rng.randrange(2) else ({}, mapping)
            content = f"opaque structured row {index}"
        else:
            marker = rng.choice(("supervision_journal", "monitoring-journal"))
            context, provenance = {}, {}
            content = f"[{marker}]\nrun={index}"
        cases.append(("sj", content, context, provenance))

        # Marker substrings and even a complete valid-looking machine header
        # remain prose when the exact envelope position is absent.
        cases.append(
            (
                None,
                rng.choice(
                    (
                        f"I quoted [file-chunk] {encoded} while reviewing the index.",
                        f"User prose before Strategy stagnation detected on branch-{index}",
                        f"My note says [supervision_journal] is a literal marker {index}.",
                        f"We discussed monitoring_journal and [file-chunk] in paragraph {index}.",
                    )
                ),
                {"topic": rng.choice(("monitoring_journal", "supervision-journal"))},
                {},
            )
        )

        # A real prefix with one randomized schema violation is fail-open.
        malformed = dict(header)
        mutation = rng.randrange(6)
        if mutation == 0:
            malformed.pop("kind")
        elif mutation == 1:
            malformed["sha256"] = malformed["sha256"].upper()
        elif mutation == 2:
            malformed["chunk"] = "0/1"
        elif mutation == 3:
            malformed["chunk"] = "9/2"
        elif mutation == 4:
            malformed["lines"] = "80-20"
        else:
            malformed["language"] = 7  # type: ignore[assignment]
        cases.append(
            (
                None,
                "[file-chunk] " + json.dumps(malformed, separators=(",", ":")),
                {},
                {},
            )
        )

        # Near-miss strategy and journal envelopes: extra prose, an unknown
        # detail key, or a marker that is not the complete first line.
        cases.extend(
            (
                (
                    None,
                    f"Strategy stagnation detected on branch-{index}\ncomment:user-authored",
                    {},
                    {},
                ),
                (
                    None,
                    f"[monitoring_journal] is documentation, not a row {index}.",
                    {},
                    {},
                ),
                (
                    None,
                    f"\n[supervision_journal]\nuser notebook entry {index}",
                    {},
                    {},
                ),
            )
        )
    return cases


def _seed_pool(store: MemoryStore, count: int) -> None:
    for index in range(count):
        store.append_trace(
            f"deployment failure case {index}: database migration rollback step {index}",
            {"scope": POOL_SCOPE, "agent": f"agent-{index}"},
        )


def test_last_residual_is_ranked_tail_disjoint_from_delivered(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_pool(store, 12)
        service = MemoryRecallService(store)

        # Reference pass without logging: it warms the lazy embeddings and
        # captures the ranking this exact store state produces. It mutates
        # nothing rank-relevant, so the logged pass below must slice the same
        # ranking at the same cut.
        reference = service.memory_recall(
            POOL_QUERY, scope=POOL_SCOPE, max_results=4, log_access=False
        )
        reference_residual = list(service.last_residual)

        results = service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        residual = service.last_residual

        assert len(results) == 4
        assert residual

        delivered_ids = [result.node.id for result in results]
        residual_ids = [result.node.id for result in residual]
        assert set(delivered_ids).isdisjoint(residual_ids)
        assert delivered_ids == [result.node.id for result in reference]
        assert residual_ids == [result.node.id for result in reference_residual]

        # Rank order holds across the cut: scores never increase from the
        # delivered slice into and through the residual.
        scores = [result.score for result in results] + [
            result.score for result in residual
        ]
        assert scores == sorted(scores, reverse=True)


def test_last_residual_empty_when_pool_fits_and_overwritten_per_call(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_pool(store, 12)
        store.append_trace(
            "billing invoice rounding bug in currency conversion",
            {"scope": "project:beta", "agent": "agent-billing"},
        )
        service = MemoryRecallService(store)

        service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        assert service.last_residual

        small = service.memory_recall(
            "billing invoice rounding currency",
            scope="project:beta",
            max_results=10,
        )
        assert small
        assert len(small) < 10
        assert service.last_residual == []

        service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        assert service.last_residual
        assert service.memory_recall("   ", scope=POOL_SCOPE) == []
        assert service.last_residual == []


def test_delivery_event_and_access_logging_ignore_residual(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_pool(store, 12)
        service = MemoryRecallService(store)

        results = service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        residual = service.last_residual

        assert len(results) == 4
        assert residual

        event = store.get_recall_event(service.last_recall_event_id)
        assert event is not None
        assert len(event.results) <= 4
        assert event.result_ids == [result.node.id for result in results]
        assert set(event.result_ids).isdisjoint(
            {result.node.id for result in residual}
        )

        # The event id is stamped onto delivered results only; the residual is
        # the raw ranked tail, untouched by event recording.
        assert all(result.recall_event_id == event.id for result in results)
        assert all(result.recall_event_id is None for result in residual)

        for result in results:
            assert store.get_node(result.node.id).access_count == 1
        for result in residual:
            assert store.get_node(result.node.id).access_count == 0


# Frozen member eligibility and pre-cap accounting
# ---------------------------------------------------------------------------


def test_production_form_classifier_matches_evaluator_on_independent_controls() -> None:
    evaluator = _load_evaluator()
    cases = _independent_form_controls()

    assert len(cases) == 8 * 64
    for expected, content, context, provenance in cases:
        production = _ballast_reason(content, context, provenance)
        evaluated = evaluator.classify_content_form(content, context, provenance)
        assert production == expected
        assert evaluated == _form_name(expected)

    # Both sides of the audit are substantial and independently generated:
    # valid machine envelopes are excluded, while randomized prose and
    # malformed lookalikes remain eligible.
    assert sum(expected is not None for expected, *_rest in cases) == 3 * 64
    assert sum(expected is None for expected, *_rest in cases) == 5 * 64


def _history_entries(outcomes: list[bool]) -> list[tuple[str, str, bool]]:
    return [
        (
            f"2026-01-{index + 2:02d}T00:00:00Z",
            f"2026-01-{index + 1:02d}T00:00:00Z",
            consumed,
        )
        for index, consumed in enumerate(outcomes)
    ]


def test_score_arithmetic_differential_matches_evaluator_and_frozen_policy() -> None:
    evaluator = _load_evaluator()
    document = json.loads(
        (ROOT / "artifacts/recall-map/relevance/policy.json").read_text()
    )
    assert hashlib.sha256(EVALUATOR_PATH.read_bytes()).hexdigest() == document[
        "bindings"
    ]["historical_evaluator"]["sha256"]
    policy = document["selected_policy"]
    means = tuple(policy["fit"]["standardization_means"])
    scales = tuple(policy["fit"]["scales"])
    assert means == RELEVANCE_FEATURE_MEANS
    assert scales == RELEVANCE_FEATURE_SCALES
    assert policy["fit"]["threshold"] == RELEVANCE_THRESHOLD

    rng = random.Random(0xD1FF3E23)
    at = "2030-01-01T00:00:00Z"
    for _index in range(256):
        matured = rng.randrange(0, 24)
        outcomes = [bool(rng.randrange(2)) for _item in range(matured)]
        consumed = sum(outcomes)
        streak = 0
        for outcome in reversed(outcomes):
            if outcome:
                break
            streak += 1
        evaluator_features = evaluator.matured_past_features(
            {"candidate": _history_entries(outcomes)}, "candidate", at
        )
        level = rng.choice(("trace", "concept", "schema"))
        expected_features = (
            float(level == "schema"),
            evaluator_features["prior_matured_log_count"],
            evaluator_features["prior_nonconsumed_log_count"],
            evaluator_features["prior_nonconsumption_streak_log"],
            evaluator_features["prior_consumption_rate"],
        )
        expected_score = sum(
            (value - mean) / scale
            for value, mean, scale in zip(
                expected_features, means, scales, strict=True
            )
        )
        history = MaturedRecallHistory.known(matured, consumed, streak)
        node = SimpleNamespace(level=level)

        assert _relevance_features(node, history) == expected_features
        assert relevance_score(node, history) == expected_score
        assert (relevance_score(node, history) >= RELEVANCE_THRESHOLD) == (
            expected_score >= policy["fit"]["threshold"]
        )


def test_no_history_is_zero_but_unavailable_history_uses_train_means() -> None:
    trace = SimpleNamespace(level="trace")
    no_history = MaturedRecallHistory.known(0, 0, 0)
    unavailable = MaturedRecallHistory.unavailable("history_read_failed")

    assert _relevance_features(trace, no_history) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert _relevance_features(trace, unavailable) == (
        0.0,
        *RELEVANCE_FEATURE_MEANS[1:],
    )
    assert relevance_score(trace, unavailable) == (
        -RELEVANCE_FEATURE_MEANS[0] / RELEVANCE_FEATURE_SCALES[0]
    )
    assert relevance_score(trace, no_history) < relevance_score(trace, unavailable)

    # Threshold comparison is inclusive.  This arithmetic-only boundary pin
    # avoids needing an integer M/C/K combination that happens to land there.
    boundary = list(RELEVANCE_FEATURE_MEANS)
    boundary[0] += RELEVANCE_THRESHOLD * RELEVANCE_FEATURE_SCALES[0]
    assert _directional_zsum(boundary) == pytest.approx(RELEVANCE_THRESHOLD)
    assert _directional_zsum(boundary) >= RELEVANCE_THRESHOLD


def _node(
    store: MemoryStore,
    node_id: str,
    content: str,
    *,
    context: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
    stats: dict[str, object] | None = None,
) -> Node:
    return store.create_node(
        level="trace",
        content=content,
        context={"scope": POOL_SCOPE, **(context or {})},
        provenance=provenance,
        stats=stats,
        node_id=node_id,
    )


def _machine_header(index: int) -> str:
    return json.dumps(
        {
            "path": f"src/machine-{index}.py",
            "kind": "source",
            "language": "python",
            "sha256": hashlib.sha256(f"machine-{index}".encode()).hexdigest(),
            "chunk": "1/2",
            "lines": "1-20",
        },
        separators=(",", ":"),
    )


def test_complete_residual_is_accounted_before_cap_and_history_is_batched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "selector.sqlite3") as store:
        high_first = _node(store, "HIGH-000", "HIGH-000 durable first survivor")
        file_chunk = _node(
            store,
            "BALLAST-FC",
            f"[file-chunk] {_machine_header(1)}\nbody",
        )
        stagnation = _node(
            store,
            "BALLAST-SS",
            "Strategy stagnation detected on selector\nreason: unchanged",
        )
        journal = _node(
            store,
            "BALLAST-SJ",
            "opaque journal row",
            provenance={"record_kind": "goal-tree-journal"},
        )
        low = _node(store, "LOW", "ordinary never-delivered trace")
        high_tail = [
            _node(
                store,
                f"HIGH-{index:03d}",
                f"HIGH-{index:03d} durable tail survivor",
            )
            for index in range(1, MAX_POOL_NODES + 2)
        ]

        raw: list[object] = [
            SimpleNamespace(node=None),
            SimpleNamespace(node=SimpleNamespace(id=" ", content="missing identity")),
            RecallResult(node=high_first, score=1.0),
            RecallResult(node=high_first, score=0.99),
            RecallResult(node=file_chunk, score=0.98),
            RecallResult(node=stagnation, score=0.97),
            RecallResult(node=journal, score=0.96),
            RecallResult(node=low, score=0.95),
            *(RecallResult(node=node, score=0.94) for node in high_tail),
        ]
        calls: list[tuple[str, ...]] = []

        def history(candidate_ids, decision_at):
            ids = tuple(candidate_ids)
            calls.append(ids)
            assert decision_at == "2026-08-23T12:00:00Z"
            assert len(ids) <= MAX_RECALL_HISTORY_CANDIDATES
            return {
                node_id: (
                    MaturedRecallHistory.known(0, 0, 0)
                    if node_id == low.id
                    else MaturedRecallHistory.known(100, 50, 50)
                )
                for node_id in ids
            }

        monkeypatch.setattr(store, "matured_recall_history", history)
        pool, selection = RecallMapBuilder(store)._pool(
            raw,  # type: ignore[arg-type]
            decision_at="2026-08-23T12:00:00Z",
        )

        assert [len(batch) for batch in calls] == [200, 3]
        requested = {node_id for batch in calls for node_id in batch}
        assert requested == {high_first.id, low.id, *(node.id for node in high_tail)}
        assert not requested & {file_chunk.id, stagnation.id, journal.id}

        assert len(pool) == MAX_POOL_NODES
        assert [member.rank for member in pool] == [2, *range(8, 207)]
        assert all(member.relevance_score >= RELEVANCE_THRESHOLD for member in pool)
        assert selection.inspected == len(raw) == 209
        assert selection.admitted == MAX_POOL_NODES
        assert selection.excluded == (2, 1, 1, 1, 1, 1, 2)
        assert SELECTION_REASON_CODES == ("iv", "du", "fc", "ss", "sj", "lr", "pc")
        assert selection.inspected == selection.admitted + sum(selection.excluded)
        assert selection.samples[-1].reason == "pc"
        assert selection.samples[-1].ordinal == 207

        compact = selection.to_dict()
        assert compact["n"] == compact["e"] + sum(compact["x"])
        assert compact["x"] == list(selection.excluded)
        assert [sample[0] for sample in compact["q"]] == ["fc", "ss"]
        assert all(0 < len(sample[1]) <= SELECTION_SAMPLE_GIST_CHARS for sample in compact["q"])
        assert compact["o"] == 4
        assert not any(
            node.id in sample[1]
            for node in [file_chunk, stagnation, journal, low, *high_tail]
            for sample in compact["q"]
        )


def test_nonempty_all_low_residual_returns_accounted_empty_selection(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "all-low.sqlite3") as store:
        nodes = [
            _node(store, f"LOW-{index}", f"new trace {index}")
            for index in range(3)
        ]
        selected, accounting = RecallMapBuilder(store)._pool(
            [RecallResult(node=node, score=10.0) for node in nodes],
            decision_at="2026-08-23T12:00:00Z",
        )

        assert selected == []
        assert accounting.excluded == (0, 0, 0, 0, 0, 3, 0)
        assert accounting.to_dict(sample_limit=0) == {
            "v": "r1",
            "n": 3,
            "e": 0,
            "x": [0, 0, 0, 0, 0, 3, 0],
            "o": 3,
        }


def test_nonempty_all_low_build_is_a_journal_only_non_offer(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "all-low-build.sqlite3") as store:
        nodes = [
            _node(
                store,
                f"LOW-BUILD-{index}",
                f"new trace {index}",
                context={"procedure_id": f"low-{index}"},
            )
            for index in range(3)
        ]
        builder = RecallMapBuilder(store)
        built = builder.build(
            [RecallResult(node=node, score=10.0) for node in nodes],
            scope=POOL_SCOPE,
            task="all-low",
            decision_at="2026-08-23T12:00:00Z",
        )

        assert built is not None
        assert built.clusters == ()
        assert built.curtailed is False
        assert built.pool_size == 0
        assert built.render_compact() == ""
        assert built.to_dict() == {
            "clusters": [],
            "pool": 0,
            "covered": 0,
            "sel": {
                "v": "r1",
                "n": 3,
                "e": 0,
                "x": [0, 0, 0, 0, 0, 3, 0],
                "o": 2,
                "q": [["lr", "new trace 0"]],
            },
        }


def test_all_filtered_build_has_a_revision_bound_negative_cache(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "all-low-negative-cache.sqlite3") as store:
        nodes = [
            _node(store, f"NEGATIVE-{index}", f"freshly filtered trace {index}")
            for index in range(3)
        ]
        residual = [RecallResult(node=node, score=10.0) for node in nodes]
        builder = RecallMapBuilder(store)

        # Seed the delivery memo so the real `_note_delivery` path is visible:
        # journal-only builds may increase its observation count, but never its
        # offer count and therefore never advance the collapse decision.
        assert builder._curtailment(POOL_SCOPE, "negative-cache").offers == 0

        first = builder.build(
            residual,
            scope=POOL_SCOPE,
            task="negative-cache",
            decision_at="2026-08-23T12:00:00Z",
        )
        assert first is not None
        assert builder.last_cache_hit is False
        assert builder._cache[first.key].templates == ()
        first_revision = builder._cache[first.key].revision

        second = builder.build(
            residual,
            scope=POOL_SCOPE,
            task="negative-cache",
            decision_at="2026-08-23T12:00:00Z",
        )
        assert second is not None
        assert builder.last_cache_hit is True
        assert second.to_dict() == first.to_dict()
        assert second.selection == first.selection
        assert second.selection is not first.selection

        content = nodes[0].content
        store.replace_node_chunks(
            nodes[0].id,
            [
                (
                    TextChunk(
                        text=content,
                        chunk_index=0,
                        token_start=0,
                        token_end=1,
                        char_start=0,
                        char_end=len(content),
                    ),
                    [1.0, 0.0, 0.0],
                )
            ],
        )
        third = builder.build(
            residual,
            scope=POOL_SCOPE,
            task="negative-cache",
            decision_at="2026-08-23T12:00:00Z",
        )
        assert third is not None
        assert builder.last_cache_hit is False
        assert third.to_dict() == first.to_dict()
        assert third.selection is not second.selection
        assert builder._cache[first.key].revision != first_revision
        assert builder._cache[first.key].templates == ()

        memo = builder._curtail_memo[first.key]
        assert memo.total == 3
        assert memo.offers == 0
        assert memo.estimate().offers == 0
        assert memo.estimate().collapse is False
        assert builder.last_curtailment.offers == 0


def test_empty_and_populated_structures_transition_without_false_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "negative-cache-transitions.sqlite3") as store:
        # A measurable corpus lets the eligible structural group clear the
        # label gate, making any accidental suppression externally visible.
        for index in range(32):
            _node(store, f"CORPUS-{index}", f"unrelated ledger vocabulary {index}")
        low = _node(store, "TRANSITION-LOW", "fresh low relevance trace")
        eligible = _node(
            store,
            "TRANSITION-ELIGIBLE",
            "durable negative cache survivor",
            context={"procedure_id": "durable-negative-cache"},
        )

        def history(candidate_ids, _decision_at):
            return {
                node_id: (
                    MaturedRecallHistory.known(100, 50, 50)
                    if node_id == eligible.id
                    else MaturedRecallHistory.known(0, 0, 0)
                )
                for node_id in candidate_ids
            }

        monkeypatch.setattr(store, "matured_recall_history", history)
        builder = RecallMapBuilder(store)
        recount_calls: list[tuple[int, int]] = []
        original_recount = builder._recount

        def recount(templates, pool):
            recount_calls.append((len(templates), len(pool)))
            return original_recount(templates, pool)

        monkeypatch.setattr(builder, "_recount", recount)

        empty_first = builder.build(
            [RecallResult(node=low, score=10.0)],
            scope=POOL_SCOPE,
            task="negative-cache-transition",
            decision_at="2026-08-23T12:00:00Z",
        )
        assert empty_first is not None
        assert builder.last_cache_hit is False
        revision = builder._cache[empty_first.key].revision
        assert builder._cache[empty_first.key].templates == ()

        populated = builder.build(
            [RecallResult(node=eligible, score=10.0)],
            scope=POOL_SCOPE,
            task="negative-cache-transition",
            decision_at="2026-08-23T12:00:00Z",
        )
        assert populated is not None
        assert builder.last_cache_hit is False
        assert populated.pool_size == 1
        assert populated.selection is not None
        assert populated.selection.admitted == 1
        assert populated.clusters
        assert builder._cache[populated.key].revision == revision
        assert builder._cache[populated.key].templates

        empty_after_populated = builder.build(
            [RecallResult(node=low, score=10.0)],
            scope=POOL_SCOPE,
            task="negative-cache-transition",
            decision_at="2026-08-23T12:00:00Z",
        )
        assert empty_after_populated is not None
        assert builder.last_cache_hit is False
        assert empty_after_populated.clusters == ()
        assert empty_after_populated.selection is not None
        assert empty_after_populated.selection.excluded == (
            0,
            0,
            0,
            0,
            0,
            1,
            0,
        )
        assert builder._cache[empty_after_populated.key].templates == ()

        empty_again = builder.build(
            [RecallResult(node=low, score=10.0)],
            scope=POOL_SCOPE,
            task="negative-cache-transition",
            decision_at="2026-08-23T12:00:00Z",
        )
        assert empty_again is not None
        assert builder.last_cache_hit is True
        assert empty_again.to_dict() == empty_after_populated.to_dict()

        # Empty templates are handled only by the explicit negative-cache
        # branch. They are never passed to `_recount` as a wildcard structure.
        assert recount_calls == []


def test_pool_orders_by_descending_frozen_score_then_original_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "score-order.sqlite3") as store:
        head = _node(store, "HEAD", "retrieval head with lower relevance")
        tied_first = _node(store, "TIED-A", "first equal-score survivor")
        tied_second = _node(store, "TIED-B", "second equal-score survivor")

        def history(candidate_ids, _decision_at):
            return {
                node_id: (
                    MaturedRecallHistory.known(100, 50, 2)
                    if node_id == head.id
                    else MaturedRecallHistory.known(100, 50, 50)
                )
                for node_id in candidate_ids
            }

        monkeypatch.setattr(store, "matured_recall_history", history)
        selected, accounting = RecallMapBuilder(store, max_pool_nodes=2)._pool(
            [
                RecallResult(node=head, score=100.0),
                RecallResult(node=tied_first, score=1.0),
                RecallResult(node=tied_second, score=0.0),
            ],
            decision_at="2026-08-23T12:00:00Z",
        )

        assert [member.node.id for member in selected] == [
            tied_first.id,
            tied_second.id,
        ]
        assert selected[0].relevance_score == selected[1].relevance_score
        assert accounting.excluded == (0, 0, 0, 0, 0, 0, 1)
        assert accounting.samples[-1].ordinal == 0


# ----------------------------------------------------------------------
# The two env-gated pool gates
# ----------------------------------------------------------------------
#
# Both gates exist because `relevance_score` cannot see the two things that
# would sink a hub: a `usefulness_score` verdict, and the fact that nobody
# ever follows the row. Neither may enter the frozen evaluator, so both enter
# as admission rules around it, each behind its own valve.
#
# The load-bearing test in this section is not either gate — it is
# `test_both_valves_unset_is_byte_identical_to_the_pre_gate_build`, which
# pins the whole build, selection journal included, against a payload
# captured from the code as it stood before any of this existed.


GATE_SCOPE = POOL_SCOPE
GATE_TASK = "pool-gate-fixture"
GATE_INSTANT = "2026-08-23T12:00:00Z"

#: One build of the fixture scenario below, captured from `recall_map.py` as
#: it stood at commit e579fb4 — the parent of the commit that added the pool
#: gates — by running `_gate_fixture_payload` against that revision of the
#: module. Recapture with:
#:
#:     git show e579fb4:src/living_memory/recall_map.py > /tmp/pre_gate.py
#:     python - <<'EOF'
#:     import importlib.util, json, sys
#:     sys.path.insert(0, "src"); sys.path.insert(0, ".")
#:     spec = importlib.util.spec_from_file_location("pre_gate", "/tmp/pre_gate.py")
#:     module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
#:     from tests.test_recall_map_pool import _gate_fixture_payload
#:     print(json.dumps(_gate_fixture_payload(module), indent=4, sort_keys=True))
#:     EOF
#:
#: It is a literal rather than a generated file on purpose: a golden that the
#: test suite can regenerate is a golden that silently re-blesses whatever the
#: code does today.
#:
#: Note what the golden happens to contain, because it is the field failure
#: this whole goal exists for: the medoid of the first cluster is ``GATEHUB1``,
#: the row whose usefulness verdict is 0.078 and whose every observable window
#: went unfollowed. With both valves unset it is still the medoid, and that is
#: correct — the gates are the operator's to arm.
PRE_GATE_GOLDEN: dict[str, object] = {
    "clusters": [
        {
            "label": "reopen escalation lesson",
            "count": 2,
            "medoid": {
                "node_id": "GATEHUB1",
                "example": (
                    "reopen lesson: the escalation loop reopened the closed "
                    "goal twice"
                ),
            },
            "ask_hint": "reopen escalation lesson",
            "plan_item": (
                "on touching reopen escalation lesson - recall "
                "'reopen escalation lesson' (2)"
            ),
        },
        {
            "label": "migration rollback drill",
            "count": 2,
            "medoid": {
                "node_id": "GATEMIGR1",
                "example": "migration rollback restored the dropped embedding column",
            },
            "ask_hint": "migration rollback drill",
            "plan_item": (
                "on touching migration rollback drill - recall "
                "'migration rollback drill' (2)"
            ),
        },
    ],
    "pool": 4,
    "covered": 4,
    "sel": {
        "v": "r1",
        "n": 5,
        "e": 4,
        "x": [0, 0, 1, 0, 0, 0, 0],
        "o": 0,
        "q": [["fc", '[file-chunk] {"path":"s…']],
    },
}


#: Documents the label gate measures rarity against. Without a corpus every
#: label is house vocabulary on a five-node store and the gate withholds the
#: whole map, which would make the byte-identity fixture pin an empty payload.
#: Sized as ``tests/test_recall_map_curtail.py`` sizes its own.
GATE_CORPUS_DOCUMENTS = 32


def _gate_nodes(store: MemoryStore) -> list[Node]:
    """A deterministic residual: two structural clusters plus one ballast row.

    Structural (stage 1) on purpose. The label cascade's later stages read
    chunk vectors, and a fixture whose bytes depend on an embedding model is a
    fixture that pins the model rather than the gates.
    """

    for index in range(GATE_CORPUS_DOCUMENTS):
        store.create_node(
            level="trace",
            content=f"quarterly ledger reconciliation entry {index}",
            context={"scope": GATE_SCOPE},
        )
    nodes = [
        _node(
            store,
            "GATEHUB1",
            "reopen lesson: the escalation loop reopened the closed goal twice",
            context={"procedure_id": "reopen_escalation_lesson"},
            stats={"usefulness_score": 0.078},
        ),
        _node(
            store,
            "GATEHUB2",
            "reopen lesson: a reopened goal inherits the stale acceptance gate",
            context={"procedure_id": "reopen_escalation_lesson"},
            stats={"usefulness_score": 0.5},
        ),
        _node(
            store,
            "GATEMIGR1",
            "migration rollback restored the dropped embedding column",
            context={"procedure_id": "migration_rollback_drill"},
            stats={"usefulness_score": 0.9},
        ),
        _node(
            store,
            "GATEMIGR2",
            "migration rollback needed a vacuum before the snapshot fit",
            context={"procedure_id": "migration_rollback_drill"},
            stats={"usefulness_score": 0.9},
        ),
        _node(
            store,
            "GATEBALLST",
            "[file-chunk] " + _machine_header(1),
        ),
    ]
    return nodes


def _gate_history(node_ids, _decision_at):
    """Windows shaped like the field cases the gates were written for.

    ``GATEHUB1`` is the hub: every window re-delivered, every observable one
    unfollowed. ``GATEHUB2`` has the same unfollowed run with no re-delivery
    behind it. The two migration rows are followed and quiet respectively.
    """

    shapes = {
        "GATEHUB1": MaturedRecallHistory.known(
            100, 100, 0, lookup_consumed=0, lookup_known=8, lookup_trailing_absent=8
        ),
        "GATEHUB2": MaturedRecallHistory.known(
            100, 50, 4, lookup_consumed=0, lookup_known=4, lookup_trailing_absent=4
        ),
        "GATEMIGR1": MaturedRecallHistory.known(
            100, 50, 4, lookup_consumed=3, lookup_known=6, lookup_trailing_absent=0
        ),
        "GATEMIGR2": MaturedRecallHistory.known(
            100, 50, 4, lookup_consumed=0, lookup_known=0, lookup_trailing_absent=0
        ),
    }
    return {
        node_id: shapes.get(node_id, MaturedRecallHistory.known(100, 50, 4))
        for node_id in node_ids
    }


def _gate_fixture_payload(module) -> dict[str, object]:
    """Build the fixture scenario against ``module`` and return the payload.

    Takes the module rather than importing it so the same scenario can be run
    against the pre-gate revision of ``recall_map.py``, which is what makes
    :data:`PRE_GATE_GOLDEN` evidence instead of a restatement.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        with MemoryStore(Path(directory) / "gate-fixture.sqlite3") as store:
            nodes = _gate_nodes(store)
            store.matured_recall_history = _gate_history  # type: ignore[method-assign]
            built = module.RecallMapBuilder(store).build(
                [
                    RecallResult(node=node, score=100.0 - index)
                    for index, node in enumerate(nodes)
                ],
                scope=GATE_SCOPE,
                task=GATE_TASK,
                decision_at=GATE_INSTANT,
            )
            assert built is not None
            return built.to_dict()


@pytest.fixture(autouse=True)
def _valves_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file starts with both valves off.

    An inherited valve from the ambient environment would make the whole file
    lie, and the byte-identity test most of all.
    """

    for name in (
        POOL_USEFULNESS_GATE_ENV,
        POOL_USEFULNESS_FLOOR_ENV,
        POOL_DEMOTION_GATE_ENV,
        POOL_DEMOTION_WINDOWS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_both_valves_unset_is_byte_identical_to_the_pre_gate_build() -> None:
    """The whole payload, not just the cluster list.

    ``sel`` is where a gate would leak first: it carries the exclusion vector,
    whose *length* changes the moment a gate code fires, and the identity-free
    examples. Comparing the serialized JSON compares both, plus the clusters,
    counts, medoids, filter block and everything else the server persists.
    """

    payload = _gate_fixture_payload(recall_map_module)

    assert payload == PRE_GATE_GOLDEN
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        PRE_GATE_GOLDEN, sort_keys=True
    )
    # The frozen seven-wide vector, unextended: this is the byte the gates
    # could most easily cost without changing a single cluster.
    assert len(payload["sel"]["x"]) == len(SELECTION_REASON_CODES) == 7


def test_the_frozen_wire_prefix_is_untouched_and_the_gate_codes_append() -> None:
    assert SELECTION_REASON_CODES == ("iv", "du", "fc", "ss", "sj", "lr", "pc")
    assert POOL_GATE_REASON_CODES == ("uf", "nf")
    assert SELECTION_LEDGER_REASON_CODES == (
        *SELECTION_REASON_CODES,
        *POOL_GATE_REASON_CODES,
    )
    # Both gate codes name something an operator set by hand, so both are
    # sampleable -- a journal that counts without ever naming gives the
    # operator no way to judge the number they chose.
    assert {"uf", "nf"} <= recall_map_module._SAMPLEABLE_REASONS


# -- gate (a): the usefulness floor ------------------------------------


def test_usefulness_gate_excludes_under_its_own_reason_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "useful.sqlite3") as store:
        sunk = _node(
            store,
            "USELESS1",
            "a verdict of nearly zero on a node the map keeps offering",
            stats={"usefulness_score": 0.078},
        )
        kept = _node(
            store,
            "USEFUL01",
            "a node the corpus actually rewarded",
            stats={"usefulness_score": 0.9},
        )
        # Exactly at the floor. "Below" is strict, so this one stays: an
        # operator setting the floor to a value their census reported has to
        # be able to predict which side of it that value lands on.
        at_floor = _node(
            store,
            "ATFLOOR01",
            "a node sitting exactly on the operator's number",
            stats={"usefulness_score": 0.25},
        )

        def history(candidate_ids, _decision_at):
            return {
                node_id: MaturedRecallHistory.known(100, 50, 4)
                for node_id in candidate_ids
            }

        monkeypatch.setattr(store, "matured_recall_history", history)
        results = [
            RecallResult(node=sunk, score=100.0),
            RecallResult(node=kept, score=99.0),
            RecallResult(node=at_floor, score=98.0),
        ]
        builder = RecallMapBuilder(store)

        off_pool, off_accounting = builder._pool(results, decision_at=GATE_INSTANT)
        assert [member.node.id for member in off_pool] == [
            sunk.id,
            kept.id,
            at_floor.id,
        ]
        assert off_accounting.excluded == (0, 0, 0, 0, 0, 0, 0)

        monkeypatch.setenv(POOL_USEFULNESS_GATE_ENV, "on")
        monkeypatch.setenv(POOL_USEFULNESS_FLOOR_ENV, "0.25")
        on_pool, on_accounting = builder._pool(results, decision_at=GATE_INSTANT)

        assert [member.node.id for member in on_pool] == [kept.id, at_floor.id]
        assert on_accounting.excluded == (0, 0, 0, 0, 0, 0, 0, 1)
        assert on_accounting.inspected == on_accounting.admitted + sum(
            on_accounting.excluded
        )

        compact = on_accounting.to_dict()
        assert compact["x"] == [0, 0, 0, 0, 0, 0, 0, 1]
        assert [sample[0] for sample in compact["q"]] == ["uf"]
        gist = compact["q"][0][1]
        assert gist and len(gist) <= SELECTION_SAMPLE_GIST_CHARS
        assert sunk.id not in gist


def test_usefulness_gate_reads_the_floor_off_the_node_not_a_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected candidate never reaches the bounded history batch."""

    with MemoryStore(tmp_path / "useful-batch.sqlite3") as store:
        sunk = _node(
            store, "USELESS2", "below the floor", stats={"usefulness_score": -0.4}
        )
        kept = _node(
            store, "USEFUL02", "above the floor", stats={"usefulness_score": 0.4}
        )

        batches: list[list[str]] = []

        def history(candidate_ids, _decision_at):
            ids = list(candidate_ids)
            batches.append(ids)
            return {
                node_id: MaturedRecallHistory.known(100, 50, 4) for node_id in ids
            }

        monkeypatch.setattr(store, "matured_recall_history", history)
        monkeypatch.setenv(POOL_USEFULNESS_GATE_ENV, "1")
        monkeypatch.setenv(POOL_USEFULNESS_FLOOR_ENV, "0")

        RecallMapBuilder(store)._pool(
            [
                RecallResult(node=sunk, score=100.0),
                RecallResult(node=kept, score=99.0),
            ],
            decision_at=GATE_INSTANT,
        )

        assert batches == [[kept.id]]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("on", ""),           # armed with no number: inert, not a guess
        ("on", "not-a-float"),
        ("on", "nan"),
        ("0", "0.25"),        # a number with no valve is still off
        ("", "0.25"),
        ("yes-please", "0.25"),
    ],
)
def test_usefulness_gate_is_inert_unless_both_halves_are_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    with MemoryStore(tmp_path / f"inert-{abs(hash((flag, value)))}.sqlite3") as store:
        sunk = _node(
            store, "USELESS3", "far below any floor", stats={"usefulness_score": -1.0}
        )

        def history(candidate_ids, _decision_at):
            return {
                node_id: MaturedRecallHistory.known(100, 50, 4)
                for node_id in candidate_ids
            }

        monkeypatch.setattr(store, "matured_recall_history", history)
        monkeypatch.setenv(POOL_USEFULNESS_GATE_ENV, flag)
        monkeypatch.setenv(POOL_USEFULNESS_FLOOR_ENV, value)

        assert recall_map_module.pool_usefulness_floor_from_env() is None
        pool, accounting = RecallMapBuilder(store)._pool(
            [RecallResult(node=sunk, score=100.0)], decision_at=GATE_INSTANT
        )
        assert [member.node.id for member in pool] == [sunk.id]
        assert accounting.excluded == (0, 0, 0, 0, 0, 0, 0)


def test_a_node_with_no_usefulness_attribute_is_not_read_as_worthless() -> None:
    """The duck-typed face, not the corpus.

    ``nodes.usefulness_score`` is ``NOT NULL DEFAULT 0.0``, so an unscored row
    really is a zero and is judged like one — the floor comes from a census
    that counted those zeroes. What must never happen is a *result object*
    from another layer, one that simply has no such attribute, being read as
    worthless because the attribute is missing.
    """

    below = recall_map_module._below_usefulness
    assert below(SimpleNamespace(usefulness_score=0.0), 0.25) is True
    assert below(SimpleNamespace(), 0.25) is False
    assert below(SimpleNamespace(usefulness_score=None), 0.25) is False
    assert below(SimpleNamespace(usefulness_score="0.0"), 0.25) is False
    assert below(SimpleNamespace(usefulness_score=True), 0.25) is False
    assert below(SimpleNamespace(usefulness_score=float("nan")), 0.25) is False


@pytest.mark.parametrize(
    "flag,expected", [("1", 3), ("true", 3), ("yes", 3), ("on", 3), ("On", 3)]
)
def test_the_valves_parse_the_drain_valve_spelling(
    monkeypatch: pytest.MonkeyPatch, flag: str, expected: int
) -> None:
    monkeypatch.setenv(POOL_DEMOTION_GATE_ENV, flag)
    monkeypatch.setenv(POOL_DEMOTION_WINDOWS_ENV, "3")
    assert recall_map_module.pool_demotion_windows_from_env() == expected

    monkeypatch.setenv(POOL_USEFULNESS_GATE_ENV, flag)
    monkeypatch.setenv(POOL_USEFULNESS_FLOOR_ENV, "0.25")
    assert recall_map_module.pool_usefulness_floor_from_env() == 0.25


@pytest.mark.parametrize("windows", ["", "0", "-2", "3.5", "three"])
def test_demotion_valve_rejects_a_window_count_that_is_not_a_count(
    monkeypatch: pytest.MonkeyPatch, windows: str
) -> None:
    monkeypatch.setenv(POOL_DEMOTION_GATE_ENV, "on")
    monkeypatch.setenv(POOL_DEMOTION_WINDOWS_ENV, windows)
    assert recall_map_module.pool_demotion_windows_from_env() is None


# -- gate (b): the unfollowed-window demotion ---------------------------


def _demoted(history, *, ask_followed: bool = False, after: int = 3) -> bool:
    return recall_map_module._demoted(
        history, ask_followed=ask_followed, after=after
    )


def test_demotion_counts_only_known_windows_and_skips_the_unknown_ones() -> None:
    """NULL is not a third flavour of "nobody followed it"."""

    # Three known windows, none followed: the row sinks at N=3.
    assert _demoted(
        MaturedRecallHistory.known(
            3, 0, 3, lookup_consumed=0, lookup_known=3, lookup_trailing_absent=3
        )
    )
    # The same three windows plus a hundred unobservable ones: still three.
    # The run is what storage counted, and it stepped over every NULL.
    assert _demoted(
        MaturedRecallHistory.known(
            103, 0, 103, lookup_consumed=0, lookup_known=3, lookup_trailing_absent=3
        )
    )
    # A corpus that has never recorded a lookup at all -- every window NULL --
    # demotes nothing. This is the cold start the tri-state column exists for.
    assert not _demoted(
        MaturedRecallHistory.known(
            500, 0, 500, lookup_consumed=0, lookup_known=0, lookup_trailing_absent=0
        )
    )
    # Two known unfollowed windows are one short of the operator's N.
    assert not _demoted(
        MaturedRecallHistory.known(
            50, 0, 50, lookup_consumed=0, lookup_known=2, lookup_trailing_absent=2
        )
    )


def test_a_lookup_inside_the_run_stops_it() -> None:
    followed = MaturedRecallHistory.known(
        20, 0, 20, lookup_consumed=1, lookup_known=9, lookup_trailing_absent=0
    )
    assert not _demoted(followed)


def test_redelivery_softens_the_demotion_and_can_never_veto_it() -> None:
    """The property the whole goal turns on.

    A hub satisfies re-delivery on every window it was ever offered in --
    that is what makes it a hub -- so a rule re-delivery could veto would make
    hubs immortal. Softening is monotone in the re-delivery share and bottoms
    out at the weight, never at zero.
    """

    def hub(run: int) -> MaturedRecallHistory:
        return MaturedRecallHistory.known(
            100, 100, 0, lookup_consumed=0, lookup_known=run,
            lookup_trailing_absent=run,
        )

    def quiet(run: int) -> MaturedRecallHistory:
        return MaturedRecallHistory.known(
            100, 0, 100, lookup_consumed=0, lookup_known=run,
            lookup_trailing_absent=run,
        )

    # Softened: what sinks a never-re-delivered row leaves the hub standing.
    assert _demoted(quiet(3), after=3)
    assert not _demoted(hub(3), after=3)
    assert not _demoted(hub(5), after=3)

    # Not vetoed: the hub sinks too, it just takes 2N windows at a weight of
    # one half. This is the arithmetic, stated as an assertion so a future
    # weight of 0 fails here rather than in the field.
    assert _demoted(hub(6), after=3)
    assert recall_map_module.DEMOTION_REDELIVERY_WEIGHT < 1.0
    assert recall_map_module.DEMOTION_REDELIVERY_WEIGHT > 0.0

    # Monotone in between: half the windows re-delivered lands between the two.
    half = MaturedRecallHistory.known(
        100, 50, 50, lookup_consumed=0, lookup_known=4, lookup_trailing_absent=4
    )
    assert _demoted(half, after=3)
    assert not _demoted(
        MaturedRecallHistory.known(
            100, 50, 50, lookup_consumed=0, lookup_known=3,
            lookup_trailing_absent=3,
        ),
        after=3,
    )


def test_an_ask_follow_holds_the_row_and_an_unreadable_history_abstains() -> None:
    sinking = MaturedRecallHistory.known(
        10, 0, 10, lookup_consumed=0, lookup_known=6, lookup_trailing_absent=6
    )
    assert _demoted(sinking)
    assert not _demoted(sinking, ask_followed=True)

    for reason in ("history_truncated", "history_read_failed", "history_format_unavailable"):
        assert not _demoted(MaturedRecallHistory.unavailable(reason))
    assert not _demoted(None)
    assert not _demoted(SimpleNamespace(available=True, matured="8"))
    # Self-inconsistent rows abstain rather than being repaired into a verdict.
    assert not _demoted(
        MaturedRecallHistory.known(
            2, 0, 2, lookup_consumed=0, lookup_known=9, lookup_trailing_absent=9
        )
    )


def test_demotion_gate_excludes_under_nf_after_the_frozen_threshold_spoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``nf`` counts what the gate newly removed, never what ``lr`` dropped."""

    with MemoryStore(tmp_path / "demote.sqlite3") as store:
        hub = _node(store, "DEMOTEHUB", "the hub the map keeps offering")
        quiet = _node(store, "DEMOTEQUI", "a row nobody has ever followed")
        low = _node(store, "DEMOTELOW", "a row the frozen threshold rejects")

        def history(candidate_ids, _decision_at):
            shapes = {
                hub.id: MaturedRecallHistory.known(
                    100, 100, 0, lookup_consumed=0, lookup_known=9,
                    lookup_trailing_absent=9,
                ),
                quiet.id: MaturedRecallHistory.known(
                    100, 50, 4, lookup_consumed=0, lookup_known=4,
                    lookup_trailing_absent=4,
                ),
                low.id: MaturedRecallHistory.known(0, 0, 0),
            }
            return {node_id: shapes[node_id] for node_id in candidate_ids}

        monkeypatch.setattr(store, "matured_recall_history", history)
        results = [
            RecallResult(node=hub, score=100.0),
            RecallResult(node=quiet, score=99.0),
            RecallResult(node=low, score=98.0),
        ]
        builder = RecallMapBuilder(store)

        off_pool, off_accounting = builder._pool(results, decision_at=GATE_INSTANT)
        assert [member.node.id for member in off_pool] == [hub.id, quiet.id]
        assert off_accounting.excluded == (0, 0, 0, 0, 0, 1, 0)

        monkeypatch.setenv(POOL_DEMOTION_GATE_ENV, "on")
        monkeypatch.setenv(POOL_DEMOTION_WINDOWS_ENV, "3")
        on_pool, on_accounting = builder._pool(
            results,
            decision_at=GATE_INSTANT,
            demote_after=recall_map_module.pool_demotion_windows_from_env(),
        )

        assert on_pool == []
        # `lr` still reports exactly the row the frozen threshold rejected --
        # the gate did not re-label it -- and `nf` reports the two it sank.
        assert on_accounting.excluded == (0, 0, 0, 0, 0, 1, 0, 0, 2)
        compact = on_accounting.to_dict()
        assert compact["x"] == [0, 0, 0, 0, 0, 1, 0, 0, 2]
        assert [sample[0] for sample in compact["q"]] == ["lr", "nf"]


def test_ask_echo_probe_comes_off_the_curtail_read_and_holds_its_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the gate consults the same window curtail judges by."""

    with MemoryStore(tmp_path / "ask-echo.sqlite3") as store:
        for index in range(GATE_CORPUS_DOCUMENTS):
            store.create_node(
                level="trace",
                content=f"quarterly ledger reconciliation entry {index}",
                context={"scope": GATE_SCOPE},
            )
        asked = _node(
            store,
            "ASKEDROW1",
            "migration rollback drill notes",
            context={"procedure_id": "migration_rollback_drill"},
        )
        ignored = _node(
            store,
            "IGNOREDR1",
            "reopen escalation lesson notes",
            context={"procedure_id": "reopen_escalation_lesson"},
        )

        def history(candidate_ids, _decision_at):
            # Comfortably over the frozen threshold, so whatever leaves the
            # pool here left because of the gate and nothing else.
            return {
                node_id: MaturedRecallHistory.known(
                    100, 50, 4, lookup_consumed=0, lookup_known=5,
                    lookup_trailing_absent=5,
                )
                for node_id in candidate_ids
            }

        monkeypatch.setattr(store, "matured_recall_history", history)
        monkeypatch.setenv(POOL_DEMOTION_GATE_ENV, "on")
        monkeypatch.setenv(POOL_DEMOTION_WINDOWS_ENV, "3")

        results = [
            RecallResult(node=asked, score=100.0),
            RecallResult(node=ignored, score=99.0),
        ]
        builder = RecallMapBuilder(store)

        # Nothing delivered yet: no ask evidence exists, so both rows sink.
        blind = builder.build(
            results, scope=GATE_SCOPE, task=GATE_TASK, decision_at=GATE_INSTANT
        )
        assert blind is not None and blind.clusters == ()
        assert builder.last_ask_follows == frozenset()

        # Now stage a window where one cluster's ask-hint was echoed by a
        # later query, exactly as `_was_followed` reads it.
        delivered = {
            "clusters": [
                {
                    "label": "migration rollback drill",
                    "ask_hint": "migration rollback drill",
                    "count": 1,
                    "medoid": {"node_id": asked.id, "example": "notes"},
                },
                {
                    "label": "reopen escalation lesson",
                    "ask_hint": "reopen escalation lesson",
                    "count": 1,
                    "medoid": {"node_id": ignored.id, "example": "notes"},
                },
            ]
        }
        store.record_recall_event(
            query="anything at all",
            scope=GATE_SCOPE,
            ambient_context={"task": GATE_TASK},
            results=[],
            recall_map=delivered,
        )
        store.record_recall_event(
            query="migration rollback drill please",
            scope=GATE_SCOPE,
            ambient_context={"task": GATE_TASK},
            results=[],
            recall_map={"clusters": []},
        )

        fresh = RecallMapBuilder(store)
        seeing = fresh.build(
            results, scope=GATE_SCOPE, task=GATE_TASK, decision_at=GATE_INSTANT
        )
        assert fresh.last_ask_follows == frozenset({asked.id})
        assert seeing is not None
        assert [cluster.medoid.node_id for cluster in seeing.clusters] == [asked.id]


def test_arming_both_gates_sinks_the_hub_the_golden_still_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the byte-identity claim: the gates are not inert.

    Same scenario, same store, same history — only the valves move. The
    fixture's first cluster was medoided by ``GATEHUB1``: usefulness 0.078,
    re-delivered on every one of its hundred windows, followed on none of the
    eight observable ones. It is exactly the row the field complaint is about,
    and with both valves armed it is gone, along with the whole cluster it was
    the medoid of.
    """

    monkeypatch.setenv(POOL_USEFULNESS_GATE_ENV, "on")
    monkeypatch.setenv(POOL_USEFULNESS_FLOOR_ENV, "0.25")
    monkeypatch.setenv(POOL_DEMOTION_GATE_ENV, "on")
    monkeypatch.setenv(POOL_DEMOTION_WINDOWS_ENV, "3")

    armed = _gate_fixture_payload(recall_map_module)

    assert armed != PRE_GATE_GOLDEN
    medoids = [cluster["medoid"]["node_id"] for cluster in armed["clusters"]]
    assert "GATEHUB1" not in medoids
    assert medoids == ["GATEMIGR1"]

    # `uf` took the 0.078 row before its history was even read; `nf` took the
    # one whose verdict was fine and whose windows were not. The ballast row
    # is still `fc`, at the index it always had.
    assert armed["sel"]["x"] == [0, 0, 1, 0, 0, 0, 0, 1, 1]
    assert PRE_GATE_GOLDEN["sel"]["x"] == [0, 0, 1, 0, 0, 0, 0]
    assert armed["sel"]["x"][: len(SELECTION_REASON_CODES)] == (
        PRE_GATE_GOLDEN["sel"]["x"]
    )


def test_the_exclusion_vector_has_exactly_one_encoding_per_outcome() -> None:
    """Trimming is an invariant, not a convention.

    If a nine-wide all-zero tail were accepted alongside the seven-wide form,
    "both valves unset" would have two legal spellings and the byte-identity
    claim above would be untestable.
    """

    accounting = recall_map_module.SelectionAccounting

    # The canonical forms round-trip.
    assert accounting(inspected=1, admitted=1, excluded=(0,) * 7).to_dict()["x"] == [
        0
    ] * 7
    assert accounting(
        inspected=2, admitted=1, excluded=(0, 0, 0, 0, 0, 0, 0, 1)
    ).to_dict()["x"] == [0, 0, 0, 0, 0, 0, 0, 1]

    # A padded tail is a different spelling of the same fact, and is rejected.
    for padded in ((0,) * 8, (0,) * 9, (0, 0, 0, 0, 0, 0, 0, 1, 0)):
        with pytest.raises(ValueError, match="canonically trimmed"):
            accounting(
                inspected=sum(padded) + 1, admitted=1, excluded=padded
            )

    # And so is a vector wider than every code this module knows.
    with pytest.raises(ValueError, match="wrong width"):
        accounting(inspected=1, admitted=1, excluded=(0,) * 6)
    with pytest.raises(ValueError, match="wrong width"):
        accounting(inspected=2, admitted=1, excluded=(0,) * 9 + (1,))
