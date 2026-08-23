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

from living_memory.chunking import TextChunk
from living_memory.models import Node
from living_memory.recall_map import (
    MAX_POOL_NODES,
    RELEVANCE_FEATURE_MEANS,
    RELEVANCE_FEATURE_SCALES,
    RELEVANCE_THRESHOLD,
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
) -> Node:
    return store.create_node(
        level="trace",
        content=content,
        context={"scope": POOL_SCOPE, **(context or {})},
        provenance=provenance,
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
