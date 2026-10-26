"""Query-qualified map residuals, diversity, and complete selection accounting."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from living_memory import recall_map as recall_map_module
from living_memory.models import Node
from living_memory.recall_map import (
    MAX_POOL_NODES, MAX_RESPONSE_CHARS, POOL_GATE_REASON_CODES,
    POOL_USEFULNESS_FLOOR_ENV, POOL_USEFULNESS_GATE_ENV,
    SELECTION_LEDGER_REASON_CODES, SELECTION_REASON_CODES,
    SELECTION_SAMPLE_GIST_CHARS, RecallMapBuilder, _ballast_reason,
)
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.score_gate import QueryRelevance, apply_score_gate
from living_memory.storage import MaturedRecallHistory, MemoryStore
from test_recall_map import seed_corpus

POOL_QUERY = "deployment failure database migration rollback"
POOL_SCOPE = "project:alpha"
FROZEN_SEL_CODE_INDEX = dict(zip(("iv", "du", "fc", "ss", "sj", "lr", "pc"), range(7)))

@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in ("LM_RECALL_MIN_SCORE", "LM_RECALL_GATE_FORM",
                 "LM_MAP_POOL_USEFULNESS_GATE", "LM_MAP_POOL_MIN_USEFULNESS",
                 "LM_MAP_CURTAIL_DECAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")

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



def test_sel_contract_selection_reason_codes_never_move_from_their_positions() -> None:
    """A reorder or a mid-tuple insertion re-means every historical `sel.q`.

    Both persistence and the evaluator read ``x`` positionally, so this tuple
    is an index assignment that happens to be spelled as a sequence. Comparing
    the whole code-to-index mapping (rather than just the tuple) is what makes
    the failure name the code that moved and the index it moved to.
    """

    assert SELECTION_REASON_CODES == ("iv", "du", "fc", "ss", "sj", "lr", "pc")
    assert {
        code: index for index, code in enumerate(SELECTION_REASON_CODES)
    } == FROZEN_SEL_CODE_INDEX
    # Seven is the width the wire promises, and duplicates would make one code
    # unreachable by `.index()` while leaving the tuple the right length.
    assert len(SELECTION_REASON_CODES) == len(set(SELECTION_REASON_CODES)) == 7



def test_sel_contract_pool_gate_reason_codes_are_append_only() -> None:
    """New gate codes may extend the vector; they may never displace one.

    The frozen seven keep their indices because the gate codes start at index
    seven, and the gate codes keep theirs because a new one goes on the end.
    This asserts the *prefix* on purpose: a third gate is a legal change and
    must not be reported here as a contract breach.
    """

    assert POOL_GATE_REASON_CODES[: len(("uf", "nf"))] == ("uf", "nf")
    assert POOL_GATE_REASON_CODES.index("uf") == 0
    assert POOL_GATE_REASON_CODES.index("nf") == 1

    frozen_width = len(SELECTION_REASON_CODES)
    assert SELECTION_LEDGER_REASON_CODES[:frozen_width] == SELECTION_REASON_CODES
    assert SELECTION_LEDGER_REASON_CODES[frozen_width:] == POOL_GATE_REASON_CODES
    # `uf` is at absolute index 7 and `nf` at 8 for the same reason the frozen
    # seven are where they are: something stored reads them by number.
    assert SELECTION_LEDGER_REASON_CODES.index("uf") == frozen_width
    assert SELECTION_LEDGER_REASON_CODES.index("nf") == frozen_width + 1
    # A gate code that collided with a frozen one would make the combined
    # vector ambiguous without changing either tuple's length.
    assert not set(POOL_GATE_REASON_CODES) & set(SELECTION_REASON_CODES)
    assert len(set(SELECTION_LEDGER_REASON_CODES)) == len(
        SELECTION_LEDGER_REASON_CODES
    )



def test_sel_contract_an_unarmed_ledger_freezes_to_exactly_seven_counts() -> None:
    """No frozen-code traffic, however heavy, may widen the vector.

    ``_SelectionLedger.freeze`` builds the vector over every code this module
    knows and then trims the tail. On a server with neither valve charged the
    gate slots are always zero, so the trim must always reach the frozen width
    -- which is what makes "seven counts" the single, checkable signature of an
    unarmed build.
    """

    ledger = recall_map_module._SelectionLedger

    # The empty residual: still seven, still all zero.
    assert ledger(inspected=0).freeze(0).excluded == (0,) * 7

    # An admitted-only residual, which is the common live shape.
    assert ledger(inspected=3).freeze(3).excluded == (0,) * 7

    # Every frozen code fired at once.
    loaded = ledger(inspected=7)
    for ordinal, code in enumerate(SELECTION_REASON_CODES):
        loaded.exclude(code, ordinal=ordinal)
    assert loaded.freeze(0).excluded == (1,) * 7

    # And each one alone -- including `pc`, the last frozen slot, whose count
    # is the one a naive trim would be tempted to treat as the tail.
    for index, code in enumerate(SELECTION_REASON_CODES):
        solo = ledger(inspected=2)
        solo.exclude(code, ordinal=0)
        solo.exclude(code, ordinal=1)
        frozen = solo.freeze(0).excluded
        assert len(frozen) == 7, code
        assert frozen[index] == 2, code
        assert sum(frozen) == 2, code



def test_sel_contract_only_an_armed_gate_widens_the_frozen_seven() -> None:
    """The other half of the trim: a wider vector *states* that a gate fired.

    A reader that finds eight or nine counts has learned something true about
    the server that emitted them, and that inference is only sound while
    arming a gate is the sole way to get there. The ``nf``-alone case is the
    load-bearing one: trimming stops at the last non-zero, so ``uf``'s slot
    survives as an explicit zero rather than being squeezed out and shifting
    ``nf`` down onto index 7.
    """

    ledger = recall_map_module._SelectionLedger

    only_uf = ledger(inspected=1)
    only_uf.exclude("uf", ordinal=0)
    assert only_uf.freeze(0).excluded == (0, 0, 0, 0, 0, 0, 0, 1)

    only_nf = ledger(inspected=1)
    only_nf.exclude("nf", ordinal=0)
    assert only_nf.freeze(0).excluded == (0, 0, 0, 0, 0, 0, 0, 0, 1)

    both = ledger(inspected=5)
    both.exclude("fc", ordinal=0)
    both.exclude("uf", ordinal=1)
    both.exclude("uf", ordinal=2)
    both.exclude("nf", ordinal=3)
    frozen = both.freeze(1).excluded
    assert frozen == (0, 0, 1, 0, 0, 0, 0, 2, 1)
    # The frozen prefix is untouched by the extension -- a reader that knows
    # only the seven keeps reading them at the same indices.
    assert frozen[: len(SELECTION_REASON_CODES)] == (0, 0, 1, 0, 0, 0, 0)
    assert frozen[len(SELECTION_REASON_CODES) :] == (2, 1)

    # Whatever the width, the accounting equation still closes over it.
    for accounting in (
        only_uf.freeze(0),
        only_nf.freeze(0),
        both.freeze(1),
    ):
        assert accounting.inspected == accounting.admitted + sum(accounting.excluded)



def test_ballast_forms_and_prose_lookalikes():
    cases = _independent_form_controls()
    assert len(cases) == 8 * 64
    for expected, content, context, provenance in cases:
        assert _ballast_reason(content, context, provenance) == expected
    assert sum(expected is not None for expected, *_ in cases) == 3 * 64


def result(node, evidence=1.0, **kwargs):
    return RecallResult(node=node, score=evidence, bm25_score=evidence, lexical_score=evidence,
                        vector_score=evidence, **kwargs)


@pytest.mark.parametrize("histories", [((2, 1, 1), (100, 50, 50)),
                                      ((100, 50, 50), (2, 1, 1)),
                                      ((0, 0, 0), (100, 50, 50))])
def test_query_evidence_survives_history_swap_and_cold_start(tmp_path, monkeypatch, histories):
    # Exercise the supplied two-candidate consumer counterexample with the
    # actual builder and SQLite. Histories differ only in exposure volume.
    monkeypatch.setenv("LM_MAP_POOL_COLD_QUOTA_GATE", "1")
    monkeypatch.setenv("LM_MAP_POOL_COLD_SLOTS", "2")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    with MemoryStore(tmp_path / "query.sqlite3") as store:
        seed_corpus(store)
        relevant = _node(store, "relevant", "explicit feedback query credit protocol",
                         context={"topic": "feedback credit"})
        unrelated = _node(store, "unrelated", "orchard irrigation inspection schedule",
                          context={"topic": "orchard irrigation"})
        ledger = dict(zip((relevant.id, unrelated.id), histories))
        history = Mock(side_effect=lambda ids, at: {
            key: MaturedRecallHistory.known(*ledger[key]) for key in ids
        })
        monkeypatch.setattr(store, "matured_recall_history", history)
        built = RecallMapBuilder(store).build([result(relevant), result(unrelated, 0.0)],
                                             scope=POOL_SCOPE)
        assert built and [c.medoid.node_id for c in built.clusters] == [relevant.id]
        assert built.selection.excluded == (0, 0, 0, 0, 0, 1, 0)
        history.assert_not_called()
        assert len(json.dumps(built.to_dict(), ensure_ascii=False, separators=(",", ":"))) <= MAX_RESPONSE_CHARS


def test_relevant_minority_topics_share_capacity_before_clustering(tmp_path):
    with MemoryStore(tmp_path / "diverse.sqlite3") as store:
        seed_corpus(store)
        majority = [_node(store, f"major-{i}", "recover from wal redo",
                          context={"topic": "wal redo"}) for i in range(8)]
        minority = _node(store, "minority", "flush disk sync",
                         context={"topic": "disk sync"})
        transfer = _node(store, "transfer", "verify log audit",
                         context={"scope": "project:other", "topic": "log audit"})
        candidates = [*(result(n) for n in majority), result(minority, 0.7), result(transfer, 0.6)]
        builder = RecallMapBuilder(store, max_pool_nodes=4)
        built = builder.build(candidates, scope="global", task="recovery")
        assert built and built.pool_size == 4
        represented = {node_id for c in built.clusters for node_id in c.member_ids}
        assert {minority.id, transfer.id} <= represented
        assert len(built.clusters) == 3
        assert built.selection.excluded[6] == 6
        assert built.selection.inspected == built.selection.admitted + sum(built.selection.excluded)
        assert builder.build(candidates, scope="global", task="recovery") == built
        assert builder.last_cache_hit


def test_complete_residual_accounting_and_no_history_read(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "accounting.sqlite3") as store:
        head = _node(store, "head", "qualified residual")
        file_chunk = _node(store, "file", f"[file-chunk] {_machine_header(1)}\nbody")
        stagnation = _node(store, "stagnation", "Strategy stagnation detected on selector\nreason: unchanged")
        journal = _node(store, "journal", "journal", provenance={"record_kind": "goal-tree-journal"})
        unrelated = _node(store, "unrelated", "no query evidence")
        tail = [_node(store, f"tail-{i}", "qualified residual") for i in range(MAX_POOL_NODES + 1)]
        raw = [SimpleNamespace(node=None), SimpleNamespace(node=SimpleNamespace(id=" ")),
               result(head), result(head), result(file_chunk), result(stagnation),
               result(journal), result(unrelated, 0.0), *(result(n) for n in tail)]
        monkeypatch.setattr(store, "matured_recall_history", Mock(side_effect=AssertionError("history must not select")))
        pool, selection = RecallMapBuilder(store)._pool(raw)
        assert len(pool) == MAX_POOL_NODES
        assert [m.rank for m in pool] == [2, *range(8, 207)]
        assert selection.inspected == 209
        assert selection.excluded == (2, 1, 1, 1, 1, 1, 2)
        assert selection.samples[-1].ordinal == 207
        assert selection.to_dict()["n"] == selection.admitted + sum(selection.excluded)


@pytest.mark.parametrize("form", ["drop", "stub"])
def test_map_shares_query_gate_context_and_correction_inheritance(tmp_path, monkeypatch, form):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    with MemoryStore(tmp_path / "corrections.sqlite3") as store:
        seed_corpus(store)
        lead = _node(store, "lead", "recall head", context={"topic": "head procedure"})
        correction = _node(store, "correction", "replacement transport rule", context={"topic": "transport correction"})
        original = _node(store, "original", "original transport rule", context={"topic": "transport original"})
        demoted = _node(store, "demoted", "contextually unrelated high vector", context={"topic": "other context"})
        ranked = [result(lead), result(correction, 0), result(original), result(demoted)]
        evidence = QueryRelevance(ranked, demotions={demoted.id: 0.01},
                                  corrections_by_superseded={original.id: [correction.id]})
        selected, residual = apply_score_gate(ranked, 1, relevance=evidence)
        assert selected == [ranked[0]]
        built = RecallMapBuilder(store).build(residual, scope=POOL_SCOPE, relevance=evidence)
        assert built
        mapped = {n for c in built.clusters for n in c.member_ids}
        assert correction.id in mapped and demoted.id not in mapped
        assert built.selection.excluded[5] == 1


@pytest.mark.parametrize("form", ["drop", "stub"])
def test_main_and_map_share_source_qualification_after_positive_history(tmp_path, monkeypatch, form):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    with MemoryStore(tmp_path / "source.sqlite3") as store:
        seed_corpus(store)
        lead = _node(store, "lead", "recall head", context={"topic": "head procedure"})
        weak = _node(store, "weak", "query source evidence", context={"topic": "query protocol"})
        fresh = _node(store, "fresh", "current source relation", context={"topic": "source relation"})
        builder = RecallMapBuilder(store)
        for history in (False, True):
            reinforced = dataclasses.replace(weak, access_count=1000, usefulness_score=1) if history else weak
            ranked = [result(lead),
                      RecallResult(reinforced, score=2 if history else 0.3,
                                   bm25_score=0.75, lexical_score=0.75, graph_score=1.5 if history else 0),
                      RecallResult(fresh, score=0.4, bm25_score=0.75, lexical_score=0.75,
                                   graph_score=0.5, source_graph_score=0.5)]
            evidence = QueryRelevance(ranked)
            selected, residual = apply_score_gate(ranked, 1, relevance=evidence)
            assert selected == ranked[:1]
            assert evidence.eligible_ids() == {lead.id, fresh.id}
            built = builder.build(residual, scope=POOL_SCOPE, relevance=evidence)
            assert built
            assert {n for c in built.clusters for n in c.member_ids} == {fresh.id}
            assert built.selection.excluded[5] == 1


def test_usefulness_floor_is_optional_and_keeps_its_accounting(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "usefulness.sqlite3") as store:
        low = _node(store, "low", "qualified but negatively reviewed", stats={"usefulness_score": -0.5})
        new = _node(store, "new", "qualified new knowledge")
        builder = RecallMapBuilder(store)
        assert len(builder._pool([result(low), result(new)])[0]) == 2
        monkeypatch.setenv(POOL_USEFULNESS_GATE_ENV, "1")
        assert len(builder._pool([result(low), result(new)])[0]) == 2
        monkeypatch.setenv(POOL_USEFULNESS_FLOOR_ENV, "0")
        pool, accounting = builder._pool([result(low), result(new)])
        assert [m.node.id for m in pool] == [new.id]
        assert accounting.excluded == (0, 0, 0, 0, 0, 0, 0, 1)


def test_empty_selection_does_not_offer_or_hide_later_query_evidence(tmp_path):
    with MemoryStore(tmp_path / "empty.sqlite3") as store:
        seed_corpus(store)
        node = _node(store, "candidate", "query protocol evidence", context={"topic": "query protocol"})
        builder = RecallMapBuilder(store)
        first = builder.build([result(node, 0)], scope=POOL_SCOPE, task="work")
        again = builder.build([result(node, 0)], scope=POOL_SCOPE, task="work")
        assert first == again and not first.clusters and first.pool_size == 0
        assert builder.last_cache_hit
        offered = builder.build([result(node)], scope=POOL_SCOPE, task="work")
        assert offered.clusters and not builder.last_cache_hit
        empty = builder.build([result(node, 0)], scope=POOL_SCOPE, task="work")
        assert not empty.clusters and not builder.last_cache_hit
