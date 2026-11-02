"""Attestation reuses immutable result tokens only within one grading pass."""

from types import SimpleNamespace

import pytest

from living_memory import attestation, feedback, grounding
from living_memory.storage import MemoryStore


@pytest.mark.parametrize("query", ["", "certificate renewal"])
def test_attest_tokenizes_results_once_per_resolution(tmp_path, monkeypatch, query):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RETRIEVAL_TUNING_POLICY", "fixed")
    batches = []

    def encode(queries):
        batches.append(list(queries))
        return [[1.0, 0.0] for _ in queries]

    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: encode)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        base = "Certificate renewal requires trusted issuer chains and dns01 propagation checks. "
        nodes = [store.create_node(
            level="trace", content=(base + f" unique service {i}. ") * 32,
            context={"scope": "project:fixture"},
        ) for i in range(8)]
        event = store.record_recall_event(query=query, scope="project:fixture", results=[
            {"node_id": node.id, "bm25_score": 1.0} for node in nodes
        ])
        evidence = [
            f"Checked certificate renewal trusted issuer chains and dns01 propagation in service {i}."
            for i in range(32)
        ]
        calls = []
        original = grounding.token_set

        def tokenize(text):
            calls.append(text)
            return original(text)

        monkeypatch.setattr(grounding, "token_set", tokenize)
        result = attestation.attest_recall_usage(store, event.id, evidence)
        # Preparation and the authoritative transaction resolve afresh, with
        # another preparation pass after anchor encoding when needed.
        one_pass = [evidence[0], *(node.content for node in nodes), *evidence[1:]]
        passes = 3 if query else 2
        assert len(calls) == len(one_pass) * passes
        assert calls == one_pass * passes
        assert result["credited"]
        assert batches == ([[query]] if query else [])
        assert bool(result["anchor_ids"]) == bool(query)
        calls.clear()
        assert attestation.attest_recall_usage(store, event.id, evidence) == dict(result, replay=True)
        assert calls == []


def test_grade_matches_independent_per_item_grounding_with_duplicate_ranks():
    resolved = [
        (0, {"bm25_score": 1.0}, SimpleNamespace(id="a", content="obsolete body")),
        (2, {"vector_score": 0.8}, SimpleNamespace(id="b", content="orchid violet garden")),
        (4, {"graph_score": 0.6}, SimpleNamespace(id="a", content="issuer chain renewal")),
        (7, {}, SimpleNamespace(id="empty", content="")),
    ]
    items = ["unrelated", "issuer chain", "issuer chain", "violet", ""]
    contents = {node.id: node.content for _, _, node in resolved}
    per_item = [grounding.ground_results(
        item, contents, min_containment=attestation.ATTESTATION_MIN_CONTAINMENT,
    ) for item in items]
    actual = attestation._grade(items, resolved)
    assert len(actual) == len(resolved)
    for entry, (rank, payload, node) in zip(actual, resolved):
        index, best = max(enumerate(per_item), key=lambda pair: pair[1][node.id].containment)
        assert entry.as_result() == {
            "node_id": node.id, "rank": rank,
            "containment": best[node.id].containment,
            "grounded": best[node.id].grounded, "evidence_index": index,
        }
        assert entry.payload is payload
    assert actual[0].evidence_index == actual[2].evidence_index == 1
    assert actual[3].containment == 0.0 and actual[3].evidence_index == 0


@pytest.mark.parametrize("items, resolved", [
    ([], [(0, {}, SimpleNamespace(id="a", content=object()))]),
    ([object()], []),
])
def test_grade_empty_paths_do_not_tokenize(monkeypatch, items, resolved):
    def forbidden(_text):
        pytest.fail("empty grading must not tokenize")

    monkeypatch.setattr(grounding, "token_set", forbidden)
    result = attestation._grade(items, resolved)
    assert len(result) == len(resolved)
    assert all(entry.containment == 0.0 and entry.evidence_index == 0 for entry in result)


@pytest.mark.parametrize("bad", ["evidence", "result"])
def test_grade_preserves_first_tokenization_failure(monkeypatch, bad):
    calls = []
    original = grounding.token_set

    def tokenize(text):
        calls.append(text)
        if text == bad:
            raise ValueError(text)
        return original(text)

    monkeypatch.setattr(grounding, "token_set", tokenize)
    with pytest.raises(ValueError, match=bad):
        attestation._grade(
            ["evidence", "later"], [(0, {}, SimpleNamespace(id="a", content="result"))],
        )
    assert calls == (["evidence"] if bad == "evidence" else ["evidence", "result"])
