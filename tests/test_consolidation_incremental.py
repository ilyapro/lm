"""The consolidation pass pays for what changed, and clusters exactly as before.

Two costs used to scale with the whole scope on every pass: re-encoding every
active trace (nodes carry no vector column any more) and scoring every trace
against every cluster in pure Python. The vector cache and the indexed
clustering remove both; these tests pin that neither changes the outcome.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import random
from typing import Any

import pytest

from living_memory import consolidation
from living_memory.consolidation import (
    _cluster_key,
    _cluster_prepared_reference,
    _cluster_traces_indexed,
    _significant_tokens,
    _trace_embeddings,
    memory_consolidate,
)
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.models import Node
from living_memory.storage import MemoryStore

SCOPE = "project:inc"


def _node(index: int, content: str, scope: str = SCOPE) -> Node:
    return Node(id=f"n{index:05d}", level="trace", content=content, scope=scope)


def _synthetic(seed: int, count: int, dimensions: int = 24) -> list[tuple[Node, set[str], list[float] | None]]:
    """Topic-mixture traces: some near-duplicates, some token-only overlaps."""

    rng = random.Random(seed)
    topics = [[rng.gauss(0, 1) for _ in range(dimensions)] for _ in range(12)]
    vocab = [f"word{index}" for index in range(80)]
    prepared = []
    for index in range(count):
        topic = rng.randrange(len(topics))
        noise = rng.choice((0.05, 0.4, 1.5))
        vector = [value + rng.gauss(0, noise) for value in topics[topic]]
        words = set(rng.sample(vocab[topic * 5 : topic * 5 + 12], 4)) | set(rng.sample(vocab, 2))
        scope = SCOPE if index % 7 else "project:other"
        embedding = None if index % 11 == 0 else vector
        prepared.append((_node(index, " ".join(sorted(words)), scope), words, embedding))
    return prepared


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_indexed_clustering_matches_the_reference_loop(seed: int) -> None:
    prepared = _synthetic(seed, 400)
    reference = _cluster_prepared_reference(prepared)
    indexed = _cluster_traces_indexed(prepared, 24)

    assert [[trace.id for trace in cluster.traces] for cluster in indexed] == [
        [trace.id for trace in cluster.traces] for cluster in reference
    ]
    assert [cluster.strategy for cluster in indexed] == [c.strategy for c in reference]
    assert [_cluster_key(cluster) for cluster in indexed] == [_cluster_key(c) for c in reference]
    assert [cluster.embedding_count for cluster in indexed] == [
        cluster.embedding_count for cluster in reference
    ]
    for mine, theirs in zip(indexed, reference):
        if theirs.embedding_sum is None:
            assert mine.embedding_sum is None
        else:
            assert mine.embedding_sum == pytest.approx(theirs.embedding_sum, abs=1e-12)
    assert len({len(cluster.traces) for cluster in reference}) > 2  # not a trivial split


def test_token_only_traces_cluster_like_the_reference() -> None:
    prepared = [(node, tokens, None) for node, tokens, _ in _synthetic(9, 200)]
    reference = _cluster_prepared_reference(prepared)
    indexed = _cluster_traces_indexed(prepared, 0)
    assert [[t.id for t in c.traces] for c in indexed] == [
        [t.id for t in c.traces] for c in reference
    ]


class _CountingEncoder(LocalEmbeddingModel):
    def __init__(self) -> None:
        super().__init__(backend="hash")
        self.encoded: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.encoded.append(text)
        return super().embed(text)


def test_second_pass_encodes_only_traces_written_since(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(30):
            store.append_trace(f"pier lamp inspection {index}", {"scope": SCOPE, "agent": "a"})
        traces = store.list_nodes(level="trace", scope=SCOPE, limit=1000)
        # Nodes carry no whole-content vector: every one is a cache miss.
        for trace in traces:
            trace.embedding = None
        encoder = _CountingEncoder()
        first = _trace_embeddings(store, traces, encoder, nullcontext)
        assert len(encoder.encoded) == 30

        store.append_trace("pier lamp inspection late entry", {"scope": SCOPE, "agent": "a"})
        traces = store.list_nodes(level="trace", scope=SCOPE, limit=1000)
        for trace in traces:
            trace.embedding = None
        encoder.encoded.clear()
        second = _trace_embeddings(store, traces, encoder, nullcontext)
        assert encoder.encoded == ["pier lamp inspection late entry"]
        # A cached vector is the fresh encode, bit for bit.
        for trace_id, vector in first.items():
            assert second[trace_id] == vector


def test_cache_misses_on_a_different_encoder(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("harbor tug roster", {"scope": SCOPE, "agent": "a"})
        node.embedding = None
        store.put_consolidation_embeddings([(node, [0.5, 0.5])], "hash:2")
        assert store.get_consolidation_embeddings([node], "hash:2") == {node.id: [0.5, 0.5]}
        assert store.get_consolidation_embeddings([node], "st:other:384") == {}


def test_guarded_pass_takes_the_guard_per_step_and_matches_the_unguarded_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")

    def seed(path: Path) -> MemoryStore:
        store = MemoryStore(path)
        for index in range(40):
            topic = ("ferry engine overhaul", "customs manifest audit")[index % 2]
            store.append_trace(f"{topic} step {index}", {"scope": SCOPE, "agent": f"a{index % 3}"})
        return store

    entries = 0

    class CountingGuard:
        def __enter__(self) -> None:
            nonlocal entries
            entries += 1

        def __exit__(self, *exc: Any) -> None:
            return None

    def members(store: MemoryStore, result: Any) -> list[list[str]]:
        return sorted(
            sorted(store.get_node(trace_id).content for trace_id in concept.source_traces)
            for concept in result.concepts_created
        )

    with seed(tmp_path / "plain.sqlite3") as plain, seed(tmp_path / "guarded.sqlite3") as guarded:
        unguarded = memory_consolidate(plain, scope=SCOPE, force=True)
        stepped = memory_consolidate(guarded, scope=SCOPE, force=True, guard=CountingGuard)
        # Ids differ between the two files (fresh ULIDs), so compare members
        # by content: the same traces must be promoted into the same concepts.
        assert members(guarded, stepped) == members(plain, unguarded)

    assert entries >= 4  # load, schemas, one per promoted cluster, decay
    assert stepped.clusters_considered == unguarded.clusters_considered
    assert len(stepped.concepts_created) == len(unguarded.concepts_created) >= 1
