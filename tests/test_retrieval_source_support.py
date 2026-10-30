"""Lexical discovery order cannot stand in for current query support."""

from dataclasses import replace
import math

import pytest

from living_memory.feedback import _method_signals
from living_memory.retrieval import MemoryRecallService, _Candidate
from living_memory.scope import resolve_scope
from living_memory.score_gate import gate_score
from living_memory.storage import MemoryStore


SCOPE = "project:source-support"
QUERY = "amber quartz basalt mica"


class FrozenEncoder:
    def __init__(self, scores):
        self.scores = scores

    def embed(self, text):
        cosine = 1.0 if text == QUERY else self.scores.get(text, 0.0)
        return [cosine, math.sqrt(1 - cosine**2)]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


def test_lexical_score_and_credit_ignore_fts_position_and_raw_scale(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "scores.sqlite3") as store:
        weak = store.create_node(level="trace", content="amber label", context={"scope": SCOPE})
        strong = store.create_node(level="trace", content=QUERY, context={"scope": SCOPE})
        service = MemoryRecallService(store, embedder=FrozenEncoder({}))
        plan = resolve_scope(scope=SCOPE)
        outcomes = []
        for order in ((weak, strong), (strong, weak)):
            monkeypatch.setattr(store, "search_content_ids", lambda *a, **kw: [
                (node.id, -(1000.0 / (index + 1))) for index, node in enumerate(order)
            ] if kw["scope"] == SCOPE else [])
            candidates = {
                weak.id: _Candidate(weak, vector_score=0.4),
                strong.id: _Candidate(strong, vector_score=0.6),
            }
            service._collect_bm25(QUERY, plan, candidates, max_results=2)
            ranked = service.rank_candidates(candidates, plan)
            assert [result.node.id for result in ranked] == [strong.id, weak.id]
            assert candidates[weak.id].bm25_score == 0.25
            assert candidates[strong.id].bm25_score == 1.0
            outcomes.append([(result.node.id, result.score, _method_signals(result, 1)) for result in ranked])
        assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("depth", [0, 1])
def test_partial_overlap_distractor_cannot_win_by_fts_order(tmp_path, reverse, depth):
    target_text = QUERY + " " + "inspection notes record mineral samples " * 40
    distractor_text = "amber"
    encoder = FrozenEncoder({target_text: 0.6, distractor_text: 0.4})
    with MemoryStore(tmp_path / "corpus.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        items = [("target", target_text), ("distractor", distractor_text)]
        for name, content in reversed(items) if reverse else items:
            store.create_node(node_id=name, level="trace", content=content,
                              context={"scope": SCOPE}, embedding=encoder.embed(content))
        service = MemoryRecallService(store, embedder=encoder, anchor_seeding=False)
        options = dict(scope=SCOPE, depth=depth, max_results=3, log_access=False, log_event=False)
        assert store.search_content_ids(QUERY, scope=SCOPE)[0][0] == "distractor"
        before = service.memory_recall(QUERY, **options)
        assert before[0].node.id == "target"
        store.create_node(node_id="competitor", level="trace", content="amber amber",
                          context={"scope": SCOPE}, embedding=encoder.embed("amber amber"))
        after = service.memory_recall(QUERY, **options)
        assert after[0].node.id == "target"
        for left in before:
            right = next(item for item in after if item.node.id == left.node.id)
            assert (left.score, left.bm25_score, left.vector_score) == (
                right.score, right.bm25_score, right.vector_score,
            )


def test_task_identifier_support_does_not_qualify_unrelated_content(tmp_path):
    with MemoryStore(tmp_path / "task.sqlite3") as store:
        target = store.create_node(level="trace", content="Cafeteria inventory.",
                                   context={"scope": SCOPE, "task": "marble-731-slate"})
        other = store.create_node(level="trace", content="marble slate 731",
                                  context={"scope": SCOPE})
        service = MemoryRecallService(store, embedder=FrozenEncoder({}))
        candidates = {}
        service._collect_bm25("marble-731-slate", resolve_scope(scope=SCOPE), candidates, max_results=2)
        assert candidates[target.id].bm25_score == 1.0
        assert 0 < candidates[other.id].bm25_score < 1.0
        ranked = service.rank_candidates(candidates, resolve_scope(scope=SCOPE))
        found = next(item for item in ranked if item.node.id == target.id)
        assert gate_score(replace(found, lexical_score=0, vector_score=0)) == 0
