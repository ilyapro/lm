"""One source preparation supplies distinct discovery and qualification scores."""

from collections import Counter

import pytest

from living_memory import embeddings, retrieval
from living_memory.scope import resolve_scope
from living_memory.storage import MemoryStore


SCOPE = "project:lexical-preparation"
QUERY = "amber quartz HTTP_429"


class Encoder:
    def embed(self, text):
        return [1.0, 0.0] if text in (QUERY, "Opaque vector source.") else [0.0, 1.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


@pytest.mark.parametrize("stemming", ["on", "off"])
@pytest.mark.parametrize("content,task", [
    ("", ""),
    ("amber quartz", None),
    ("amber quartz", ""),
    ("Cafeteria inventory.", QUERY),
    ("HTTP_429", "HTTP_503"),
    ("_collect_bm25 src/living_memory/retrieval.py", "issue-7319"),
    ("alphaBeta/x_7.end", "taskName-8"),
    ("name.", "continued"),
    ("HTTP_", "429"),
    ("Ёлка миграции обновления", "если не готово"),
    ("ΑΒΣ Straße İstanbul e\u0301", "Σσς 中文 Δelta-７"),
    ("a\u200db/", "\u0301c.d"),
])
def test_separate_source_and_task_match_concatenated_coverage(
    tmp_path, monkeypatch, stemming, content, task,
):
    monkeypatch.setenv("LM_TOKENIZE_CYRILLIC_STEM", stemming)
    embeddings.reset_cyrillic_stem_cache()
    try:
        terms = retrieval._lexical_terms
        combined = content + "\n" + (task or "")
        assert terms(combined) == terms(content) | terms(task or "")
        with MemoryStore(tmp_path / "coverage.sqlite3") as store:
            node = store.create_node(level="trace", content=content or " ",
                                     context={"scope": SCOPE, "task": task})
            monkeypatch.setattr(store, "search_content_ids", lambda *a, **kw: [(node.id, -1)])
            service = retrieval.MemoryRecallService(store, embedder=Encoder())
            for query in (QUERY, combined, "the", "7319 если не готово"):
                candidates = {}
                service._collect_bm25(query, resolve_scope(scope=SCOPE), candidates, max_results=1)
                candidate = candidates[node.id]
                assert candidate.lexical_score_prepared
                assert candidate.lexical_score == retrieval._lexical_support(terms(query), node.content)
                assert candidate.bm25_score == retrieval._lexical_support(
                    terms(query), node.content + "\n" + (task or ""),
                )
    finally:
        embeddings.reset_cyrillic_stem_cache()


def test_recall_prepares_each_body_once_and_reads_fresh_source(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "regex")
    monkeypatch.delenv("LM_RECALL_SCHEMA_TRIGGER", raising=False)
    encoder = Encoder()
    bodies = {
        "content": QUERY + " source procedure.",
        "task": "Cafeteria inventory.",
        "vector": "Opaque vector source.",
        "trigger": "Opaque trigger source.",
        "graph": "Opaque graph source.",
    }
    path = tmp_path / "recall.sqlite3"
    with MemoryStore(path, chunk_embedder=encoder.embed_batch) as store, MemoryStore(path) as writer:
        for name, content in bodies.items():
            context = {"scope": SCOPE}
            if name == "task":
                context["task"] = QUERY
            if name == "trigger":
                context["trigger"] = QUERY
            store.create_node(node_id=name, level="schema" if name == "trigger" else "concept",
                              content=content, context=context, embedding=encoder.embed(content))
        store.create_connection("content", "graph", "requires")
        service = retrieval.MemoryRecallService(store, embedder=encoder, anchor_seeding=False)
        calls = Counter()
        tokenize = retrieval.tokenize

        def count(text, **kwargs):
            if kwargs.get("evidence"):
                calls[text] += 1
            return tokenize(text, **kwargs)

        monkeypatch.setattr(retrieval, "tokenize", count)

        def recall():
            calls.clear()
            service.memory_recall(QUERY, scope=SCOPE, depth=1, max_results=2,
                                  log_access=False, log_event=False, repair_pending=False)
            assert all(calls[body] == 1 for body in bodies.values())
            assert not any(calls[body + "\n" + QUERY] for body in bodies.values())
            return {result.node_id: result for result in service.last_query_relevance.ranked}

        found = recall()
        assert set(found) == set(bodies)
        assert found["content"].bm25_score == found["content"].lexical_score == 1
        assert found["task"].bm25_score == 1
        assert found["task"].lexical_score == 0
        assert found["vector"].methods == ("vector",)
        assert found["trigger"].methods == ("trigger",)
        assert found["graph"].methods == ("graph",)
        assert all(found[name].lexical_score == 0 for name in ("vector", "trigger", "graph"))
        assert "task" not in service.last_query_relevance.eligible_ids()

        bodies["task"] = QUERY + " updated source."
        writer.update_node("task", content=bodies["task"], context={"scope": SCOPE, "task": ""})
        changed = recall()
        assert changed["task"].lexical_score == 1
        assert changed["task"].bm25_score == 1
        assert "task" in service.last_query_relevance.eligible_ids()
