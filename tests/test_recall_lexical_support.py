"""Source qualification uses current text even when FTS competition changes."""

from dataclasses import replace
import math

import pytest

from living_memory.embeddings import tokenize
from living_memory.models import Node
from living_memory.recall_map import RecallMapBuilder
from living_memory.retrieval import MemoryRecallService, RecallResult, _expanded_query
from living_memory.score_gate import gate_score, passes_gate, trigger_gate_score
from living_memory.storage import MemoryStore


SCOPE = "project:lexical-support"
QUERY = "Which database snapshot should be restored after a failed migration?"
TEXTS = {
    "target": "Database migration rollback: restore the verified snapshot before restarting the deployment worker.",
    "botanical": "Botanical observation: violet plants grow beside the amber pond.",
    "picnic": "The database team schedules a picnic beside the lake after the maintenance window.",
    "competitor": "Database snapshot should be restored after a failed migration.",
}


class FrozenEncoder:
    def __init__(self, query, scores=None):
        self.query = query
        self.scores = scores or {}

    def embed(self, text):
        cosine = 1.0 if text == self.query else self.scores.get(text, 0.0)
        return [cosine, math.sqrt(1 - cosine**2)]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    monkeypatch.setenv("LM_RECALL_GATE_FORM", "drop")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "regex")
    monkeypatch.delenv("LM_RECALL_SCHEMA_TRIGGER", raising=False)


def add(store, encoder, name, content, **kwargs):
    return store.create_node(level="trace", node_id=name, content=content,
                             context={"scope": SCOPE}, embedding=encoder.embed(content), **kwargs)


def snapshot(store, service, query, depth=0):
    delivered = service.memory_recall(query, scope=SCOPE, depth=depth, max_results=5,
                                      log_access=False, log_event=False)
    relevance = service.last_query_relevance
    by_id = {r.node_id: r for r in relevance.ranked}
    pool, _ = RecallMapBuilder(store)._pool(relevance.ranked, relevance=relevance)
    return {
        "results": by_id,
        "scores": {key: gate_score(r, relevance.demotions, plan=relevance.plan,
                                   causal_mode=relevance.causal_mode) for key, r in by_id.items()},
        "qualified": relevance.eligible_ids(),
        "mapped": {member.node.id for member in pool},
        "delivered": {r.node_id: r.withheld for r in delivered},
        "fts": [node.id for node, _ in store.search_content(_expanded_query(query), scope=SCOPE, limit=40)],
    }


@pytest.mark.parametrize("form", ["drop", "stub"])
def test_fts_competition_changes_order_without_changing_source_qualification(tmp_path, monkeypatch, form):
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    encoder = FrozenEncoder(QUERY, {TEXTS["target"]: 0.7666788101196289,
                                   TEXTS["picnic"]: 0.25728386640548706,
                                   TEXTS["botanical"]: 0.11858166754245758})
    with MemoryStore(tmp_path / "competition.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        for name in ("target", "botanical", "picnic"):
            add(store, encoder, name, TEXTS[name])
        service = MemoryRecallService(store, embedder=encoder, anchor_seeding=False)
        before = snapshot(store, service, QUERY)
        add(store, encoder, "competitor", TEXTS["competitor"])
        after = snapshot(store, service, QUERY)
        assert before["fts"].index("picnic") == 1
        assert after["fts"].index("picnic") == 2
        assert before["results"]["picnic"].bm25_score == 0.5
        assert after["results"]["picnic"].bm25_score == pytest.approx(1 / 3)
        for name in ("target", "picnic"):
            left, right = (arm["results"][name] for arm in (before, after))
            assert left.node.content == right.node.content
            assert left.lexical_score == right.lexical_score > 0
            assert left.vector_score == right.vector_score
            assert before["scores"][name] == after["scores"][name]
        for arm in (before, after):
            assert "target" in arm["qualified"] & arm["mapped"]
            assert "picnic" not in arm["qualified"] | arm["mapped"]
            assert arm["delivered"]["picnic"] == "below_threshold"
            assert arm["delivered"]["target"] is None


@pytest.mark.parametrize("depth", [2, "causal"])
def test_genuine_relation_activation_survives_fts_competition(tmp_path, depth):
    query = "alpha beta gamma delta"
    encoder = FrozenEncoder(query, {"opaque leaf": 0.6, "opaque tail": 0.1})
    with MemoryStore(tmp_path / "graph.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        for name, text in (("root", "alpha"), ("leaf", "opaque leaf"), ("tail", "opaque tail")):
            add(store, encoder, name, text)
        store.create_connection("root", "leaf", "requires")
        store.create_connection("leaf", "tail", "requires")
        service = MemoryRecallService(store, embedder=encoder, anchor_seeding=False)
        before = snapshot(store, service, query, depth)
        add(store, encoder, "competitor", query + " alpha beta gamma delta")
        after = snapshot(store, service, query, depth)
        assert before["results"]["root"].bm25_score > after["results"]["root"].bm25_score
        assert before["results"]["leaf"].graph_score > after["results"]["leaf"].graph_score
        for name in ("leaf", "tail"):
            assert before["results"][name].source_graph_score == after["results"][name].source_graph_score > 0
            assert before["scores"][name] == after["scores"][name]
            assert (name in before["qualified"]) == (name in after["qualified"])
        if depth == "causal":
            assert "leaf" in before["qualified"] & after["qualified"]


def test_source_seed_bound_does_not_use_fts_tie_order(tmp_path, monkeypatch):
    import living_memory.retrieval as retrieval

    monkeypatch.setattr(retrieval, "GRAPH_SEED_LIMIT", 1)
    query = "quartz pressure"
    encoder = FrozenEncoder(query, {"opaque leaf": 0.1})
    with MemoryStore(tmp_path / "seeds.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        add(store, encoder, "a-root", "quartz pressure with an extended inspection description")
        add(store, encoder, "leaf", "opaque leaf")
        store.create_connection("a-root", "leaf", "requires")
        service = MemoryRecallService(store, embedder=encoder, anchor_seeding=False)
        before = snapshot(store, service, query, depth=1)
        # Both sources have complete coverage. Only discovery prefers the
        # short competitor; source seed ties use source identity.
        add(store, encoder, "z-competitor", query + " quartz pressure")
        after = snapshot(store, service, query, depth=1)
        # The leaf remains independently found through its frozen vector.
        assert after["fts"][0] == "z-competitor"
        leaf = after["results"]["leaf"]
        assert leaf.graph_score == 0
        assert leaf.source_graph_score == before["results"]["leaf"].source_graph_score == pytest.approx(0.85)
        assert before["scores"]["leaf"] == after["scores"]["leaf"]


@pytest.mark.parametrize("query,source,other", [
    ("HTTP_429", "Return HTTP_429 after throttling.", "Return HTTP_503 during downtime."),
    ("_collect_bm25", "The _collect_bm25 function expands the query.", "The collect function reads a stream."),
    ("7319", "Ticket 7319 records the procedure.", "Ticket 7320 records another procedure."),
    ("quartz pressure", "Inspect quartz pressure before release.", "Inspect quartz colour before release."),
])
def test_lexical_only_identifiers_and_source_coverage_remain_available(tmp_path, query, source, other):
    encoder = FrozenEncoder(query)
    with MemoryStore(tmp_path / "identifier.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        add(store, encoder, "source", source)
        add(store, encoder, "other", other)
        service = MemoryRecallService(store, embedder=encoder, anchor_seeding=False)
        arm = snapshot(store, service, query)
        found = arm["results"]["source"]
        assert found.vector_score == 0
        assert found.lexical_score == 1
        assert arm["delivered"]["source"] is None
        assert "source" in arm["mapped"]
        assert "other" not in arm["qualified"]


@pytest.mark.parametrize("mode", ["", "name"])
@pytest.mark.parametrize("trigger,query", [("HTTP_429_RETRY", "HTTP_503_ABORT"), ("если не готово", "готово")])
def test_trigger_matching_keeps_identifier_and_condition_terms(tmp_path, monkeypatch, mode, trigger, query):
    monkeypatch.setenv("LM_RECALL_SCHEMA_TRIGGER", mode)
    encoder = FrozenEncoder(query)
    with MemoryStore(tmp_path / "trigger.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        schema = store.create_node(level="schema", node_id="schema", content=trigger,
                                   context={"scope": SCOPE, "trigger": trigger},
                                   embedding=encoder.embed(trigger))
        service = MemoryRecallService(store, embedder=encoder, anchor_seeding=False)
        arm = snapshot(store, service, query)
        found = arm["results"][schema.id]
        assert found.trigger_score == 0
        # Term coverage is not an entailment or polarity classifier.
        # An exact name remains available with both trigger policies.
        encoder.query = trigger
        positive = snapshot(store, service, trigger)
        assert positive["delivered"][schema.id] is None


def test_discovery_fields_cannot_substitute_for_missing_source_evidence():
    node = Node(id="source", level="schema", content="source", scope=SCOPE)
    result = RecallResult(node, score=1, bm25_score=1, graph_score=1.5,
                          trigger_score=0.975, trigger_coverage=0.6)
    for rank in (0, 0.1, 0.5, 1):
        changed = replace(result, bm25_score=rank)
        assert gate_score(changed) == gate_score(result)
        assert trigger_gate_score(changed) == trigger_gate_score(result)
    unsupported = replace(result, trigger_score=0)
    assert gate_score(unsupported) == 0
    assert not passes_gate(unsupported, 0.35)
    assert "429" in tokenize("HTTP_429", evidence=True)
    assert len(set(tokenize("если не готово", evidence=True))) == 3
