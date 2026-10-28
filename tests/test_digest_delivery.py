"""Known source boundaries survive digest persistence and bounded recall."""

import pytest

from living_memory import consolidation as c
from living_memory.delivery import _relevant_excerpt
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.models import Node
from living_memory.server import create_mcp_server
from living_memory.source_text import passage_spans
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


LEAD = (
    "The common delivery policy "
    + "describes recurring coordination details " * 9
    + "and preserves current conditions."
)
FACT = "For the amber gasket use torque 17 only after the calibration check passes."
QUERY = "amber gasket torque calibration"
SCOPE = "project:digest-delivery"
LEGACY = LEAD + "\n• " + FACT


@pytest.mark.parametrize("fact", [
    FACT,
    "Do not\napply amber gasket torque before the calibration check passes.",
    "For the amber gasket see ref.\nCalibration must pass before applying torque.",
])
def test_fresh_digest_delivers_the_complete_matching_passage(fact):
    sources = [Node("lead", "trace", LEAD), Node("fact", "trace", fact)]
    assert [len(passage_spans(source.content)) for source in sources] == [1, 1]
    digest = c._synthesize_digest(sources, best=sources[0])
    assert len(passage_spans(digest)) == 2
    excerpt = _relevant_excerpt(digest, 300, QUERY)
    assert fact in excerpt
    assert len(excerpt) <= 300
    assert excerpt.endswith("[Context omitted]")


@pytest.mark.parametrize("extra_budget", [-1, 0, 1])
def test_digest_admission_counts_the_whole_passage_separator(extra_budget):
    sources = [Node("lead", "trace", LEAD), Node("fact", "trace", FACT)]
    complete = LEAD + "\n\n• " + FACT
    budget = len(complete) + extra_budget
    digest = c._synthesize_digest(sources, best=sources[0], max_chars=budget)
    assert len(digest) <= budget
    assert digest == (LEAD if extra_budget < 0 else complete)


def test_opaque_source_keeps_soft_bullet_boundary_and_incomplete_delivery():
    source = Node("opaque", "trace", LEGACY)
    digest = c._synthesize_digest([source], best=source)
    assert digest == LEGACY
    assert len(passage_spans(digest)) == 1
    excerpt = _relevant_excerpt(digest, 300, QUERY)
    assert FACT not in excerpt
    assert len(excerpt) <= 300
    assert excerpt.endswith("[Incomplete passage]")


@pytest.fixture(autouse=True)
def deterministic_embeddings(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "300")
    monkeypatch.setattr(LocalEmbeddingModel, "embed", lambda self, text: (
        [1.0, 0.0] if "amber" in text else [0.8, 0.6]
    ))


def seed(store):
    traces = [
        store.append_trace(text, {"scope": SCOPE, "agent": f"writer-{i}"},
                           feedback={"usefulness_score": 1.0 if i == 0 else 0.0})
        for i, text in enumerate((LEAD, FACT))
    ]
    result = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
    concept, = result.concepts_created
    assert concept.content.startswith(LEAD)
    peers = [
        store.create_node(level="concept", content=f"The {name} workshop keeps inspection records.",
                          context={"scope": f"project:{name}"}, embedding=[1.0, 0.0],
                          stats={"confidence": 0.1})
        for name in ("north", "south")
    ]
    promoted, = c._cross_scope_promotion(store, concept)
    assert promoted.content.startswith(concept.content)
    return traces, concept, promoted, peers


def assert_delivered(mcp, node, fact):
    response = mcp.tools["memory_recall"](
        QUERY, scope=node.scope, depth=0, max_results=10,
    )
    result = next(item for item in response["results"] if item["id"] == node.id)
    assert result["delivery"] == "snippet"
    assert fact in result["content"]
    assert len(result["content"]) <= 300
    assert result["content"].endswith(("[Context omitted]", "[Incomplete source context]"))
    assert result["content_ref"] == {"node_id": node.id, "full_content_chars": len(node.content)}
    lookup = mcp.tools["memory_lookup"](node_id=node.id)["results"][0]
    assert lookup["content"] == node.content


@pytest.mark.parametrize("refresh", [None, "consolidate", "teach"])
def test_persisted_digest_recall_and_source_based_refresh(tmp_path, refresh):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        traces, concept, promoted, peers = seed(store)
        peer_contents = {node.id: node.content for node in peers}
        if refresh:
            for node in (concept, promoted):
                store.update_node(node.id, content=node.content.replace("\n\n• ", "\n• "))
        legacy_content = store.get_node(concept.id).content

    mcp = create_mcp_server(path, mcp_factory=FakeMCP)
    try:
        store = mcp.memory_store
        expected_fact = FACT
        expected_sources = {trace.id for trace in traces}
        if refresh:
            before = mcp.tools["memory_recall"](
                QUERY, scope=SCOPE, depth=0, max_results=10,
            )
            old = next(item for item in before["results"] if item["id"] == concept.id)
            assert old["delivery"] == "snippet"
            assert FACT not in old["content"]
            assert old["content"].endswith("[Incomplete passage]")
            assert len(old["content"]) <= 300
            assert store.get_node(concept.id).content == legacy_content
            assert mcp.tools["memory_lookup"](node_id=concept.id)["results"][0]["content"] == legacy_content
            if refresh == "consolidate":
                # Existing promotions consume the current project digest.
                result = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
                assert result.concepts_created == []
                assert [node.id for node in result.concepts_updated] == [concept.id]
            else:
                expected_fact = "Do not\napply amber gasket torque until the calibration check passes."
                mcp.tools["memory_teach"](traces[1].id, expected_fact)
                correction, = store.list_connections(target_id=traces[1].id, relation_type="supersedes")
                expected_sources = {traces[0].id, correction.source_id}
                assert mcp.tools["memory_lookup"](node_id=traces[1].id)["results"][0]["content"] == FACT

        current = store.get_node(concept.id)
        global_node = store.get_node(promoted.id)
        assert set(current.source_traces) == expected_sources
        assert set(global_node.source_traces) == expected_sources
        assert set(global_node.materializer_sources) == {concept.id, *(peer.id for peer in peers)}
        assert current.context == concept.context
        assert global_node.context == promoted.context
        assert {node.id for node in store.list_nodes(level="concept", scope=SCOPE)} == {concept.id}
        assert {node.id for node in store.list_nodes(level="concept", scope="global")} == {promoted.id}
        assert {node_id: store.get_node(node_id).content for node_id in peer_contents} == peer_contents
        for node in (current, global_node):
            assert not node.decayed
            assert set(node.materializer_sources) <= {
                edge.target_id if edge.source_id == node.id else edge.source_id
                for edge in store.list_connections_for_nodes([node.id])[node.id] if edge.type == "related"
            }
            assert_delivered(mcp, node, expected_fact)
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()
