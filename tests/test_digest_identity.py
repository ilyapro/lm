"""Repeated source-based materialization preserves literal passage identity."""

from collections import Counter

import pytest

from living_memory.consolidation import memory_consolidate
from living_memory.server import create_mcp_server
from living_memory.source_text import passage_spans
from test_lookup import FakeMCP


COMMON = "A database import incident required checksum verification before reopening writes."


@pytest.mark.parametrize("scope_order", [(0, 1, 2), (2, 1, 0)])
@pytest.mark.parametrize("clauses", [
    tuple(f"Independent observation {i}." for i in range(3)),
    (
        "• Keep writes closed until checksum verification succeeds; a partial match does not authorize reopening.",
        "• • Keep writes closed until checksum verification succeeds; a partial match does not authorize reopening.",
        "Do not\nreopen writes unless checksum verification succeeds.",
    ),
])
def test_repeated_consolidation_and_promotion_preserve_exact_passages(
    tmp_path, scope_order, clauses,
):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    try:
        sources = {}
        owners = {}
        for scope_index in scope_order:
            scope = f"project:digest-{scope_index}"
            traces = [
                store.append_trace(
                    COMMON + "\n\n" + clause,
                    {"scope": scope, "agent": f"writer-{scope_index}"},
                    feedback={"usefulness_score": 1.0 if i == scope_index else 0.0},
                )
                for i, clause in enumerate(clauses)
            ]
            sources[scope] = traces
            result = memory_consolidate(store, scope=scope, force=True, min_cluster_size=3)
            concept, = result.concepts_created
            owners[scope] = concept.id
            assert concept.content.startswith(traces[scope_index].content)

        expected = Counter([COMMON, *clauses])
        global_node, = store.list_nodes(level="concept", scope="global")
        owners["global"] = global_node.id
        all_source_ids = {trace.id for traces in sources.values() for trace in traces}

        for fold in range(3):
            response = mcp.tools["memory_lookup"](node_ids=list(owners.values()), view="full")
            assert {node["id"] for node in response["results"]} == set(owners.values())
            for node in response["results"]:
                content = node["content"]
                assert Counter(content[start:end] for start, end, _ in passage_spans(content)) == expected
                current = store.get_node(node["id"])
                source_ids = (all_source_ids if current.scope == "global" else
                              {trace.id for trace in sources[current.scope]})
                assert set(current.source_traces) == source_ids
                if current.scope == "global":
                    assert set(current.materializer_sources) == set(owners.values()) - {current.id}

            if fold == 2:
                break
            for scope_index in reversed(scope_order):
                scope = f"project:digest-{scope_index}"
                lead_index = (scope_index + fold + 1) % len(clauses)
                for i, trace in enumerate(sources[scope]):
                    store.update_node(trace.id, stats={"usefulness_score": 1.0 if i == lead_index else 0.0})
                result = memory_consolidate(store, scope=scope, force=True, min_cluster_size=3)
                assert result.concepts_created == []
                concept, = result.concepts_updated
                assert concept.id == owners[scope]
                assert concept.content.startswith(sources[scope][lead_index].content)
            assert [node.id for node in store.list_nodes(level="concept", scope="global")] == [global_node.id]

        for traces in sources.values():
            for trace in traces:
                assert store.get_node(trace.id).content == trace.content
    finally:
        mcp.auto_consolidation.close()
        store.close()
