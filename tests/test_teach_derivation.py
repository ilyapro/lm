"""A correction reaches recorded derivatives before the next ordinary read."""

from collections import Counter
from dataclasses import asdict

import pytest

from living_memory import consolidation
from living_memory.consolidation import (
    _assess_cluster_eras,
    _cross_scope_promotion,
    _materialize_procedural_schemas,
    _materialize_cluster,
    _TraceCluster,
    memory_teach,
)
from living_memory.delivery import DeliveryHistory, shape_recall_results
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.feedback import apply_pending_recall_feedback
from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_transport_identity import FakeMCP


SCOPE = "project:workshop"
STEPS = [
    "Workshop release requires the retired paper permit.",
    "Workshop release checks the package checksum.",
    "Workshop release records the new package version.",
]
CORRECTION = "Workshop release uses the signed manifest; no paper permit is required."


class Encoder:
    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture(autouse=True)
def deterministic_recall(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    monkeypatch.setenv("LM_RECALL_GATE_FORM", "drop")
    monkeypatch.setattr(LocalEmbeddingModel, "embed", lambda self, text: [1.0, 0.0])


def procedure(store, scope=SCOPE, group="release", *, declared=True, texts=STEPS):
    traces = [
        store.append_trace(
            text,
            {"scope": scope, "procedure_id": "workshop_release", "task_pattern": group,
             "agent": "writer", "timestamp": "2025-01-01T00:00:00Z",
             **({"step_order": index + 1} if declared else {})},
        )
        for index, text in enumerate(texts)
    ]
    nodes = _materialize_procedural_schemas(store, traces, min_cluster_size=3)
    derived = next(node for node, _ in nodes if node.scope == scope and node.context["task_pattern"] == group)
    return traces, derived


def recall(service):
    return service.memory_recall(
        "workshop release", scope=SCOPE, max_results=10, log_access=False, log_event=False,
    )


def assert_withheld_source(results, source_id, correction_id):
    # A corrected source may remain discoverable as an unqualified reference.
    for item in results:
        if item["id"] == source_id:
            assert item["delivery"] == "below_threshold"
            assert item["content"] == ""
            assert item["content_ref"]["node_id"] == source_id
            assert item["superseded"] is True
            assert item["superseded_by"] == [correction_id]


@pytest.mark.parametrize("separate_writer", [False, True])
def test_teach_then_warm_recall_delivers_current_binding_steps(tmp_path, separate_writer):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        store.set_chunk_embedder(lambda texts: [Encoder().embed(text) for text in texts])
        traces, schema = procedure(store)
        _, elsewhere = procedure(store, scope="project:other")
        # Same label and scope, independent group and records.
        _, same_title = procedure(store, group="independent", texts=[text + " Independently." for text in STEPS])
        service = MemoryRecallService(store, embedder=Encoder())
        assert any(r.node.id == schema.id and STEPS[0] in r.node.content for r in recall(service))
        controls = {node.id: asdict(store.get_node(node.id)) for node in [elsewhere, same_title]}

        if separate_writer:
            with MemoryStore(path) as writer:
                taught = memory_teach(writer, traces[0].id, CORRECTION)
        else:
            taught = memory_teach(store, traces[0].id, CORRECTION)

        current = store.get_node(schema.id)
        assert current.level == "schema" and not current.decayed
        assert current.context["procedure"] == [CORRECTION, *STEPS[1:]]
        results = recall(service)
        assert any(r.node.id == schema.id and r.node.content == current.content for r in results)
        assert any(r.node.id == taught.corrective_trace.id for r in results)
        assert STEPS[0] not in current.content
        delivered = shape_recall_results(
            results, delivery_history=DeliveryHistory(), snippet_max_chars=1000,
            context_value_max_chars=1000, session_dedup=False, view="knowledge",
            correction_refs={r.node.id: r.correction_ids for r in results},
        )
        assert next(item for item in delivered if item["id"] == schema.id)["content"] == current.content
        assert next(item for item in delivered if item["id"] == taught.corrective_trace.id)["content"] == CORRECTION
        assert_withheld_source(delivered, traces[0].id, taught.corrective_trace.id)
        assert all(r.node.scope != "project:other" for r in results)
        assert {node_id: asdict(store.get_node(node_id)) for node_id in controls} == controls
        historical = store.get_node(traces[0].id)
        assert historical.content == STEPS[0]
        assert historical.corrections[-1]["new"] == CORRECTION

        # Reuse the same reader after another correction, including cached vectors.
        replacement = "Workshop release validates the artifact signature."
        memory_teach(store, taught.corrective_trace.id, replacement)
        assert store.get_node(schema.id).context["procedure"] == [replacement, *STEPS[1:]]
        assert any(r.node.id == schema.id and replacement in r.node.content for r in recall(service))


@pytest.mark.parametrize("declared", [True, False])
def test_mcp_teach_recall_and_exact_historical_lookup(tmp_path, declared):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    with mcp.memory_store as store:
        traces, schema = procedure(store, declared=declared)
        recall_args = {"scope": SCOPE, "ambient_context": {"transport_session_id": "reader"}}
        first = mcp.tools["memory_recall"]("workshop release", **recall_args)
        assert next(item for item in first["results"] if item["id"] == schema.id)["delivery"] == "full"
        repeat = mcp.tools["memory_recall"]("workshop release", **recall_args)
        assert next(item for item in repeat["results"] if item["id"] == schema.id)["delivery"] == "session_duplicate"
        taught = mcp.tools["memory_teach"](traces[0].id, CORRECTION)
        results = mcp.tools["memory_recall"]("workshop release", **recall_args)["results"]
        delivered = next(item for item in results if item["id"] == schema.id)
        assert delivered["level"] == ("schema" if declared else "concept")
        assert delivered["delivery"] == "full"
        assert delivered["content"] == store.get_node(schema.id).content
        assert all(text in delivered["content"] for text in [CORRECTION, *STEPS[1:]])
        assert STEPS[0] not in delivered["content"]
        correction_id = taught["corrective_trace"]["id"]
        assert next(item for item in results if item["id"] == correction_id)["content"] == CORRECTION
        assert_withheld_source(results, traces[0].id, correction_id)
        old = mcp.tools["memory_lookup"](node_id=traces[0].id)["results"][0]
        assert old["content"] == STEPS[0]
        assert old["provenance"]["corrections"][-1]["new"] == CORRECTION
        assert old["superseded"] is True
        assert old["superseded_by"] == [correction_id]


def test_later_materialization_cannot_undo_teach_or_erase_independent_steps(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces, schema = procedure(store)
        taught = memory_teach(store, traces[0].id, CORRECTION)
        # A background pass can already hold these append-only sources when
        # teach commits. Its later guarded write must use current validity.
        _materialize_procedural_schemas(store, traces, min_cluster_size=3)
        assert store.get_node(schema.id).context["procedure"] == [CORRECTION, *STEPS[1:]]
        _materialize_procedural_schemas(store, [*traces, taught.corrective_trace], min_cluster_size=3)
        assert store.get_node(schema.id).context["procedure"] == [CORRECTION, *STEPS[1:]]


def test_teaching_a_replacement_reaches_derivatives_still_citing_its_predecessor(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces, schema = procedure(store)
        # Recorded input still names the original, while its replacement is
        # the source being corrected now.
        with monkeypatch.context() as previous:
            previous.setattr(consolidation, "_refresh_derived_from", lambda *_: None)
            earlier = memory_teach(store, traces[0].id, "Workshop release uses a temporary manifest.")
        assert STEPS[0] in store.get_node(schema.id).content
        memory_teach(store, earlier.corrective_trace.id, CORRECTION)
        assert store.get_node(schema.id).context["procedure"] == [CORRECTION, *STEPS[1:]]


def test_incidental_recall_lineage_and_unrecorded_citations_are_independent(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces, _ = procedure(store)
        service = MemoryRecallService(store, embedder=Encoder())
        identity = {"agent": "writer", "transport_session_id": "derivation-test"}
        service.memory_recall("workshop release", scope=SCOPE, ambient_context=identity)
        independent = store.append_trace("The workshop door stays locked overnight.", {"scope": SCOPE, **identity})
        independent = apply_pending_recall_feedback(store, independent, reinforce_results=False).trace
        assert traces[0].id in independent.source_traces
        cited = store.create_node(
            level="concept", content="Independent electrical constraint.", context={"scope": SCOPE},
            provenance={"source_traces": [traces[0].id]},
        )
        before = {node.id: asdict(node) for node in [independent, cited]}
        memory_teach(store, traces[0].id, CORRECTION)
        assert {node_id: asdict(store.get_node(node_id)) for node_id in before} == before
        assert not _assess_cluster_eras(store, [traces[0], independent]).excluded.get(independent.id)
        corrected_independent = memory_teach(store, independent.id, "The workshop door uses the night latch.")
        assessment = _assess_cluster_eras(store, [traces[1], corrected_independent.corrective_trace])
        assert traces[1].id not in assessment.excluded


def test_evidence_carrier_keeps_whole_independent_records_and_correction(tmp_path):
    records = ["Workshop release case: " + ("Detailed observation. " * 100) + str(i) for i in range(3)]
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces, carrier = procedure(store, declared=False, texts=records)
        assert carrier.level == "concept"
        replacement = ("Workshop release revised case: " + "Complete substantive evidence. " * 90).rstrip()
        taught = memory_teach(store, traces[0].id, replacement)
        current = store.get_node(carrier.id)
        assert current.level == "concept" and "procedure" not in current.context
        assert all(text in current.content for text in [replacement, *records[1:]])
        assert records[0] not in current.content
        assert taught.corrective_trace.id in current.source_traces
        assert store.get_node(traces[0].id).content == records[0]


def cluster(store, scope, texts):
    traces = [
        store.append_trace(text, {"scope": scope, "agent": f"writer-{i}"},
                           feedback={"usefulness_score": 1.0 if i == 0 else 0.0})
        for i, text in enumerate(texts)
    ]
    group = _TraceCluster(scope, traces, Counter({"workshop": len(traces)}))
    (concept, _), = _materialize_cluster(store, group, min_cluster_size=1)
    store.update_node(concept.id, embedding=[1.0, 0.0])
    return traces, store.get_node(concept.id)


def test_cluster_correction_reaches_cross_scope_promotion_without_reconsolidation(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.set_chunk_embedder(lambda texts: [[1.0, 0.0] for _ in texts])
        traces, concept = cluster(store, SCOPE, STEPS)
        _, other = cluster(store, "project:other", ["Workshop inventory is checked weekly."])
        _, third = cluster(store, "project:third", ["Workshop tools are kept dry."])
        global_node, = _cross_scope_promotion(store, concept, phase_number=4)
        assert global_node is not None and STEPS[0] in global_node.content
        controls = {node.id: asdict(store.get_node(node.id)) for node in [other, third]}
        taught = memory_teach(store, traces[0].id, CORRECTION)
        for node_id in [concept.id, global_node.id]:
            current = store.get_node(node_id)
            assert STEPS[0] not in current.content
            assert CORRECTION in current.content
            assert not current.decayed
        assert taught.corrective_trace.id in store.get_node(concept.id).source_traces
        assert {node_id: asdict(store.get_node(node_id)) for node_id in controls} == controls
        stale_cluster = _TraceCluster(SCOPE, traces, Counter({"workshop": len(traces)}))
        _materialize_cluster(store, stale_cluster, min_cluster_size=1)
        current = store.get_node(concept.id)
        assert CORRECTION in current.content and STEPS[0] not in current.content
        assert taught.corrective_trace.id in current.source_traces


def test_teach_rollback_preserves_all_nodes_edges_chunks_and_feedback(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces, schema = procedure(store)
        service = MemoryRecallService(store, embedder=Encoder())
        service.memory_recall("workshop release", scope=SCOPE)
        before = list(store.connection.iterdump())
        update = consolidation._refresh_derived_node

        def fail_after_write(store, node, **kwargs):
            update(store, node, **kwargs)
            raise RuntimeError("failed derived write")

        monkeypatch.setattr(consolidation, "_refresh_derived_node", fail_after_write)
        with pytest.raises(RuntimeError, match="failed derived write"):
            memory_teach(store, traces[0].id, CORRECTION)
        assert list(store.connection.iterdump()) == before
        assert not store.connection.in_transaction
        assert store.get_node(schema.id).context["procedure"] == STEPS


def test_explicit_derivation_without_a_writer_becomes_historical(tmp_path):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    with mcp.memory_store as store:
        original = store.append_trace(STEPS[0], {"scope": SCOPE})
        derived = store.create_node(level="schema", content="An opaque derived requirement.", context={"scope": SCOPE})
        store.create_connection(derived.id, original.id, "related", metadata={"kind": "derived_from"})
        memory_teach(store, original.id, CORRECTION)
        assert all(result.node.id != derived.id for result in recall(MemoryRecallService(store, embedder=Encoder())))
        history = mcp.tools["memory_lookup"](node_id=derived.id)["results"][0]
        assert history["content"] == derived.content
        assert history["decayed"] and "corrected source" in history["decay_reason"]
