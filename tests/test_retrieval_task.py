"""Public recall checks for task text as lexical evidence."""

from pathlib import Path

from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore


def test_unscoped_recall_reaches_task_only_fact_without_promoting_same_task_noise(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "task.sqlite3") as store:
        target = store.append_trace(
            "The integrated comparison confirmed the revision survived landing. "
            "Supervisor review should inspect the recorded result before release.",
            {"scope": "project:marble", "task": "harbor-914-slate"},
        )
        same_task = store.append_trace(
            "The cafeteria inventory was counted on Friday.",
            {"scope": "project:marble", "task": "harbor-914-slate"},
        )
        other_project = store.append_trace(
            "Quartz packaging labels passed inspection.",
            {"scope": "project:quartz", "task": "harbor-914-slate"},
        )
        store.append_trace(
            "The archive checksum passed staging.",
            {"scope": "project:marble", "task": "harbor-915-slate"},
        )
        decayed = store.append_trace(
            "A historical review was superseded.",
            {"scope": "project:marble", "task": "harbor-914-slate"},
        )
        store.soft_delete_node(decayed.id, reason="superseded")
        store.append_trace(
            "A procedure tag from another record must not enter task search.",
            {"scope": "global", "procedure": "harbor-914-slate"},
        )

        short = memory_recall(store, "harbor-914-slate", max_results=6)
        short_ids = {result.node.id for result in short}
        assert target.id in short_ids
        assert decayed.id not in short_ids
        assert any("bm25" in result.methods for result in short if result.node.id == target.id)

        contextual = memory_recall(
            store, "harbor-914-slate supervisor landing integrated comparison", max_results=6
        )
        contextual_ids = [result.node.id for result in contextual]
        assert contextual_ids.index(target.id) < contextual_ids.index(same_task.id)
        if other_project.id in contextual_ids:
            assert contextual_ids.index(target.id) < contextual_ids.index(other_project.id)

        scoped = memory_recall(
            store,
            "harbor-914-slate supervisor landing integrated comparison",
            scope="project:marble",
            max_results=6,
        )
        assert target.id in {result.node.id for result in scoped}
        assert all(result.node.scope == "project:marble" for result in scoped)


def test_identifier_phrase_survives_crowded_partial_matches_with_content_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    # More decoys than either lexical or vector discovery admits at limit=6.
    # Every decoy contains the identifier's words, but in a different order,
    # and is a stronger content-vector match than the recorded-task fact.
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    task = "harbor-914-slate"
    query = f"{task} supervisor landing integrated comparison"
    content = (
        "The supervisor confirmed the integrated comparison after landing. "
        + "Documented review evidence remains available for an independent assessment. " * 4
    )
    decoy_content = (
        "harbor slate 914 supervisor landing integrated comparison "
        "orchard transport inventory apples pears oranges peaches crates "
        "boats tables chairs roof road stone trees batch "
    )
    embedder = LocalEmbeddingModel()
    query_vector = embedder.embed(query)
    target_similarity = cosine_similarity(query_vector, embedder.embed(content))
    with MemoryStore(tmp_path / "crowded.sqlite3") as store:
        target = store.append_trace(content, {"scope": "project:marble", "task": task})
        for index in range(160):
            decoy = decoy_content + str(index)
            assert cosine_similarity(query_vector, embedder.embed(decoy)) > target_similarity
            store.append_trace(decoy, {"scope": "project:quartz", "task": f"catalog-{index}"})

        short = memory_recall(store, task, max_results=6)
        assert target.id in {result.node.id for result in short}
        contextual = memory_recall(store, query, max_results=6)
        reached = next(result for result in contextual if result.node.id == target.id)
        assert reached.bm25_score == 1.0
        # Lexical admission must retain the already computed content evidence
        # even when stronger vectors fill the vector-only discovery shortlist.
        assert reached.vector_score > 0.0
