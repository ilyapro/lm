from pathlib import Path

from living_memory.retrieval import memory_recall
from living_memory.scope import resolve_scope
from living_memory.storage import MemoryStore


def test_explicit_project_scope_is_isolated_from_other_projects(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        alpha = store.append_trace(
            "Alpha project deploy uses migration 42",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        beta = store.append_trace(
            "Beta project deploy uses migration 99",
            {"scope": "project:beta", "agent": "agent-b"},
        )
        global_trace = store.append_trace(
            "All projects should run migration checks before deploy",
            {"scope": "global", "agent": "agent-c"},
        )

        alpha_results = memory_recall(store, "deploy migration", scope="project:alpha", max_results=10)
        alpha_ids = {result.node.id for result in alpha_results}

        assert alpha.id in alpha_ids
        assert global_trace.id in alpha_ids
        assert beta.id not in alpha_ids


def test_ambient_session_searches_session_project_then_global(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        session = store.append_trace(
            "Session note says rollback deploy now",
            {"scope": "session:s1", "agent": "agent-a"},
        )
        project = store.append_trace(
            "Project note says rollback deploy after health checks",
            {"scope": "project:alpha", "agent": "agent-b"},
        )
        global_trace = store.append_trace(
            "Global note says rollback deploy requires approval",
            {"scope": "global", "agent": "agent-c"},
        )
        other_project = store.append_trace(
            "Other project rollback deploy is unrelated",
            {"scope": "project:beta", "agent": "agent-d"},
        )

        results = memory_recall(
            store,
            "rollback deploy",
            ambient_context={"session_id": "s1", "project": "alpha"},
            max_results=10,
        )
        ids = [result.node.id for result in results]

        assert session.id in ids
        assert project.id in ids
        assert global_trace.id in ids
        assert other_project.id not in ids
        assert results[0].node.scope == "session:s1"


def test_implicit_project_scope_can_be_inferred_from_query(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        alpha = store.append_trace(
            "Queue worker restart policy for alpha",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        beta = store.append_trace(
            "Queue worker restart policy for beta",
            {"scope": "project:beta", "agent": "agent-b"},
        )

        plan = resolve_scope(query="what is alpha queue policy", store=store)
        assert plan.scopes == ("project:alpha", "global")

        results = memory_recall(store, "alpha queue policy", max_results=10)
        ids = {result.node.id for result in results}
        assert alpha.id in ids
        assert beta.id not in ids


def test_implicit_project_scope_can_be_inferred_from_russian_query(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        alpha = store.append_trace(
            "Политика перезапуска очереди для альфа",
            {"scope": "project:альфа", "agent": "agent-a"},
        )
        beta = store.append_trace(
            "Политика перезапуска очереди для бета",
            {"scope": "project:бета", "agent": "agent-b"},
        )

        plan = resolve_scope(query="какая политика очереди у альфа", store=store)
        assert plan.scopes == ("project:альфа", "global")

        results = memory_recall(store, "политика очереди альфа", max_results=10)
        ids = {result.node.id for result in results}
        assert alpha.id in ids
        assert beta.id not in ids
