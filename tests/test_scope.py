from pathlib import Path

import pytest

from living_memory.config import MemoryConfig
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
        store.append_trace(
            "All projects should run migration checks before deploy",
            {"scope": "global", "agent": "agent-c"},
        )

        plan = resolve_scope(query="deploy migration", scope="project:alpha", store=store)
        assert plan.scopes == ("project:alpha", "global")

        alpha_results = memory_recall(store, "deploy migration", scope="project:alpha", max_results=10)
        alpha_ids = {result.node.id for result in alpha_results}

        assert alpha.id in alpha_ids
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


def test_partial_token_overlap_does_not_infer_project_scope(tmp_path: Path) -> None:
    # A single shared token is coincidence, not a deliberate mention: a broad
    # query must not be silently narrowed into an unrelated project's scope.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace(
            "Ingestion worker batches records",
            {"scope": "project:data-pipeline", "agent": "agent-a"},
        )
        store.append_trace(
            "Invoices post nightly",
            {"scope": "project:billing-service", "agent": "agent-b"},
        )

        plan = resolve_scope(query="pipeline docs", store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("global",)

        plan = resolve_scope(query="service restart guide", store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("global",)


def test_full_token_coverage_infers_project_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace(
            "Ingestion worker batches records",
            {"scope": "project:data-pipeline", "agent": "agent-a"},
        )

        plan = resolve_scope(query="where are the data pipeline docs", store=store)
        assert plan.scopes == ("project:data-pipeline", "global")


def test_whole_name_substring_infers_project_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace(
            "Ingestion worker batches records",
            {"scope": "project:data-pipeline", "agent": "agent-a"},
        )

        plan = resolve_scope(query="update the data-pipeline readme", store=store)
        assert plan.scopes == ("project:data-pipeline", "global")


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


def _default_scoped_store(tmp_path: Path, default_scope: str) -> MemoryStore:
    return MemoryStore(
        MemoryConfig(db_path=tmp_path / "memory.sqlite3", default_scope=default_scope)
    )


def test_scopeless_fallback_widens_search_to_configured_default_project(
    tmp_path: Path,
) -> None:
    # The request stays global (feedback closure pins that divergence), but the
    # search includes the deployment's declared default project scope.
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        plan = resolve_scope(query="checkpoint cursor resume offsets", store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("project:alpha", "global")


def test_explicit_global_scope_is_not_widened_by_default_scope(tmp_path: Path) -> None:
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        plan = resolve_scope(query="checkpoint cursor resume offsets", scope="global", store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("global",)


def test_ambient_scope_wins_over_default_scope_widening(tmp_path: Path) -> None:
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        plan = resolve_scope(
            query="checkpoint cursor resume offsets",
            ambient_context={"scope": "project:beta"},
            store=store,
        )
        assert plan.requested_scope == "project:beta"
        assert plan.scopes == ("project:beta", "global")


def test_deliberate_query_mention_wins_over_default_scope_widening(tmp_path: Path) -> None:
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        store.append_trace(
            "Ingestion worker batches records",
            {"scope": "project:data-pipeline", "agent": "agent-a"},
        )

        plan = resolve_scope(query="where are the data pipeline docs", store=store)
        assert plan.requested_scope == "project:data-pipeline"
        assert plan.scopes == ("project:data-pipeline", "global")


def test_global_default_scope_does_not_widen_fallback_plan(tmp_path: Path) -> None:
    with _default_scoped_store(tmp_path, "global") as store:
        plan = resolve_scope(query="checkpoint cursor resume offsets", store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("global",)


def test_scopeless_recall_reaches_default_scoped_memory(tmp_path: Path) -> None:
    # A trace remembered without a scope inherits the configured default; a
    # scope-less recall of its content must find it even when the query shares
    # no token with the project name — the deployment's own memory must not
    # be reachable only through a deliberate project mention.
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        marker = store.append_trace(
            "checkpoint cursor resumes from the last acked offset",
            {"agent": "agent-a"},
        )
        assert marker.scope == "project:alpha"

        results = memory_recall(store, "checkpoint cursor acked offset", max_results=5)
        assert marker.id in {result.node.id for result in results}


@pytest.mark.parametrize(
    ("raw_scope", "expected_scope"),
    [
        ("rise/_critique", "project:rise/_critique"),
        ("breakthrough/gemini-flash-contract", "project:breakthrough/gemini-flash-contract"),
        (
            "ocpa-generative-action-substrate-v1",
            "project:ocpa-generative-action-substrate-v1",
        ),
        ("future-goal/implementation-node", "project:future-goal/implementation-node"),
    ],
)
def test_write_path_canonicalizes_bare_project_scopes(
    tmp_path: Path,
    raw_scope: str,
    expected_scope: str,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            f"Trace stored under {raw_scope}",
            {"scope": raw_scope, "agent": "agent-a"},
        )

        assert trace.scope == expected_scope
        assert trace.context["scope"] == expected_scope
        assert store.list_nodes(scope=expected_scope) == [trace]
        assert store.list_nodes(scope=raw_scope) == []


def test_write_path_preserves_already_canonical_project_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "Trace already scoped to the LM project",
            {"scope": "project:lm", "agent": "agent-a"},
        )

        assert trace.scope == "project:lm"
        assert trace.context["scope"] == "project:lm"


def test_write_path_preserves_global_scope_outside_projects(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "Universal lesson that should remain global",
            {"scope": "global", "agent": "agent-a"},
        )

        assert trace.scope == "global"
        assert trace.context["scope"] == "global"


def test_write_path_rejects_unknown_scope_prefix(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        with pytest.raises(ValueError, match="unsupported scope prefix: external"):
            store.append_trace(
                "Unknown prefixes should fail loudly",
                {"scope": "external:vendor", "agent": "agent-a"},
            )
