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

        plan = resolve_scope(scope="project:alpha", store=store)
        assert plan.scopes == ("project:alpha", "global")
        assert plan.restricted

        alpha_results = memory_recall(store, "deploy migration", scope="project:alpha", max_results=10)
        alpha_ids = {result.node.id for result in alpha_results}

        assert alpha.id in alpha_ids
        assert beta.id not in alpha_ids


def _rollback_notes(store: MemoryStore) -> dict[str, str]:
    notes = {
        "session:s1": "Session note says rollback deploy now",
        "project:alpha": "Project note says rollback deploy after health checks",
        "global": "Global note says rollback deploy requires approval",
        "project:beta": "Other project rollback deploy waits for the release train",
    }
    return {
        scope: store.append_trace(content, {"scope": scope, "agent": "agent-a"}).id
        for scope, content in notes.items()
    }


def test_ambient_session_ranks_first_without_hiding_other_projects(tmp_path: Path) -> None:
    # Ambient metadata is not a restriction: it orders the answer, and the
    # other project's note stays reachable behind the session, project and
    # global notes.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rollback_notes(store)

        results = memory_recall(
            store,
            "rollback deploy",
            ambient_context={"session_id": "s1", "project": "alpha"},
            max_results=10,
        )
        ranked = [result.node.id for result in results]

        assert set(ids.values()) <= set(ranked)
        assert results[0].node.scope == "session:s1"
        assert ranked.index(ids["project:beta"]) > ranked.index(ids["project:alpha"])


def test_explicit_session_scope_restricts_to_session_project_and_global(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rollback_notes(store)

        plan = resolve_scope(scope="session:s1", ambient_context={"project": "alpha"}, store=store)
        assert plan.scopes == ("session:s1", "project:alpha", "global")

        results = memory_recall(
            store,
            "rollback deploy",
            scope="session:s1",
            ambient_context={"project": "alpha"},
            max_results=10,
        )
        ranked = {result.node.id for result in results}
        assert ranked == {ids["session:s1"], ids["project:alpha"], ids["global"]}


def test_scopeless_plan_searches_the_whole_store(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        plan = resolve_scope(store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("global", "*")
        assert plan.search_scopes == (None,)
        assert plan.allows("project:anything") and plan.allows("session:any")
        assert plan.rank("project:anything") == plan.rank("global")

        plan = resolve_scope(ambient_context={"session_id": "s1"}, store=store)
        assert plan.requested_scope == "session:s1"
        assert plan.scopes == ("session:s1", "global", "*")
        assert plan.boost_steps("session:s1") == 1
        assert plan.boost_steps("project:other") == 0


@pytest.mark.parametrize(
    "ambient",
    [None, {"session_id": "s1"}, {"project": "beta"}, {"cwd": "/work/beta"}],
)
def test_scopeless_recall_reaches_another_projects_fact(
    tmp_path: Path, ambient: dict[str, str] | None
) -> None:
    # Neither a project name hiding inside a query word ("mm" in "commit")
    # nor session/project metadata may make another project's fact unreachable.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace("Unrelated note about the mm widget", {"scope": "project:mm"})
        store.append_trace("Beta keeps release notes in the wiki", {"scope": "project:beta"})
        fact = store.append_trace(
            "Snapshot the private state ref before the immediate commit",
            {"scope": "project:ae"},
        )

        results = memory_recall(
            store,
            "private snapshot immediate commit",
            ambient_context=ambient,
            max_results=5,
        )
        assert fact.id in {result.node.id for result in results}


def test_wider_search_adds_no_noise_a_single_scope_would_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Weak vector-only matches fill free slots inside one scope too (and the
    # gate always delivers a top result); spreading the same memory over thirty
    # projects and searching them all must not add more of them.
    def noise(name: str, *, spread: bool) -> tuple[int, int]:
        with MemoryStore(tmp_path / f"{name}.sqlite3") as store:
            fact = store.append_trace(
                "Checkpoint cursor resumes from the last acked offset",
                {"scope": "project:ingest"},
            )
            fillers = {
                store.append_trace(
                    f"Team {index} lunch rota and parking badge renewal schedule",
                    {"scope": f"project:filler-{index}" if spread else "project:ingest"},
                ).id
                for index in range(30)
            }
            scope = None if spread else "project:ingest"
            answer = memory_recall(store, "checkpoint cursor acked offset", scope=scope, max_results=5)
            assert answer[0].node.id == fact.id
            nonsense = memory_recall(store, "qzvkrw plbxtn", scope=scope, max_results=5)
            return (
                sum(result.node.id in fillers for result in answer),
                sum(result.node.id in fillers for result in nonsense),
            )

    for gate in ("", "0.35"):  # off, and the deployed threshold
        monkeypatch.setenv("LM_RECALL_MIN_SCORE", gate)
        spread = noise(f"spread{gate}", spread=True)
        single = noise(f"single{gate}", spread=False)
        assert spread[0] <= single[0] and spread[1] <= single[1]
    assert spread[0] == 0


def _default_scoped_store(tmp_path: Path, default_scope: str) -> MemoryStore:
    return MemoryStore(
        MemoryConfig(db_path=tmp_path / "memory.sqlite3", default_scope=default_scope)
    )


def test_scopeless_plan_ranks_configured_default_project_first(tmp_path: Path) -> None:
    # The request stays global (feedback closure pins that divergence); the
    # deployment's declared default project ranks first in a whole-store search.
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        plan = resolve_scope(store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("project:alpha", "global", "*")


def test_explicit_global_scope_is_not_widened_by_default_scope(tmp_path: Path) -> None:
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        plan = resolve_scope(scope="global", store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("global",)


def test_ambient_scope_is_an_explicit_restriction(tmp_path: Path) -> None:
    with _default_scoped_store(tmp_path, "project:alpha") as store:
        plan = resolve_scope(ambient_context={"scope": "project:beta"}, store=store)
        assert plan.requested_scope == "project:beta"
        assert plan.scopes == ("project:beta", "global")


def test_global_default_scope_does_not_add_a_preference(tmp_path: Path) -> None:
    with _default_scoped_store(tmp_path, "global") as store:
        plan = resolve_scope(store=store)
        assert plan.requested_scope == "global"
        assert plan.scopes == ("global", "*")


def test_scopeless_recall_reaches_default_scoped_memory(tmp_path: Path) -> None:
    # A trace remembered without a scope inherits the configured default; a
    # scope-less recall of its content must find it even when the query shares
    # no token with the project name.
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
