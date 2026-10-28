from pathlib import Path

import pytest

from living_memory import consolidation, storage
from living_memory.consolidation import memory_teach, prepare_teach
from living_memory.decay import memory_forget
from living_memory.storage import MemoryStore


def test_memory_teach_creates_corrective_trace_and_supersedes_original(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        original = store.append_trace(
            "status endpoint is /healthz",
            {"scope": "project:api", "agent": "agent-a"},
            feedback={"confidence": 0.5, "usefulness_score": 0.4},
        )

        taught = memory_teach(
            store,
            original.id,
            "status endpoint is /readyz",
            confidence=0.95,
            context={"agent": "agent-b"},
        )

        assert taught.corrective_trace.level == "trace"
        assert taught.corrective_trace.content == "status endpoint is /readyz"
        assert taught.corrective_trace.scope == "project:api"
        assert taught.corrective_trace.confidence <= 0.5
        assert taught.corrective_trace.usefulness_score > taught.original.usefulness_score
        assert taught.supersedes.source_id == taught.corrective_trace.id
        assert taught.supersedes.target_id == original.id
        assert taught.supersedes.type == "supersedes"
        assert taught.original.corrections[-1]["new"] == "status endpoint is /readyz"

        ranked = store.search_content("status endpoint", scope="project:api")
        assert ranked[0][0].id == taught.corrective_trace.id


def test_memory_forget_soft_deletes_without_removing_node(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "temporary observation can be forgotten",
            {"scope": "global", "agent": "agent-a"},
        )

        forgotten = memory_forget(store, trace.id, "operator requested")

        assert forgotten.decayed is True
        assert forgotten.decay_reason == "operator requested"
        assert store.get_node(trace.id) is not None
        assert store.list_nodes(level="trace", scope="global") == []
        assert store.list_nodes(level="trace", scope="global", include_decayed=True)[0].id == trace.id


def test_memory_teach_rejects_missing_or_empty_corrections(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace("original fact", {"scope": "global", "agent": "agent-a"})

        with pytest.raises(ValueError):
            memory_teach(store, trace.id, " ")
        with pytest.raises(KeyError):
            memory_teach(store, "missing", "replacement fact")


@pytest.mark.parametrize("confidence,usefulness", [(0.8, 1.4), (0.1, -2.0)])
def test_teach_preserves_source_history_across_repeated_corrections(
    tmp_path, monkeypatch, confidence, usefulness,
):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        original = store.create_node(
            level="trace", content="Status endpoint is /healthz",
            context={
                "scope": "project:api", "agent": "author", "task": "old-task",
                "session_id": "old-session", "session": "old-session-alias",
                "transport_session_id": "old-transport", "procedure_id": "deploy",
                "step_order": 2, "step_description": "old step wording",
            },
            stats={
                "confidence": confidence, "usefulness_score": usefulness,
                "unique_agents": 3, "access_count": 7,
                "last_accessed": "2026-01-01T00:00:00+00:00", "temporal_hint": "current",
            },
            provenance={
                "source_traces": ["prior-source"], "inspection": {"verified": True},
                "corrections": [{"timestamp": "2026-01-01T00:00:00+00:00",
                                 "old": "status unknown", "new": "check status", "by": "inspector"}],
            },
        )
        corrections = list(original.corrections)
        identities = [
            {},
            {"agent": "reviewer", "task": "review", "session_id": "review-session",
             "transport_session_id": "review-transport"},
        ]
        correction_ids = set()
        for index, identity in enumerate(identities, 1):
            now = f"2026-02-0{index}T00:00:00+00:00"
            monkeypatch.setattr(storage, "_utc_now", lambda: now)
            monkeypatch.setattr(consolidation, "_utc_now", lambda: now)
            text = f"Status endpoint is /readyz{index}"
            taught = memory_teach(store, original.id, text, context=identity)
            teacher = identity.get("agent", "teacher")
            corrections.append({"timestamp": now, "old": original.content, "new": text, "by": teacher})
            current = store.get_node(original.id)
            assert taught.original == current
            assert current.content == original.content
            assert current.context == original.context
            assert current.timestamp == original.timestamp
            assert current.created_at == original.created_at
            assert current.updated_at == now
            assert current.source_traces == original.source_traces
            assert current.corrections == corrections
            assert current.provenance == {**original.provenance, "corrections": corrections}
            assert current.stats == {
                **original.stats, "confidence": min(confidence, 0.25),
                "usefulness_score": min(usefulness, -0.25),
            }
            corrective = taught.corrective_trace
            correction_ids.add(corrective.id)
            assert corrective.context == {
                "scope": "project:api", "procedure_id": "deploy", "step_order": 2,
                **identity, "agent": teacher, "timestamp": now,
            }
            assert corrective.timestamp == now
            assert corrective.confidence == 0.5
            assert taught.supersedes.metadata == {"by": teacher, "correction": text}
        assert len(correction_ids) == 2
        assert {edge.source_id for edge in store.list_connections(
            target_id=original.id, relation_type="supersedes",
        )} == correction_ids


def test_teach_uses_source_changes_committed_after_preparation(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        original = store.append_trace("Status endpoint is /healthz", {"scope": "project:api"})
        prepared = prepare_teach(store, original.id, "Status endpoint is /readyz")
        with MemoryStore(path) as writer:
            writer.add_correction(original.id, old=original.content, new="Check readiness", by="inspector")
            current = writer.get_node(original.id)
            current = writer.update_node(
                original.id, stats={"confidence": 0.1, "usefulness_score": -2.0, "unique_agents": 4},
                context={**current.context, "subject": "current API"},
                provenance={**current.provenance, "source_traces": ["new-source"], "reviewed": True},
            )
        taught = memory_teach(store, original.id, "Status endpoint is /readyz", preparation=prepared)
        assert taught.corrective_trace.id == prepared.corrective_id
        assert taught.corrective_trace.context["subject"] == "current API"
        assert taught.original == store.get_node(original.id)
        assert taught.original.stats == current.stats
        assert taught.original.corrections[:-1] == current.corrections
        assert taught.original.provenance == {
            **current.provenance, "corrections": taught.original.corrections,
        }

