"""Frozen MCP slot case and its delivery/accounting invariants."""

from __future__ import annotations

import json
import time
from pathlib import Path

from living_memory.chunking import TextChunk
from living_memory.server import create_mcp_server

from fixtures.recall_distinct_slots import (
    CORPUS, CORPUS_SHA256, FakeMCP, SCOPE, STEM, _run, _vector,
)


def test_frozen_paired_answer_preserves_distinct_facts() -> None:
    result = _run(expect_baseline=False)
    assert result["corpus_sha256"] == CORPUS_SHA256
    assert result["queries"]["loss"]["keys"] == ["survey_a", "inventory"]
    assert result["loss_repeat_slots"] == 0
    assert result["necessary_mcp_calls"] == 1
    assert result["queries"]["reference"]["keys"] == [
        "survey_a", "survey_b", "inventory"
    ]
    assert result["queries"]["distinctions"]["keys"] == [
        "new_rule", "positive", "negative", "id_alpha", "id_beta"
    ]


def test_candidate_wire_bytes_ignore_ambient_encoder_and_second(monkeypatch) -> None:
    recorded = json.loads(Path("artifacts/recall-distinct-slots/candidate.json").read_text())
    observed = []
    for backend in ("hash", "auto"):
        monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", backend)
        # Start the second run after a real wall-clock second transition.
        if observed:
            before = int(time.time())
            while int(time.time()) == before:
                time.sleep(0.01)
        result = _run(expect_baseline=False)
        observed.append(result)
        for name in ("loss", "reference", "distinctions"):
            assert {k: v for k, v in result["queries"][name].items()
                    if k != "elapsed_ms"} == {
                k: v for k, v in recorded["queries"][name].items()
                if k != "elapsed_ms"
            }
        assert result["necessary_response_bytes"] == recorded["necessary_response_bytes"]
        assert result["necessary_lookup_bytes"] == recorded["necessary_lookup_bytes"]
    assert observed[0]["queries"]["distinctions"]["response_bytes"] == 3734


def test_selected_event_lookup_and_intentional_refetch(tmp_path, monkeypatch) -> None:
    for knob in (
        "LM_RECALL_NEAR_DUP_COSINE", "LM_RECALL_NEAR_DUP_LENGTH_RATIO",
        "LM_NEAR_DUP_IDENTIFIER_VETO", "LM_RECALL_MIN_SCORE",
        "LM_DELIVERY_SESSION_DEDUP", "LM_RECALL_REPEAT_GATING",
    ):
        monkeypatch.delenv(knob, raising=False)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    ids = {
        key: store.create_node(
            level="trace", content=content,
            context={"scope": SCOPE, "agent": "synthetic-fixture"},
        ).id
        for key, content, _angle in CORPUS
    }
    store.create_connection(ids["new_rule"], ids["old_rule"], "supersedes")
    recall = mcp.tools["memory_recall"]
    recall(query=STEM, scope=SCOPE, max_results=len(CORPUS))
    width = len(store.list_node_chunks(ids["survey_a"])[0].embedding)
    for key, _content, angle in CORPUS:
        store.replace_node_chunks(ids[key], [(
            TextChunk(text=key, chunk_index=0, token_start=0, token_end=1,
                      char_start=0, char_end=len(key)),
            _vector(width, angle),
        )])

    context = {"transport_session_id": "synthetic-session"}
    answer = recall(query=STEM, scope=SCOPE, max_results=2, depth=0,
                    ambient_context=context)
    delivered = [entry["node"]["id"] for entry in answer["results"]]
    assert delivered == [ids["survey_a"], ids["inventory"]]
    event = store.get_recall_event(answer["recall_event_id"])
    assert event is not None and event.result_ids == delivered
    assert ids["survey_b"] not in event.result_ids
    assert mcp.tools["memory_lookup"](node_id=ids["survey_b"])["results"][0][
        "content"
    ] == dict((key, content) for key, content, _ in CORPUS)["survey_b"]

    repeated = recall(query=STEM, scope=SCOPE, max_results=2, depth=0,
                      ambient_context=context, used=[ids["survey_a"]])
    assert repeated["results"][0]["node"]["id"] == ids["survey_a"]
    assert repeated["results"][0]["delivery"] == "session_duplicate"
    assert repeated["feedback_marks"]["accepted"] == 1
    repeated_event = store.get_recall_event(repeated["recall_event_id"])
    assert repeated_event is not None
    assert repeated_event.result_ids == [
        entry["node"]["id"] for entry in repeated["results"]
    ]
