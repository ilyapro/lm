"""Selection and delivery share chunk observations within one recall only."""

from collections import Counter
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from living_memory import near_dup, server, storage
from living_memory.chunking import TextChunk
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.storage import MemoryStore

from test_lookup import FakeMCP


SCOPE = "project:duplicate-observation"
CONTENTS = (
    "The harbour survey runs at slack water; the office countersigns on Tuesday.",
    "At slack water the harbour survey runs; Tuesday is when the office countersigns.",
    "The inventory is short; the missing crates are logged at the ferry berth.",
    "The quay gate closes at dusk; the night watch keeps the entrance clear.",
)


def configured_environment(monkeypatch):
    for name in (
        "LM_RECALL_MIN_SCORE", "LM_RECALL_REPEAT_GATING",
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "LM_DELIVERY_SESSION_DEDUP",
        "LM_RECALL_NEAR_DUP_COSINE", "LM_RECALL_NEAR_DUP_LENGTH_RATIO",
        "LM_NEAR_DUP_IDENTIFIER_VETO", "LM_RECALL_SCHEMA_DEDUP",
        "LM_RECALL_SCHEMA_TRIGGER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setattr(storage, "_utc_now", lambda: "2099-01-01T00:00:00Z")


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    configured_environment(monkeypatch)


def put_vector(store, node_id, degrees):
    angle = math.radians(degrees)
    store.replace_node_chunks(node_id, [(
        TextChunk(text=node_id, chunk_index=0, token_start=0, token_end=1,
                  char_start=0, char_end=len(node_id)),
        [math.cos(angle), math.sin(angle)],
    )])


def make_recall(tmp_path, monkeypatch, *, case="ordinary", limit=2, view="knowledge"):
    """Freeze retrieval inputs; keep the service's selection and accounting real."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    specs = list(zip(("bearer", "repeat", "distinct", "later"), CONTENTS, (0, 3, 70, 140)))
    if case == "omitted-bearer":
        specs = [
            ("first", CONTENTS[0], 0),
            ("omitted", CONTENTS[0], 70),
            ("bearer", CONTENTS[1], 74),
            ("repeat", "The office countersigns on Tuesday; the harbour survey runs at slack water.", 78),
            ("distinct", CONTENTS[2], 140),
            ("later", CONTENTS[3], 180),
        ]
        limit = 3
    elif case == "identifier":
        specs[0] = ("bearer", "Permit PX-17: " + CONTENTS[0], 0)
        specs[1] = ("repeat", "Permit PX-18: " + CONTENTS[1], 3)
    elif case == "length":
        specs[1] = ("repeat", CONTENTS[1] + " The pilot records the tide and checks the mooring before departure.", 3)
    # The tail must not enter the bounded observation even though it is ranked.
    if case == "bounded":
        specs.append(("outside", "The warehouse roof needs an inspection.", 180))
    ranked = []
    for index, (node_id, content, degrees) in enumerate(specs):
        node = store.create_node(level="trace", content=content, node_id=node_id,
                                 context={"scope": SCOPE})
        put_vector(store, node_id, degrees)
        ranked.append(RecallResult(node=node, score=1.0 - index * 0.05,
                                   bm25_score=0.5, vector_score=0.5, graph_score=0.0,
                                   trigger_score=0.0, scope_rank=0, methods=("vector",)))
    if case == "unknown":
        store.replace_node_chunks("repeat", [])
    elif case == "zero":
        store.replace_node_chunks("repeat", [(TextChunk(
            text="repeat", chunk_index=0, token_start=0, token_end=1,
            char_start=0, char_end=6), [0.0, 0.0])])
    elif case == "mixed":
        original = store.list_node_chunks
        monkeypatch.setattr(store, "list_node_chunks", lambda node_id: (
            [SimpleNamespace(dimensions=2, embedding=[1.0, 0.0]),
             SimpleNamespace(dimensions=3, embedding=[1.0, 0.0, 0.0])]
            if node_id == "repeat" else original(node_id)
        ))
    services = []

    def fixed_rank(service, *args, **kwargs):
        services[:] = [service]
        return [replace(row, node=store.get_node(row.node.id)) for row in ranked]

    monkeypatch.setattr(MemoryRecallService, "rank_candidates", fixed_rank)
    monkeypatch.setattr(MemoryRecallService, "_collect_bm25", lambda *a, **kw: None)
    monkeypatch.setattr(MemoryRecallService, "_collect_vector", lambda *a, **kw: [])
    monkeypatch.setattr(MemoryRecallService, "_collect_query_demotions", lambda *a, **kw: {})
    monkeypatch.setattr(MemoryRecallService, "_collect_hub_demotions", lambda *a, **kw: {})
    if case == "session":
        monkeypatch.setattr(store, "delivered_node_ids", lambda session: {"repeat"})
    if case == "repeat-gated":
        monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "1")
        monkeypatch.setattr(server, "should_gate_fingerprint", lambda *a: True)
        monkeypatch.setattr(store, "fingerprint_delivered_node_ids", lambda fingerprint: {"repeat"})
    reads, pools = [], []
    list_chunks, mean_pool = store.list_node_chunks, near_dup._mean_pool

    def chunks(node_id):
        reads.append(node_id)
        return list_chunks(node_id)

    def pool(vectors):
        pools.append(vectors)
        return mean_pool(vectors)

    monkeypatch.setattr(store, "list_node_chunks", chunks)
    monkeypatch.setattr(near_dup, "_mean_pool", pool)

    def run():
        reads.clear()
        pools.clear()
        before = {row.node.id: store.get_node(row.node.id).stats.get("access_count", 0) for row in ranked}
        answer = mcp.tools["memory_recall"](
            "harbour survey", scope=SCOPE, depth=0, max_results=limit, view=view,
            ambient_context={"transport_session_id": "fixture-session"},
        )
        event_id = answer["recall_event_id"]
        if limit <= 0:
            assert event_id is None and answer["results"] == []
            assert all(store.get_node(row.node.id).stats.get("access_count", 0)
                       == before[row.node.id] for row in ranked)
            return {"selected": [], "residual": [], "results": [],
                    "event_ids": [], "access_ids": [], "reads": list(reads),
                    "pools": len(pools)}
        event = store.get_recall_event(event_id)
        assert event is not None
        entries = answer["results"]
        selected = [entry.get("node", entry)["id"] for entry in entries]
        assert list(event.result_ids) == selected
        access_ids = [row.node.id for row in ranked
                      if store.get_node(row.node.id).stats.get("access_count", 0) > before[row.node.id]]
        assert set(access_ids) == set(selected)
        for entry in entries:
            node_id = entry.get("node", entry)["id"]
            if view == "knowledge" or "content_ref" in entry:
                assert entry["content_ref"]["node_id"] == node_id
        # Generated event ids are not semantic output; verify the relationship
        # before normalizing them for comparisons between fresh stores.
        if view == "full":
            assert all(entry.get("recall_event_id", event_id) == event_id for entry in entries)
            entries = [{**entry, "recall_event_id": "event"} if "recall_event_id" in entry
                       else entry for entry in entries]
        return {"selected": selected,
                "residual": [row.node.id for row in services[0].last_residual],
                "results": entries, "event_ids": list(event.result_ids),
                "access_ids": access_ids, "reads": list(reads), "pools": len(pools)}

    run.store = store
    return run


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_selection_and_delivery_observe_the_bounded_pool_once(tmp_path, monkeypatch, view):
    result = make_recall(tmp_path, monkeypatch, view=view)()
    assert result["selected"] == ["bearer", "distinct"]
    assert result["residual"] == ["repeat", "later"]
    assert result["reads"] == ["bearer", "repeat", "distinct", "later"]
    assert result["pools"] == 4
    assert [row["delivery"] for row in result["results"]] == ["full", "full"]


@pytest.mark.parametrize("case", ["unknown", "mixed", "zero", "identifier", "length"])
def test_unknown_vectors_and_guards_keep_the_fact_without_rereading(tmp_path, monkeypatch, case):
    result = make_recall(tmp_path, monkeypatch, case=case)()
    assert result["selected"] == ["bearer", "repeat"]
    assert result["reads"] == ["bearer", "repeat", "distinct", "later"]
    assert [row["delivery"] for row in result["results"]] == ["full", "full"]


def test_omitted_pool_bearer_is_replaced_in_answer_order(tmp_path, monkeypatch):
    result = make_recall(tmp_path, monkeypatch, case="omitted-bearer")()
    assert result["selected"] == ["first", "bearer", "repeat"]
    assert result["residual"] == ["omitted", "distinct", "later"]
    assert result["results"][2]["delivery"] == "near_duplicate"
    assert result["results"][2]["content_ref"]["duplicate_of"] == "bearer"
    assert set(Counter(result["reads"]).values()) == {1}
    assert "outside" not in result["reads"]


def test_next_recall_reads_changed_chunks(tmp_path, monkeypatch):
    run = make_recall(tmp_path, monkeypatch)
    assert run()["selected"] == ["bearer", "distinct"]
    with MemoryStore(run.store.db_path) as writer:
        put_vector(writer, "repeat", 140)
    # Isolate duplicate observation from intentional session repeat selection.
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")
    after = run()
    assert after["selected"] == ["bearer", "repeat"]
    assert after["reads"] == ["bearer", "repeat", "distinct", "later"]


def test_disabled_near_dup_never_reads_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_RECALL_NEAR_DUP_COSINE", "0")
    result = make_recall(tmp_path, monkeypatch)()
    assert result["selected"] == ["bearer", "repeat"]
    assert result["reads"] == []
    assert result["pools"] == 0


def test_score_gate_observes_only_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.1")
    result = make_recall(tmp_path, monkeypatch)()
    assert result["selected"] == ["bearer", "repeat"]
    assert result["reads"] == ["bearer", "repeat"]
    assert result["results"][1]["delivery"] == "near_duplicate"


@pytest.mark.parametrize("case", ["session", "repeat-gated"])
def test_intentional_repeat_selection_keeps_session_delivery(tmp_path, monkeypatch, case):
    result = make_recall(tmp_path, monkeypatch, case=case)()
    assert result["selected"] == ["bearer", "repeat"]
    assert result["results"][1]["delivery"] == "session_duplicate"
    assert result["reads"] == ["bearer", "repeat", "distinct", "later"]


@pytest.mark.parametrize("limit,expected", [(0, []), (8, ["bearer", "repeat", "distinct", "later"])])
def test_empty_and_short_answers_read_only_delivered_nodes(tmp_path, monkeypatch, limit, expected):
    result = make_recall(tmp_path, monkeypatch, limit=limit)()
    assert result["reads"] == expected
    if limit == 0:
        assert result["event_ids"] == [] and result["pools"] == 0


def test_residual_beyond_the_selection_pool_is_not_observed(tmp_path, monkeypatch):
    result = make_recall(tmp_path, monkeypatch, case="bounded")()
    assert result["residual"] == ["repeat", "later", "outside"]
    assert result["reads"] == ["bearer", "repeat", "distinct", "later"]
