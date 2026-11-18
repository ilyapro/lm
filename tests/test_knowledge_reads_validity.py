"""Knowledge reads preserve relation validity and their own scope/selection."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from living_memory.embeddings import LocalEmbeddingModel
from living_memory.server import create_mcp_server
from test_transport_identity import _structured


@pytest.mark.parametrize("surface", ["filter", "recent", "concepts", "prompt"])
@pytest.mark.parametrize("scope", ["project:read-validity", "global"])
@pytest.mark.parametrize("correction_state", ["active", "decayed", "missing", "foreign", "global"])
def test_native_knowledge_reads_share_relation_validity(
    tmp_path, monkeypatch, surface, scope, correction_state,
):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    level = "concept" if surface in ("concepts", "prompt") else "trace"
    original_content = "Amber hatch pressure rule: open first.\n\n  Исходник.\t\n"
    context = {
        "scope": scope, "task_pattern": "amber-hatch", "procedure_id": "amber-operation",
        "detail": {"nested": ["full context", "x" * 800]},
    }

    async def scenario():
        async with Client(mcp) as client:
            async def remember(content, context):
                result = _structured(await client.call_tool("memory_remember", {
                    "content": content, "context": context,
                }))
                return result["node"]["id"]

            original = await remember(original_content, context)
            untouched = await remember("Amber hatch seal: inspect before opening.", context)
            foreign = await remember("Foreign selection sentinel.", {
                **context, "scope": "project:foreign-validity",
            })
            for node_id in (original, untouched, foreign):
                store.update_node(node_id, level=level)

            async def read_surface():
                before = list(store.connection.iterdump())
                if surface == "filter":
                    response = _structured(await client.call_tool("memory_lookup", {
                        "scope": scope, "task_pattern": "amber-hatch", "level": level,
                    }))
                    entries = response["results"]
                    expected = store.list_nodes_by_context(
                        scope=scope, context_filters={"task_pattern": "amber-hatch"}, level=level,
                    )
                elif surface == "prompt":
                    response = await client.get_prompt("memory://prompt/retrieval_context", {
                        "scope": scope, "task": "",
                    })
                    body = response.messages[0].content.text
                    assert foreign not in body
                    assert original in body and untouched in body
                    assert list(store.connection.iterdump()) == before
                    return body
                else:
                    uri = "memory://recent" if surface == "recent" else (
                        "memory://global/concepts" if scope == "global"
                        else "memory://project/read-validity/concepts"
                    )
                    response = json.loads((await client.read_resource(uri))[0].text)
                    entries = response["traces" if surface == "recent" else "concepts"]
                    expected = store.list_nodes(
                        level=level, scope=None if surface == "recent" else scope,
                        include_decayed=False, limit=20 if surface == "recent" else 100_000,
                    )
                assert [entry["id"] for entry in entries] == [node.id for node in expected]
                assert response["count"] == len(entries)
                assert list(store.connection.iterdump()) == before
                return {entry["id"]: entry for entry in entries}

            before = await read_surface()
            exact_before = _structured(await client.call_tool("memory_lookup", {
                "node_id": original, "scope": scope,
            }))["results"][0]
            assert exact_before["superseded"] is False
            assert exact_before["superseded_by"] == []
            if surface != "prompt":
                assert before[original].get("superseded", False) is False
                assert before[original]["content"] == original_content

            correction_scope = {
                "foreign": "project:foreign-validity", "global": "global",
            }.get(correction_state, scope)
            correction = await remember(
                f"Correction to {original}: release pressure first. CORRECTION_BODY_SENTINEL",
                {"scope": correction_scope},
            )
            await client.call_tool("memory_connect", {
                "id_a": correction, "id_b": original, "relation_type": "supersedes",
            })
            edges = store.list_connections(
                source_id=correction, target_id=original, relation_type="supersedes",
            )
            assert len(edges) == 1 and edges[0].metadata == {}
            assert store.get_node(original).corrections == []
            if correction_state == "decayed":
                store.soft_delete_node(correction, "expired correction")
            elif correction_state == "missing":
                # Retain the known edge while its counterpart is absent.
                get_node_scopes = store.get_node_scopes
                monkeypatch.setattr(store, "get_node_scopes", lambda ids: {
                    key: scope for key, scope in get_node_scopes(ids).items() if key != correction
                })

            after = await read_surface()
            readable = correction_state != "missing" and (
                surface == "recent" or correction_scope == scope
                or (surface == "prompt" and correction_scope == "global")
            )
            references = [correction] if readable else []
            if surface == "prompt":
                original_line = next(line for line in after.splitlines() if f"id={original} " in line)
                control_line = next(line for line in after.splitlines() if f"id={untouched} " in line)
                assert "superseded=true" in original_line
                assert f"superseded_by={json.dumps(references)}" in original_line
                assert "superseded=false superseded_by=[]" in control_line
                assert "CORRECTION_BODY_SENTINEL" not in after
            else:
                entry = after[original]
                assert entry["superseded"] is True
                assert entry["superseded_by"] == references
                assert entry["decayed"] is False
                assert entry["provenance"]["corrections"] == []
                assert entry["content"] == original_content
                assert entry["context"]["detail"] == context["detail"]
                assert {k: v for k, v in entry.items() if k not in ("superseded", "superseded_by")} == {
                    k: v for k, v in before[original].items() if k not in ("superseded", "superseded_by")
                }
                assert after[untouched]["superseded"] is False
                assert after[untouched]["superseded_by"] == []
            if not readable and surface != "recent":
                assert correction not in json.dumps(after)
                assert "CORRECTION_BODY_SENTINEL" not in json.dumps(after)

            exact_after = _structured(await client.call_tool("memory_lookup", {
                "node_id": original, "scope": scope,
            }))["results"][0]
            assert exact_after["superseded"] is True
            assert exact_after["superseded_by"] == (
                [correction] if correction_state != "missing" and correction_scope in (scope, "global") else []
            )

    try:
        asyncio.run(scenario())
    finally:
        store.close()


@pytest.mark.parametrize("writer", ["teach", "teach_mixed", "historical_duplicate", "historical_inferred"])
def test_native_reads_preserve_teach_and_historical_relation_semantics(tmp_path, monkeypatch, writer):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    scope = "project:read-validity"

    async def scenario():
        async with Client(mcp) as client:
            original = _structured(await client.call_tool("memory_remember", {
                "content": "Amber hatch opening is allowed before approval.",
                "context": {"scope": scope, "task_pattern": "amber-hatch"},
            }))["node"]["id"]
            replacement = "Amber hatch opening is forbidden before approval."
            explicit = writer.startswith("teach")
            if explicit:
                correction = _structured(await client.call_tool("memory_teach", {
                    "trace_id": original, "correction": replacement,
                }))["corrective_trace"]["id"]
                if writer == "teach_mixed":
                    store.create_connection(correction, original, "supersedes", merge=True,
                                            metadata={"rule": "R1c", "kind": "content_correction"})
            else:
                correction = _structured(await client.call_tool("memory_remember", {
                    "content": replacement, "context": {"scope": scope},
                }))["node"]["id"]
                metadata = ({"kind": "duplicate_content"} if writer == "historical_duplicate"
                            else {"rule": "R1c", "kind": "content_correction"})
                store.create_connection(correction, original, "supersedes", metadata=metadata)

            async def check(level):
                before = list(store.connection.iterdump())
                filtered = _structured(await client.call_tool("memory_lookup", {
                    "scope": scope, "task_pattern": "amber-hatch", "level": level,
                }))["results"]
                uri = "memory://recent" if level == "trace" else "memory://project/read-validity/concepts"
                resource = json.loads((await client.read_resource(uri))[0].text)
                entries = resource["traces" if level == "trace" else "concepts"]
                for selection in (filtered, entries):
                    entry = next(item for item in selection if item["id"] == original)
                    assert entry["superseded"] is explicit
                    assert entry["superseded_by"] == ([correction] if explicit else [])
                    assert bool(entry["provenance"]["corrections"]) is explicit
                    assert entry["decayed"] is False
                if level == "concept":
                    prompt = await client.get_prompt("memory://prompt/retrieval_context", {"scope": scope, "task": ""})
                    line = next(line for line in prompt.messages[0].content.text.splitlines()
                                if f"id={original} " in line)
                    assert f"superseded={str(explicit).lower()}" in line
                    assert f"superseded_by={json.dumps([correction] if explicit else [])}" in line
                assert list(store.connection.iterdump()) == before

            await check("trace")
            store.update_node(original, level="concept")
            await check("concept")

    try:
        asyncio.run(scenario())
    finally:
        store.close()


def test_prompt_projects_validity_for_schemas_and_decision_history(tmp_path, monkeypatch):
    from living_memory.prompts import ContextSchema, DecisionHistory, format_retrieval_context
    from living_memory.scope import resolve_scope
    from living_memory.storage import MemoryStore

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        schema = store.create_node(level="schema", content="Hatch recipe", context={"trigger": "hatch"})
        decision = store.append_trace("Hatch decision")
        correction = store.append_trace("Corrected hatch recipe")
        store.create_connection(correction.id, schema.id, "supersedes")
        store.soft_delete_node(correction.id, "historical correction")
        before = list(store.connection.iterdump())
        block = format_retrieval_context(
            [], schemas=[ContextSchema(schema, 0.5)],
            decision_history=[DecisionHistory(decision, ("Rejected approach",), 0.5)],
            task="hatch", plan=resolve_scope(), store=store,
            min_confidence=0.0, retrieval_policy="balanced",
        )
        schema_line = next(line for line in block.splitlines() if f"id={schema.id} " in line)
        decision_line = next(line for line in block.splitlines() if f"id={decision.id} " in line)
        assert f'superseded=true superseded_by=["{correction.id}"]' in schema_line
        assert "superseded=false superseded_by=[]" in decision_line
        assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("selectors", [{}, {"context_key": "task_pattern"}, {"view": "full"},
                                      {"task_pattern": "amber-hatch"}])
def test_lookup_keeps_one_snapshot_across_teach(tmp_path, monkeypatch, selectors):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    db = tmp_path / "memory.sqlite3"
    reader, writer = create_mcp_server(db), create_mcp_server(db)
    store = reader.memory_store
    scope = "project:read-validity"
    original = store.append_trace("Open the hatch first.", {
        "scope": scope, "task_pattern": "amber-hatch",
    })
    args = {"scope": scope, **selectors}
    if "task_pattern" not in selectors:
        args["node_id"] = original.id

    async def teach():
        async with Client(writer) as client:
            result = await client.call_tool("memory_teach", {
                "trace_id": original.id, "correction": "Release pressure first.",
            })
            return result.data["corrective_trace"]["id"]

    async def scenario():
        async with Client(reader) as client:
            before = (await client.call_tool("memory_lookup", args)).data
            read_pairs = store.list_supersedes_pairs
            with ThreadPoolExecutor(max_workers=1) as executor, monkeypatch.context() as patch:
                corrections = []

                def teach_then_read_pairs(ids):
                    corrections.append(executor.submit(asyncio.run, teach()).result(timeout=30))
                    return read_pairs(ids)

                patch.setattr(store, "list_supersedes_pairs", teach_then_read_pairs)
                during = (await client.call_tool("memory_lookup", args)).data
            after = (await client.call_tool("memory_lookup", args)).data
            assert during == before
            correction, = corrections
            for response, expected in [(before, []), (after, [correction])]:
                row = next(row for row in response["results"] if row["id"] == original.id)
                assert len(row["provenance"]["corrections"]) == len(expected)
                assert row["superseded"] == bool(expected)
                assert row["superseded_by"] == expected

    try:
        asyncio.run(scenario())
    finally:
        for mcp in (reader, writer):
            mcp.auto_consolidation.close()
            mcp.memory_store.close()


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("threshold", ["off", "0.35"])
def test_recall_keeps_selected_knowledge_consistent_across_teach(tmp_path, monkeypatch, view, threshold):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", threshold)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")
    db = tmp_path / "memory.sqlite3"
    reader, writer = create_mcp_server(db), create_mcp_server(db)
    store = reader.memory_store
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    text = "Amber hatch before pressure release."
    revised = "Amber hatch after pressure release."
    scope = "project:read-validity"

    async def teach(original):
        async with Client(writer) as client:
            result = await client.call_tool("memory_teach", {
                "trace_id": original, "correction": revised, "context": {"scope": scope},
            })
            return result.data["corrective_trace"]["id"]

    async def scenario():
        async with Client(reader) as client:
            original = (await client.call_tool("memory_remember", {
                "content": text, "context": {"scope": scope},
            })).data["node"]["id"]

            async def recall():
                response = (await client.call_tool("memory_recall", {
                    "query": text, "depth": 0, "max_results": 5, "view": view,
                })).data
                entries = response["results"]
                assert store.get_recall_event(response["recall_event_id"]).result_ids == [
                    entry.get("node", entry)["id"] for entry in entries
                ]
                return entries

            before = await recall()
            read_pairs = store.list_supersedes_pairs
            corrections = []
            with ThreadPoolExecutor(max_workers=1) as executor, monkeypatch.context() as patch:
                def teach_then_read_pairs(ids):
                    if not corrections:
                        corrections.append(executor.submit(asyncio.run, teach(original)).result(timeout=30))
                    return read_pairs(ids)

                patch.setattr(store, "list_supersedes_pairs", teach_then_read_pairs)
                during = await recall()
            after = await recall()
            correction, = corrections

            def content_and_validity(entries):
                return [(entry.get("node", entry)["id"], entry.get("node", entry)["content"],
                         entry["delivery"], entry["superseded"], entry["superseded_by"],
                         bool(entry.get("node", entry).get("provenance", {}).get("corrections")))
                        for entry in entries]

            old = content_and_validity(before)
            new = content_and_validity(after)
            assert old == [(original, text, "full", False, [], False)]
            assert new[0][:5] == (correction, revised, "full", False, [])
            stale, = [row for row in new if row[0] == original]
            assert stale[3:] == (True, [correction], True)
            if threshold != "off":
                assert stale[1:3] == ("", "below_threshold")
            assert content_and_validity(during) in (old, new)
            for node_id in (original, correction):
                assert store.get_node(node_id).access_count == sum(
                    entry.get("node", entry)["id"] == node_id
                    for entries in (before, during, after) for entry in entries
                )
            assert len(store.list_recall_events()) == 3

    try:
        asyncio.run(scenario())
    finally:
        for mcp in (reader, writer):
            mcp.auto_consolidation.close()
            mcp.memory_store.close()


def test_broad_recall_reuses_windows_across_lookup_and_scope_move(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_RECALL_MAP": "0",
        "LM_DELIVERY_SESSION_DEDUP": "0",
    }.items():
        monkeypatch.setenv(key, value)
    db = tmp_path / "memory.sqlite3"
    reader, writer = create_mcp_server(db), create_mcp_server(db)
    store = reader.memory_store
    peer = writer.memory_store
    query = "amber hatch pressure release"
    target = "window-000"
    counts = []
    on_validity = None
    rows = store.iter_chunk_embedding_rows
    pairs = store.list_supersedes_pairs
    record = store.record_recall

    def counted_rows(**kwargs):
        for row in rows(**kwargs):
            counts[-1][1] += 1
            yield row

    def validity(ids):
        counts[-1][0] += 1
        if counts[-1][0] == 1 and on_validity:
            on_validity()
        return pairs(ids)

    def checkpoint_then_record(**kwargs):
        # The consumption savepoint has not yet touched the database. A pinned
        # selection snapshot here would block a complete checkpoint on the peer.
        assert tuple(peer.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()) == (0, 0, 0)
        return record(**kwargs)

    async def lookup():
        async with Client(writer) as client:
            response = await client.call_tool("memory_lookup", {"node_id": target})
            assert response.data["results"][0]["content"] == query

    async def scenario():
        nonlocal on_validity
        encoder = LocalEmbeddingModel()
        for i in range(128):
            content = query if i == 0 else f"spare parts inventory {i}"
            store.create_node(node_id=f"window-{i:03}", level="trace",
                              content=content, embedding=encoder.embed(content),
                              context={"scope": f"project:s{i % 4}"})
        monkeypatch.setattr(store, "list_supersedes_pairs", validity)
        monkeypatch.setattr(store, "iter_chunk_embedding_rows", counted_rows)
        monkeypatch.setattr(store, "record_recall", checkpoint_then_record)
        peer.connection.execute("PRAGMA busy_timeout = 0")
        async with Client(reader) as client:
            async def recall():
                counts.append([0, 0])
                result = (await client.call_tool("memory_recall", {
                    "query": query, "depth": 0, "max_results": 1,
                })).data
                entry, = result["results"]
                assert (entry["id"], entry["content"], entry["delivery"]) == (target, query, "full")
                assert not entry["superseded"]
                assert store.get_recall_event(result["recall_event_id"]).result_ids == [target]
                assert store.get_node(target).access_count == len(counts)
                assert len(store.list_recall_events()) == len(counts)
                return entry["scope"]

            assert await recall() == "project:s0"
            assert await recall() == "project:s0"
            with ThreadPoolExecutor(max_workers=1) as executor:
                on_validity = lambda: executor.submit(asyncio.run, lookup()).result(timeout=30)
                assert await recall() == "project:s0"
            on_validity = None
            assert peer.connection.execute("SELECT count(*) FROM recall_lookup_events").fetchone()[0] == 1
            assert await recall() == "project:s0"
            assert counts == [[1, 128], [1, 0], [1, 0], [1, 0]]

            on_validity = lambda: peer.update_node(target, context={"scope": "project:s1"})
            assert await recall() == "project:s0"
            on_validity = None
            assert await recall() == "project:s1"
            assert counts[-2:] == [[1, 0], [1, 64]]

    try:
        asyncio.run(scenario())
    finally:
        for mcp in (reader, writer):
            mcp.auto_consolidation.close()
            mcp.memory_store.close()


def test_mcp_selection_does_not_cache_rolled_back_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")
    db = tmp_path / "memory.sqlite3"
    reader, writer = create_mcp_server(db), create_mcp_server(db)
    store = reader.memory_store
    scope = "project:inside"
    outside = {"scope": "project:outside"}
    query = "amber hatch pressure release"
    vector = LocalEmbeddingModel().embed(query)
    try:
        for name in ("first", "second"):
            store.create_node(node_id=name, level="trace", content=f"Inventory {name}",
                              context={"scope": scope}, embedding=vector)

        async def scenario():
            async with Client(reader) as client:
                async def recall():
                    response = await client.call_tool("memory_recall", {
                        "query": query, "depth": 0, "scope": scope,
                    })
                    return {row["id"] for row in response.data["results"]}

                assert await recall() == {"first", "second"}
                peer = writer.memory_store
                peer.connection.execute("BEGIN")
                peer.update_node("first", context=outside)
                uncommitted = peer.chunk_corpus_revisions()
                pairs = store.list_supersedes_pairs

                def rollback_during_read(ids):
                    assert peer.connection.in_transaction
                    peer.connection.rollback()
                    return pairs(ids)

                with monkeypatch.context() as patch:
                    patch.setattr(store, "list_supersedes_pairs", rollback_during_read)
                    assert await recall() == {"first", "second"}
                peer.update_node("second", context=outside)
                assert store.chunk_corpus_revisions() == uncommitted
                assert await recall() == {"first"}
                assert len(store.list_recall_events()) == 3

        asyncio.run(scenario())
    finally:
        for mcp in (reader, writer):
            mcp.auto_consolidation.close()
            mcp.memory_store.close()
