"""The graph channel's entry from query space, at the retrieval boundary.

Five contracts are under test here, and they are the five the mechanism can
fail at.

*A jargon query reaches an anchor-only node.* The falsifiable one. The node's
content is English, the query is the operator's Russian jargon, and the measured
cosine between them is 0.17 — under the decoys, which is why it sits outside the
top 5. It becomes reachable through an anchor written from a *different* past
query («глянь мердж-реквест 9443» for «посмотри мердж-реквест 9806», measured
0.916 with the shipped encoder): different verb, different ticket number, so the
anchor is not a copy of the query under test and nothing in the store contains
the evaluated query's words.

*Cold start is free.* With no anchors — or on a query class no anchor
resembles — ranking must be what it was before anchors existed. That is asserted
against the actual pre-anchor implementation, loaded out of git at
:data:`REFERENCE_COMMIT` and run side by side over the same store, rather than
against the new code with a flag flipped: a flag comparison can only prove the
flag is wired, never that the surrounding refactor left the ranking alone. The
same reference arm is what makes the regression above a regression — it fails on
the old code because the old code is what produced the failing arm.

*An anchor never demotes the node it seeds.* Extra evidence must not cost a
candidate anything. It used to: an anchor seed was a new way for an
already-strong candidate to acquire a *small* graph score, which tripped the
graph-weight floor and moved up to a quarter of the per-scope weight off the
lexical/vector evidence carrying it. That arm is pinned against the code that
had the defect (:data:`PRE_FIX_COMMIT`), so the fixture has to keep provoking it
for the invariant to mean anything — and the floor itself has to keep firing
unchanged for a graph score that came from an ordinary traversal.

*An anchor is never an answer.* Anchors are query text, not knowledge. No anchor
id and no anchor's query text may appear anywhere a caller can read.

*An anchor never crosses scope.* An anchor seeds only where its own scope is
admitted by the ``ScopePlan``, and the nodes it names are filtered by that same
plan.

Why a subprocess
----------------

Every claim here is a claim about what the shipped multilingual encoder does to
the operator's jargon, and ``scripts/test.sh`` exports
``LIVING_MEMORY_EMBEDDING_BACKEND=hash`` so the suite does not load torch. Under
the hash fallback the premise does not exist. Clearing the variable in-process
is not enough — ``embeddings._MODEL_CACHE`` would then hand the real model to
later tests that mock ``sentence_transformers``, and ``torch`` would stay in
``sys.modules`` for tests that assert it was never imported. Both were observed
failing that way for ``test_chunk_tail_regression``, whose shape this follows:
one child process measures, and every assertion lives here, in the parent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from living_memory.query_anchors import (
    ANCHOR_EDGE_WEIGHT,
    ANCHOR_MATCH_COSINE_THRESHOLD,
)
from living_memory.retrieval import GRAPH_SEED_LIMIT, VECTOR_MATCH_THRESHOLD

#: The commit whose ``retrieval.py`` is "today" for the byte-identity check —
#: the anchor-store-edges merge, the last state of the file before the graph
#: channel gained an entry from query space. Pinned rather than ``HEAD`` on
#: purpose: once this node's work is committed, ``HEAD`` is the *new* code and
#: comparing against it would silently become a comparison of the change with
#: itself.
REFERENCE_COMMIT = "ed20653"

#: The commit that first gave the graph channel an entry from query space, and
#: with it the demotion this node fixes: an anchor seed handed an already-strong
#: candidate a small graph score, which tripped the graph-weight floor and moved
#: up to a quarter of the per-scope weight off the evidence that was carrying it
#: (``result.md`` section 6). Pinned so the monotonicity tests below have an arm
#: that still *has* the defect — without it "anchors never lower a score" could
#: pass on a fixture that never provoked one.
PRE_FIX_COMMIT = "71d4005"

SCOPE = "project:demo"
OTHER_SCOPE = "project:other"

#: The node the jargon query must find. English, about merge-request review,
#: and invisible to both existing content channels for a Russian jargon query.
REVIEW_NODE = (
    "Merge request review checklist: verify migration ordering, confirm the "
    "rollback path, and require a green pipeline before approval."
)

NEIGHBOUR_NODE = (
    "Pipeline gate configuration for review approvals lives in ci/review.yml"
)

#: What the query *does* look like to the four existing channels: Russian, on
#: unrelated subjects, each outscoring the English target — which is why the
#: target sits outside the top 5 without an anchor.
DECOYS = (
    "Мердж-реквест 9443 закрыт без ревью: ветка удалена вместе с историей",
    "Реквест на доступ к стенду 9806 отправлен администратору кластера",
    "Глянь логи ночного прогона: упал тест мультиязычного поиска",
    "Посмотри отчёт по нагрузке: очередь реквестов выросла втрое",
    "Ревью дизайна интерфейса перенесено на следующую неделю",
    "Мердж ветки релиза заблокирован конфликтом в файле миграции",
    "Реквест 9443 в трекере переведён в статус ожидания",
)

#: The pair the regression rests on: different verb *and* different ticket
#: number. Measured query-to-anchor 0.916, query-to-content 0.172.
JARGON_QUERY = "посмотри мердж-реквест 9806"
ANCHOR_QUERY = "глянь мердж-реквест 9443"

#: A second pair whose query-to-content cosine (0.065) sits *below*
#: :data:`VECTOR_MATCH_THRESHOLD`, so the vector channel cannot produce the node
#: as a candidate at all and "no other channel found it" is literal rather than
#: a matter of ranking. Same jargon verb, different ticket — the «поревьювь»
#: class the goal names, measured at 0.960.
INVISIBLE_QUERY = "поревьювь мердж-реквест 9806"
INVISIBLE_ANCHOR = "поревьювь мердж-реквест 9443"

#: Queries and modes the cold-start comparison sweeps. Each reaches a different
#: combination of channels; ``causal`` and ``decision`` are here because they
#: are the two modes that rewrite how the graph is walked and scored, which is
#: where an anchor seed could do damage unnoticed.
COLD_START_MATRIX = (
    ("deployment restart after schema migration", 1),
    ("why did the checkout deploy incident happen", "causal"),
    ("backup snapshot before mutating database", 1),
    ("merge request review checklist rollback", 2),
    (JARGON_QUERY, 1),
    ("глянь логи ночного прогона", 1),
    ("deployment restart after schema migration", 0),
    ("deployment restart after schema migration", "decision"),
)

#: Anchors for the "no anchor resembles this" arm: a real, non-trivial anchor
#: population that simply belongs to other classes of question.
UNRELATED_ANCHOR_QUERIES = (
    "разложи 3d-модель по слоям для печати",
    "какая погода в осло в феврале",
    "recipe for sourdough starter hydration",
)

TOP_K = 5
MAX_RESULTS = 10


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------

#: Measurement only — every assertion is in the parent. Reads its inputs as
#: JSON on stdin and writes one JSON object to stdout, so the corpus above is
#: defined exactly once.
PROBE = r"""
import json, subprocess, sys, tempfile, types
from pathlib import Path

payload = json.load(sys.stdin)

from living_memory import query_anchors as qa
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.retrieval import MemoryRecallService
from living_memory.scope import ScopeResolver
from living_memory.storage import MemoryStore

SCOPE = payload["scope"]
OTHER_SCOPE = payload["other_scope"]
REVIEW = payload["review_node"]
NEIGHBOUR = payload["neighbour_node"]
DECOYS = payload["decoys"]
JARGON = payload["jargon_query"]
ANCHOR_Q = payload["anchor_query"]
INVISIBLE = payload["invisible_query"]
INVISIBLE_A = payload["invisible_anchor"]
MAX_RESULTS = payload["max_results"]

model = LocalEmbeddingModel()
model.warmup()
if model._sentence_transformer() is None:
    print(json.dumps({"model_available": False}))
    raise SystemExit(0)


def store_at(directory, name="memory.sqlite3"):
    return MemoryStore(Path(directory) / name)


def service(store, anchors=True):
    return MemoryRecallService(store, embedder=model, anchor_seeding=anchors)


def recall(svc, query, scope=SCOPE, depth=1, max_results=MAX_RESULTS):
    return svc.memory_recall(
        query, scope=scope, depth=depth, max_results=max_results,
        log_access=False, log_event=False,
    )


def ids(results):
    return [r.node.id for r in results]


def blob(results):
    return json.dumps([r.to_dict() for r in results], sort_keys=True)


def warm(svc):
    # The lazy embedding backfill rewrites updated_at, and _collect_bm25 reads
    # node rows before it runs -- so an undrained store hands the first arm of
    # a comparison pre-backfill rows and the second post-backfill ones.
    for scope in (SCOPE, "global"):
        svc.memory_recall("warm the lazy embedding backfill", scope=scope,
                          log_access=False, log_event=False)


def anchor(store, query, targets, scope=SCOPE, consumptions=1):
    out = None
    for _ in range(consumptions):
        out = qa.upsert_anchor(store, query, scope, model.embed(query), targets=list(targets))
    return out


def module_at(commit, name):
    source = subprocess.run(
        ["git", "-C", payload["repo_root"], "show",
         commit + ":src/living_memory/retrieval.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, "<retrieval@" + commit + ">", "exec"), module.__dict__)
    # The frozen file still passes ``query`` to the resolver, which no longer
    # reads it. Every comparison names its scope, so the plans it gets are the
    # restricted ones it was written against.
    module.ScopeResolver = QueryTolerantResolver
    return module


class QueryTolerantResolver(ScopeResolver):
    def resolve(self, *, query="", **kwargs):
        return super().resolve(**kwargs)


_reference_module = module_at(payload["reference_commit"], "living_memory._retrieval_pre_anchor")
_pre_fix_module = module_at(payload["pre_fix_commit"], "living_memory._retrieval_pre_floor_fix")


def reference_service(store):
    # No relative imports in that file, so it binds to the same storage,
    # feedback and scope modules this process already imported: the code under
    # test is the only difference between the two services.
    return _reference_module.MemoryRecallService(store, embedder=model)


def pre_fix_service(store, anchors=True):
    return _pre_fix_module.MemoryRecallService(store, embedder=model, anchor_seeding=anchors)


def jargon_corpus(store):
    target = store.append_trace(REVIEW, {"scope": SCOPE, "agent": "reviewer"})
    neighbour = store.append_trace(NEIGHBOUR, {"scope": SCOPE, "agent": "reviewer"})
    store.create_connection(target.id, neighbour.id, "related", weight=1.0)
    for text in DECOYS:
        store.append_trace(text, {"scope": SCOPE, "agent": "operator"})
    return target.id, neighbour.id


def mixed_corpus(store):
    nodes = {}
    nodes["deploy"] = store.append_trace(
        "Deployment runbook: restart the server after the schema migration and "
        "run the write smoke test before declaring the rollout done.",
        {"scope": SCOPE, "agent": "operator"},
        feedback={"confidence": 0.7, "usefulness_score": 0.6},
    ).id
    nodes["incident"] = store.append_trace(
        "Missing migration file caused the checkout deploy incident last Friday",
        {"scope": SCOPE, "agent": "operator"},
    ).id
    nodes["effect"] = store.append_trace(
        "Checkout deploy incident happened during the release window",
        {"scope": SCOPE, "agent": "operator"},
    ).id
    nodes["review"] = store.append_trace(REVIEW, {"scope": SCOPE}).id
    nodes["global"] = store.append_trace(
        "Restart the MCP server after any additive schema migration", {"scope": "global"}
    ).id
    for text in DECOYS:
        store.append_trace(text, {"scope": SCOPE, "agent": "operator"})
    nodes["schema"] = store.create_node(
        level="schema",
        content="Always take a sqlite backup-API snapshot before mutating the live database",
        context={"scope": SCOPE, "trigger": "backup snapshot before mutating database"},
    ).id
    store.create_connection(nodes["incident"], nodes["effect"], "caused", weight=1.0)
    store.create_connection(nodes["deploy"], nodes["review"], "related", weight=0.9)
    return nodes


result = {"model_available": True}
jargon_vector = model.embed(JARGON)
invisible_vector = model.embed(INVISIBLE)
result["cosines"] = {
    "jargon_to_content": cosine_similarity(jargon_vector, model.embed(REVIEW)),
    "jargon_to_anchor": cosine_similarity(jargon_vector, model.embed(ANCHOR_Q)),
    "invisible_to_content": cosine_similarity(invisible_vector, model.embed(REVIEW)),
    "invisible_to_anchor": cosine_similarity(invisible_vector, model.embed(INVISIBLE_A)),
}

# --- the regression: outside top 5 on the old code, inside it with anchors ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-jargon-") as directory:
    with store_at(directory) as store:
        target_id, neighbour_id = jargon_corpus(store)
        anchor(store, ANCHOR_Q, [target_id], consumptions=4)
        reference = reference_service(store)
        warm(reference)
        before = recall(reference, JARGON)
        without = recall(service(store, anchors=False), JARGON)
        with_anchors = recall(service(store, anchors=True), JARGON)
        matched = next((r for r in with_anchors if r.node.id == target_id), None)
        hopped = next((r for r in with_anchors if r.node.id == neighbour_id), None)
        result["jargon"] = {
            "target": target_id,
            "neighbour": neighbour_id,
            "before_ids": ids(before),
            "without_ids": ids(without),
            "with_ids": ids(with_anchors),
            "without_matches_reference": blob(without) == blob(before),
            "target_graph_score": None if matched is None else matched.graph_score,
            "target_vector_score": None if matched is None else matched.vector_score,
            "target_bm25_score": None if matched is None else matched.bm25_score,
            "target_methods": [] if matched is None else list(matched.methods),
            "neighbour_graph_score": None if hopped is None else hopped.graph_score,
            "neighbour_methods": [] if hopped is None else list(hopped.methods),
        }

# --- the blind spot: BFS used to need another channel to seed it ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-blind-") as directory:
    with store_at(directory) as store:
        target = store.append_trace(REVIEW, {"scope": SCOPE})
        without = recall(service(store, anchors=False), INVISIBLE)
        anchor(store, INVISIBLE_A, [target.id], consumptions=4)
        with_anchors = recall(service(store, anchors=True), INVISIBLE)
        first = with_anchors[0] if with_anchors else None
        result["blind_spot"] = {
            "target": target.id,
            "without_ids": ids(without),
            "with_ids": ids(with_anchors),
            "scores": None if first is None else {
                "bm25": first.bm25_score, "vector": first.vector_score,
                "graph": first.graph_score, "trigger": first.trigger_score,
                "methods": list(first.methods), "path": list(first.path),
            },
        }

# --- activation is the anchor's cosine times its edge weight ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-activation-") as directory:
    with store_at(directory) as store:
        target = store.append_trace(REVIEW, {"scope": SCOPE})
        anchor(store, INVISIBLE_A, [target.id], consumptions=1)
        once = recall(service(store), INVISIBLE)
        for _ in range(3):
            anchor(store, INVISIBLE_A, [target.id], consumptions=1)
        saturated = recall(service(store), INVISIBLE)
        edges = store.list_query_anchor_edges(target_id=target.id)
        result["activation"] = {
            "similarity": cosine_similarity(invisible_vector, model.embed(INVISIBLE_A)),
            "once": once[0].graph_score if once else None,
            "saturated": saturated[0].graph_score if saturated else None,
            "edge_weight": edges[0].weight if edges else None,
            "edge_hits": edges[0].hits if edges else None,
        }

# --- cold start: no anchors at all ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-cold-") as directory:
    with store_at(directory) as store:
        mixed_corpus(store)
        reference = reference_service(store)
        current = service(store)
        warm(reference)
        divergences = []
        for query, depth in payload["cold_start_matrix"]:
            for scope in (SCOPE, "global"):
                expected = blob(recall(reference, query, scope=scope, depth=depth))
                actual = blob(recall(current, query, scope=scope, depth=depth))
                if actual != expected:
                    divergences.append(f"{query!r} depth={depth!r} scope={scope}")
        result["cold_start"] = {
            "anchor_count": store.count_query_anchors(),
            "divergences": divergences,
            "comparisons": len(payload["cold_start_matrix"]) * 2,
        }

# --- anchors present, none close enough ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-unrelated-") as directory:
    with store_at(directory) as store:
        nodes = mixed_corpus(store)
        for query in payload["unrelated_anchor_queries"]:
            anchor(store, query, [nodes["review"]], consumptions=4)
        reference = reference_service(store)
        current = service(store)
        warm(reference)
        divergences = []
        for query, depth in payload["cold_start_matrix"]:
            expected = blob(recall(reference, query, depth=depth))
            actual = blob(recall(current, query, depth=depth))
            if actual != expected:
                divergences.append(f"{query!r} depth={depth!r}")
        result["unresembled"] = {
            "anchor_count": store.count_query_anchors(),
            "divergences": divergences,
            "comparisons": len(payload["cold_start_matrix"]),
        }

# --- the scan itself: skipped when there is nothing to scan, cached when not ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-scan-") as directory:
    with store_at(directory) as store:
        nodes = mixed_corpus(store)
        svc = service(store)
        counter = {"scans": 0}
        original = store.iter_query_anchor_vectors

        def counting(*a, **k):
            counter["scans"] += 1
            return original(*a, **k)

        store.iter_query_anchor_vectors = counting
        counts = {}
        recall(svc, "deployment restart after schema migration")
        counts["empty_corpus"] = counter["scans"]
        # The anchor *write* path runs a scan of its own for cosine dedup, so
        # the counter is reset around each recall.
        anchor(store, ANCHOR_Q, [nodes["review"]])
        counter["scans"] = 0
        recall(svc, JARGON)
        counts["first_recall"] = counter["scans"]
        recall(svc, JARGON)
        counts["second_recall"] = counter["scans"]
        anchor(store, "поревьювь реквест EZ-13871", [nodes["deploy"]])
        counter["scans"] = 0
        revived = recall(svc, "поревьювь реквест EZ-12826")
        counts["after_new_anchor"] = counter["scans"]
        counts["new_anchor_target_found"] = nodes["deploy"] in ids(revived)
        del store.iter_query_anchor_vectors
        result["scan_counts"] = counts

# --- depth=0 turns the graph off, and anchors are an entry into the graph ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-depth0-") as directory:
    with store_at(directory) as store:
        target = store.append_trace(REVIEW, {"scope": SCOPE})
        anchor(store, INVISIBLE_A, [target.id], consumptions=4)
        svc = service(store)
        result["graph_off"] = {
            "target": target.id,
            "depth_0": ids(recall(svc, INVISIBLE, depth=0)),
            "depth_1": ids(recall(svc, INVISIBLE, depth=1)),
        }

# --- an anchor seed must never cost the node it seeds ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-monotone-") as directory:
    with store_at(directory) as store:
        seeded = None
        nodes = {}
        for index, text in enumerate(DECOYS):
            nodes["d%d" % index] = store.append_trace(
                text, {"scope": SCOPE, "agent": "operator"}
            ).id
        nodes["review"] = store.append_trace(REVIEW, {"scope": SCOPE}).id
        # A real edge out of the seeded node, so the same recall also carries a
        # candidate whose graph score comes from the ordinary traversal.
        store.create_connection(nodes["d0"], nodes["review"], "related", weight=0.9)
        # The live regime, where the floor's deficit is largest: online weight
        # learning drives bm25 to ~0 and leaves graph at a few percent, so
        # lifting graph to 0.25 takes almost a quarter of the weight off the
        # vector channel that is actually carrying every candidate.
        store.set_retrieval_weights(SCOPE, bm25=0.05, vector=0.93, graph=0.02)
        seeded = nodes["d0"]
        # One consumption, so the edge sits at ANCHOR_EDGE_WEIGHT and the
        # activation is small -- which is the case that used to lose score.
        anchor(store, ANCHOR_Q, [seeded], consumptions=1)
        weights = store.get_retrieval_weights(SCOPE).normalized()
        # Four arms compare byte for byte below, so the lazy embedding backfill
        # has to be drained before the first of them rather than by it.
        warm(service(store, anchors=False))

        def arm(results):
            return {
                r.node.id: {
                    "score": r.score,
                    "bm25": r.bm25_score,
                    "vector": r.vector_score,
                    "graph": r.graph_score,
                    "methods": list(r.methods),
                }
                for r in results
            }

        old_off = recall(pre_fix_service(store, anchors=False), JARGON)
        old_on = recall(pre_fix_service(store, anchors=True), JARGON)
        new_off = recall(service(store, anchors=False), JARGON)
        new_on = recall(service(store, anchors=True), JARGON)
        pre_anchor = recall(reference_service(store), JARGON)
        result["monotonicity"] = {
            "seeded": seeded,
            "traversed": nodes["review"],
            "weights": {
                "bm25": weights.bm25,
                "vector": weights.vector,
                "graph": weights.graph,
            },
            "old_off": arm(old_off),
            "old_on": arm(old_on),
            "new_off": arm(new_off),
            "new_on": arm(new_on),
            "old_off_ids": ids(old_off),
            "old_on_ids": ids(old_on),
            "new_off_ids": ids(new_off),
            "new_on_ids": ids(new_on),
            "new_off_matches_pre_fix": blob(new_off) == blob(old_off),
            "new_off_matches_pre_anchor": blob(new_off) == blob(pre_anchor),
        }

# --- an anchor is never an answer ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-leak-") as directory:
    with store_at(directory) as store:
        target_id, _neighbour_id = jargon_corpus(store)
        upsert = anchor(store, ANCHOR_Q, [target_id], consumptions=4)
        anchor_id = upsert.anchor.id
        svc = service(store)
        results = svc.memory_recall(JARGON, scope=SCOPE, max_results=MAX_RESULTS)
        payload_blob = blob(results)
        event = store.get_recall_event(svc.last_recall_event_id)
        event_blob = json.dumps(event.to_dict(), sort_keys=True)
        result["leak"] = {
            "anchor_id": anchor_id,
            "anchor_query": ANCHOR_Q,
            "target": target_id,
            "result_ids": ids(results),
            "payload_has_anchor_id": anchor_id in payload_blob,
            "payload_has_anchor_query": ANCHOR_Q in payload_blob,
            "paths": [list(r.path) for r in results],
            "event_has_anchor_id": anchor_id in event_blob,
            "event_has_anchor_query": ANCHOR_Q in event_blob,
            "anchor_is_a_node": store.get_node(anchor_id) is not None,
            "nodes_carrying_anchor_text": store.connection.execute(
                "SELECT COUNT(*) FROM nodes WHERE content LIKE ?", ("%" + ANCHOR_Q + "%",)
            ).fetchone()[0],
            "connections_touching_anchor": store.connection.execute(
                "SELECT COUNT(*) FROM connections WHERE source_id = ? OR target_id = ?",
                (anchor_id, anchor_id),
            ).fetchone()[0],
        }

# --- an anchor never crosses scope ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-scope-") as directory:
    with store_at(directory) as store:
        target = store.append_trace(REVIEW, {"scope": SCOPE})
        anchor(store, INVISIBLE_A, [target.id], scope=OTHER_SCOPE, consumptions=4)
        svc = service(store)
        isolation = {
            "target": target.id,
            "from_demo_with_other_scope_anchor": ids(recall(svc, INVISIBLE, scope=SCOPE)),
            "from_other_scope": ids(recall(svc, INVISIBLE, scope=OTHER_SCOPE)),
        }
        anchor(store, INVISIBLE_A, [target.id], scope=SCOPE, consumptions=4)
        isolation["after_same_scope_anchor"] = ids(recall(service(store), INVISIBLE, scope=SCOPE))
        result["scope_isolation"] = isolation

with tempfile.TemporaryDirectory(prefix="lm-anchor-plan-") as directory:
    with store_at(directory) as store:
        in_plan = store.append_trace(REVIEW, {"scope": "global"})
        out_of_plan = store.append_trace(NEIGHBOUR, {"scope": OTHER_SCOPE})
        anchor(store, INVISIBLE_A, [in_plan.id, out_of_plan.id], scope="global", consumptions=4)
        result["plan_filter"] = {
            "in_plan": in_plan.id,
            "out_of_plan": out_of_plan.id,
            "ids": ids(recall(service(store), INVISIBLE, scope="global")),
        }

# --- soft deletion, on both ends of the edge ---
with tempfile.TemporaryDirectory(prefix="lm-anchor-decay-") as directory:
    with store_at(directory) as store:
        target = store.append_trace(REVIEW, {"scope": SCOPE})
        anchor(store, INVISIBLE_A, [target.id], consumptions=4)
        before_ids = ids(recall(service(store), INVISIBLE))
        store.soft_delete_node(target.id, "test")
        result["decayed_target"] = {
            "target": target.id,
            "before": before_ids,
            "after": ids(recall(service(store), INVISIBLE)),
        }

with tempfile.TemporaryDirectory(prefix="lm-anchor-decay2-") as directory:
    with store_at(directory) as store:
        target = store.append_trace(REVIEW, {"scope": SCOPE})
        upsert = anchor(store, INVISIBLE_A, [target.id], consumptions=4)
        live = ids(recall(service(store), INVISIBLE))
        store.decay_query_anchor(upsert.anchor.id, qa.ANCHOR_TTL_DECAY_REASON)
        decayed = ids(recall(service(store), INVISIBLE))
        store.reinforce_query_anchor(upsert.anchor.id)
        result["decayed_anchor"] = {
            "target": target.id,
            "live": live,
            "decayed": decayed,
            "revived": ids(recall(service(store), INVISIBLE)),
        }

print(json.dumps(result))
"""


def run_probe() -> dict[str, Any]:
    """Run the model-dependent probe in a child process and return its JSON."""

    root = Path(__file__).resolve().parent.parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])]
    )
    # The whole point of the child: the real encoder, with nothing of it left
    # behind in this process.
    environment.pop("LIVING_MEMORY_EMBEDDING_BACKEND", None)
    # The reference and pre-fix services are retrieval.py frozen at older
    # commits, bound to the live storage and embeddings modules. Their BM25
    # expansion predates the Cyrillic prefix terms the current
    # ``_expanded_query`` emits, so with Russian stemming on the two services
    # would differ in the BM25 query of every Cyrillic probe query -- a
    # tokenizer difference, not the anchor difference these comparisons
    # isolate. Hold the switch at its pre-stemming behaviour for the whole
    # probe so the code under test stays the only difference.
    environment["LM_TOKENIZE_CYRILLIC_STEM"] = "off"

    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        input=json.dumps(
            {
                "repo_root": str(root),
                "reference_commit": REFERENCE_COMMIT,
                "pre_fix_commit": PRE_FIX_COMMIT,
                "scope": SCOPE,
                "other_scope": OTHER_SCOPE,
                "review_node": REVIEW_NODE,
                "neighbour_node": NEIGHBOUR_NODE,
                "decoys": list(DECOYS),
                "jargon_query": JARGON_QUERY,
                "anchor_query": ANCHOR_QUERY,
                "invisible_query": INVISIBLE_QUERY,
                "invisible_anchor": INVISIBLE_ANCHOR,
                "cold_start_matrix": [list(item) for item in COLD_START_MATRIX],
                "unrelated_anchor_queries": list(UNRELATED_ANCHOR_QUERIES),
                "max_results": MAX_RESULTS,
            }
        ),
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    assert completed.returncode == 0, f"probe failed:\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    measured = run_probe()
    if not measured.get("model_available"):
        pytest.skip(
            "sentence-transformers model is not in the local cache; these tests measure "
            "what the encoder does to the operator's jargon and are meaningless against "
            "the hash fallback"
        )
    return measured


# ---------------------------------------------------------------------------
# the premise
# ---------------------------------------------------------------------------


def test_encoder_puts_query_to_query_far_above_query_to_content(
    probe: dict[str, Any]
) -> None:
    """The measurement the whole mechanism rests on, re-measured every run.

    Without this the regression below could keep passing for the wrong reason:
    an anchor that matched because the encoder had drifted into calling
    everything similar would still return the target.
    """

    cosines = probe["cosines"]
    assert cosines["jargon_to_anchor"] > ANCHOR_MATCH_COSINE_THRESHOLD
    assert cosines["jargon_to_content"] < 0.3
    assert cosines["jargon_to_anchor"] > 4 * cosines["jargon_to_content"]
    # The second pair's node is below the vector channel's own floor, which is
    # what lets a later test say "no other channel found it" literally.
    assert cosines["invisible_to_content"] < VECTOR_MATCH_THRESHOLD
    assert cosines["invisible_to_anchor"] > ANCHOR_MATCH_COSINE_THRESHOLD


# ---------------------------------------------------------------------------
# the falsifiable regression
# ---------------------------------------------------------------------------


def test_jargon_query_reaches_anchor_only_node(probe: dict[str, Any]) -> None:
    """Outside the top 5 on the pre-anchor code, inside it with anchors."""

    jargon = probe["jargon"]
    target = jargon["target"]

    assert jargon["without_matches_reference"], (
        "the anchors-off arm no longer reproduces the pre-anchor implementation, "
        "so 'fails on the old code' is not what is being demonstrated"
    )
    assert target not in jargon["before_ids"][:TOP_K], (
        f"the pre-anchor code already had the target in its top {TOP_K}: "
        f"{jargon['before_ids'][:TOP_K]}"
    )
    assert target in jargon["with_ids"][:TOP_K], (
        f"target missing from the anchored top {TOP_K}: {jargon['with_ids'][:TOP_K]}"
    )
    # Measured 8 -> 5 against seven same-language decoys. Stated as a rank move
    # rather than only a membership flip so a failure says how far it slipped,
    # and so a version that lands the target at 5 by accident rather than by
    # lift cannot pass.
    before_rank = jargon["before_ids"].index(target) + 1 if target in jargon["before_ids"] else None
    after_rank = jargon["with_ids"].index(target) + 1
    assert before_rank is None or after_rank < before_rank, (
        f"anchors did not lift the target: {before_rank} -> {after_rank}"
    )


def test_anchor_only_node_arrives_through_the_graph_channel(
    probe: dict[str, Any]
) -> None:
    """Scored as graph evidence, not as a fifth channel with its own field."""

    jargon = probe["jargon"]
    assert "graph" in jargon["target_methods"], jargon["target_methods"]
    assert "trigger" not in jargon["target_methods"], "anchors must not mint trigger score"
    assert jargon["target_bm25_score"] == 0.0, "BM25 was never going to find this node"
    # The lift is the graph channel's, several times what the vector channel
    # could see on its own — which is the whole claim.
    assert jargon["target_graph_score"] > 4 * jargon["target_vector_score"]
    assert jargon["target_graph_score"] <= 1.0


def test_anchor_seed_opens_one_hop(probe: dict[str, Any]) -> None:
    """The hop the mechanism is named for: one edge past the anchor's target."""

    jargon = probe["jargon"]
    assert jargon["neighbour_graph_score"] is not None, "the anchor seed opened no hop"
    assert 0.0 < jargon["neighbour_graph_score"] < jargon["target_graph_score"]
    assert "graph" in jargon["neighbour_methods"]


def test_anchor_seed_opens_the_graph_with_no_other_candidate(
    probe: dict[str, Any]
) -> None:
    """The blind spot itself: BFS used to need another channel to seed it.

    The store holds one node the query matches on no channel at all — so with
    anchors off the recall is empty, and the old guard (``candidates`` alone)
    could never have run the graph.
    """

    blind = probe["blind_spot"]
    assert blind["without_ids"] == []
    assert blind["with_ids"] == [blind["target"]]
    scores = blind["scores"]
    assert scores["graph"] > 0.0
    assert scores["bm25"] == 0.0
    assert scores["vector"] == 0.0
    assert scores["trigger"] == 0.0
    assert scores["methods"] == ["graph"]


def test_anchor_activation_is_cosine_times_edge_weight(probe: dict[str, Any]) -> None:
    """The seed carries the anchor's confidence, not a constant.

    One grounded consumption seeds at ``cosine x ANCHOR_EDGE_WEIGHT``; four
    saturate the edge and seed at the cosine itself. Both factors stay in
    [0, 1], which is what keeps an anchor seed inside the range the graph
    channel's existing seeds occupy instead of inflating it.
    """

    activation = probe["activation"]
    similarity = activation["similarity"]
    assert activation["edge_weight"] == pytest.approx(1.0)
    assert activation["edge_hits"] == 4
    assert activation["once"] == pytest.approx(similarity * ANCHOR_EDGE_WEIGHT, abs=1e-6)
    assert activation["saturated"] == pytest.approx(similarity, abs=1e-6)
    assert activation["saturated"] > activation["once"]
    assert 0.0 < activation["saturated"] <= 1.0


# ---------------------------------------------------------------------------
# an anchor never demotes the node it seeds
# ---------------------------------------------------------------------------
#
# Three invariants, on one fixture that provokes the defect on the pre-fix
# code: the anchored node is a strong vector candidate the anchor also names,
# under live-shaped weights (bm25 ~0, graph a few percent) where lifting the
# graph weight to its 0.25 floor is a ~23-point tax on the vector channel.


def test_the_fixture_still_provokes_the_demotion_on_the_pre_fix_code(
    probe: dict[str, Any]
) -> None:
    """Guard: without this the monotonicity test could pass on a flat fixture.

    On :data:`PRE_FIX_COMMIT` the anchor hands its own target a small graph
    score, the floor moves weight off the vector evidence carrying it, and the
    node the anchor was seeding loses score and rank.
    """

    monotone = probe["monotonicity"]
    seeded = monotone["seeded"]
    # The floor is live: the learned graph weight is below the 0.25 it is
    # lifted to, so crossing graph_score > 0 really does re-weight the blend.
    assert monotone["weights"]["graph"] < 0.25

    before = monotone["old_off"][seeded]
    after = monotone["old_on"][seeded]
    # The anchor gave it graph evidence...
    assert before["graph"] == 0.0
    assert after["graph"] > 0.0
    # ...and that cost it score, and a rank.
    assert after["score"] < before["score"]
    assert (before["score"] - after["score"]) / before["score"] > 0.05
    assert monotone["old_on_ids"].index(seeded) > monotone["old_off_ids"].index(seeded)


def test_an_anchor_seed_never_lowers_a_candidates_score(probe: dict[str, Any]) -> None:
    """MONOTONICITY. Anchors on scores no candidate below anchors off."""

    monotone = probe["monotonicity"]
    off = monotone["new_off"]
    on = monotone["new_on"]

    # Nothing the anchors-off arm returned may go missing, which is how the
    # two items of ``result.md`` section 6 were lost.
    assert set(off) <= set(on)
    for node_id, before in off.items():
        assert on[node_id]["score"] >= before["score"], node_id

    # And specifically the node the anchor seeded, which is the one that lost.
    seeded = monotone["seeded"]
    assert on[seeded]["score"] >= off[seeded]["score"]
    assert monotone["new_on_ids"].index(seeded) <= monotone["new_off_ids"].index(seeded)


def test_anchors_off_is_byte_identical_with_a_matching_anchor_present(
    probe: dict[str, Any]
) -> None:
    """BASELINE UNTOUCHED, on the corpus where anchors actually fire.

    The other byte-identity tests switch anchors off where no anchor matches.
    Here one does, and the anchors-off arm must still be exactly the pre-fix
    and pre-anchor ranking — that is what keeps the published 0.5769 / 0.3540
    baseline a valid comparison for the anchors-on arm.
    """

    monotone = probe["monotonicity"]
    assert monotone["new_off_matches_pre_fix"]
    assert monotone["new_off_matches_pre_anchor"]


def test_a_traversal_graph_score_still_triggers_the_floor(
    probe: dict[str, Any]
) -> None:
    """BASELINE UNTOUCHED, part two: only *anchor*-derived activation changed.

    The node at the far end of a real edge is scored through the floored blend
    in every arm, identically. A fix that took the better of the floored and
    unfloored blends for every graph candidate — rather than only where an
    anchor moved the activation — would raise this node's score here and
    quietly retune the floor for the whole corpus.
    """

    monotone = probe["monotonicity"]
    traversed = monotone["traversed"]
    for arm in ("old_off", "old_on", "new_off", "new_on"):
        assert monotone[arm][traversed]["graph"] > 0.0, arm
        assert "graph" in monotone[arm][traversed]["methods"], arm
    baseline = monotone["old_off"][traversed]["score"]
    assert monotone["new_off"][traversed]["score"] == baseline
    assert monotone["new_on"][traversed]["score"] == baseline
    assert monotone["old_on"][traversed]["score"] == baseline


# ---------------------------------------------------------------------------
# cold start is free
# ---------------------------------------------------------------------------


def test_cold_start_ranking_is_byte_identical_to_pre_anchor_code(
    probe: dict[str, Any]
) -> None:
    """No anchors: the pre-anchor implementation and this one agree exactly."""

    cold = probe["cold_start"]
    assert cold["anchor_count"] == 0
    assert cold["comparisons"] == len(COLD_START_MATRIX) * 2
    assert cold["divergences"] == []


def test_unresembled_query_class_ranks_identically_to_pre_anchor_code(
    probe: dict[str, Any]
) -> None:
    """Anchors present, none close enough: still the pre-anchor ranking.

    The steady state for most queries against a real anchor corpus, and the
    half of "cold start is free" that an empty table cannot demonstrate.
    """

    unresembled = probe["unresembled"]
    assert unresembled["anchor_count"] == len(UNRELATED_ANCHOR_QUERIES)
    assert unresembled["comparisons"] == len(COLD_START_MATRIX)
    assert unresembled["divergences"] == []


def test_empty_anchor_corpus_runs_no_anchor_scan(probe: dict[str, Any]) -> None:
    """Cold start is free in cost, not only in output.

    With no live anchor the read path stops at the indexed revision probe.
    Pulling a corpus of zero BLOBs would be cheap, but doing it on every recall
    of a database that will never have anchors is the kind of cost only ever
    noticed in aggregate. With anchors, the scan happens once and the cache
    serves the rest — until a new anchor moves the revision.
    """

    counts = probe["scan_counts"]
    assert counts["empty_corpus"] == 0
    assert counts["first_recall"] == 1
    assert counts["second_recall"] == 1
    assert counts["after_new_anchor"] == 1
    assert counts["new_anchor_target_found"] is True


def test_graph_off_skips_anchors_entirely(probe: dict[str, Any]) -> None:
    """``depth=0`` turns the graph off, and anchors are an entry into the graph."""

    graph_off = probe["graph_off"]
    assert graph_off["depth_0"] == []
    assert graph_off["depth_1"] == [graph_off["target"]]


# ---------------------------------------------------------------------------
# an anchor is never an answer
# ---------------------------------------------------------------------------


def test_anchor_never_appears_in_a_response(probe: dict[str, Any]) -> None:
    """No anchor id and no anchor query text, anywhere a caller can read.

    Checked over the serialized results *and* the persisted recall event, and
    then at the source: an anchor is not a node, so there is nothing for FTS or
    the graph to return in the first place.
    """

    leak = probe["leak"]
    assert leak["target"] in leak["result_ids"]
    assert leak["anchor_id"] not in leak["result_ids"]
    assert leak["payload_has_anchor_id"] is False
    assert leak["payload_has_anchor_query"] is False
    assert leak["event_has_anchor_id"] is False
    assert leak["event_has_anchor_query"] is False
    assert all(leak["anchor_id"] not in path for path in leak["paths"])
    assert leak["anchor_is_a_node"] is False
    assert leak["nodes_carrying_anchor_text"] == 0
    assert leak["connections_touching_anchor"] == 0


# ---------------------------------------------------------------------------
# an anchor never crosses scope
# ---------------------------------------------------------------------------


def test_anchor_does_not_seed_outside_its_own_scope(probe: dict[str, Any]) -> None:
    """An anchor in a scope the plan does not admit contributes nothing.

    Both directions, so the negative result is the scope rule rather than a
    broken fixture: the out-of-plan anchor is invisible, and the same anchor
    written into the plan's own scope does seed.
    """

    isolation = probe["scope_isolation"]
    assert isolation["from_demo_with_other_scope_anchor"] == []
    # The project:other plan admits the anchor, but its target lives in
    # project:demo, which that plan does not admit — so still nothing.
    assert isolation["from_other_scope"] == []
    assert isolation["after_same_scope_anchor"] == [isolation["target"]]


def test_anchor_target_outside_the_plan_is_not_admitted(probe: dict[str, Any]) -> None:
    """A global anchor seeds a global plan, and only the nodes that plan allows."""

    plan_filter = probe["plan_filter"]
    assert plan_filter["ids"] == [plan_filter["in_plan"]]


def test_decayed_anchor_target_is_never_seeded(probe: dict[str, Any]) -> None:
    """A forgotten node stays forgotten, however strong the anchor edge."""

    decayed = probe["decayed_target"]
    assert decayed["before"] == [decayed["target"]]
    assert decayed["after"] == []


def test_decayed_anchor_stops_seeding_until_revived(probe: dict[str, Any]) -> None:
    """An anchor that aged out is not consulted until something revives it."""

    decayed = probe["decayed_anchor"]
    assert decayed["live"] == [decayed["target"]]
    assert decayed["decayed"] == []
    assert decayed["revived"] == [decayed["target"]]


# ---------------------------------------------------------------------------
# bounds that need no encoder
# ---------------------------------------------------------------------------


def test_anchor_seeds_are_bounded_by_the_graph_seed_limit() -> None:
    """An anchor with more edges than the seed block admits cannot outgrow it.

    Runs in-process with whatever backend is ambient: the bound is arithmetic,
    not semantic, and holding it here means the fetch it guards is bounded even
    when the encoder is a stub.
    """

    import tempfile

    from living_memory import query_anchors as qa
    from living_memory.embeddings import LocalEmbeddingModel
    from living_memory.retrieval import MemoryRecallService
    from living_memory.scope import ScopePlan
    from living_memory.storage import MemoryStore

    overflow = 12
    embedder = LocalEmbeddingModel()
    with tempfile.TemporaryDirectory(prefix="lm-anchor-bound-") as directory:
        with MemoryStore(Path(directory) / "memory.sqlite3") as store:
            # Ascending edge weights, so "kept the heaviest" is a statement the
            # assertion can actually distinguish from "kept the first fifty".
            targets = [
                (
                    store.append_trace(f"bounded target {index}", {"scope": SCOPE}).id,
                    (index + 1) / 100.0,
                )
                for index in range(GRAPH_SEED_LIMIT + overflow)
            ]
            query = "bounded anchor query"
            qa.upsert_anchor(store, query, SCOPE, embedder.embed(query), targets=targets)

            service = MemoryRecallService(store, embedder=embedder)
            plan = ScopePlan(requested_scope=SCOPE, scopes=(SCOPE, "global"))
            seeds = service._collect_anchor_seeds(plan, embedder.embed(query))

            assert len(seeds) == GRAPH_SEED_LIMIT
            assert all(value > 0.0 for value in seeds.values())
            assert set(seeds) == {node_id for node_id, _weight in targets[overflow:]}
