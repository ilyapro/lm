"""The harness's query-anchor buckets, and the self-reinforcement guard.

Three things are pinned here, all on fixture databases under
``LIVING_MEMORY_EMBEDDING_BACKEND=hash``:

* **fresh-node visibility** — a relevant node created after the matched anchor
  last learned anything, and not itself one of that anchor's targets, must stay
  reachable once the anchor seeds its older targets into the graph channel. The
  slice must not be worse with anchors than without. This is the gate against
  "the rich get richer": anchors, grounding and usefulness are one loop, and
  without this bucket the loop can quietly bury new memory while every headline
  metric improves.
* **exact repeat versus fingerprint-disjoint** — the two subsets are counted and
  scored apart. An anchor whose stored query is byte-identical to the eval query
  is a lookup table; the generalization claim lives in the disjoint subset,
  where a *different* past query has to be the bridge.
* **cold start** — with no anchor rows the annotation is empty and every anchor
  bucket reports zero items rather than vanishing or throwing.

The disjoint-and-matching case is constructible under the hash backend because
:func:`living_memory.query_anchors.upsert_anchor` takes the embedding as an
argument: the anchor is stored under a *different* query string carrying the
eval query's vector, which is exactly the mechanism under test — one past query
reaching a new one it does not share a fingerprint with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from living_memory import query_anchors as qa
from living_memory import retrieval_harness as harness
from living_memory.config import MemoryConfig
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore

SCOPE = "project:anchor-slice"

#: Before every node the fixture cares about, so "fresh" is unambiguous.
ANCHOR_STAMP = "2026-01-10T00:00:00Z"
OLD_NODE_STAMP = "2026-01-05T00:00:00Z"
FRESH_NODE_STAMP = "2026-03-01T00:00:00Z"

#: The remembered situation. The eval queries below are (a) this string exactly
#: and (b) a different string carrying this string's vector.
ANCHOR_QUERY = "поревьювь мердж-реквест 9443 по чеклисту"
REPEAT_QUERY = ANCHOR_QUERY
DISJOINT_QUERY = "глянь мердж-реквест 9443 по чеклисту"


def _set_node_created_at(store: MemoryStore, node_id: str, created_at: str) -> None:
    with store.connection as conn:
        conn.execute(
            "UPDATE nodes SET created_at = ?, updated_at = ? WHERE id = ?",
            (created_at, created_at, node_id),
        )


def _goldset_record(
    query_id: str,
    query: str,
    relevant: list[str],
    *,
    stratum: str = "content_grounded",
    max_results: int = 5,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query,
        "scope": SCOPE,
        "ambient_context": None,
        "depth": 1,
        "max_results": max_results,
        "relevant_node_ids": relevant,
        "stratum": stratum,
        "tail": False,
        "source_event_id": None,
        "provenance": {"origin": "anchor-slice-fixture"},
    }


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Old anchor targets, one fresh relevant node, and anchors over them."""

    root = tmp_path_factory.mktemp("anchor-slice")
    db_path = root / "source.sqlite3"
    store = MemoryStore(MemoryConfig(db_path=db_path))

    # Eight old nodes the anchor points at. Eight rather than one so the guard
    # is a real crowding test: they all become graph seeds competing for the
    # same five result slots the fresh node needs one of. Their vocabulary is
    # deliberately disjoint from the eval queries -- no shared token, no shared
    # digit -- so bm25 and the vector channel cannot reach them and the ONLY
    # way they enter the result list is the anchor seed.
    old_ids: list[str] = []
    for index in range(8):
        node = store.create_node(
            level="trace",
            content=(
                f"deployment runbook appendix {index}: rollback window, paging "
                f"policy and advisory checklist for the feed workers, part {index}"
            ),
            context={"scope": SCOPE},
        )
        _set_node_created_at(store, node.id, OLD_NODE_STAMP)
        old_ids.append(node.id)

    # The node the anchor cannot know about: created after the anchor's last
    # reinforcement, and never one of its targets.
    fresh = store.create_node(
        level="trace",
        content=(
            "мердж-реквест 9443 по чеклисту: свежий разбор, миграция чексуммы "
            "zr9042 откатывается перед мержем"
        ),
        context={"scope": SCOPE},
    )
    _set_node_created_at(store, fresh.id, FRESH_NODE_STAMP)

    snapshot_before = root / "no-anchors.sqlite3"
    store.close()
    harness.create_snapshot(db_path, snapshot_before)

    store = MemoryStore(MemoryConfig(db_path=db_path))
    embedder = LocalEmbeddingModel(model_name=store.config.embedding_model)

    # (1) the exact-repeat anchor: stored under the eval query's own text, so
    #     its fingerprint is the eval query's fingerprint.
    repeat = qa.upsert_anchor(
        store, ANCHOR_QUERY, SCOPE, embedder.embed(ANCHOR_QUERY), old_ids, ANCHOR_STAMP
    )
    # (2) the fingerprint-disjoint anchor: a *different* remembered question
    #     carrying the disjoint eval query's vector. Different fingerprint, and
    #     the vector match still fires -- one past query bridging to another.
    disjoint = qa.upsert_anchor(
        store,
        "мердж-реквест 9443 чеклист ревью",
        SCOPE,
        embedder.embed(DISJOINT_QUERY),
        old_ids,
        ANCHOR_STAMP,
    )
    assert repeat.created and disjoint.created
    assert repeat.anchor.id != disjoint.anchor.id

    snapshot_after = root / "with-anchors.sqlite3"
    store.close()
    harness.create_snapshot(db_path, snapshot_after)

    goldset = root / "goldset.jsonl"
    items = [
        # Fresh relevant node, anchor points only at older ones -> in the slice.
        _goldset_record("content_grounded-repeat", REPEAT_QUERY, [fresh.id]),
        _goldset_record(
            "content_grounded-disjoint",
            DISJOINT_QUERY,
            [fresh.id],
            stratum="cross_lingual",
        ),
        # Same remembered query, but the relevant node IS an anchor target ->
        # matched, and never in the slice: that is a lookup, not visibility of
        # something the anchor could not know about.
        _goldset_record("content_grounded-target", REPEAT_QUERY, [old_ids[0]]),
    ]
    goldset.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in items)
        + "\n",
        encoding="utf-8",
    )

    return {
        "root": root,
        "db_path": db_path,
        "snapshot_before": snapshot_before,
        "snapshot_after": snapshot_after,
        "goldset": goldset,
        "old_ids": old_ids,
        "fresh_id": fresh.id,
        "repeat_anchor_id": repeat.anchor.id,
        "disjoint_anchor_id": disjoint.anchor.id,
    }


def _annotations(db_path: Path, goldset: Path) -> dict[str, harness.AnchorAnnotation]:
    items = harness.load_goldset(goldset)
    with harness.working_copy(db_path) as working:
        store = MemoryStore(MemoryConfig(db_path=working))
        try:
            service = MemoryRecallService(store)
            return harness.annotate_anchors(service, items)
        finally:
            store.close()


def _run(snapshot: Path, goldset: Path, report: Path, *, anchors: bool) -> dict[str, Any]:
    argv = [
        "run",
        "--snapshot",
        str(snapshot),
        "--goldset",
        str(goldset),
        "--report",
        str(report),
        "--agreement-sample",
        "0",
    ]
    if not anchors:
        argv.append("--no-anchors")
    assert harness.main(argv) == 0
    return json.loads(report.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def test_annotation_splits_exact_repeat_from_fingerprint_disjoint(
    corpus: dict[str, Any],
) -> None:
    annotations = _annotations(corpus["snapshot_after"], corpus["goldset"])

    repeat = annotations["content_grounded-repeat"]
    assert repeat.exact_repeat is True
    assert repeat.fingerprint_anchor_id == corpus["repeat_anchor_id"]

    disjoint = annotations["content_grounded-disjoint"]
    # The whole point: a different past question reaches this query. No shared
    # fingerprint, and the vector match still fires.
    assert disjoint.exact_repeat is False
    assert disjoint.fingerprint_anchor_id is None
    assert disjoint.matched is True
    assert disjoint.best_similarity is not None
    assert disjoint.best_similarity >= qa.ANCHOR_MATCH_COSINE_THRESHOLD


def test_fresh_relevant_node_is_the_one_the_anchor_cannot_know_about(
    corpus: dict[str, Any],
) -> None:
    annotations = _annotations(corpus["snapshot_after"], corpus["goldset"])

    fresh_item = annotations["content_grounded-repeat"]
    assert fresh_item.anchor_as_of == ANCHOR_STAMP
    assert set(fresh_item.target_ids) == set(corpus["old_ids"])
    assert fresh_item.fresh_relevant_ids == (corpus["fresh_id"],)
    assert fresh_item.in_fresh_slice is True

    # A relevant node that IS the anchor's target is a lookup, not visibility of
    # something new: matched anchor, empty fresh set, out of the slice.
    target_item = annotations["content_grounded-target"]
    assert target_item.matched is True
    assert target_item.fresh_relevant_ids == ()
    assert target_item.in_fresh_slice is False


def test_annotation_is_empty_without_anchor_rows(corpus: dict[str, Any]) -> None:
    assert _annotations(corpus["snapshot_before"], corpus["goldset"]) == {}


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------


def test_fresh_node_slice_scores_only_the_fresh_relevant_nodes(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    report = _run(
        corpus["snapshot_after"],
        corpus["goldset"],
        tmp_path / "with.json",
        anchors=True,
    )
    metrics = report["metrics"]
    slice_block = metrics["fresh_node_visibility"]

    # Two of the three items qualify; the anchor-target item does not.
    assert slice_block["items"] == 2
    assert slice_block["fresh_relevant_nodes"] == 2
    assert slice_block["events_with_useful"] == 2
    # One useful node per event -- the fresh one -- not the full relevant set.
    assert slice_block["useful_results"] == 2

    holdout = metrics["anchor_holdout"]
    assert holdout["annotated_items"] == 3
    assert holdout["exact_repeat"]["events"] == 2
    assert holdout["fingerprint_disjoint"]["events"] == 1

    coverage = metrics["anchor_coverage"]
    assert coverage["items"] == 3
    assert coverage["matched"] == 3
    assert coverage["matched_with_live_targets"] == 3
    assert coverage["fresh_slice_items"] == 2
    assert coverage["match_floor"] == qa.ANCHOR_MATCH_COSINE_THRESHOLD
    assert coverage["nearest_similarity"]["max"] == pytest.approx(1.0, abs=1e-6)


def test_cold_start_reports_empty_anchor_buckets_rather_than_dropping_them(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    report = _run(
        corpus["snapshot_before"],
        corpus["goldset"],
        tmp_path / "cold.json",
        anchors=True,
    )
    metrics = report["metrics"]
    assert metrics["anchor_coverage"]["items"] == 0
    assert metrics["fresh_node_visibility"]["items"] == 0
    assert metrics["anchor_holdout"]["annotated_items"] == 0
    assert report["anchor_annotations"] == []
    assert report["provenance"]["anchor_corpus"]["live_anchors"] == 0


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


def test_anchors_do_not_bury_the_fresh_node(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    """The self-reinforcement gate: same snapshot, same goldset, one switch."""

    with_anchors = _run(
        corpus["snapshot_after"],
        corpus["goldset"],
        tmp_path / "with.json",
        anchors=True,
    )
    without_anchors = _run(
        corpus["snapshot_after"],
        corpus["goldset"],
        tmp_path / "without.json",
        anchors=False,
    )

    assert with_anchors["provenance"]["anchor_seeding"] is True
    assert without_anchors["provenance"]["anchor_seeding"] is False
    # Both arms read the same corpus, so bucket membership is identical and the
    # comparison is like for like. Without this the two arms could be scoring
    # different item sets.
    assert with_anchors["anchor_annotations"] == without_anchors["anchor_annotations"]

    on = with_anchors["metrics"]["fresh_node_visibility"]
    off = without_anchors["metrics"]["fresh_node_visibility"]
    assert on["events_with_useful"] == off["events_with_useful"] == 2
    for metric in ("hit@1", "hit@5", "hit@10", "mrr"):
        assert on[metric] >= off[metric], (metric, on[metric], off[metric])
    # And the fresh node is genuinely reachable, not merely equally unreachable.
    assert on["hit@5"] > 0.0

    # The anchors really did put their older targets into the running -- the
    # crowding the slice guards against is present, not hypothetical. Those
    # nodes share no token and no digit with the query, so nothing but the
    # anchor seed can have produced them.
    targets = set(corpus["old_ids"])
    seeded = {
        run["query_id"]: len(targets.intersection(run["ranked_node_ids"]))
        for run in with_anchors["runs"]
    }
    unseeded = {
        run["query_id"]: len(targets.intersection(run["ranked_node_ids"]))
        for run in without_anchors["runs"]
    }
    assert unseeded == {key: 0 for key in unseeded}
    assert seeded["content_grounded-repeat"] > 0
    assert seeded["content_grounded-disjoint"] > 0


def test_slice_reports_a_buried_fresh_node_the_overall_bucket_hides() -> None:
    """Why the slice has to score the fresh node *alone*.

    Constructed ranking: the anchor's own old target is rank 1 and the fresh
    relevant node is rank 7. The overall bucket sees a hit@5 of 1.0 -- the item
    "found something relevant" -- while the fresh node is nowhere near the top
    five. Only a bucket whose useful set is the fresh node alone can report
    that, which is exactly the failure "the rich get richer" produces.
    """

    item = harness.GoldsetItem(
        query_id="content_grounded-buried",
        query=ANCHOR_QUERY,
        scope=SCOPE,
        ambient_context=None,
        depth=1,
        max_results=10,
        relevant_node_ids=("old-1", "fresh-1"),
        stratum="content_grounded",
        tail=False,
        source_event_id=None,
        provenance={},
    )
    ranked = ["old-1", *[f"filler-{index}" for index in range(5)], "fresh-1"]
    run = harness.HarnessRun(
        item=item,
        results=tuple(
            harness.HarnessResult(
                node_id=node_id,
                rank=index + 1,
                level="trace",
                scope=SCOPE,
                score=1.0 - index / 100,
                bm25_score=0.0,
                vector_score=0.0,
                graph_score=1.0 - index / 100,
                trigger_score=0.0,
                methods=("graph",),
            )
            for index, node_id in enumerate(ranked)
        ),
    )
    annotation = harness.AnchorAnnotation(
        query_id=item.query_id,
        stratum=item.stratum,
        scopes=(SCOPE,),
        fingerprint="fp",
        exact_repeat=True,
        fingerprint_anchor_id="anchor-1",
        matched_anchor_ids=("anchor-1",),
        best_similarity=1.0,
        nearest_similarity=1.0,
        target_ids=("old-1",),
        anchor_as_of=ANCHOR_STAMP,
        fresh_relevant_ids=("fresh-1",),
    )

    metrics = harness.compute_metrics([run], {item.query_id: annotation})
    assert metrics["overall"]["hit@5"] == 1.0
    assert metrics["overall"]["hit@1"] == 1.0
    slice_block = metrics["fresh_node_visibility"]
    assert slice_block["items"] == 1
    assert slice_block["events_with_useful"] == 1
    assert slice_block["hit@5"] == 0.0
    assert slice_block["hit@10"] == 1.0
    assert slice_block["mrr"] == pytest.approx(1 / 7)


def test_anchor_seeding_switch_matches_an_unpopulated_corpus(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    """``--no-anchors`` must equal "there are no anchors", not approximate it.

    Otherwise the anchor-free arm of the holdout run would be its own
    configuration rather than today's shipped behaviour.
    """

    ablated = _run(
        corpus["snapshot_after"],
        corpus["goldset"],
        tmp_path / "ablated.json",
        anchors=False,
    )
    unpopulated = _run(
        corpus["snapshot_before"],
        corpus["goldset"],
        tmp_path / "unpopulated.json",
        anchors=True,
    )
    assert [run["ranked_node_ids"] for run in ablated["runs"]] == [
        run["ranked_node_ids"] for run in unpopulated["runs"]
    ]
    for key in ("overall", "per_stratum", "tail", "per_channel"):
        assert ablated["metrics"][key] == unpopulated["metrics"][key]
