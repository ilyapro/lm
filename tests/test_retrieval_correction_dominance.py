"""Correction-dominance ordering invariant in ``rank_candidates``.

Whenever a superseded node and its superseding correction both qualify for
one recall, the correction must rank strictly above the stale node —
regardless of feedback multipliers, in every mode (causal/decision
included). The 0.2x/1.2x correction multipliers in
``feedback_weighted_score`` remain the soft prior; the structural pass at
the end of ``rank_candidates`` is what these tests prove. Superseded nodes
are never hard-filtered: with the correction absent or decayed they keep
surfacing at their scored position, flagged via ``RecallResult.superseded``.

Replay honesty note: in the recorded animal-planet corpus every supersedes
edge post-dates the recorded recalls, so no recorded event carries both
endpoints of a pair — ``ap_baseline.py compare --metrics
correction_dominance`` reports pairs=0 on both dev and eval (see
artifacts/animal-planet/failures/cases/ap-correction-ordering-*.json, which
encode the at-risk surface, not observed inversions). Zero replay
violations is therefore near-vacuous evidence; the seeded-random property
sweep and the deterministic at-risk regressions below are the real proof.

The property sweep runs through ``replay.make_ranking_service`` on purpose:
it proves the invariant on the exact code path the replay harness measures,
and its shim store raises on any SQL beyond the pinned supersedes query, so
a regression that adds store calls to ``rank_candidates`` fails here too.
"""

from __future__ import annotations

import random
from pathlib import Path

from living_memory.models import Node, RetrievalWeights
from living_memory.replay import make_ranking_service
from living_memory.retrieval import MemoryRecallService, RecallResult, _Candidate
from living_memory.scope import ScopePlan
from living_memory.storage import MemoryStore

PROJECT_SCOPE = "project:app"
PROJECT_PLAN = ScopePlan(requested_scope=PROJECT_SCOPE, scopes=(PROJECT_SCOPE, "global"))

SEED = 20260814
SCENARIOS = 350


def _service(pairs, weights=(0.5, 0.5, 0.0)):
    bm25, vector, graph = weights

    def resolver(scope: str) -> RetrievalWeights:
        return RetrievalWeights(scope=scope, bm25=bm25, vector=vector, graph=graph)

    return make_ranking_service(resolver, pairs)


def _node(node_id: str, scope: str = PROJECT_SCOPE, **stats) -> Node:
    return Node(id=node_id, level="trace", content=f"content of {node_id}", scope=scope, **stats)


def _rank(pairs, candidates, plan=PROJECT_PLAN, weights=(0.5, 0.5, 0.0), **modes):
    return _service(pairs, weights).rank_candidates(candidates, plan, **modes)


def _ids(results) -> list[str]:
    return [result.node.id for result in results]


# --- Deterministic regressions ----------------------------------------------


def _at_risk_candidates() -> dict[str, _Candidate]:
    """The ap-correction-ordering-* at-risk shape.

    A stale node whose compounded feedback multiplier sits at the 1.4
    FEEDBACK_MULTIPLIER_CAP and whose base evidence is far stronger than its
    correction's: even after the 0.2x superseded penalty it outscores the
    fresh correction at default stats, so only the structural pass can order
    them correctly.
    """

    return {
        "stale": _Candidate(
            node=_node("stale", confidence=1.0, usefulness_score=1.0, access_count=10_000),
            bm25_score=1.0,
            vector_score=0.64,
        ),
        "correction": _Candidate(node=_node("correction"), bm25_score=0.05, vector_score=0.25),
        "bystander": _Candidate(node=_node("bystander"), bm25_score=0.5, vector_score=0.5),
    }


def test_cap_saturated_stale_never_outranks_its_fresh_correction() -> None:
    results = _rank([("correction", "stale")], _at_risk_candidates())
    position = {nid: index for index, nid in enumerate(_ids(results))}
    assert position["correction"] < position["stale"]

    by_id = {result.node.id: result for result in results}
    # The soft prior alone loses this shape: the stale node still outscores
    # the correction, proving the ordering came from the structural pass.
    assert by_id["stale"].score > by_id["correction"].score
    assert by_id["stale"].superseded is True
    assert by_id["correction"].superseded is False
    assert by_id["bystander"].superseded is False


def test_promotion_is_minimal_bystanders_keep_their_positions() -> None:
    # The correction is lifted to sit directly above the stale node; the
    # stronger bystander stays at the top and the stale node is not demoted.
    results = _rank([("correction", "stale")], _at_risk_candidates())
    assert _ids(results) == ["bystander", "correction", "stale"]


def test_without_the_edge_the_same_shape_ranks_by_score() -> None:
    results = _rank([], _at_risk_candidates())
    assert _ids(results) == ["stale", "bystander", "correction"]
    assert all(result.superseded is False for result in results)


def _chain_candidates(length: int) -> tuple[dict[str, _Candidate], list[tuple[str, str]]]:
    """v1..vN where each v(i+1) supersedes v(i); v1 carries the entrenched
    feedback and the strongest evidence, later corrections progressively
    weaker (the realistic teach-chain shape)."""

    candidates: dict[str, _Candidate] = {
        "top_bystander": _Candidate(node=_node("top_bystander"), bm25_score=1.0, vector_score=0.62),
        "tail_bystander": _Candidate(node=_node("tail_bystander"), bm25_score=0.05, vector_score=0.09),
    }
    pairs: list[tuple[str, str]] = []
    for index in range(length):
        stats = (
            {"confidence": 1.0, "usefulness_score": 1.0, "access_count": 10_000}
            if index == 0
            else {}
        )
        candidates[f"v{index + 1}"] = _Candidate(
            node=_node(f"v{index + 1}", **stats),
            bm25_score=0.9 - 0.2 * index,
            vector_score=0.5 - 0.1 * index,
        )
        if index:
            pairs.append((f"v{index + 1}", f"v{index}"))
    return candidates, pairs


def test_chain_of_four_lifts_transitively_to_a_fixpoint() -> None:
    candidates, pairs = _chain_candidates(4)
    results = _rank(pairs, candidates)
    position = {nid: index for index, nid in enumerate(_ids(results))}
    assert position["v4"] < position["v3"] < position["v2"] < position["v1"]
    # The whole chain lands where its best-ranked (stale) member sat;
    # uninvolved results keep their relative order around it.
    assert _ids(results) == ["top_bystander", "v4", "v3", "v2", "v1", "tail_bystander"]
    flags = {result.node.id: result.superseded for result in results}
    assert flags == {
        "top_bystander": False,
        "v4": False,
        "v3": True,
        "v2": True,
        "v1": True,
        "tail_bystander": False,
    }


def test_diamond_topology_orders_every_edge() -> None:
    candidates = {
        "bottom": _Candidate(
            node=_node("bottom", confidence=1.0, usefulness_score=1.0, access_count=5_000),
            bm25_score=1.0,
            vector_score=0.6,
        ),
        "mid_a": _Candidate(node=_node("mid_a"), bm25_score=0.5, vector_score=0.3),
        "mid_b": _Candidate(node=_node("mid_b"), bm25_score=0.4, vector_score=0.25),
        "top": _Candidate(node=_node("top"), bm25_score=0.1, vector_score=0.12),
    }
    pairs = [("mid_a", "bottom"), ("mid_b", "bottom"), ("top", "mid_a"), ("top", "mid_b")]
    results = _rank(pairs, candidates)
    position = {nid: index for index, nid in enumerate(_ids(results))}
    assert position["top"] < position["mid_a"] < position["bottom"]
    assert position["top"] < position["mid_b"] < position["bottom"]


def test_two_node_supersedes_cycle_keeps_score_order_and_flags_both() -> None:
    candidates = {
        "x": _Candidate(node=_node("x"), bm25_score=1.0),
        "y": _Candidate(node=_node("y"), bm25_score=0.5),
    }
    pairs = [("x", "y"), ("y", "x")]
    first = _rank(pairs, candidates)
    second = _rank(pairs, candidates)
    # Mutually contradictory constraints cannot both hold: ranking must
    # terminate, stay deterministic, and fall back to score order.
    assert _ids(first) == ["x", "y"]
    assert _ids(first) == _ids(second)
    assert all(result.superseded is True for result in first)


def test_three_node_supersedes_cycle_terminates_deterministically() -> None:
    candidates = {
        "x": _Candidate(node=_node("x"), bm25_score=1.0),
        "y": _Candidate(node=_node("y"), bm25_score=0.6),
        "z": _Candidate(node=_node("z"), bm25_score=0.3),
    }
    pairs = [("x", "y"), ("y", "z"), ("z", "x")]
    first = _rank(pairs, candidates)
    second = _rank(pairs, candidates)
    assert set(_ids(first)) == {"x", "y", "z"}
    assert _ids(first) == _ids(second)


def test_self_supersedes_edge_is_inert_for_ordering() -> None:
    candidates = {
        "x": _Candidate(node=_node("x"), bm25_score=0.4),
        "y": _Candidate(node=_node("y"), bm25_score=1.0),
    }
    results = _rank([("x", "x")], candidates)
    assert _ids(results) == ["y", "x"]
    flags = {result.node.id: result.superseded for result in results}
    assert flags == {"y": False, "x": True}


def test_absent_correction_leaves_stale_surfacing_flagged() -> None:
    candidates = _at_risk_candidates()
    del candidates["correction"]
    results = _rank([("correction", "stale")], candidates)
    # No hard filter: the stale node still surfaces at its scored position
    # (the 0.2x soft-prior penalty still applies), carrying the flag.
    assert _ids(results) == ["bystander", "stale"]
    assert results[-1].superseded is True


def test_decayed_correction_leaves_stale_surfacing_flagged() -> None:
    candidates = _at_risk_candidates()
    candidates["correction"] = _Candidate(
        node=Node(
            id="correction",
            level="trace",
            content="content of correction",
            scope=PROJECT_SCOPE,
            decayed=True,
        ),
        bm25_score=0.05,
        vector_score=0.25,
    )
    results = _rank([("correction", "stale")], candidates)
    # The decayed correction never reaches ranking; the stale node surfaces
    # flagged at its soft-prior-penalized position instead of being dropped.
    assert _ids(results) == ["bystander", "stale"]
    assert results[-1].superseded is True


def test_causal_mode_still_enforces_dominance() -> None:
    candidates = _at_risk_candidates()
    # Causal mode multiplies graph-connected results by 1.5x on top of the
    # feedback multiplier — the stale node gets every boost available.
    candidates["stale"].graph_score = 1.2
    results = _rank([("correction", "stale")], candidates, causal_mode=True)
    position = {nid: index for index, nid in enumerate(_ids(results))}
    assert position["correction"] < position["stale"]


def test_decision_mode_still_enforces_dominance() -> None:
    results = _rank([("correction", "stale")], _at_risk_candidates(), decision_mode=True)
    position = {nid: index for index, nid in enumerate(_ids(results))}
    assert position["correction"] < position["stale"]


def test_recall_result_superseded_is_additive_with_false_default() -> None:
    result = RecallResult(node=_node("n1"), score=1.0)
    assert result.superseded is False
    payload = result.to_dict()
    assert payload["superseded"] is False
    # Additive only: every pre-existing key is still emitted (MCP contract).
    assert set(payload) == {
        "node",
        "score",
        "bm25_score",
        "vector_score",
        "graph_score",
        "trigger_score",
        "methods",
        "path",
        "recall_event_id",
        "superseded",
    }
    flagged = RecallResult(node=_node("n1"), score=1.0, superseded=True)
    assert flagged.to_dict()["superseded"] is True


# --- Seeded-random property sweep -------------------------------------------


def _random_plan(rng: random.Random) -> ScopePlan:
    project = f"project:p{rng.randrange(3)}"
    kind = rng.randrange(4)
    if kind == 0:
        return ScopePlan(requested_scope=project, scopes=(project, "global"))
    if kind == 1:
        session = f"session:s{rng.randrange(2)}"
        return ScopePlan(requested_scope=session, scopes=(session, project, "global"))
    if kind == 2:
        return ScopePlan(requested_scope="global", scopes=("global",))
    return ScopePlan(requested_scope="global", scopes=(project, "global"))


def _random_pairs(rng: random.Random, ids: list[str]) -> list[tuple[str, str]]:
    """Random supersedes topology: pairs, chains (length <=4), diamonds,
    cycles, absent corrections, self-loops. Tuples are (correction, stale)
    matching the connections table's (source_id, target_id)."""

    pairs: list[tuple[str, str]] = []
    pool = list(ids)
    rng.shuffle(pool)

    def take(count: int) -> list[str]:
        taken = pool[:count]
        del pool[:count]
        return taken

    shape = rng.randrange(6)
    if shape == 1:
        for _ in range(rng.randint(1, 3)):
            if len(pool) >= 2:
                stale, correction = take(2)
                pairs.append((correction, stale))
    elif shape == 2:
        chain = take(rng.randint(2, 4))
        for older, newer in zip(chain, chain[1:], strict=False):
            pairs.append((newer, older))
    elif shape == 3 and len(pool) >= 4:
        bottom, mid_a, mid_b, top = take(4)
        pairs += [(mid_a, bottom), (mid_b, bottom), (top, mid_a), (top, mid_b)]
    elif shape == 4:
        cycle = take(rng.randint(2, 3))
        if len(cycle) >= 2:
            for current, following in zip(cycle, cycle[1:] + cycle[:1], strict=False):
                pairs.append((current, following))
    elif shape == 5:
        chain = take(3)
        for older, newer in zip(chain, chain[1:], strict=False):
            pairs.append((newer, older))
        if len(pool) >= 2:
            stale, correction = take(2)
            pairs.append((correction, stale))
        if rng.random() < 0.5 and len(pool) >= 2:
            first, second = take(2)
            pairs += [(first, second), (second, first)]
    if ids and rng.random() < 0.35:
        pairs.append((f"phantom-{rng.randrange(100)}", rng.choice(ids)))
    if ids and rng.random() < 0.10:
        loop = rng.choice(ids)
        pairs.append((loop, loop))
    return pairs


def _random_candidate(
    rng: random.Random,
    node_id: str,
    plan: ScopePlan,
    role: str,
) -> _Candidate:
    scope = "project:outside-plan" if rng.random() < 0.05 else rng.choice(plan.scopes)
    level = "schema" if rng.random() < 0.10 else "trace"
    context = {"is_rejected_alternative": True} if rng.random() < 0.08 else {}
    decayed = rng.random() < (0.20 if role == "correction" else 0.08)
    if role == "stale" and rng.random() < 0.6:
        # Adversarial: entrenched feedback at the 1.4 cap plus dominant
        # evidence, the shape the soft prior loses.
        stats = {"confidence": 1.0, "usefulness_score": 1.0, "access_count": 10_000}
        scores = {"bm25_score": 1.0, "vector_score": rng.choice([0.5, 0.64, 0.9])}
    elif role == "correction" and rng.random() < 0.6:
        stats = {}
        scores = {"bm25_score": rng.choice([0.0, 0.05, 0.2]), "vector_score": rng.uniform(0.08, 0.3)}
    else:
        stats = {
            "confidence": rng.choice([0.0, 0.3, 0.5, 0.8, 1.0]),
            "usefulness_score": rng.choice([-1.0, -0.4, 0.0, 0.3, 0.8, 1.0, 1.0]),
            "access_count": rng.choice([0, 1, 5, 40, 600, 10_000]),
        }
        scores = {}
        if rng.random() < 0.75:
            scores["bm25_score"] = rng.choice([1.0, 0.5, 0.33, 0.25, 0.1])
        if rng.random() < 0.8:
            scores["vector_score"] = round(rng.uniform(0.08, 1.0), 3)
        if rng.random() < 0.35:
            scores["graph_score"] = round(rng.uniform(0.05, 1.5), 3)
    if level == "schema" and rng.random() < 0.5:
        scores["trigger_score"] = round(0.95 + 0.05 * rng.random(), 4)
    node = Node(
        id=node_id,
        level=level,  # type: ignore[arg-type]
        content=f"content of {node_id}",
        scope=scope,
        decayed=decayed,
        context=context,
        **stats,
    )
    return _Candidate(node=node, **scores)


def _cycle_edges(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Edges participating in any cycle: (c, s) such that s reaches c."""

    adjacency: dict[str, set[str]] = {}
    for correction, stale in edges:
        adjacency.setdefault(correction, set()).add(stale)

    def reaches(start: str, goal: str) -> bool:
        seen: set[str] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            if current == goal:
                return True
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(adjacency.get(current, ()))
        return False

    return {(c, s) for (c, s) in edges if reaches(s, c)}


def test_property_sweep_correction_dominance_invariant() -> None:
    rng = random.Random(SEED)
    scenarios_with_enforced_edges = 0
    scenarios_where_scores_alone_would_violate = 0
    scenarios_with_absent_or_decayed_correction = 0

    for scenario in range(SCENARIOS):
        node_count = rng.randint(4, 16)
        ids = [f"n{scenario}-{index}" for index in range(node_count)]
        pairs = _random_pairs(rng, ids)
        stale_ids = {stale for (_correction, stale) in pairs}
        correction_ids = {correction for (correction, _stale) in pairs}
        plan = _random_plan(rng)
        candidates = {}
        for node_id in ids:
            role = (
                "stale"
                if node_id in stale_ids
                else "correction"
                if node_id in correction_ids
                else "plain"
            )
            candidates[node_id] = _random_candidate(rng, node_id, plan, role)
        weights = (rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0), rng.uniform(0.0, 0.6))
        causal_mode = rng.random() < 0.25
        decision_mode = not causal_mode and rng.random() < 0.15
        modes = {"causal_mode": causal_mode, "decision_mode": decision_mode}

        output = _rank(pairs, candidates, plan, weights, **modes)
        rerun = _rank(pairs, candidates, plan, weights, **modes)
        baseline = _rank([], candidates, plan, weights, **modes)

        output_ids = _ids(output)
        # Deterministic, cycles included.
        assert _ids(rerun) == output_ids
        # The pass reorders; it never drops or admits anything, so a stale
        # node whose correction is absent/decayed keeps surfacing.
        assert set(output_ids) == set(_ids(baseline))
        # Flag comes from the supersedes mapping alone, present or not.
        for result in output:
            assert result.superseded is (result.node.id in stale_ids)
        for result in baseline:
            assert result.superseded is False

        position = {nid: index for index, nid in enumerate(output_ids)}
        present_edges = {
            (correction, stale)
            for (correction, stale) in pairs
            if correction != stale and correction in position and stale in position
        }
        enforceable = present_edges - _cycle_edges(present_edges)
        scores = {result.node.id: result.score for result in output}
        for correction, stale in enforceable:
            assert position[correction] < position[stale], (
                f"scenario {scenario}: correction {correction} at "
                f"{position[correction]} did not outrank {stale} at {position[stale]}"
            )
            if scores[correction] <= scores[stale]:
                scenarios_where_scores_alone_would_violate += 1
        scenarios_with_enforced_edges += bool(enforceable)
        for correction, stale in pairs:
            if correction != stale and stale in position and correction not in position:
                scenarios_with_absent_or_decayed_correction += 1
                break

        # Any prefix (max_results truncation happens downstream) that keeps
        # the stale node keeps its correction, ranked above.
        if output_ids:
            prefix = output_ids[: rng.randint(1, len(output_ids))]
            kept = set(prefix)
            for correction, stale in enforceable:
                if stale in kept:
                    assert correction in kept
                    assert prefix.index(correction) < prefix.index(stale)

        # Results on no supersedes edge keep their relative order exactly.
        involved = {nid for pair in pairs for nid in pair}
        assert [nid for nid in output_ids if nid not in involved] == [
            nid for nid in _ids(baseline) if nid not in involved
        ]

    # The sweep must not be vacuously green.
    assert scenarios_with_enforced_edges >= 100
    assert scenarios_where_scores_alone_would_violate >= 25
    assert scenarios_with_absent_or_decayed_correction >= 25


# --- End-to-end through a real store ----------------------------------------


def test_memory_recall_orders_and_flags_through_real_supersedes_edges(
    tmp_path: Path,
) -> None:
    """Full path: real SQLite supersedes query, access logging, event replace."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        stale = store.append_trace(
            "beacon endpoint timeout is 30 seconds for beacon retries",
            {"scope": "project:cd", "agent": "agent-a"},
        )
        correction = store.append_trace(
            "beacon endpoint timeout corrected to 5 seconds",
            {"scope": "project:cd", "agent": "agent-b"},
        )
        store.update_node(stale.id, stats={"usefulness_score": 1.0, "confidence": 1.0})
        store.create_connection(correction.id, stale.id, "supersedes")

        results = MemoryRecallService(store).memory_recall(
            "beacon endpoint timeout",
            scope="project:cd",
            max_results=5,
        )
        ids = _ids(results)
        assert stale.id in ids and correction.id in ids
        assert ids.index(correction.id) < ids.index(stale.id)
        flags = {result.node.id: result.superseded for result in results}
        assert flags[stale.id] is True
        assert flags[correction.id] is False
        # The flag survives the record-access/event-id replace() calls and
        # reaches the serialized payload.
        by_id = {result.node.id: result for result in results}
        assert by_id[stale.id].recall_event_id is not None
        assert by_id[stale.id].to_dict()["superseded"] is True
