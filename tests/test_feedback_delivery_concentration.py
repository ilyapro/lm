"""Closed-loop proof that damped reinforcement reduces delivery concentration.

The delivery->reinforcement loop under test: memory_recall records a recall
event, the next compatible memory_remember implicitly reinforces every
delivered result (apply_pending_recall_feedback), and feedback_weighted_score
turns the accrued usefulness into a rank boost — so delivery itself breeds
more delivery. The damped rule in feedback.py scales positive usefulness
increments by max(floor, 1-usefulness) / (1 + damping * ln(1+access_count)).

Simulation design (seeded, deterministic under the hash embedding backend):

* Corpus: 5 topics x 6 paraphrases = 30 traces with a designated ground-truth
  mapping (a topic's paraphrases are relevant to that topic's queries). All
  paraphrases of a topic share an identical token structure and differ only in
  one codeword, so within-topic base-score differences are query-driven, not
  wording noise.
* Generic queries (~2/3, "anchor incident review") tie all six paraphrases;
  the boost leader wins, gets delivered, gets reinforced — the entrenchment
  engine. Variant queries (~1/3, "anchor incident <codeword>") clearly favor
  one paraphrase at a base-score edge of ~1.36-1.41x — below the legacy
  boost ceiling (FEEDBACK_MULTIPLIER_CAP 1.4) but above what the damped rule
  lets an entrenched node accrue over this horizon, so they probe whether
  entrenchment can override a better relevance match ("capture").
* Retrieval weights are frozen (learning_rate=0) at the vector-dominant
  learned-production values so the ONLY difference between the two arms is
  the usefulness increment rule; the legacy arm reproduces the historical
  flat 0.1*signal increment by pinning the knobs to their neutral values
  (floor 1.0, damping 0.0 — pinned exact by the neutrality test below).
* depth=0 recalls isolate the usefulness channel from graph-edge effects;
  follow-up notes share the session (strong pending-event match) and echo
  exactly the four corpus tokens that appear in no query (`caused`,
  `misconfiguration`, `mitigation`, `documented`), so closing the loop never
  pollutes the ranked corpus. That echo is load-bearing since credit became
  grounded (feedback.RECALL_CREDIT_POLICIES): reinforcement now reaches only
  results whose content the consuming trace used, so a vocabulary-disjoint
  note would reinforce nothing and the entrenchment engine this test measures
  would not run at all. Because every paraphrase of every topic shares an
  identical token structure, the note grounds in whichever paraphrase was
  delivered at an identical containment of 0.2961 (vs the 0.25 threshold),
  keeping the arms symmetric and the loop seed-independent.
* Volume stays under 100 traces per scope so auto-consolidation never fires.

Assertion margins were calibrated against seeds {7, 11, 13, 17, 23}; the two
parametrized seeds are not cherry-picked — every measured seed passes with
the asserted slack (topic-mean top1 delta >= 0.083 vs asserted 0.05, top3
delta >= 0.055 vs 0.03, capture delta >= 0.259 vs 0.15).
"""

from __future__ import annotations

import collections
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import living_memory.feedback as feedback_module
from living_memory.feedback import (
    _positive_reinforcement_gain,
    apply_retrieval_feedback,
)
from living_memory.models import Node
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)


SCOPE = "project:concentration"
ROUNDS = 12
RECALLS_PER_TOPIC_TURN = 3

TOPICS: dict[str, dict[str, Any]] = {
    "redis": {
        "anchor": "redis cache eviction pressure",
        "variants": ["alpha", "bravo", "gamma", "delta", "sigma", "omega"],
    },
    "auth": {
        "anchor": "auth token refresh expiry",
        "variants": ["kilo", "lima", "mike", "nova", "papa", "zulu"],
    },
    "deploy": {
        "anchor": "deploy rollout migration failure",
        "variants": ["quill", "tango", "vireo", "waltz", "xenon", "rombo"],
    },
    "queue": {
        "anchor": "queue consumer backlog latency",
        "variants": ["ember", "flint", "haven", "ivory", "ochre", "umber"],
    },
    "storage": {
        "anchor": "disk volume capacity alert",
        "variants": ["cedar", "birch", "aspen", "maple", "rowan", "alder"],
    },
}


def _paraphrase_content(anchor: str, variant: str, index: int) -> str:
    # Identical token structure for every paraphrase of a topic; only the
    # variant codeword differs (the index is digits-only and dropped by the
    # tokenizer), so within-topic base differences are variant-driven.
    return (
        f"{anchor} incident caused by {variant} misconfiguration "
        f"{variant} mitigation documented {index}"
    )


@dataclass(frozen=True, slots=True)
class SimOutcome:
    top1_share: float
    top3_share: float
    global_top1_share: float
    global_top3_share: float
    hit_rate: float
    capture_rate: float
    distinct_delivered: int
    total_deliveries: int
    events_consumed: int
    max_corpus_usefulness: float


def _run_closed_loop(db_path: Path, *, seed: int) -> SimOutcome:
    """Drive seeded recall->remember cycles through the real service layer."""

    mcp = create_mcp_server(db_path, mcp_factory=FakeMCP)
    store = mcp.memory_store
    rng = random.Random(seed)

    ground_truth: dict[str, set[str]] = {}
    variant_node: dict[tuple[str, str], str] = {}
    for topic, spec in TOPICS.items():
        ids: set[str] = set()
        for index, variant in enumerate(spec["variants"]):
            remembered = mcp.tools["memory_remember"](
                _paraphrase_content(spec["anchor"], variant, index),
                {
                    "scope": SCOPE,
                    "agent": "corpus-loader",
                    "task": "corpus-load",
                    "session_id": "corpus-load",
                },
            )
            ids.add(remembered["node"]["id"])
            variant_node[(topic, variant)] = remembered["node"]["id"]
        ground_truth[topic] = ids

    # Freeze the weights at the vector-dominant learned-production values so
    # both arms rank under identical weights for the whole horizon and the
    # only divergence is the usefulness increment rule.
    store.set_retrieval_weights(SCOPE, bm25=0.10, vector=0.85, graph=0.05, learning_rate=0.0)

    ambient = {"agent": "sim-agent", "task": "sim-task", "session_id": "sim-session"}
    note_context = {
        "scope": SCOPE,
        "agent": "sim-agent",
        "task": "sim-task",
        "session_id": "sim-session",
    }

    deliveries: collections.Counter[str] = collections.Counter()
    topic_deliveries: dict[str, collections.Counter[str]] = {
        topic: collections.Counter() for topic in TOPICS
    }
    hits: list[float] = []
    variant_queries = 0
    variant_captured = 0  # variant recalls where the favored paraphrase lost
    variant_cursor: dict[str, int] = {topic: 0 for topic in TOPICS}
    events_consumed = 0
    note_index = 0
    topic_names = list(TOPICS)

    for _round in range(ROUNDS):
        order = rng.sample(topic_names, len(topic_names))
        for topic in order:
            spec = TOPICS[topic]
            for _slot in range(RECALLS_PER_TOPIC_TURN):
                if rng.random() < 1.0 / 3.0:
                    variant = spec["variants"][variant_cursor[topic] % len(spec["variants"])]
                    variant_cursor[topic] += 1
                    query = f"{spec['anchor']} incident {variant}"
                else:
                    variant = None
                    query = f"{spec['anchor']} incident review"
                recalled = mcp.tools["memory_recall"](
                    query,
                    scope=SCOPE,
                    max_results=1,
                    depth=0,
                    ambient_context=dict(ambient),
                )
                delivered_ids = [result["node"]["id"] for result in recalled["results"]]
                for node_id in delivered_ids:
                    deliveries[node_id] += 1
                    topic_deliveries[topic][node_id] += 1
                if variant is not None and delivered_ids:
                    variant_queries += 1
                    if delivered_ids[0] != variant_node[(topic, variant)]:
                        variant_captured += 1
                if delivered_ids:
                    relevant = ground_truth[topic]
                    hits.append(
                        sum(1 for node_id in delivered_ids if node_id in relevant)
                        / len(delivered_ids)
                    )
                else:
                    hits.append(0.0)
            note_index += 1
            noted = mcp.tools["memory_remember"](
                "session follow-up: applied the documented mitigation for the "
                "misconfiguration it caused, verified and logged after review "
                f"pass {note_index}",
                dict(note_context),
            )
            events_consumed += len(noted["implicit_feedback"]["recall_event_ids"])

    total = sum(deliveries.values())
    ranked = deliveries.most_common()
    topic_top1: list[float] = []
    topic_top3: list[float] = []
    for counter in topic_deliveries.values():
        topic_total = sum(counter.values())
        assert topic_total > 0
        common = counter.most_common()
        topic_top1.append(common[0][1] / topic_total)
        topic_top3.append(sum(count for _node_id, count in common[:3]) / topic_total)

    max_corpus_usefulness = max(
        store.get_node(node_id).usefulness_score
        for ids in ground_truth.values()
        for node_id in ids
    )
    return SimOutcome(
        top1_share=statistics.mean(topic_top1),
        top3_share=statistics.mean(topic_top3),
        global_top1_share=ranked[0][1] / total,
        global_top3_share=sum(count for _node_id, count in ranked[:3]) / total,
        hit_rate=statistics.mean(hits),
        capture_rate=variant_captured / variant_queries if variant_queries else 0.0,
        distinct_delivered=len(ranked),
        total_deliveries=total,
        events_consumed=events_consumed,
        max_corpus_usefulness=max_corpus_usefulness,
    )


@pytest.fixture()
def _sim_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.delenv("LM_RETRIEVAL_TUNING_POLICY", raising=False)
    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)


@pytest.mark.parametrize("seed", [7, 17])
def test_damped_rule_reduces_delivery_concentration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _sim_env: None,
    seed: int,
) -> None:
    damped = _run_closed_loop(tmp_path / "damped.sqlite3", seed=seed)

    with monkeypatch.context() as legacy_knobs:
        legacy_knobs.setattr(feedback_module, "USEFULNESS_GAIN_HEADROOM_FLOOR", 1.0)
        legacy_knobs.setattr(feedback_module, "USEFULNESS_GAIN_ACCESS_DAMPING", 0.0)
        legacy = _run_closed_loop(tmp_path / "legacy.sqlite3", seed=seed)

    # Identical seeded schedules: the arms saw the same number of recalls and
    # both closed the implicit feedback loop on every cycle.
    assert damped.total_deliveries == legacy.total_deliveries
    assert legacy.events_consumed == damped.events_consumed
    assert legacy.events_consumed >= ROUNDS * len(TOPICS) * RECALLS_PER_TOPIC_TURN

    # The pathology genuinely emerges under the legacy rule: leaders saturate
    # to usefulness 1.0 and capture a real share of better-matched queries.
    assert legacy.max_corpus_usefulness == pytest.approx(1.0)
    assert legacy.top1_share >= 0.6
    assert legacy.capture_rate >= 0.15

    # Damped arm: strictly lower top-1/top-3 delivery concentration by a real
    # margin, per-topic and globally.
    assert damped.top1_share <= legacy.top1_share - 0.05
    assert damped.top3_share <= legacy.top3_share - 0.03
    assert damped.global_top1_share < legacy.global_top1_share
    assert damped.global_top3_share <= legacy.global_top3_share - 0.03
    # Entrenchment stops overriding clearly better relevance matches, spreads
    # deliveries over more distinct nodes, and stays short of saturation.
    assert damped.capture_rate <= legacy.capture_rate - 0.15
    assert damped.distinct_delivered > legacy.distinct_delivered
    assert damped.max_corpus_usefulness <= 0.9

    # No ground-truth regression: damping redistributes deliveries among
    # relevant nodes instead of promoting irrelevant ones.
    assert damped.hit_rate >= legacy.hit_rate - 0.02


def test_neutral_knobs_reproduce_legacy_flat_increment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy arm of the simulation is exactly the historical rule."""

    monkeypatch.setattr(feedback_module, "USEFULNESS_GAIN_HEADROOM_FLOOR", 1.0)
    monkeypatch.setattr(feedback_module, "USEFULNESS_GAIN_ACCESS_DAMPING", 0.0)
    for usefulness in (-1.0, -0.3, 0.0, 0.5, 0.9, 1.0):
        for access in (0, 5, 600):
            node = Node(
                id="probe",
                level="trace",
                content="probe",
                usefulness_score=usefulness,
                access_count=access,
            )
            assert _positive_reinforcement_gain(node) == pytest.approx(1.0)

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("entrenched probe", {"scope": "project:probe"})
        store.update_node(node.id, stats={"usefulness_score": 0.5, "access_count": 600})
        result = type("Result", (), {"node_id": node.id, "vector_score": 1.0})()
        apply_retrieval_feedback(store, result, useful=True, signal=1.0)
        assert store.get_node(node.id).usefulness_score == pytest.approx(0.6)


def test_gain_shrinks_with_usefulness_and_access() -> None:
    """Entrenched nodes need disproportionately more evidence per unit of boost."""

    def gain(usefulness: float, access: int) -> float:
        return _positive_reinforcement_gain(
            Node(
                id="probe",
                level="trace",
                content="probe",
                usefulness_score=usefulness,
                access_count=access,
            )
        )

    assert gain(0.0, 0) == pytest.approx(1.0)
    # Monotone in usefulness down to the headroom floor.
    assert gain(0.0, 0) > gain(0.4, 0) > gain(0.7, 0)
    assert gain(0.8, 0) == gain(1.0, 0) == pytest.approx(
        feedback_module.USEFULNESS_GAIN_HEADROOM_FLOOR
    )
    # Monotone in delivered-access volume at any usefulness level.
    assert gain(0.5, 0) > gain(0.5, 5) > gain(0.5, 50) > gain(0.5, 600)
    # Negative usefulness recovers with full headroom (access still damps).
    assert gain(-1.0, 0) == pytest.approx(1.0)
    assert gain(-1.0, 600) == gain(0.0, 600)
    # The production-shaped entrenched node gains an order of magnitude slower.
    assert gain(1.0, 600) < gain(0.0, 0) / 20.0


def test_explicit_negative_feedback_keeps_full_strength(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("entrenched but wrong", {"scope": "project:probe"})
        store.update_node(node.id, stats={"usefulness_score": 0.9, "access_count": 600})
        result = type("Result", (), {"node_id": node.id, "vector_score": 1.0})()
        apply_retrieval_feedback(store, result, useful=False, signal=1.0)
        assert store.get_node(node.id).usefulness_score == pytest.approx(0.8)


def test_saturation_stays_reachable_under_damping(tmp_path: Path) -> None:
    """The headroom floor never zeroes gains: strong evidence still saturates."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("nearly saturated", {"scope": "project:probe"})
        store.update_node(node.id, stats={"usefulness_score": 0.9, "access_count": 600})
        result = type("Result", (), {"node_id": node.id, "vector_score": 1.0})()
        for _ in range(40):
            apply_retrieval_feedback(store, result, useful=True, signal=5.0)
            if store.get_node(node.id).usefulness_score >= 1.0:
                break
        assert store.get_node(node.id).usefulness_score == pytest.approx(1.0)
