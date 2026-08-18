"""Contracts for the shared content-grounding module.

``living_memory.grounding`` is the single implementation of the
IDF-containment measure. Two callers depend on it meaning the same thing:
the live credit loop (``feedback.apply_pending_recall_feedback``) and the
offline harness label (``replay._containment`` / ``replay.apply_grounding``).
These tests pin the arithmetic, and pin that neither caller re-implements it.
"""

from __future__ import annotations

import inspect
import math

import pytest

from living_memory import grounding
from living_memory.grounding import (
    DEFAULT_MIN_CONTAINMENT,
    Grounding,
    build_idf,
    containment,
    ground_results,
    ground_token_sets,
    token_set,
)


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------


def test_build_idf_is_log_one_plus_documents_over_document_frequency() -> None:
    idf = build_idf([{"a", "b"}, {"a", "c"}, {"a"}])

    assert idf["a"] == pytest.approx(math.log(1.0 + 3 / 3))
    assert idf["b"] == pytest.approx(math.log(1.0 + 3 / 1))
    assert idf["c"] == pytest.approx(math.log(1.0 + 3 / 1))
    # Ubiquitous tokens keep a positive but small weight; rare ones dominate.
    assert 0.0 < idf["a"] < idf["b"]


def test_build_idf_counts_documents_not_occurrences() -> None:
    """A token repeated inside one document is still one document."""

    assert build_idf([["a", "a", "a"]]) == pytest.approx(build_idf([["a"]]))


def test_build_idf_of_empty_corpus_is_empty() -> None:
    assert build_idf([]) == {}


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


def test_containment_is_one_when_the_trace_covers_every_node_token() -> None:
    idf = build_idf([{"alpha", "beta"}, {"alpha", "beta", "gamma"}])

    assert containment({"alpha", "beta"}, {"alpha", "beta", "gamma"}, idf) == pytest.approx(1.0)


def test_containment_is_zero_without_overlap() -> None:
    idf = build_idf([{"alpha"}, {"gamma"}])

    assert containment({"alpha"}, {"gamma"}, idf) == 0.0


@pytest.mark.parametrize(
    ("node_tokens", "trace_tokens"),
    [(None, {"a"}), ({"a"}, None), (frozenset(), {"a"}), ({"a"}, frozenset())],
)
def test_containment_of_an_unmeasurable_pair_is_zero(node_tokens, trace_tokens) -> None:
    assert containment(node_tokens, trace_tokens, {"a": 1.0}) == 0.0


def test_containment_is_zero_when_the_node_carries_no_idf_mass() -> None:
    """Unknown tokens weigh nothing, so the ratio has no denominator."""

    assert containment({"unseen"}, {"unseen"}, {}) == 0.0


def test_containment_weights_rare_tokens_above_boilerplate() -> None:
    """The point of the IDF: sharing an identifier beats sharing filler.

    Two nodes with the same *count* of shared tokens score differently when
    what they share differs in rarity — this is what stops a boilerplate
    preamble from grounding an unrelated node.
    """

    documents = [
        {"the", "note", "zr9042"},
        {"the", "note", "kubernetes"},
        {"the", "note", "palette"},
        {"the", "note", "zr9042", "rollback"},
    ]
    idf = build_idf(documents)
    trace = {"the", "note", "zr9042"}

    shares_identifier = containment({"the", "note", "zr9042"}, trace, idf)
    shares_boilerplate = containment({"the", "note", "kubernetes"}, trace, idf)

    assert shares_identifier == pytest.approx(1.0)
    assert shares_boilerplate < 0.5
    assert shares_identifier > shares_boilerplate


def test_containment_is_directional() -> None:
    """Containment asks what share of the *node* the trace covers.

    A short node fully quoted by a long trace is grounded; the same pair read
    the other way round is not. Credit must follow the first reading.
    """

    node = {"zr9042", "alembic"}
    trace = {"zr9042", "alembic", "rollback", "drain", "workers", "staging"}
    idf = build_idf([node, trace])

    assert containment(node, trace, idf) == pytest.approx(1.0)
    assert containment(trace, node, idf) < 1.0


# ---------------------------------------------------------------------------
# Event-level grading
# ---------------------------------------------------------------------------


GROUNDED = "alembic migration checksum zr9042 failure fixed by pinning sqlalchemy-utils"
UNRELATED = "kubernetes ingress annotation cheat sheet for cert-manager wildcards"
TRACE = (
    "Deployed the alembic migration checksum zr9042 failure fix by pinning "
    "sqlalchemy-utils on staging."
)


def test_ground_results_separates_used_from_merely_delivered() -> None:
    graded = ground_results(TRACE, {"used": GROUNDED, "ignored": UNRELATED})

    assert graded["used"].grounded is True
    assert graded["ignored"].grounded is False
    assert graded["used"].containment > graded["ignored"].containment
    assert isinstance(graded["used"], Grounding)
    assert graded["used"].node_id == "used"


def test_ground_results_threshold_is_the_only_gate() -> None:
    """The verdict is exactly ``containment >= min_containment``."""

    graded = ground_results(TRACE, {"ignored": UNRELATED})
    value = graded["ignored"].containment

    assert ground_results(TRACE, {"ignored": UNRELATED}, min_containment=value)[
        "ignored"
    ].grounded is True
    assert ground_results(
        TRACE, {"ignored": UNRELATED}, min_containment=value + 1e-9
    )["ignored"].grounded is False


def test_ground_results_on_no_results_does_no_work() -> None:
    assert ground_results(TRACE, {}) == {}


def test_ground_results_delegates_to_ground_token_sets() -> None:
    """The text entry point tokenizes and then runs the shared arithmetic."""

    contents = {"used": GROUNDED, "ignored": UNRELATED}
    from_text = ground_results(TRACE, contents)
    from_tokens = ground_token_sets(
        token_set(TRACE), {key: token_set(value) for key, value in contents.items()}
    )

    assert from_text == from_tokens


def test_default_threshold_is_the_replay_default() -> None:
    from living_memory import replay

    assert DEFAULT_MIN_CONTAINMENT == 0.25
    assert replay.DEFAULT_MIN_CONTAINMENT is DEFAULT_MIN_CONTAINMENT


# ---------------------------------------------------------------------------
# Single source of truth
# ---------------------------------------------------------------------------


def test_replay_containment_delegates_to_the_shared_module() -> None:
    """``replay._containment`` must call the module, not re-derive the ratio."""

    from living_memory import replay

    source = inspect.getsource(replay._containment)
    assert "_idf_containment(" in source
    assert "math.log" not in source
    assert "sum(" not in source


def test_feedback_grounds_through_the_shared_module() -> None:
    """The live credit loop must not carry its own containment arithmetic."""

    from living_memory import feedback

    source = inspect.getsource(feedback.apply_pending_recall_feedback)
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "ground_results(" in code
    for forbidden in ("build_idf", "math.log", "document_frequency", "Counter("):
        assert forbidden not in code, f"{forbidden} re-derives grounding in the live path"
    assert feedback.RECALL_CREDIT_MIN_CONTAINMENT is DEFAULT_MIN_CONTAINMENT


def test_module_exposes_one_containment_implementation() -> None:
    """Exactly one place in the package computes the IDF-weighted ratio."""

    implementations = [
        name
        for name, value in vars(grounding).items()
        if callable(value) and name in {"containment"}
    ]
    assert implementations == ["containment"]
    assert "shared / total" in inspect.getsource(containment)
