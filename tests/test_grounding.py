"""Contracts for the shared content-grounding module.

``living_memory.grounding`` is the single implementation of the
IDF-containment measure. Two callers depend on it meaning the same thing:
the live credit loop (``feedback.apply_pending_recall_feedback``) and the
offline harness label (``replay._containment`` / ``replay.apply_grounding``).
These tests pin the arithmetic, and pin that neither caller re-implements it.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from living_memory import grounding
from living_memory.grounding import (
    CALIBRATED_MIN_CONTAINMENT,
    DEFAULT_MIN_CONTAINMENT,
    MIN_CONTAINMENT_ENV_VAR,
    Grounding,
    build_idf,
    containment,
    ground_results,
    ground_token_sets,
    resolve_min_containment,
    token_set,
)

RECALIBRATION_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "artifacts" / "grounding" / "recalibration-2026-09.json"
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


def test_default_threshold_is_the_recalibrated_value() -> None:
    """The constant is the value the September 2026 recalibration adopted.

    ``artifacts/grounding/recalibration-2026-09.json`` states the criterion and
    the sweep; its ``adopted`` key and the constant must not drift apart. The
    process default equals the constant unless the environment overrides it.
    """

    from living_memory import replay

    assert CALIBRATED_MIN_CONTAINMENT == 0.22
    artifact = json.loads(RECALIBRATION_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["adopted"] == CALIBRATED_MIN_CONTAINMENT
    assert DEFAULT_MIN_CONTAINMENT == resolve_min_containment(
        os.environ.get(MIN_CONTAINMENT_ENV_VAR)
    )
    assert replay.DEFAULT_MIN_CONTAINMENT is DEFAULT_MIN_CONTAINMENT


# ---------------------------------------------------------------------------
# Environment override
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.2", 0.2), (" 0.15 ", 0.15), ("1", 1.0), ("1e-2", 0.01), ("0.999", 0.999)],
)
def test_resolve_min_containment_accepts_a_decimal_in_the_unit_interval(
    raw: str, expected: float
) -> None:
    assert resolve_min_containment(raw, fallback=0.4) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "abc", "0,25", "0", "-0.1", "1.0001", "2", "nan", "inf", "-inf"],
)
def test_resolve_min_containment_falls_back_on_unusable_values(raw: str | None) -> None:
    """Unset or blank means no override; garbage or out of (0, 1] is ignored."""

    assert resolve_min_containment(raw, fallback=0.4) == 0.4


def test_resolve_min_containment_default_fallback_is_the_calibrated_constant() -> None:
    assert resolve_min_containment(None) == CALIBRATED_MIN_CONTAINMENT
    assert resolve_min_containment("garbage") == CALIBRATED_MIN_CONTAINMENT


_CONSUMER_PROBE = """
import json
from living_memory import attestation, feedback, grounding, replay
print(json.dumps({
    "grounding": grounding.DEFAULT_MIN_CONTAINMENT,
    "feedback": feedback.RECALL_CREDIT_MIN_CONTAINMENT,
    "attestation": attestation.ATTESTATION_MIN_CONTAINMENT,
    "replay": replay.DEFAULT_MIN_CONTAINMENT,
    "label_config": replay.LabelConfig().min_containment,
}))
"""


def _consumers_in_fresh_process(env_value: str | None) -> dict[str, float]:
    """Import every consumer in a new interpreter with the override set (or unset)."""

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env.pop(MIN_CONTAINMENT_ENV_VAR, None)
    if env_value is not None:
        env[MIN_CONTAINMENT_ENV_VAR] = env_value
    completed = subprocess.run(
        [sys.executable, "-c", _CONSUMER_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_env_override_reaches_every_consumer_at_import() -> None:
    """The override is read once at import and every caller binds that value."""

    values = _consumers_in_fresh_process("0.2")

    assert values == {key: 0.2 for key in values}
    assert set(values) == {"grounding", "feedback", "attestation", "replay", "label_config"}


@pytest.mark.parametrize("env_value", [None, "not-a-number", "1.5", "0"])
def test_env_override_falls_back_to_the_constant_at_import(env_value: str | None) -> None:
    values = _consumers_in_fresh_process(env_value)

    assert values == {key: CALIBRATED_MIN_CONTAINMENT for key in values}


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
