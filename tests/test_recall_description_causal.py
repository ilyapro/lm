"""Both feedback arms carry the same recall protocol and causal debugging hint."""

from living_memory.server import (
    _IRRELEVANT_FIELD_DESCRIPTION,
    _RECALL_DESCRIPTION,
    _RECALL_DESCRIPTION_MANDATORY,
    _USED_FIELD_DESCRIPTION,
    _explicit_feedback_texts,
)


def test_both_arms_keep_causal_hint_within_budget() -> None:
    for arm in ("optional", "mandatory"):
        text = _explicit_feedback_texts(arm)["recall"]
        assert "Depth 'causal' when debugging." in text
        assert len(text) < 1024


def test_mandatory_arm_extends_the_shared_recall_protocol() -> None:
    assert _RECALL_DESCRIPTION_MANDATORY == (
        _RECALL_DESCRIPTION
        + " You MUST mark each recall's results on your next call: used ids as "
        "used, off-topic ids as irrelevant."
    )


def test_optional_arm_keeps_query_local_feedback() -> None:
    optional = _explicit_feedback_texts("optional")
    assert optional["recall"] is _RECALL_DESCRIPTION
    assert optional["used"] == _USED_FIELD_DESCRIPTION
    assert optional["irrelevant"] == _IRRELEVANT_FIELD_DESCRIPTION
    assert "you used" in optional["used"]
    assert "did not fit their query" in optional["irrelevant"]
