"""The mandatory memory_recall description carries the depth 'causal' hint.

The first fit of the mandatory arm into the 1024-char budget dropped
"Depth 'causal' when debugging."; it is back by rewording. The optional arm's
current texts are pinned by sha256; its recall description now makes lookup
conditional on insufficient delivered knowledge.
"""

from __future__ import annotations

import hashlib

from living_memory.server import (
    _IRRELEVANT_FIELD_DESCRIPTION,
    _RECALL_DESCRIPTION,
    _RECALL_DESCRIPTION_MANDATORY,
    _USED_FIELD_DESCRIPTION,
    _explicit_feedback_texts,
)

MAX_DESCRIPTION_CHARS = 1024

OPTIONAL_SHA256 = {
    "recall": "10d3843d1348e83b1b85051ab8e08a7a8c7ac6ea3391808d4cc51433e0f6e5be",
    "used": "4f999afcd41803aa54c9df439f4bca2301ef1f620b5804b5fc2efa1b4f8b8dd1",
    "irrelevant": "9996b53a921be8d0b205633827ef299acf6a08316e1d4472ac5b489c63b9ccae",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_mandatory_recall_has_causal_hint_within_budget() -> None:
    text = _explicit_feedback_texts("mandatory")["recall"]
    assert text is _RECALL_DESCRIPTION_MANDATORY
    assert "Depth 'causal' when debugging." in text
    assert len(text) <= MAX_DESCRIPTION_CHARS, len(text)


def test_mandatory_recall_keeps_the_binding_sentences() -> None:
    text = _RECALL_DESCRIPTION_MANDATORY
    for sentence in (
        "You MUST recall BEFORE acting",
        "You MUST recall MID-WORK",
        "Default: when uncertain, recall.",
        "A level:schema result is a binding procedure — follow it literally.",
        "You MUST mark each recall's results on your next call: used ids as "
        "used, off-topic ids as irrelevant.",
    ):
        assert sentence in text, sentence


def test_optional_arm_texts_match_current_pins() -> None:
    optional = _explicit_feedback_texts("optional")
    assert optional["recall"] is _RECALL_DESCRIPTION
    assert optional["used"] == _USED_FIELD_DESCRIPTION
    assert optional["irrelevant"] == _IRRELEVANT_FIELD_DESCRIPTION
    assert {name: _sha(optional[name]) for name in OPTIONAL_SHA256} == OPTIONAL_SHA256
