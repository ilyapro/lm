"""Imperative-tone metrics for the MCP server instructions.

These tests pin the qualitative properties that make the instructions
imperative rather than advisory: MUST counts, action-lifecycle hooks,
semantic negative-policy coverage, absence of weak language, and
superintelligence framing. They guard against regressions toward a softer,
advisory tone without requiring every prohibition to use one exact phrase.
"""

from __future__ import annotations

import pytest

from living_memory.server import _server_instructions

MAX_INSTRUCTION_BYTES = 9000


@pytest.fixture(scope="module")
def text() -> str:
    return _server_instructions("global")


def test_required_keywords_present(text: str) -> None:
    needed = [
        "MUST",
        "BEFORE",
        "AFTER",
        "recall before",
        "remember after",
        "memory_teach",
        "level:schema",
        "cross-project",
    ]
    missing = [kw for kw in needed if kw not in text]
    assert not missing, f"missing required keywords: {missing}"


def test_must_count_at_least_ten(text: str) -> None:
    must_count = text.count("MUST")
    assert must_count >= 10, (
        f"imperative tone requires >=10 MUST instances, found {must_count}"
    )


def test_negative_policy_is_hard_and_concrete(text: str) -> None:
    """Negative policy is semantic, not a literal `MUST NOT` spelling check.

    The server instructions intentionally use hard prohibitions such as
    `DO NOT`, `What NOT to store`, named anti-patterns, and concrete forbidden
    storage categories. This keeps the contract user-facing: weak advice is not
    enough, but the exact phrase `MUST NOT` is not required.
    """

    lowered = text.lower()
    hard_negative_markers = [
        "must not",
        "do not",
        "what not to store",
    ]
    found_markers = [marker for marker in hard_negative_markers if marker in lowered]
    assert found_markers, (
        "negative policy must use hard prohibitive language such as DO NOT or "
        "What NOT to store; weak advisory phrasing is not enough"
    )

    required_sections = [
        "## Anti-patterns",
        "## What NOT to store",
    ]
    missing_sections = [section for section in required_sections if section not in text]
    assert not missing_sections, (
        f"negative policy must include concrete prohibition sections: {missing_sections}"
    )

    forbidden_storage_categories = [
        "routine actions",
        "copies of code",
        "speculation",
        "unverified plans",
        "verified facts only",
    ]
    missing_categories = [
        category for category in forbidden_storage_categories if category not in lowered
    ]
    assert not missing_categories, (
        "What NOT to store must forbid concrete storage categories, missing: "
        f"{missing_categories}"
    )

    weak_negative_phrases = [
        "try not to",
        "consider not",
        "avoid if possible",
        "you may skip",
    ]
    found_weak = [phrase for phrase in weak_negative_phrases if phrase in lowered]
    assert not found_weak, (
        f"negative policy must be mandatory rather than advisory, found: {found_weak}"
    )


def test_before_count_at_least_four(text: str) -> None:
    before_count = text.count("BEFORE")
    assert before_count >= 4, (
        f"need >=4 BEFORE-style action-lifecycle hooks, found {before_count}"
    )


def test_after_count_at_least_two(text: str) -> None:
    after_count = text.count("AFTER")
    assert after_count >= 2, (
        f"need >=2 AFTER-style action-lifecycle hooks, found {after_count}"
    )


def test_tool_specific_action_hooks(text: str) -> None:
    """At least 4 concrete tool names must appear in the hook section."""

    tool_names = ["Edit", "Write", "Bash", "grep", "mkdir", "curl"]
    covered = [name for name in tool_names if name in text]
    assert len(covered) >= 4, (
        f"need >=4 concrete tool-name hooks, found only: {covered}"
    )


def test_anti_pattern_markers_at_least_three(text: str) -> None:
    """At least 3 anti-pattern-style markers must be present."""

    markers = ["anti-pattern", "do not", "avoid", "wrong", "mistake", "re-invent"]
    lowered = text.lower()
    found = [m for m in markers if m in lowered]
    assert len(found) >= 3, (
        f"need >=3 anti-pattern markers, found only: {found}"
    )


def test_anti_pattern_section_has_examples(text: str) -> None:
    """The anti-pattern section must enumerate at least 3 named failure modes."""

    named_failures = [
        "mkdir-vs-API",
        "shell-watchdog-vs-loop",
        "grep-before-recall",
        "silent-correction",
        "lookup-table-as-learning",
    ]
    present = [name for name in named_failures if name in text]
    assert len(present) >= 3, (
        f"need >=3 named failure modes, found: {present}"
    )


def test_no_weak_language(text: str) -> None:
    """Advisory phrasing weakens imperative tone — must be absent."""

    weak = ["you can also", "consider ", "might want", "feel free"]
    lowered = text.lower()
    found = [w for w in weak if w in lowered]
    assert not found, f"weak/advisory language found: {found}"


def test_superintelligence_framing_in_opening(text: str) -> None:
    """First paragraph must frame agent+LM as one cognitive system."""

    head = text[:600].lower()
    phrases = [
        "cognitive system",
        "single intelligence",
        "agent and lm",
        "one cognitive",
        "superintelligence",
    ]
    matched = [p for p in phrases if p in head]
    assert matched, (
        "first paragraph must frame agent+LM as one cognitive system; "
        f"none of {phrases} found in opening"
    )


def test_size_within_mcp_budget(text: str) -> None:
    """Instructions must stay bounded without enforcing the obsolete 6.5KB cap."""

    size = len(text.encode("utf-8"))
    assert size <= MAX_INSTRUCTION_BYTES, (
        f"instructions size {size} bytes exceeds {MAX_INSTRUCTION_BYTES}; "
        "the MCP instruction contract allows the current imperative content, "
        "but intentional growth needs an explicit test-budget update"
    )


def test_default_scope_substituted(text: str) -> None:
    """The default scope literal must still appear so existing tests keep passing."""

    assert "Your default scope is global" in text
    custom = _server_instructions("project:custom-scope")
    assert "Your default scope is project:custom-scope" in custom


def test_action_hook_pairs_recall_with_each_phase(text: str) -> None:
    """Each BEFORE hook must pair with a 'recall' obligation in the same vicinity."""

    lines = text.split("\n")
    before_indices = [i for i, line in enumerate(lines) if line.startswith("### BEFORE")]
    assert len(before_indices) >= 4, (
        f"expected at least 4 '### BEFORE' headers, found {len(before_indices)}"
    )
    for idx in before_indices:
        window = "\n".join(lines[idx : idx + 6]).lower()
        assert "recall" in window, (
            f"BEFORE header at line {idx} has no 'recall' obligation in next 6 lines: "
            f"{lines[idx]!r}"
        )


def test_after_hook_pairs_remember_with_each_phase(text: str) -> None:
    """Each AFTER hook must pair with a 'remember' obligation in the same vicinity."""

    lines = text.split("\n")
    after_indices = [i for i, line in enumerate(lines) if line.startswith("### AFTER")]
    assert len(after_indices) >= 2, (
        f"expected at least 2 '### AFTER' headers, found {len(after_indices)}"
    )
    for idx in after_indices:
        window = "\n".join(lines[idx : idx + 6]).lower()
        assert "remember" in window, (
            f"AFTER header at line {idx} has no 'remember' obligation in next 6 lines: "
            f"{lines[idx]!r}"
        )


def test_correction_loop_section_present(text: str) -> None:
    """`memory_teach` must be framed as the correction mechanism, not a side feature."""

    assert "memory_teach" in text
    assert "correction" in text.lower(), (
        "memory_teach must be framed as the correction loop"
    )


def test_cross_project_transfer_explicit(text: str) -> None:
    """Cross-project knowledge transfer must be an explicit section."""

    assert "cross-project" in text.lower()
    lowered = text.lower()
    assert "broaden recall" in lowered or "across projects" in lowered or "across scopes" in lowered, (
        "cross-project section must instruct broadening recall across scopes"
    )
