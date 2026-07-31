"""Imperative-tone metrics for the MCP server instructions.

These tests pin the qualitative properties that make the instructions
imperative rather than advisory: MUST counts, action-lifecycle hooks,
semantic negative-policy coverage, absence of weak language, and
superintelligence framing. They guard against regressions toward a softer,
advisory tone without requiring every prohibition to use one exact phrase.
"""

from __future__ import annotations

import re

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


def test_write_diet_remember_only_non_derivable(text: str) -> None:
    """memory_remember must be gated hard on re-derivation from code/git.

    Keeps its name from the binary-gate era; the criterion itself is economic
    now (test_write_guidance_economic_criterion) while the hard ONLY framing,
    the storable categories, and the What-NOT-to-store backstop pinned here
    remain.
    """

    assert "ONLY for what is expensive to re-derive" in text, (
        "write policy must gate memory_remember on the cost of re-deriving, "
        "as a hard ONLY rule rather than advice"
    )
    assert "git history" in text
    lowered = text.lower()
    categories = ["recipes", "pitfalls", "refutations", "external contracts"]
    missing = [c for c in categories if c not in lowered]
    assert not missing, (
        f"the write gate must name the storable categories, missing: {missing}"
    )
    assert "re-derivable" in lowered, (
        "What NOT to store must forbid cheaply re-derivable facts explicitly"
    )


def test_write_diet_procedure_form_triple(text: str) -> None:
    """Reusable know-how must be prescribed as trigger/task_pattern/procedure."""

    idx = text.find("procedure form")
    assert idx != -1, "write policy must prescribe procedure form for know-how"
    window = text[idx : idx + 400]
    for token in ("trigger", "task_pattern", "procedure"):
        assert token in window, (
            f"procedure form must name {token!r} next to the prescription"
        )
    assert "context.task_pattern" in text and "context.procedure_id" in text, (
        "procedure form must point at the real writable context fields"
    )


def test_write_diet_closure_note_not_journal(text: str) -> None:
    """Closing work = one short note of the non-derivable, never a journal."""

    idx = text.find("## Closing out work")
    assert idx != -1, "instructions must keep a Closing out work section"
    closure = text[idx:].split("\n## ")[0].lower()
    assert "closure note" in closure
    assert "diff" in closure and "git history" in closure and "cannot show" in closure, (
        "closure guidance must be the universal criterion — the note carries "
        "only what the diff and git history cannot show"
    )
    for example in ("invariant", "pitfall"):
        assert example in closure, (
            f"the closure criterion must keep {example!r} as a short "
            "parenthetical example"
        )
    assert "done-journal-dump" in text, (
        "the DONE-journal dump must be a NAMED anti-pattern"
    )
    anti_section = text[text.find("## Anti-patterns") :].split("\n## ")[0]
    assert "done-journal-dump" in anti_section, (
        "done-journal-dump must live inside the Anti-patterns section"
    )
    assert "node X DONE" in text, (
        "the journal anti-pattern must show the concrete 'node X DONE' shape"
    )
    journal_entry = anti_section[anti_section.find("done-journal-dump") :].lower()
    assert "recall" in journal_entry, (
        "the journal anti-pattern must state its recall consequence in "
        "universal terms"
    )


def test_closure_redaction_no_what_changed_element(text: str) -> None:
    """The closure note stays the criterion redaction, never an element formula.

    Operator-final (2026-07-31): the note's content derives from the same
    expensive-to-re-derive criterion as the rest of the write policy, so a
    what-changed element — derivable from the diff — must not come back, with
    or without plus-joins.
    """

    idx = text.find("## Closing out work")
    assert idx != -1, "instructions must keep a Closing out work section"
    closure = text[idx:].split("\n## ")[0]
    assert (
        "closure note carrying only what the diff and git history cannot show"
        in closure
    ), "the closure guidance must keep the endorsed criterion redaction"
    assert "(e.g. the invariant to preserve, the pitfall that cost time)" in closure, (
        "the criterion must keep its short parenthetical examples"
    )
    assert "what changed" not in closure.lower(), (
        "the closure guidance must not require a what-changed element — it is "
        "derivable from the diff"
    )


def test_write_guidance_register_no_measurement_artifacts(text: str) -> None:
    """Guidance is universal criteria; diagnosis measurements must not leak in."""

    assert "%" not in text, (
        "no storage/delivery percentages — measurement numbers do not belong "
        "in the instruction register"
    )
    assert " + " not in text, (
        "no plus-joined report formulas — closing-out guidance is a universal "
        "criterion with short parenthetical examples, not a template"
    )
    assert re.search(r"\b\d+(?:\.\d+)?x\b", text) is None, (
        "no redundancy multipliers — measurement ratios do not belong in the "
        "instruction register"
    )


def test_write_guidance_economic_criterion(text: str) -> None:
    """The remember-gate is economic — the cost of re-derivation, not its
    possibility.

    The highest-value stored knowledge IS derivable from code/git, just
    expensively; the gate must show derivable-but-expensive positives and
    pair them, in the same section, with the trivially re-derivable
    execution-journal counter-example.
    """

    idx = text.find("## Write policy")
    assert idx != -1, "instructions must keep a Write policy section"
    policy = text[idx:].split("\n## ")[0]
    lowered = policy.lower()

    assert "expensive to re-derive" in lowered, (
        "the remember-gate must be the COST of re-deriving from code/git "
        "history, not binary derivability"
    )
    assert "cannot be re-derived" not in text and "non-derivable" not in text, (
        "the superseded binary non-derivability phrasing must not survive "
        "anywhere in the instructions"
    )

    positives = [
        "contract smeared across thousands of lines",
        "measurement findings",
        "refuted hypothesis",
    ]
    missing = [p for p in positives if p not in lowered]
    assert not missing, (
        "derivable-but-expensive knowledge must appear as short parenthetical "
        f"positive examples, missing: {missing}"
    )

    assert "execution journal" in lowered and "trivially" in lowered, (
        "the economic gate must pair, in the same section, the counter-example "
        "of an execution journal that re-derives trivially from git history"
    )
    assert "'did x'" in lowered and "'node n done'" in lowered, (
        "the journal counter-example must show its concrete shapes"
    )
    assert "do not store" in lowered, (
        "the trivially re-derivable case must end in a hard prohibition"
    )
