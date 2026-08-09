"""Imperative-tone contract for the LM protocol across its delivery channels.

Supersedes the single-channel contract that pinned everything inside
``_server_instructions``. Measured live 2026-08-09 (claude-code 2.1.217):
clients clip MCP server instructions at 2048 chars ("… [truncated]"), and
GigaCode drops them entirely — so 74% of the old 7.9k-char protocol (every
AFTER hook, all anti-patterns, the write policy) never reached any agent.
The protocol therefore ships over two channels:

- **instructions** (the only text guaranteed visible before any tool is
  considered): identity framing, the three laws, the bootstrap directive
  telling deferred-tool sessions to load the LM tools, and scope policy.
  Budget: 2048 chars — clients compare string length, not bytes.
- **tool descriptions** (reach every client that can call tools at all,
  including GigaCode): the detailed binding hooks. Budget: 1024 chars each,
  the limit OpenAI-compatible clients (codex) enforce per function
  description.

The old 7.9k-char protocol is not restated anywhere in full (operator
decision 2026-08-09: a full-text side channel is waste). Instead the
compression works by abstraction: every dropped example or rationale must
be subsumed by a broader phrase that still carries its point — see
test_legacy_protocol_abstract_coverage for the audited mapping.

These tests pin the imperative register of both channels, the budgets, and
the abstract coverage — guarding against regressions toward advisory tone,
against silent overflow past what clients actually deliver, and against
nuance quietly dying in compression.
"""

from __future__ import annotations

import re

import pytest

from living_memory.server import (
    _CONSOLIDATE_DESCRIPTION,
    _RECALL_DESCRIPTION,
    _REMEMBER_DESCRIPTION,
    _TEACH_DESCRIPTION,
    _server_instructions,
)

# Client-side delivery limits, measured empirically — see module docstring.
MAX_INSTRUCTION_CHARS = 2048
MAX_DESCRIPTION_CHARS = 1024

DESCRIPTIONS = {
    "memory_recall": _RECALL_DESCRIPTION,
    "memory_remember": _REMEMBER_DESCRIPTION,
    "memory_teach": _TEACH_DESCRIPTION,
    "memory_consolidate": _CONSOLIDATE_DESCRIPTION,
}


@pytest.fixture(scope="module")
def instructions() -> str:
    return _server_instructions("global")


@pytest.fixture(scope="module")
def union(instructions: str) -> str:
    return "\n\n".join([instructions, *DESCRIPTIONS.values()])


# ── Channel budgets ─────────────────────────────────────────────────────────


def test_instructions_within_client_budget(instructions: str) -> None:
    size = len(instructions)
    assert size <= MAX_INSTRUCTION_CHARS, (
        f"instructions are {size} chars; clients clip at "
        f"{MAX_INSTRUCTION_CHARS} and everything past the limit is invisible "
        "to every agent — cut, do not raise the budget"
    )


def test_each_description_within_client_budget() -> None:
    oversize = {
        name: len(text)
        for name, text in DESCRIPTIONS.items()
        if len(text) > MAX_DESCRIPTION_CHARS
    }
    assert not oversize, (
        f"tool descriptions exceed the {MAX_DESCRIPTION_CHARS}-char "
        f"per-function client limit: {oversize} — cut, do not raise the budget"
    )


# ── Instructions channel: identity, laws, bootstrap ─────────────────────────


def test_superintelligence_framing_in_opening(instructions: str) -> None:
    """First paragraph must frame agent+LM as one cognitive system."""

    head = instructions[:600].lower()
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


def test_instructions_carry_the_three_laws(instructions: str) -> None:
    """The always-visible channel must state all three laws imperatively."""

    assert "You MUST recall BEFORE you act" in instructions
    assert "You MUST remember at the moment of insight" in instructions
    assert "You MUST teach the moment a belief changes" in instructions
    assert "supersedes edge" in instructions, (
        "the teach law must explain WHY: only teach creates the supersedes edge"
    )


def test_instructions_bootstrap_directive(instructions: str) -> None:
    """Deferred-tool sessions must be told to load the LM tools up front.

    In current claude-code every MCP tool is deferred behind a search step
    even for a single-tool server, so descriptions are a pull channel; the
    instructions are the only push channel left and must trigger the pull.
    """

    for tool in DESCRIPTIONS:
        assert tool in instructions, (
            f"bootstrap directive must name {tool} so agents know what to load"
        )
    lowered = instructions.lower()
    assert "defers" in lowered or "deferred" in lowered
    assert "load the living memory tools now" in lowered, (
        "the bootstrap directive must be an immediate imperative, not advice"
    )
    assert "as protocol" in lowered


def test_default_scope_substituted(instructions: str) -> None:
    assert "Your default scope is global" in instructions
    custom = _server_instructions("project:custom-scope")
    assert "Your default scope is project:custom-scope" in custom


# ── Imperative register, both channels ──────────────────────────────────────


def test_must_counts(instructions: str, union: str) -> None:
    assert instructions.count("MUST") >= 4, (
        "the three laws plus the bootstrap directive each carry a MUST"
    )
    assert union.count("MUST") >= 8, (
        "every protocol-bearing description carries at least one MUST"
    )


def test_negative_policy_is_hard_and_concrete(union: str) -> None:
    lowered = union.lower()
    hard_negative_markers = ["do not", "not count", "never"]
    found = [m for m in hard_negative_markers if m in lowered]
    assert len(found) >= 2, (
        "negative policy must use hard prohibitive language such as DO NOT "
        f"or NEVER; found only {found}"
    )

    weak_negative_phrases = [
        "try not to",
        "consider not",
        "avoid if possible",
        "you may skip",
    ]
    found_weak = [p for p in weak_negative_phrases if p in lowered]
    assert not found_weak, (
        f"negative policy must be mandatory rather than advisory: {found_weak}"
    )


def test_no_weak_language(union: str) -> None:
    weak = ["you can also", "consider ", "might want", "feel free"]
    lowered = union.lower()
    found = [w for w in weak if w in lowered]
    assert not found, f"weak/advisory language found: {found}"


def test_no_measurement_artifacts(union: str) -> None:
    """Guidance is universal criteria; diagnosis measurements must not leak in."""

    assert "%" not in union, (
        "no storage/delivery percentages — measurement numbers do not belong "
        "in the instruction register"
    )
    assert " + " not in union, (
        "no plus-joined report formulas — guidance is a universal criterion "
        "with short parenthetical examples, not a template"
    )
    assert re.search(r"\b\d+(?:\.\d+)?x\b", union) is None, (
        "no redundancy multipliers — measurement ratios do not belong in the "
        "instruction register"
    )


def test_named_anti_patterns_all_present(union: str) -> None:
    """All six named failure modes survive the move onto the tools."""

    named_failures = [
        "mkdir-vs-API",
        "shell-watchdog-vs-loop",
        "grep-before-recall",
        "silent-correction",
        "lookup-table-as-learning",
        "done-journal-dump",
    ]
    missing = [name for name in named_failures if name not in union]
    assert not missing, f"named anti-patterns lost in compression: {missing}"


def test_cross_project_transfer_explicit(union: str) -> None:
    lowered = union.lower()
    assert "cross-project" in lowered
    assert "across scopes" in lowered or "across projects" in lowered, (
        "cross-project transfer must instruct broadening recall across scopes"
    )
    assert "omit" in lowered and "scope" in lowered, (
        "read-broad guidance must say to omit scope when investigating"
    )


# ── memory_recall description: the BEFORE hooks ─────────────────────────────


def test_recall_hooks_concrete_triggers() -> None:
    """The compressed BEFORE hooks must keep concrete, recognisable triggers."""

    text = _RECALL_DESCRIPTION
    assert "You MUST recall BEFORE" in text
    triggers = [
        "editing any file",
        "writes state",
        "git push",
        "DROP",
        "grep",
        "watchdog",
        "architecture",
        "cold start",
        "'how does X work'",
    ]
    missing = [t for t in triggers if t not in text]
    assert not missing, f"BEFORE triggers lost in compression: {missing}"


def test_recall_default_and_economics() -> None:
    text = _RECALL_DESCRIPTION.lower()
    assert "when uncertain, recall" in text
    assert "grep only after recall returns nothing" in text, (
        "grep-before-recall must keep its ordering rule, not just its name"
    )


def test_recall_schema_activation() -> None:
    text = _RECALL_DESCRIPTION
    assert "level:schema" in text
    assert "follow it literally" in text, (
        "schema results must stay binding procedures, not suggestions"
    )


def test_recall_content_ref_refetch_path() -> None:
    text = _RECALL_DESCRIPTION
    assert "content_ref" in text and "memory_lookup" in text, (
        "agents must keep the re-fetch path for non-full deliveries"
    )


# ── memory_remember description: write policy ───────────────────────────────


def test_remember_economic_gate() -> None:
    """The remember-gate is economic — the cost of re-derivation, not its
    possibility."""

    text = _REMEMBER_DESCRIPTION
    lowered = text.lower()
    assert "ONLY for what is" in text and "expensive to re-derive" in lowered, (
        "write policy must gate memory_remember on the cost of re-deriving, "
        "as a hard ONLY rule rather than advice"
    )
    assert "cost, not" in lowered and "possibility" in lowered
    assert "cannot be re-derived" not in text and "non-derivable" not in text, (
        "the superseded binary non-derivability phrasing must not return"
    )

    categories = ["recipes", "pitfalls", "refutations", "external contracts"]
    missing = [c for c in categories if c not in lowered]
    assert not missing, f"storable categories lost: {missing}"

    positives = [
        "contract smeared across thousands of lines",
        "measurement findings",
    ]
    missing = [p for p in positives if p not in lowered]
    assert not missing, (
        f"derivable-but-expensive positive examples lost: {missing}"
    )

    assert "execution journal" in lowered and "trivially" in lowered, (
        "the economic gate must pair with the execution-journal counter-example"
    )
    assert "'did x'" in lowered and "'node n done'" in lowered, (
        "the journal counter-example must keep its concrete shapes"
    )
    assert "do not store" in lowered, (
        "the trivially re-derivable case must end in a hard prohibition"
    )


def test_remember_timing_is_moment_of_insight() -> None:
    text = _REMEMBER_DESCRIPTION.lower()
    assert "at the moment of insight" in text
    assert "before your next action" in text
    assert "deferred remembers are usually lost" in text, (
        "the timing law must state the loss consequence, not just the timing"
    )


def test_remember_verified_facts_only() -> None:
    text = _REMEMBER_DESCRIPTION.lower()
    assert "verified facts only" in text
    forbidden = [
        "routine actions",
        "copies of code",
        "speculation",
        "unverified plans",
    ]
    missing = [f for f in forbidden if f not in text]
    assert not missing, f"forbidden storage categories lost: {missing}"


def test_remember_closure_note_not_journal() -> None:
    """Closing work = one short note of the non-derivable, never a journal.

    Operator-final (2026-07-31): the note derives from the same
    expensive-to-re-derive criterion, so a what-changed element — derivable
    from the diff — must not come back.
    """

    text = _REMEMBER_DESCRIPTION
    assert (
        "closure note carrying only what the diff and git history cannot show"
        in text
    ), "the closure guidance must keep the endorsed criterion redaction"
    assert "(e.g. the invariant to preserve, the pitfall that cost time)" in text, (
        "the criterion must keep its short parenthetical examples"
    )
    assert "what changed" not in text.lower(), (
        "the closure guidance must not require a what-changed element — it is "
        "derivable from the diff"
    )
    assert "done-journal-dump" in text, (
        "the DONE-journal dump must stay a NAMED anti-pattern next to the "
        "closure rule"
    )


def test_remember_redirects_corrections_to_teach() -> None:
    text = _REMEMBER_DESCRIPTION
    assert "NEVER" in text and "memory_teach" in text, (
        "remember must redirect corrections to teach in hard terms"
    )


# ── memory_teach description: the correction loop ───────────────────────────


def test_teach_correction_loop() -> None:
    text = _TEACH_DESCRIPTION
    assert "You MUST teach the moment a belief changes" in text
    assert "trace_id" in text
    assert "supersedes edge" in text
    assert "UPDATE/CORRECTION wording does NOT count" in text, (
        "the remember-as-correction failure mode must stay pinned verbatim"
    )
    assert "silent-correction" in text
    assert "outranking" in text, (
        "teach must explain the ranking consequence of a missing supersedes edge"
    )


# ── Abstract coverage of the legacy 7.9k protocol ───────────────────────────


def test_legacy_protocol_abstract_coverage(instructions: str, union: str) -> None:
    """Every point of the legacy protocol is carried — restored verbatim
    where the phrase itself was load-bearing, or subsumed by a broader
    abstraction where it was an example.

    Audited mapping (2026-08-09) of what compression did to each legacy
    nuance. Items asserted below were restored because no abstraction
    carried their point; the rest are intentionally absent, each subsumed
    by a phrase that generalises it:

    - mkdir/echo>/sed -i/tee, 'goals state' examples  -> "Bash that writes
      state ... recall the domain noun, not mkdir/echo"
    - curl POST / gh pr merge / ALTER / 'git push origin master'
      -> "destructive or external commands (rm, git push, DROP, migrate
      ... verb plus target)"
    - "domain answers are invented instead of retrieved", "act blind"
      -> "Without recall you act blind — inventing what memory already
      holds" (now covers actions AND answers)
    - "grows ONLY through consistent cycles" -> the without-recall /
      without-remember / without-teach triad
    - "the most valuable knowledge IS derivable, just expensive to
      rediscover", "recon findings", "a refuted hypothesis" -> "The gate:
      cost, not possibility" plus the positive examples kept
    - "find the precedent before designing from scratch" -> "(recall
      cross-project first)"
    - "'file X exports Y, not Z'", "function names, config keys", "Short
      beats long" -> "One concrete fact per trace — paths, identifiers,
      the WHY"
    - "reference file paths instead" -> "never ... copies of code" plus
      "paths, identifiers"
    - "Scope is inferred when omitted; explicit is better" -> "Write with
      a specific scope"
    - "omit scope or pass a wildcard" -> "omit scope"
    - "schemas exist to prevent repeated failure modes" -> "binding
      procedure — follow it literally"
    - root cause / working fix trigger enumeration -> "a non-obvious
      discovery"
    - transport_session_id auto-stamping detail -> server behaviour needing
      no agent action; the actionable half is the task/agent/session_id
      line asserted below.

    Do NOT delete assertions here to make a budget cut pass — find a
    stronger abstraction instead.
    """

    # Restored because the phrase itself changes behaviour:
    assert "'I wish I had known this earlier'" in _REMEMBER_DESCRIPTION, (
        "the remember self-test is the strongest known nudge against "
        "under-remembering"
    )
    assert "drowns the rare real lesson in recall" in _REMEMBER_DESCRIPTION, (
        "the journal ban must keep its systemic WHY — junk degrades every "
        "future recall, not just this trace"
    )
    assert "the WHY" in _REMEMBER_DESCRIPTION, (
        "trace-quality craft: a fact without its why is half a fact"
    )
    assert "conventions" in _RECALL_DESCRIPTION, (
        "file-edit recall covers conventions, not only past edits and "
        "rejections"
    )
    assert "inventing what memory already holds" in instructions, (
        "the blind/invent merge is the universal consequence of skipping "
        "recall"
    )
    assert "link work across sessions" in instructions, (
        "correlation identity: task/agent/session_id in context is the "
        "actionable half of the legacy block"
    )
    for token in ("`task`", "`agent`", "`session_id`"):
        assert token in instructions

    # The abstractions the mapping above relies on must themselves stay:
    for anchor in (
        "not mkdir/echo",
        "verb plus target",
        "cost, not possibility",
        "recall cross-project first",
        "One concrete fact per trace",
        "binding procedure",
        "non-obvious discovery",
    ):
        assert anchor in union, (
            f"abstraction {anchor!r} carries legacy nuances — losing it "
            "silently drops everything mapped onto it"
        )


# ── memory_consolidate description: promotion path ──────────────────────────


def test_consolidate_procedure_form_triple() -> None:
    """Reusable know-how must be prescribed as trigger/task_pattern/procedure."""

    text = _CONSOLIDATE_DESCRIPTION
    idx = text.find("procedure form")
    assert idx != -1, "consolidate must prescribe procedure form for know-how"
    window = text[idx : idx + 400]
    for token in ("trigger", "task_pattern", "procedure"):
        assert token in window, (
            f"procedure form must name {token!r} next to the prescription"
        )
    assert "context.task_pattern" in text and "context.procedure_id" in text, (
        "procedure form must point at the real writable context fields"
    )
    assert "level:schema" in text
    assert "re-invented" in text, (
        "lookup-table-as-learning must keep its consequence, not just its name"
    )
