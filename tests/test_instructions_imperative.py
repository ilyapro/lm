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

One part of the instructions channel is not written here at all: the recall
map section, composed at ``initialize`` from what earlier recalls persisted.
Everything above pins the *static* half; the block at the end of this module
pins the populated one to the same standard — the same budget across scope
names, the same register bans run through the same scanners, the same pinned
phrases still present underneath it, and — the point of a personalized
channel — a section that is a function of real history rather than a
constant.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from living_memory.instructions_map import HEADING, MAX_SECTION_CHARS
from living_memory.server import (
    _CONSOLIDATE_DESCRIPTION,
    _RECALL_DESCRIPTION,
    _REMEMBER_DESCRIPTION,
    _TEACH_DESCRIPTION,
    _instructions_with_map,
    _server_instructions,
)
from living_memory.storage import MemoryStore

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


@pytest.fixture(scope="module")
def channels(instructions: str) -> dict[str, str]:
    """Every protocol-bearing text, by the channel that delivers it."""

    return {"instructions": instructions, **DESCRIPTIONS}


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


# ── The budget funds protocol only ──────────────────────────────────────────

# A capped channel spends every char on something. An agent following the
# protocol cannot act on a mechanism that is off unless an operator flips it,
# so each char describing one is taken from a trigger that would have changed
# behaviour. Measured cost of the last regression (2026-08-15):
# _RECALL_DESCRIPTION spent 43 of its 1024 chars on "Experimental repeat
# gating is default-off.", and the opening sentence — the highest-value
# position in the whole channel — had been truncated to "Retrieve memories."
# to pay for it. Operator machinery belongs in docs and docstrings, which
# have no budget; this channel carries protocol.
NON_DEFAULT_MACHINERY_PATTERNS = {
    "experimental": r"experiment(?:al|s|ing)?\b",
    "default-off": r"default[-\s]off\b|\boff by default\b|\bdisabled by default\b",
    "opt-in": r"\bopt(?:s|ed)?[-\s]?in\b",
    "beta": r"\bbeta\b",
    "feature-flag": r"\bfeature[-\s]?(?:flag|gate)",
    "env-var machinery": r"\bLM_[A-Z0-9_]+|\benv(?:ironment)?[-\s]?var",
}

# One planted advert per pattern: the scanner must be able to fail.
PLANTED_ADVERTS = {
    "experimental": "Experimental repeat gating collapses repeated recalls.",
    "default-off": "The fingerprint gate is default-off.",
    "opt-in": "Trailing-stub drop is a strict opt-in.",
    "beta": "The causal walker ships in beta.",
    "feature-flag": "Cross-scope admission sits behind a feature flag.",
    "env-var machinery": "Set LM_RECALL_REPEAT_GATING=1 to enable gating.",
}

# Protocol wording that must NOT trip the scanner — a ban worded too widely
# (on bare "default", "in", "gate") would silently force real protocol out of
# the channel, which is the very failure it exists to prevent.
LEGITIMATE_PROTOCOL_WORDING = (
    "Your default scope is global.",
    "Default: when uncertain, recall.",
    "Retrieve memories by text, vector, and graph signals.",
    "Read broad: omit scope to transfer across scopes.",
    "A level:schema result is a binding procedure — follow it literally.",
    "Non-full results carry a content_ref — refetch via memory_lookup.",
    "recipes, pitfalls, refutations, external contracts, measurement findings",
    "You and Living Memory form ONE cognitive system; LM supplies durable memory.",
    "entering anything new — a session, task, message, thought, or direction",
    "Store universal rules in global so they can become schemas.",
)


def _machinery_hits(text: str) -> dict[str, list[str]]:
    return {
        label: found
        for label, pattern in NON_DEFAULT_MACHINERY_PATTERNS.items()
        if (found := re.findall(pattern, text, flags=re.IGNORECASE))
    }


def test_protocol_channels_advertise_no_non_default_machinery(
    channels: dict[str, str],
) -> None:
    """No protocol-bearing text may spend budget on non-default machinery."""

    offenders = {
        name: hits for name, text in channels.items() if (hits := _machinery_hits(text))
    }
    assert not offenders, (
        f"protocol channels advertise machinery an agent cannot act on: "
        f"{offenders} — the channel is capped at "
        f"{MAX_DESCRIPTION_CHARS} chars per description and "
        f"{MAX_INSTRUCTION_CHARS} for instructions, and every char it spends "
        "here is taken from a binding trigger; document the mechanism in "
        "docs/ or a docstring instead"
    )


def test_machinery_scanner_fires_on_planted_adverts() -> None:
    """The ban must be able to fail — one control per pattern."""

    assert set(PLANTED_ADVERTS) == set(NON_DEFAULT_MACHINERY_PATTERNS), (
        "every banned-machinery pattern needs a planted advert proving it "
        "fires, or the ban can rot into a no-op"
    )
    for label, advert in PLANTED_ADVERTS.items():
        assert label in _machinery_hits(advert), (
            f"pattern {label!r} did not fire on {advert!r} — the ban would "
            "not have caught the advert it exists to catch"
        )


def test_machinery_scanner_spares_legitimate_protocol_wording() -> None:
    """The ban must not fire on protocol wording that merely looks similar."""

    false_positives = {
        phrase: hits
        for phrase in LEGITIMATE_PROTOCOL_WORDING
        if (hits := _machinery_hits(phrase))
    }
    assert not false_positives, (
        f"the banned-machinery terms fire on legitimate protocol wording: "
        f"{false_positives} — narrow the pattern; a ban that hits real "
        "protocol pushes protocol out of the channel"
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
    assert "recall the task at hand" in lowered, (
        "loading and the first recall are one prescribed step — otherwise "
        "the first-call slot goes to the task and recall never happens "
        "(cold-start miss measured 2026-08-09 in two sessions)"
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
    """Every named failure mode survives — command-free (operator decision
    2026-08-09): grep-before-recall, mkdir-vs-API and shell-watchdog-vs-loop
    are retired as names and subsumed by universal classes —
    world-before-memory (memory before searching/measuring the world) and
    the inventing-a-mechanism clause of the state trigger."""

    named_failures = [
        "world-before-memory",
        "silent-correction",
        "lookup-table-as-learning",
        "done-journal-dump",
    ]
    missing = [name for name in named_failures if name not in union]
    assert not missing, f"named anti-patterns lost in compression: {missing}"
    for retired in ("grep-before-recall", "mkdir-vs-API", "shell-watchdog-vs-loop"):
        assert retired not in union, (
            f"retired command-bound name {retired!r} crept back — its class "
            "already covers it"
        )


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


def test_recall_opens_by_naming_the_retrieval_signals() -> None:
    """The opening sentence is the highest-value position in the channel.

    It is what a client shows in a collapsed tool list, so it must say what
    recall actually does — retrieval over text, vector and graph signals —
    rather than be squeezed down to a stub to pay for something else.
    """

    opener = _RECALL_DESCRIPTION.split(".", 1)[0]
    missing = [s for s in ("text", "vector", "graph") if s not in opener]
    assert not missing, (
        f"the opening sentence {opener!r} does not name the retrieval "
        f"signals {missing} — it is the first thing every client shows"
    )
    size = len(_RECALL_DESCRIPTION)
    assert size < MAX_DESCRIPTION_CHARS, (
        f"memory_recall is {size} chars, flush against the "
        f"{MAX_DESCRIPTION_CHARS}-char client limit: keep headroom so the "
        "next protocol wording fix does not have to truncate a trigger"
    )


def test_recall_hooks_universal_action_classes() -> None:
    """The BEFORE hooks are command-free action classes (operator decision
    2026-08-09): every trigger names a universal shape of action, so unlisted
    concrete commands (curl, ALTER, a benchmark, a UI click) are covered by
    construction instead of falling through an enumeration."""

    text = _RECALL_DESCRIPTION
    assert "You MUST recall BEFORE acting" in text
    classes = [
        "changing any artifact",
        "creating or mutating state",
        "irreversible or outward-facing",
        "pulling knowledge from the world",
        "measuring",
        "choosing a design or approach",
        "entering anything new",
        # thought/direction were promoted out of this enumeration into the
        # mid-work law (operator directive 2026-08-19) — pinned by
        # test_recall_standing_epistemic_trigger as "every new turn of
        # thought", not here.
    ]
    missing = [c for c in classes if c not in text]
    assert not missing, f"universal action classes lost: {missing}"
    for command in ("grep", "mkdir", "git push", "DROP", "Bash", "rm "):
        assert command not in text, (
            f"command-bound trigger {command!r} crept back — triggers must "
            "stay universal action classes"
        )


def test_recall_default_and_economics() -> None:
    text = _RECALL_DESCRIPTION.lower()
    assert "when uncertain, recall" in text
    assert "only after recall returns nothing" in text, (
        "world-before-memory must keep its ordering rule, not just its name"
    )
    assert "memory first" in text, (
        "the world-vs-memory ordering must be stated as a hard default "
        "(cold-start miss measured 2026-08-09: a fresh session re-measured "
        "the 2048 limit and answered falsely while the fact sat in memory)"
    )


def test_recall_standing_epistemic_trigger() -> None:
    """Every new turn of thought is a trigger — proactive, not remedial.

    Field measurement 2026-08-19 (30 tree sessions on the deployment host):
    recall was a session-start ritual — median position 0.02 of session
    length, 1 of 57 calls mid-session — because every trigger was bound to
    an action boundary, and a linear pass crosses its biggest boundary once,
    at the start. Operator directive 2026-08-19: the mid-work trigger is the
    *turn of thought itself* — ask what memory holds nearby before the turn
    hardens, so related knowledge shapes the reasoning instead of being
    missed. Failure states (stuck, surprised) are kept only as lagging
    markers meaning the recall is overdue — they must never become the
    primary trigger again, because by then the related fact has already
    been missed once. 'If only I knew' is the prospective mirror of
    remember's retrospective self-test ('I wish I had known this earlier').
    The thought/direction member of the old entering-anything-new
    enumeration lives here now, promoted from list item to law."""

    text = _RECALL_DESCRIPTION
    assert "MUST recall MID-WORK" in text, (
        "the mid-work law must stay a MUST, not advice"
    )
    assert "at every new turn of thought" in text, (
        "the trigger is the turn of thought itself — proactive coverage of "
        "related knowledge, not a remedy for being stuck"
    )
    assert "nothing related is missed" in text, (
        "the law must state its purpose: completeness of related knowledge, "
        "not unblocking"
    )
    assert "stuck or surprised means overdue" in text, (
        "failure states stay as lagging markers, subordinate to the "
        "turn-of-thought trigger"
    )
    assert "'if only I knew' means recall NOW" in text, (
        "the prospective self-test is the strongest known nudge against "
        "grinding on without asking memory"
    )


def test_instructions_law_one_covers_stuck(instructions: str) -> None:
    """The always-visible channel must extend recall past uncertainty to
    stuckness — the epistemic state agents actually reach mid-work."""

    assert "When uncertain or stuck, recall" in instructions


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

    - mkdir/echo>/sed -i/tee, 'goals state' examples  -> "creating or
      mutating state ... recall the domain concept before inventing one"
      (also subsumes the watchdog/poll-loop hook)
    - rm / curl POST / gh pr merge / DROP / ALTER / 'git push origin
      master' -> "any irreversible or outward-facing step (recall the
      action plus its target)"
    - grep/rg/find, fresh measurements, probes -> "pulling knowledge from
      the world — searching, measuring, probing (memory first ...
      anti-pattern: world-before-memory)"
    - CLAUDE.md's "recall for every new thought/direction/message"
      -> message stays under "entering anything new — session, task,
      message"; thought and direction were promoted to the mid-work law
      "at every new turn of thought" (operator directive 2026-08-19; the
      client file now only draws attention)
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
        "before inventing one",
        "action plus target",
        "world-before-memory",
        "cost, not possibility",
        "recall cross-project",
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


# ── The recall map section: the one populated part of the channel ───────────
#
# Everything above this line pins text written in server.py. The map section
# is composed at ``initialize`` from what earlier recalls persisted, so it is
# the only part of the instructions an operator cannot read off the source —
# and the only part whose content changes between sessions. It therefore gets
# the same contract, not a weaker one.
#
# The pin set is not restated here. These tests re-run the *existing* contract
# functions above against the populated text, so a phrase pinned once is
# pinned in both halves of the channel by construction: adding an assertion
# above automatically extends the populated contract too, and no literal can
# drift between two copies because there is one copy.


#: The budget applies to the WHOLE string, and the static text ends with the
#: default-scope line — so its length grows one-for-one with the scope name
#: and "global" is the *shortest* case, not a representative one. Measured
#: 2026-08-20, before the compression that made room for the map: "global"
#: was 2042 of 2048 while ``project:custom-scope`` was already 2056 and
#: ``project:a-fairly-long-project-name`` 2070 — both over budget, and both
#: invisible to a budget test that only ever measured "global".
SCOPE_LENGTHS = (
    "global",
    "project:lm",
    "project:custom-scope",
    "project:a-fairly-long-project-name",
    "project:a-fairly-long-scope-name-for-headroom",
)

#: A history whose map fills the section: enough clusters to exhaust the label
#: slots, with ``more`` set so the breadth marker is on too. This is what the
#: channel looks like for work that keeps coming back to the same subsystems.
FULL_HISTORY = (
    ("instructions channel", 9),
    ("recall map clustering", 7),
    ("delivery diet", 5),
    ("transport identity", 4),
    ("consolidation", 3),
    ("latency budget", 2),
)

#: Two histories with no label, and no substring of a label, in common. The
#: anti-hardcoding pin needs both directions: each section must name its own
#: labels *and* none of the other's, which a constant string satisfies in
#: neither direction and a "some section is present" check satisfies in both
#: for the wrong reason.
HISTORY_ONE = (("deploy recipes", 3), ("bundle layout", 2))
HISTORY_TWO = (("transport identity", 5), ("latency budget", 4))

#: Fixture names of the channel texts the contracts above consume.
CHANNEL_FIXTURES = frozenset({"instructions", "union", "channels"})

#: Below this the collection has rotted — an import error, a rename, or a
#: signature change would otherwise leave the populated channel pinned by an
#: empty loop that passes.
MIN_CHANNEL_CONTRACTS = 12


def _map_payload(
    *labelled: tuple[str, int], pool: int = 60, more: int = 0
) -> dict[str, Any]:
    """A persisted ``recall_map`` payload, shaped as ``RecallMap.to_dict``."""

    clusters = [
        {
            "label": label,
            "count": count,
            "medoid": {"node_id": f"node-{index}", "example": f"an example of {label}"},
            "ask_hint": f"what does memory hold about {label}",
        }
        for index, (label, count) in enumerate(labelled)
    ]
    payload: dict[str, Any] = {
        "clusters": clusters,
        "pool": pool,
        "covered": sum(count for _, count in labelled),
    }
    if more:
        payload["more"] = more
    return payload


def _store_with_history(db_path: Path, *maps: dict[str, Any]) -> MemoryStore:
    """A store that has already delivered ``maps``, oldest call first.

    Written through ``record_recall_event`` rather than through the recall
    tool: what the instructions channel reads is the persisted history, and
    going through the builder would couple these tests to clustering
    thresholds without exercising one line of the composition under test.
    """

    store = MemoryStore(db_path)
    for index, payload in enumerate(maps):
        store.record_recall_event(
            query=f"what does memory hold about topic {index}",
            scope="global",
            recall_map=payload,
        )
    return store


def _populated(store: MemoryStore, scope: str) -> str:
    """The instructions this store composes, asserted to actually carry a map.

    Every test below that claims something about a populated channel goes
    through here, so none of them can quietly pass against an empty section.
    """

    text = _instructions_with_map(store, scope)
    assert HEADING in text, (
        "the history seeded for this test composed no map section — the "
        "populated-channel contract would pass vacuously against the static "
        "text it is supposed to be stronger than"
    )
    return text


def _section_of(text: str, scope: str) -> str:
    """Everything the splice added past the static text."""

    static = _server_instructions(scope)
    assert text.startswith(static), (
        "the map section must be additive and tail-placed; the static text is "
        "no longer this text's prefix"
    )
    return text[len(static) :]


# Snapshot taken HERE, at import time, before a single test below is defined:
# ``globals()`` at this point holds exactly the contracts above. Collecting
# later would sweep these tests into their own pin set and recurse.
_STATIC_CHANNEL_CONTRACTS: tuple[tuple[str, Any], ...] = tuple(
    (name, obj)
    for name, obj in sorted(globals().items())
    if name.startswith("test_")
    and callable(obj)
    and (params := frozenset(inspect.signature(obj).parameters))
    and params <= CHANNEL_FIXTURES
)


def _call_channel_contract(contract: Any, instructions_text: str) -> None:
    """Run one collected contract against ``instructions_text``.

    The populated instructions are substituted for the ``instructions``
    fixture, and the ``union``/``channels`` views are rebuilt on top of them
    exactly as their fixtures do — so the register scanners run over the map
    section, and the presence pins are checked in its company.
    """

    supplied: dict[str, Any] = {
        "instructions": instructions_text,
        "union": "\n\n".join([instructions_text, *DESCRIPTIONS.values()]),
        "channels": {"instructions": instructions_text, **DESCRIPTIONS},
    }
    contract(**{p: supplied[p] for p in inspect.signature(contract).parameters})


def _run_channel_contracts(instructions_text: str) -> list[str]:
    """Every channel contract above, re-run against ``instructions_text``.

    Returns the contracts that ran, so a caller can prove the loop was not
    empty.
    """

    assert len(_STATIC_CHANNEL_CONTRACTS) >= MIN_CHANNEL_CONTRACTS, (
        f"only {len(_STATIC_CHANNEL_CONTRACTS)} channel contracts collected, "
        f"expected at least {MIN_CHANNEL_CONTRACTS} — the populated channel "
        "is being pinned by an almost-empty loop"
    )
    ran = []
    for name, contract in _STATIC_CHANNEL_CONTRACTS:
        _call_channel_contract(contract, instructions_text)
        ran.append(name)
    return ran


@pytest.fixture(scope="module")
def map_valve_open() -> Iterator[None]:
    """``LM_RECALL_MAP=0`` in the ambient environment must not silence a test.

    The valve is a real rollback and its off-state is pinned in
    tests/test_instructions_refresh.py. Here it would turn every populated
    assertion into an assertion about the static text.
    """

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("LM_RECALL_MAP", raising=False)
        yield


@pytest.fixture(scope="module")
def populated_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MemoryStore]:
    store = _store_with_history(
        tmp_path_factory.mktemp("populated") / "memory.sqlite3",
        _map_payload(*HISTORY_ONE),
        _map_payload(*FULL_HISTORY, more=4),
    )
    try:
        yield store
    finally:
        store.close()


@pytest.fixture(scope="module")
def populated_instructions(
    populated_store: MemoryStore, map_valve_open: None
) -> str:
    return _populated(populated_store, "global")


# ── Budget, with the section actually in the string ─────────────────────────


@pytest.mark.parametrize("scope", SCOPE_LENGTHS)
def test_populated_instructions_within_client_budget(
    populated_store: MemoryStore, map_valve_open: None, scope: str
) -> None:
    """The budget test above measures ``_server_instructions("global")``.

    That is the shortest text this channel ever serves: no map, and the
    scope name that costs the fewest chars. What clients actually receive is
    this — a real scope name and whatever the persisted history composed.
    """

    text = _populated(populated_store, scope)
    size = len(text)
    assert size <= MAX_INSTRUCTION_CHARS, (
        f"instructions for scope {scope!r} with a populated map section are "
        f"{size} chars; clients clip at {MAX_INSTRUCTION_CHARS} and the map "
        "sits at the tail, so the clip eats it first — cut, do not raise the "
        "budget"
    )


@pytest.mark.parametrize("scope", SCOPE_LENGTHS)
def test_budget_holds_for_the_largest_section_composable(scope: str) -> None:
    """Headroom for any history, not just the one seeded here.

    The section's content is whatever a session happened to recall, so a
    budget that holds for one sample history and not for a full one fails in
    the field rather than in this suite. ``MAX_SECTION_CHARS`` is the
    composer's own cap, so this is the worst case that can ever be spliced.
    """

    text = _server_instructions(scope, map_section="m" * MAX_SECTION_CHARS)
    size = len(text)
    assert size <= MAX_INSTRUCTION_CHARS, (
        f"a full {MAX_SECTION_CHARS}-char map section takes scope {scope!r} "
        f"to {size} chars, past the {MAX_INSTRUCTION_CHARS} clip: the channel "
        "has no headroom for a history richer than today's — cut the static "
        "text or the section cap, do not raise the budget"
    )


# ── Zero displacement, and the same register ────────────────────────────────


def test_populated_channel_displaces_no_pinned_phrase(
    populated_instructions: str,
) -> None:
    """Every contract above, re-run with the map section in the string.

    This is the whole zero-displacement guarantee and the register guarantee
    at once: the presence pins say the protocol survives underneath the
    section, and the ban scanners — ``%``, ``" + "``, ``\\d+x``, the machinery
    patterns, the weak-language list — now run over a text that includes it.
    Because they are the same function objects, the section's register cannot
    drift from the static text's: tightening one tightens both.
    """

    ran = _run_channel_contracts(populated_instructions)
    assert len(ran) >= MIN_CHANNEL_CONTRACTS, ran


def test_channel_contracts_actually_scan_the_map_section() -> None:
    """The re-run must be able to fail on the section — one control per ban.

    A pin set that passes no matter what is spliced pins nothing. Each poison
    below is placed in the section and nowhere else, so the only way it can be
    caught is by reading the section — and each is asserted against the
    *named* contract that owns its ban, so a control cannot be quietly
    satisfied by some unrelated assertion failing first.
    """

    owners = {
        "test_protocol_channels_advertise_no_non_default_machinery": tuple(
            PLANTED_ADVERTS.values()
        ),
        "test_no_measurement_artifacts": (
            "memory also holds: 43% of storage",
            "memory also holds: what changed + the invariant",
            "memory also holds: a 3x smaller candidate pool",
        ),
        "test_no_weak_language": (
            "consider recalling deploys before touching them",
        ),
    }
    collected = dict(_STATIC_CHANNEL_CONTRACTS)
    for name, poisons in owners.items():
        contract = collected.get(name)
        assert contract is not None, (
            f"{name} is no longer collected — the ban it owns would stop "
            "reaching the map section without a single test turning red"
        )
        for poison in poisons:
            text = _server_instructions("global", map_section=poison)
            assert poison in text
            with pytest.raises(AssertionError):
                _call_channel_contract(contract, text)
            # And the whole re-run, which is what the populated contract uses.
            with pytest.raises(AssertionError):
                _run_channel_contracts(text)


def test_map_section_advertises_no_non_default_machinery(
    populated_instructions: str,
) -> None:
    """The machinery ban, aimed at the section alone.

    Redundant with the re-run above by design: the section is composed from
    user data — context values, file paths, the wording of past queries — so
    it is the one part of this channel that can acquire a banned phrase
    without anyone editing a line of source. Scanned on its own so a failure
    names the section instead of the whole text.
    """

    section = _section_of(populated_instructions, "global")
    hits = _machinery_hits(section)
    assert not hits, (
        f"the composed map section advertises machinery an agent cannot act "
        f"on: {hits} — labels come from stored data and reach the channel "
        "unedited; the composer must drop such a label, not ship it"
    )


# ── Degradation, and the anti-hardcoding pin ────────────────────────────────


@pytest.mark.parametrize("scope", SCOPE_LENGTHS)
def test_empty_history_degrades_to_the_static_text(
    tmp_path: Path, map_valve_open: None, scope: str
) -> None:
    """No persisted map, no section — byte-identical, not "almost".

    A heading over nothing, or a stray blank line, is a permanent one-line
    tax on every session of every fresh install for zero information.
    """

    with _store_with_history(tmp_path / "memory.sqlite3") as store:
        composed = _instructions_with_map(store, scope)

    assert composed == _server_instructions(scope), (
        "instructions composed from an empty history must be byte-identical "
        "to the static text"
    )
    assert HEADING not in composed


def test_section_reflects_its_own_history_and_no_other(
    tmp_path: Path, map_valve_open: None
) -> None:
    """The section is a function of persisted history, not a constant.

    Two histories, two stores, one scope. Separate stores are the point: the
    composer merges a *window* of recent rows, so writing both histories into
    one store would make "none of the other's labels" a statement about
    ordering. Isolated, it is a statement about where the content comes from.

    A fixed expected string passes neither direction of this; a check that
    only asserts "a section is present" passes both directions against a
    hardcoded one.
    """

    with _store_with_history(
        tmp_path / "one.sqlite3", _map_payload(*HISTORY_ONE)
    ) as store_one, _store_with_history(
        tmp_path / "two.sqlite3", _map_payload(*HISTORY_TWO)
    ) as store_two:
        first = _populated(store_one, "global")
        second = _populated(store_two, "global")

    assert first != second, (
        "two different persisted histories composed the same instructions — "
        "the section is not reading history"
    )
    # And the difference lives entirely in the section.
    static = _server_instructions("global")
    assert first.startswith(static) and second.startswith(static)

    one, two = _section_of(first, "global"), _section_of(second, "global")
    for label, count in HISTORY_ONE:
        assert f"{label}({count})" in one, (
            f"the section composed from history one does not name {label!r} "
            "with the count that history recorded"
        )
        assert label not in two, (
            f"{label!r} belongs to history one and reached a section composed "
            "from history two — the section carries content from somewhere "
            "other than the store it was composed from"
        )
    for label, count in HISTORY_TWO:
        assert f"{label}({count})" in two, (
            f"the section composed from history two does not name {label!r} "
            "with the count that history recorded"
        )
        assert label not in one


def test_section_follows_the_history_as_it_grows(
    tmp_path: Path, map_valve_open: None
) -> None:
    """Same store, one more delivery: the channel personalizes to the latest.

    The complement of the two-store pin — that one covers *where* the content
    comes from, this one covers *when*. A section cached into a constant at
    first composition would pass that test and fail this one.
    """

    with _store_with_history(
        tmp_path / "memory.sqlite3", _map_payload(*HISTORY_ONE)
    ) as store:
        before = _section_of(_populated(store, "global"), "global")
        store.record_recall_event(
            query="what does memory hold about transports",
            scope="global",
            recall_map=_map_payload(*HISTORY_TWO),
        )
        after = _section_of(_populated(store, "global"), "global")

    assert after != before, (
        "a newly delivered map did not change the section — the channel is "
        "not tracking history, it is repeating a snapshot"
    )
    newest, older = HISTORY_TWO[0][0], HISTORY_ONE[0][0]
    assert newest not in before
    assert after.index(newest) < after.index(older), (
        "the newest map must lead: this channel personalizes a session from "
        "what memory has lately been mapping, not from the oldest thing it "
        "still remembers"
    )
