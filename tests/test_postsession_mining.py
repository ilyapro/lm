"""Candidate mining: what fires, what does not, and that quotes stay verbatim.

The miner is the only part of the extractor that reads a transcript, so its
contract is narrow and testable without a model: a candidate exists only when
*all* legs of its shape are present, and the evidence it carries must be a
literal slice of the session it came from -- everything downstream trusts that.
"""

from __future__ import annotations

from typing import Any

import pytest

from living_memory.postsession.mining import (
    CANDIDATE_KINDS,
    DEFAULT_MINING,
    Candidate,
    MiningConfig,
    hard_identifiers,
    identifiers,
    mine,
    mining_stats,
    paragraphs,
    quote_window,
    tool_command,
)
from living_memory.postsession.session import (
    FileMutation,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def _record(**overrides: Any) -> SessionRecord:
    base: dict[str, Any] = {
        "session_key": "claude:test",
        "source": "claude",
        "cli": "claude",
        "path": "/tmp/t.jsonl",
        "transcript_sha256": "a" * 64,
        "record_count": 400,
        "repo": "/repo",
    }
    base.update(overrides)
    return SessionRecord(**base)


def _span(index: int, field: str = "") -> SourceSpan:
    return SourceSpan("claude", "/tmp/t.jsonl", index, field)


def _call(
    ordinal: int,
    index: int,
    command: str,
    result: str | None = None,
    *,
    ok: bool | None = None,
    name: str = "Bash",
) -> ToolCall:
    return ToolCall(
        ordinal=ordinal,
        name=name,
        server=None,
        arguments={"command": command},
        span=_span(index),
        ok=ok,
        result_text=result,
    )


def _mutation(ordinal: int, index: int, path: str, diff: str = "") -> FileMutation:
    return FileMutation(
        ordinal=ordinal,
        path=path,
        kind="update",
        span=_span(index),
        unified_diff=diff or f"--- a/{path}\n+++ b/{path}\n@@\n+changed\n",
    )


def _turn(index: int, role: str, text: str) -> Turn:
    return Turn(index=index, role=role, text=text, at=None, span=_span(index))


RED = "Traceback (most recent call last):\n  File 'src/app.py', line 3\nValueError: bad token\n"
GREEN = "12 passed in 3.20s"


def _red_green_record() -> SessionRecord:
    return _record(
        tool_calls=[
            _call(0, 10, "python -m pytest tests/test_app.py -q", RED, ok=False),
            _call(1, 30, "python -m pytest tests/test_app.py -q", GREEN, ok=True),
        ],
        file_mutations=[_mutation(0, 20, "src/app.py")],
    )


# --------------------------------------------------------------------------
# resolved_failure
# --------------------------------------------------------------------------


def test_resolved_failure_needs_red_then_edit_then_green() -> None:
    candidates = mine(_red_green_record())

    assert [item.kind for item in candidates] == ["resolved_failure"]
    found = candidates[0]
    assert "ValueError: bad token" in found.evidence_text
    assert found.signals["changed_paths"] == ["src/app.py"]
    assert found.span.record_index == 10


def test_resolved_failure_without_an_edit_between_is_not_a_lesson() -> None:
    """Red then green with nothing changed is a flake, not a fix."""

    record = _record(
        tool_calls=[
            _call(0, 10, "python -m pytest -q", RED, ok=False),
            _call(1, 30, "python -m pytest -q", GREEN, ok=True),
        ],
    )
    assert mine(record) == []


def test_resolved_failure_without_a_green_rerun_is_not_a_lesson() -> None:
    record = _record(
        tool_calls=[_call(0, 10, "python -m pytest -q", RED, ok=False)],
        file_mutations=[_mutation(0, 20, "src/app.py")],
    )
    assert [item.kind for item in mine(record)] == []


def test_rerun_matching_tolerates_a_changed_flag() -> None:
    """A re-run routinely gains or drops a flag; byte equality would miss it."""

    record = _record(
        tool_calls=[
            _call(0, 10, "python -m pytest tests/test_app.py -q", RED, ok=False),
            _call(1, 30, "python -m pytest tests/test_app.py -q -x", GREEN, ok=True),
        ],
        file_mutations=[_mutation(0, 20, "src/app.py")],
    )
    assert [item.kind for item in mine(record)] == ["resolved_failure"]


def test_rerun_matching_rejects_an_unrelated_command() -> None:
    record = _record(
        tool_calls=[
            _call(0, 10, "python -m pytest tests/test_app.py -q", RED, ok=False),
            _call(1, 30, "npm run build --workspace web", GREEN, ok=True),
        ],
        file_mutations=[_mutation(0, 20, "src/app.py")],
    )
    assert mine(record) == []


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def test_measurement_requires_the_number_to_be_reused() -> None:
    reused = _record(
        tool_calls=[_call(0, 10, "wc -l src/app.py", "  4821 src/app.py")],
        turns=[_turn(40, "assistant", "4821 lines is over the split threshold, so I split it.")],
    )
    unused = _record(tool_calls=[_call(0, 10, "wc -l src/app.py", "  4821 src/app.py")])

    assert [item.kind for item in mine(reused)] == ["measurement"]
    assert mine(unused) == []


def test_measurement_ignores_file_modes_years_and_epochs() -> None:
    """The first train pass produced 539 candidates, nearly all of these."""

    record = _record(
        tool_calls=[
            _call(0, 10, "git show HEAD", "100644 blob 2026 1755561234567 src/app.py")
        ],
        turns=[
            _turn(40, "assistant", "mode 100644 in 2026 at 1755561234567 as expected"),
        ],
    )
    assert mine(record) == []


def test_measurement_accepts_a_unit_number_from_a_non_measuring_command() -> None:
    record = _record(
        tool_calls=[_call(0, 10, "hostname -I", "latency was 214ms on that host")],
        turns=[_turn(40, "assistant", "214ms is above the 100ms budget so we cache it")],
    )
    assert [item.kind for item in mine(record)] == ["measurement"]


# --------------------------------------------------------------------------
# external_contract
# --------------------------------------------------------------------------


def test_external_contract_needs_a_probe_and_a_documented_reply() -> None:
    record = _record(
        tool_calls=[
            _call(0, 10, "gh api /repos/o/r/pulls", '{"error": "Not Found"}\n404 Not Found'),
        ]
    )
    candidates = mine(record)
    assert [item.kind for item in candidates] == ["external_contract"]
    assert candidates[0].signals["subject"] == "gh"


def test_reading_a_local_file_that_prints_usage_is_not_an_external_contract() -> None:
    """``sed -n '1,80p' scripts/x.py`` is that file quoting itself."""

    record = _record(
        tool_calls=[_call(0, 10, "sed -n '1,80p' scripts/x.py", "Usage: x.py [--flag]")]
    )
    assert mine(record) == []


def test_probe_subject_is_the_segment_that_probed() -> None:
    record = _record(
        tool_calls=[
            _call(0, 10, "cd /tmp && kubectl explain pod --help", "unknown flag: --help"),
        ]
    )
    candidates = mine(record)
    assert candidates and candidates[0].signals["subject"] == "kubectl"


# --------------------------------------------------------------------------
# refuted_hypothesis and operator_correction
# --------------------------------------------------------------------------


def test_refuted_hypothesis_pairs_on_a_shared_anchor() -> None:
    record = _record(
        turns=[
            _turn(10, "assistant", "I expect `storage.find_similar` to read nodes.embedding."),
            _turn(20, "assistant", "Actually `storage.find_similar` reads chunks now."),
        ]
    )
    candidates = mine(record)
    assert [item.kind for item in candidates] == ["refuted_hypothesis"]
    assert "storage.find_similar" in candidates[0].identifiers


def test_refuted_hypothesis_will_not_pair_on_a_bare_number() -> None:
    """Two paragraphs both saying ``2026`` share nothing."""

    record = _record(
        turns=[
            _turn(10, "assistant", "I expect 2026 rows."),
            _turn(20, "assistant", "Actually 2026 was the year, not the count."),
        ]
    )
    assert mine(record) == []


def test_operator_correction_requires_work_after_it() -> None:
    acted_on = _record(
        turns=[
            _turn(5, "assistant", "I will refactor the loader first."),
            _turn(10, "user", "no, that's wrong - recall before you touch anything"),
        ],
        tool_calls=[_call(0, 20, "python -m pytest -q", GREEN, ok=True)],
    )
    ignored = _record(
        turns=[
            _turn(5, "assistant", "I will refactor the loader first."),
            _turn(10, "user", "no, that's wrong - recall before you touch anything"),
        ]
    )
    assert [item.kind for item in mine(acted_on)] == ["operator_correction"]
    assert mine(ignored) == []


# --------------------------------------------------------------------------
# evidence integrity and shared primitives
# --------------------------------------------------------------------------


def test_every_evidence_quote_is_verbatim_transcript_text() -> None:
    """Nothing downstream may invent facts, so quotes must be literal slices."""

    record = _red_green_record()
    sources = [call.result_text or "" for call in record.tool_calls]
    sources += [f"$ {tool_command(call)}" for call in record.tool_calls]

    for candidate in mine(record):
        for item in candidate.evidence:
            assert any(item.quote in text for text in sources), item.quote


def test_ordinals_are_dense_and_transcript_ordered() -> None:
    record = _record(
        tool_calls=[
            _call(0, 10, "python -m pytest -q", RED, ok=False),
            _call(1, 30, "python -m pytest -q", GREEN, ok=True),
            _call(2, 60, "wc -l src/app.py", "  4821 src/app.py"),
        ],
        file_mutations=[_mutation(0, 20, "src/app.py")],
        turns=[_turn(80, "assistant", "4821 lines, so we split it")],
    )
    candidates = mine(record)
    assert [item.ordinal for item in candidates] == list(range(len(candidates)))
    positions = [item.span.record_index for item in candidates]
    assert positions == sorted(positions)


def test_per_kind_cap_protects_the_rare_classes() -> None:
    """A thousand red-green pairs must not crowd out one refuted hypothesis."""

    calls: list[ToolCall] = []
    mutations: list[FileMutation] = []
    for index in range(20):
        base = index * 10
        calls.append(_call(index * 2, base, f"pytest tests/test_{index}.py -q", RED, ok=False))
        mutations.append(_mutation(index, base + 2, f"src/m{index}.py"))
        calls.append(_call(index * 2 + 1, base + 4, f"pytest tests/test_{index}.py -q", GREEN, ok=True))
    record = _record(
        tool_calls=calls,
        file_mutations=mutations,
        turns=[
            _turn(500, "assistant", "I expect `mod.fn` to be pure."),
            _turn(505, "assistant", "Actually `mod.fn` mutates its argument."),
        ],
        record_count=600,
    )
    kinds = [item.kind for item in mine(record)]
    assert kinds.count("resolved_failure") == DEFAULT_MINING.max_per_kind
    assert "refuted_hypothesis" in kinds


def test_mining_is_deterministic() -> None:
    record = _red_green_record()
    assert [item.to_dict() for item in mine(record)] == [
        item.to_dict() for item in mine(record)
    ]


def test_config_can_disable_the_number_reuse_requirement() -> None:
    record = _record(tool_calls=[_call(0, 10, "wc -l src/app.py", "  4821 src/app.py")])
    loose = MiningConfig(require_number_reuse=False)
    assert [item.kind for item in mine(record, config=loose)] == ["measurement"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see src/living_memory/storage.py:587", "src/living_memory/storage.py"),
        ("pass --strict-mcp-config next time", "--strict-mcp-config"),
        ("raised ModuleNotFoundError there", "ModuleNotFoundError"),
        ("took 214ms", "214ms"),
        ("call storage.find_similar()", "storage.find_similar"),
    ],
)
def test_hard_identifiers_find_the_anchor(text: str, expected: str) -> None:
    assert expected in hard_identifiers(text)


def test_identifiers_are_grouped_and_deduplicated() -> None:
    grouped = identifiers("a.py and a.py plus --flag")
    assert grouped["file"] == ("a.py",)
    assert grouped["flag"] == ("--flag",)


def test_quote_window_snaps_to_line_boundaries_and_never_paraphrases() -> None:
    text = "first line\nsecond line with MARK inside\nthird line"
    window = quote_window(text, text.index("MARK"), text.index("MARK") + 4, before=5, after=5)
    assert window in text
    assert "MARK" in window


def test_paragraphs_split_on_blank_lines() -> None:
    assert paragraphs("one\n\ntwo\n\n\nthree") == ("one", "two", "three")


def test_tool_command_handles_string_list_and_path_arguments() -> None:
    string_call = ToolCall(0, "Bash", None, {"command": "ls -la"}, _span(1))
    list_call = ToolCall(1, "exec_command", None, {"cmd": ["ls", "-la"]}, _span(2))
    path_call = ToolCall(2, "Read", None, {"file_path": "src/app.py"}, _span(3))
    assert tool_command(string_call) == "ls -la"
    assert tool_command(list_call) == "ls -la"
    assert tool_command(path_call) == "Read src/app.py"


def test_mining_stats_reports_every_kind_even_at_zero() -> None:
    stats = mining_stats(mine(_red_green_record()))
    assert set(stats["by_kind"]) == set(CANDIDATE_KINDS)
    assert stats["candidates"] == 1
    assert stats["evidence_quotes"] == 4


def test_candidate_evidence_text_joins_every_quote() -> None:
    candidate: Candidate = mine(_red_green_record())[0]
    assert candidate.evidence_text.count("\n") >= len(candidate.evidence) - 1
