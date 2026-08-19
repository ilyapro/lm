"""The extractor end to end, plus the dedup layer it proposes through.

Everything here runs offline against :class:`FakeJudge`, so the assertions are
about the *pipeline*, not about a model's taste: which stage vetoes what, what
attribution lands on a proposed op, and that a near-duplicate of an existing
node never becomes one. The one thing that is emphatically not tested here is
"did the judge find the right insight" -- there is no answer key for that, and
inventing one would be the shortcut this stage is forbidden to take.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from living_memory.postsession.dedup import (
    DEDUP_COSINE_THRESHOLD,
    DedupConfig,
    Deduper,
    NodeSnapshot,
    NullMemoryIndex,
    StaticMemoryIndex,
    dedup_breakdown,
)
from living_memory.postsession.gate import DEFAULT_GATE, REJECTION_CODES, Gate
from living_memory.postsession.insights import (
    INSIGHT_SCHEMA,
    PROMPT_DIR,
    ExtractionConfig,
    attribution_context,
    build_op,
    build_payload,
    build_prompt_task,
    extract_session,
    extraction_summary,
    scope_for,
    source_transport_id,
)
from living_memory.postsession.judge import (
    FakeJudge,
    JudgeRefused,
    JudgeUnavailable,
    check_schema,
    validation_errors,
)
from living_memory.postsession.mining import CANDIDATE_KINDS, mine
from living_memory.postsession.session import (
    FileMutation,
    ProposedOpError,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
    WriteInteraction,
)

RED = "Traceback (most recent call last):\n  File 'src/app.py', line 3\nValueError: bad token\n"
GREEN = "12 passed in 3.20s"

FACT = (
    "`pytest -q` on top of `addopts = \"-q\"` in pyproject.toml collapses the report "
    "to one line, which is why the ValueError in src/app.py looked like a silent pass."
)


def _span(index: int) -> SourceSpan:
    return SourceSpan("claude", "/home/operator/p/lm/t.jsonl", index)


def _record(**overrides: Any) -> SessionRecord:
    base: dict[str, Any] = {
        "session_key": "claude:abc-123",
        "source": "claude",
        "cli": "claude",
        "path": "/home/operator/p/lm/t.jsonl",
        "transcript_sha256": "b" * 64,
        "record_count": 200,
        "repo": "/home/operator/p/lm",
        "transport_session_ids": ("transport-xyz",),
        "tool_calls": [
            ToolCall(
                0,
                "Bash",
                None,
                {"command": "python -m pytest -q"},
                _span(10),
                ok=False,
                result_text=RED + "pyproject.toml has addopts = \"-q\"\n",
            ),
            ToolCall(
                1,
                "Bash",
                None,
                {"command": "python -m pytest -q"},
                _span(30),
                ok=True,
                result_text=GREEN,
            ),
        ],
        "file_mutations": [
            FileMutation(0, "src/app.py", "update", _span(20), unified_diff="@@\n+ok\n")
        ],
    }
    base.update(overrides)
    return SessionRecord(**base)


def _insight(fact: str = FACT) -> dict[str, Any]:
    return {
        "verdict": "insight",
        "fact": fact,
        "why_durable": "rediscovering this costs a debugging cycle",
        "identifiers": ["pyproject.toml", "src/app.py"],
        "evidence_locator": _span(10).locator(),
    }


REFUSAL = {"verdict": "none", "reason": "nothing durable here"}


def _run(responses: list[Any], *, record: SessionRecord | None = None, **kwargs: Any):
    record = record or _record()
    judge = FakeJudge(responses, validate=False)
    return (
        extract_session(
            record,
            judge=judge,
            gate=Gate(DEFAULT_GATE),
            deduper=kwargs.pop("deduper", Deduper(StaticMemoryIndex([]))),
            **kwargs,
        ),
        judge,
    )


# --------------------------------------------------------------------------
# prompts and schema
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", CANDIDATE_KINDS)
def test_every_candidate_kind_has_a_directive_spliced_into_one_base_prompt(kind: str) -> None:
    task = build_prompt_task(kind)
    directive = (PROMPT_DIR / "kinds" / f"{kind}.md").read_text(encoding="utf-8").strip()

    assert directive in task
    assert "{kind_directive}" not in task
    # the shared rules must be identical for every kind
    assert "MUST appear\nverbatim in the quotes" in task or "verbatim in the quotes" in task
    assert "Refusing is a correct, expected answer" in task


def test_the_output_schema_is_one_the_judge_adapter_can_enforce() -> None:
    check_schema(INSIGHT_SCHEMA)

    assert validation_errors({"verdict": "none"}, INSIGHT_SCHEMA) == []
    assert validation_errors(_insight(), INSIGHT_SCHEMA) == []
    assert validation_errors({"verdict": "maybe"}, INSIGHT_SCHEMA)
    assert validation_errors({"verdict": "none", "extra": 1}, INSIGHT_SCHEMA)


def test_payload_carries_the_span_and_never_the_transcript() -> None:
    record = _record()
    candidate = mine(record)[0]
    payload = build_payload(record, candidate)

    assert [item["locator"] for item in payload["evidence"]]
    assert "ValueError: bad token" in json.dumps(payload)
    # nothing that would let the judge answer from outside the span
    assert "turns" not in payload
    assert "tool_calls" not in payload
    assert set(payload) == {
        "candidate_kind",
        "session",
        "evidence",
        "mined_identifiers",
        "mining_signals",
        "already_written",
        "already_recalled",
    }


def test_payload_shows_what_the_session_already_stored() -> None:
    record = _record(
        writes=[
            WriteInteraction(0, "remember", _span(50), content="already stored this fact")
        ]
    )
    payload = build_payload(record, mine(record)[0])
    assert payload["already_written"] == ["already stored this fact"]


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------


def test_attribution_names_the_extractor_and_the_source_session() -> None:
    record = _record()
    context = attribution_context(record, mine(record)[0])

    assert context["agent"] == "extractor:claude"
    assert context["task"]
    assert context["session_id"] == "claude:abc-123"
    assert context["transport_session_id"] == "transport-xyz"
    assert context["scope"] == "project:lm"
    assert context["source_span"].startswith("claude:")


def test_transport_id_falls_back_through_writes_then_the_cli_session() -> None:
    from_write = _record(
        transport_session_ids=(),
        writes=[WriteInteraction(0, "remember", _span(50), transport_session_id="w-1")],
    )
    from_cli = _record(transport_session_ids=(), cli_session_id="cli-9")

    assert source_transport_id(from_write) == "w-1"
    assert source_transport_id(from_cli) == "cli-9"
    assert source_transport_id(_record(transport_session_ids=(), cli_session_id=None)) == ""


def test_scope_collapses_a_worktree_to_its_project() -> None:
    assert scope_for(_record(repo="/home/operator/p/lm")) == "project:lm"
    assert scope_for(_record(repo=None, cwd=None)) == "global"


def test_an_explicit_scope_is_required_so_session_id_cannot_hijack_it() -> None:
    """``_ambient_scope`` derives ``session:<id>`` when no scope is given."""

    context = attribution_context(_record(), mine(_record())[0])
    assert context["scope"].startswith("project:")
    assert context["session_id"] != context["scope"]


def test_build_op_refuses_content_that_embeds_its_own_context() -> None:
    """The measured 2026-08-17 write defect, asserted structurally."""

    record = _record()
    candidate = mine(record)[0]
    leaked = FACT + ' {"scope": "project:lm", "agent": "x", "task": "y"}'

    with pytest.raises(ProposedOpError, match="context key"):
        build_op(record, candidate, leaked)


def test_build_op_produces_a_validated_remember_op_with_evidence() -> None:
    record = _record()
    candidate = mine(record)[0]
    op = build_op(record, candidate, FACT)

    assert op.kind == "remember"
    assert op.payload["content"] == FACT
    assert op.provenance.session_key == record.session_key
    assert op.provenance.transcript_sha256 == record.transcript_sha256
    assert op.provenance.source_span == candidate.span.locator()
    assert op.evidence and all(item.quote.strip() for item in op.evidence)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


def test_an_accepted_fact_becomes_exactly_one_proposed_op() -> None:
    result, judge = _run([_insight()])

    assert len(result.ops) == 1
    assert [item.status for item in result.outcomes] == ["proposed"]
    assert result.judge_calls == 1
    assert judge.calls[0]["task"].startswith("Extract at most ONE")


def test_a_judge_refusal_is_a_first_class_outcome_not_an_error() -> None:
    result, _ = _run([REFUSAL])

    assert result.ops == []
    assert [(item.status, item.code) for item in result.outcomes] == [
        ("rejected", "judge_refused")
    ]


def test_the_gate_vetoes_the_judge() -> None:
    """A model that answers with an execution journal is overruled here."""

    journal = _insight(
        "Added the retry loop to src/app.py and pyproject.toml, and the suite is green."
    )
    result, _ = _run([journal])

    assert result.ops == []
    assert result.outcomes[0].code == "journal_phrasing"
    assert result.outcomes[0].fact == journal["fact"]


def test_an_invented_identifier_is_caught_after_the_judge() -> None:
    result, _ = _run([_insight("The bug lives in src/living_memory/retrieval.py:120 and is fatal.")])

    assert result.outcomes[0].code == "ungrounded_identifier"


def test_a_judge_outage_is_reported_separately_from_a_gate_rejection() -> None:
    """"Judge down for 40 spans" and "gate rejected 40 facts" are opposites."""

    result, _ = _run([JudgeUnavailable("backend down")])

    assert result.outcomes[0].code == "judge_unavailable"
    assert result.candidates == 1


def test_a_raised_judge_refusal_maps_to_the_refusal_code() -> None:
    result, _ = _run([JudgeRefused("declined")])
    assert result.outcomes[0].code == "judge_refused"


def test_the_per_session_judge_budget_is_enforced_and_reported() -> None:
    record = _record(
        turns=[Turn(0, "assistant", "214ms is over the 100ms budget", None, _span(40))],
        tool_calls=[
            ToolCall(0, "Bash", None, {"command": "hostname -I"}, _span(10), result_text="took 214ms"),
        ],
        file_mutations=[],
    )
    candidates = mine(record)
    assert candidates

    result, _ = _run(
        [REFUSAL] * 10, record=record, config=ExtractionConfig(max_judge_calls=0)
    )
    assert result.judge_calls == 0
    assert {item.code for item in result.outcomes} == {"budget_exhausted"}


def test_a_session_with_no_candidates_costs_nothing() -> None:
    result, judge = _run([], record=_record(tool_calls=[], file_mutations=[]))

    assert (result.candidates, result.judge_calls, result.ops) == (0, 0, [])
    assert judge.calls == []


def test_summary_reports_yield_and_the_reason_breakdown() -> None:
    accepted, _ = _run([_insight()])
    rejected, _ = _run(
        [
            _insight(
                "Added the retry loop to src/app.py and to pyproject.toml, then re-ran "
                "the suite until every check was green at 12 passed."
            )
        ]
    )
    summary = extraction_summary([accepted, rejected])

    assert summary["sessions"] == 2
    assert summary["proposed"] == 1
    assert summary["rejection_reasons"] == {"journal_phrasing": 1}
    assert summary["by_kind"]["resolved_failure"]["candidates"] == 2


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------


def test_dedup_reuses_the_query_anchor_cosine_precedent() -> None:
    from living_memory.query_anchors import ANCHOR_DEDUP_COSINE_THRESHOLD

    assert DEDUP_COSINE_THRESHOLD == ANCHOR_DEDUP_COSINE_THRESHOLD == 0.95


def test_containment_catches_a_near_duplicate_the_server_would_not() -> None:
    """``_insert_node`` only links byte-identical content inside one scope."""

    existing = NodeSnapshot("01EXISTING", FACT, scope="project:lm")
    deduper = Deduper(StaticMemoryIndex([existing]))
    verdict = deduper.check(FACT + " (noticed again today)", scope="project:lm")

    assert verdict.duplicate
    assert (verdict.channel, verdict.node_id) == ("containment", "01EXISTING")
    assert verdict.code == "duplicate_of_existing_node"


def test_an_unrelated_fact_is_not_a_duplicate() -> None:
    deduper = Deduper(StaticMemoryIndex([NodeSnapshot("01OTHER", FACT, scope="project:lm")]))
    verdict = deduper.check(
        "openstack server reboot --hard is the only thing that recreates the "
        "libvirt domain when a hot-attached port stays DOWN.",
        scope="project:lm",
    )
    assert not verdict.duplicate


def test_the_embedding_channel_catches_what_containment_misses() -> None:
    class _Encoder:
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0] if "reboot" in text else [0.0, 1.0]

    node = NodeSnapshot("01VEC", "a paraphrase with no shared rare tokens", scope="project:lm")
    index = StaticMemoryIndex(
        [node], embeddings={"01VEC": [1.0, 0.0]}, encoder=_Encoder()
    )
    verdict = Deduper(index).check(
        "only `openstack server reboot --hard` recreates the libvirt domain",
        scope="project:lm",
    )

    assert verdict.duplicate
    assert (verdict.channel, verdict.node_id) == ("embedding", "01VEC")
    assert verdict.cosine >= DEDUP_COSINE_THRESHOLD


def test_within_run_dedup_collapses_two_spans_that_taught_one_thing() -> None:
    deduper = Deduper(StaticMemoryIndex([]))
    deduper.remember_proposal("s#0", FACT)

    assert deduper.check_within_run(FACT).code == "duplicate_within_run"
    assert not deduper.check_within_run("something else entirely about --flags").duplicate

    deduper.reset()
    assert not deduper.check_within_run(FACT).duplicate


def test_a_duplicate_is_a_rejection_with_its_own_code_not_a_silent_drop() -> None:
    deduper = Deduper(StaticMemoryIndex([NodeSnapshot("01EXISTING", FACT, scope="project:lm")]))
    result, _ = _run([_insight()], deduper=deduper)

    assert result.ops == []
    assert result.outcomes[0].code == "duplicate_of_existing_node"
    assert result.outcomes[0].dedup["node_id"] == "01EXISTING"


def test_no_index_reports_unavailable_rather_than_pretending_everything_is_new() -> None:
    deduper = Deduper(NullMemoryIndex())
    verdict = deduper.check(FACT, scope="project:lm")

    assert (verdict.duplicate, verdict.available) == (False, False)
    assert dedup_breakdown([verdict])["unavailable"] == 1


def test_dedup_looks_across_scopes_because_recall_reads_broad() -> None:
    """A twin sitting in ``global`` is still a duplicate to a reader."""

    elsewhere = NodeSnapshot("01GLOBAL", FACT, scope="global")
    index = StaticMemoryIndex([elsewhere])

    assert Deduper(index).check(FACT, scope="project:lm").duplicate
    scoped = Deduper(index, config=DedupConfig(cross_scope=False))
    assert not scoped.check(FACT, scope="project:lm").duplicate
    assert scoped.check(FACT, scope="global").duplicate


def test_dedup_thresholds_are_configurable_without_touching_the_rules() -> None:
    index = StaticMemoryIndex([NodeSnapshot("01E", FACT, scope="project:lm")])
    strict = Deduper(index, config=DedupConfig(min_containment=0.999))
    assert not strict.check(FACT + " and one more clause about --flags", scope="project:lm").duplicate


# --------------------------------------------------------------------------
# the committed eval artifact
# --------------------------------------------------------------------------

EVAL_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "artifacts" / "post-session" / "insights-eval.json"
)


@pytest.fixture(scope="module")
def eval_report() -> dict[str, Any]:
    if not EVAL_ARTIFACT.is_file():
        pytest.skip(f"{EVAL_ARTIFACT} not built; run scripts/postsession_insights.py --report")
    return json.loads(EVAL_ARTIFACT.read_text(encoding="utf-8"))


def test_the_eval_report_proves_the_reserved_split_was_never_read() -> None:
    """Checked on the bytes, because that is what an auditor would grep."""

    if not EVAL_ARTIFACT.is_file():
        pytest.skip("eval artifact not built")
    text = EVAL_ARTIFACT.read_text(encoding="utf-8")
    report = json.loads(text)

    assert report["splits_read"] == ["eval", "train"]
    assert report["split_policy"]["reserved_split_read"] is False
    assert ("hold" + "out") not in text


def test_the_eval_report_carries_a_machine_readable_rejection_breakdown(
    eval_report: dict[str, Any],
) -> None:
    reasons = eval_report["extraction"]["rejection_reasons"]

    assert reasons, "a run that rejected nothing has no gate"
    assert set(reasons) <= set(REJECTION_CODES) | {"budget_exhausted"}
    assert sum(reasons.values()) + eval_report["extraction"]["proposed"] > 0


def test_every_op_in_the_eval_report_passed_the_structural_invariants(
    eval_report: dict[str, Any],
) -> None:
    checks = eval_report["structural_checks"]

    assert checks["vacuous"] is False, "no ops means the checks proved nothing"
    assert checks["all_pass"] is True
    for name in (
        "attribution_agent",
        "attribution_task",
        "attribution_session_id",
        "attribution_transport_session_id",
        "content_free_of_context",
        "identifiers_grounded_in_evidence",
        "with_evidence",
        "with_provenance",
    ):
        assert checks[name] == checks["ops"], name


def test_the_gate_discrimination_transfers_from_train_to_eval(
    eval_report: dict[str, Any],
) -> None:
    """The generalization claim, held to the numbers the run published."""

    block = eval_report["gate_calibration"]["generalization"]

    assert block["eval_journal_rejected"] >= 0.5
    assert block["eval_organic_rejected"] <= 0.15
    assert block["eval_gap"] >= 0.4
    assert abs(block["transfer_loss"]) <= 0.15


def test_the_eval_report_exercised_every_stage(eval_report: dict[str, Any]) -> None:
    assert eval_report["mining"]["candidates"] > 0
    assert eval_report["judge"]["calls"] > 0
    assert eval_report["dedup"]["index_available"] is True
    assert eval_report["proposed_ops_total"] > 0
