"""The missed-insight extractor: mined span -> judge -> gate -> dedup -> op.

This is the stage that closes the write half of Living Memory without asking
the agent to have cooperated. It exists because of a measurement, not a hunch:
on the goal tree ``memory_recall`` is a startup ritual (median position 0.02 of
session length; exactly 1 of 57 sessions recalled mid-session) and
``memory_remember`` is a closing ritual (median 0.96), so an insight discovered
in the middle of a session is either batched until the end or lost. This module
is the safety net under that discipline. It is not a replacement for it: it
runs offline, it proposes rather than writes, and everything it proposes is
attributed to ``extractor:<cli>`` so nobody can mistake it for the agent's own
learning.

The pipeline, and why each stage cannot be merged into its neighbour
-------------------------------------------------------------------
1. :mod:`.mining` -- deterministic, no LLM. The judge never sees a transcript,
   only a mined span, which is what bounds the cost and makes hallucination
   detectable.
2. this module -- one judge call per candidate, asking for ONE fact with its
   WHY and its concrete identifiers, or an explicit refusal. Refusal is a
   first-class answer; most spans hold nothing durable.
3. :mod:`.gate` -- deterministic, and it can veto the judge. A model asked
   "is this an insight?" says yes far too often, and says it in execution-
   journal prose.
4. :mod:`.dedup` -- against memory that already exists, because the server's
   only write-time dedup is exact byte identity inside one scope.

Attribution, and the defect it has to avoid
-------------------------------------------
Every proposed op carries ``agent="extractor:<cli>"``, ``context.task``,
``context.session_id`` and ``context.transport_session_id`` set to the *source*
session's transport id. That last one is deliberate: the live server's
``_with_transport_identity`` only ``setdefault``s the key, so an explicit value
wins and the extracted trace links back to the session it came from rather than
to the extractor's own connection.

There is a measured write defect to avoid while doing this: a malformed call
can serialize the whole context block into ``content`` (107 nodes affected,
measured 2026-08-17). :func:`build_op` asserts structurally that it has not
happened, and the gate's ``context_leak`` rule is the second lock.

Nothing here writes to Living Memory. The runner child owns writes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .dedup import DEFAULT_DEDUP, Deduper
from .gate import DEFAULT_GATE, Gate, GateVerdict, SessionContext
from .judge import (
    Judge,
    JudgeError,
    JudgeInvalidOutput,
    JudgeRefused,
    JudgeUnavailable,
)
from .mining import DEFAULT_MINING, Candidate, MiningConfig, mine
from .session import (
    Evidence,
    ProposedOp,
    ProposedOpError,
    SessionRecord,
)

__all__ = [
    "INSIGHT_SCHEMA",
    "PROMPT_DIR",
    "ExtractionConfig",
    "DEFAULT_EXTRACTION",
    "Outcome",
    "SessionExtraction",
    "build_prompt_task",
    "build_payload",
    "attribution_context",
    "scope_for",
    "build_op",
    "extract_session",
    "extraction_summary",
]

#: Where the judge prompt and its per-kind directives live. Kept as files so a
#: prompt change is a reviewable diff rather than a string edit buried in code.
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

#: Where the per-kind directive is spliced into the base prompt.
KIND_PLACEHOLDER = "{kind_directive}"

#: The judge's output contract. ``additionalProperties: false`` matters: the
#: adapter's validator is strict, so a model that invents a field fails loudly
#: instead of smuggling an unchecked value past the gate.
INSIGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["insight", "none"]},
        "fact": {"type": "string", "maxLength": 1600},
        "why_durable": {"type": "string", "maxLength": 400},
        "identifiers": {
            "type": "array",
            "items": {"type": "string", "maxLength": 200},
            "maxItems": 12,
        },
        "evidence_locator": {"type": "string", "maxLength": 300},
        "reason": {"type": "string", "maxLength": 300},
    },
    "required": ["verdict"],
}


@lru_cache(maxsize=None)
def _prompt_text(name: str) -> str:
    path = PROMPT_DIR / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - a packaging failure, not a run failure
        raise RuntimeError(f"missing judge prompt: {path}") from exc


def build_prompt_task(kind: str) -> str:
    """The base task with the directive for ``kind`` spliced in.

    One base plus five directives rather than five prompts: the refusal rules
    and the anti-journal rules must be identical for every class, and copying
    them five times is how they drift apart.
    """

    directive = _prompt_text(f"kinds/{kind}.md")
    base = _prompt_text("insight_extraction.md")
    if KIND_PLACEHOLDER not in base:  # pragma: no cover - guards a prompt edit
        raise RuntimeError(f"{PROMPT_DIR / 'insight_extraction.md'} lost {KIND_PLACEHOLDER}")
    # Plain substitution, not ``str.format``: the prompt quotes JSON literals
    # like {"verdict": "none"} and format() reads those as field names.
    return base.replace(KIND_PLACEHOLDER, directive)


# --------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Knobs for one extraction run. Tuned on ``train``, reported on ``eval``."""

    mining: MiningConfig = DEFAULT_MINING
    judge_timeout_s: float = 120.0
    #: Quotes handed to the judge per candidate.
    max_evidence: int = 5
    #: Characters kept per quote before the judge adapter bounds the payload.
    max_quote_chars: int = 1600
    #: How many of the session's own writes/recalls are shown so the judge can
    #: avoid restating them.
    max_context_items: int = 6
    max_context_item_chars: int = 240
    #: Judge calls per session. A safety belt on cost, reported when it bites.
    max_judge_calls: int = 12
    #: ``context.task`` stamped on every proposed op.
    task: str = "post-session-extraction/insight-extraction-gate"


DEFAULT_EXTRACTION = ExtractionConfig()


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened to one candidate. Every candidate produces exactly one."""

    kind: str
    ordinal: int
    locator: str
    status: str
    code: str = ""
    detail: str = ""
    fact: str = ""
    why_durable: str = ""
    dedup: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ordinal": self.ordinal,
            "locator": self.locator,
            "status": self.status,
            "code": self.code,
            "detail": self.detail,
            "fact": self.fact,
            "why_durable": self.why_durable,
            "dedup": dict(self.dedup),
            "signals": dict(self.signals),
        }


@dataclass(slots=True)
class SessionExtraction:
    """Everything one session produced, proposals and refusals alike."""

    session_key: str
    source: str
    cli: str
    candidates: int = 0
    judge_calls: int = 0
    outcomes: list[Outcome] = field(default_factory=list)
    ops: list[ProposedOp] = field(default_factory=list)
    judge_stats: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_ops: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "session_key": self.session_key,
            "source": self.source,
            "cli": self.cli,
            "candidates": self.candidates,
            "judge_calls": self.judge_calls,
            "proposed": len(self.ops),
            "outcomes": [item.to_dict() for item in self.outcomes],
        }
        if include_ops:
            data["ops"] = [op.to_dict() for op in self.ops]
        return data


# --------------------------------------------------------------------------
# Payload, attribution, op construction
# --------------------------------------------------------------------------


def build_payload(
    record: SessionRecord,
    candidate: Candidate,
    *,
    config: ExtractionConfig = DEFAULT_EXTRACTION,
) -> dict[str, Any]:
    """The bounded payload for one judge call.

    The judge sees the mined span and never the transcript. ``already_written``
    and ``already_recalled`` are included precisely so a fact the session
    already stored, or was already shown, can be refused by the judge rather
    than having to be caught later by the gate.
    """

    def _clip(text: str) -> str:
        limit = config.max_context_item_chars
        return text[:limit] + ("…" if len(text) > limit else "")

    written = [
        _clip(write.content)
        for write in record.writes
        if write.content
    ][: config.max_context_items]
    recalled = [
        _clip(node.content)
        for recall in record.recalls
        for node in recall.delivered
        if node.content
    ][: config.max_context_items]

    return {
        "candidate_kind": candidate.kind,
        "session": {
            "cli": record.cli,
            "source": record.source,
            "repo": record.repo,
            "started_at": record.started_at,
        },
        "evidence": [
            {"locator": item.locator, "quote": item.quote[: config.max_quote_chars]}
            for item in candidate.evidence[: config.max_evidence]
        ],
        "mined_identifiers": list(candidate.identifiers[:20]),
        "mining_signals": dict(candidate.signals),
        "already_written": written,
        "already_recalled": recalled,
    }


_SCOPE_CLEAN = re.compile(r"[^A-Za-z0-9_.-]+")


def scope_for(record: SessionRecord, *, default: str = "global") -> str:
    """``project:<repo basename>`` when the session had a repository.

    Setting an explicit ``scope`` also protects the op from a subtlety of the
    live server: with no ``scope``, ambient scope resolution would derive
    ``session:<session_id>`` from the ``session_id`` this stage is required to
    stamp, and every extracted trace would land in a per-session scope nobody
    ever recalls from.
    """

    root = record.repo or record.cwd
    if not root:
        return default
    name = Path(root).name.strip()
    if not name:
        return default
    cleaned = _SCOPE_CLEAN.sub("-", name).strip("-")
    return f"project:{cleaned}" if cleaned else default


def source_transport_id(record: SessionRecord) -> str:
    """The transport id of the SOURCE session, or the best identifier we have.

    Preference order is deliberate: the transport id the live server itself
    recorded during the session is the join key the ``recall_events`` rows use,
    and only if the transcript never carried one do we fall back to the CLI's
    own session uuid.
    """

    for value in record.transport_session_ids:
        if value:
            return value
    for write in record.writes:
        if write.transport_session_id:
            return write.transport_session_id
    return record.cli_session_id or ""


def attribution_context(
    record: SessionRecord,
    candidate: Candidate,
    *,
    config: ExtractionConfig = DEFAULT_EXTRACTION,
    scope: str | None = None,
) -> dict[str, Any]:
    """The ``context`` block every proposed remember op carries."""

    return {
        "scope": scope or scope_for(record),
        "agent": f"extractor:{record.cli}",
        "task": config.task,
        "session_id": record.session_key,
        "transport_session_id": source_transport_id(record),
        "source": record.source,
        "source_span": candidate.span.locator(),
        "transcript_sha256": record.transcript_sha256 or "",
        "candidate_kind": candidate.kind,
        "extractor": "postsession.insights",
    }


def _content_carries_context(content: str, context: Mapping[str, Any]) -> str:
    """The first context key that appears *as a key* inside ``content``.

    Asserting against the keys of the context block we are about to attach is
    stronger than pattern matching for a generic shape: it catches the exact
    defect -- the context ending up inside the trace body -- whatever the
    serialization looked like. Prose that merely mentions the word ``scope`` is
    untouched; only ``"scope":`` counts.
    """

    for key in context:
        if f'"{key}":' in content or f"'{key}':" in content:
            return key
    return ""


def build_op(
    record: SessionRecord,
    candidate: Candidate,
    fact: str,
    *,
    config: ExtractionConfig = DEFAULT_EXTRACTION,
    scope: str | None = None,
) -> ProposedOp:
    """A validated ``remember`` proposal, or :class:`ProposedOpError`.

    Structural, not stylistic: the op must name the session it came from, the
    exact bytes of that transcript, the span inside it, and at least one
    verbatim quote -- and ``content`` must be the fact and nothing else.
    """

    context = attribution_context(record, candidate, config=config, scope=scope)
    leaked = _content_carries_context(fact, context)
    if leaked:
        raise ProposedOpError(
            f"remember: content embeds the context key {leaked!r} "
            "(the 2026-08-17 context-leak defect)"
        )
    op = ProposedOp.for_session(
        "remember",
        {"content": fact, "context": context},
        record,
        candidate.span,
        [
            Evidence(quote=item.quote, locator=item.locator)
            for item in candidate.evidence[: config.max_evidence]
        ],
    )
    return op.validate()


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _judge_failure_code(error: JudgeError) -> str:
    if isinstance(error, JudgeRefused):
        return "judge_refused"
    if isinstance(error, JudgeInvalidOutput):
        return "judge_invalid"
    if isinstance(error, JudgeUnavailable):
        return "judge_unavailable"
    return "judge_invalid"


def extract_session(
    record: SessionRecord,
    *,
    judge: Judge,
    gate: Gate | None = None,
    deduper: Deduper | None = None,
    config: ExtractionConfig = DEFAULT_EXTRACTION,
    scope: str | None = None,
    candidates: Sequence[Candidate] | None = None,
) -> SessionExtraction:
    """Run the whole pipeline over one session. Never raises on judge failure.

    A judge that is down must not lose the deterministic work: every candidate
    still gets an outcome with a code, so the run report distinguishes "the
    gate rejected 40 facts" from "the judge was unavailable for 40 spans",
    which are opposite conclusions about the extractor's quality.
    """

    gate = gate or Gate(DEFAULT_GATE)
    deduper = deduper if deduper is not None else Deduper(config=DEFAULT_DEDUP)
    mined = list(candidates if candidates is not None else mine(record, config=config.mining))
    result = SessionExtraction(
        session_key=record.session_key,
        source=record.source,
        cli=record.cli,
        candidates=len(mined),
    )
    if not mined:
        return result

    context = SessionContext.from_record(record)
    resolved_scope = scope or scope_for(record)

    for candidate in mined:
        if result.judge_calls >= config.max_judge_calls:
            result.outcomes.append(
                Outcome(
                    candidate.kind,
                    candidate.ordinal,
                    candidate.span.locator(),
                    "skipped",
                    "budget_exhausted",
                    f"session judge budget of {config.max_judge_calls} calls spent",
                )
            )
            continue

        payload = build_payload(record, candidate, config=config)
        task = build_prompt_task(candidate.kind)
        result.judge_calls += 1
        try:
            answer = judge.judge(
                task, INSIGHT_SCHEMA, payload, timeout_s=config.judge_timeout_s
            )
        except JudgeError as error:
            result.outcomes.append(
                Outcome(
                    candidate.kind,
                    candidate.ordinal,
                    candidate.span.locator(),
                    "rejected",
                    _judge_failure_code(error),
                    str(error)[:200],
                )
            )
            continue

        if str(answer.get("verdict")) != "insight":
            result.outcomes.append(
                Outcome(
                    candidate.kind,
                    candidate.ordinal,
                    candidate.span.locator(),
                    "rejected",
                    "judge_refused",
                    str(answer.get("reason", ""))[:200],
                )
            )
            continue

        fact = str(answer.get("fact") or "").strip()
        why = str(answer.get("why_durable") or "").strip()
        verdict: GateVerdict = gate.check(fact, candidate=candidate, context=context)
        if not verdict.accepted:
            result.outcomes.append(
                Outcome(
                    candidate.kind,
                    candidate.ordinal,
                    candidate.span.locator(),
                    "rejected",
                    verdict.code,
                    verdict.detail,
                    fact=fact,
                    why_durable=why,
                    signals=verdict.signals,
                )
            )
            continue

        seen = deduper.check_within_run(fact)
        if not seen.duplicate:
            seen = deduper.check(fact, scope=resolved_scope)
        if seen.duplicate:
            result.outcomes.append(
                Outcome(
                    candidate.kind,
                    candidate.ordinal,
                    candidate.span.locator(),
                    "rejected",
                    seen.code,
                    f"{seen.channel} match against {seen.node_id}",
                    fact=fact,
                    why_durable=why,
                    dedup=seen.to_dict(),
                )
            )
            continue

        try:
            op = build_op(record, candidate, fact, config=config, scope=resolved_scope)
        except ProposedOpError as error:
            result.outcomes.append(
                Outcome(
                    candidate.kind,
                    candidate.ordinal,
                    candidate.span.locator(),
                    "rejected",
                    "context_leak" if "context key" in str(error) else "judge_invalid",
                    str(error)[:200],
                    fact=fact,
                    why_durable=why,
                )
            )
            continue

        deduper.remember_proposal(f"{record.session_key}#{candidate.ordinal}", fact)
        result.ops.append(op)
        result.outcomes.append(
            Outcome(
                candidate.kind,
                candidate.ordinal,
                candidate.span.locator(),
                "proposed",
                "",
                "",
                fact=fact,
                why_durable=why,
                dedup=seen.to_dict(),
                signals=verdict.signals,
            )
        )

    result.judge_stats = [item.as_dict() for item in getattr(judge, "stats", ())]
    return result


def extraction_summary(results: Iterable[SessionExtraction]) -> dict[str, Any]:
    """Aggregate counts for the run report: yield plus the reason breakdown."""

    sessions = 0
    candidates = 0
    judge_calls = 0
    proposed = 0
    by_kind: dict[str, dict[str, int]] = {}
    reasons: dict[str, int] = {}
    for result in results:
        sessions += 1
        candidates += result.candidates
        judge_calls += result.judge_calls
        proposed += len(result.ops)
        for outcome in result.outcomes:
            bucket = by_kind.setdefault(
                outcome.kind, {"candidates": 0, "proposed": 0, "rejected": 0, "skipped": 0}
            )
            bucket["candidates"] += 1
            if outcome.status == "proposed":
                bucket["proposed"] += 1
            elif outcome.status == "skipped":
                bucket["skipped"] += 1
            else:
                bucket["rejected"] += 1
                reasons[outcome.code] = reasons.get(outcome.code, 0) + 1
    return {
        "sessions": sessions,
        "candidates": candidates,
        "judge_calls": judge_calls,
        "proposed": proposed,
        "proposal_rate": round(proposed / candidates, 4) if candidates else 0.0,
        "by_kind": {name: bucket for name, bucket in sorted(by_kind.items())},
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
    }
