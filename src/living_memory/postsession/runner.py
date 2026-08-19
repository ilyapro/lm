"""The post-session extraction runner: proposals -> gate -> bounded writes.

This is the one component of the stage allowed to write, and everything it
writes goes through the live MCP interface (:mod:`.client`) — never the SQLite
file. The pipeline it drives is built entirely from the sibling children::

    SessionRecord --(insights.extract_session)-->    remember proposals
    SessionRecord --(corrections.detect_corrections)--> teach proposals
    SessionRecord --(evidence.proposed_attestations)--> attest proposals
                              |
                    gate verdict check (this module)
                              |
            attribution -> budget -> ledger -> LM-side dedup
                              |
                 memory_remember / memory_teach / memory_attest

Three properties this module owns, each provable:

**Halt on gate failure.** ``artifacts/post-session/usage-baseline.json``
pre-registers the consumption bar and names this file as its enforcer
(``pre_registered_bar.enforced_by``). :func:`evaluate_gate` reads a verdict
produced by ``scripts/counterfactual_consumption.py`` and applies the bar's own
decision procedure — the relative floor against a concurrently measured organic
control first, the protocol-keyed absolute floor as the anti-degenerate
backstop, ``INCONCLUSIVE`` (never PASS) when the control itself cannot
discriminate. The floor is read from ``absolute_floor_by_protocol``; the
deprecated scalar is never read, exactly as the artifact instructs. Anything
but PASS means exit :data:`EXIT_GATE` and zero writes. Before the first
extracted cohort exists there is nothing to measure, so a self-validated
organic-control verdict authorizes a *bounded* bootstrap
(``control_bootstrap``): once the ledger shows more than
``bootstrap_write_allowance`` nodes ever written, a control-only verdict stops
authorizing and the runner demands a measurement of its own cohort — an
extractor whose traces are not recalled must signal and stop, not keep
writing.

**Idempotency.** Every executed op lands in a per-``(transcript sha256, op
fingerprint)`` ledger under the state dir; a re-run over the same transcript
proposes the same ops and skips them all. The ledger is a cache, not the
proof: if it is lost, the write path still asks the *server* what exists —
``memory_attest`` is idempotent per (event, evidence) on the server's own
ledger, and remember/teach proposals are checked against live memory through
:class:`~.client.McpMemoryIndex` (recall + full-content lookup), by extraction
identity (same transcript sha + source span) and by IDF containment. Exact
content dedup on the server is byte-identity within one scope only, which is
why the runner does its own near-duplicate check before every write.

**Attribution.** :func:`enforce_attribution` stamps every write context with
``agent="extractor:<cli>"``, ``task``, ``session_id``,
``transport_session_id`` — the *source* session's transport id, which wins over
the extractor's own connection because the server only ``setdefault``s it —
and ``source_transcript_sha256``. Corrections go only through
``memory_teach``; nothing here mutates another agent's trace.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .client import ClientError, LiveMemoryClient, McpMemoryIndex
from .corrections import (
    DEFAULT_CONFIG as DEFAULT_CORRECTIONS,
    DetectionResult,
    DetectorConfig,
    detect_corrections,
)
from .dedup import DEFAULT_DEDUP, DedupConfig, Deduper, NullMemoryIndex, _containment
from .evidence import proposed_attestations
from .gate import DEFAULT_GATE, Gate
from .insights import (
    DEFAULT_EXTRACTION,
    ExtractionConfig,
    SessionExtraction,
    extract_session,
    source_transport_id,
)
from .judge import canonical_json, default_redactor
from .session import ProposedOp, SessionRecord

__all__ = [
    "DEFAULT_BASELINE",
    "DEFAULT_STATE_DIR",
    "DEFAULT_VERDICT",
    "EXIT_GATE",
    "EXIT_OK",
    "EXIT_USAGE",
    "ExtractionRunner",
    "GateDecision",
    "Ledger",
    "OpDecision",
    "RunnerConfig",
    "SessionProposals",
    "build_report",
    "enforce_attribution",
    "evaluate_gate",
    "op_fingerprint",
]

EXIT_OK = 0
EXIT_USAGE = 2
#: The gate did not authorize this run: verdict failed, inconclusive, missing
#: or stale. The runner exits with this code having written NOTHING.
EXIT_GATE = 3

RUNNER_VERSION = "runner/1"

DEFAULT_BASELINE = Path("artifacts/post-session/usage-baseline.json")
DEFAULT_VERDICT = Path("artifacts/post-session/counterfactual-selfcheck.json")
DEFAULT_STATE_DIR = Path("~/.local/state/living-memory/postsession")

#: How many nodes a control-only (bootstrap) verdict may ever authorize. Once
#: the ledger shows more, only a measured extracted-cohort verdict passes.
DEFAULT_BOOTSTRAP_ALLOWANCE = 200


# --------------------------------------------------------------------------
# gate: the pre-registered bar, enforced
# --------------------------------------------------------------------------


def _parse_instant(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return None, f"{path} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path} does not hold a JSON object"
    return data, ""


@dataclass(frozen=True)
class GateDecision:
    """The verdict that did or did not authorize this run."""

    status: str  # PASS | FAIL | INCONCLUSIVE | MISSING | STALE
    mode: str = ""  # control_bootstrap | extracted_vs_control | ""
    reasons: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def authorized(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "authorized": self.authorized,
            "mode": self.mode,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


def _control_admissibility(
    selfcheck: Mapping[str, Any],
    bar: Mapping[str, Any],
    pinned: str,
    label: str,
) -> list[str]:
    """Why the organic control cannot serve as a reference, empty if it can."""

    problems: list[str] = []
    rule = bar.get("inconclusive_rule") or {}
    min_nodes = int(rule.get("min_control_nodes") or 0)
    min_rates = rule.get("min_control_rate_by_protocol") or {}
    min_rate = float(min_rates.get(pinned) or 0.0)

    kind = str(selfcheck.get("cohort_kind") or "")
    if not kind.startswith("organic"):
        problems.append(f"{label}: cohort_kind {kind!r} is not an organic control")
    if not selfcheck.get("passed"):
        problems.append(
            f"{label}: the harness did not self-validate on this control "
            "(selfcheck.passed is false) — it may not judge anything"
        )
    nodes = int(selfcheck.get("cohort_nodes") or 0)
    if nodes < min_nodes:
        problems.append(
            f"{label}: control cohort holds {nodes} nodes, below "
            f"min_control_nodes {min_nodes} — it cannot discriminate"
        )
    rate = selfcheck.get("counterfactual_rate")
    if rate is None or float(rate) < min_rate:
        problems.append(
            f"{label}: control rate {rate!r} falls below "
            f"min_control_rate_by_protocol[{pinned}]={min_rate}"
        )
    return problems


def _protocol_fields(report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = report.get("protocol") or {}
    meta = report.get("meta") or {}
    return {
        "as_of": meta.get("as_of"),
        "replay_since": protocol.get("replay_since"),
        "min_containment": protocol.get("min_containment"),
        "within_days": protocol.get("within_days"),
    }


def evaluate_gate(
    baseline_path: Path,
    verdict_path: Path,
    control_path: Path | None = None,
    *,
    now: datetime | None = None,
    max_age_days: float = 14.0,
    written_nodes_total: int = 0,
    bootstrap_allowance: int = DEFAULT_BOOTSTRAP_ALLOWANCE,
) -> GateDecision:
    """Apply the pre-registered bar to a counterfactual consumption verdict.

    ``written_nodes_total`` is what the ledger says this extractor has ever
    written; it bounds how long a control-only verdict keeps authorizing.
    The decision procedure is the one published in the baseline artifact —
    see the module docstring. Never raises: every failure is a decision.
    """

    now = now or datetime.now(UTC)
    details: dict[str, Any] = {
        "baseline": str(baseline_path),
        "verdict": str(verdict_path),
        "control": str(control_path) if control_path else None,
        "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_age_days": max_age_days,
    }

    baseline, problem = _load_json(baseline_path)
    if baseline is None:
        return GateDecision("MISSING", reasons=(f"pre-registered bar: {problem}",), details=details)
    bar = baseline.get("pre_registered_bar")
    if not isinstance(bar, dict):
        return GateDecision(
            "MISSING",
            reasons=(f"{baseline_path} carries no pre_registered_bar block",),
            details=details,
        )

    pinned = str(bar.get("pinned_protocol") or "")
    floors = bar.get("absolute_floor_by_protocol") or {}
    relative_floor = bar.get("relative_floor")
    if not pinned or pinned not in floors or relative_floor is None:
        return GateDecision(
            "MISSING",
            reasons=(
                "the bar is unreadable: pinned_protocol, relative_floor and "
                "absolute_floor_by_protocol[pinned] are all required "
                "(the deprecated scalar absolute_floor is never read)",
            ),
            details=details,
        )
    relative_floor = float(relative_floor)
    absolute_floor = float(floors[pinned])
    details.update(
        {
            "pinned_protocol": pinned,
            "relative_floor": relative_floor,
            "absolute_floor": absolute_floor,
        }
    )

    verdict, problem = _load_json(verdict_path)
    if verdict is None:
        return GateDecision("MISSING", reasons=(f"gate verdict: {problem}",), details=details)
    details["verdict_sha256"] = hashlib.sha256(
        verdict_path.read_bytes()
    ).hexdigest()

    meta = verdict.get("meta") or {}
    if meta.get("artifact") != "counterfactual_consumption":
        return GateDecision(
            "MISSING",
            reasons=(
                f"{verdict_path} is not a scripts/counterfactual_consumption.py "
                f"report (meta.artifact={meta.get('artifact')!r})",
            ),
            details=details,
        )
    generated_at = _parse_instant(meta.get("generated_at_utc"))
    details["verdict_generated_at"] = meta.get("generated_at_utc")
    if generated_at is None:
        return GateDecision(
            "STALE",
            reasons=(f"{verdict_path} carries no parseable meta.generated_at_utc",),
            details=details,
        )
    age = now - generated_at
    details["verdict_age_days"] = round(age.total_seconds() / 86400.0, 2)
    if age > timedelta(days=max_age_days) or age < timedelta(days=-1):
        return GateDecision(
            "STALE",
            reasons=(
                f"verdict is {details['verdict_age_days']} days old, over the "
                f"{max_age_days}-day limit — re-run "
                "scripts/counterfactual_consumption.py before writing",
            ),
            details=details,
        )

    selfcheck = verdict.get("selfcheck")
    if not isinstance(selfcheck, dict):
        return GateDecision(
            "MISSING", reasons=(f"{verdict_path} carries no selfcheck block",), details=details
        )

    replay_rule = bar.get("replay_set_rule") or {}
    validated_by = str(replay_rule.get("validated_by") or "")
    cohort_kind = str(selfcheck.get("cohort_kind") or "")
    details["cohort_kind"] = cohort_kind

    # ---- bootstrap: the verdict IS the organic control ---------------------
    if cohort_kind.startswith("organic") and control_path is None:
        problems = _control_admissibility(selfcheck, bar, pinned, "control")
        replay_since = str((_protocol_fields(verdict) or {}).get("replay_since") or "")
        if validated_by and replay_since < validated_by:
            problems.append(
                f"control replayed traffic from {replay_since!r}, before the "
                f"validated arm {validated_by!r} pinned by replay_set_rule"
            )
        details["organic_rate"] = selfcheck.get("counterfactual_rate")
        if problems:
            return GateDecision("INCONCLUSIVE", "control_bootstrap", tuple(problems), details)
        details["written_nodes_total"] = written_nodes_total
        details["bootstrap_allowance"] = bootstrap_allowance
        if written_nodes_total > bootstrap_allowance:
            return GateDecision(
                "FAIL",
                "control_bootstrap",
                (
                    f"this extractor has already written {written_nodes_total} nodes "
                    f"on control-only verdicts (allowance {bootstrap_allowance}); "
                    "measure the extracted cohort with "
                    "scripts/counterfactual_consumption.py and pass its report as "
                    "--gate-verdict with the organic control as --gate-control",
                ),
                details,
            )
        return GateDecision(
            "PASS",
            "control_bootstrap",
            (
                "no extracted cohort exists to measure yet; the self-validated "
                "organic control authorizes a bounded bootstrap write",
            ),
            details,
        )

    # ---- extracted cohort against a concurrent organic control ------------
    if control_path is None:
        return GateDecision(
            "MISSING",
            "extracted_vs_control",
            (
                f"verdict cohort_kind {cohort_kind!r} needs an organic control "
                "measured under the identical protocol; pass it via --gate-control",
            ),
            details,
        )
    control, problem = _load_json(control_path)
    if control is None:
        return GateDecision(
            "MISSING", "extracted_vs_control", (f"organic control: {problem}",), details
        )
    control_check = control.get("selfcheck")
    if not isinstance(control_check, dict):
        return GateDecision(
            "MISSING",
            "extracted_vs_control",
            (f"{control_path} carries no selfcheck block",),
            details,
        )

    ours, theirs = _protocol_fields(verdict), _protocol_fields(control)
    mismatched = [key for key in ours if ours[key] != theirs[key]]
    details["protocol"] = ours
    if mismatched:
        return GateDecision(
            "INCONCLUSIVE",
            "extracted_vs_control",
            (
                "the two cohorts were not measured under the identical protocol "
                f"({', '.join(f'{k}: {ours[k]!r} vs {theirs[k]!r}' for k in mismatched)}); "
                "the comparison is void whatever the two rates are",
            ),
            details,
        )
    replay_since = str(ours.get("replay_since") or "")
    problems = _control_admissibility(control_check, bar, pinned, "control")
    if validated_by and replay_since < validated_by:
        problems.append(
            f"replay set starts {replay_since!r}, before the validated arm "
            f"{validated_by!r} pinned by replay_set_rule"
        )
    organic_rate = float(control_check.get("counterfactual_rate") or 0.0)
    extracted_rate = float(selfcheck.get("counterfactual_rate") or 0.0)
    details["organic_rate"] = organic_rate
    details["extracted_rate"] = extracted_rate
    details["required_relative"] = round(relative_floor * organic_rate, 4)
    if problems:
        return GateDecision("INCONCLUSIVE", "extracted_vs_control", tuple(problems), details)

    reasons: list[str] = []
    if extracted_rate < relative_floor * organic_rate:
        reasons.append(
            f"extracted rate {extracted_rate} is below {relative_floor} x organic "
            f"({relative_floor * organic_rate:.4f})"
        )
    if extracted_rate < absolute_floor:
        reasons.append(
            f"extracted rate {extracted_rate} is below the "
            f"absolute_floor_by_protocol[{pinned}] backstop {absolute_floor}"
        )
    if reasons:
        return GateDecision("FAIL", "extracted_vs_control", tuple(reasons), details)
    return GateDecision(
        "PASS",
        "extracted_vs_control",
        (
            f"extracted {extracted_rate} >= {relative_floor} x organic "
            f"{organic_rate} and >= floor {absolute_floor}",
        ),
        details,
    )


# --------------------------------------------------------------------------
# idempotency ledger
# --------------------------------------------------------------------------


def op_fingerprint(op: ProposedOp) -> str:
    """A stable identity for one op, independent of run-time context noise.

    Only what the write *does* is hashed: the fact and its scope, the corrected
    node and its correction, the attested event and its canonical evidence
    digest (the same digest the server's own attestation ledger keys on).
    Attribution context is excluded on purpose — a runner restarted with a
    different task label must still recognise its earlier write.
    """

    payload = op.payload
    if op.kind == "remember":
        context = payload.get("context") or {}
        identity: dict[str, Any] = {
            "kind": "remember",
            "content": str(payload.get("content") or ""),
            "scope": str(context.get("scope") or ""),
        }
    elif op.kind == "teach":
        correction = payload.get("correction")
        identity = {
            "kind": "teach",
            "trace_id": str(payload.get("trace_id") or ""),
            "correction": correction
            if isinstance(correction, str)
            else canonical_json(correction, indent=None),
        }
    elif op.kind == "attest":
        items = [str(item) for item in payload.get("evidence") or []]
        digest = hashlib.sha256(
            "\x1f".join(item.strip() for item in items if item.strip()).encode("utf-8")
        ).hexdigest()
        try:  # keep byte-parity with the server's own ledger key when possible
            from living_memory.attestation import canonical_evidence

            _, digest = canonical_evidence(items)
        except Exception:  # noqa: BLE001 - unbounded evidence still fingerprints
            pass
        identity = {
            "kind": "attest",
            "recall_event_id": str(payload.get("recall_event_id") or ""),
            "evidence_sha256": digest,
        }
    else:  # pragma: no cover - ProposedOp.validate rejects unknown kinds
        identity = {"kind": op.kind, "payload": canonical_json(payload, indent=None)}
    return hashlib.sha256(
        canonical_json(identity, indent=None).encode("utf-8")
    ).hexdigest()


class Ledger:
    """Append-only JSONL of executed ops, keyed (transcript sha256, fingerprint).

    Corrupt or truncated lines are skipped on load: the ledger is an
    optimization and a bootstrap counter, and the write path behind it re-checks
    against the live server anyway. Thread-safe because proposal workers may
    consult it concurrently.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.path = state_dir / "ledger.jsonl"
        self._seen: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            sha = str(entry.get("transcript_sha256") or "")
            fingerprint = str(entry.get("fingerprint") or "")
            if sha and fingerprint:
                self._seen[(sha, fingerprint)] = entry

    def __len__(self) -> int:
        return len(self._seen)

    def has(self, transcript_sha256: str, fingerprint: str) -> bool:
        with self._lock:
            return (transcript_sha256, fingerprint) in self._seen

    def written_nodes_total(self) -> int:
        """Nodes this extractor has ever created (remember + teach entries)."""

        with self._lock:
            return sum(
                1 for entry in self._seen.values() if entry.get("kind") in ("remember", "teach")
            )

    def record(
        self,
        transcript_sha256: str,
        fingerprint: str,
        *,
        kind: str,
        session_key: str,
        result_id: str = "",
        written_at: str | None = None,
    ) -> None:
        entry = {
            "transcript_sha256": transcript_sha256,
            "fingerprint": fingerprint,
            "kind": kind,
            "session_key": session_key,
            "result_id": result_id,
            "written_at": written_at
            or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with self._lock:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._seen[(transcript_sha256, fingerprint)] = entry


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

#: Context keys every written trace must carry, and the report proves it.
REQUIRED_CONTEXT_KEYS = (
    "agent",
    "task",
    "session_id",
    "transport_session_id",
    "source_transcript_sha256",
)


def enforce_attribution(
    op: ProposedOp, record: SessionRecord, *, task: str
) -> ProposedOp:
    """Stamp the required attribution onto an op's write context.

    ``agent`` and ``transport_session_id`` are *overridden*, not defaulted:
    whatever a producing stage wrote, the trace must say which extractor wrote
    it and which source session it came from. An empty transport id stays an
    explicit empty string so the server's ``setdefault`` cannot substitute the
    extractor's own connection identity for the source session's.
    """

    payload = dict(op.payload)
    context = dict(payload.get("context") or {})
    context["agent"] = f"extractor:{record.cli}"
    context.setdefault("task", task)
    context.setdefault("session_id", record.cli_session_id or record.session_key)
    context["transport_session_id"] = source_transport_id(record)
    context["source_transcript_sha256"] = (
        record.transcript_sha256 or op.provenance.transcript_sha256
    )
    context.setdefault("source_session_key", record.session_key)
    context.setdefault("source_span", op.provenance.source_span)
    context.setdefault("extraction_runner", RUNNER_VERSION)
    payload["context"] = context
    return ProposedOp(
        kind=op.kind, payload=payload, provenance=op.provenance, evidence=op.evidence
    )


# --------------------------------------------------------------------------
# configuration and bookkeeping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerConfig:
    """Everything one run holds constant."""

    task: str = "post-session-extraction/extraction-runner"
    stages: tuple[str, ...] = ("insights", "corrections", "attest")
    dry_run: bool = True
    #: Node-writing ops (remember + teach) accepted per session / per run.
    max_ops_per_session: int = 3
    max_total_ops: int = 24
    #: Attest ops per session / per run; idempotent server-side, still bounded.
    max_attests_per_session: int = 8
    max_total_attests: int = 64
    #: Containment at which the LM-side pre-write check calls it a duplicate.
    lm_dup_containment: float = 0.75
    insights: ExtractionConfig = DEFAULT_EXTRACTION
    corrections: DetectorConfig = DEFAULT_CORRECTIONS


@dataclass
class OpDecision:
    """What happened to one proposed op. Every op gets exactly one."""

    kind: str
    session_key: str
    fingerprint: str
    status: str
    reason: str = ""
    detail: str = ""
    locator: str = ""
    preview: str = ""
    node_id: str = ""
    supersedes_edge_id: str = ""
    attestation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "session_key": self.session_key,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "locator": self.locator,
            "preview": self.preview,
        }
        if self.node_id:
            data["node_id"] = self.node_id
        if self.supersedes_edge_id:
            data["supersedes_edge_id"] = self.supersedes_edge_id
        if self.attestation:
            data["attestation"] = dict(self.attestation)
        return data


@dataclass
class SessionProposals:
    """One session's proposals plus the per-stage accounting behind them."""

    record: SessionRecord
    ops: list[ProposedOp] = field(default_factory=list)
    attests: list[ProposedOp] = field(default_factory=list)
    stage_stats: dict[str, Any] = field(default_factory=dict)
    judge_stats: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _op_preview(op: ProposedOp, limit: int = 240) -> str:
    payload = op.payload
    if op.kind == "remember":
        text = str(payload.get("content") or "")
    elif op.kind == "teach":
        correction = payload.get("correction")
        text = correction if isinstance(correction, str) else canonical_json(
            correction, indent=None
        )
        text = f"teach {payload.get('trace_id')}: {text}"
    else:
        text = f"attest {payload.get('recall_event_id')} ({len(payload.get('evidence') or [])} items)"
    text = default_redactor(text)
    return text[:limit] + ("…" if len(text) > limit else "")


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


class ExtractionRunner:
    """Drives propose -> decide -> execute for a set of session records."""

    def __init__(
        self,
        *,
        config: RunnerConfig,
        ledger: Ledger,
        client: LiveMemoryClient | None = None,
        judge_factory: Callable[[], Any] | None = None,
        gate: Gate | None = None,
        deduper: Deduper | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.client = client
        self.judge_factory = judge_factory
        self.gate = gate or Gate(DEFAULT_GATE)
        if deduper is not None:
            self.deduper = deduper
        elif client is not None:
            # cross_scope=False ON PURPOSE for the MCP index: the server
            # resolves a scope-less recall to global only, so searching "every
            # scope" is not a capability this transport has. Passing the write's
            # target scope makes the server search that scope PLUS global —
            # exactly the view a reader recalling in that project would get.
            self.deduper = Deduper(
                McpMemoryIndex(client),
                config=DedupConfig(cross_scope=False),
            )
        else:
            self.deduper = Deduper(NullMemoryIndex(), config=DEFAULT_DEDUP)
        self.judge_stats: list[Any] = []

    # ---- proposal phase ----------------------------------------------------

    def propose(self, record: SessionRecord) -> SessionProposals:
        """Run the three producing stages over one session. Never raises."""

        proposals = SessionProposals(record=record)
        if "insights" in self.config.stages:
            self._propose_insights(proposals)
        if "corrections" in self.config.stages:
            self._propose_corrections(proposals)
        if "attest" in self.config.stages:
            self._propose_attests(proposals)
        return proposals

    def _propose_insights(self, proposals: SessionProposals) -> None:
        if self.judge_factory is None:
            proposals.stage_stats["insights"] = {"skipped": "no judge configured"}
            return
        try:
            judge = self.judge_factory()
            result: SessionExtraction = extract_session(
                proposals.record,
                judge=judge,
                gate=self.gate,
                deduper=self.deduper,
                config=self.config.insights,
            )
        except Exception as exc:  # noqa: BLE001 - one stage must not kill the run
            proposals.errors.append(f"insights: {exc}")
            proposals.stage_stats["insights"] = {"error": str(exc)[:300]}
            return
        proposals.ops.extend(result.ops)
        proposals.judge_stats.extend(getattr(judge, "stats", ()))
        proposals.stage_stats["insights"] = result.to_dict(include_ops=False)

    def _propose_corrections(self, proposals: SessionProposals) -> None:
        if self.judge_factory is None:
            proposals.stage_stats["corrections"] = {"skipped": "no judge configured"}
            return
        try:
            judge = self.judge_factory()
            result: DetectionResult = detect_corrections(
                proposals.record, judge, config=self.config.corrections
            )
        except Exception as exc:  # noqa: BLE001
            proposals.errors.append(f"corrections: {exc}")
            proposals.stage_stats["corrections"] = {"error": str(exc)[:300]}
            return
        # Teach first in the op list: corrections are the rarer, higher-value
        # write, so the per-session budget should spend on them first.
        proposals.ops[:0] = result.ops
        proposals.judge_stats.extend(getattr(judge, "stats", ()))
        proposals.stage_stats["corrections"] = result.as_dict()

    def _propose_attests(self, proposals: SessionProposals) -> None:
        try:
            ops = proposed_attestations(proposals.record)
        except Exception as exc:  # noqa: BLE001
            proposals.errors.append(f"attest: {exc}")
            proposals.stage_stats["attest"] = {"error": str(exc)[:300]}
            return
        proposals.attests.extend(ops)
        proposals.stage_stats["attest"] = {
            "recall_events": len(proposals.record.recall_event_ids),
            "proposed": len(ops),
        }

    # ---- decide + execute --------------------------------------------------

    def _lm_duplicate(self, op: ProposedOp) -> tuple[str, dict[str, Any]]:
        """Ask the live server whether this write already exists.

        Returns ``(reason, details)``; an empty reason means "not a duplicate
        as far as the server shows". This is the fallback that holds when the
        local ledger is lost. Attest ops skip it: the server's own attestation
        ledger replays them idempotently.
        """

        if self.client is None or op.kind == "attest":
            return "", {"checked": 0, "available": self.client is not None}
        payload = op.payload
        scope = (payload.get("context") or {}).get("scope")
        if op.kind == "remember":
            fact = str(payload.get("content") or "")
        else:
            correction = payload.get("correction")
            fact = correction if isinstance(correction, str) else canonical_json(
                correction, indent=None
            )
            # The corrective trace lands in the ORIGINAL node's scope, so that
            # is where its twin must be searched for; a scope-less recall would
            # resolve to global only and miss it.
            try:
                found = self.client.lookup_nodes([str(payload.get("trace_id") or "")])
                results = found.get("results") or []
                if results and results[0].get("scope"):
                    scope = results[0]["scope"]
            except ClientError:
                pass
        index = McpMemoryIndex(self.client)
        try:
            snapshots = index.candidates(fact, scope=scope, limit=8)
        except ClientError as exc:
            return "", {"checked": 0, "error": str(exc)[:200]}
        best = 0.0
        best_id = ""
        for snapshot in snapshots:
            context = index.last_contexts.get(snapshot.node_id, {})
            same_transcript = op.provenance.transcript_sha256 and (
                context.get("source_transcript_sha256") == op.provenance.transcript_sha256
                or context.get("transcript_sha256") == op.provenance.transcript_sha256
            )
            same_span = context.get("source_span") == op.provenance.source_span
            same_correction = (
                op.kind == "teach"
                and context.get("corrected_node_id") == payload.get("trace_id")
            )
            if same_transcript and (same_span or same_correction):
                return "lm_prior_extraction", {
                    "node_id": snapshot.node_id,
                    "checked": len(snapshots),
                }
            value = _containment(fact, snapshot.content)
            if value > best:
                best, best_id = value, snapshot.node_id
        if best >= self.config.lm_dup_containment:
            return "lm_duplicate", {
                "node_id": best_id,
                "containment": round(best, 4),
                "checked": len(snapshots),
            }
        return "", {"checked": len(snapshots), "best_containment": round(best, 4)}

    def _execute(self, op: ProposedOp, decision: OpDecision) -> None:
        """One write through the live MCP interface. Client must be present."""

        assert self.client is not None
        payload = op.payload
        context = payload.get("context") or {}
        if op.kind == "remember":
            response = self.client.remember(str(payload["content"]), dict(context))
            decision.node_id = str((response.get("node") or {}).get("id") or "")
            decision.status = "written"
        elif op.kind == "teach":
            response = self.client.teach(
                str(payload["trace_id"]),
                payload["correction"],
                confidence=payload.get("confidence"),
                context=dict(context),
            )
            decision.node_id = str(
                (response.get("corrective_trace") or {}).get("id") or ""
            )
            decision.supersedes_edge_id = str(
                (response.get("supersedes") or {}).get("id") or ""
            )
            decision.status = "written"
        else:
            response = self.client.attest(
                str(payload["recall_event_id"]),
                [str(item) for item in payload.get("evidence") or []],
                context=dict(context),
            )
            decision.attestation = {
                "attestation_id": response.get("attestation_id"),
                "grounded_node_ids": list(response.get("grounded_node_ids") or []),
                "grounded": len(response.get("grounded_node_ids") or []),
                "credited": bool(response.get("credited")),
                "replay": bool(response.get("replay")),
            }
            decision.status = "written"

    def decide_and_execute(
        self, all_proposals: Sequence[SessionProposals]
    ) -> list[OpDecision]:
        """Apply attribution, budgets, ledger, LM dedup; then write (or not).

        Order inside a session is teach-before-remember (see
        :meth:`_propose_corrections`); sessions keep the caller's order so the
        budget is deterministic. Every drop is a logged decision, never a
        silent truncation.
        """

        decisions: list[OpDecision] = []
        total_ops = 0
        total_attests = 0
        for proposals in all_proposals:
            record = proposals.record
            session_ops = 0
            session_attests = 0
            for op in proposals.ops + proposals.attests:
                op = enforce_attribution(op, record, task=self.config.task)
                fingerprint = op_fingerprint(op)
                decision = OpDecision(
                    kind=op.kind,
                    session_key=record.session_key,
                    fingerprint=fingerprint,
                    status="proposed",
                    locator=op.provenance.source_span,
                    preview=_op_preview(op),
                )
                decisions.append(decision)

                if op.kind == "attest":
                    if session_attests >= self.config.max_attests_per_session:
                        decision.status = "dropped"
                        decision.reason = "budget_session_attests"
                        continue
                    if total_attests >= self.config.max_total_attests:
                        decision.status = "dropped"
                        decision.reason = "budget_total_attests"
                        continue
                else:
                    if session_ops >= self.config.max_ops_per_session:
                        decision.status = "dropped"
                        decision.reason = "budget_session_ops"
                        continue
                    if total_ops >= self.config.max_total_ops:
                        decision.status = "dropped"
                        decision.reason = "budget_total_ops"
                        continue

                sha = op.provenance.transcript_sha256
                if self.ledger.has(sha, fingerprint):
                    decision.status = "skipped"
                    decision.reason = "ledger_duplicate"
                    continue

                reason, detail = self._lm_duplicate(op)
                if reason:
                    decision.status = "skipped"
                    decision.reason = reason
                    decision.detail = canonical_json(detail, indent=None)
                    continue

                # The op is accepted; it consumes budget whether or not this
                # is a dry run, so a dry run predicts exactly what a write run
                # would do.
                if op.kind == "attest":
                    session_attests += 1
                    total_attests += 1
                else:
                    session_ops += 1
                    total_ops += 1

                if self.config.dry_run:
                    decision.status = "would_write"
                    continue

                if self.client is None:
                    decision.status = "failed"
                    decision.reason = "no_client"
                    continue
                try:
                    self._execute(op, decision)
                except ClientError as exc:
                    decision.status = "failed"
                    decision.reason = "client_error"
                    decision.detail = str(exc)[:300]
                    continue
                self.ledger.record(
                    sha,
                    fingerprint,
                    kind=op.kind,
                    session_key=record.session_key,
                    result_id=decision.node_id
                    or str(decision.attestation.get("attestation_id") or ""),
                )
        return decisions


# --------------------------------------------------------------------------
# run report
# --------------------------------------------------------------------------


def _status_counts(decisions: Sequence[OpDecision]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for decision in decisions:
        by_status[decision.status] = by_status.get(decision.status, 0) + 1
        if decision.reason:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
    return {
        "by_status": dict(sorted(by_status.items())),
        "by_reason": dict(sorted(reasons.items())),
    }


def build_report(
    *,
    mode: str,
    splits_read: Sequence[str],
    gate_decision: GateDecision,
    proposals: Sequence[SessionProposals],
    decisions: Sequence[OpDecision],
    config: RunnerConfig,
    ledger: Ledger,
    generated_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The run report: what was proposed, decided, written, and under whose
    authority. Counts everywhere; quoted content only as redacted previews."""

    ops_decisions = [d for d in decisions if d.kind in ("remember", "teach")]
    attest_decisions = [d for d in decisions if d.kind == "attest"]
    written = [d for d in decisions if d.status == "written"]
    judge_stats = [item for p in proposals for item in p.judge_stats]

    report: dict[str, Any] = {
        "generator": "scripts/post_session_extract.py",
        "runner_version": RUNNER_VERSION,
        "generated_at": generated_at
        or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "splits_read": list(splits_read),
        "gate": gate_decision.to_dict(),
        "budgets": {
            "max_ops_per_session": config.max_ops_per_session,
            "max_total_ops": config.max_total_ops,
            "max_attests_per_session": config.max_attests_per_session,
            "max_total_attests": config.max_total_attests,
            "dropped": sum(1 for d in decisions if d.status == "dropped"),
        },
        "sessions": [
            {
                "session_key": p.record.session_key,
                "source": p.record.source,
                "cli": p.record.cli,
                "proposed_ops": len(p.ops),
                "proposed_attests": len(p.attests),
                "stages": p.stage_stats,
                "errors": list(p.errors),
            }
            for p in proposals
        ],
        "ops": {
            "proposed": len(ops_decisions),
            **_status_counts(ops_decisions),
        },
        "attestations": {
            "proposed": len(attest_decisions),
            **_status_counts(attest_decisions),
            "applied": sum(
                1 for d in attest_decisions if d.status == "written"
            ),
            "replayed": sum(
                1
                for d in attest_decisions
                if d.attestation.get("replay")
            ),
            "grounded_results": sum(
                int(d.attestation.get("grounded") or 0) for d in attest_decisions
            ),
            "credited_events": sum(
                1 for d in attest_decisions if d.attestation.get("credited")
            ),
        },
        "written_nodes": [d.node_id for d in written if d.node_id],
        "teach_edges": [
            {
                "corrective_trace_id": d.node_id,
                "supersedes_edge_id": d.supersedes_edge_id,
            }
            for d in written
            if d.kind == "teach"
        ],
        "attribution": {
            "required_keys": list(REQUIRED_CONTEXT_KEYS),
            "note": "enforce_attribution stamps every accepted op; see decisions",
        },
        "judge": {
            "calls": len(judge_stats),
            "total_cost_usd": round(
                sum(float(getattr(s, "total_cost_usd", 0.0)) for s in judge_stats), 4
            ),
            "outcomes": _count(getattr(s, "outcome", "?") for s in judge_stats),
            "backends": _count(getattr(s, "backend", "?") for s in judge_stats),
        },
        "ledger": {
            "path": str(ledger.path),
            "entries": len(ledger),
            "written_nodes_total": ledger.written_nodes_total(),
        },
        "decisions": [d.to_dict() for d in decisions],
    }
    if extra:
        report.update(dict(extra))
    return report


def _count(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
