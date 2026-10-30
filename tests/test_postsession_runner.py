"""Tests for the post-session extraction runner and its CLI.

Layers:

* the consumption gate, against the *published* baseline artifact and against
  synthetic verdicts covering PASS / FAIL / INCONCLUSIVE / MISSING / STALE —
  the pre-registered bar names ``runner.py`` as its enforcer, so the decision
  procedure is exercised branch by branch;
* the idempotency ledger and the op fingerprint;
* attribution enforcement — every write context carries the extractor identity
  and the SOURCE session's transport id;
* budgets — drops are logged decisions, never silent truncation;
* the CLI — dry-run by default, seal required for the reserved split, exit 3
  with zero writes when the gate does not authorize;
* one live end-to-end write scenario against a throwaway server subprocess on
  a spare port with a scratch database — never the live unit, never the live
  database file — proving real writes, real supersedes edges, real grounded
  attestation, ledger suppression, and a positive prior-extraction lookup.
"""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from living_memory.postsession.client import (
    ClientError, LiveMemoryClient, McpMemoryIndex, PreDispatchError, resolve_token,
)
from living_memory.postsession.judge import FakeJudge
from living_memory.postsession.runner import (
    DEFAULT_BOOTSTRAP_ALLOWANCE,
    EXIT_GATE,
    EXIT_USAGE,
    REQUIRED_CONTEXT_KEYS,
    ExtractionRunner,
    GateDecision,
    Ledger,
    LedgerError,
    OpDecision,
    RunnerConfig,
    SessionProposals,
    build_report,
    enforce_attribution,
    evaluate_gate,
    op_fingerprint,
)
from living_memory.postsession.session import (
    ProposedOp,
)

from postsession_support import (
    CORRECTION, EXTRACTED_FACT, ORIGINAL_FACT, ROOT_DIR, TOKEN,
    _live_record, _load_cli, live_server, make_record, make_remember, make_teach,
)

BASELINE = ROOT_DIR / "artifacts" / "post-session" / "usage-baseline.json"
SELFCHECK = ROOT_DIR / "artifacts" / "post-session" / "counterfactual-selfcheck.json"
CLAUDE_FIXTURE = (
    ROOT_DIR
    / "tests"
    / "fixtures"
    / "postsession"
    / "home"
    / ".claude"
    / "projects"
    / "-home-user-p-demo"
    / "00000002-0000-4000-8000-000000000000.jsonl"
)

#: Pinned "now" so gate-age tests never depend on the wall clock.
NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# synthetic consumption verdicts
# --------------------------------------------------------------------------


def write_verdict(
    path: Path,
    *,
    cohort_kind: str = "organic_holdout",
    rate: float = 0.1159,
    nodes: int = 949,
    passed: bool = True,
    generated_at: str = "2026-08-19T01:32:55Z",
    replay_since: str = "2026-08-15T00:00:00Z",
    as_of: str = "2026-08-19T00:00:00Z",
    min_containment: float = 0.25,
    within_days: int = 7,
) -> Path:
    payload = {
        "meta": {"artifact": "counterfactual_consumption", "generated_at_utc": generated_at, "as_of": as_of},
        "protocol": {
            "replay_since": replay_since,
            "min_containment": min_containment,
            "within_days": within_days,
        },
        "selfcheck": {
            "cohort_kind": cohort_kind,
            "cohort_nodes": nodes,
            "counterfactual_rate": rate,
            "passed": passed,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def test_gate_passes_bootstrap_on_the_published_artifacts() -> None:
    decision = evaluate_gate(BASELINE, SELFCHECK, now=NOW)
    assert decision.status == "PASS"
    assert decision.authorized
    assert decision.mode == "control_bootstrap"
    # The floor must be the protocol-keyed one, never the deprecated scalar.
    published = json.loads(BASELINE.read_text())["pre_registered_bar"]
    assert decision.details["absolute_floor"] == pytest.approx(
        published["absolute_floor_by_protocol"]["counterfactual_post_cutoff"]
    )
    assert decision.details["absolute_floor"] != published["absolute_floor"]


def test_gate_missing_verdict_is_not_authorized(tmp_path: Path) -> None:
    decision = evaluate_gate(BASELINE, tmp_path / "absent.json", now=NOW)
    assert decision.status == "MISSING"
    assert not decision.authorized
    assert any("absent.json" in reason for reason in decision.reasons)


def test_gate_missing_baseline_is_not_authorized(tmp_path: Path) -> None:
    decision = evaluate_gate(tmp_path / "no-bar.json", SELFCHECK, now=NOW)
    assert decision.status == "MISSING"
    assert not decision.authorized


def test_gate_stale_verdict_is_not_authorized(tmp_path: Path) -> None:
    verdict = write_verdict(tmp_path / "v.json", generated_at="2026-07-01T00:00:00Z")
    decision = evaluate_gate(BASELINE, verdict, now=NOW, max_age_days=14)
    assert decision.status == "STALE"
    assert not decision.authorized
    assert any("re-run" in reason for reason in decision.reasons)


def test_gate_unvalidated_harness_cannot_authorize(tmp_path: Path) -> None:
    verdict = write_verdict(tmp_path / "v.json", passed=False)
    decision = evaluate_gate(BASELINE, verdict, now=NOW)
    assert decision.status == "INCONCLUSIVE"
    assert not decision.authorized


def test_gate_bootstrap_allowance_stops_unmeasured_writing(tmp_path: Path) -> None:
    verdict = write_verdict(tmp_path / "v.json")
    decision = evaluate_gate(
        BASELINE, verdict, now=NOW, written_nodes_total=DEFAULT_BOOTSTRAP_ALLOWANCE + 1
    )
    assert decision.status == "FAIL"
    assert not decision.authorized
    assert any("allowance" in reason for reason in decision.reasons)


def test_gate_extracted_needs_a_control(tmp_path: Path) -> None:
    verdict = write_verdict(tmp_path / "v.json", cohort_kind="extracted")
    decision = evaluate_gate(BASELINE, verdict, now=NOW)
    assert decision.status == "MISSING"
    assert not decision.authorized


@pytest.mark.parametrize(
    ("extracted_rate", "organic_rate", "status", "needle"),
    [
        # relative 0.5x of 0.12 is 0.06; the pinned absolute floor is 0.05.
        (0.10, 0.12, "PASS", ""),
        (0.055, 0.12, "FAIL", "below 0.5 x organic"),
        (0.04, 0.12, "FAIL", "backstop"),
        (0.30, 0.12, "PASS", ""),
    ],
)
def test_gate_extracted_vs_control_decision_rule(
    tmp_path: Path, extracted_rate: float, organic_rate: float, status: str, needle: str
) -> None:
    verdict = write_verdict(
        tmp_path / "extracted.json", cohort_kind="extracted", rate=extracted_rate, nodes=40
    )
    control = write_verdict(tmp_path / "control.json", rate=organic_rate)
    decision = evaluate_gate(BASELINE, verdict, control, now=NOW)
    assert decision.status == status
    assert decision.mode == "extracted_vs_control"
    assert decision.details["extracted_rate"] == pytest.approx(extracted_rate)
    assert decision.details["organic_rate"] == pytest.approx(organic_rate)
    if needle:
        assert any(needle in reason for reason in decision.reasons)


def test_gate_inadmissible_control_is_inconclusive_never_pass(tmp_path: Path) -> None:
    # Control below min_control_rate_by_protocol[counterfactual]=0.1, and an
    # extracted rate that would clear every floor if the control were usable.
    verdict = write_verdict(tmp_path / "extracted.json", cohort_kind="extracted", rate=0.5)
    weak_control = write_verdict(tmp_path / "control.json", rate=0.05)
    decision = evaluate_gate(BASELINE, verdict, weak_control, now=NOW)
    assert decision.status == "INCONCLUSIVE"
    assert not decision.authorized

    tiny_control = write_verdict(tmp_path / "tiny.json", rate=0.2, nodes=50)
    decision = evaluate_gate(BASELINE, verdict, tiny_control, now=NOW)
    assert decision.status == "INCONCLUSIVE"


def test_gate_protocol_mismatch_is_void(tmp_path: Path) -> None:
    verdict = write_verdict(
        tmp_path / "extracted.json", cohort_kind="extracted", rate=0.5
    )
    control = write_verdict(
        tmp_path / "control.json", rate=0.12, replay_since="2026-08-16T00:00:00Z"
    )
    decision = evaluate_gate(BASELINE, verdict, control, now=NOW)
    assert decision.status == "INCONCLUSIVE"
    assert any("identical protocol" in reason for reason in decision.reasons)


# --------------------------------------------------------------------------
# fingerprint + ledger
# --------------------------------------------------------------------------


def test_op_fingerprint_ignores_context_noise_but_not_content() -> None:
    record = make_record()
    op = make_remember(record, "The scratch DB lives under /tmp and is never the live file.")
    stamped = enforce_attribution(op, record, task="task-a")
    restamped = enforce_attribution(op, record, task="task-b")
    assert op_fingerprint(stamped) == op_fingerprint(restamped)

    other = make_remember(record, "A different fact entirely, with different bytes.")
    assert op_fingerprint(op) != op_fingerprint(other)

    teach = make_teach(record, "01AAAAAAAAAAAAAAAAAAAAAAAA", "The port is 8081, not 8080.")
    assert op_fingerprint(teach) != op_fingerprint(op)


def test_ledger_round_trip_reload_and_corruption_refusal(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "state")
    assert len(ledger) == 0
    assert not ledger.has("sha", "fp")
    ledger.record("sha", "fp", kind="remember", session_key="s", result_id="n1")
    ledger.record("sha", "fp2", kind="attest", session_key="s", result_id="a1")
    assert ledger.has("sha", "fp")
    assert ledger.written_nodes_total() == 1  # attest entries create no node

    reloaded = Ledger(tmp_path / "state")
    assert len(reloaded) == 2
    assert reloaded.has("sha", "fp")
    assert reloaded.has("sha", "fp2")
    assert reloaded.written_nodes_total() == 1

    with open(ledger.path, "a", encoding="utf-8") as handle:
        handle.write('not json at all\n{"broken":\n')
    with pytest.raises(LedgerError):
        Ledger(tmp_path / "state")


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------


def test_enforce_attribution_stamps_every_required_key() -> None:
    record = make_record()
    # A corrections-style op arrives with agent="extractor" and no transport id.
    op = make_teach(record, "01AAAAAAAAAAAAAAAAAAAAAAAA", "The flag is --write, not --commit.")
    op = ProposedOp(
        kind=op.kind,
        payload={**op.payload, "context": {**op.payload["context"], "agent": "extractor"}},
        provenance=op.provenance,
        evidence=op.evidence,
    )
    stamped = enforce_attribution(op, record, task="post-session-extraction/extraction-runner")
    context = stamped.payload["context"]
    for key in REQUIRED_CONTEXT_KEYS:
        assert context.get(key), f"missing required context key {key}"
    assert context["agent"] == "extractor:claude"
    assert context["transport_session_id"] == "source-transport-1"
    assert context["source_transcript_sha256"] == record.transcript_sha256
    assert context["session_id"] == record.cli_session_id
    assert context["source_session_key"] == record.session_key
    # Existing keys survive: corrections' own audit trail is not erased.
    assert context["corrected_node_id"] == "01AAAAAAAAAAAAAAAAAAAAAAAA"


def test_enforce_attribution_pins_empty_transport_id_explicitly() -> None:
    record = make_record(transport_session_ids=(), cli_session_id=None)
    op = make_remember(record, "fact with no transport identity")
    stamped = enforce_attribution(op, record, task="t")
    # Present and empty: the server's setdefault must not substitute the
    # extractor's own connection identity for the source session's.
    assert "transport_session_id" in stamped.payload["context"]
    assert stamped.payload["context"]["transport_session_id"] == ""


# --------------------------------------------------------------------------
# budgets + dry-run decisions
# --------------------------------------------------------------------------


def _runner(tmp_path: Path, **config: Any) -> ExtractionRunner:
    return ExtractionRunner(
        config=RunnerConfig(dry_run=True, **config),
        ledger=Ledger(tmp_path / "state"),
    )


def test_budget_drops_are_logged_decisions_never_silent(tmp_path: Path) -> None:
    record = make_record()
    ops = [make_remember(record, f"distinct fact number {i} about module_{i}.py") for i in range(6)]
    runner = _runner(tmp_path, max_ops_per_session=2, max_total_ops=10)
    decisions = runner.decide_and_execute([SessionProposals(record=record, ops=ops)])
    assert len(decisions) == 6  # every proposed op got exactly one decision
    statuses = [d.status for d in decisions]
    assert statuses.count("would_write") == 2
    dropped = [d for d in decisions if d.status == "dropped"]
    assert len(dropped) == 4
    assert all(d.reason == "budget_session_ops" for d in dropped)

    report = build_report(
        mode="dry-run",
        splits_read=[],
        gate_decision=evaluate_gate(BASELINE, SELFCHECK, now=NOW),
        proposals=[SessionProposals(record=record, ops=ops)],
        decisions=decisions,
        config=runner.config,
        ledger=runner.ledger,
    )
    assert report["budgets"]["dropped"] == 4
    assert report["ops"]["by_reason"]["budget_session_ops"] == 4


def test_total_budget_spans_sessions(tmp_path: Path) -> None:
    records = [make_record(session_key=f"claude:s{i}", transcript_sha256=f"{i:02x}" * 32) for i in range(3)]
    proposals = [
        SessionProposals(record=r, ops=[make_remember(r, f"fact for {r.session_key}")])
        for r in records
    ]
    runner = _runner(tmp_path, max_ops_per_session=3, max_total_ops=2)
    decisions = runner.decide_and_execute(proposals)
    assert [d.status for d in decisions] == ["would_write", "would_write", "dropped"]
    assert decisions[-1].reason == "budget_total_ops"


def test_dry_run_consults_ledger_and_writes_nothing(tmp_path: Path) -> None:
    record = make_record()
    op = make_remember(record, "an already-written fact")
    runner = _runner(tmp_path)
    fingerprint = op_fingerprint(enforce_attribution(op, record, task=runner.config.task))
    runner.ledger.record(
        record.transcript_sha256 or "", fingerprint, kind="remember", session_key=record.session_key
    )
    decisions = runner.decide_and_execute([SessionProposals(record=record, ops=[op])])
    assert decisions[0].status == "skipped"
    assert decisions[0].reason == "ledger_duplicate"
    # Dry run + no client: nothing could have been written anywhere.
    assert runner.client is None


def test_propose_wires_all_three_stages(tmp_path: Path) -> None:
    record = make_record()
    runner = ExtractionRunner(
        config=RunnerConfig(dry_run=True),
        ledger=Ledger(tmp_path / "state"),
        judge_factory=lambda: FakeJudge([{"verdict": "none", "reason": "fake"}] * 100),
    )
    proposals = runner.propose(record)
    assert set(proposals.stage_stats) == {"insights", "corrections", "attest"}
    assert proposals.ops == []  # a refusing judge proposes nothing
    assert proposals.attests == []  # no recall_event_ids in this record


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_requires_exactly_one_input(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    assert cli.main([]) == EXIT_USAGE
    assert "exactly one" in capsys.readouterr().err


def test_cli_seal_required_for_the_reserved_split(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    code = cli.main(["--split", "holdout", "--judge", "none"])
    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "sealed" in err


def test_cli_seal_holds_in_session_key_mode(tmp_path: Path) -> None:
    """A holdout session requested by key is refused without the seal flag."""

    cli = _load_cli()
    index = tmp_path / "corpus-index.jsonl"
    row = {
        "session_key": "claude:sneaky-holdout-read",
        "source": "claude",
        "cli": "claude",
        "path": str(CLAUDE_FIXTURE),
        "split": "holdout",
    }
    index.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli.select_records(
            cli.build_parser().parse_args(
                ["--session", "claude:sneaky-holdout-read", "--index", str(index)]
            ),
            lambda message: None,
        )
    assert "sealed" in str(excinfo.value)


def test_cli_halts_exit3_and_writes_nothing_when_verdict_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    out = tmp_path / "report.json"
    state = tmp_path / "state"
    code = cli.main(
        [
            "--transcript",
            str(CLAUDE_FIXTURE),
            "--write",  # even an explicit write run must halt before connecting
            "--url",
            "http://127.0.0.1:1/mcp/",  # unreachable: proves we never got there
            "--gate-verdict",
            str(tmp_path / "no-verdict.json"),
            "--baseline",
            str(BASELINE),
            "--state-dir",
            str(state),
            "--out",
            str(out),
            "--judge",
            "none",
        ]
    )
    assert code == EXIT_GATE
    err = capsys.readouterr().err
    assert "GATE MISSING" in err
    assert "Nothing was written" in err
    report = json.loads(out.read_text())
    assert report["mode"] == "halted"
    assert report["gate"]["authorized"] is False
    assert report["written_nodes"] == []
    assert not (state / "ledger.jsonl").exists()


def test_cli_dry_run_is_the_default_and_reports(tmp_path: Path) -> None:
    cli = _load_cli()
    out = tmp_path / "report.json"
    code = cli.main(
        [
            "--transcript",
            str(CLAUDE_FIXTURE),
            "--judge",
            "fake",
            "--baseline",
            str(BASELINE),
            "--gate-verdict",
            str(SELFCHECK),
            "--max-verdict-age-days",
            "100000",  # the artifact ages; this test must not
            "--state-dir",
            str(tmp_path / "state"),
            "--out",
            str(out),
            "--workers",
            "1",
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["mode"] == "dry-run"
    assert report["gate"]["status"] == "PASS"
    assert report["written_nodes"] == []
    assert report["sessions"], "the fixture session was read"
    assert set(report["sessions"][0]["stages"]) == {"insights", "corrections", "attest"}
    # A dry run never writes to Living Memory: no ledger entry can exist.
    assert not (tmp_path / "state" / "ledger.jsonl").exists()


# --------------------------------------------------------------------------
# live end-to-end against a throwaway server
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ok", "connect", "timeout", "negative", "cleanup"])
def test_preflight_transport_contract(monkeypatch, mode):
    import fastmcp

    events = []

    class Transport:
        def __init__(self, url, **kwargs):
            assert url == "http://127.0.0.1:9876/mcp/"
            assert kwargs == {"auth": TOKEN, "timeout": 0.25, "init_timeout": 0.25}

        async def __aenter__(self):
            if mode == "connect":
                raise TimeoutError("initialization timed out")
            return self

        async def ping(self):
            events.append("ping")
            if mode == "timeout":
                raise TimeoutError("ping timed out")
            return mode != "negative"

        async def call_tool(self, *args):
            pytest.fail("preflight must not dispatch a tool")

        async def __aexit__(self, *args):
            if mode == "cleanup":
                raise OSError("cleanup failed")

    monkeypatch.setattr(fastmcp, "Client", Transport)
    client = LiveMemoryClient("http://127.0.0.1:9876/mcp/", token=TOKEN, timeout=0.25)
    if mode in {"ok", "cleanup"}:
        assert client.ping() is True
    else:
        with pytest.raises(ClientError) as caught:
            client.ping()
        assert isinstance(caught.value, PreDispatchError) == (mode == "connect")
    assert events == ([] if mode == "connect" else ["ping"])
    assert client.calls == ["ping"]


def test_preflight_token_resolution(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("LM_AUTH_TOKEN='file-token'\n")
    assert resolve_token("explicit", environ={"LM_AUTH_TOKEN": "env"}, env_file=env_file) == "explicit"
    assert resolve_token(environ={"LM_AUTH_TOKEN": "env"}, env_file=env_file) == "env"
    assert resolve_token(environ={}, env_file=env_file) == "file-token"
    assert resolve_token(environ={}, env_file=None) is None


@pytest.mark.parametrize("token", [TOKEN, "invalid-test-token", None])
def test_preflight_http_auth(live_server, token):
    client = LiveMemoryClient(live_server["url"], token=token, timeout=2)
    if token == TOKEN:
        assert client.ping() is True
    else:
        with pytest.raises(PreDispatchError):
            client.ping()
    assert client.calls == ["ping"]


def test_preflight_unavailable_host():
    # Reserve a port without listening so no other process can accept the call.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        client = LiveMemoryClient(
            f"http://127.0.0.1:{sock.getsockname()[1]}/mcp/", token=TOKEN, timeout=0.25,
        )
        with pytest.raises(PreDispatchError):
            client.ping()


def test_preflight_protocol_timeout():
    from fastmcp import FastMCP
    from mcp import types

    server = FastMCP("slow-ping")
    received = []

    async def slow_ping(request):
        received.append(request)
        await asyncio.sleep(1)
        return types.ServerResult(types.EmptyResult())

    server._mcp_server.request_handlers[types.PingRequest] = slow_ping
    client = LiveMemoryClient(server, timeout=0.1)
    with pytest.raises(ClientError, match="Timed out") as caught:
        client.ping()
    assert not isinstance(caught.value, PreDispatchError)
    assert len(received) == 1


def test_preflight_needs_no_status_report_or_sql(tmp_path, monkeypatch):
    from living_memory import server as server_module

    server = server_module.create_mcp_server(tmp_path / "preflight.sqlite3")
    store = server.memory_store
    client = LiveMemoryClient(server)
    sql = []
    status_calls = []

    def unavailable_report(*args, **kwargs):
        status_calls.append("memory_status")
        raise RuntimeError("report unavailable")

    try:
        store.connection.set_trace_callback(sql.append)
        assert client.call("memory_status", {})
        assert sql, "the real report still reads the database"
        monkeypatch.setattr(server_module, "status_view", unavailable_report)
        with pytest.raises(ClientError):
            client.call("memory_status", {})
        assert status_calls == ["memory_status"]
        status_calls.clear()
        sql.clear()
        assert client.ping() is True
        assert status_calls == []
        assert sql == []
    finally:
        store.connection.set_trace_callback(None)
        server.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("flag", ["--write", "--lm-dedup"])
def test_cli_failed_preflight_stops_before_judge_and_write(
    live_server, tmp_path, monkeypatch, capsys, flag,
):
    cli = _load_cli()
    monkeypatch.setattr(cli, "evaluate_gate", lambda *args, **kwargs: GateDecision("PASS"))
    monkeypatch.setattr(cli, "select_records", lambda *args: ([make_record()], [], {}))

    def forbidden(*args, **kwargs):
        pytest.fail("failed preflight must stop before judge or runner construction")

    monkeypatch.setattr(cli, "make_judge_factory", forbidden)
    monkeypatch.setattr(cli, "ExtractionRunner", forbidden)
    assert cli.main([
        "--transcript", "unused", "--state-dir", str(tmp_path), flag,
        "--url", live_server["url"], "--token", "invalid-test-token", "--timeout", "0.5",
    ]) == EXIT_USAGE
    assert "cannot reach the Living Memory host" in capsys.readouterr().err
    assert not (tmp_path / "ledger.jsonl").exists()


def test_live_write_scenario_attribution_idempotency_and_ledger_loss(
    live_server: dict[str, Any], tmp_path: Path
) -> None:
    from living_memory.postsession.evidence import proposed_attestations

    client = LiveMemoryClient(live_server["url"], token=TOKEN, timeout=30.0)
    client.ping()

    # Seed the scratch world as some *other* agent: a node that a later recall
    # delivers, which our synthetic source session then contradicts and uses.
    seeded = client.remember(
        ORIGINAL_FACT,
        {"scope": "project:runner-live", "agent": "codex", "task": "seeding"},
    )
    original_id = seeded["node"]["id"]
    recalled = client.recall("deploy port default", scope="project:runner-live")
    event_id = recalled["recall_event_id"]
    assert event_id
    delivered_ids = {
        entry["id"] for entry in recalled["results"]
    }
    assert original_id in delivered_ids

    record = _live_record(event_id, original_id)
    ops = [
        make_teach(record, original_id, CORRECTION),
        make_remember(record, EXTRACTED_FACT, scope="project:runner-live"),
    ]
    attests = proposed_attestations(record)
    assert attests, "the synthetic diff must yield attestation evidence"

    def run(state_dir: Path) -> list[OpDecision]:
        runner = ExtractionRunner(
            config=RunnerConfig(dry_run=False),
            ledger=Ledger(state_dir),
            client=client,
        )
        return runner.decide_and_execute(
            [SessionProposals(record=record, ops=list(ops), attests=list(attests))]
        )

    # ---- run 1: everything writes -----------------------------------------
    first = run(tmp_path / "state-a")
    by_kind = {d.kind: d for d in first}
    assert by_kind["teach"].status == "written"
    assert by_kind["teach"].node_id, "teach returned the corrective trace id"
    assert by_kind["teach"].supersedes_edge_id, "teach created a supersedes edge"
    assert by_kind["remember"].status == "written"
    assert by_kind["remember"].node_id
    assert by_kind["attest"].status == "written", by_kind["attest"].detail
    assert by_kind["attest"].attestation["grounded"] >= 1
    assert by_kind["attest"].attestation["replay"] is False

    # Attribution, verified on the server, not in our own bookkeeping.
    for decision in (by_kind["remember"], by_kind["teach"]):
        fetched = client.lookup_nodes([decision.node_id], view="full")["results"]
        assert fetched, f"{decision.kind} node exists on the server"
        context = fetched[0]["context"]
        assert context["agent"] == "extractor:claude"
        assert context["transport_session_id"] == "live-source-transport"
        assert context["source_transcript_sha256"] == record.transcript_sha256
        assert context["task"]
        assert context["session_id"] == "live-source-session"

    def count_extracted_fact_nodes() -> int:
        found = client.recall(EXTRACTED_FACT, scope="project:runner-live", max_results=10)
        ids = [
            entry["id"]
            for entry in found["results"]
            if entry["id"]
        ]
        if not ids:
            return 0
        full = client.lookup_nodes(ids)["results"]
        return sum(1 for node in full if node["content"] == EXTRACTED_FACT)

    def corrections_on_original() -> int:
        node = client.lookup_nodes([original_id])["results"][0]
        return len(node["provenance"]["corrections"])

    assert count_extracted_fact_nodes() == 1
    assert corrections_on_original() == 1

    # ---- run 2: same ledger -> zero new writes ----------------------------
    second = run(tmp_path / "state-a")
    assert {d.status for d in second} == {"skipped"}
    assert {d.reason for d in second} == {"ledger_duplicate"}
    assert count_extracted_fact_nodes() == 1
    assert corrections_on_original() == 1

    # ---- run 3: fresh ledger -> positive prior-extraction evidence --------
    third = run(tmp_path / "state-b")  # a fresh, empty state dir
    by_kind3 = {d.kind: d for d in third}
    assert by_kind3["remember"].status == "skipped"
    assert by_kind3["remember"].reason in ("lm_prior_extraction", "lm_duplicate")
    assert by_kind3["teach"].status == "skipped"
    assert by_kind3["teach"].reason in ("lm_prior_extraction", "lm_duplicate")
    # The server's own attestation ledger replays; nothing is applied twice.
    assert by_kind3["attest"].status == "written"
    assert by_kind3["attest"].attestation["replay"] is True
    assert count_extracted_fact_nodes() == 1
    assert corrections_on_original() == 1


def test_live_tool_error_preserves_unknown_effect(
    live_server: dict[str, Any], tmp_path: Path
) -> None:
    client = LiveMemoryClient(live_server["url"], token=TOKEN, timeout=15.0)
    record = make_record(
        session_key="claude:bad-teach-session", transcript_sha256="ef" * 32
    )
    # A tool error has no generic no-effect guarantee. The runner must retain
    # UNKNOWN even though this particular request names a missing node.
    ops = [make_teach(record, "01ZZZZZZZZZZZZZZZZZZZZZZZZ", "corrects a phantom node")]
    runner = ExtractionRunner(
        config=RunnerConfig(dry_run=False),
        ledger=Ledger(tmp_path / "state"),
        client=client,
    )
    decisions = runner.decide_and_execute([SessionProposals(record=record, ops=ops)])
    assert decisions[0].status == "unknown"
    assert decisions[0].reason == "unconfirmed_effect"
    assert len(runner.ledger) == 1
    assert runner.ledger.entries()[0]["state"] == "unknown"


@pytest.mark.parametrize("content", [
    "src/app.py restart is forbidden before approval.",
    "src/app.py restart is allowed after approval.",
    "src/app.py restart is not allowed before approval.",
])
def test_lm_duplicate_preserves_distinct_claims_from_the_same_span(
    tmp_path: Path, content: str,
) -> None:
    record = make_record()
    op = make_remember(record, content)
    existing = {
        "id": "prior", "scope": "project:demo",
        "content": "src/app.py restart is allowed before approval.",
        "context": {
            "source_transcript_sha256": record.transcript_sha256,
            "source_span": op.provenance.source_span,
        },
    }

    class Client:
        def recall(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"node": existing, "delivery": "full"}]}

    runner = ExtractionRunner(config=RunnerConfig(), ledger=Ledger(tmp_path), client=Client())
    reason, _ = runner._lm_duplicate(op)
    assert reason == ""


@pytest.mark.parametrize("change, expected", [
    ({}, "lm_prior_extraction"),
    ({"scope": "global"}, ""),
    ({"context": {"source_transcript_sha256": "other", "source_span": "same"}}, ""),
    ({"context": {}}, ""),
])
def test_lm_duplicate_requires_exact_content_scope_and_source(
    tmp_path: Path, change: dict[str, Any], expected: str,
) -> None:
    record = make_record()
    op = make_remember(record, "src/app.py restart is forbidden before approval.")
    existing = {
        "id": "prior", "scope": "project:demo", "content": op.payload["content"],
        "context": {
            "source_transcript_sha256": record.transcript_sha256,
            "source_span": op.provenance.source_span,
        },
        **change,
    }

    class Client:
        def recall(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"node": existing, "delivery": "full"}]}

    runner = ExtractionRunner(config=RunnerConfig(), ledger=Ledger(tmp_path), client=Client())
    reason, _ = runner._lm_duplicate(op)
    assert reason == expected


def test_mcp_index_never_compares_a_preview_when_full_lookup_is_unavailable() -> None:
    class Client:
        def recall(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{
                "node": {"id": "partial", "scope": "project:demo", "content": "same preview"},
                "delivery": "snippet",
            }]}

        def lookup_nodes(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise ClientError("unavailable")

    index = McpMemoryIndex(Client())
    assert index.candidates("same preview", scope="project:demo") == []
    assert index.last_contexts == {}


@pytest.mark.parametrize("corrected_node_id, expected", [(None, ""), ("other", ""), ("original", "lm_prior_extraction")])
def test_plain_remember_does_not_replace_a_teach_operation(
    tmp_path: Path, corrected_node_id: str | None, expected: str,
) -> None:
    record = make_record()
    op = make_teach(record, "original", "src/app.py restart is forbidden before approval.")
    existing = {
        "id": "prior", "scope": "project:demo", "content": op.payload["correction"],
        "context": {
            "source_transcript_sha256": record.transcript_sha256,
            "source_span": op.provenance.source_span,
            "corrected_node_id": corrected_node_id,
        },
    }

    class Client:
        def recall(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"node": existing, "delivery": "full"}]}

        def lookup_nodes(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"id": "original", "scope": "project:demo", "superseded_by": ["prior"]}]}

    runner = ExtractionRunner(config=RunnerConfig(), ledger=Ledger(tmp_path), client=Client())
    reason, _ = runner._lm_duplicate(op)
    assert reason == expected
