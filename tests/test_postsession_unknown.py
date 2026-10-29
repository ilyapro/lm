"""Mutation recovery at the native runner/ledger boundary, with disposable recipients."""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from living_memory.postsession.client import ClientError, LiveMemoryClient, PreDispatchError
from living_memory.postsession.evidence import proposed_attestations
from living_memory.postsession.runner import (
    EXIT_GATE, EXIT_UNKNOWN, ExtractionRunner, Ledger, RunnerConfig,
    SessionProposals, enforce_attribution, op_fingerprint,
)
from test_postsession_runner import (
    ORIGINAL_FACT, TOKEN, _live_record, _load_cli, live_server,
    make_record, make_remember, make_teach,
)


class Recipient:
    url = "synthetic://recipient"

    def __init__(self, failure=None):
        self.failure = failure
        self.effects = []

    def recall(self, *args, **kwargs):
        return {"results": []}

    def remember(self, content, context):
        if isinstance(self.failure, PreDispatchError):
            raise self.failure
        self.effects.append((content, context))
        if self.failure:
            raise self.failure
        return {"node": {"id": f"node-{len(self.effects)}"}}


def run(state, client, record, ops):
    return ExtractionRunner(
        config=RunnerConfig(dry_run=False), ledger=Ledger(state), client=client,
    ).decide_and_execute([SessionProposals(record=record, ops=ops)])


def test_unknown_does_not_block_an_independent_operation(tmp_path):
    record = make_record()
    op = make_remember(record, "First operation has an unknown outcome.")
    client = Recipient(ClientError("reply lost"))
    first = run(tmp_path, client, record, [op])
    assert first[0].status == "unknown"
    client.failure = None
    second = run(tmp_path, client, record, [op, make_remember(record, "Independent fact.")])
    assert [d.status for d in second] == ["unknown", "written"]
    assert len(client.effects) == 2
    assert len(Ledger(tmp_path)) == 2


def test_known_pre_dispatch_failure_resumes_the_saved_request(tmp_path):
    record = make_record()
    client = Recipient(PreDispatchError("connection not established"))
    op = make_remember(record, "A saved fact with stable attribution.")
    assert run(tmp_path, client, record, [op])[0].reason == "pre_dispatch"
    saved = Ledger(tmp_path).entries()[0]
    assert saved["state"] == "ready"
    assert client.effects == []
    client.failure = None
    runner = ExtractionRunner(
        config=RunnerConfig(dry_run=False, task="new task"),
        ledger=Ledger(tmp_path), client=client,
    )
    assert runner.reconcile()[0].status == "written"
    assert client.effects == [(op.payload["content"], saved["operation"]["payload"]["context"])]
    assert Ledger(tmp_path).entries()[0]["state"] == "applied"
    assert runner.reconcile() == []


@pytest.mark.parametrize("boundary", ["file_sync", "replace", "directory_sync"])
def test_durable_save_failure_never_dispatches(tmp_path, monkeypatch, boundary):
    record = make_record()
    client = Recipient()
    ledger = Ledger(tmp_path)
    ledger.entries()  # Create the lock/directory before targeting save itself.

    def fail(*args):
        raise OSError("durable save refused")

    if boundary == "file_sync":
        monkeypatch.setattr(os, "fsync", fail)
    elif boundary == "replace":
        monkeypatch.setattr(os, "replace", fail)
    else:
        monkeypatch.setattr(Ledger, "_sync_dir", fail)
    runner = ExtractionRunner(config=RunnerConfig(dry_run=False), ledger=ledger, client=client)
    decision = runner.decide_and_execute([
        SessionProposals(record=record, ops=[make_remember(record, "Never sent.")])
    ])[0]
    assert decision.reason == "ledger_error"
    assert client.effects == []


def test_completion_save_failure_keeps_unknown_and_result_id(tmp_path, monkeypatch):
    record = make_record()
    client = Recipient()
    op = make_remember(record, "Applied but local completion is unavailable.")
    real_save = Ledger._save
    calls = 0

    def save(ledger):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("completion disk failure")
        real_save(ledger)

    monkeypatch.setattr(Ledger, "_save", save)
    first = run(tmp_path, client, record, [op])[0]
    assert (first.status, first.node_id) == ("unknown", "node-1")
    assert Ledger(tmp_path).entries()[0]["state"] == "unknown"
    assert run(tmp_path, client, record, [op])[0].status == "unknown"
    assert len(client.effects) == 1


def test_recipient_change_cannot_reopen_unknown(tmp_path):
    record = make_record()
    op = make_remember(record, "Target identity matters.")
    run(tmp_path, Recipient(ClientError("reply lost")), record, [op])
    other = Recipient()
    other.url = "synthetic://another-recipient"
    assert run(tmp_path, other, record, [op])[0].reason == "recipient_mismatch"
    assert other.effects == []


def test_old_completed_entries_remain_completed(tmp_path):
    (tmp_path / "ledger.jsonl").write_text(json.dumps({
        "transcript_sha256": "sha", "fingerprint": "fp", "kind": "teach",
        "session_key": "session", "result_id": "node",
    }) + "\n")
    assert Ledger(tmp_path).has("sha", "fp")


def test_client_distinguishes_dispatch_and_preserves_received_result(monkeypatch):
    import fastmcp

    class Transport:
        mode = "connect"
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            if self.mode == "connect":
                raise OSError("connect failed")
            return self

        async def call_tool(self, *args):
            type(self).calls += 1
            if self.mode == "call":
                raise OSError("reply lost")
            return SimpleNamespace(structured_content={"node": {"id": "applied"}})

        async def __aexit__(self, *args):
            if self.mode == "cleanup":
                raise OSError("cleanup failed")

    monkeypatch.setattr(fastmcp, "Client", Transport)
    client = LiveMemoryClient("synthetic://recipient")
    with pytest.raises(PreDispatchError):
        client.remember("fact", {})
    assert Transport.calls == 0
    Transport.mode = "call"
    with pytest.raises(ClientError) as caught:
        client.remember("fact", {})
    assert not isinstance(caught.value, PreDispatchError)
    Transport.mode = "cleanup"
    assert client.remember("fact", {}) == {"node": {"id": "applied"}}


def _concurrent_writer(state, effects, ready, start):
    class FileRecipient(Recipient):
        def remember(self, content, context):
            with open(effects, "a") as handle:
                handle.write("effect\n")
            return {"node": {"id": "applied"}}

    record = make_record()
    ledger = Ledger(state)  # Both processes load before either dispatches.
    ready.set()
    assert start.wait(15)
    ExtractionRunner(config=RunnerConfig(dry_run=False), ledger=ledger, client=FileRecipient()).decide_and_execute([
        SessionProposals(record=record, ops=[make_remember(record, "Concurrent operation.")])
    ])


def test_concurrent_cli_writers_share_one_claim(tmp_path):
    ctx = multiprocessing.get_context("fork")
    start = ctx.Event()
    ready = [ctx.Event(), ctx.Event()]
    effects = tmp_path / "effects"
    processes = [ctx.Process(target=_concurrent_writer, args=(tmp_path / "state", effects, r, start)) for r in ready]
    try:
        for proc in processes:
            proc.start()
        assert all(r.wait(15) for r in ready)
        start.set()
        for proc in processes:
            proc.join(15)
            assert proc.exitcode == 0
        assert effects.read_text().splitlines() == ["effect"]
        assert Ledger(tmp_path / "state").entries()[0]["state"] == "applied"
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(5)


class LostReplyClient(LiveMemoryClient):
    lost_tool = ""
    result = None
    mutations = 0

    async def _call_async(self, tool, arguments):
        if tool in {"memory_remember", "memory_teach", "memory_attest"}:
            self.mutations += 1
        response = await super()._call_async(tool, arguments)
        if tool == self.lost_tool:
            self.result = response
            raise ClientError("recipient applied; response lost")
        return response


class MissingLookupClient(LostReplyClient):
    unavailable = False

    def recall(self, *args, **kwargs):
        if self.unavailable:
            raise ClientError("recall unavailable")
        return {"results": []}

    def lookup_nodes(self, *args, **kwargs):
        if self.unavailable:
            raise ClientError("lookup unavailable")
        return super().lookup_nodes(*args, **kwargs)


def _native_op(client, kind, tag):
    record = make_record(transcript_sha256=tag * 32)
    if kind == "remember":
        return record, make_remember(record, f"config/{tag}.env uses an isolated cache directory.")
    old = client.remember(f"config/{tag}.env uses the obsolete cache path.", {"scope": "project:demo"})
    return record, make_teach(record, old["node"]["id"], f"config/{tag}.env uses the corrected cache path.")


@pytest.mark.parametrize("kind,tag", [("remember", "12"), ("teach", "34")])
@pytest.mark.parametrize("unavailable", [False, True])
def test_native_applied_lost_reply_and_finite_confirmation(live_server, tmp_path, kind, tag, unavailable):
    client = LostReplyClient(live_server["url"], token=TOKEN)
    tag = tag if not unavailable else {"12": "9a", "34": "bc"}[tag]
    record, op = _native_op(client, kind, tag)
    client.lost_tool = f"memory_{kind}"
    assert run(tmp_path, client, record, [op])[0].status == "unknown"
    result_id = client.result["node" if kind == "remember" else "corrective_trace"]["id"]
    saved = Ledger(tmp_path).entries()[0]
    assert saved["state"] == "unknown"
    missing = MissingLookupClient(live_server["url"], token=TOKEN)
    missing.unavailable = unavailable
    assert run(tmp_path, missing, record, [op])[0].status == "unknown"
    assert missing.mutations == 0

    # Direct recipient confirmation closes the original key even if semantic
    # search is empty. Unavailable lookup keeps UNKNOWN, including with an id.
    runner = ExtractionRunner(config=RunnerConfig(), ledger=Ledger(tmp_path), client=missing)
    key = f"{saved['transcript_sha256']}:{saved['fingerprint']}"
    if unavailable:
        assert runner.reconcile(operation=key, node_id=result_id)[0].status == "unknown"
    missing.unavailable = False
    resolved = runner.reconcile(operation=key, node_id=result_id)[0]
    assert (resolved.status, resolved.node_id) == ("reconciled", result_id)
    assert missing.mutations == 0
    assert len(Ledger(tmp_path)) == 1
    assert Ledger(tmp_path).entries()[0]["state"] == "applied"
    assert run(tmp_path, missing, record, [op])[0].reason == "ledger_duplicate"
    with sqlite3.connect(live_server["db"]) as db:
        # Count this operation's actual recipient effects, independent of recall.
        assert db.execute("SELECT count(*) FROM nodes WHERE id = ?", (result_id,)).fetchone()[0] == 1
    if kind == "teach":
        original = missing.lookup_nodes([op.payload["trace_id"]])["results"][0]
        assert original["superseded_by"] == [result_id]


def _interrupted_native_writer(url, state, record, op, result_path, after_effect):
    class InterruptedClient(LiveMemoryClient):
        async def _call_async(self, tool, arguments):
            if tool != f"memory_{op.kind}":
                return await super()._call_async(tool, arguments)
            if after_effect:
                result = await super()._call_async(tool, arguments)
                Path(result_path).write_text(json.dumps(result))
            os._exit(19)

    run(state, InterruptedClient(url, token=TOKEN), record, [op])


@pytest.mark.parametrize("after_effect", [False, True])
def test_process_interruption_preserves_unknown_before_and_after_effect(live_server, tmp_path, after_effect):
    client = LiveMemoryClient(live_server["url"], token=TOKEN)
    record, op = _native_op(client, "teach", "56" if after_effect else "78")
    state = tmp_path / "state"
    result_path = tmp_path / "recipient-result.json"
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_interrupted_native_writer, args=(client.url, state, record, op, result_path, after_effect))
    try:
        proc.start()
        proc.join(30)
        assert proc.exitcode == 19
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
    assert Ledger(state).entries()[0]["state"] == "unknown"
    missing = MissingLookupClient(client.url, token=TOKEN)
    assert run(state, missing, record, [op])[0].status == "unknown"
    assert missing.mutations == 0
    original = client.lookup_nodes([op.payload["trace_id"]])["results"][0]
    assert len(original["superseded_by"]) == int(after_effect)
    if after_effect:
        # Found provenance and the actual correction relation finish the same op.
        reconciled = ExtractionRunner(config=RunnerConfig(), ledger=Ledger(state), client=client).reconcile()
        assert reconciled[0].status == "reconciled"
        assert reconciled[0].node_id == json.loads(result_path.read_text())["corrective_trace"]["id"]


def test_attest_unknown_replays_receiver_identity_without_duplicate_credit(live_server, tmp_path):
    client = LostReplyClient(live_server["url"], token=TOKEN)
    seeded = client.remember(ORIGINAL_FACT, {"scope": "project:attest-unknown"})["node"]["id"]
    event = client.recall(ORIGINAL_FACT, scope="project:attest-unknown")["recall_event_id"]
    record = _live_record(event, seeded)
    op = proposed_attestations(record)[0]
    client.lost_tool = "memory_attest"
    assert run(tmp_path, client, record, [op])[0].status == "unknown"
    receipt = client.result
    assert receipt["credited"] and receipt["grounded_node_ids"]
    with sqlite3.connect(live_server["db"]) as db:
        before = db.execute("SELECT usefulness_score FROM nodes WHERE id = ?", (seeded,)).fetchone()[0]
    client.lost_tool = ""
    runner = ExtractionRunner(config=RunnerConfig(dry_run=False), ledger=Ledger(tmp_path), client=client)
    result = runner.reconcile()[0]
    assert result.status == "written"
    assert result.attestation["replay"] is True
    assert result.attestation["attestation_id"] == receipt["attestation_id"]
    with sqlite3.connect(live_server["db"]) as db:
        assert db.execute("SELECT count(*) FROM recall_attestations WHERE recall_event_id = ?", (event,)).fetchone()[0] == 1
        assert db.execute("SELECT usefulness_score FROM nodes WHERE id = ?", (seeded,)).fetchone()[0] == before
    assert len(Ledger(tmp_path)) == 1


def test_cli_unknown_is_visible_without_a_proposal_or_paid_judge(tmp_path, monkeypatch, capsys):
    cli = _load_cli()
    client = Recipient(ClientError("lost reply"))
    record = make_record()
    run(tmp_path, client, record, [make_remember(record, "Saved unknown operation.")])
    monkeypatch.setattr(cli, "LiveMemoryClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(cli, "make_judge_factory", lambda *args: pytest.fail("reconciliation must not pay a judge"))
    args = ["--reconcile", "--state-dir", str(tmp_path), "--url", client.url]
    assert cli.main(args) == EXIT_UNKNOWN
    assert "UNKNOWN" in capsys.readouterr().err
    # Mutation continuation still enforces the existing gate.
    assert cli.main(args + ["--write", "--gate-verdict", str(tmp_path / "missing")]) == EXIT_GATE
    assert len(client.effects) == 1


def test_native_pre_dispatch_auth_failure_can_resume(live_server, tmp_path):
    record = make_record(transcript_sha256="de" * 32)
    op = make_remember(record, "A connection authentication refusal does not send a tool mutation.")
    rejected = LiveMemoryClient(live_server["url"], token="invalid-test-token", timeout=2)
    decision = run(tmp_path, rejected, record, [op])[0]
    assert (decision.status, decision.reason) == ("failed", "pre_dispatch")
    assert Ledger(tmp_path).entries()[0]["state"] == "ready"
    accepted = LiveMemoryClient(live_server["url"], token=TOKEN)
    assert ExtractionRunner(
        config=RunnerConfig(dry_run=False), ledger=Ledger(tmp_path), client=accepted,
    ).reconcile()[0].status == "written"
    assert len(Ledger(tmp_path)) == 1


def test_teach_confirmation_requires_the_actual_relation(live_server, tmp_path):
    client = LiveMemoryClient(live_server["url"], token=TOKEN)
    record, op = _native_op(client, "teach", "ef")
    op = enforce_attribution(op, record, task="test")
    # Equal text and even supplied correction metadata do not create an edge.
    twin = client.remember(op.payload["correction"], {
        **op.payload["context"], "scope": "project:demo",
    })["node"]["id"]
    ledger = Ledger(tmp_path)
    assert ledger.begin(op, client.url)[0]
    runner = ExtractionRunner(config=RunnerConfig(), ledger=ledger, client=client)
    key = f"{record.transcript_sha256}:{op_fingerprint(op)}"
    assert runner.reconcile(operation=key, node_id=twin)[0].status == "unknown"
    assert ledger.entries()[0]["state"] == "unknown"


def test_cli_missing_output_does_not_reopen_applied_operation(tmp_path, monkeypatch):
    cli = _load_cli()
    client = Recipient()
    record = make_record()
    op = make_remember(record, "The report is not write admission state.")
    # Use a known unsent request so --reconcile --write exercises the CLI's
    # output handling without proposal generation or a paid judge.
    client.failure = PreDispatchError("not connected")
    run(tmp_path / "state", client, record, [op])
    client.failure = None
    monkeypatch.setattr(cli, "LiveMemoryClient", lambda *args, **kwargs: client)
    from living_memory.postsession.runner import GateDecision
    monkeypatch.setattr(cli, "evaluate_gate", lambda *args, **kwargs: GateDecision("PASS"))
    blocked_output = tmp_path / "output"
    blocked_output.mkdir()
    args = ["--reconcile", "--write", "--state-dir", str(tmp_path / "state"), "--out", str(blocked_output)]
    assert cli.main(args) == 2
    assert len(client.effects) == 1
    assert Ledger(tmp_path / "state").entries()[0]["state"] == "applied"
    assert cli.main(args[:-2]) == 0
    assert len(client.effects) == 1
