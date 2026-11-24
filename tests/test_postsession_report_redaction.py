"""Exercise both extraction report writers with synthetic data only."""

import json
from types import SimpleNamespace

import pytest

from living_memory.postsession.runner import GateDecision
from scripts import post_session_extract as cli
from scripts import postsession_field_run as field


def report_fields():
    return {
        "detail": 'PASSWORD="alpha bravo"; status=ok',
        "nested": [{"App_Api_Key": ["synthetic field value"]}],
        "ordinary": ['путь "quoted"\\file\nnext', 3, True, None],
        "token_count": 5,
    }


def check_report(path, field_name):
    body = path.read_text(encoding="utf-8")
    report = json.loads(body)
    assert "alpha bravo" not in body
    assert "synthetic field value" not in body
    fields = report[field_name]
    assert fields["detail"] == "PASSWORD=<redacted>; status=ok"
    assert fields["nested"] == [{"App_Api_Key": "<redacted>"}]
    assert fields["ordinary"] == report_fields()["ordinary"]
    assert fields["token_count"] == 5
    return report


@pytest.mark.parametrize("split, label", [("eval", "<reserved-split:h>"), ("holdout", "HOLDOUT")])
def test_cli_report(tmp_path, split, label):
    path = tmp_path / "cli.json"
    cli._write_report(path, {"fields": report_fields(), "label": "HOLDOUT"}, splits_read=[split])
    assert check_report(path, "fields")["label"] == label


def test_field_report(tmp_path, monkeypatch):
    (tmp_path / "preregistration.json").write_text("{}")
    monkeypatch.setattr(field, "DEFAULT_STATE_DIR", tmp_path)
    monkeypatch.setattr(field, "evaluate_gate", lambda *args, **kwargs: GateDecision("PASS"))
    monkeypatch.setattr(field, "_select_sessions", lambda *args: ([], {"eligible": 0, **report_fields()}))
    assert field.cmd_extract(SimpleNamespace(split="eval", state_dir=tmp_path)) == 0
    report = check_report(tmp_path / "extract-eval.json", "selection")
    assert report["judge"]["calls"] == 0
    assert report["splits_read"] == ["eval"]
