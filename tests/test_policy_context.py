"""Pytest owns policy defaults across fixture scopes and child processes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("inherited", [False, True], ids=["clean", "host-policies"])
def test_policy_context_precedes_probes_and_allows_overrides(
    tmp_path: Path, inherited: bool
) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_probe.py").write_text(
        '''
import json
import os
import subprocess
import sys

import pytest

from living_memory.feedback import _implicit_link_policy, _recall_credit_policy
from living_memory.score_gate import gate_form_from_env, min_score_from_env


def policy():
    return [_implicit_link_policy(), _recall_credit_policy(),
            min_score_from_env(), gate_form_from_env()]


def child_policy():
    return json.loads(subprocess.check_output(
        [sys.executable, "-c",
         "import json; from test_probe import policy; print(json.dumps(policy()))"],
        text=True, timeout=30,
    ))


@pytest.fixture(scope="module")
def probe():
    assert policy() == ["all", "grounded", None, "drop"]
    return child_policy()


def test_default(probe):
    assert probe == policy() == ["all", "grounded", None, "drop"]
    assert os.environ["LIVING_MEMORY_EMBEDDING_BACKEND"] == "hash"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["LIVING_MEMORY_DEPS_DIR"] == "fixture-deps"
    assert os.environ["LM_DB_PATH"] == "fixture.sqlite3"
    assert os.environ["LM_DELIVERY_SNIPPET_CHARS"] == "321"


def test_override(monkeypatch):
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", "credited")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "all")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    monkeypatch.setenv("LM_RECALL_GATE_FORM", "stub")
    assert policy() == child_policy() == ["credited", "all", 0.35, "stub"]


def test_restored():
    assert policy() == child_policy() == ["all", "grounded", None, "drop"]
''',
        encoding="utf-8",
    )
    environment = dict(os.environ)
    policies = {
        "LM_IMPLICIT_LINK_POLICY": "credited",
        "LM_RECALL_CREDIT_POLICY": "grounded_negative",
        "LM_RECALL_MIN_SCORE": "0.35",
        "LM_RECALL_GATE_FORM": "stub",
    }
    for name, value in policies.items():
        if inherited:
            environment[name] = value
        else:
            environment.pop(name, None)
    environment.update(
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
        LIVING_MEMORY_EMBEDDING_BACKEND="hash",
        HF_HUB_OFFLINE="1",
        LIVING_MEMORY_DEPS_DIR="fixture-deps",
        LM_DB_PATH="fixture.sqlite3",
        LM_DELIVERY_SNIPPET_CHARS="321",
    )
    completed = subprocess.run(
        [sys.executable, "-c", '''
import os
import pytest

def settings():
    return {k: v for k, v in os.environ.items()
            if k.startswith(("LM_", "LIVING_MEMORY_", "HF_")) or k == "PYTHONPATH"}

before = settings()
result = pytest.main(["-q", "test_probe.py"])
assert settings() == before, "pytest must restore its caller's environment"
raise SystemExit(result)
'''],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "3 passed" in completed.stdout
