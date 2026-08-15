"""A deleted test directory must not abort the whole run.

The merge gate derives its targeted-test paths from ``git diff --name-only``,
which lists deleted files too, so a change that removes the last Python file in
a directory hands pytest a path that no longer exists. Without the root
conftest hook pytest answers with a usage error (exit 4) and the suite never
runs — a deletion cannot be verified by the suite it shrinks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import conftest


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_CONFTEST = REPO_ROOT / "conftest.py"


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    return env


def _sandbox(tmp_path: Path) -> Path:
    """A throwaway project carrying this repository's real root conftest."""

    shutil.copy2(ROOT_CONFTEST, tmp_path / "conftest.py")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_present.py").write_text(
        "def test_present() -> None:\n    assert True\n", encoding="utf-8"
    )
    return tmp_path


def _run_pytest(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-q", *args],
        cwd=cwd,
        env=_child_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_missing_target_is_dropped_when_another_target_survives(tmp_path: Path) -> None:
    project = _sandbox(tmp_path)

    completed = _run_pytest(project, "suite", "deleted/machinery/dir")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 passed" in completed.stdout
    assert "deleted/machinery/dir" in completed.stderr


def test_every_target_missing_still_fails_as_a_usage_error(tmp_path: Path) -> None:
    project = _sandbox(tmp_path)

    completed = _run_pytest(project, "tsets")

    assert completed.returncode == 4, completed.stdout + completed.stderr
    assert "file or directory not found: tsets" in completed.stdout + completed.stderr


def test_existing_targets_are_left_exactly_as_requested(tmp_path: Path) -> None:
    kept, dropped = conftest.prune_missing_target_paths(
        ["tests", "tests/test_stale_target_paths.py::test_kept"], REPO_ROOT
    )

    assert kept == ["tests", "tests/test_stale_target_paths.py::test_kept"]
    assert dropped == []


def test_node_id_target_follows_the_existence_of_its_file(tmp_path: Path) -> None:
    kept, dropped = conftest.prune_missing_target_paths(
        ["tests", "tests/test_retired_machinery.py::test_gone"], REPO_ROOT
    )

    assert kept == ["tests"]
    assert dropped == ["tests/test_retired_machinery.py::test_gone"]


def test_all_missing_targets_are_returned_untouched(tmp_path: Path) -> None:
    args = ["gone/one", "gone/two"]

    kept, dropped = conftest.prune_missing_target_paths(args, REPO_ROOT)

    assert kept == args
    assert dropped == []
