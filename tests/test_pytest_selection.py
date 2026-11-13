"""Checkout pytest preserves explicit target selection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "conftest.py").symlink_to(root / "conftest.py")
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = selection_suite\n", encoding="utf-8"
    )
    suite = tmp_path / "selection_suite"
    suite.mkdir()
    (suite / "__init__.py").touch()
    (suite / "test_present.py").write_text(
        "def test_first():\n    assert True\n\n"
        "def test_second():\n    assert True\n",
        encoding="utf-8",
    )
    return tmp_path


def _run_pytest(
    project: Path, *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        LIVING_MEMORY_EMBEDDING_BACKEND="hash",
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-q",
         "-c", str(project / "pytest.ini"), "--confcutdir", str(project), *args],
        cwd=cwd or project,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "args, count",
    [
        ((), 2),
        (("selection_suite",), 2),
        (("selection_suite/test_present.py",), 2),
        (("selection_suite/test_present.py::test_first",), 1),
        (("--pyargs", "selection_suite.test_present"), 2),
    ],
    ids=["bare", "directory", "file", "nodeid", "pyargs"],
)
def test_valid_selection(project: Path, args: tuple[str, ...], count: int) -> None:
    completed = _run_pytest(project, *args)

    assert completed.returncode == pytest.ExitCode.OK, completed.stdout + completed.stderr
    assert f"{count} passed" in completed.stdout


@pytest.mark.parametrize(
    "missing",
    [
        "selection_suite/test_absent.py",
        "absent_directory",
        "selection_suite/test_absent.py::test_required",
        "selection_suite/test_present.py::test_absent",
    ],
    ids=["file", "directory", "file-nodeid", "nodeid"],
)
@pytest.mark.parametrize("with_present", [False, True], ids=["alone", "mixed"])
def test_missing_target_fails(project: Path, missing: str, with_present: bool) -> None:
    args = ["selection_suite/test_present.py::test_first"] if with_present else []
    completed = _run_pytest(project, *args, missing)

    assert completed.returncode == pytest.ExitCode.USAGE_ERROR, completed.stdout + completed.stderr
    assert missing in completed.stderr


def test_missing_target_before_valid_target_fails(project: Path) -> None:
    completed = _run_pytest(project, "absent_directory", "selection_suite")

    assert completed.returncode == pytest.ExitCode.USAGE_ERROR, completed.stdout + completed.stderr


def test_missing_pyargs_module_fails(project: Path) -> None:
    completed = _run_pytest(
        project, "--pyargs", "selection_suite.test_present", "selection_suite.test_absent"
    )

    assert completed.returncode == pytest.ExitCode.USAGE_ERROR, completed.stdout + completed.stderr


@pytest.mark.parametrize("node", ["test_first", "test_absent"])
def test_overlapping_nodeid_with_keep_duplicates(project: Path, node: str) -> None:
    completed = _run_pytest(
        project, "--keep-duplicates", "selection_suite/test_present.py",
        f"selection_suite/test_present.py::{node}",
    )

    if node == "test_first":
        assert completed.returncode == pytest.ExitCode.OK, completed.stdout + completed.stderr
        assert "3 passed" in completed.stdout
    else:
        assert completed.returncode == pytest.ExitCode.USAGE_ERROR, completed.stdout + completed.stderr


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_selection_from_subdirectory(project: Path, absolute: bool) -> None:
    target = "selection_suite/test_present.py::test_second"
    target = str(project / target) if absolute else f"../{target}"
    completed = _run_pytest(project, target, cwd=project / "selection_suite")

    assert completed.returncode == pytest.ExitCode.OK, completed.stdout + completed.stderr
    assert "1 passed" in completed.stdout

    completed = _run_pytest(
        project, target, str(project / "test_absent.py"), cwd=project / "selection_suite"
    )

    assert completed.returncode == pytest.ExitCode.USAGE_ERROR, completed.stdout + completed.stderr
