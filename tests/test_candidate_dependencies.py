"""Exercise the compare command's candidate interpreter without a service."""

import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("environment", ["checkout", "override", "foreign-only"])
def test_compare_uses_only_selected_dependencies(tmp_path, monkeypatch, capsys, environment):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import observe_live_task_recall as observer

    installed = tmp_path / "installation"
    candidate = tmp_path / "repository"
    package = candidate / "src/living_memory"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (candidate / "pyproject.toml").touch()
    (package / "server.py").write_text(
        "from types import SimpleNamespace\n"
        "from fixture_delivery_dependency import digest\n"
        "def create_mcp_server(path):\n"
        "    return SimpleNamespace(memory_store=SimpleNamespace(close=lambda: None))\n"
        "def _loaded_code_identity():\n"
        f"    return dict(status='known', scheme={observer.CODE_IDENTITY_SCHEME!r}, digest=digest)\n"
    )
    subprocess.run(["git", "init", "-q", str(candidate)], check=True)
    subprocess.run(["git", "add", "."], cwd=candidate, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-qm", "candidate"], cwd=candidate, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True).strip()

    cache = installed / ".cache/python-deps"
    override = tmp_path / "selected"
    foreign = candidate / ".cache/python-deps"
    for directory, value in [(cache, "a"), (override, "b"), (foreign, "c")]:
        stale_package = directory / "living_memory"
        stale_package.mkdir(parents=True)
        (stale_package / "__init__.py").write_text("raise ImportError('stale package')")
        (directory / "fixture_delivery_dependency.py").write_text(f"digest = {value!r} * 64")

    monkeypatch.chdir(candidate)
    monkeypatch.delenv("LIVING_MEMORY_DEPS_DIR", raising=False)
    paths = [foreign, package.parent]
    if environment != "foreign-only":
        paths.append(cache)
    if environment == "override":
        monkeypatch.setenv("LIVING_MEMORY_DEPS_DIR", "../selected")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, paths)))
    digest = "b" * 64 if environment == "override" else "a" * 64
    monkeypatch.setattr(observer, "_health", lambda *args: {
        "ok": True, "boot_id": "fixture", "code_identity": {
            "status": "known", "scheme": observer.CODE_IDENTITY_SCHEME, "digest": digest,
        },
    })

    code = observer.main(["compare", "--candidate-root", ".", "--installed-root", str(installed),
                          "--published-commit", commit])
    report = json.loads(capsys.readouterr().out)
    assert code == 1  # The installation contains dependencies only.
    assert report["disk"]["status"] == "mismatch"
    if environment == "foreign-only":
        assert report["process"] == {"status": "unknown", "reason": "candidate function identity unavailable"}
    else:
        assert report["process"]["status"] == "match"
        assert report["process"]["candidate_digest"] == digest
