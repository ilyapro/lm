"""Dependency preparation borrows ready caches and owns installation targets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MARKER = ".living-memory-deps-ready"
CONSUMER = """
if __name__ == '__main__':
    import fastmcp, json, os, pathlib, sys
    with open(os.environ['SETUP_TEST_LOG'], 'a') as log:
        log.write(json.dumps({'consumer': __file__, 'deps': fastmcp.__file__,
                              'args': sys.argv[1:]}) + '\\n')
"""


class Setup:
    def __init__(self, root: Path):
        self.root = root / "candidate"
        self.main = root / "main" / ".cache" / "python-deps"
        self.deps = self.root / ".cache" / "python-deps"
        self.log = root / "calls.jsonl"
        scripts = self.root / "scripts"
        scripts.mkdir(parents=True)
        for name in ("setup-python.sh", "test.sh", "server.sh"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        spec = b'[project]\nname = "setup-fixture"\nversion = "0.0.0"\n'
        (self.root / "pyproject.toml").write_bytes(spec)
        self.fingerprint = hashlib.sha256(spec).hexdigest()
        package = self.root / "src" / "living_memory"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "server.py").write_text(CONSUMER)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        git = bin_dir / "git"
        git.write_text('#!/bin/sh\nprintf "%s\\n" "$SETUP_TEST_COMMON_GIT"\n')
        git.chmod(0o755)
        python = bin_dir / "python"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "if sys.argv[1:3] == ['-m', 'pip']:\n"
            "    target = pathlib.Path(sys.argv[sys.argv.index('--target') + 1])\n"
            "    with open(os.environ['SETUP_TEST_LOG'], 'a') as log:\n"
            "        log.write(json.dumps({'target': str(target), 'resolved': str(target.resolve())}) + '\\n')\n"
            "    if os.environ.get('SETUP_TEST_WHEELS'):\n"
            "        wheels = sorted(pathlib.Path(os.environ['SETUP_TEST_WHEELS']).glob('*.whl'))\n"
            f"        os.execv(sys.executable, [sys.executable, {str(Path(importlib.util.find_spec('pip').origin).parent)!r}, *sys.argv[3:-1],\n"
            "                 '--no-index', '--no-deps', *map(str, wheels)])\n"
            "    if os.environ.get('SETUP_TEST_FAIL') == 'pip':\n"
            "        sys.exit(1)\n"
            "    target.mkdir(parents=True, exist_ok=True)\n"
            "    broken = os.environ.get('SETUP_TEST_FAIL') == 'imports'\n"
            "    (target / 'fastmcp.py').write_text('raise ImportError(\"unusable\")' if broken else '')\n"
            f"    (target / 'pytest.py').write_text({CONSUMER!r})\n"
            "else:\n"
            "    os.execv(sys.executable, [sys.executable, '-S', *sys.argv[1:]])\n"
        )
        python.chmod(0o755)
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
            "HOME": str(root),
            "PYTHON": str(python),
            "SETUP_TEST_LOG": str(self.log),
            "SETUP_TEST_COMMON_GIT": str(root / "main" / ".git"),
        }

    def cache(self, path: Path, state: str = "ready") -> None:
        path.mkdir(parents=True)
        (path / "fastmcp.py").write_text(
            'raise ImportError("unusable")' if state == "imports" else ""
        )
        (path / "pytest.py").write_text(CONSUMER)
        (path / "sentinel").write_text("owned by this cache")
        if state != "unmarked":
            (path / MARKER).write_text(
                ("different-spec" if state == "mismatch" else self.fingerprint) + "\n"
            )

    def link(self) -> None:
        self.deps.parent.mkdir(parents=True)
        self.deps.symlink_to(self.main, target_is_directory=True)

    def run(self, script: str = "setup-python.sh", *args: str, ok: bool = True) -> None:
        result = subprocess.run(
            ["bash", str(self.root / "scripts" / script), *args],
            cwd=self.root, env=self.env, text=True, capture_output=True, timeout=20,
        )
        assert (result.returncode == 0) == ok, result.stdout + result.stderr

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def snapshot(path: Path) -> dict:
    return {
        str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in path.rglob("*") if p.is_file()
    }


def wheels(path: Path, version: str) -> None:
    """Small offline wheels exercise pip's actual target upgrade behavior."""
    path.mkdir()
    modules = {
        "fastmcp": "",
        "pytest": CONSUMER,
        "huggingface_hub": f"__version__ = {version!r}\n",
        "transformers": (
            "from importlib.metadata import version\n"
            f"if version('huggingface_hub') != {version!r}:\n"
            "    raise ImportError('hub metadata does not match the installed encoder')\n"
        ),
        "sentence_transformers": "import transformers\n",
    }
    for name, source in modules.items():
        info = f"{name}-{version}.dist-info"
        files = {
            f"{name}/__init__.py": source,
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: setup-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        }
        if name == "transformers" and version == "1.0":
            files["transformers/removed_module.py"] = ""
        files[f"{info}/RECORD"] = "".join(f"{name},,\n" for name in files) + f"{info}/RECORD,,\n"
        with zipfile.ZipFile(path / f"{name}-{version}-py3-none-any.whl", "w") as wheel:
            for name, content in files.items():
                wheel.writestr(name, content)


def test_upgrade_replaces_the_environment_seen_by_server(setup: Setup, tmp_path: Path) -> None:
    for version in ("1.0", "2.0"):
        wheel_dir = tmp_path / version
        wheels(wheel_dir, version)
        setup.env["SETUP_TEST_WHEELS"] = str(wheel_dir)
        with (setup.root / "pyproject.toml").open("a") as spec:
            spec.write(f"# dependency set {version}\n")
        setup.run()

    server = setup.root / "src" / "living_memory" / "server.py"
    server.write_text(
        "from importlib.metadata import distributions, version\n"
        "from pathlib import Path\n"
        "import os, sentence_transformers, transformers\n"
        "target = Path(os.environ['SETUP_TEST_TARGET'])\n"
        "assert version('huggingface_hub') == '2.0'\n"
        "assert sorted((d.metadata['Name'], d.version) for d in distributions(path=[str(target)])) == "
        "[(name, '2.0') for name in ['fastmcp', 'huggingface_hub', 'pytest', 'sentence_transformers', 'transformers']]\n"
        "assert not (target / 'transformers' / 'removed_module.py').exists()\n"
        + CONSUMER
    )
    setup.env["SETUP_TEST_TARGET"] = str(setup.deps)
    setup.env["PYTHONDONTWRITEBYTECODE"] = "1"
    before = snapshot(setup.deps)
    setup.run("server.sh")
    setup.run()
    assert snapshot(setup.deps) == before
    assert len([call for call in setup.calls() if "target" in call]) == 2


@pytest.mark.parametrize("cache", ["new-link", "existing-link", "local"])
def test_ready_cache_is_reused_without_writes(setup: Setup, cache: str) -> None:
    target = setup.deps if cache == "local" else setup.main
    setup.cache(target)
    if cache == "existing-link":
        setup.link()
    before = snapshot(target)

    setup.run()
    setup.run()

    assert setup.calls() == []
    assert setup.deps.resolve() == target
    assert snapshot(target) == before


@pytest.mark.parametrize("linked", [False, True], ids=["new", "existing-link"])
@pytest.mark.parametrize("state", ["mismatch", "imports", "unmarked"])
def test_unusable_shared_cache_is_preserved(setup: Setup, linked: bool, state: str) -> None:
    setup.cache(setup.main, state)
    before = snapshot(setup.main)
    if linked:
        setup.link()

    setup.run()
    setup.run()

    assert not setup.deps.is_symlink()
    assert setup.calls() == [{"target": str(setup.deps), "resolved": str(setup.deps)}]
    assert (setup.deps / MARKER).read_text() == setup.fingerprint + "\n"
    assert snapshot(setup.main) == before


def test_dangling_borrowed_link_becomes_local(setup: Setup) -> None:
    setup.link()
    setup.run()
    assert not setup.main.exists()
    assert not setup.deps.is_symlink()
    assert setup.calls() == [{"target": str(setup.deps), "resolved": str(setup.deps)}]


@pytest.mark.parametrize("linked", [False, True], ids=["directory", "symlink"])
@pytest.mark.parametrize("state", ["ready", "mismatch"])
def test_explicit_override_owns_selected_target(setup: Setup, linked: bool, state: str) -> None:
    setup.cache(setup.main)
    before = snapshot(setup.main)
    selected = setup.root / "selected"
    target = setup.root / "override-target" if linked else selected
    setup.cache(target, state)
    if linked:
        selected.symlink_to(target, target_is_directory=True)
    setup.env["LIVING_MEMORY_DEPS_DIR"] = str(selected)

    setup.run()
    setup.run()

    expected = [] if state == "ready" else [{"target": str(selected), "resolved": str(target)}]
    assert setup.calls() == expected
    assert selected.is_symlink() == linked
    assert not setup.deps.exists()
    assert (target / MARKER).read_text() == setup.fingerprint + "\n"
    assert snapshot(setup.main) == before


@pytest.mark.parametrize("script", ["test.sh", "server.sh"])
@pytest.mark.parametrize("override", [False, True])
def test_consumers_use_installed_directory(setup: Setup, script: str, override: bool) -> None:
    setup.cache(setup.main, "mismatch")
    before = snapshot(setup.main)
    setup.link()
    selected = setup.root / "selected" if override else setup.deps
    if override:
        setup.env["LIVING_MEMORY_DEPS_DIR"] = str(selected)

    setup.run(script, "consumer-argument")

    install, consumer = setup.calls()
    assert install == {"target": str(selected), "resolved": str(selected)}
    assert consumer["deps"] == str(selected / "fastmcp.py")
    assert consumer["args"] == ["consumer-argument"]
    assert snapshot(setup.main) == before


@pytest.mark.parametrize("failure", ["pip", "imports"])
def test_failed_local_preparation_leaves_borrowed_cache_intact(setup: Setup, failure: str) -> None:
    setup.cache(setup.main, "mismatch")
    setup.link()
    before = snapshot(setup.main)
    setup.env["SETUP_TEST_FAIL"] = failure

    setup.run(ok=False)

    assert not setup.deps.is_symlink()
    assert not (setup.deps / MARKER).exists()
    assert snapshot(setup.main) == before


@pytest.mark.parametrize("consumer", ["conftest", "living_memory"])
@pytest.mark.parametrize("override", ["unset", "empty", "selected"])
def test_checkout_imports_use_selected_directory(tmp_path: Path, consumer: str, override: str) -> None:
    shutil.copy2(ROOT / "conftest.py", tmp_path / "conftest.py")
    if consumer == "living_memory":
        bootstrap = ROOT / "living_memory" / "__init__.py"
        if not bootstrap.is_file():
            pytest.skip("This src-layout checkout has no import bootstrap")
        package = tmp_path / "living_memory"
        package.mkdir()
        shutil.copy2(bootstrap, package / "__init__.py")
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    default = tmp_path / ".cache" / "python-deps"
    selected = tmp_path / "selected"
    for directory in (default, selected):
        directory.mkdir(parents=True)
        (directory / "setup_dependency_probe.py").write_text("")
    env = dict(os.environ)
    env.pop("LIVING_MEMORY_DEPS_DIR", None)
    if override != "unset":
        env["LIVING_MEMORY_DEPS_DIR"] = str(selected) if override == "selected" else ""
    env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    result = subprocess.run(
        [sys.executable, "-c", f"import {consumer}; import setup_dependency_probe as probe; print(probe.__file__)"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = selected if override == "selected" else default
    assert result.stdout.strip() == str(expected / "setup_dependency_probe.py")
