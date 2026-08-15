"""Pytest path setup for checkout-local development."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, Sequence

import pytest

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DEPS_DIR = ROOT_DIR / ".cache" / "python-deps"

if SRC_DIR.exists():
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)

if DEPS_DIR.exists():
    deps = str(DEPS_DIR)
    if deps not in sys.path:
        insert_at = 1 if str(SRC_DIR) in sys.path else 0
        sys.path.insert(insert_at, deps)


def prune_missing_target_paths(
    args: Sequence[str], invocation_dir: Path
) -> tuple[list[str], list[str]]:
    """Split requested targets into (kept, dropped-because-they-do-not-exist).

    Nothing is dropped unless at least one requested target survives: a run
    whose every target is missing keeps its original arguments so pytest still
    reports the usage error.
    """

    kept: list[str] = []
    dropped: list[str] = []
    for arg in args:
        path_part = arg.split("::", 1)[0]
        if not path_part or (invocation_dir / path_part).exists():
            kept.append(arg)
        else:
            dropped.append(arg)
    if not kept:
        return list(args), []
    return kept, dropped


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Ignore requested test targets that the change under test deleted.

    The merge gate runs targeted tests by handing pytest the directory of every
    changed ``*.py`` file from ``git diff --name-only base...branch``. That list
    includes files the change *deleted*, so removing the last Python file in a
    directory points pytest at a path that is gone, and pytest aborts the whole
    run with a usage error (exit 4) before collecting a single test — the suite
    never runs, and a deletion-only change cannot be verified at all.

    Drop such targets loudly, and only while another target survives, so a
    mistyped path (``pytest tsets/``) still fails the way it always has.
    """

    if getattr(config.option, "pyargs", False):
        return
    kept, dropped = prune_missing_target_paths(
        config.args, Path(config.invocation_params.dir)
    )
    if not dropped:
        return
    print(
        "conftest: ignoring requested test target(s) that do not exist: "
        + ", ".join(dropped),
        file=sys.stderr,
    )
    config.args = kept


@pytest.fixture(autouse=True)
def _clear_lm_policy_env() -> Iterator[None]:
    """Keep ambient MCP server env vars from changing unit-test defaults."""

    saved = {
        name: os.environ.pop(name, None)
        for name in (
            "LM_AUTH_TOKEN",
            "LM_AUTO_CONSOLIDATE_POLICY",
            "LM_RETRIEVAL_TUNING_POLICY",
            "LM_DECAY_SWEEP_INTERVAL_SEC",
        )
    }
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
