"""Pytest path setup for checkout-local development."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

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
