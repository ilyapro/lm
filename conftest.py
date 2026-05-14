"""Pytest path setup for checkout-local development."""

from __future__ import annotations

import sys
from pathlib import Path

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
