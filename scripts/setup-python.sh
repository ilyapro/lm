#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DEPS_DIR="${LIVING_MEMORY_DEPS_DIR:-${ROOT_DIR}/.cache/python-deps}"
MARKER="${DEPS_DIR}/.living-memory-deps-ready"

mkdir -p "${DEPS_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${MARKER}" && "${MARKER}" -nt "${ROOT_DIR}/pyproject.toml" && "${MARKER}" -nt "${BASH_SOURCE[0]}" ]]; then
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import fastmcp
import pytest
PY
    then
        exit 0
    fi
fi

"${PYTHON_BIN}" -m pip install --upgrade --target "${DEPS_DIR}" "${ROOT_DIR}[test]"

"${PYTHON_BIN}" - <<'PY'
import fastmcp
import pytest
PY

touch "${MARKER}"
