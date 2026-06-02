#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
DEPS_DIR="${LIVING_MEMORY_DEPS_DIR:-${ROOT_DIR}/.cache/python-deps}"

"${SCRIPT_DIR}/setup-python.sh"

export PYTHONPATH="${ROOT_DIR}/src:${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m living_memory.server "$@"
