#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
DEPS_DIR="${LIVING_MEMORY_DEPS_DIR:-${ROOT_DIR}/.cache/python-deps}"

"${SCRIPT_DIR}/setup-python.sh"

export PYTHONPATH="${ROOT_DIR}/src:${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Default to the lightweight hash embedding backend so the suite does not load
# torch/sentence-transformers — that real-model load makes the full suite
# exceed the merge gate's test timeout. Tests that exercise the real model path
# clear this var themselves. Override by exporting it explicitly.
export LIVING_MEMORY_EMBEDDING_BACKEND="${LIVING_MEMORY_EMBEDDING_BACKEND:-hash}"

exec "${PYTHON_BIN}" -m pytest "$@"
