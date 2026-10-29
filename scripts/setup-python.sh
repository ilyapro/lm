#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
DEPS_DIR="${LIVING_MEMORY_DEPS_DIR:-${ROOT_DIR}/.cache/python-deps}"
MARKER="${DEPS_DIR}/.living-memory-deps-ready"

# Cache readiness is tied to the contents of pyproject.toml.
_deps_fingerprint() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${ROOT_DIR}/pyproject.toml" 2>/dev/null | awk '{print $1}'
    else
        "${PYTHON_BIN}" -c "import hashlib;print(hashlib.sha256(open('${ROOT_DIR}/pyproject.toml','rb').read()).hexdigest())" 2>/dev/null
    fi
}

want_fp="$(_deps_fingerprint)"

_deps_ready() {
    local marker="$1/.living-memory-deps-ready"
    [[ -f "${marker}" && -n "${want_fp}" && "$(cat "${marker}" 2>/dev/null)" == "${want_fp}" ]] || return 1
    PYTHONPATH="${ROOT_DIR}/src:$1${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -B - <<'PY' >/dev/null 2>&1
import fastmcp
import pytest
PY
}

# Borrow the main checkout's cache only after a read-only readiness check.
if [[ -z "${LIVING_MEMORY_DEPS_DIR:-}" && ! -e "${DEPS_DIR}" && ! -L "${DEPS_DIR}" ]]; then
    common_git="$(git -C "${ROOT_DIR}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [[ -n "${common_git}" ]]; then
        main_deps="$(dirname "${common_git}")/.cache/python-deps"
        if [[ -d "${main_deps}" && "${main_deps}" != "${DEPS_DIR}" \
              ]] && _deps_ready "${main_deps}"; then
            mkdir -p "${ROOT_DIR}/.cache"
            ln -s "${main_deps}" "${DEPS_DIR}"
            exit 0
        fi
    fi
fi

if _deps_ready "${DEPS_DIR}"; then
    exit 0
fi

# A default symlink is borrowed storage. Replace only the link before writing;
# an explicit override authorizes installation into the caller's chosen target.
if [[ -z "${LIVING_MEMORY_DEPS_DIR:-}" && -L "${DEPS_DIR}" ]]; then
    rm -- "${DEPS_DIR}"
fi
mkdir -p "${DEPS_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m pip install --upgrade --target "${DEPS_DIR}" "${ROOT_DIR}[test]"

"${PYTHON_BIN}" - <<'PY'
import fastmcp
import pytest
PY

printf '%s\n' "${want_fp}" > "${MARKER}"
