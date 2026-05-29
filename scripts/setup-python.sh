#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DEPS_DIR="${LIVING_MEMORY_DEPS_DIR:-${ROOT_DIR}/.cache/python-deps}"
MARKER="${DEPS_DIR}/.living-memory-deps-ready"

# Fingerprint of the dependency-defining inputs. We reinstall only when this
# content changes — NOT on mtime changes. A git worktree checkout (used by the
# AE merge gate for every node attempt) resets file mtimes, so the previous
# `-nt pyproject.toml` test treated the freshly-checked-out pyproject as "newer"
# than the cache and forced a multi-GB torch/sentence-transformers reinstall on
# every gate run, blowing the gate test timeout. A content hash is stable.
_deps_fingerprint() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${ROOT_DIR}/pyproject.toml" 2>/dev/null | awk '{print $1}'
    else
        "${PYTHON_BIN}" -c "import hashlib;print(hashlib.sha256(open('${ROOT_DIR}/pyproject.toml','rb').read()).hexdigest())" 2>/dev/null
    fi
}

# Reuse the prebuilt cache from the main checkout when this tree has none yet.
# Fresh AE node-exec worktrees (.worktrees/_node_exec_*) start without a
# .cache/python-deps, so without this they cold-reinstall ~5GB on every gate
# run. Only auto-link when the deps dir was not explicitly overridden.
if [[ -z "${LIVING_MEMORY_DEPS_DIR:-}" && ! -e "${DEPS_DIR}" ]]; then
    common_git="$(git -C "${ROOT_DIR}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [[ -n "${common_git}" ]]; then
        main_deps="$(dirname "${common_git}")/.cache/python-deps"
        if [[ -d "${main_deps}" && "${main_deps}" != "${DEPS_DIR}" \
              && -f "${main_deps}/.living-memory-deps-ready" ]]; then
            mkdir -p "${ROOT_DIR}/.cache"
            ln -s "${main_deps}" "${DEPS_DIR}"
        fi
    fi
fi

mkdir -p "${DEPS_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Idempotent: skip the install when the cache already matches the current
# dependency spec (fingerprint) and the runtime imports cleanly.
want_fp="$(_deps_fingerprint)"
if [[ -f "${MARKER}" && -n "${want_fp}" && "$(cat "${MARKER}" 2>/dev/null)" == "${want_fp}" ]]; then
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

printf '%s\n' "${want_fp}" > "${MARKER}"
