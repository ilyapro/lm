#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
DEPS_DIR="${LIVING_MEMORY_DEPS_DIR:-${ROOT_DIR}/.cache/python-deps}"

# Cache readiness is tied to the contents of pyproject.toml.
_deps_fingerprint() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${ROOT_DIR}/pyproject.toml" 2>/dev/null | awk '{print $1}'
    else
        "${PYTHON_BIN}" -c "import hashlib;print(hashlib.sha256(open('${ROOT_DIR}/pyproject.toml','rb').read()).hexdigest())" 2>/dev/null
    fi
}

# This marker attests to a complete target prepared by this writer.
want_fp="target-v1:$(_deps_fingerprint)"

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
        if [[ -d "${main_deps}" && "${main_deps}" != "${DEPS_DIR}" ]] \
            && _deps_ready "${main_deps}"; then
            mkdir -p "${ROOT_DIR}/.cache"
            ln -s "${main_deps}" "${DEPS_DIR}"
            exit 0
        fi
    fi
fi

if _deps_ready "${DEPS_DIR}"; then
    exit 0
fi

# Serialize writers at the owned destination, including explicit symlink aliases.
mkdir -p "$(dirname "${DEPS_DIR}")"
if [[ -n "${LIVING_MEMORY_DEPS_DIR:-}" ]]; then
    target="$(realpath -m -- "${DEPS_DIR}")"
else
    target="$(cd "$(dirname "${DEPS_DIR}")" && pwd)/$(basename "${DEPS_DIR}")"
fi
mkdir -p "$(dirname "${target}")"
exec 9>"${target}.lock"
flock 9
if _deps_ready "${DEPS_DIR}"; then
    exit 0
fi

# A default symlink is borrowed storage; only its local link may be replaced.
# Explicit overrides own their resolved destination and retain the symlink.
if [[ -z "${LIVING_MEMORY_DEPS_DIR:-}" && -L "${DEPS_DIR}" ]]; then
    rm -- "${DEPS_DIR}"
fi
rm -f -- "${target}/.living-memory-deps-ready"
stage="$(mktemp -d "${target}.tmp.XXXXXX")"
previous="${stage}.previous"
_cleanup() {
    if [[ -d "${previous}" && ! -e "${target}" ]]; then
        mv -T -- "${previous}" "${target}"
    fi
    rm -rf -- "${stage}" "${previous}"
}
trap _cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Install into an empty tree: pip target upgrades do not remove old dist-info.
PYTHONPATH= "${PYTHON_BIN}" -m pip install --target "${stage}" "${ROOT_DIR}[test]"

# Validate the encoder once at preparation, before publishing readiness.
PYTHONPATH="${ROOT_DIR}/src:${stage}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -B - <<'PY'
import fastmcp
import pytest
import sentence_transformers
PY

printf '%s\n' "${want_fp}" > "${stage}/.living-memory-deps-ready"
if [[ -e "${target}" ]]; then
    mv -T -- "${target}" "${previous}"
fi
mv -T -- "${stage}" "${target}"
