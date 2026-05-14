#!/usr/bin/env bash
set -euo pipefail

if ! command -v npm >/dev/null 2>&1 && [[ -n "${NVM_BIN:-}" && -x "${NVM_BIN}/npm" ]]; then
  export PATH="${NVM_BIN}:$PATH"
fi

npm test
