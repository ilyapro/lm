#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset LIVING_MEMORY_EMBEDDING_BACKEND LIVING_MEMORY_CHUNK_TOKENIZER
exec "${SCRIPT_DIR}/test.sh"
