#!/bin/bash

# Low-priority wrapper for LMCache cleanup jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CMD=("$PYTHON_BIN" "$SCRIPT_DIR/cleanup_cache.py" "$@")

if command -v ionice >/dev/null 2>&1; then
  CMD=(ionice -c3 "${CMD[@]}")
fi

if command -v nice >/dev/null 2>&1; then
  CMD=(nice -n19 "${CMD[@]}")
fi

exec "${CMD[@]}"
