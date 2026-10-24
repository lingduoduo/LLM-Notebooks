#!/usr/bin/env bash
# Compatibility entry point; all service behavior lives in ../services.py.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/../services.py" start "$@"
