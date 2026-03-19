#!/bin/bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${PYTHON_EXE}" "${SCRIPT_DIR}/chat.py" "$@"
