#!/usr/bin/env bash
# run.sh — Launch the full pipeline (now a single Python process).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${PROJECT_DIR}/python/main.py" "$@"
