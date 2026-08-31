#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m activeview.scripts.analyze_stage_d_human_viewpoint "$@"
