#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${REPO_ROOT}/.tmp"
mkdir -p "${TMPDIR}"
python -m activeview.scripts.analyze_stage_d_predictability \
  --output "${SCRIPT_DIR}/result.json" \
  --analysis "${SCRIPT_DIR}/analysis.md"
