#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
exec "${PYTHON_BIN:-python}" -m activeview.scripts.analyze_stage_d_overnight \
  --data-root "${ACTIVEVIEW_DATA_ROOT:-$ROOT/../../data/ActiveView}" "$@"
