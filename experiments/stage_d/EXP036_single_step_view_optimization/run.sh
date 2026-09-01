#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
exec "${PYTHON_BIN:-python}" -m activeview.scripts.run_stage_d_dense_campaign \
  --data-root "${ACTIVEVIEW_DATA_ROOT:-$ROOT/../../data/ActiveView}" "$@"
