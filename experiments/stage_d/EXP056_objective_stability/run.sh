#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-$ROOT/../../data/ActiveView}"
PYTHON="/home/zxf/anaconda3/envs/habitat/bin/python"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u "$ROOT/activeview/scripts/run_exp056_objective_stability.py" --data-root "$DATA_ROOT" --device "${EXP056_DEVICE:-cuda:0}"
