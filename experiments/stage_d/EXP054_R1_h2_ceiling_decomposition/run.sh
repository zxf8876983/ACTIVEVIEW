#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${ROOT}/../../data/ActiveView}"
PYTHONPATH="${ROOT}" /home/zxf/anaconda3/envs/habitat/bin/python -u "${ROOT}/activeview/scripts/analyze_exp054_h2_ceiling.py" \
  --data-root "${DATA_ROOT}" --device "${EXP054_DEVICE:-cuda:0}" --r1 \
  --output-dir "${ROOT}/experiments/stage_d/EXP054_R1_h2_ceiling_decomposition"
