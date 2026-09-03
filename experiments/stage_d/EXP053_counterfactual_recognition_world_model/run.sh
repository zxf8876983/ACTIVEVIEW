#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${ROOT}/../../data/ActiveView}"
PYTHONPATH="${ROOT}" python "${ROOT}/activeview/scripts/train_exp053_counterfactual_recognition.py" \
  --data-root "${DATA_ROOT}" --device "${EXP053_DEVICE:-cpu}" --epochs 15 --batch-size 256
