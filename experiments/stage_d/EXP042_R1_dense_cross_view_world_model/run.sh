#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-${ROOT}/../../data/ActiveView}"
RGB_ROOT="${ACTIVEVIEW_RGB_FEATURE_ROOT:-/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4}"
export ACTIVEVIEW_RGB_FEATURE_ROOT="$RGB_ROOT"
python -m activeview.scripts.run_stage_d_exp042r1_045 --data-root "$DATA_ROOT" --epochs 12 --workers 4 --device cuda:0 --variants A B C D E F
