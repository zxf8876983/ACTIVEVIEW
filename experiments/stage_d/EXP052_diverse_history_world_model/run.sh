#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-/home/zxf/WorkSpace/code/data/ActiveView}"
PYTHONPATH="$REPO_ROOT" /home/zxf/anaconda3/envs/habitat/bin/python \
  "$REPO_ROOT/activeview/scripts/train_exp052_diverse_history_wm.py" \
  --data-root "$DATA_ROOT" --device cuda:0 --epochs 15 --batch-size 256 --workers 4
