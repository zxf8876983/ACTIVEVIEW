#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-/home/zxf/WorkSpace/code/data/ActiveView}"
PYTHONPATH="$REPO_ROOT" /home/zxf/anaconda3/envs/habitat/bin/python \
  "$REPO_ROOT/activeview/scripts/run_exp051_r2_real_eval.py" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$DATA_ROOT/experiments/stage_d/EXP050_joint_rollout_revision/joint_revision_final.pth" \
  --device cuda:0
