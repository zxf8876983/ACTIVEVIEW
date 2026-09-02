#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${ACTIVEVIEW_HABITAT_PYTHON:-/home/zxf/anaconda3/envs/habitat/bin/python}"
exec "${PYTHON_BIN}" -m activeview.scripts.run_stage_d_exp046_048 "$@"
