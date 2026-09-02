#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec python "${REPO_ROOT}/activeview/scripts/check_exp051_r1_prerequisites.py" \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${REPO_ROOT}/experiments/stage_d/EXP051_R1_closed_loop_revision"
