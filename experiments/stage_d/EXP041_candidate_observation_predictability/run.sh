#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec python -m activeview.scripts.run_stage_d_exp041_044 --data-root "${ACTIVEVIEW_DATA_ROOT:-${REPO_ROOT}/../../data/ActiveView}" --variants A B C
