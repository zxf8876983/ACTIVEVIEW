#!/usr/bin/env bash
set -euo pipefail
python -m activeview.scripts.run_stage_d_dense_campaign --data-root "${ACTIVEVIEW_DATA_ROOT:-../../data/ActiveView}"
