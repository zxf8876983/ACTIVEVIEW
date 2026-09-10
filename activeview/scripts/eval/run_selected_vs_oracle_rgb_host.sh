#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zxf/WorkSpace/code/code/ActiveView"
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/zxf/anaconda3/envs/habitat/bin/python3.9}"
DATA_ROOT="/home/zxf/WorkSpace/code/data/ActiveView"
SCENE_ROOT="/home/zxf/WorkSpace/code/code/robot/DATA/hm3d-train"
ARCHIVE_ROOT="$DATA_ROOT/datasets/offline/habitat-train/00006-00087"
MOTION_MANIFEST="$DATA_ROOT/datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json"
OUTPUT="$REPO_ROOT/experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization"

cd "$REPO_ROOT"

if [[ ! -x "$HABITAT_PYTHON" ]]; then
    echo "Habitat Python is not executable: $HABITAT_PYTHON" >&2
    exit 2
fi

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is unavailable; run this script from the GPU host." >&2
    exit 2
}
nvidia-smi

"$HABITAT_PYTHON" - <<'PY'
import sys

import habitat_sim
import magnum

print("python:", sys.executable)
print("habitat_sim: PASS")
print("magnum: PASS")
PY

export ACTIVEVIEW_HOST_RUNNER="$REPO_ROOT/activeview/scripts/eval/run_selected_vs_oracle_rgb_host.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$HABITAT_PYTHON" activeview/scripts/eval/render_selected_vs_oracle_rgb_worker.py \
    --case-manifest "$OUTPUT/case_manifest.json" \
    --output "$OUTPUT/qualitative_rgb" \
    --data-root "$DATA_ROOT" \
    --scene-root "$SCENE_ROOT" \
    --archive-root "$ARCHIVE_ROOT" \
    --motion-manifest "$MOTION_MANIFEST" \
    --runtime-status "$OUTPUT/qualitative_rgb/runtime_status.json"

"$HABITAT_PYTHON" activeview/scripts/eval/visualize_reduced12_selected_vs_oracle.py

echo "Targeted qualitative RGB rendering and visualization completed."
