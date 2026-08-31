#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${ACTIVEVIEW_HABITAT_PYTHON:-/home/zxf/anaconda3/envs/habitat/bin/python}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp_habitat}"
mkdir -p "${TMPDIR}"
exec "${PYTHON_BIN}" -m activeview.scripts.analyze_stage_d_semantic_bev "$@"
