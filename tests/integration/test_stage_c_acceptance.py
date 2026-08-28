import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.validate_stage_c import validate


DATA_ROOT = Path("/home/zxf/WorkSpace/code/data/ActiveView")
DATASET_ROOT = DATA_ROOT / "datasets/policy_v11_5"
STAGE_B_ROOT = DATASET_ROOT / "stage_b"
STAGE_C_ROOT = DATASET_ROOT / "stage_c"


@pytest.mark.integration
def test_stage_c_real_artifacts_pass_independent_validation(tmp_path):
    required = [
        STAGE_C_ROOT / "stage_c_feature_summary.json",
        STAGE_C_ROOT / "evaluations/pairwise_mlp_evaluation_summary.json",
        STAGE_C_ROOT / "evaluations/set_ranker_evaluation_summary.json",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Stage C runtime artifacts are not available")
    summaries = [path for path in required[1:]]
    report = validate(
        dataset_root=DATASET_ROOT,
        stage_b_root=STAGE_B_ROOT,
        stage_c_root=STAGE_C_ROOT,
        eval_summaries=summaries,
        report_path=tmp_path / "stage_c_validation.json",
    )
    assert report["passed"], json.dumps(report, ensure_ascii=False)
