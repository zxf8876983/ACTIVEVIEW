from pathlib import Path

import pytest

from activeview.core.paths import get_data_root
from activeview.scripts.validate_stage_b import validate


def test_stage_b_real_artifact_acceptance(tmp_path: Path):
    dataset_root = get_data_root() / "datasets/policy_v11_5"
    stage_b_root = dataset_root / "stage_b"
    if not (stage_b_root / "stage_b_summary.json").exists():
        pytest.skip("Stage B output is not available")
    report = validate(dataset_root, stage_b_root, report_path=tmp_path / "validation_report.json")
    assert report["passed"], report
