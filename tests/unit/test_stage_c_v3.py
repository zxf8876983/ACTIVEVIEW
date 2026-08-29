import json

import numpy as np
import pytest
import torch

from activeview.active_view.stage_c_v3_predictability import _audit
from activeview.active_view.stage_c_v3_teacher import (
    future_perception_vector,
)
from activeview.active_view.stage_c_v3_topk import run_topk_audit
from activeview.active_view.utility_predictor import FuturePerceptionTeacherMLP


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_future_teacher_vector_uses_only_persisted_fields():
    vector = future_perception_vector({"predicted_label_id": 3, "logp_true": -0.7, "entropy": 0.2, "correct": True})
    assert vector.shape == (17,)
    assert vector[3] == 1.0
    assert vector[16] == pytest.approx(0.2)


def test_future_teacher_forward_is_candidate_aligned():
    model = FuturePerceptionTeacherMLP().eval()
    current = torch.randn(2, 275)
    geometry = torch.randn(2, 4, 11)
    future = torch.randn(2, 4, 17)
    output = model(current, geometry, future, torch.ones(2, 4, dtype=torch.bool))
    assert output.shape == (2, 4)


def test_future_teacher_candidate_permutation_permutates_outputs():
    model = FuturePerceptionTeacherMLP().eval()
    current = torch.randn(1, 275)
    geometry = torch.randn(1, 4, 11)
    future = torch.randn(1, 4, 17)
    permutation = torch.tensor([2, 0, 3, 1])
    first = model(current, geometry, future)
    second = model(current, geometry[:, permutation], future[:, permutation])
    assert torch.allclose(first[:, permutation], second, atol=1e-6, rtol=0.0)


def test_predictability_audit_reports_train_only_neighbours():
    train = [
        {"geometry": np.asarray([0.0, 0.0]), "utility": 1.0},
        {"geometry": np.asarray([1.0, 1.0]), "utility": -1.0},
        {"geometry": np.asarray([2.0, 2.0]), "utility": 1.0},
        {"geometry": np.asarray([3.0, 3.0]), "utility": -1.0},
        {"geometry": np.asarray([4.0, 4.0]), "utility": 1.0},
    ]
    val = [{"geometry": np.asarray([1.1, 1.1]), "utility": -1.0}]
    result = _audit(train, val, "geometry", 5, disagreement_threshold=2.0)
    assert result["train_reference_count"] == 5
    assert result["val_query_count"] == 1
    assert result["same_neighborhood_sign_conflict_rate"] == 1.0


def test_topk_audit_tie_breaks_by_geodesic_then_viewpoint(tmp_path):
    prediction = {
        "episode_id": "e0",
        "candidate_viewpoint_ids": [9, 3],
        "predicted_utilities": [1.0, 1.0],
    }
    utility = {
        "episode_id": "e0",
        "candidates": [
            {"viewpoint_id": 9, "geodesic_distance_m": 2.0, "utility": 1.0},
            {"viewpoint_id": 3, "geodesic_distance_m": 1.0, "utility": 0.8},
        ],
        "oracle": {
            "candidate_oracle_viewpoint_id": 9,
            "safe_oracle_viewpoint_id": 9,
            "safe_oracle_utility": 1.0,
            "safe_oracle_stays": False,
        },
    }
    predictions_path = tmp_path / "predictions.jsonl"
    utility_path = tmp_path / "utility.jsonl"
    output_path = tmp_path / "topk.json"
    _write_jsonl(predictions_path, [prediction])
    _write_jsonl(utility_path, [utility])
    result = run_topk_audit(predictions_path=predictions_path, stage_b_utility_path=utility_path, output_path=output_path)
    assert result["reports"]["1"]["candidate_oracle_hit_rate"] == 0.0
    assert result["reports"]["2"]["candidate_oracle_hit_rate"] == 1.0
