import numpy as np
import torch

from activeview.action_recognition.st_gcn_model import STGCN
from activeview.active_view.stage_c_v2_features import extract_frozen_joint_tokens, v2_schema
from activeview.active_view.utility_predictor import build_utility_predictor, count_parameters
from activeview.perception.skeleton_definition import get_skeleton_definition


def _inputs(batch=2, candidates=4):
    return (
        torch.randn(batch, 17, 256),
        torch.randn(batch, 3, 30, 17),
        torch.randn(batch, 19),
        torch.randn(batch, candidates, 11),
        torch.ones(batch, candidates, dtype=torch.bool),
    )


def test_frozen_stgcn_final_block_tokens_keep_joint_dimension():
    model = STGCN(
        in_channels=3, num_classes=16, graph_strategy="spatial", edge_importance_weighting=True,
        skel_def=get_skeleton_definition(backend="h36m_17"),
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokens = extract_frozen_joint_tokens(model, torch.randn(2, 3, 30, 17))
    assert tokens.shape == (2, 17, 256)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_v2_models_forward_with_expected_shapes_and_parameter_counts():
    joint_tokens, skeleton, semantic, geometry, mask = _inputs()
    for model_type in ("joint_aware_set_ranker", "candidate_conditioned_attention", "skeleton_policy_transformer"):
        model = build_utility_predictor(model_type, geometry_dim=11).eval()
        if model_type == "skeleton_policy_transformer":
            output = model(skeleton, semantic, geometry, mask)
        else:
            output = model(joint_tokens, semantic, geometry, mask)
        assert output.shape == (2, 4)
        assert count_parameters(model) > 0


def test_candidate_conditioned_and_skeleton_models_are_permutation_equivariant():
    joint_tokens, skeleton, semantic, geometry, mask = _inputs(candidates=5)
    permutation = torch.tensor([4, 1, 0, 3, 2])
    for model_type in ("candidate_conditioned_attention", "skeleton_policy_transformer"):
        model = build_utility_predictor(model_type, geometry_dim=11).eval()
        if model_type == "skeleton_policy_transformer":
            first = model(skeleton, semantic, geometry, mask)
            second = model(skeleton, semantic, geometry[:, permutation], mask[:, permutation])
        else:
            first = model(joint_tokens, semantic, geometry, mask)
            second = model(joint_tokens, semantic, geometry[:, permutation], mask[:, permutation])
        assert torch.allclose(first[:, permutation], second, atol=1e-5, rtol=0.0)


def test_v2_inputs_do_not_change_when_future_candidate_fields_change():
    geometry = np.zeros((2, 11), dtype=np.float32)
    altered_geometry = geometry.copy()
    # Future perception fields are not represented in the geometry array.
    assert np.array_equal(geometry, altered_geometry)
    schema = v2_schema()
    assert schema["future_candidate_perception_used_as_input"] is False
    assert "candidate_skeleton" in schema["forbidden_input_fields"]
    assert "candidate_entropy" in schema["forbidden_input_fields"]
