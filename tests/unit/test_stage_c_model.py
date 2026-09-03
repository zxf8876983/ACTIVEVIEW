import torch

from activeview.methods.active_view.policy import move_stay_decision
from activeview.methods.active_view.utility_predictor import (
    MoveStaySetUtilityRanker,
    PairwiseUtilityMLP,
    SetUtilityRanker,
)


def test_models_return_candidate_utilities_and_set_ranker_is_permutation_equivariant():
    current = torch.randn(1, 275)
    geometry = torch.randn(1, 3, 11)
    mask = torch.ones(1, 3, dtype=torch.bool)
    pairwise = PairwiseUtilityMLP().eval()
    set_ranker = SetUtilityRanker().eval()
    assert pairwise(current, geometry, mask).shape == (1, 3)
    first = set_ranker(current, geometry, mask)
    permutation = torch.tensor([2, 0, 1])
    shuffled = set_ranker(current, geometry[:, permutation], mask[:, permutation])
    assert torch.allclose(first[:, permutation], shuffled, atol=1e-6, rtol=0.0)


def test_move_stay_ranker_has_current_only_gate_and_expected_shapes():
    current = torch.randn(2, 275)
    geometry = torch.randn(2, 3, 11)
    mask = torch.ones(2, 3, dtype=torch.bool)
    model = MoveStaySetUtilityRanker().eval()
    assert model(current, geometry, mask).shape == (2, 3)
    assert model.move_logits(current).shape == (2,)


def test_move_stay_gate_uses_fixed_half_probability_threshold():
    assert move_stay_decision(0.49) is True
    assert move_stay_decision(0.5) is False
    assert move_stay_decision(0.9) is False
