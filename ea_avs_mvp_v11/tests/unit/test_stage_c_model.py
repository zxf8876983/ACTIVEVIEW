import torch

from ea_avs_mvp_v11.active_view.utility_predictor import PairwiseUtilityMLP, SetUtilityRanker


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
