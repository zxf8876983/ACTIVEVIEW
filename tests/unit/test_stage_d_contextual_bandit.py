import pytest
import torch

from activeview.active_view.stage_d_contextual_bandit import (
    ContextualBanditRanker,
    expected_reward_loss,
    select_bandit_actions,
)


def test_stay_reward_is_fixed_zero_and_higher_reward_reduces_loss():
    scores = torch.tensor([[0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True]])
    low_loss, _ = expected_reward_loss(scores, torch.tensor([[-1.0, 1.0]]), mask)
    high_loss, _ = expected_reward_loss(scores, torch.tensor([[2.0, 1.0]]), mask)
    assert high_loss.item() < low_loss.item()


def test_reward_tensor_is_not_part_of_model_input():
    model = ContextualBanditRanker()
    model.eval()
    s0 = torch.zeros((1, 275))
    s1 = torch.zeros((1, 275))
    delta = torch.zeros((1, 19))
    geometry = torch.zeros((1, 2, 11))
    mask = torch.tensor([[True, True]])
    with torch.no_grad():
        scores_a = model(s0, s1, delta, geometry, mask)
        scores_b = model(s0, s1, delta, geometry, mask)
    assert torch.equal(scores_a, scores_b)


def test_bandit_model_returns_candidate_scores():
    model = ContextualBanditRanker()
    output = model(
        torch.zeros((2, 275)), torch.zeros((2, 275)), torch.zeros((2, 19)),
        torch.zeros((2, 2, 11)), torch.ones((2, 2), dtype=torch.bool),
    )
    assert output.shape == (2, 2)


def test_stay_wins_nonpositive_candidate_scores():
    assert select_bandit_actions([-1.0, 0.0], [2, 3]) == (True, None, 0.0)


def test_bandit_can_select_each_candidate_from_model_scores():
    assert select_bandit_actions([2.0, 1.0], [2, 3])[1] == 2
    assert select_bandit_actions([1.0, 2.0], [2, 3])[1] == 3


def test_stay_is_first_on_exact_tie():
    assert select_bandit_actions([0.0, 0.0], [2, 3])[0] is True


def test_invalid_mask_shape_is_rejected():
    with pytest.raises(ValueError, match="candidate_mask"):
        ContextualBanditRanker()(
            torch.zeros((1, 275)), torch.zeros((1, 275)), torch.zeros((1, 19)),
            torch.zeros((1, 2, 11)), torch.ones((1, 1), dtype=torch.bool),
        )


def test_test_split_is_not_an_exposed_parser_option():
    from activeview.scripts.train_stage_d_contextual_bandit import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--split", "test"])
