import torch

from activeview.active_view.policy_training import policy_loss, utility_gap_pairwise_loss


def test_gap_loss_prefers_correct_ordering():
    target = torch.tensor([[4.0, 1.0]])
    mask = torch.tensor([[True, True]])
    correct = torch.tensor([[3.0, 0.5]], requires_grad=True)
    reversed_scores = torch.tensor([[-0.5, 3.0]], requires_grad=True)

    correct_loss = utility_gap_pairwise_loss(correct, target, mask)
    reversed_loss = utility_gap_pairwise_loss(reversed_scores, target, mask)

    assert correct_loss < reversed_loss
    reversed_loss.backward()
    assert torch.isfinite(reversed_scores.grad).all()


def test_larger_target_gap_gets_larger_wrong_order_penalty():
    mask = torch.tensor([[True, True]])
    predicted = torch.tensor([[0.9, 1.0]])
    small_gap = torch.tensor([[1.0, 0.9]])
    large_gap = torch.tensor([[10.0, 0.9]])

    small_loss = utility_gap_pairwise_loss(predicted, small_gap, mask)
    large_loss = utility_gap_pairwise_loss(predicted, large_gap, mask)

    assert large_loss > small_loss


def test_gap_loss_includes_stay_and_excludes_padding():
    mask = torch.tensor([[True, False]])
    target = torch.tensor([[-2.0, 999.0]])
    predicted = torch.tensor([[-1.0, 1000.0]], requires_grad=True)
    changed_padding = torch.tensor([[-1.0, -1000.0]], requires_grad=True)

    first = utility_gap_pairwise_loss(predicted, target, mask)
    second = utility_gap_pairwise_loss(changed_padding, target, mask)

    assert torch.isfinite(first)
    assert torch.isfinite(second)
    assert torch.allclose(first, second)


def test_gap_loss_handles_equal_and_single_candidate_sets():
    equal_target = torch.tensor([[0.0, 0.0]])
    equal_pred = torch.tensor([[1.0, -1.0]], requires_grad=True)
    equal_mask = torch.tensor([[True, True]])
    equal_loss = utility_gap_pairwise_loss(equal_pred, equal_target, equal_mask)

    single_pred = torch.tensor([[2.0, 999.0]], requires_grad=True)
    single_target = torch.tensor([[1.0, 999.0]])
    single_mask = torch.tensor([[True, False]])
    single_loss = utility_gap_pairwise_loss(single_pred, single_target, single_mask)

    assert torch.isfinite(equal_loss) and equal_loss.item() == 0.0
    assert torch.isfinite(single_loss)


def test_stage_c_loss_zero_gap_weight_preserves_existing_objective():
    predicted = torch.tensor([[0.5, -0.2, 3.0]], requires_grad=True)
    target = torch.tensor([[0.4, -0.1, 99.0]])
    mask = torch.tensor([[True, True, False]])

    explicit_zero = policy_loss(
        predicted, target, mask, lambda_gap=0.0, tau_gap=1.0, max_gap_weight=10.0
    )

    valid = mask.bool()
    huber = torch.nn.functional.smooth_l1_loss(predicted[valid], target[valid])
    gt = torch.cat([torch.zeros((1, 1)), target], dim=1)
    scores = torch.cat([torch.zeros((1, 1)), predicted], dim=1)
    rank_mask = torch.cat([torch.ones((1, 1), dtype=torch.bool), valid], dim=1)
    gt = gt.masked_fill(~rank_mask, float("-inf"))
    scores = scores.masked_fill(~rank_mask, float("-inf"))
    target_distribution = torch.softmax(gt / 0.5, dim=1)
    log_prediction = torch.log_softmax(scores / 0.5, dim=1)
    log_prediction = torch.where(rank_mask, log_prediction, torch.zeros_like(log_prediction))
    existing_total = huber - (target_distribution * log_prediction).sum(dim=1).mean()

    assert torch.allclose(existing_total, explicit_zero["total"])
    assert explicit_zero["gap_ranking"].item() == 0.0
