import torch

from activeview.methods.active_view.training import policy_loss
from activeview.evaluation.metrics import move_stay_metrics, regression_metrics


def test_stage_c_loss_masks_padding_and_is_finite():
    predicted = torch.tensor([[0.5, -0.2, 3.0]], requires_grad=True)
    target = torch.tensor([[0.4, -0.1, 99.0]])
    mask = torch.tensor([[True, True, False]])
    losses = policy_loss(predicted, target, mask)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_regression_and_move_stay_metrics():
    metrics = regression_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert metrics["mae"] == 0.0 and metrics["spearman"] == 1.0
    move = move_stay_metrics([{"safe_oracle_stays": False, "predicted_stays": False}, {"safe_oracle_stays": True, "predicted_stays": True}])
    assert move["accuracy"] == 1.0 and move["f1"] == 1.0
