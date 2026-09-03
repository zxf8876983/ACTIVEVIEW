import torch

from activeview.active_view.stage_d_counterfactual_recognition import CounterfactualRecognitionModel


def test_counterfactual_recognition_accepts_two_and_three_view_histories() -> None:
    model = CounterfactualRecognitionModel()
    for history_size in (2, 3):
        output = model(
            torch.randn(4, history_size, 16),
            torch.randn(4, history_size, 768),
            torch.randn(4, history_size, 102),
            torch.randn(4, history_size, 9),
            torch.randn(4, 9),
            torch.ones(4, history_size, dtype=torch.bool),
        )
        assert output.shape == (4, 16)


def test_counterfactual_recognition_has_no_future_target_input() -> None:
    model = CounterfactualRecognitionModel()
    assert not any(name.startswith("target") for name, _ in model.named_parameters())

