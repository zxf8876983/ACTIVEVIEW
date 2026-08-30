import numpy as np
import pytest
import torch

from activeview.active_view.stage_d_rgb_context import (
    DINO_EMBED_DIM,
    EXP024_INPUT_DIM,
    RGBContextUtilityRegressor,
    RGBObservationKey,
    _preprocess_rgb,
    observation_keys_from_feature_rows,
)


def _row(episode_id: str = "e", s0: int = 1, s1: int = 2) -> dict:
    return {"episode_id": episode_id, "scene_id": "scene", "region": "bedroom", "record_id": "record", "s0_viewpoint_id": s0, "s1_viewpoint_id": s1}


def test_observation_keys_use_only_s0_and_s1_and_deduplicate() -> None:
    keys, mapping = observation_keys_from_feature_rows([_row("e1", 1, 2), _row("e2", 1, 3)])
    assert {(key.viewpoint_id, key.record_id) for key in keys} == {(1, "record"), (2, "record"), (3, "record")}
    assert mapping["e1"][0] == RGBObservationKey("scene", "bedroom", "record", 1)


def test_rgb_context_regressor_has_513d_input_and_shared_projector() -> None:
    model = RGBContextUtilityRegressor()
    output = model(torch.zeros((2, 128)), torch.zeros((2, 1)), torch.zeros((2, DINO_EMBED_DIM)), torch.zeros((2, DINO_EMBED_DIM)))
    assert model.input_dim == EXP024_INPUT_DIM == 513
    assert output.shape == (2,)
    assert model.rgb_projector[0].in_features == DINO_EMBED_DIM


def test_rgb_preprocessing_is_deterministic_224_imagenet_normalized() -> None:
    rgb = np.full((2, 256, 256, 3), 255, dtype=np.uint8)
    first = _preprocess_rgb(rgb)
    second = _preprocess_rgb(rgb)
    assert first.shape == (2, 3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.allclose(first[:, 0], torch.full_like(first[:, 0], (1.0 - 0.485) / 0.229))


def test_duplicate_episode_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        observation_keys_from_feature_rows([_row(), _row()])
