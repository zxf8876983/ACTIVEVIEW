import numpy as np
import torch

from activeview.active_view.stage_d_rgb_spatial import (
    SPATIAL_PATCH_GRID,
    SPATIAL_TOKEN_COUNT,
    SpatialRGBUtilityRegressor,
    dino_spatial_embeddings,
)
from activeview.active_view.stage_d_rgb_context import DINO_EMBED_DIM


class _FakeDino(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> object:
        batch = pixel_values.size(0)
        values = torch.arange(
            batch * (SPATIAL_PATCH_GRID * SPATIAL_PATCH_GRID + 1) * DINO_EMBED_DIM,
            dtype=torch.float32,
            device=pixel_values.device,
        ).remainder(1000).reshape(batch, SPATIAL_PATCH_GRID * SPATIAL_PATCH_GRID + 1, DINO_EMBED_DIM)
        return type("Output", (), {"last_hidden_state": values})()


def test_patch_grid_and_spatial_pool_shape() -> None:
    images = np.zeros((2, 256, 256, 3), dtype=np.uint8)
    values = dino_spatial_embeddings(_FakeDino(), images, torch.device("cpu"))
    assert values.shape == (2, SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM)
    assert values.dtype == np.float16


def test_spatial_regressor_has_shared_projector_and_513d_input() -> None:
    model = SpatialRGBUtilityRegressor()
    assert model.input_dim == 513
    assert model.spatial_encoder.num_layers == 1
    assert model.rgb_projector[0].in_features == DINO_EMBED_DIM
    x = torch.zeros((2, 128))
    u = torch.zeros((2, 1))
    rgb = torch.zeros((2, SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM))
    assert model(x, u, rgb, rgb).shape == (2,)
