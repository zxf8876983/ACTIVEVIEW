import numpy as np
import torch

from activeview.active_view.stage_d_depth_spatial import (
    SpatialRGBDUtilityRegressor,
    depth_spatial_features,
)


def test_depth_spatial_features_shape_and_valid_statistics() -> None:
    depth = np.full((256, 256), 2.0, dtype=np.float32)
    depth[:64, :64] = np.nan
    features = depth_spatial_features(depth)
    assert features.shape == (16, 4)
    assert features.dtype == np.float16
    assert np.isfinite(features).all()
    assert float(features[0, 3]) == 0.0
    assert float(features[1, 0]) == 2.0


def test_spatial_rgbd_regressor_output_shape_and_input_contract() -> None:
    model = SpatialRGBDUtilityRegressor()
    output = model(
        torch.zeros(2, 128),
        torch.zeros(2, 1),
        torch.zeros(2, 16, 768),
        torch.zeros(2, 16, 768),
        torch.zeros(2, 16, 4),
        torch.zeros(2, 16, 4),
    )
    assert output.shape == (2,)
    assert model.input_dim == 609
