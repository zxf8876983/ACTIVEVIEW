import numpy as np
from activeview.active_view.stage_d_semantic_bev import (
    BEV_CHANNELS,
    BEV_SHAPE,
    bev_cell,
    normalize_category,
    pool_bev,
    project_world_samples,
    world_to_s1,
)


def test_category_aliases_and_unknowns_are_deterministic():
    assert normalize_category("sofa") == "couch"
    assert normalize_category("kitchen_cabinet") == "cabinet"
    assert normalize_category("floor") == "other_object"


def test_world_to_s1_and_bev_cell_use_s1_origin():
    identity = np.eye(3)
    assert np.allclose(world_to_s1([1.0, 0.0, 2.0], [1.0, 0.0, 1.0], identity), [0.0, 0.0, 1.0])
    assert bev_cell([0.0, 0.0, 0.0]) == (40, 40)
    assert bev_cell([9.0, 0.0, 0.0]) is None


def test_projected_endpoint_is_occupied_and_ray_is_free_without_conflict():
    points = np.full((1, 2, 3), np.nan, dtype=np.float32)
    points[0, 0] = [0.0, 0.0, 1.0]
    points[0, 1] = [0.0, 0.0, 2.0]
    labels = np.array([[1, 0]], dtype=np.int32)
    bev = project_world_samples(
        points,
        labels,
        camera_world=[0.0, 0.0, 0.0],
        s1_position=[0.0, 0.0, 0.0],
        s1_rotation_matrix=np.eye(3),
        semantic_channels={1: BEV_CHANNELS["chair"]},
    )
    assert bev.dtype == np.uint8 and bev.shape == BEV_SHAPE
    assert bev[BEV_CHANNELS["occupied"], 50, 40] == 1
    assert bev[BEV_CHANNELS["free"], 50, 40] == 0
    assert bev[BEV_CHANNELS["observed"]].sum() > 0


def test_pool_shape_is_fixed():
    pooled = pool_bev(np.zeros(BEV_SHAPE, dtype=np.uint8))
    assert pooled.shape == (15 * 10 * 10,)
